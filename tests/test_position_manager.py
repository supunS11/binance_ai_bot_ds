import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import exchange
import execution
import market_structure
import risk_manager
import signal_engine
from position_manager import (
    BREAKEVEN_ACTIVE, DCA_ACTIVE, DCA_PENDING, PENDING_LIMIT_FILL, RETRACEMENT_PENDING, TP1_PENDING,
    PositionManager, _more_favorable, _order_type, _resolve_tp1_price, _structure_stop_candidate,
)


# _close() journals every outcome for real (signal_journal.append_outcome),
# and _finalize_retracement_entry journals every settle for real
# (signal_journal.append_retracement_settle) - neither side effect is
# something these tests care about, so both are patched out for the whole
# module rather than in every single test that reaches one.
_journal_patchers = []


def setUpModule():
    global _journal_patchers
    _journal_patchers = [
        patch("position_manager.signal_journal.append_outcome"),
        patch("position_manager.signal_journal.append_retracement_settle"),
    ]
    for patcher in _journal_patchers:
        patcher.start()


def tearDownModule():
    for patcher in _journal_patchers:
        patcher.stop()


def _plan(side="BUY"):
    if side == "SELL":
        return {
            "symbol": "BTCUSDT",
            "side": side,
            "entry_price": 100,
            "sl_price": 102,
            "tp1_price": 98,
            "tp2_price": 96,
            "breakeven_price": 99.98,
            "quantity": 1.0,
            "tp1_quantity": 0.5,
            "tp2_quantity": 0.5,
        }

    return {
        "symbol": "BTCUSDT",
        "side": side,
        "entry_price": 100,
        "sl_price": 98,
        "tp1_price": 102,
        "tp2_price": 104,
        "breakeven_price": 100.02,
        "quantity": 1.0,
        "tp1_quantity": 0.5,
        "tp2_quantity": 0.5,
    }


def _candle(high, low, close=None):
    return {
        "open_time": 0, "open": high, "high": high, "low": low,
        "close": high if close is None else close, "volume": 1, "closed": False,
    }


class OrderTypeFieldTests(unittest.TestCase):
    """Real bug, confirmed against v7's proven-working
    find_matching_open_algo_order: the algo-order list endpoint returns
    the type under `orderType`, not `type`. Checking `type` alone matches
    nothing, ever - every "missing order" self-heal attempt then tries to
    place a genuine duplicate and gets rejected with -4130 forever."""

    def test_prefers_the_real_orderType_field(self):
        self.assertEqual(_order_type({"orderType": "STOP_MARKET"}), "STOP_MARKET")

    def test_falls_back_to_type_if_orderType_is_absent(self):
        self.assertEqual(_order_type({"type": "TAKE_PROFIT_MARKET"}), "TAKE_PROFIT_MARKET")

    def test_missing_both_fields_is_empty_not_a_crash(self):
        self.assertEqual(_order_type({}), "")
        self.assertEqual(_order_type(None), "")

    def test_find_open_order_matches_against_the_real_field_shape(self):
        # No "type" key at all - only what Binance actually returns.
        real_tp2 = {"orderType": "TAKE_PROFIT_MARKET", "closePosition": True, "algoId": "real_tp2"}

        with patch.object(exchange, "get_open_algo_orders", return_value=[real_tp2]):
            found = PositionManager._find_open_order("BTCUSDT", "TAKE_PROFIT_MARKET", close_position=True)

        self.assertIsNotNone(found)
        self.assertEqual(found["algoId"], "real_tp2")


class SaveAndLoadStateTests(unittest.TestCase):
    """position_manager.STATE_PATH's own comment explains why this exists
    - a full-fidelity snapshot of self.positions, preferred over
    exchange-shape reconciliation on restart wherever available."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "position_state.json"

    def test_save_then_load_round_trips_exactly(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True}, trade_id="BTCUSDT_123")

        manager.save_state(self.path)
        loaded = PositionManager.load_state(self.path)

        self.assertEqual(loaded, manager.positions)

    def test_save_creates_the_parent_directory_if_missing(self):
        nested_path = Path(self._tmpdir.name) / "nested" / "position_state.json"
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})

        manager.save_state(nested_path)

        self.assertTrue(nested_path.exists())

    def test_save_leaves_no_leftover_temp_file(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})
        manager.save_state(self.path)

        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_load_missing_file_returns_empty_dict(self):
        loaded = PositionManager.load_state(self.path)  # never saved
        self.assertEqual(loaded, {})

    def test_load_corrupt_json_returns_empty_dict_not_raise(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not valid json")

        loaded = PositionManager.load_state(self.path)  # must not raise

        self.assertEqual(loaded, {})

    def test_load_non_dict_json_returns_empty_dict(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(["not", "a", "dict"]))

        loaded = PositionManager.load_state(self.path)

        self.assertEqual(loaded, {})

    def test_save_failure_does_not_raise(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})

        # A directory where the file should be - mkdir/open both fail.
        bad_path = Path(self._tmpdir.name) / "not_writable_dir"
        bad_path.mkdir()

        manager.save_state(bad_path)  # must not raise


class TryRestoreFromSavedStateTests(unittest.TestCase):
    def _live_position(self, side="BUY", quantity=1.0):
        return {"symbol": "BTCUSDT", "side": side, "entry_price": 100.0, "quantity": quantity}

    def _saved_position(self, side="BUY", quantity=1.0, stage=DCA_ACTIVE):
        return {"side": side, "quantity": quantity, "stage": stage, "entry_price": 98.0}

    def test_no_saved_entry_for_symbol_returns_false(self):
        manager = PositionManager()
        result = manager._try_restore_from_saved_state("BTCUSDT", self._live_position(), {})
        self.assertFalse(result)
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_matching_side_and_quantity_restores_verbatim(self):
        manager = PositionManager()
        saved = self._saved_position()

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(), {"BTCUSDT": saved}
        )

        self.assertTrue(result)
        self.assertEqual(manager.positions["BTCUSDT"], saved)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_side_mismatch_is_rejected(self):
        manager = PositionManager()
        saved = self._saved_position(side="SELL")

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(side="BUY"), {"BTCUSDT": saved}
        )

        self.assertFalse(result)
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_quantity_within_one_percent_tolerance_is_accepted(self):
        manager = PositionManager()
        saved = self._saved_position(quantity=100.0)

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(quantity=100.5), {"BTCUSDT": saved}
        )

        self.assertTrue(result)

    def test_quantity_beyond_tolerance_is_rejected(self):
        # Real motivation: a manual partial close (or any other out-of-
        # band change) since the last save makes the saved snapshot
        # unreliable - same discipline as the XNYUSDT investigation.
        manager = PositionManager()
        saved = self._saved_position(quantity=100.0)

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(quantity=50.0), {"BTCUSDT": saved}
        )

        self.assertFalse(result)
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_zero_live_quantity_is_rejected(self):
        manager = PositionManager()
        saved = self._saved_position(quantity=100.0)

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(quantity=0), {"BTCUSDT": saved}
        )

        self.assertFalse(result)

    def test_missing_saved_quantity_is_rejected(self):
        manager = PositionManager()
        saved = {"side": "BUY", "stage": DCA_ACTIVE, "entry_price": 98.0}  # no quantity key

        result = manager._try_restore_from_saved_state(
            "BTCUSDT", self._live_position(), {"BTCUSDT": saved}
        )

        self.assertFalse(result)


class ReconcileOnStartupTests(unittest.TestCase):
    def _live_position(self, symbol="BTCUSDT", side="BUY", entry=100.0, qty=1.0):
        return {"symbol": symbol, "side": side, "entry_price": entry, "quantity": qty}

    def test_no_open_positions_does_nothing(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[]):
            manager.reconcile_on_startup()

        self.assertEqual(manager.open_count(), 0)

    def test_matching_saved_state_is_preferred_over_exchange_shape_guessing(self):
        saved = {"side": "BUY", "quantity": 1.0, "stage": DCA_ACTIVE, "entry_price": 98.0}

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(PositionManager, "load_state", return_value={"BTCUSDT": saved}), \
             patch.object(exchange, "get_open_algo_orders") as get_orders:
            manager = PositionManager()
            manager.reconcile_on_startup()

        self.assertEqual(manager.positions["BTCUSDT"], saved)
        get_orders.assert_not_called()  # never fell through to _adopt_position's own guessing

    def test_no_matching_saved_state_falls_back_to_exchange_shape_guessing(self):
        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(PositionManager, "load_state", return_value={}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[
                 {"type": "STOP_MARKET", "triggerPrice": "98", "algoId": "sl1"},
             ]):
            manager = PositionManager()
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["sl_order_id"], "sl1")

    def test_already_tracked_symbol_is_not_re_adopted(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders") as get_orders:
            manager.reconcile_on_startup()

        get_orders.assert_not_called()

    def test_full_order_set_found_adopts_with_real_prices_and_ids(self):
        manager = PositionManager()
        open_orders = [
            {"type": "STOP_MARKET", "triggerPrice": "98", "algoId": "sl1"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "false", "triggerPrice": "102", "origQty": "0.8", "algoId": "tp1_1"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "104", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["sl_price"], 98.0)
        self.assertEqual(position["tp1_price"], 102.0)
        self.assertEqual(position["tp2_price"], 104.0)

    def test_adopts_correctly_against_the_real_orderType_field_shape(self):
        # No "type" key, no "origQty" key - only what Binance's algo-order
        # endpoint actually returns (confirmed against v7's proven-working
        # parsing), so this proves the fix against reality, not a guess.
        manager = PositionManager()
        open_orders = [
            {"orderType": "STOP_MARKET", "stopPrice": "98", "algoId": "sl1"},
            {"orderType": "TAKE_PROFIT_MARKET", "closePosition": False, "stopPrice": "102", "quantity": "0.8", "algoId": "tp1_1"},
            {"orderType": "TAKE_PROFIT_MARKET", "closePosition": True, "stopPrice": "104", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["sl_price"], 98.0)
        self.assertEqual(position["tp1_price"], 102.0)
        self.assertEqual(position["tp2_price"], 104.0)
        self.assertEqual(position["tp1_quantity"], 0.8)
        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")
        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertTrue(position["trade_id"].startswith("BTCUSDT_RECOVERED_"))

    def test_only_sl_and_tp2_found_means_tp1_already_resolved(self):
        manager = PositionManager()
        open_orders = [
            {"type": "STOP_MARKET", "triggerPrice": "100.02", "algoId": "sl2"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "104", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertEqual(position["tp1_order_id"], "")

    def test_tp1_pending_adoption_stores_a_real_risk_distance(self):
        # SL is still the genuine original here (TP1 hasn't resolved yet),
        # so entry-to-sl is a trustworthy original risk distance.
        manager = PositionManager()
        open_orders = [
            {"type": "STOP_MARKET", "triggerPrice": "98", "algoId": "sl1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["risk_distance"], 2.0)

    def test_adopted_position_has_no_break_confirmation_data(self):
        # No original signal/candle to check this against - stays
        # unresolved forever, same honesty policy as confluence_ratio.
        manager = PositionManager()
        open_orders = [{"type": "STOP_MARKET", "triggerPrice": "98", "algoId": "sl1"}]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertIsNone(position["structure_level"])
        self.assertIsNone(position["trigger_candle_open_time"])
        self.assertIsNone(position["break_confirmed_by_close"])

    def test_breakeven_active_adoption_has_no_recoverable_risk_distance(self):
        # Real bug found live (2026-08-09): sl here is the REAL exchange
        # order's price, which by BREAKEVEN_ACTIVE time is the breakeven
        # price (~entry), not the true original stop - that original
        # distance was lost before this restart. Storing
        # abs(entry-sl)=0.02 here instead of None would silently
        # reintroduce the billions-R bug the next time this position's
        # MAE/MFE gets computed at close.
        manager = PositionManager()
        open_orders = [
            {"type": "STOP_MARKET", "triggerPrice": "100.02", "algoId": "sl2"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "104", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertIsNone(position["risk_distance"])

    def test_early_breakeven_promoted_position_is_recognized_even_with_tp1_still_open(self):
        # Real bug found live (2026-08-10): the early-1R trigger moves the
        # stop WITHOUT touching TP1 (still a live, resting order) - so
        # "TP1 order absent" alone can no longer tell "already promoted"
        # apart from "still TP1_PENDING". A stop already sitting on the
        # profit side of entry is unambiguous either way. Getting this
        # wrong doesn't just corrupt risk_distance (same billions-R class
        # of bug via a smaller, filter-evading magnitude) - it also means
        # an eventual stop-out on this position would log as a full
        # SL_HIT instead of the breakeven it actually is.
        manager = PositionManager()
        open_orders = [
            # BUY, entry=100 - a stop at 100.02 is above entry, only
            # possible if this SL was already moved to breakeven.
            {"type": "STOP_MARKET", "triggerPrice": "100.02", "algoId": "sl2"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "false", "triggerPrice": "102", "quantity": "0.5", "algoId": "tp1_1"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "104", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertIsNone(position["risk_distance"])
        # TP1 tracking is still correctly wired up too - it's a real,
        # still-open order, not something to self-heal/recover.
        self.assertEqual(position["tp1_order_id"], "tp1_1")

    def test_sell_side_stop_on_profit_side_is_also_recognized(self):
        # SELL, entry=100 - a stop at or below entry is on the profit
        # side (mirrors the BUY case, where profit-side is above entry).
        manager = PositionManager()
        open_orders = [
            {"type": "STOP_MARKET", "triggerPrice": "99.98", "algoId": "sl2"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "false", "triggerPrice": "98", "quantity": "0.5", "algoId": "tp1_1"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "96", "algoId": "tp2_1"},
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(side="SELL", entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=open_orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertIsNone(position["risk_distance"])

    def test_no_stop_loss_found_places_an_emergency_stop(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl, \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.3):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        place_sl.assert_called_once()
        self.assertEqual(position["sl_order_id"], "emergency_sl")
        # Emergency stop is at least the configured minimum distance away.
        self.assertLessEqual(position["sl_price"], 100.0 * (1 - 0.003))

    def test_no_orders_at_all_still_produces_a_trackable_position(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["tp1_order_id"], "")
        self.assertEqual(position["tp2_order_id"], "")
        self.assertIsNotNone(position["tp1_price"])
        self.assertIsNotNone(position["tp2_price"])

    def test_adopted_position_starts_with_no_trailing_profit_locked(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}):
            manager.reconcile_on_startup()

        self.assertFalse(manager.positions["BTCUSDT"]["trailing_stop_locked_profit"])

    def test_emergency_stop_placement_failure_does_not_raise(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position()]), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")):
            manager.reconcile_on_startup()  # must not raise

        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(manager.positions["BTCUSDT"]["sl_order_id"], "")

    def _dca_shape_open_orders(self):
        # No STOP_MARKET at all - TP1 (partial) + TP2 (full) both resting,
        # exactly what execution.enter_trade_dca_pending produces and
        # nothing else in this codebase does. See _recover_dca_pending_position.
        return [
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "false", "triggerPrice": "102", "origQty": "0.8", "algoId": "tp1_1"},
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "104", "algoId": "tp2_1"},
        ]

    def _single_tp_dca_shape_open_orders(self):
        # config.TP_STATIC_ROI_ENABLED shape - exactly ONE full-position
        # TP resting, no partial, no SL. See
        # _recover_dca_pending_single_tp_position.
        return [
            {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "triggerPrice": "106", "algoId": "tp_solo"},
        ]

    class _FakeCandleStore:
        def __init__(self, candles):
            self._candles = candles

        def get(self, symbol):
            return self._candles

    class _FakeFeed:
        def __init__(self, candles):
            self.candles = ReconcileOnStartupTests._FakeCandleStore(candles)

    def test_dca_pending_shape_with_feed_recovers_dca_pending_stage(self):
        manager = PositionManager()
        feed = self._FakeFeed(["candle"])  # non-empty is all _adopt_position checks

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]), \
             patch.object(market_structure, "average_true_range", return_value=0.5), \
             patch.object(risk_manager, "compute_dca_price", return_value=96.0) as compute_dca, \
             patch.object(exchange, "place_stop_loss") as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["dca_price"], 96.0)
        self.assertIsNone(position["sl_order_id"])
        self.assertFalse(position["dca_applied"])
        self.assertEqual(position["tp1_price"], 102.0)
        self.assertEqual(position["tp2_price"], 104.0)
        self.assertEqual(position["tp1_quantity"], 0.8)
        self.assertIsNone(position["risk_distance"])  # honest - no recoverable original stop
        place_sl.assert_not_called()  # no emergency stop - the DCA mechanism survived instead
        compute_dca.assert_called_once_with(100.0, "BUY", [], atr=0.5)

    def test_dca_pending_shape_without_feed_falls_back_to_emergency_stop(self):
        # No feed passed (matches every existing call site/test above,
        # and a real run with WS_ENABLED off) - can't recompute dca_price
        # without candles, so this must degrade to the same safe,
        # already-proven emergency-stop path as any other no-SL position,
        # not silently drop the position or crash.
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl:
            manager.reconcile_on_startup()  # feed defaults to None

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        place_sl.assert_called_once()
        self.assertEqual(position["sl_order_id"], "emergency_sl")

    def test_dca_pending_shape_but_dca_disabled_falls_back_to_emergency_stop(self):
        # Same TP1+TP2-no-SL shape, but DCA_ENABLED=False means this
        # codebase's only producer of that shape isn't active - treat it
        # like any other anomalous no-SL position instead of assuming DCA.
        manager = PositionManager()
        feed = self._FakeFeed(["candle"])

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", False), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        place_sl.assert_called_once()

    def test_dca_pending_recovery_with_no_candles_falls_back_to_emergency_stop(self):
        manager = PositionManager()
        feed = self._FakeFeed([])  # feed present but empty history

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        place_sl.assert_called_once()

    def test_dca_pending_recovery_with_uncomputable_dca_price_falls_back(self):
        manager = PositionManager()
        feed = self._FakeFeed(["candle"])

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]), \
             patch.object(market_structure, "average_true_range", return_value=0.5), \
             patch.object(risk_manager, "compute_dca_price", return_value=None), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        place_sl.assert_called_once()

    def test_single_tp_dca_pending_shape_with_feed_recovers_dca_pending_stage(self):
        manager = PositionManager()
        feed = self._FakeFeed(["candle"])

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._single_tp_dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]), \
             patch.object(market_structure, "average_true_range", return_value=0.5), \
             patch.object(risk_manager, "compute_dca_price", return_value=96.0) as compute_dca, \
             patch.object(exchange, "place_stop_loss") as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertTrue(position["single_tp"])
        self.assertEqual(position["dca_price"], 96.0)
        self.assertIsNone(position["sl_order_id"])
        self.assertFalse(position["dca_applied"])
        self.assertEqual(position["tp_price"], 106.0)
        self.assertEqual(position["tp_order_id"], "tp_solo")
        self.assertIsNone(position["tp1_price"])
        self.assertIsNone(position["tp2_price"])
        self.assertIsNone(position["risk_distance"])  # honest - no recoverable original stop
        place_sl.assert_not_called()  # no emergency stop - the DCA mechanism survived instead
        compute_dca.assert_called_once_with(100.0, "BUY", [], atr=0.5)

    def test_single_tp_dca_pending_recovery_with_no_candles_falls_back_to_emergency_stop(self):
        manager = PositionManager()
        feed = self._FakeFeed([])  # feed present but empty history

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=100.0)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._single_tp_dca_shape_open_orders()), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "emergency_sl"}) as place_sl:
            manager.reconcile_on_startup(feed)

        position = manager.positions["BTCUSDT"]
        place_sl.assert_called_once()
        self.assertEqual(position["sl_order_id"], "emergency_sl")

    def _dca_active_open_orders(self, sl_tag="dcaSL1787000000000", tp_tag="dcaTP1787000000000"):
        # Same shape a genuine post-DCA position always has: one real
        # full-position SL, one real full-position TP, no partial TP -
        # the tag (see _execute_dca) is the only thing that lets
        # _adopt_position tell this apart from an ordinary BREAKEVEN_
        # ACTIVE position, which looks identical otherwise.
        return [
            {
                "type": "STOP_MARKET", "closePosition": "true",
                "triggerPrice": "0.038", "algoId": "sl_real", "clientAlgoId": sl_tag,
            },
            {
                "type": "TAKE_PROFIT_MARKET", "closePosition": "true",
                "triggerPrice": "0.043", "algoId": "tp_real", "clientAlgoId": tp_tag,
            },
        ]

    def test_dca_active_shape_with_the_tag_is_recovered_as_dca_active(self):
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_active_open_orders()):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertEqual(position["sl_price"], 0.038)
        self.assertEqual(position["tp_price"], 0.043)
        self.assertEqual(position["sl_order_id"], "sl_real")
        self.assertEqual(position["tp_order_id"], "tp_real")
        self.assertTrue(position["dca_applied"])
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertIsNone(position["risk_distance"])  # honest - no original stop survives a restart
        # DCA_ACTIVE has no TP1/TP2 concept - the keys exist (every
        # position dict shape does, config.TP_STATIC_ROI_ENABLED needs
        # this consistency) but are always None here.
        self.assertIsNone(position["tp1_price"])
        self.assertIsNone(position["tp2_price"])
        self.assertTrue(position["single_tp"])
        # config.DCA_PRESSURE_CHECK_ENABLED - no record of the check
        # survives a restart, same honesty policy as risk_distance above.
        self.assertIsNone(position["dca_pressure_confirmed"])

    def test_same_shape_without_the_tag_is_not_treated_as_dca_active(self):
        # An ordinary post-TP1 BREAKEVEN_ACTIVE position (SL already
        # promoted, TP1 already resolved) has this exact same order
        # shape - only the missing tag distinguishes it, and it must
        # fall through to the existing generic adoption logic instead.
        manager = PositionManager()
        orders = self._dca_active_open_orders()
        orders[0].pop("clientAlgoId")  # no tag at all - real orders predating this fix

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertNotIn("dca_applied", position)

    def test_a_different_clientalgoid_is_not_treated_as_dca_active(self):
        # Only this codebase's own dcaSL-prefixed tag counts - an
        # unrelated auto-generated clientAlgoId (every real order has
        # one) must not false-positive.
        manager = PositionManager()
        orders = self._dca_active_open_orders(sl_tag="x-Cb7ytekJ7f08390857d3692432277d")

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)

    def test_dca_active_recovery_finds_a_resting_dcatrail_tagged_order(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - Binance keeps a
        # real TRAILING_STOP_MARKET resting across a bot restart; this
        # only has to notice it via the same clientAlgoId-based
        # disambiguation the SL/TP tags already use, not recreate it.
        manager = PositionManager()
        orders = self._dca_active_open_orders() + [
            {
                "type": "TRAILING_STOP_MARKET", "closePosition": "true",
                "algoId": "trail_real", "clientAlgoId": "dcaTrail1787000000000",
            },
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertEqual(position["dca_trail_order_id"], "trail_real")

    def test_dca_active_recovery_without_a_trail_order_leaves_it_none(self):
        # The normal case - no flag, no trail order ever placed.
        manager = PositionManager()

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=self._dca_active_open_orders()):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertIsNone(position["dca_trail_order_id"])

    def test_dca_active_recovery_ignores_an_untagged_trailing_stop_market(self):
        # Only the dcaTrail-prefixed tag counts - an unrelated real
        # TRAILING_STOP_MARKET (not this feature's doing) must not
        # false-positive.
        manager = PositionManager()
        orders = self._dca_active_open_orders() + [
            {
                "type": "TRAILING_STOP_MARKET", "closePosition": "true",
                "algoId": "trail_unrelated", "clientAlgoId": "x-Cb7ytekJ7f08390857d3692432277d",
            },
        ]

        with patch.object(exchange, "get_all_open_positions", return_value=[self._live_position(entry=0.0404)]), \
             patch.object(exchange, "get_open_algo_orders", return_value=orders):
            manager.reconcile_on_startup()

        position = manager.positions["BTCUSDT"]
        self.assertIsNone(position["dca_trail_order_id"])


class RealShadowOpenCountTests(unittest.TestCase):
    """open_count() alone conflates real exchange exposure with shadow-only
    tracking (config.SHADOW_ONLY_TRIGGERS) - real_open_count()/
    shadow_open_count() split it so the heartbeat's OPEN_POSITIONS can mean
    "real trades" again."""

    def _add(self, manager, symbol, shadow):
        manager.positions[symbol] = dict(_plan(), stage=DCA_PENDING, shadow=shadow, trade_id=symbol)

    def test_all_real_counts_only_as_real(self):
        manager = PositionManager()
        self._add(manager, "BTCUSDT", shadow=False)
        self._add(manager, "ETHUSDT", shadow=False)

        self.assertEqual(manager.real_open_count(), 2)
        self.assertEqual(manager.shadow_open_count(), 0)
        self.assertEqual(manager.open_count(), 2)

    def test_all_shadow_counts_only_as_shadow(self):
        manager = PositionManager()
        self._add(manager, "BTCUSDT", shadow=True)
        self._add(manager, "ETHUSDT", shadow=True)

        self.assertEqual(manager.real_open_count(), 0)
        self.assertEqual(manager.shadow_open_count(), 2)
        self.assertEqual(manager.open_count(), 2)

    def test_mixed_real_and_shadow_split_correctly(self):
        manager = PositionManager()
        self._add(manager, "BTCUSDT", shadow=False)
        self._add(manager, "ETHUSDT", shadow=True)
        self._add(manager, "SOLUSDT", shadow=True)

        self.assertEqual(manager.real_open_count(), 1)
        self.assertEqual(manager.shadow_open_count(), 2)
        self.assertEqual(manager.open_count(), 3)

    def test_empty_manager_is_zero_both(self):
        manager = PositionManager()

        self.assertEqual(manager.real_open_count(), 0)
        self.assertEqual(manager.shadow_open_count(), 0)


class ReconcileClosedPositionsTests(unittest.TestCase):
    """poll_live only ever detects a real position's close by watching
    specific remembered order ids reach FINISHED - a manual close on the
    exchange (or ADL/liquidation) auto-cancels those same orders instead,
    which poll_live's checks can never recognize. reconcile_closed_positions
    is the separate reconciliation pass that catches exactly that gap -
    real incident (2026-08-29): TRXUSDT/XPINUSDT both closed by hand on the
    exchange, left an orphaned resting DCA order live indefinitely."""

    def _tracked_position(self, stage, shadow=False):
        manager = PositionManager()
        manager.positions["BTCUSDT"] = dict(_plan(), stage=stage, shadow=shadow, trade_id="t1")
        return manager

    def test_position_still_real_is_left_alone(self):
        manager = self._tracked_position(DCA_PENDING)

        with patch.object(exchange, "_fetch_all_open_positions", return_value=[{"symbol": "BTCUSDT"}]), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            manager.reconcile_closed_positions()

        self.assertIn("BTCUSDT", manager.positions)
        cancel_all.assert_not_called()

    def test_position_gone_in_eligible_stage_is_cleaned_up_and_closed(self):
        for stage in (TP1_PENDING, BREAKEVEN_ACTIVE, DCA_PENDING, DCA_ACTIVE):
            with self.subTest(stage=stage):
                manager = self._tracked_position(stage)

                with patch.object(exchange, "_fetch_all_open_positions", return_value=[]), \
                     patch.object(exchange, "cancel_all_open_orders") as cancel_all:
                    manager.reconcile_closed_positions()

                cancel_all.assert_called_once_with("BTCUSDT")
                self.assertNotIn("BTCUSDT", manager.positions)

    def test_shadow_position_is_never_touched(self):
        manager = self._tracked_position(DCA_PENDING, shadow=True)

        with patch.object(exchange, "_fetch_all_open_positions", return_value=[]), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            manager.reconcile_closed_positions()

        self.assertIn("BTCUSDT", manager.positions)
        cancel_all.assert_not_called()

    def test_pending_limit_fill_stage_is_never_touched(self):
        manager = self._tracked_position(PENDING_LIMIT_FILL)

        with patch.object(exchange, "_fetch_all_open_positions", return_value=[]), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            manager.reconcile_closed_positions()

        self.assertIn("BTCUSDT", manager.positions)
        cancel_all.assert_not_called()

    def test_retracement_pending_stage_is_never_touched(self):
        manager = self._tracked_position(RETRACEMENT_PENDING)

        with patch.object(exchange, "_fetch_all_open_positions", return_value=[]), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            manager.reconcile_closed_positions()

        self.assertIn("BTCUSDT", manager.positions)
        cancel_all.assert_not_called()

    def test_failed_fetch_fails_open_not_closed(self):
        manager = self._tracked_position(DCA_PENDING)

        with patch.object(exchange, "_fetch_all_open_positions", side_effect=Exception("timeout")), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            manager.reconcile_closed_positions()  # must not raise

        self.assertIn("BTCUSDT", manager.positions)
        cancel_all.assert_not_called()

    def test_outcome_string_is_closed_externally(self):
        manager = self._tracked_position(DCA_ACTIVE)

        with patch.object(exchange, "_fetch_all_open_positions", return_value=[]), \
             patch.object(exchange, "cancel_all_open_orders"), \
             patch("position_manager.signal_journal.append_outcome") as append_outcome:
            manager.reconcile_closed_positions()

        append_outcome.assert_called_once()
        self.assertEqual(append_outcome.call_args.args[1], "CLOSED_EXTERNALLY")

    def test_tp_finished_this_tick_gets_its_real_outcome_not_the_generic_one(self):
        # Regression guard for the exact ordering main._poll_positions
        # relies on: poll_live must get first crack at a real fill before
        # reconcile_closed_positions ever runs, or every ordinary TP/SL
        # fill would get mislabeled CLOSED_EXTERNALLY.
        manager = PositionManager()
        manager.positions["BTCUSDT"] = dict(
            _plan(), stage=DCA_PENDING, shadow=False, trade_id="t1",
            single_tp=True, tp_order_id="tp_order_1",
        )

        with patch.object(exchange, "get_algo_order_status", return_value="FINISHED"), \
             patch.object(exchange, "cancel_all_open_orders"), \
             patch("position_manager.signal_journal.append_outcome") as append_outcome:
            outcome = manager.poll_live("BTCUSDT")

            # Simulates the exchange snapshot fetched moments later this
            # same tick already reflecting the fill.
            with patch.object(exchange, "_fetch_all_open_positions", return_value=[]):
                manager.reconcile_closed_positions()

        self.assertEqual(outcome, "STATIC_TP_HIT")
        self.assertNotIn("BTCUSDT", manager.positions)  # already closed by poll_live
        append_outcome.assert_called_once()
        self.assertEqual(append_outcome.call_args.args[1], "STATIC_TP_HIT")  # not CLOSED_EXTERNALLY


class ReentryCooldownTests(unittest.TestCase):
    def test_symbol_never_closed_is_not_in_cooldown(self):
        manager = PositionManager()
        self.assertFalse(manager.is_in_cooldown("BTCUSDT"))

    def test_symbol_closed_recently_is_in_cooldown(self):
        manager = PositionManager()
        manager._closed_at["BTCUSDT"] = time.time()

        with patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 900):
            self.assertTrue(manager.is_in_cooldown("BTCUSDT"))

    def test_close_starts_the_cooldown(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})

        with patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 900), \
             patch("position_manager.signal_journal.append_outcome"):
            manager._close("BTCUSDT", "SHADOW_SL_HIT")

        self.assertTrue(manager.is_in_cooldown("BTCUSDT"))

    def test_cooldown_expires_after_the_configured_window(self):
        manager = PositionManager()
        manager._closed_at["BTCUSDT"] = 0.0  # far in the past

        with patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 1):
            self.assertFalse(manager.is_in_cooldown("BTCUSDT"))

    def test_zero_cooldown_disables_it(self):
        manager = PositionManager()
        manager._closed_at["BTCUSDT"] = time.time()

        with patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 0):
            self.assertFalse(manager.is_in_cooldown("BTCUSDT"))

    def test_mark_entry_failure_starts_the_cooldown_too(self):
        # Real bug found live (STGUSDT/DEXEUSDT, 2026-08-08): a failed
        # entry never reaches register()/_close(), so is_in_cooldown()'s
        # only trigger never fired for it - a symbol that keeps failing
        # entry for a persistent reason got retried on every single eval
        # cycle with no backoff at all.
        manager = PositionManager()

        with patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 900):
            self.assertFalse(manager.is_in_cooldown("STGUSDT"))
            manager.mark_entry_failure("STGUSDT")
            self.assertTrue(manager.is_in_cooldown("STGUSDT"))


class RegisterTests(unittest.TestCase):
    def test_shadow_registration_has_no_order_ids(self):
        manager = PositionManager()
        position = manager.register(_plan(), {"shadow": True})

        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(manager.open_count(), 1)
        self.assertTrue(position["shadow"])
        self.assertIsNone(position["sl_order_id"])
        self.assertEqual(position["stage"], TP1_PENDING)

    def test_no_real_entry_price_uses_the_planned_entry_unchanged(self):
        # No "real_entry_price" key at all (shadow mode's own execution_
        # result shape) - must be byte-identical to the old behavior.
        manager = PositionManager()
        position = manager.register(_plan(), {"shadow": True})

        self.assertEqual(position["entry_price"], 100)
        self.assertEqual(position["breakeven_price"], 100.02)  # _plan()'s own value, untouched
        self.assertEqual(position["risk_distance"], 2.0)
        self.assertEqual(position["mae_price"], 100)
        self.assertEqual(position["mfe_price"], 100)

    def test_real_entry_price_corrects_entry_breakeven_and_risk_distance(self):
        # entry=100 planned, sl=98 (_plan()) - a real fill at 100.5 (0.5%
        # slippage) must NOT touch sl_price/tp1_price/tp2_price (real
        # structure levels, unaffected by entry slippage), but does
        # correct entry_price itself, the breakeven derived from it, and
        # risk_distance (98 -> 100.5, was 100 -> 98 = 2.0, now 2.5).
        manager = PositionManager()
        execution_result = {"shadow": False, "real_entry_price": 100.5}

        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            position = manager.register(_plan(), execution_result)

        self.assertEqual(position["entry_price"], 100.5)
        self.assertEqual(position["sl_price"], 98)  # unshifted
        self.assertEqual(position["tp1_price"], 102)  # unshifted
        self.assertEqual(position["tp2_price"], 104)  # unshifted
        self.assertAlmostEqual(position["breakeven_price"], 100.5 * 1.0002)
        self.assertEqual(position["risk_distance"], 2.5)
        self.assertEqual(position["mae_price"], 100.5)
        self.assertEqual(position["mfe_price"], 100.5)

    def test_real_entry_price_matching_the_plan_exactly_is_a_noop(self):
        manager = PositionManager()
        execution_result = {"shadow": False, "real_entry_price": 100}
        position = manager.register(_plan(), execution_result)

        self.assertEqual(position["entry_price"], 100)
        self.assertEqual(position["breakeven_price"], 100.02)  # _plan()'s own value
        self.assertEqual(position["risk_distance"], 2.0)

    def test_real_entry_price_of_zero_is_treated_as_unavailable(self):
        # exchange.resolve_market_fill_price never actually returns 0 (it
        # falls back to the planned price itself first) - defends against
        # that invariant being violated rather than relying on it blindly.
        manager = PositionManager()
        execution_result = {"shadow": False, "real_entry_price": 0}
        position = manager.register(_plan(), execution_result)

        self.assertEqual(position["entry_price"], 100)
        self.assertEqual(position["breakeven_price"], 100.02)

    def test_trade_id_is_stored_and_threaded_through_to_the_outcome_journal(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True}, trade_id="BTCUSDT_123456")
        self.assertEqual(manager.positions["BTCUSDT"]["trade_id"], "BTCUSDT_123456")

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            outcome = manager._close("BTCUSDT", "SHADOW_SL_HIT")

        self.assertEqual(outcome, "SHADOW_SL_HIT")
        args, kwargs = append_outcome.call_args
        self.assertEqual(args, ("BTCUSDT", "SHADOW_SL_HIT", "BTCUSDT_123456"))
        self.assertIn("mae_r_multiple", kwargs)
        self.assertIn("mfe_r_multiple", kwargs)
        self.assertEqual(kwargs["early_breakeven_applied"], False)

    def test_close_reports_early_breakeven_applied_when_it_fired(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True}, trade_id="BTCUSDT_123456")
        manager.positions["BTCUSDT"]["early_breakeven_applied"] = True

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            manager._close("BTCUSDT", "SHADOW_BREAKEVEN_STOP_HIT")

        _, kwargs = append_outcome.call_args
        self.assertEqual(kwargs["early_breakeven_applied"], True)

    def test_close_reports_dca_pressure_confirmed_when_the_check_ran(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True}, trade_id="BTCUSDT_123456")
        manager.positions["BTCUSDT"]["dca_pressure_confirmed"] = False

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            manager._close("BTCUSDT", "SHADOW_SL_HIT")

        _, kwargs = append_outcome.call_args
        self.assertEqual(kwargs["dca_pressure_confirmed"], False)

    def test_close_reports_dca_pressure_confirmed_as_none_when_it_never_ran(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True}, trade_id="BTCUSDT_123456")

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            manager._close("BTCUSDT", "SHADOW_SL_HIT")

        _, kwargs = append_outcome.call_args
        self.assertIsNone(kwargs["dca_pressure_confirmed"])

    def test_live_registration_extracts_order_ids(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        position = manager.register(_plan(), execution_result)

        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")

    def test_confluence_ratio_is_carried_from_the_plan(self):
        manager = PositionManager()
        position = manager.register(dict(_plan(), confluence_ratio=0.25), {"shadow": True})

        self.assertEqual(position["confluence_ratio"], 0.25)
        self.assertFalse(position["early_breakeven_applied"])

    def test_risk_distance_is_read_from_the_plan_when_present(self):
        manager = PositionManager()
        position = manager.register(dict(_plan(), risk_distance=1.75), {"shadow": True})

        self.assertEqual(position["risk_distance"], 1.75)

    def test_risk_distance_falls_back_to_entry_minus_sl_when_missing_from_plan(self):
        # _plan() fixture: entry=100, sl=98
        manager = PositionManager()
        position = manager.register(_plan(), {"shadow": True})

        self.assertEqual(position["risk_distance"], 2)

    def test_break_confirmation_fields_are_carried_from_the_plan(self):
        manager = PositionManager()
        position = manager.register(
            dict(_plan(), structure_level=98.5, trigger_candle_open_time=12345),
            {"shadow": True},
        )

        self.assertEqual(position["structure_level"], 98.5)
        self.assertEqual(position["trigger_candle_open_time"], 12345)
        self.assertIsNone(position["break_confirmed_by_close"])


class EarlyBreakevenEligibilityTests(unittest.TestCase):
    """config.EARLY_BREAKEVEN_ENABLED - every trade still waiting on TP1
    gets pulled to breakeven once it's moved EARLY_BREAKEVEN_R_MULTIPLE R
    in profit (evidence: 2026-08-10, 28% of LOSS trades ran 1.0R+ before
    fully reversing, completely unprotected). No longer gated on
    confluence_ratio - that was the original design (2026-08-09) but real
    data showed confluence didn't correlate with outcome at all. These
    test the eligibility check in isolation from any exchange or shadow-
    candle mechanics."""

    def _position(self, **overrides):
        position = {
            "side": "BUY",
            "entry_price": 100,
            "sl_price": 98,
            "breakeven_price": 100.02,
            "stage": TP1_PENDING,
            "confluence_ratio": 0.25,
            "early_breakeven_applied": False,
        }
        position.update(overrides)
        return position

    def test_disabled_config_is_never_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", False):
            self.assertFalse(manager._is_early_breakeven_candidate(self._position()))

    def test_already_applied_is_never_a_candidate_again(self):
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertFalse(manager._is_early_breakeven_candidate(
                self._position(early_breakeven_applied=True)
            ))

    def test_wrong_stage_is_not_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertFalse(manager._is_early_breakeven_candidate(
                self._position(stage=BREAKEVEN_ACTIVE)
            ))

    def test_dca_pending_is_also_a_candidate(self):
        # Real gap found live (2026-08-17, operator question): a DCA-
        # pending position is ALSO still waiting on TP1 with no real SL
        # placed yet - the same condition TP1_PENDING represents. Before
        # this fix, this stage check was `!= TP1_PENDING` only, so while
        # config.DCA_ENABLED is True (every position starts in
        # DCA_PENDING) this could never arm at all.
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertTrue(manager._is_early_breakeven_candidate(
                self._position(stage=DCA_PENDING)
            ))

    def test_single_tp_dca_pending_is_never_a_candidate(self):
        # config.TP_STATIC_ROI_ENABLED - deliberately kept simple: no
        # early-arm mechanisms layered on a single-TP DCA_PENDING
        # position, just the DCA-vs-single-TP race.
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertFalse(manager._is_early_breakeven_candidate(
                self._position(stage=DCA_PENDING, single_tp=True)
            ))

    def test_zero_risk_distance_is_not_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertFalse(manager._is_early_breakeven_candidate(
                self._position(entry_price=100, sl_price=100)
            ))

    def test_any_confluence_ratio_is_a_candidate_including_none(self):
        # Real bug this replaces: gating on confluence_ratio meant a
        # position with no original signal (e.g. startup-reconciliation-
        # adopted) or a high-confluence trade never got this protection
        # at all - neither restriction is evidence-backed anymore.
        manager = PositionManager()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True):
            self.assertTrue(manager._is_early_breakeven_candidate(
                self._position(confluence_ratio=None)
            ))
            self.assertTrue(manager._is_early_breakeven_candidate(
                self._position(confluence_ratio=0.75)
            ))
            self.assertTrue(manager._is_early_breakeven_candidate(
                self._position(confluence_ratio=0.25)
            ))


class ProfitProtectionEligibilityTests(unittest.TestCase):
    """config.PROFIT_PROTECTION_ENABLED - same shape as
    EarlyBreakevenEligibilityTests above, a different metric (% of TP1's
    own ROI instead of an R-multiple of risk_distance)."""

    def _position(self, **overrides):
        position = {
            "side": "BUY",
            "entry_price": 100,
            "tp1_price": 110,
            "stage": TP1_PENDING,
            "profit_protection_applied": False,
        }
        position.update(overrides)
        return position

    def test_disabled_config_is_never_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False):
            self.assertFalse(manager._is_profit_protection_candidate(self._position()))

    def test_already_applied_is_never_a_candidate_again(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True):
            self.assertFalse(manager._is_profit_protection_candidate(
                self._position(profit_protection_applied=True)
            ))

    def test_wrong_stage_is_not_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True):
            self.assertFalse(manager._is_profit_protection_candidate(
                self._position(stage=BREAKEVEN_ACTIVE)
            ))

    def test_missing_tp1_price_is_not_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True):
            self.assertFalse(manager._is_profit_protection_candidate(
                self._position(tp1_price=None)
            ))

    def test_otherwise_eligible_position_is_a_candidate(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False):
            self.assertTrue(manager._is_profit_protection_candidate(self._position()))

    def test_single_tp_dca_pending_is_never_a_candidate(self):
        # config.TP_STATIC_ROI_ENABLED - see the identical note in
        # EarlyBreakevenEligibilityTests. Checked explicitly (not just
        # relying on tp1_price being None for this shape) so this stays
        # true even if that ever changes.
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True):
            self.assertFalse(manager._is_profit_protection_candidate(
                self._position(stage=DCA_PENDING, single_tp=True, tp1_price=None, tp_price=106)
            ))

    def test_dca_pending_is_also_a_candidate(self):
        # See EarlyBreakevenEligibilityTests.test_dca_pending_is_also_a_
        # candidate - identical real gap, same fix.
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False):
            self.assertTrue(manager._is_profit_protection_candidate(
                self._position(stage=DCA_PENDING)
            ))

    def test_static_tp1_is_not_a_candidate_when_the_skip_flag_is_on(self):
        # config.PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED - real
        # motivation (2026-08-22): TP1 under TP_STATIC_ROI_ENABLED is a
        # small, fixed ROI% target, not the variable/potentially-large one
        # the activation tiers were built around (TREEUSDT, 2026-08-21:
        # armed at 8% of a 10%-ROI TP1, reversed before reaching it,
        # closed for ~4% - a fraction of an already-small target).
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED", True):
            self.assertFalse(manager._is_profit_protection_candidate(self._position()))

    def test_static_tp1_still_a_candidate_when_the_skip_flag_is_off(self):
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED", False):
            self.assertTrue(manager._is_profit_protection_candidate(self._position()))

    def test_structure_tp1_is_unaffected_by_the_skip_flag(self):
        # The skip only fires when TP_STATIC_ROI_ENABLED is ALSO True -
        # a structure-based TP1 (the flag off) keeps today's behavior
        # regardless of PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED.
        manager = PositionManager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED", True):
            self.assertTrue(manager._is_profit_protection_candidate(self._position()))


class Tp2ProfitProtectionEligibilityTests(unittest.TestCase):
    """config.PROFIT_PROTECTION_TP2_LEG_ENABLED - _is_tp2_profit_
    protection_candidate, same shape as ProfitProtectionEligibilityTests
    above but for the post-genuine-TP1-fill BREAKEVEN_ACTIVE leg."""

    def _position(self, **overrides):
        position = {
            "side": "BUY",
            "entry_price": 100,
            "tp2_price": 110,
            "stage": BREAKEVEN_ACTIVE,
            "profit_protection_applied": False,
        }
        position.update(overrides)
        return position

    def _is_candidate(self, **overrides):
        manager = PositionManager()
        return manager._is_tp2_profit_protection_candidate(self._position(**overrides))

    def test_disabled_master_flag_is_never_a_candidate(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True):
            self.assertFalse(self._is_candidate())

    def test_disabled_tp2_leg_flag_is_never_a_candidate(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", False):
            self.assertFalse(self._is_candidate())

    def test_already_applied_is_never_a_candidate_again(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True):
            self.assertFalse(self._is_candidate(profit_protection_applied=True))

    def test_wrong_stage_is_not_a_candidate(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True):
            self.assertFalse(self._is_candidate(stage=TP1_PENDING))
            self.assertFalse(self._is_candidate(stage=DCA_ACTIVE))
            self.assertFalse(self._is_candidate(stage=DCA_PENDING))

    def test_missing_tp2_price_is_not_a_candidate(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True):
            self.assertFalse(self._is_candidate(tp2_price=None))

    def test_otherwise_eligible_position_is_a_candidate(self):
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True):
            self.assertTrue(self._is_candidate())


class ProfitProtectionPriceReachedTests(unittest.TestCase):
    def _position(self, side="BUY", entry_price=100, tp1_price=110):
        return {"side": side, "entry_price": entry_price, "tp1_price": tp1_price}

    def test_none_price_is_not_reached(self):
        manager = PositionManager()

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60):
            self.assertFalse(
                manager._profit_protection_price_reached(self._position(), None)
            )

    def test_buy_reaches_trigger_at_the_lock_price(self):
        # lock price = 106 (see ComputeProfitProtectionLockPriceTests) -
        # tp1 ROI here is 100%, above PROFIT_PROTECTION_HIGH_TP1_ROI_
        # THRESHOLD_PCT's default (50), so it's pinned high to keep this
        # test on the plain (non-tiered) activation path it's actually
        # about - see ComputeProfitProtectionLockPriceHighTp1RoiTests.
        manager = PositionManager()

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            self.assertTrue(manager._profit_protection_price_reached(self._position(), 106))
            self.assertFalse(manager._profit_protection_price_reached(self._position(), 105.9))

    def test_sell_reaches_trigger_at_the_lock_price(self):
        manager = PositionManager()
        position = self._position(side="SELL", entry_price=100, tp1_price=90)

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            self.assertTrue(manager._profit_protection_price_reached(position, 94))
            self.assertFalse(manager._profit_protection_price_reached(position, 94.1))

    def test_lock_price_unavailable_is_never_reached(self):
        manager = PositionManager()

        with patch.object(config, "LEVERAGE", 10):
            self.assertFalse(manager._profit_protection_price_reached(
                self._position(tp1_price=None), 106
            ))


class EarlyBreakevenPriceReachedTests(unittest.TestCase):
    def _position(self, side="BUY", entry_price=100, sl_price=98):
        return {"side": side, "entry_price": entry_price, "sl_price": sl_price}

    def test_none_price_is_not_reached(self):
        with patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0):
            self.assertFalse(
                PositionManager._early_breakeven_price_reached(self._position(), None)
            )

    def test_buy_reaches_trigger_at_full_r_multiple(self):
        # risk_distance = 2 (100 - 98), R multiple 1.0 -> needs +2 favorable
        with patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0):
            self.assertTrue(
                PositionManager._early_breakeven_price_reached(self._position(), 102)
            )
            self.assertFalse(
                PositionManager._early_breakeven_price_reached(self._position(), 101)
            )

    def test_sell_reaches_trigger_at_full_r_multiple(self):
        position = self._position(side="SELL", entry_price=100, sl_price=102)

        with patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0):
            self.assertTrue(PositionManager._early_breakeven_price_reached(position, 98))
            self.assertFalse(PositionManager._early_breakeven_price_reached(position, 99))

    def test_smaller_r_multiple_triggers_earlier(self):
        with patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5):
            # only +1 favorable needed now (0.5 * risk_distance 2)
            self.assertTrue(
                PositionManager._early_breakeven_price_reached(self._position(), 101)
            )


class MaeMfeTrackingTests(unittest.TestCase):
    """config.MAE_TRACKING_ENABLED - the diagnostic that tells apart a
    trade that went wrong from the first tick (near-zero MFE) from one
    that was solidly in profit before fully reversing (large MFE), which
    look identical as a plain WIN/LOSS outcome."""

    def _position(self, side="BUY", entry_price=100, sl_price=98, **overrides):
        position = {
            "side": side, "entry_price": entry_price, "sl_price": sl_price,
            "mae_price": entry_price, "mfe_price": entry_price,
            "risk_distance": abs(entry_price - sl_price),
        }
        position.update(overrides)
        return position

    def test_disabled_config_leaves_mae_mfe_untouched(self):
        position = self._position()

        with patch.object(config, "MAE_TRACKING_ENABLED", False):
            PositionManager._update_mae_mfe(position, 90, 110)

        self.assertEqual(position["mae_price"], 100)
        self.assertEqual(position["mfe_price"], 100)

    def test_buy_tracks_low_as_adverse_and_high_as_favorable(self):
        position = self._position(side="BUY")

        with patch.object(config, "MAE_TRACKING_ENABLED", True):
            PositionManager._update_mae_mfe(position, 97, 103)

        self.assertEqual(position["mae_price"], 97)
        self.assertEqual(position["mfe_price"], 103)

    def test_sell_tracks_high_as_adverse_and_low_as_favorable(self):
        position = self._position(side="SELL", entry_price=100, sl_price=102)

        with patch.object(config, "MAE_TRACKING_ENABLED", True):
            PositionManager._update_mae_mfe(position, 97, 103)

        self.assertEqual(position["mae_price"], 103)
        self.assertEqual(position["mfe_price"], 97)

    def test_single_price_variant_updates_both_extremes_from_one_sample(self):
        position = self._position(side="BUY")

        with patch.object(config, "MAE_TRACKING_ENABLED", True):
            PositionManager._update_mae_mfe(position, 105)  # no high_or_price given

        self.assertEqual(position["mae_price"], 100)  # unchanged, 105 isn't adverse
        self.assertEqual(position["mfe_price"], 105)

    def test_extremes_only_ever_move_in_the_worse_or_better_direction(self):
        # A later, less-extreme sample must not undo an already-recorded
        # worst/best price.
        position = self._position(side="BUY")

        with patch.object(config, "MAE_TRACKING_ENABLED", True):
            PositionManager._update_mae_mfe(position, 95, 105)
            PositionManager._update_mae_mfe(position, 98, 101)

        self.assertEqual(position["mae_price"], 95)
        self.assertEqual(position["mfe_price"], 105)

    def test_none_price_is_ignored(self):
        position = self._position()

        with patch.object(config, "MAE_TRACKING_ENABLED", True):
            PositionManager._update_mae_mfe(position, None)

        self.assertEqual(position["mae_price"], 100)
        self.assertEqual(position["mfe_price"], 100)

    def test_r_multiples_are_normalized_to_risk_distance(self):
        # entry=100, sl=98 -> risk_distance=2
        position = self._position(mae_price=97, mfe_price=103)

        mae_r, mfe_r = PositionManager._mae_mfe_r_multiples(position)

        self.assertAlmostEqual(mae_r, 1.5)  # (100-97)/2
        self.assertAlmostEqual(mfe_r, 1.5)  # (103-100)/2

    def test_r_multiples_use_the_stored_risk_distance_not_the_live_sl_price(self):
        # Real bug found live (2026-08-09): once a trade is promoted to
        # breakeven, sl_price legitimately moves to ~entry_price (shadow
        # mode mutates it directly; a reconciled-while-already-breakeven
        # position picks it up from the real exchange order). Recomputing
        # risk_distance from that moved sl_price at close time divided
        # real MAE/MFE price distances by a near-zero breakeven-buffer
        # distance instead of the original ~2.0, producing R-multiples in
        # the billions. The fixed risk_distance field must be used
        # instead, completely ignoring wherever sl_price ended up.
        position = self._position(
            sl_price=98, mae_price=97, mfe_price=104,
            risk_distance=2.0,  # captured once at entry, before promotion
        )
        position["sl_price"] = 100.02  # moved to breakeven, as it really would be

        mae_r, mfe_r = PositionManager._mae_mfe_r_multiples(position)

        self.assertAlmostEqual(mae_r, 1.5)  # (100-97)/2, NOT /0.02
        self.assertAlmostEqual(mfe_r, 2.0)  # (104-100)/2, NOT /0.02

    def test_r_multiples_are_none_when_risk_distance_is_unknown(self):
        # The stored risk_distance can itself be None (a position adopted
        # via startup reconciliation while already BREAKEVEN_ACTIVE has no
        # recoverable original risk distance) - must stay honestly
        # unknown rather than falling back to a live sl_price-derived
        # value that would reintroduce the same bug.
        position = self._position(risk_distance=None)

        mae_r, mfe_r = PositionManager._mae_mfe_r_multiples(position)

        self.assertIsNone(mae_r)
        self.assertIsNone(mfe_r)

    def test_r_multiples_are_none_when_risk_distance_is_zero(self):
        position = self._position(entry_price=100, sl_price=100)

        mae_r, mfe_r = PositionManager._mae_mfe_r_multiples(position)

        self.assertIsNone(mae_r)
        self.assertIsNone(mfe_r)


class _FakeCandleStore:
    """Minimal stand-in for ws_client.CandleStore - resolve_break_confirmations
    only ever calls .get(symbol)."""

    def __init__(self, candles_by_symbol):
        self.candles_by_symbol = candles_by_symbol

    def get(self, symbol):
        return self.candles_by_symbol.get(symbol, [])


class ResolveBreakConfirmationsTests(unittest.TestCase):
    """The entry fires the instant a live/forming candle breaks structure
    - this is the look-back, a candle later, checking whether price
    actually held beyond the level once that candle finished, or snapped
    back inside first (just a wick, not a real break)."""

    def _manager_with_position(self, side="BUY", structure_level=98.0, trigger_candle_open_time=1000):
        manager = PositionManager()
        manager.register(
            dict(_plan(side), structure_level=structure_level, trigger_candle_open_time=trigger_candle_open_time),
            {"shadow": True},
        )
        return manager

    def test_stays_unresolved_while_the_trigger_candle_is_still_forming(self):
        manager = self._manager_with_position()
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 99.0, "closed": False}],
        })

        manager.resolve_break_confirmations(candles)

        self.assertIsNone(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_stays_unresolved_when_the_trigger_candle_is_not_in_the_buffer_yet(self):
        manager = self._manager_with_position()
        candles = _FakeCandleStore({"BTCUSDT": []})

        manager.resolve_break_confirmations(candles)

        self.assertIsNone(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_buy_break_confirmed_when_close_holds_above_the_level(self):
        # side=BUY, structure_level=98.0 - a bullish break of 98 that's
        # still closing above 98 once the candle finishes is a real break.
        manager = self._manager_with_position(side="BUY", structure_level=98.0)
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 99.5, "closed": True}],
        })

        manager.resolve_break_confirmations(candles)

        self.assertTrue(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_buy_break_rejected_when_close_snaps_back_below_the_level(self):
        manager = self._manager_with_position(side="BUY", structure_level=98.0)
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 97.5, "closed": True}],
        })

        manager.resolve_break_confirmations(candles)

        self.assertFalse(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_sell_break_confirmed_when_close_holds_below_the_level(self):
        manager = self._manager_with_position(side="SELL", structure_level=102.0)
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 101.0, "closed": True}],
        })

        manager.resolve_break_confirmations(candles)

        self.assertTrue(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_sell_break_rejected_when_close_snaps_back_above_the_level(self):
        manager = self._manager_with_position(side="SELL", structure_level=102.0)
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 102.5, "closed": True}],
        })

        manager.resolve_break_confirmations(candles)

        self.assertFalse(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_already_resolved_positions_are_not_recomputed(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["break_confirmed_by_close"] = True
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 50.0, "closed": True}],  # would flip to False
        })

        manager.resolve_break_confirmations(candles)

        self.assertTrue(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_missing_structure_level_or_trigger_time_is_skipped_without_crashing(self):
        # e.g. a startup-reconciliation-adopted position, which has no
        # original signal/candle data at all.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["structure_level"] = None
        candles = _FakeCandleStore({
            "BTCUSDT": [{"open_time": 1000, "close": 99.5, "closed": True}],
        })

        manager.resolve_break_confirmations(candles)  # must not raise

        self.assertIsNone(manager.positions["BTCUSDT"]["break_confirmed_by_close"])

    def test_close_journals_break_confirmed_by_close(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["break_confirmed_by_close"] = False

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            manager._close("BTCUSDT", "SHADOW_SL_HIT")

        _, kwargs = append_outcome.call_args
        self.assertEqual(kwargs["break_confirmed_by_close"], False)


class StructureStopCandidateTests(unittest.TestCase):
    """config.STRUCTURE_STOP_MANAGEMENT_ENABLED - the pure "what would
    structure say" query, independent of whether the feature is on."""

    def _position(self, side="BUY", entry_price=100):
        return {"side": side, "entry_price": entry_price}

    def test_none_when_candles_falsy(self):
        self.assertIsNone(_structure_stop_candidate(self._position(), None))
        self.assertIsNone(_structure_stop_candidate(self._position(), []))

    def test_none_when_structure_unavailable(self):
        with patch.object(market_structure, "structure_state", return_value={"available": False}):
            self.assertIsNone(_structure_stop_candidate(self._position(), ["candle"]))

    def test_none_when_no_swing_in_the_favorable_direction(self):
        with patch.object(
            market_structure, "structure_state",
            return_value={"available": True, "last_swing_low": None},
        ):
            self.assertIsNone(_structure_stop_candidate(self._position(side="BUY"), ["candle"]))

    def test_buy_uses_swing_when_it_beats_breakeven(self):
        with patch.object(
            market_structure, "structure_state",
            return_value={"available": True, "last_swing_low": 100.5},
        ):
            candidate = _structure_stop_candidate(self._position(side="BUY"), ["candle"])
        self.assertAlmostEqual(candidate, 100.5)  # 100.5 > breakeven(~100.02)

    def test_buy_clamps_to_breakeven_when_swing_is_worse(self):
        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02), patch.object(
            market_structure, "structure_state",
            return_value={"available": True, "last_swing_low": 99.0},
        ):
            candidate = _structure_stop_candidate(self._position(side="BUY"), ["candle"])
        self.assertAlmostEqual(candidate, 100.02)  # clamped up to breakeven, not 99.0

    def test_sell_uses_swing_when_it_beats_breakeven(self):
        with patch.object(
            market_structure, "structure_state",
            return_value={"available": True, "last_swing_high": 99.5},
        ):
            candidate = _structure_stop_candidate(self._position(side="SELL"), ["candle"])
        self.assertAlmostEqual(candidate, 99.5)  # 99.5 < breakeven(~99.98)

    def test_sell_clamps_to_breakeven_when_swing_is_worse(self):
        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02), patch.object(
            market_structure, "structure_state",
            return_value={"available": True, "last_swing_high": 101.0},
        ):
            candidate = _structure_stop_candidate(self._position(side="SELL"), ["candle"])
        self.assertAlmostEqual(candidate, 99.98)  # clamped down to breakeven, not 101.0


class MoreFavorableTests(unittest.TestCase):
    def test_buy_higher_is_more_favorable(self):
        self.assertTrue(_more_favorable("BUY", 101, 100))
        self.assertFalse(_more_favorable("BUY", 99, 100))
        self.assertFalse(_more_favorable("BUY", 100, 100))  # strict, not >=

    def test_sell_lower_is_more_favorable(self):
        self.assertTrue(_more_favorable("SELL", 99, 100))
        self.assertFalse(_more_favorable("SELL", 101, 100))
        self.assertFalse(_more_favorable("SELL", 100, 100))


class EarlyBreakevenLockPriceTests(unittest.TestCase):
    def _position(self, side="BUY", entry_price=100, risk_distance=2.0):
        return {"side": side, "entry_price": entry_price, "risk_distance": risk_distance}

    def test_disabled_ignores_candles_and_uses_the_fixed_distance(self):
        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", False), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 999},
             ):
            price = PositionManager._early_breakeven_lock_price(self._position(), ["candle"])

        self.assertAlmostEqual(price, 100.6)  # fixed lock only - structure ignored entirely

    def test_enabled_uses_the_structure_candidate_when_available(self):
        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 100.8},
             ):
            price = PositionManager._early_breakeven_lock_price(self._position(), ["candle"])

        self.assertAlmostEqual(price, 100.8)

    def test_enabled_falls_back_to_fixed_distance_when_no_swing(self):
        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3), \
             patch.object(market_structure, "structure_state", return_value={"available": False}):
            price = PositionManager._early_breakeven_lock_price(self._position(), ["candle"])

        self.assertAlmostEqual(price, 100.6)

    def test_enabled_falls_back_when_candles_missing(self):
        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3):
            price = PositionManager._early_breakeven_lock_price(self._position(), None)

        self.assertAlmostEqual(price, 100.6)


class ReplaceSlOrderTests(unittest.TestCase):
    def _manager_with_position(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(_plan(), execution_result)
        return manager

    def test_success_replaces_and_returns_replaced_true(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}) as place:
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason"
            )

        self.assertIsNone(outcome)
        self.assertTrue(replaced)
        self.assertEqual(manager.positions["BTCUSDT"]["sl_order_id"], "sl_new")
        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], 101.0)
        cancel.assert_called_once_with("BTCUSDT", "sl1")
        place.assert_called_once_with("BTCUSDT", "BUY", 101.0)

    def test_uses_the_real_exchange_order_not_a_stale_local_id(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["sl_order_id"] = "stale_local_id"
        real_sl = {"type": "STOP_MARKET", "closePosition": True, "algoId": "real_sl_on_exchange"}

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[real_sl]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}):
            manager._replace_sl_order(manager.positions["BTCUSDT"], 101.0, "test reason")

        cancel.assert_called_once_with("BTCUSDT", "real_sl_on_exchange")

    def test_ground_truth_check_fails_retries_without_replacing(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", side_effect=RuntimeError("timeout")), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss") as place:
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason"
            )

        self.assertIsNone(outcome)
        self.assertFalse(replaced)
        cancel.assert_not_called()
        place.assert_not_called()

    def test_already_closed_with_close_if_not_open_true_calls_close(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", return_value=None), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason", close_if_not_open=True,
            )

        self.assertEqual(outcome, "TP1_THEN_POSITION_ALREADY_CLOSED")
        self.assertFalse(replaced)
        cancel_all.assert_called_once_with("BTCUSDT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_already_closed_with_close_if_not_open_false_does_not_call_close(self):
        # Trailing (close_if_not_open=False) must never guess an outcome
        # here - it would misclassify a likely protected close as a LOSS
        # (TP1_THEN_POSITION_ALREADY_CLOSED). The next poll's own status
        # check resolves and journals the real outcome instead.
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", return_value=None), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason", close_if_not_open=False,
            )

        self.assertIsNone(outcome)
        self.assertFalse(replaced)
        cancel_all.assert_not_called()
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_minus_2021_delegates_to_close_remainder_at_market(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(
                 exchange, "place_stop_loss",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason"
            )

        self.assertEqual(outcome, "BREAKEVEN_TRIGGER_MARKET_CLOSE")  # neither profit flag set
        self.assertFalse(replaced)
        market_close.assert_called_once()

    def test_minus_2021_on_dca_active_position_closes_as_dca_sl_hit(self):
        # Real VPS evidence (2026-08-17, SCRUSDT): DCA only fires while
        # price is already moving hard against the position, so by the
        # time the post-DCA SL order goes out, price has often already
        # traded through it - this -2021 fallback is DCA_ACTIVE's most
        # common real SL-hit path, not an edge case. It must not fall
        # through to the generic pre-DCA BREAKEVEN_TRIGGER_MARKET_CLOSE
        # label, which hid this exact loss from DCA-specific analysis.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = DCA_ACTIVE

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(
                 exchange, "place_stop_loss",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason"
            )

        self.assertEqual(outcome, "DCA_SL_HIT")
        self.assertFalse(replaced)
        market_close.assert_called_once()

    def test_other_error_logs_and_does_not_raise(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")):
            outcome, replaced = manager._replace_sl_order(
                manager.positions["BTCUSDT"], 101.0, "test reason"
            )  # must not raise

        self.assertIsNone(outcome)
        self.assertFalse(replaced)


class TrailStopIfImprovedTests(unittest.TestCase):
    def _manager_with_position(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(_plan(), execution_result)
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        manager.positions["BTCUSDT"]["sl_price"] = 100.02  # flat breakeven
        return manager

    def test_disabled_is_a_noop(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", False), \
             patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_no_candles_is_a_noop(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], None)

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_no_swing_available_is_a_noop(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(market_structure, "structure_state", return_value={"available": False}), \
             patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_candidate_not_more_favorable_is_a_noop(self):
        manager = self._manager_with_position()  # sl_price=100.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 100.02},
             ), \
             patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_improved_candidate_replaces_the_sl(self):
        manager = self._manager_with_position()  # sl_price=100.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}) as place:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        place.assert_called_once_with("BTCUSDT", "BUY", 101.5)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["sl_price"], 101.5)
        self.assertTrue(position["trailing_stop_locked_profit"])
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)  # never transitions

    def test_replace_failure_does_not_set_the_locked_profit_flag(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", side_effect=RuntimeError("timeout")):
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        self.assertFalse(manager.positions["BTCUSDT"]["trailing_stop_locked_profit"])

    def test_minus_2021_returns_the_market_close_outcome(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(
                 exchange, "place_stop_loss",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertEqual(outcome, "BREAKEVEN_TRIGGER_MARKET_CLOSE")
        market_close.assert_called_once()

    def test_already_closed_ground_truth_returns_none_without_closing(self):
        manager = self._manager_with_position()

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", return_value=None), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager._trail_stop_if_improved(manager.positions["BTCUSDT"], ["candle"])

        self.assertIsNone(outcome)
        cancel_all.assert_not_called()
        self.assertTrue(manager.has_open_position("BTCUSDT"))


class TrailProfitProtectionIfImprovedTests(unittest.TestCase):
    """The continuous trailing companion to the profit-protection arm -
    see risk_manager.compute_profit_protection_trailing_floor and
    position_manager._trail_profit_protection_if_improved. Entry=100,
    tp1=102 (see _plan()) -> tp1 ROI at LEVERAGE=10 is 20%."""

    def _manager_with_position(self, peak_price=101.2, sl_price=100.6):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(_plan(), execution_result)
        position = manager.positions["BTCUSDT"]
        position["stage"] = BREAKEVEN_ACTIVE
        position["profit_protection_applied"] = True
        position["profit_protection_profit_locked"] = True
        position["profit_protection_peak_price"] = peak_price
        position["sl_price"] = sl_price
        return manager

    def test_not_armed_is_a_noop(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["profit_protection_applied"] = False

        with patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_profit_protection_if_improved(
                manager.positions["BTCUSDT"], 105.0
            )

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_missing_current_price_is_a_noop(self):
        manager = self._manager_with_position()

        with patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_profit_protection_if_improved(
                manager.positions["BTCUSDT"], None
            )

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_new_peak_ratchets_the_stop_up(self):
        # LOCK_PCT_OF_TP1=10 -> floor=100.2 (fixed); RETRACE_PCT=50 of the
        # entry->102.4 gain (2.4) retained = 1.2 -> retrace floor=101.2.
        # max(100.2, 101.2)=101.2, which beats the current sl_price=100.6.
        manager = self._manager_with_position(peak_price=101.2, sl_price=100.6)

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}) as place:
            outcome = manager._trail_profit_protection_if_improved(
                manager.positions["BTCUSDT"], 102.4
            )

        self.assertIsNone(outcome)
        place.assert_called_once_with("BTCUSDT", "BUY", 101.2)
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["sl_price"], 101.2)
        self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)

    def test_pullback_below_peak_keeps_the_remembered_peak_and_does_not_loosen(self):
        manager = self._manager_with_position(peak_price=102.4, sl_price=101.2)

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "place_stop_loss") as place:
            outcome = manager._trail_profit_protection_if_improved(
                manager.positions["BTCUSDT"], 101.8
            )

        self.assertIsNone(outcome)
        place.assert_not_called()
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
        self.assertAlmostEqual(position["sl_price"], 101.2)

    def test_replace_failure_leaves_sl_price_untouched(self):
        manager = self._manager_with_position(peak_price=101.2, sl_price=100.6)

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "_fetch_open_position_detail", side_effect=RuntimeError("timeout")):
            outcome = manager._trail_profit_protection_if_improved(
                manager.positions["BTCUSDT"], 102.4
            )

        self.assertIsNone(outcome)
        self.assertAlmostEqual(manager.positions["BTCUSDT"]["sl_price"], 100.6)


class BreakevenStopOutcomeTests(unittest.TestCase):
    def test_trailing_takes_precedence_over_early_lock(self):
        position = {"trailing_stop_locked_profit": True, "early_breakeven_profit_locked": True}
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=False), "TRAILING_STOP_PROFIT_HIT"
        )
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=True), "SHADOW_TRAILING_STOP_PROFIT_HIT"
        )

    def test_early_lock_when_no_trailing(self):
        position = {"trailing_stop_locked_profit": False, "early_breakeven_profit_locked": True}
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=False), "EARLY_BREAKEVEN_PROFIT_HIT"
        )
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=True), "SHADOW_EARLY_BREAKEVEN_PROFIT_HIT"
        )

    def test_profit_protection_takes_precedence_over_early_lock(self):
        position = {
            "trailing_stop_locked_profit": False,
            "profit_protection_profit_locked": True,
            "early_breakeven_profit_locked": True,
        }
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=False), "PROFIT_PROTECTION_HIT"
        )
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=True), "SHADOW_PROFIT_PROTECTION_HIT"
        )

    def test_trailing_takes_precedence_over_profit_protection(self):
        position = {"trailing_stop_locked_profit": True, "profit_protection_profit_locked": True}
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=False), "TRAILING_STOP_PROFIT_HIT"
        )

    def test_flat_scratch_when_neither(self):
        position = {"trailing_stop_locked_profit": False, "early_breakeven_profit_locked": False}
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=False), "BREAKEVEN_STOP_HIT"
        )
        self.assertEqual(
            PositionManager._breakeven_stop_outcome(position, shadow=True), "SHADOW_BREAKEVEN_STOP_HIT"
        )

    def test_missing_flags_default_to_flat_scratch(self):
        self.assertEqual(PositionManager._breakeven_stop_outcome({}, shadow=False), "BREAKEVEN_STOP_HIT")


class EarlyBreakevenProfitLockedDerivationTests(unittest.TestCase):
    """Real correctness fix: the flag used to be set from the CONFIG value
    (EARLY_BREAKEVEN_LOCK_R_MULTIPLE > 0), not the actual computed price -
    with structure-awareness on, a real profit lock can happen even when
    the config multiple is 0 (structure alone provides it), which the old
    logic would have silently misclassified as a flat scratch."""

    def test_structure_candidate_above_breakeven_sets_the_flag_even_with_zero_config_lock(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(_plan(), execution_result)

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0), \
             patch.object(config, "PROFIT_PROTECTION_ENABLED", False), \
             patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.0},
             ), \
             patch.object(exchange, "get_mark_price", return_value=102.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}):
            manager.poll_live("BTCUSDT", candles=["candle"])

        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["early_breakeven_profit_locked"])
        self.assertEqual(position["sl_price"], 101.0)


class PollShadowTests(unittest.TestCase):
    def setUp(self):
        # EARLY_BREAKEVEN_ENABLED now defaults True and is no longer
        # confluence-gated - without this, a candle favorable enough to
        # hit TP1 is *also* favorable enough to satisfy the 1R early-
        # breakeven trigger, and since that check now runs first, tests
        # meant to exercise a genuine TP1 hit would silently pass through
        # the early-breakeven path instead (same end state, wrong code
        # path actually verified). Off by default here; the tests that
        # actually exercise early breakeven turn it back on locally.
        self.early_breakeven_patcher = patch.object(config, "EARLY_BREAKEVEN_ENABLED", False)
        # Same reasoning as early_breakeven_patcher above -
        # PROFIT_PROTECTION_ENABLED now defaults True in the loaded .env,
        # and is checked even before early breakeven - without this, a
        # candle favorable enough to hit TP1 can also satisfy the
        # profit-protection lock price and silently hijack tests meant to
        # exercise the plain TP1/trailing/breakeven paths.
        self.profit_protection_patcher = patch.object(config, "PROFIT_PROTECTION_ENABLED", False)
        self.early_breakeven_patcher.start()
        self.profit_protection_patcher.start()

    def tearDown(self):
        self.early_breakeven_patcher.stop()
        self.profit_protection_patcher.stop()

    def _manager_with_position(self, side="BUY", confluence_ratio=None):
        manager = PositionManager()
        manager.register(dict(_plan(side), confluence_ratio=confluence_ratio), {"shadow": True})
        return manager

    def test_tp1_pending_sl_hit_closes_as_sl(self):
        manager = self._manager_with_position()
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=99, low=97))  # low <= sl(98)

        self.assertEqual(outcome, "SHADOW_SL_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_tp1_pending_tp1_hit_moves_to_breakeven_and_stays_open(self):
        # Real profit lock now, not flat breakeven - entry=100, sl=98 ->
        # risk_distance=2, EARLY_BREAKEVEN_LOCK_R_MULTIPLE=0.3 (default)
        # -> 100 + 0.3*2 = 100.6. See test_tp1_finished_promotes_to_
        # breakeven (the live-mode equivalent) for the same math.
        manager = self._manager_with_position()
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # high >= tp1(102)

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertEqual(position["sl_price"], 100.6)
        self.assertTrue(position["early_breakeven_profit_locked"])

    def test_ambiguous_candle_hitting_both_is_conservatively_sl(self):
        manager = self._manager_with_position()
        # low(97) <= sl(98) AND high(103) >= tp1(102) in the same candle
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=103, low=97))

        self.assertEqual(outcome, "SHADOW_SL_HIT")

    def test_breakeven_stage_tp2_hit_closes_as_tp2(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to breakeven
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=105, low=101))  # tp2(104) hit

        self.assertEqual(outcome, "SHADOW_TP2_HIT")

    def test_breakeven_stage_stop_hit_closes_as_breakeven_stop(self):
        # Promotion now locks real profit at 100.6 (see
        # test_tp1_pending_tp1_hit_moves_to_breakeven_and_stays_open), so a
        # stop-out here is a genuine small win, not a flat scratch -
        # EARLY_BREAKEVEN_PROFIT_HIT, not the generic BREAKEVEN_STOP_HIT.
        # See test_breakeven_stage_stop_hit_is_a_flat_scratch_when_lock_
        # multiple_is_zero below for the case that still produces this
        # outcome.
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to breakeven
        outcome = manager.poll_shadow(
            "BTCUSDT", _candle(high=100.7, low=100.5)
        )  # low <= locked sl(100.6)

        self.assertEqual(outcome, "SHADOW_EARLY_BREAKEVEN_PROFIT_HIT")

    def test_breakeven_stage_stop_hit_is_a_flat_scratch_when_lock_multiple_is_zero(self):
        # EARLY_BREAKEVEN_LOCK_R_MULTIPLE=0 preserves the original flat-
        # breakeven (fee-buffer-only) behavior exactly - compute_early_
        # breakeven_price falls through to compute_breakeven_price.
        # BREAKEVEN_BUFFER_PCT pinned to _plan()'s own hardcoded fixture
        # assumption (breakeven_price=100.02) - see the analogous note in
        # PollLiveTests.test_early_breakeven_triggers_before_the_normal_
        # tp1_check for why.
        with patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            manager = self._manager_with_position()
            manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to breakeven
            outcome = manager.poll_shadow(
                "BTCUSDT", _candle(high=100.5, low=100.0)
            )  # low <= breakeven(100.02)

        self.assertEqual(outcome, "SHADOW_BREAKEVEN_STOP_HIT")

    def test_no_candle_returns_none(self):
        manager = self._manager_with_position()
        self.assertIsNone(manager.poll_shadow("BTCUSDT", None))

    def test_unknown_symbol_returns_none(self):
        manager = PositionManager()
        self.assertIsNone(manager.poll_shadow("NOPE", _candle(100, 99)))

    def test_sell_side_uses_inverted_high_low_logic(self):
        manager = self._manager_with_position(side="SELL")
        # SELL: sl=102 (above entry). A candle that stays below the stop
        # must not close the position.
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=101, low=99))
        self.assertIsNone(outcome)

        # Now push the high through the SELL stop (102) -> should close.
        outcome = manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))
        self.assertEqual(outcome, "SHADOW_SL_HIT")

    def test_early_breakeven_triggers_before_tp1_reached(self):
        # entry=100, sl=98 (risk=2), R multiple 0.5 -> trigger at close=101,
        # well below tp1(102) - isolates the early trigger from a genuine
        # TP1 hit. Lock multiple forced to 0 here so this test stays
        # decoupled from the profit-lock pricing - see
        # EarlyBreakevenProfitLockTests for that.
        # BREAKEVEN_BUFFER_PCT pinned to _plan()'s own hardcoded fixture
        # assumption (breakeven_price=100.02) - see the analogous note in
        # PollLiveTests.test_early_breakeven_triggers_before_the_normal_
        # tp1_check for why.
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            manager = self._manager_with_position()
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=101.5, low=99, close=101))

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["early_breakeven_applied"])
        self.assertFalse(position["early_breakeven_profit_locked"])
        self.assertEqual(position["sl_price"], position["breakeven_price"])

    def test_price_not_moved_enough_stays_pending(self):
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5):
            manager = self._manager_with_position()
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=100.8, low=99, close=100.5))

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertFalse(position["early_breakeven_applied"])

    def test_triggers_regardless_of_confluence_ratio(self):
        # No longer gated on confluence - real evidence (2026-08-10)
        # showed confluence didn't predict outcome, while MFE distribution
        # did: 28% of losses ran 1.0R+ before fully reversing, completely
        # unprotected. A high-confluence trade gets the same protection.
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5):
            manager = self._manager_with_position(confluence_ratio=0.75)
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=101.5, low=99, close=101))

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["early_breakeven_applied"])

    def test_early_breakeven_locks_real_profit_when_configured(self):
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3):
            manager = self._manager_with_position()
            manager.poll_shadow("BTCUSDT", _candle(high=101.5, low=99, close=101))

        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["early_breakeven_profit_locked"])
        self.assertAlmostEqual(position["sl_price"], 100.6)  # entry=100, risk=2, lock 0.3R

    def test_stop_hit_after_a_locked_early_breakeven_is_a_real_win_not_a_scratch(self):
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.5), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3):
            manager = self._manager_with_position()
            manager.poll_shadow("BTCUSDT", _candle(high=101.5, low=99, close=101))  # locks sl=100.6
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=100.7, low=100.5))  # low <= 100.6

        self.assertEqual(outcome, "SHADOW_EARLY_BREAKEVEN_PROFIT_HIT")

    def test_mae_mfe_track_the_full_candle_range_not_just_the_close(self):
        # BUY, entry=100, sl=98, tp1=102 - low stays above sl and high
        # stays below tp1, so the position survives this poll untouched
        # and both extremes are purely from the range tracking.
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=101, low=98.5, close=99.5))

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["mae_price"], 98.5)
        self.assertEqual(position["mfe_price"], 101)

    def test_profit_protection_arms_then_keeps_trailing_the_peak(self):
        # entry=100, tp1=102 -> tp1 ROI=20% at LEVERAGE=10. Arm trigger
        # (60% of that)=101.2. LOCK_PCT_OF_TP1=10% -> floor=100.2 fixed;
        # RETRACE_PCT=50% of the entry->peak gain retained.
        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            manager = self._manager_with_position()
            manager.poll_shadow("BTCUSDT", _candle(high=101.2, low=100.5, close=101.2))

            position = manager.positions["BTCUSDT"]
            self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
            self.assertTrue(position["profit_protection_applied"])
            self.assertAlmostEqual(position["profit_protection_peak_price"], 101.2)
            self.assertAlmostEqual(position["sl_price"], 100.6)  # retrace: (1.2)*0.5

            # A new higher peak (102.4) should pull the floor up further,
            # ratchet-only - the low here (100.7) stays above the old
            # sl_price (100.6) so this candle doesn't close the trade.
            manager.poll_shadow("BTCUSDT", _candle(high=102.4, low=100.7))

            self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
            self.assertAlmostEqual(position["sl_price"], 101.2)  # retrace: (2.4)*0.5

            # A pullback that doesn't make a new peak must not loosen the
            # stop back down.
            manager.poll_shadow("BTCUSDT", _candle(high=101.5, low=101.3))

            self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
            self.assertAlmostEqual(position["sl_price"], 101.2)

    def test_mae_mfe_disabled_leaves_tracking_at_entry(self):
        with patch.object(config, "MAE_TRACKING_ENABLED", False):
            manager = self._manager_with_position()
            manager.poll_shadow("BTCUSDT", _candle(high=101, low=98.5, close=99.5))

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["mae_price"], 100)
        self.assertEqual(position["mfe_price"], 100)

    def test_journal_receives_mae_mfe_r_multiples_on_close(self):
        manager = self._manager_with_position()  # BUY, entry=100, sl=98, risk=2

        with patch("position_manager.signal_journal.append_outcome") as append_outcome:
            # Moved favorably to 101 within the same candle before also
            # touching down to 97 (low(97) <= sl(98)) -> closes as
            # SHADOW_SL_HIT, but with real favorable excursion recorded
            # along the way - exactly the case MFE exists to distinguish
            # from a trade that went straight down.
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=101, low=97))

        self.assertEqual(outcome, "SHADOW_SL_HIT")
        append_outcome.assert_called_once()
        _, kwargs = append_outcome.call_args
        self.assertAlmostEqual(kwargs["mae_r_multiple"], 1.5)  # (100-97)/2
        self.assertAlmostEqual(kwargs["mfe_r_multiple"], 0.5)  # (101-100)/2

    def test_trailing_applies_a_tighter_sl_mid_breakeven_active(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to flat breakeven

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ):
            outcome = manager.poll_shadow(
                "BTCUSDT", _candle(high=102, low=101.6), candles=["candle"]
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["sl_price"], 101.5)
        self.assertTrue(position["trailing_stop_locked_profit"])

    def test_trailing_does_not_apply_when_disabled_even_with_candles_present(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to flat breakeven
        original_sl = manager.positions["BTCUSDT"]["sl_price"]

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", False), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ):
            manager.poll_shadow("BTCUSDT", _candle(high=102, low=101.6), candles=["candle"])

        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], original_sl)

    def test_trailing_does_not_loosen_the_stop(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to flat breakeven
        original_sl = manager.positions["BTCUSDT"]["sl_price"]

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 90.0},  # worse than breakeven
             ):
            manager.poll_shadow("BTCUSDT", _candle(high=102, low=101), candles=["candle"])

        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], original_sl)

    def test_tp2_hit_short_circuits_before_trailing_is_attempted(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to flat breakeven

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(market_structure, "structure_state") as structure_mock:
            outcome = manager.poll_shadow(
                "BTCUSDT", _candle(high=105, low=101), candles=["candle"]
            )  # tp2(104) hit

        self.assertEqual(outcome, "SHADOW_TP2_HIT")
        structure_mock.assert_not_called()

    def test_trailed_profit_reported_as_trailing_stop_profit_hit(self):
        manager = self._manager_with_position()
        manager.poll_shadow("BTCUSDT", _candle(high=103, low=99))  # promote to flat breakeven

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ):
            manager.poll_shadow("BTCUSDT", _candle(high=102, low=101.6), candles=["candle"])

        outcome = manager.poll_shadow("BTCUSDT", _candle(high=101.6, low=101.4))  # low <= 101.5
        self.assertEqual(outcome, "SHADOW_TRAILING_STOP_PROFIT_HIT")


class ProfitProtectionTp2LegPollShadowTests(PollShadowTests):
    """config.PROFIT_PROTECTION_TP2_LEG_ENABLED - shadow counterpart to
    ProfitProtectionTp2LegPollLiveTests. _plan() BUY: entry=100, tp2=104
    -> same 102.4 arm / 101.2 floor arithmetic (uses the candle's CLOSE
    as the arm price, same convention _try_early_promotions_shadow
    already uses for its own arm check)."""

    def _breakeven_manager(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        return manager

    def test_arms_on_the_tp2_leg_and_locks_the_trailing_floor(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=102.5, low=102.0, close=102.4))

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["profit_protection_applied"])
        self.assertTrue(position["profit_protection_profit_locked"])
        self.assertEqual(position["profit_protection_target"], "tp2_price")
        self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
        self.assertAlmostEqual(position["sl_price"], 101.2)

    def test_below_the_lock_price_does_not_arm(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60):
            manager.poll_shadow("BTCUSDT", _candle(high=102.1, low=101.9, close=102.0))

        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["profit_protection_applied"])
        self.assertEqual(position["sl_price"], 98)  # untouched

    def test_no_op_when_the_tp2_leg_flag_is_off(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", False), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60):
            manager.poll_shadow("BTCUSDT", _candle(high=102.5, low=102.0, close=102.4))

        self.assertFalse(manager.positions["BTCUSDT"]["profit_protection_applied"])

    def test_continues_trailing_after_arming_on_a_later_poll(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            manager.poll_shadow("BTCUSDT", _candle(high=102.5, low=102.0, close=102.4))  # arms, SL -> 101.2

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            outcome = manager.poll_shadow("BTCUSDT", _candle(high=103.0, low=102.6, close=103.0))

        self.assertIsNone(outcome)
        # New peak=103.0 (high, BUY side), retrace 50% of 3.0 = 1.5 -> 101.5.
        self.assertAlmostEqual(manager.positions["BTCUSDT"]["sl_price"], 101.5)


class PollLiveTests(unittest.TestCase):
    def setUp(self):
        # EARLY_BREAKEVEN_ENABLED/MAE_TRACKING_ENABLED now default True -
        # without this, a test that doesn't mock exchange.get_mark_price
        # makes a real, unmocked network call every poll_live(), which is
        # slow, non-deterministic, and (since this fixture's entry/sl sit
        # around 100) a real BTCUSDT mark price would trivially clear the
        # early-breakeven trigger and hijack tests meant to exercise a
        # completely different code path (SL_HIT, TP2_HIT_DIRECT, etc.).
        # Off by default here; tests that actually exercise either
        # feature turn the relevant one back on locally.
        self.early_breakeven_patcher = patch.object(config, "EARLY_BREAKEVEN_ENABLED", False)
        self.mae_tracking_patcher = patch.object(config, "MAE_TRACKING_ENABLED", False)
        # Same reasoning as early_breakeven_patcher above - off by default
        # so tests not about this feature aren't hijacked by an unmocked
        # exchange.get_mark_price(); ProfitProtectionPollLiveTests turns
        # it back on locally.
        self.profit_protection_patcher = patch.object(config, "PROFIT_PROTECTION_ENABLED", False)
        self.early_breakeven_patcher.start()
        self.mae_tracking_patcher.start()
        self.profit_protection_patcher.start()

    def tearDown(self):
        self.early_breakeven_patcher.stop()
        self.mae_tracking_patcher.stop()
        self.profit_protection_patcher.stop()

    def _manager_with_position(self, confluence_ratio=None):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "sl_order": {"algoId": "sl1"},
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(dict(_plan(), confluence_ratio=confluence_ratio), execution_result)
        return manager

    def test_tp1_finished_promotes_to_breakeven(self):
        # A genuine TP1 fill now locks real profit (EARLY_BREAKEVEN_LOCK_
        # R_MULTIPLE, default 0.3), not just a flat fee-buffer scratch -
        # price has, by definition, already moved at least TP1_R_MULTIPLE
        # R in the position's favor by this point. entry=100, sl=98 ->
        # risk_distance=2 -> lock = 100 + 0.3*2 = 100.6.
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertEqual(position["sl_order_id"], "sl2")
        self.assertTrue(position["early_breakeven_profit_locked"])
        cancel.assert_called_once_with("BTCUSDT", "sl1")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 100.6)

    def test_breakeven_promotion_cancels_the_real_sl_not_a_stale_local_id(self):
        # Real bug seen live: local tracking's sl_order_id can be stale
        # (e.g. from a reconciliation mismatch) while a real SL is still
        # open under a different id - cancelling the stale id cancels
        # nothing, and the new placement then fails with -4130 forever.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["sl_order_id"] = "stale_local_id"

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        real_sl_order = {"type": "STOP_MARKET", "closePosition": True, "algoId": "real_sl_on_exchange"}

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[real_sl_order]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}):
            manager.poll_live("BTCUSDT")

        cancel.assert_called_once_with("BTCUSDT", "real_sl_on_exchange")

    def test_tp1_finished_but_position_already_closed_gives_up_cleanly(self):
        # TP1 filling can coincide with the original SL also having fired
        # (or manual intervention) - the position is genuinely gone, so
        # this must close out tracking instead of retrying forever.
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value=None), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all, \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "TP1_THEN_POSITION_ALREADY_CLOSED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        new_sl.assert_not_called()
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_tp1_finished_but_position_check_fails_retries_next_poll(self):
        # A transient network/backoff error while checking ground truth
        # must NOT be treated as "position closed" - that would abandon a
        # still-open, still-unprotected position.
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", side_effect=RuntimeError("timeout")), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], TP1_PENDING)
        cancel.assert_not_called()
        new_sl.assert_not_called()

    def test_breakeven_placement_immediately_triggers_closes_at_market(self):
        # Binance rejects a stop that would fire the instant it's placed -
        # that means price already passed the (now real-profit-locked, see
        # test_tp1_finished_promotes_to_breakeven) level, so the remainder
        # must be closed at market instead of left unprotected. Still a
        # real locked-profit exit, not a scratch - hence EARLY_BREAKEVEN_
        # PROFIT_HIT, not the generic BREAKEVEN_TRIGGER_MARKET_CLOSE.
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(
                 exchange,
                 "_fetch_open_position_detail",
                 return_value={"quantity": 0.5},
             ), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all, \
             patch.object(
                 exchange,
                 "place_stop_loss",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "EARLY_BREAKEVEN_PROFIT_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        market_close.assert_called_once_with("BTCUSDT", "BUY", 0.5)
        # The original SL (via the targeted cancel before the failed
        # replace attempt) and everything else on the symbol (via the
        # comprehensive cancel-all after the market close) get cleaned up.
        cancel.assert_called_once_with("BTCUSDT", "sl1")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_sl_finished_closes_and_cancels_all_open_orders(self):
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "sl1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "SL_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_tp2_finished_directly_closes_as_tp2_hit_direct(self):
        manager = self._manager_with_position()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp2_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "TP2_HIT_DIRECT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_breakeven_stage_sl_finished_closes(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "sl1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "BREAKEVEN_STOP_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_breakeven_stage_tp2_finished_closes(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp2_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "TP2_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_shadow_position_is_ignored_by_poll_live(self):
        manager = PositionManager()
        manager.register(_plan(), {"shadow": True})

        outcome = manager.poll_live("BTCUSDT")
        self.assertIsNone(outcome)

    def test_unknown_symbol_returns_none(self):
        manager = PositionManager()
        self.assertIsNone(manager.poll_live("NOPE"))

    def test_missing_order_id_is_never_sent_to_the_status_lookup(self):
        # A blank algoId is a guaranteed -1102 from Binance on every call -
        # this must short-circuit locally instead of hitting the exchange.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""

        with patch.object(exchange, "get_algo_order_status", return_value="NEW") as status, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": ""}):
            manager.poll_live("BTCUSDT")

        called_ids = {call.args[1] for call in status.call_args_list}
        self.assertNotIn("", called_ids)

    def test_missing_tp1_order_is_recovered(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_partial", return_value={"algoId": "tp1_new"}
             ) as recover:
            manager.poll_live("BTCUSDT")

        recover.assert_called_once()
        self.assertEqual(manager.positions["BTCUSDT"]["tp1_order_id"], "tp1_new")

    def test_missing_tp1_order_already_exists_on_exchange_is_resynced_not_duplicated(self):
        # The real bug seen live: local tracking lost the id while the
        # real order is still there - placing another gets rejected with
        # -4130 forever. Must adopt the real id instead of duplicating.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""
        real_tp1 = {"type": "TAKE_PROFIT_MARKET", "closePosition": False, "algoId": "real_tp1"}

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[real_tp1]), \
             patch.object(exchange, "place_take_profit_partial") as place:
            manager.poll_live("BTCUSDT")

        place.assert_not_called()
        self.assertEqual(manager.positions["BTCUSDT"]["tp1_order_id"], "real_tp1")

    def test_missing_tp2_order_is_recovered_in_either_stage(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp2_order_id"] = ""
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_full", return_value={"algoId": "tp2_new"}
             ) as recover:
            manager.poll_live("BTCUSDT")

        recover.assert_called_once()
        self.assertEqual(manager.positions["BTCUSDT"]["tp2_order_id"], "tp2_new")

    def test_tp1_market_close_instead_when_price_already_passed_it(self):
        # -2021 on TP1 specifically means price already passed that level -
        # take the partial at market instead of retrying a placement that
        # can never succeed.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange,
                 "place_take_profit_partial",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}):
            manager.poll_live("BTCUSDT")

        market_close.assert_called_once_with("BTCUSDT", "BUY", 0.5)

    def test_tp1_recovery_is_not_attempted_once_in_breakeven_stage(self):
        # TP1 is already resolved by the time BREAKEVEN_ACTIVE is reached -
        # a blank tp1_order_id there is expected (it filled), not missing.
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_take_profit_partial") as recover:
            manager.poll_live("BTCUSDT")

        recover.assert_not_called()

    def test_recovery_failure_is_handled_gracefully(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["tp1_order_id"] = ""

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_partial", side_effect=RuntimeError("rejected")
             ):
            outcome = manager.poll_live("BTCUSDT")  # must not raise

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["tp1_order_id"], "")

    def test_early_breakeven_triggers_before_the_normal_tp1_check(self):
        manager = self._manager_with_position()

        # Lock multiple forced to 0 here so this test stays decoupled from
        # the profit-lock pricing itself - see EarlyBreakevenProfitLockTests.
        # BREAKEVEN_BUFFER_PCT pinned to _plan()'s own hardcoded fixture
        # assumption (breakeven_price=100.02) so the live recompute here
        # matches it exactly - otherwise the live value (now 0.15% by
        # default) would diverge from the fixture's stale flat value and
        # register as a real profit lock instead of the flat scratch this
        # test is actually about.
        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02), \
             patch.object(exchange, "get_mark_price", return_value=102.0), \
             patch.object(exchange, "get_algo_order_status") as status_mock, \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["early_breakeven_applied"])
        self.assertFalse(position["early_breakeven_profit_locked"])
        self.assertEqual(position["sl_order_id"], "sl2")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", position["breakeven_price"])
        cancel.assert_called_once_with("BTCUSDT", "sl1")
        # Never reached the normal TP1/SL/TP2 status checks this cycle.
        status_mock.assert_not_called()

    def test_triggers_regardless_of_confluence_ratio(self):
        # No longer gated on confluence - real evidence (2026-08-10)
        # showed confluence didn't predict outcome, while MFE distribution
        # did: 28% of losses ran 1.0R+ before fully reversing, completely
        # unprotected. A high-confluence trade gets the same protection.
        manager = self._manager_with_position(confluence_ratio=0.75)

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0), \
             patch.object(exchange, "get_mark_price", return_value=102.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["early_breakeven_applied"])

    def test_early_breakeven_places_locked_profit_stop_when_configured(self):
        manager = self._manager_with_position()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3), \
             patch.object(exchange, "get_mark_price", return_value=102.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}) as new_sl:
            manager.poll_live("BTCUSDT")

        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["early_breakeven_profit_locked"])
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 100.6)  # entry=100, risk=2, lock 0.3R
        self.assertEqual(position["sl_price"], 100.6)

    def test_sl_hit_after_a_locked_early_breakeven_reports_profit_hit_outcome(self):
        manager = self._manager_with_position()

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 1.0), \
             patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3), \
             patch.object(exchange, "get_mark_price", return_value=102.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl2"}):
            manager.poll_live("BTCUSDT")  # promotes to a locked-profit stop

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "sl2" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "EARLY_BREAKEVEN_PROFIT_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_mark_price_never_fetched_when_both_mae_tracking_and_early_breakeven_are_off(self):
        manager = self._manager_with_position(confluence_ratio=0.25)

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", False), \
             patch.object(config, "MAE_TRACKING_ENABLED", False), \
             patch.object(exchange, "get_mark_price") as mark_price_mock, \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        mark_price_mock.assert_not_called()

    def test_mae_tracking_still_fetches_mark_price_when_early_breakeven_is_off(self):
        manager = self._manager_with_position(confluence_ratio=0.25)

        with patch.object(config, "EARLY_BREAKEVEN_ENABLED", False), \
             patch.object(config, "MAE_TRACKING_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=101.0) as mark_price_mock, \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        mark_price_mock.assert_called_once_with("BTCUSDT")
        self.assertEqual(manager.positions["BTCUSDT"]["mfe_price"], 101.0)

    def test_trailing_applies_a_tighter_sl_mid_breakeven_active(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        manager.positions["BTCUSDT"]["sl_price"] = 100.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}) as place:
            outcome = manager.poll_live("BTCUSDT", candles=["candle"])

        self.assertIsNone(outcome)
        place.assert_called_once_with("BTCUSDT", "BUY", 101.5)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["sl_price"], 101.5)
        self.assertTrue(position["trailing_stop_locked_profit"])

    def test_trailing_does_not_apply_when_disabled_even_with_candles_present(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        manager.positions["BTCUSDT"]["sl_price"] = 100.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", False), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "place_stop_loss") as place:
            manager.poll_live("BTCUSDT", candles=["candle"])

        place.assert_not_called()
        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], 100.02)

    def test_tp2_finished_short_circuits_before_trailing_is_attempted(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp2_1" else "NEW"

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all, \
             patch.object(market_structure, "structure_state") as structure_mock:
            outcome = manager.poll_live("BTCUSDT", candles=["candle"])

        self.assertEqual(outcome, "TP2_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")
        structure_mock.assert_not_called()

    def test_trailed_profit_reported_as_trailing_stop_profit_hit(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        manager.positions["BTCUSDT"]["sl_price"] = 100.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 101.5},
             ), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}):
            manager.poll_live("BTCUSDT", candles=["candle"])  # trails sl_price -> 101.5

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "sl_trailed" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "TRAILING_STOP_PROFIT_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")


class ProfitProtectionTp2LegPollLiveTests(PollLiveTests):
    """config.PROFIT_PROTECTION_TP2_LEG_ENABLED - a genuine TP1 fill's
    remainder (BREAKEVEN_ACTIVE, profit_protection_applied still False)
    gets its own fresh-arm, reusing the exact same ACTIVATION_PCT_OF_TP1/
    LOCK_PCT_OF_TP1/RETRACE_PCT math as the pre-TP1 case, just against
    tp2_price instead of tp1_price. _plan() BUY: entry=100, tp2=104 ->
    tp2 move=4, tp2 ROI=(4/100)*10*100=40%. Arm trigger: 60% of that=24%
    -> 102.4 (mirrors ProfitProtectionPollLiveTests' own TP1-relative
    arithmetic, just against the wider tp2 distance)."""

    def _breakeven_manager(self):
        manager = self._manager_with_position()
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        return manager

    def test_arms_on_the_tp2_leg_and_locks_the_trailing_floor(self):
        # Lock: worst-case LOCK_PCT_OF_TP1=10% of tp2 ROI -> 100.4;
        # retrace RETRACE_PCT=50% of the entry->peak gain (2.4) retained
        # = 1.2 -> 101.2. Floor is the max of the two (101.2).
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=102.4), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.2}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_locked"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["profit_protection_applied"])
        self.assertTrue(position["profit_protection_profit_locked"])
        self.assertEqual(position["profit_protection_target"], "tp2_price")
        self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
        self.assertAlmostEqual(position["sl_price"], 101.2)
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 101.2)

    def test_below_the_lock_price_does_not_arm(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=102.0):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["profit_protection_applied"])
        self.assertEqual(position["sl_price"], 98)  # untouched

    def test_no_op_when_the_tp2_leg_flag_is_off(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", False), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=102.4):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertFalse(manager.positions["BTCUSDT"]["profit_protection_applied"])

    def test_no_op_when_already_armed(self):
        # An early-promoted position (pre-TP1 profit protection or early
        # breakeven) that's already armed must not re-arm/re-lock here.
        manager = self._breakeven_manager()
        manager.positions["BTCUSDT"]["profit_protection_applied"] = True
        manager.positions["BTCUSDT"]["profit_protection_target"] = "tp2_price"
        manager.positions["BTCUSDT"]["sl_price"] = 100.6

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.2}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}):
            manager.poll_live("BTCUSDT")

        # Continues trailing via the EXISTING mechanism instead (not a
        # fresh arm) - floor at peak=103.0: retrace 50% of 3.0 = 1.5 -> 101.5.
        self.assertAlmostEqual(manager.positions["BTCUSDT"]["sl_price"], 101.5)

    def test_continues_trailing_after_arming_on_a_later_poll(self):
        manager = self._breakeven_manager()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=102.4), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.2}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_locked"}):
            manager.poll_live("BTCUSDT")  # arms, SL -> 101.2

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_TP2_LEG_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.2}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}) as trail_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        # New peak=103.0, retrace 50% of 3.0 = 1.5 -> 101.5 - more
        # favorable than the 101.2 armed floor, so it ratchets forward.
        trail_sl.assert_called_once_with("BTCUSDT", "BUY", 101.5)


class ProfitProtectionPollLiveTests(PollLiveTests):
    """Integration coverage for config.PROFIT_PROTECTION_ENABLED through
    the real poll_live() dispatch, on top of PollLiveTests' setUp (which
    keeps EARLY_BREAKEVEN_ENABLED/MAE_TRACKING_ENABLED/
    PROFIT_PROTECTION_ENABLED off by default so an unmocked
    exchange.get_mark_price() can't hijack unrelated tests)."""

    def test_profit_protection_promotes_before_tp1_and_locks_the_trailing_floor(self):
        # _plan() BUY: entry=100, tp1=102 -> tp1 move=2, tp1 ROI=(2/100)*
        # 10*100=20%. Arm trigger: 60% of that=12% -> trigger price=101.2
        # (see ComputeProfitProtectionLockPriceTests) - mark price reaches
        # it exactly, so peak_price seeds at 101.2 too. Floor: worst-case
        # LOCK_PCT_OF_TP1=10% of TP1 ROI=2% -> 100.2; retrace RETRACE_PCT=
        # 50% of the entry->peak gain (1.2) retained = 0.6 -> 100.6. Floor
        # is the max of the two (100.6) - retrace dominates here.
        manager = self._manager_with_position()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_mark_price", return_value=101.2), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_locked"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["profit_protection_applied"])
        self.assertTrue(position["profit_protection_profit_locked"])
        self.assertAlmostEqual(position["profit_protection_peak_price"], 101.2)
        self.assertAlmostEqual(position["sl_price"], 100.6)
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 100.6)

    def test_below_the_lock_price_does_not_promote(self):
        manager = self._manager_with_position()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertFalse(position["profit_protection_applied"])

    def test_profit_protection_is_checked_before_early_breakeven_when_both_would_fire(self):
        manager = self._manager_with_position()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "EARLY_BREAKEVEN_ENABLED", True), \
             patch.object(config, "EARLY_BREAKEVEN_R_MULTIPLE", 0.1), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", False), \
             patch.object(exchange, "get_mark_price", return_value=101.2), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_locked"}):
            manager.poll_live("BTCUSDT")

        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["profit_protection_profit_locked"])
        self.assertFalse(position["early_breakeven_applied"])

    def test_continues_trailing_after_arming_on_a_later_poll(self):
        # Same arm as test_profit_protection_promotes_before_tp1_and_locks_
        # the_trailing_floor (SL -> 100.6), then a later poll with a higher
        # mark price (102.4) should ratchet the stop further via the
        # BREAKEVEN_ACTIVE branch of poll_live, not just the one-time arm.
        manager = self._manager_with_position()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_mark_price", return_value=101.2), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_locked"}):
            manager.poll_live("BTCUSDT")  # arms, SL -> 100.6

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(exchange, "get_mark_price", return_value=102.4), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 1.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_trailed"}) as trail_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        trail_sl.assert_called_once_with("BTCUSDT", "BUY", 101.2)
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["profit_protection_peak_price"], 102.4)
        self.assertAlmostEqual(position["sl_price"], 101.2)


def _pending_order_status(status, executed_qty=0.0, avg_price=0.0, orig_qty=1.0):
    return {"status": status, "executed_qty": executed_qty, "avg_price": avg_price, "orig_qty": orig_qty}


def _pending_manager(side="BUY"):
    manager = PositionManager()
    execution_result = {"shadow": False, "entry_order": {"orderId": "limit1"}}
    manager.register_pending_entry(_plan(side), execution_result, trade_id="BTCUSDT_1")
    return manager


class RegisterPendingEntryTests(unittest.TestCase):
    """config.LIMIT_ENTRY_MODE_ENABLED - a resting limit entry has no
    protective orders yet (unlike register(), which reads real SL/TP ids
    off a synchronously-filled market order) - see
    position_manager.poll_pending_entry for where those get placed."""

    def test_live_registration_has_no_protective_orders_yet(self):
        manager = _pending_manager()
        position = manager.positions["BTCUSDT"]

        self.assertEqual(position["stage"], PENDING_LIMIT_FILL)
        self.assertIsNone(position["sl_order_id"])
        self.assertIsNone(position["tp1_order_id"])
        self.assertIsNone(position["tp2_order_id"])
        self.assertEqual(position["filled_quantity"], 0.0)
        self.assertEqual(position["limit_order_id"], "limit1")
        self.assertIsNone(position["mae_price"])
        self.assertIsNone(position["mfe_price"])

    def test_reserves_a_max_total_positions_slot_immediately(self):
        # No separate pending-entry accounting - same dict as register(),
        # so has_open_position/open_count already work correctly for a
        # resting entry with no new code, which is the whole basis for
        # not building a separate PendingEntryManager class.
        manager = _pending_manager()

        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(manager.open_count(), 1)

    def test_shadow_registration_has_no_limit_order_id(self):
        manager = PositionManager()
        position = manager.register_pending_entry(_plan(), {"shadow": True}, trade_id="BTCUSDT_1")
        self.assertIsNone(position["limit_order_id"])
        self.assertTrue(position["shadow"])


class PollPendingEntryFillTests(unittest.TestCase):
    def test_unknown_status_is_a_noop_retry_next_poll(self):
        manager = _pending_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("UNKNOWN")):
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], PENDING_LIMIT_FILL)

    def test_immediate_full_fill_places_sl_tp1_tp2_and_transitions(self):
        manager = _pending_manager()

        with patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("FILLED", executed_qty=1.0, avg_price=100.0)), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl1"}) as place_sl, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}) as place_tp2, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as place_tp1:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")
        self.assertEqual(position["filled_quantity"], 1.0)
        self.assertEqual(position["entry_price"], 100.0)  # real avg_price, not the planned entry
        self.assertEqual(position["risk_distance"], abs(100.0 - position["sl_price"]))
        place_sl.assert_called_once()
        place_tp2.assert_called_once()
        place_tp1.assert_called_once()

    def test_partial_fill_places_sl_and_tp2_but_defers_tp1(self):
        manager = _pending_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=100.0)), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch.object(exchange, "place_take_profit_partial") as place_tp1:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], PENDING_LIMIT_FILL)
        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")
        self.assertEqual(position["filled_quantity"], 0.4)
        place_tp1.assert_not_called()

    def test_second_poll_growing_to_full_fill_places_the_deferred_tp1_only(self):
        manager = _pending_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=100.0)), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch.object(exchange, "place_take_profit_partial") as place_tp1_first:
            manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        place_tp1_first.assert_not_called()

        with patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("FILLED", executed_qty=1.0, avg_price=100.0)), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as place_tp1_second, \
             patch.object(exchange, "place_stop_loss") as place_sl_again, \
             patch.object(exchange, "place_take_profit_full") as place_tp2_again:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertAlmostEqual(position["tp1_quantity"], 0.5)  # sized off the FINAL filled qty (1.0), not the first partial
        place_tp1_second.assert_called_once()
        place_sl_again.assert_not_called()  # SL/TP2 only placed on the FIRST fill
        place_tp2_again.assert_not_called()

    def test_sl_placement_failure_closes_the_filled_quantity_and_journals(self):
        manager = _pending_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("FILLED", executed_qty=1.0, avg_price=100.0)), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all, \
             patch("position_manager.signal_journal.append_outcome") as append_outcome:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertEqual(outcome, "LIMIT_FILL_SL_PLACEMENT_FAILED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        cancel_all.assert_called_once_with("BTCUSDT")
        append_outcome.assert_called_once()


class PollPendingEntryExpiryTests(unittest.TestCase):
    def test_zero_fill_expiry_cancels_and_drops_without_cooldown(self):
        manager = _pending_manager()
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW")), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch("position_manager.signal_journal.append_outcome") as append_outcome:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertEqual(outcome, "LIMIT_EXPIRED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        self.assertFalse(manager.is_in_cooldown("BTCUSDT"))
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")
        append_outcome.assert_called_once_with("BTCUSDT", "LIMIT_EXPIRED_UNFILLED", "BTCUSDT_1")

    def test_zero_fill_invalidation_cancels_and_sets_cooldown(self):
        manager = _pending_manager()  # BUY, sl_price=98

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch.object(config, "SYMBOL_REENTRY_COOLDOWN_SECONDS", 900), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW")), \
             patch.object(exchange, "cancel_order"), \
             patch("position_manager.signal_journal.append_outcome"):
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=_candle(high=99, low=97))  # low <= sl(98)

        self.assertEqual(outcome, "LIMIT_INVALIDATED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        self.assertTrue(manager.is_in_cooldown("BTCUSDT"))

    def test_not_yet_expired_and_not_invalidated_is_a_noop(self):
        manager = _pending_manager()

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW")):
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=_candle(high=101, low=99))

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))


class PollPendingEntryRaceTests(unittest.TestCase):
    def test_cancel_races_a_fill_and_still_protects_it(self):
        # A fill can land microseconds before the cancel takes effect -
        # Binance doesn't guarantee atomicity between them. The
        # post-cancel re-check must catch this instead of treating it as
        # a clean, unfilled cancel.
        manager = _pending_manager()
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000  # force the expiry path

        order_statuses = iter([
            _pending_order_status("NEW", executed_qty=0.0),  # the initial check
            _pending_order_status("FILLED", executed_qty=1.0, avg_price=100.0),  # post-cancel re-check
        ])

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch.object(exchange, "get_order_status", side_effect=lambda *a, **k: next(order_statuses)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}):
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["filled_quantity"], 1.0)
        self.assertEqual(position["sl_order_id"], "sl1")
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")


class PollPendingEntryPartialThenExpireTests(unittest.TestCase):
    def test_partial_fill_then_expiry_settles_the_remainder_without_touching_existing_sl_tp2(self):
        manager = _pending_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=100.0)), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch.object(exchange, "place_take_profit_partial") as place_tp1_first:
            manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        place_tp1_first.assert_not_called()
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000  # force expiry

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=100.0)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_stop_loss") as place_sl_again, \
             patch.object(exchange, "place_take_profit_full") as place_tp2_again, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as place_tp1_second:
            outcome = manager.poll_pending_entry("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)  # not closed - settled into TP1_PENDING with real filled qty
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["sl_order_id"], "sl1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        place_sl_again.assert_not_called()
        place_tp2_again.assert_not_called()
        place_tp1_second.assert_called_once()
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")


class PollShadowPendingEntryTests(unittest.TestCase):
    def _shadow_manager(self, side="BUY"):
        manager = PositionManager()
        manager.register_pending_entry(_plan(side), {"shadow": True}, trade_id="BTCUSDT_1")
        return manager

    def test_no_candle_returns_none(self):
        manager = self._shadow_manager()
        self.assertIsNone(manager.poll_shadow_pending_entry("BTCUSDT", None))

    def test_candle_touching_entry_fills_and_transitions(self):
        manager = self._shadow_manager()  # entry=100
        outcome = manager.poll_shadow_pending_entry("BTCUSDT", _candle(high=101, low=99))  # range covers 100

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], TP1_PENDING)
        self.assertEqual(position["filled_quantity"], position["quantity"])
        self.assertEqual(position["mae_price"], 100)
        self.assertEqual(position["mfe_price"], 100)

    def test_candle_touching_both_entry_and_sl_assumes_sl_first_no_fill(self):
        manager = self._shadow_manager()  # entry=100, sl=98

        with patch("position_manager.signal_journal.append_outcome"):
            outcome = manager.poll_shadow_pending_entry("BTCUSDT", _candle(high=101, low=97))  # covers both

        self.assertEqual(outcome, "LIMIT_INVALIDATED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_candle_reaching_sl_without_ever_touching_entry_is_invalidated(self):
        manager = self._shadow_manager()  # entry=100, sl=98

        with patch("position_manager.signal_journal.append_outcome"):
            outcome = manager.poll_shadow_pending_entry("BTCUSDT", _candle(high=99.5, low=97.5))  # reaches sl, not entry

        self.assertEqual(outcome, "LIMIT_INVALIDATED_UNFILLED")

    def test_wall_clock_expiry_with_no_candle_touch(self):
        manager = self._shadow_manager()
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "LIMIT_ENTRY_EXPIRY_SECONDS", 600), \
             patch("position_manager.signal_journal.append_outcome"):
            outcome = manager.poll_shadow_pending_entry("BTCUSDT", _candle(high=99.9, low=99.5))  # neither entry(100) nor sl(98)

        self.assertEqual(outcome, "LIMIT_EXPIRED_UNFILLED")


class ReconcilePendingEntriesOnStartupTests(unittest.TestCase):
    def test_disabled_config_does_nothing(self):
        manager = PositionManager()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", False), \
             patch.object(config, "RETRACEMENT_ENTRY_ENABLED", False), \
             patch.object(config, "DCA_RESTING_ORDER_ENABLED", False), \
             patch.object(exchange, "get_all_open_orders") as get_orders:
            manager.reconcile_pending_entries_on_startup()

        get_orders.assert_not_called()

    def test_retracement_entry_enabled_alone_still_sweeps(self):
        # config.RETRACEMENT_ENTRY_ENABLED places the exact same plain
        # LIMIT order type LIMIT_ENTRY_MODE_ENABLED does - this sweep must
        # run for either flag, not just the older one.
        manager = PositionManager()
        open_orders = [{"symbol": "ETHUSDT", "orderId": "limit9", "type": "LIMIT"}]

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", False), \
             patch.object(config, "RETRACEMENT_ENTRY_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("ETHUSDT", "limit9")

    def test_stray_resting_limit_order_is_cancelled(self):
        manager = PositionManager()
        open_orders = [{"symbol": "ETHUSDT", "orderId": "limit9", "type": "LIMIT"}]

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("ETHUSDT", "limit9")

    def test_algo_orders_in_the_same_response_are_left_alone(self):
        # Proves this doesn't collide with reconcile_on_startup's own
        # SL/TP algo-order handling - only plain LIMIT orders are touched.
        manager = PositionManager()
        open_orders = [{"symbol": "ETHUSDT", "orderId": "algo1", "type": "STOP_MARKET"}]

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_not_called()

    # config.DCA_RESTING_ORDER_ENABLED - a resting DCA-add order belongs
    # to an already-open, real exchange position, unlike a pending ENTRY
    # (nothing recoverable). The blanket "cancel every resting LIMIT
    # order" behavior above would silently strip that protection on
    # every restart without this exception.

    def test_dca_flag_alone_still_sweeps(self):
        # Neither LIMIT_ENTRY_MODE_ENABLED nor RETRACEMENT_ENTRY_ENABLED -
        # DCA_RESTING_ORDER_ENABLED alone must still trigger the sweep
        # (an untagged stray still gets cancelled either way).
        manager = PositionManager()
        open_orders = [{"symbol": "ETHUSDT", "orderId": "limit9", "type": "LIMIT"}]

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", False), \
             patch.object(config, "RETRACEMENT_ENTRY_ENABLED", False), \
             patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("ETHUSDT", "limit9")

    def test_tagged_dca_order_matching_a_tracked_position_is_preserved(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": False})
        open_orders = [{
            "symbol": "BTCUSDT", "orderId": "dca_real_1", "type": "LIMIT",
            "clientOrderId": f"{execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX}1787600000000",
        }]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_not_called()
        self.assertEqual(manager.positions["BTCUSDT"]["dca_order_id"], "dca_real_1")

    def test_tagged_dca_order_matching_an_already_populated_dca_order_id_is_preserved(self):
        # Real bug found live (2026-08-26, MEUSDT/UBUSDT): reconcile_on_
        # startup's own preferred path (_try_restore_from_saved_state)
        # restores the saved dca_order_id verbatim BEFORE this runs - the
        # common case for a real resting DCA order, not a rare one. The
        # old `not position.get("dca_order_id")` condition failed here
        # and cancelled a live resting order every restart.
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": False})
        manager.positions["BTCUSDT"]["dca_order_id"] = "dca_real_1"
        open_orders = [{
            "symbol": "BTCUSDT", "orderId": "dca_real_1", "type": "LIMIT",
            "clientOrderId": f"{execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX}1787600000000",
        }]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_not_called()
        self.assertEqual(manager.positions["BTCUSDT"]["dca_order_id"], "dca_real_1")

    def test_tagged_dca_order_mismatching_an_already_tracked_dca_order_id_is_cancelled(self):
        # A genuine mismatch (a DIFFERENT order_id than what's already
        # tracked) stays conservative - still cancelled, not silently
        # overwritten.
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": False})
        manager.positions["BTCUSDT"]["dca_order_id"] = "dca_old_stale"
        open_orders = [{
            "symbol": "BTCUSDT", "orderId": "dca_different_1", "type": "LIMIT",
            "clientOrderId": f"{execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX}1787600000000",
        }]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("BTCUSDT", "dca_different_1")

    def test_tagged_dca_order_with_no_matching_position_is_still_cancelled(self):
        # State lost/corrupt, or the position resolved some other way
        # before this ran - the tag alone isn't a blanket exemption.
        manager = PositionManager()
        open_orders = [{
            "symbol": "BTCUSDT", "orderId": "dca_orphan_1", "type": "LIMIT",
            "clientOrderId": f"{execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX}1787600000000",
        }]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("BTCUSDT", "dca_orphan_1")

    def test_tagged_dca_order_matching_a_non_dca_pending_position_is_cancelled(self):
        # A tagged order matching a symbol that's tracked but NOT
        # DCA_PENDING (e.g. already promoted to BREAKEVEN_ACTIVE) is not
        # a legitimate match either - falls through to cancel.
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": False})
        manager.positions["BTCUSDT"]["stage"] = BREAKEVEN_ACTIVE
        open_orders = [{
            "symbol": "BTCUSDT", "orderId": "dca_stale_1", "type": "LIMIT",
            "clientOrderId": f"{execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX}1787600000000",
        }]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("BTCUSDT", "dca_stale_1")

    def test_untagged_order_for_a_dca_pending_symbol_is_still_cancelled(self):
        # Only the tag identifies a legitimate DCA-add order - an
        # ordinary pending-entry LIMIT order that happens to share a
        # symbol with a DCA_PENDING position gets no special treatment.
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": False})
        open_orders = [{"symbol": "BTCUSDT", "orderId": "limit9", "type": "LIMIT"}]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_all_open_orders", return_value=open_orders), \
             patch.object(exchange, "cancel_order") as cancel_order:
            manager.reconcile_pending_entries_on_startup()

        cancel_order.assert_called_once_with("BTCUSDT", "limit9")
        self.assertFalse(manager.positions["BTCUSDT"]["dca_order_id"])


def _retracement_plan(side="BUY", dca=True, single_tp=False):
    plan = dict(_plan(side))

    if dca:
        plan["dca_price"] = 96 if side == "BUY" else 104

    if single_tp:
        plan.update({
            "tp1_price": None, "tp2_price": None, "tp1_quantity": None, "tp2_quantity": None,
            "tp_price": 106 if side == "BUY" else 94, "single_tp": True,
        })

    return plan


def _retracement_manager(
    side="BUY", dca=True, single_tp=False, retracement_price=99.8, shadow=False,
    retracement_timeout_seconds=None, used_deep_retracement=None,
):
    manager = PositionManager()
    execution_result = {
        "shadow": shadow,
        "entry_order": None if shadow else {"orderId": "limit1"},
        "retracement_price": retracement_price,
    }

    if retracement_timeout_seconds is not None:
        execution_result["retracement_timeout_seconds"] = retracement_timeout_seconds

    if used_deep_retracement is not None:
        execution_result["used_deep_retracement"] = used_deep_retracement

    with patch.object(config, "DCA_ENABLED", dca):
        manager.register_retracement_pending(
            _retracement_plan(side, dca=dca, single_tp=single_tp), execution_result, trade_id="BTCUSDT_1",
        )

    return manager


class RegisterRetracementPendingTests(unittest.TestCase):
    def test_live_registration_has_the_expected_shape(self):
        manager = _retracement_manager()
        position = manager.positions["BTCUSDT"]

        self.assertEqual(position["stage"], RETRACEMENT_PENDING)
        self.assertEqual(position["limit_order_id"], "limit1")
        self.assertEqual(position["retracement_price"], 99.8)
        self.assertFalse(position["shadow"])
        self.assertTrue(position["is_dca"])
        self.assertEqual(position["plan"]["entry_price"], 100)
        self.assertEqual(position["plan"]["sl_price"], 98)

    def test_shadow_registration_has_no_limit_order_id(self):
        manager = _retracement_manager(shadow=True)
        position = manager.positions["BTCUSDT"]

        self.assertIsNone(position["limit_order_id"])
        self.assertTrue(position["shadow"])

    def test_is_dca_captures_config_at_registration_time(self):
        dca_manager = _retracement_manager(dca=True)
        plain_manager = _retracement_manager(dca=False)

        self.assertTrue(dca_manager.positions["BTCUSDT"]["is_dca"])
        self.assertFalse(plain_manager.positions["BTCUSDT"]["is_dca"])

    def test_reserves_a_max_total_positions_slot_immediately(self):
        manager = _retracement_manager()

        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(manager.open_count(), 1)

    def test_stores_timeout_seconds_and_deep_flag_from_execution_result(self):
        # config.RETRACEMENT_DEPTH_AWARE_ENABLED
        manager = _retracement_manager(retracement_timeout_seconds=600, used_deep_retracement=True)
        position = manager.positions["BTCUSDT"]

        self.assertEqual(position["retracement_timeout_seconds"], 600)
        self.assertTrue(position["used_deep_retracement"])

    def test_defaults_timeout_seconds_when_execution_result_omits_it(self):
        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300):
            manager = _retracement_manager()

        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["retracement_timeout_seconds"], 300)
        self.assertFalse(position["used_deep_retracement"])


class ResolveRetracementMarketFallbackTests(unittest.TestCase):
    def _manager_with_position(self, filled_quantity=0.0):
        manager = _retracement_manager()
        position = manager.positions["BTCUSDT"]
        position["plan"]["quantity"] = 1.0
        return manager, position

    def test_already_fully_filled_places_no_market_order(self):
        manager, position = self._manager_with_position()

        with patch.object(exchange, "place_market_order") as market_order:
            total_quantity, entry_price, used_fallback, error = manager._resolve_retracement_market_fallback(
                position, 1.0, 99.8
            )

        market_order.assert_not_called()
        self.assertIsNone(error)
        self.assertEqual(total_quantity, 1.0)
        self.assertEqual(entry_price, 99.8)
        self.assertFalse(used_fallback)

    def test_zero_fill_places_a_full_market_fallback(self):
        manager, position = self._manager_with_position()

        with patch.object(exchange, "place_market_order", return_value={"orderId": 2}) as market_order, \
             patch.object(exchange, "resolve_market_fill_price", return_value=100.5):
            total_quantity, entry_price, used_fallback, error = manager._resolve_retracement_market_fallback(
                position, 0.0, None
            )

        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        self.assertIsNone(error)
        self.assertEqual(total_quantity, 1.0)
        self.assertEqual(entry_price, 100.5)
        self.assertTrue(used_fallback)

    def test_partial_fill_blends_with_the_market_fallback(self):
        manager, position = self._manager_with_position()

        with patch.object(exchange, "place_market_order", return_value={"orderId": 2}) as market_order, \
             patch.object(exchange, "resolve_market_fill_price", return_value=100.5):
            total_quantity, entry_price, used_fallback, error = manager._resolve_retracement_market_fallback(
                position, 0.4, 99.5
            )

        market_order.assert_called_once_with("BTCUSDT", "BUY", 0.6)
        self.assertIsNone(error)
        self.assertAlmostEqual(total_quantity, 1.0)
        self.assertAlmostEqual(entry_price, 99.5 * 0.4 + 100.5 * 0.6)
        self.assertTrue(used_fallback)

    def test_market_order_failure_returns_an_error(self):
        manager, position = self._manager_with_position()

        with patch.object(exchange, "place_market_order", side_effect=RuntimeError("boom")):
            total_quantity, entry_price, used_fallback, error = manager._resolve_retracement_market_fallback(
                position, 0.0, None
            )

        self.assertIsNone(total_quantity)
        self.assertIsNone(entry_price)
        self.assertIn("boom", error)


class ResolveTp1PriceTests(unittest.TestCase):
    """config.TP_STATIC_ROI_ENABLED - a static TP1 is a pure function of
    entry_price, so unlike a structure-resolved one it must be recomputed
    when the real entry_price differs from the plan's own (real bug found
    live, 2026-08-21: a retracement fill better than the trigger price
    left TP1 computed off the stale trigger)."""

    def test_non_static_plan_returns_the_original_tp1_unchanged(self):
        plan = {"tp1_price": 102, "tp1_static_roi_pct": None}
        result = _resolve_tp1_price(plan, entry_price=99, side="BUY")
        self.assertEqual(result, 102)

    def test_static_plan_recomputes_from_the_real_entry_price(self):
        plan = {"tp1_price": 104.0, "tp1_static_roi_pct": 40}  # originally computed off entry=100

        with patch.object(config, "LEVERAGE", 10):
            result = _resolve_tp1_price(plan, entry_price=99, side="BUY")

        # 40% ROI / 10x leverage = 4% price move, off the REAL entry (99),
        # not the plan's original one (100, which gave 104.0).
        self.assertAlmostEqual(result, 99 * 1.04)

    def test_static_plan_mirrors_for_sell(self):
        plan = {"tp1_price": 96.0, "tp1_static_roi_pct": 40}

        with patch.object(config, "LEVERAGE", 10):
            result = _resolve_tp1_price(plan, entry_price=101, side="SELL")

        self.assertAlmostEqual(result, 101 * 0.96)

    def test_recompute_failure_falls_back_to_the_original_plan_value(self):
        plan = {"tp1_price": 104.0, "tp1_static_roi_pct": 40}

        with patch.object(config, "LEVERAGE", 0):  # makes price_at_roi_pct fail
            result = _resolve_tp1_price(plan, entry_price=99, side="BUY")

        self.assertEqual(result, 104.0)


class FinalizeRetracementEntryTests(unittest.TestCase):
    def test_dca_dual_tp_places_tp1_tp2_no_sl_and_becomes_dca_pending(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}) as tp2, \
             patch.object(exchange, "place_stop_loss") as sl:
            outcome = manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        self.assertIsNone(outcome)
        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["stage"], DCA_PENDING)
        self.assertEqual(final["entry_price"], 99.5)
        self.assertEqual(final["sl_price"], 98)  # structure-anchored, unchanged by the real fill
        self.assertIsNone(final["sl_order_id"])  # DCA_PENDING never gets a real SL
        self.assertEqual(final["tp1_order_id"], "tp1_1")
        self.assertEqual(final["tp2_order_id"], "tp2_1")
        self.assertEqual(final["dca_price"], 96)
        sl.assert_not_called()
        tp1.assert_called_once()
        tp2.assert_called_once()

    def test_static_tp1_is_recomputed_from_the_real_fill_not_the_stale_trigger(self):
        # Real bug found live (2026-08-21, SLXUSDT): TP1 stayed computed
        # off the planned trigger price even though the retracement limit
        # filled at a meaningfully better real price - the real exchange
        # order (and the tracked value) must reflect the ACTUAL entry.
        manager = PositionManager()
        execution_result = {"shadow": False, "entry_order": {"orderId": "limit1"}, "retracement_price": 99.8}
        plan = _retracement_plan(dca=True, single_tp=False)
        plan["tp1_price"] = 104.0  # as originally computed off the planned entry_price=100
        plan["tp1_static_roi_pct"] = 40

        with patch.object(config, "DCA_ENABLED", True):
            manager.register_retracement_pending(plan, execution_result, trade_id="BTCUSDT_1")

        position = manager.positions["BTCUSDT"]

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            manager._finalize_retracement_entry(position, 99.0, 1.0, "LIMIT")  # real fill: 99.0, not the planned 100

        final = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(final["tp1_price"], 99.0 * 1.04)  # 40% ROI/10x off the REAL entry
        tp1.assert_called_once()
        self.assertAlmostEqual(tp1.call_args.args[3], 99.0 * 1.04)  # the REAL order uses the same price

    def test_static_tp_price_is_recomputed_from_the_real_fill_when_single_tp(self):
        # config.TP2_ENABLED=False - the same real bug class as the
        # tp1_price recompute test above, but for a single_tp plan's
        # tp_price: a static-ROI single-TP built from a retracement entry
        # must also reflect the REAL settled fill, not the stale planned
        # trigger price. _resolve_tp1_price itself reads plan["tp1_price"]/
        # plan.get("tp1_static_roi_pct"), so both are set here even
        # though tp1_price is never actually used for order placement in
        # single_tp mode - it's still the source this recompute reads.
        manager = PositionManager()
        execution_result = {"shadow": False, "entry_order": {"orderId": "limit1"}, "retracement_price": 99.8}
        plan = _retracement_plan(dca=True, single_tp=True)
        plan["tp_price"] = 104.0  # as originally computed off the planned entry_price=100
        plan["tp1_price"] = 104.0
        plan["tp1_static_roi_pct"] = 40

        with patch.object(config, "DCA_ENABLED", True):
            manager.register_retracement_pending(plan, execution_result, trade_id="BTCUSDT_1")

        position = manager.positions["BTCUSDT"]

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_1"}) as tp_full:
            manager._finalize_retracement_entry(position, 99.0, 1.0, "LIMIT")  # real fill: 99.0, not the planned 100

        final = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(final["tp_price"], 99.0 * 1.04)  # 40% ROI/10x off the REAL entry
        tp_full.assert_called_once()
        self.assertAlmostEqual(tp_full.call_args.args[2], 99.0 * 1.04)  # the REAL order uses the same price

    def test_single_tp_structure_price_is_unaffected_by_retracement_fill(self):
        # Mirrors compute_targets' own structure-anchored TP2/SL - a real
        # level, not a pure function of entry_price, so unlike the static-
        # ROI case above it must NOT drift with the real fill price.
        manager = PositionManager()
        execution_result = {"shadow": False, "entry_order": {"orderId": "limit1"}, "retracement_price": 99.8}
        plan = _retracement_plan(dca=True, single_tp=True)  # tp1_static_roi_pct left unset (None)

        with patch.object(config, "DCA_ENABLED", True):
            manager.register_retracement_pending(plan, execution_result, trade_id="BTCUSDT_1")

        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_1"}):
            manager._finalize_retracement_entry(position, 99.0, 1.0, "LIMIT")

        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["tp_price"], 106)  # unchanged from _retracement_plan's own default

    def test_dca_single_tp_places_only_the_full_tp(self):
        manager = _retracement_manager(dca=True, single_tp=True)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_1"}) as tp_full:
            outcome = manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        self.assertIsNone(outcome)
        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["stage"], DCA_PENDING)
        self.assertTrue(final["single_tp"])
        self.assertEqual(final["tp_order_id"], "tp_1")
        self.assertFalse(final["tp1_order_id"])
        self.assertFalse(final["tp2_order_id"])
        tp1.assert_not_called()

    def test_non_dca_places_sl_then_tp1_tp2_and_becomes_tp1_pending(self):
        manager = _retracement_manager(dca=False, single_tp=False)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_1"}) as sl, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        self.assertIsNone(outcome)
        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["stage"], TP1_PENDING)
        self.assertEqual(final["sl_order_id"], "sl_1")
        sl.assert_called_once_with("BTCUSDT", "BUY", 98)

    def test_non_dca_sl_failure_closes_at_market_and_returns_outcome(self):
        manager = _retracement_manager(dca=False, single_tp=False)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market") as close_market:
            outcome = manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        self.assertEqual(outcome, "RETRACEMENT_SL_PLACEMENT_FAILED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.0)

    def test_quantity_and_tp_split_reflect_the_real_settled_fill_not_the_plan(self):
        # The real point of this whole mechanism: a blended limit+market-
        # fallback fill can differ from the originally planned quantity -
        # TP1/TP2 must size off THAT, not silently reuse the plan's own.
        manager = _retracement_manager(dca=False, single_tp=False)
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_1"}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            manager._finalize_retracement_entry(position, 99.0, 1.7, "LIMIT")

        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["quantity"], 1.7)
        self.assertAlmostEqual(final["tp1_quantity"], 0.85)
        self.assertAlmostEqual(final["tp2_quantity"], 0.85)
        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.85, 102)

    def test_shadow_finalize_places_no_real_orders(self):
        manager = _retracement_manager(dca=True, single_tp=False, shadow=True)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full") as tp2, \
             patch.object(exchange, "place_stop_loss") as sl:
            outcome = manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        self.assertIsNone(outcome)
        final = manager.positions["BTCUSDT"]
        self.assertEqual(final["stage"], DCA_PENDING)
        self.assertTrue(final["shadow"])
        tp1.assert_not_called()
        tp2.assert_not_called()
        sl.assert_not_called()

    # config.RETRACEMENT_ENTRY_ENABLED observability (2026-08-25) -
    # signal_journal.append_retracement_settle is called with the real
    # settled entry_price and the fill_type/fill_lag this method received,
    # regardless of dca/single_tp/shadow shape or a downstream SL failure.

    def test_journals_the_real_entry_price_and_fill_type(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        position = manager.positions["BTCUSDT"]
        trade_id = position["trade_id"]

        with patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager._finalize_retracement_entry(position, 99.5, 1.0, "MARKET_FALLBACK")

        append_settle.assert_called_once()
        args = append_settle.call_args.args
        self.assertEqual(args[0], "BTCUSDT")
        self.assertEqual(args[1], trade_id)
        self.assertEqual(args[2], 99.5)
        self.assertEqual(args[3], "MARKET_FALLBACK")
        self.assertIsInstance(args[4], float)

    def test_journals_even_when_the_post_settle_sl_placement_fails(self):
        # entry_price/fill_type are known and worth keeping even when the
        # position closes right back out from a downstream SL failure.
        manager = _retracement_manager(dca=False, single_tp=False)
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market"), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        append_settle.assert_called_once()

    def test_fill_lag_reflects_real_elapsed_time_since_the_limit_was_placed(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        position = manager.positions["BTCUSDT"]
        position["limit_placed_at"] = time.time() - 184.2

        with patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager._finalize_retracement_entry(position, 99.5, 1.0, "LIMIT")

        fill_lag = append_settle.call_args.args[4]
        self.assertAlmostEqual(fill_lag, 184.2, delta=1.0)


class PollRetracementPendingTests(unittest.TestCase):
    def test_shadow_position_is_a_noop(self):
        manager = _retracement_manager(shadow=True)

        with patch.object(exchange, "get_order_status") as get_status:
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        get_status.assert_not_called()

    def test_unknown_status_is_a_noop_retry_next_poll(self):
        manager = _retracement_manager()

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("UNKNOWN")), \
             patch.object(exchange, "cancel_order") as cancel_order:
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)
        cancel_order.assert_not_called()

    def test_still_resting_keeps_waiting(self):
        manager = _retracement_manager()

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW")), \
             patch.object(exchange, "cancel_order") as cancel_order:
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)
        cancel_order.assert_not_called()

    def test_deep_timeout_keeps_resting_past_the_global_base_timeout(self):
        # config.RETRACEMENT_DEPTH_AWARE_ENABLED - the per-position value
        # must win, not the global config. Global patched to a SHORTER
        # value than elapsed time to prove that: if the code accidentally
        # still read the global, this would incorrectly expire.
        manager = _retracement_manager(dca=True, single_tp=False, retracement_timeout_seconds=600)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 400

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW", executed_qty=0.0)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order:
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)
        cancel_order.assert_not_called()
        market_order.assert_not_called()

    def test_deep_timeout_still_expires_once_its_own_window_elapses(self):
        manager = _retracement_manager(dca=True, single_tp=False, retracement_timeout_seconds=600)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 700

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW", executed_qty=0.0)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 2}) as market_order, \
             patch.object(exchange, "resolve_market_fill_price", return_value=101.0), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)

    def test_full_fill_settles_with_the_real_fill_price_and_no_market_fallback(self):
        manager = _retracement_manager(dca=True, single_tp=False)

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("FILLED", executed_qty=1.0, avg_price=99.8)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 99.8)
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")
        market_order.assert_not_called()  # nothing left to fall back for

    def test_invalidated_before_any_fill_drops_unfilled_and_never_falls_back(self):
        manager = _retracement_manager()
        candle = _candle(high=100, low=97)  # BUY sl_price=98 - touched, never filled

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW", executed_qty=0.0)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order:
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=candle)

        self.assertEqual(outcome, "RETRACEMENT_INVALIDATED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")
        market_order.assert_not_called()

    def test_expiry_with_zero_fill_falls_back_to_a_full_market_order(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW", executed_qty=0.0)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 2}) as market_order, \
             patch.object(exchange, "resolve_market_fill_price", return_value=101.0), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 101.0)
        self.assertEqual(position["quantity"], 1.0)
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)

    def test_expiry_with_a_partial_fill_blends_with_the_market_fallback(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=99.5)), \
             patch.object(exchange, "cancel_order"), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 2}) as market_order, \
             patch.object(exchange, "resolve_market_fill_price", return_value=100.5), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["entry_price"], 99.5 * 0.4 + 100.5 * 0.6)
        self.assertAlmostEqual(position["quantity"], 1.0)
        market_order.assert_called_once_with("BTCUSDT", "BUY", 0.6)

    def test_fill_race_after_cancel_is_caught_by_the_post_cancel_recheck(self):
        # Same discipline poll_pending_entry's own race test already
        # documents: a fill can land microseconds before the cancel takes
        # effect, so the post-cancel re-check must catch it instead of
        # treating this as a clean, unfilled expiry.
        manager = _retracement_manager(dca=True, single_tp=False)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        order_statuses = iter([
            _pending_order_status("NEW", executed_qty=0.0),  # initial check
            _pending_order_status("FILLED", executed_qty=1.0, avg_price=99.8),  # post-cancel re-check
        ])

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", side_effect=lambda *a, **k: next(order_statuses)), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 99.8)
        market_order.assert_not_called()  # the race fill covered the full quantity
        cancel_order.assert_called_once_with("BTCUSDT", "limit1")

    # config.RETRACEMENT_ENTRY_ENABLED observability (2026-08-25) - the
    # journaled fill_type must reflect whether the market fallback was
    # actually needed, not just whether the position settled successfully.

    def test_full_fill_journals_fill_type_limit(self):
        manager = _retracement_manager(dca=True, single_tp=False)

        with patch.object(exchange, "get_order_status", return_value=_pending_order_status("FILLED", executed_qty=1.0, avg_price=99.8)), \
             patch.object(exchange, "cancel_order"), \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        market_order.assert_not_called()
        self.assertEqual(append_settle.call_args.args[3], "LIMIT")

    def test_expired_fallback_journals_fill_type_market_fallback(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("NEW", executed_qty=0.0)), \
             patch.object(exchange, "cancel_order"), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 2}), \
             patch.object(exchange, "resolve_market_fill_price", return_value=101.0), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertEqual(append_settle.call_args.args[3], "MARKET_FALLBACK")

    def test_partial_fill_plus_fallback_journals_fill_type_market_fallback(self):
        manager = _retracement_manager(dca=True, single_tp=False)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "get_order_status", return_value=_pending_order_status("PARTIALLY_FILLED", executed_qty=0.4, avg_price=99.5)), \
             patch.object(exchange, "cancel_order"), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 2}), \
             patch.object(exchange, "resolve_market_fill_price", return_value=100.5), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            manager.poll_retracement_pending("BTCUSDT", latest_candle=None)

        self.assertEqual(append_settle.call_args.args[3], "MARKET_FALLBACK")


class PollShadowRetracementPendingTests(unittest.TestCase):
    def test_touches_retracement_price_settles_immediately(self):
        manager = _retracement_manager(dca=True, single_tp=False, shadow=True, retracement_price=99.8)
        candle = _candle(high=100, low=99.5)  # touches 99.8, never reaches sl=98

        with patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 99.8)
        self.assertTrue(position["shadow"])
        self.assertEqual(append_settle.call_args.args[3], "LIMIT")

    def test_touches_sl_before_retracement_invalidates(self):
        manager = _retracement_manager(shadow=True, retracement_price=99.8)
        candle = _candle(high=99.5, low=97.5)  # clears sl=98, never reaches retracement=99.8

        outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertEqual(outcome, "SHADOW_RETRACEMENT_INVALIDATED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_both_touched_in_the_same_candle_assumes_the_adverse_side_first(self):
        manager = _retracement_manager(shadow=True, retracement_price=99.8)
        candle = _candle(high=100, low=97)  # spans both sl=98 and retracement=99.8

        outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertEqual(outcome, "SHADOW_RETRACEMENT_INVALIDATED_UNFILLED")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_expiry_falls_back_to_the_candle_close(self):
        manager = _retracement_manager(dca=True, single_tp=False, shadow=True, retracement_price=99.8)
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 1000
        candle = _candle(high=100.5, low=100.2, close=100.3)  # never reaches retracement or sl

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}), \
             patch("position_manager.signal_journal.append_retracement_settle") as append_settle:
            outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 100.3)
        self.assertEqual(append_settle.call_args.args[3], "MARKET_FALLBACK")

    def test_still_waiting_is_a_noop(self):
        manager = _retracement_manager(shadow=True, retracement_price=99.8)
        candle = _candle(high=100.5, low=100.2)  # never reaches retracement or sl

        outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)

    def test_live_position_is_a_noop(self):
        manager = _retracement_manager(shadow=False)
        candle = _candle(high=100, low=99.5)

        outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)

    def test_deep_timeout_keeps_waiting_past_the_global_base_timeout(self):
        # config.RETRACEMENT_DEPTH_AWARE_ENABLED - same per-position-wins
        # proof as the live poller's version: global patched to a SHORTER
        # value than elapsed time.
        manager = _retracement_manager(
            dca=True, single_tp=False, shadow=True, retracement_price=99.8,
            retracement_timeout_seconds=600,
        )
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 400
        candle = _candle(high=100.5, low=100.2)  # never reaches retracement or sl

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300):
            outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], RETRACEMENT_PENDING)

    def test_deep_timeout_still_expires_once_its_own_window_elapses(self):
        manager = _retracement_manager(
            dca=True, single_tp=False, shadow=True, retracement_price=99.8,
            retracement_timeout_seconds=600,
        )
        manager.positions["BTCUSDT"]["limit_placed_at"] = time.time() - 700
        candle = _candle(high=100.5, low=100.2, close=100.3)  # never reaches retracement or sl

        with patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": "tp1_1"}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp2_1"}):
            outcome = manager.poll_shadow_retracement_pending("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertEqual(position["entry_price"], 100.3)


def _dca_plan(side="BUY"):
    plan = dict(_plan(side))

    if side == "SELL":
        plan.update({"dca_price": 104, "dca_quantity": 1.0, "atr": 1.0})
    else:
        plan.update({"dca_price": 96, "dca_quantity": 1.0, "atr": 1.0})

    return plan


def _dca_single_tp_plan(side="BUY"):
    # config.TP_STATIC_ROI_ENABLED shape - tp1/tp2 fields None, tp_price/
    # single_tp set instead (mirrors risk_manager.build_trade_plan's own
    # single_tp branch).
    plan = _dca_plan(side)
    plan.update({
        "tp1_price": None, "tp2_price": None, "tp1_quantity": None, "tp2_quantity": None,
        "tp_price": 106 if side == "BUY" else 94, "single_tp": True,
    })
    return plan


_DCA_RESULT_PLAN = {
    "entry_price": 98.0, "sl_price": 94.0, "tp_price": 106.0,
    "quantity": 2.0, "risk_distance": 4.0,
}


class RegisterDcaPendingTests(unittest.TestCase):
    def test_shadow_registration_has_no_sl_order_and_dca_pending_stage(self):
        manager = PositionManager()
        position = manager.register_dca_pending(_dca_plan(), {"shadow": True})

        self.assertTrue(manager.has_open_position("BTCUSDT"))
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertIsNone(position["sl_order_id"])
        self.assertFalse(position["dca_applied"])
        self.assertEqual(position["dca_price"], 96)
        self.assertEqual(position["dca_quantity"], 1.0)

    def test_live_registration_never_reads_an_sl_order_even_if_present(self):
        # execution.enter_trade_dca_pending never returns an "sl_order"
        # key at all (see its own docstring) - this proves register_dca_
        # pending doesn't accidentally look for one either.
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        position = manager.register_dca_pending(_dca_plan(), execution_result)

        self.assertIsNone(position["sl_order_id"])
        self.assertEqual(position["tp1_order_id"], "tp1_1")
        self.assertEqual(position["tp2_order_id"], "tp2_1")

    # config.DCA_RESTING_ORDER_ENABLED - dca_order_id, same treatment as
    # tp1_order_id/tp2_order_id/tp_order_id above.

    def test_live_registration_stores_dca_order_id_when_present(self):
        manager = PositionManager()
        execution_result = {"shadow": False, "dca_order": {"orderId": 555}}
        position = manager.register_dca_pending(_dca_plan(), execution_result)

        self.assertEqual(position["dca_order_id"], 555)

    def test_shadow_registration_dca_order_id_is_none(self):
        manager = PositionManager()
        position = manager.register_dca_pending(
            _dca_plan(), {"shadow": True, "dca_order": {"orderId": 555}},
        )

        self.assertIsNone(position["dca_order_id"])

    def test_live_registration_dca_order_id_none_when_absent(self):
        # config.DCA_RESTING_ORDER_ENABLED off, or placement failed -
        # execution_result never carries a "dca_order" key at all.
        manager = PositionManager()
        position = manager.register_dca_pending(_dca_plan(), {"shadow": False})

        self.assertFalse(position["dca_order_id"])

    def test_atr_is_carried_from_the_plan(self):
        manager = PositionManager()
        position = manager.register_dca_pending(dict(_dca_plan(), atr=2.5), {"shadow": True})
        self.assertEqual(position["atr"], 2.5)

    def test_real_entry_price_corrects_entry_breakeven_and_risk_distance(self):
        # Same fix as RegisterTests' equivalent - dca_price (a real
        # structure level, independent of entry slippage) stays untouched.
        manager = PositionManager()
        execution_result = {"shadow": False, "real_entry_price": 100.5}

        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            position = manager.register_dca_pending(_dca_plan(), execution_result)

        self.assertEqual(position["entry_price"], 100.5)
        self.assertEqual(position["dca_price"], 96)  # unshifted
        self.assertAlmostEqual(position["breakeven_price"], 100.5 * 1.0002)
        self.assertEqual(position["risk_distance"], 2.5)

    def test_no_real_entry_price_uses_the_planned_entry_unchanged(self):
        manager = PositionManager()
        position = manager.register_dca_pending(_dca_plan(), {"shadow": True})

        self.assertEqual(position["entry_price"], 100)
        self.assertEqual(position["breakeven_price"], 100.02)
        self.assertEqual(position["risk_distance"], 2.0)

    def test_dual_tp_plan_is_not_single_tp(self):
        manager = PositionManager()
        position = manager.register_dca_pending(_dca_plan(), {"shadow": True})

        self.assertFalse(position["single_tp"])
        self.assertIsNone(position["tp_price"])
        self.assertIsNone(position["tp_order_id"])

    def test_single_tp_plan_builds_the_single_tp_shape(self):
        manager = PositionManager()
        execution_result = {"shadow": False, "tp_order": {"algoId": "tp_solo"}}
        position = manager.register_dca_pending(_dca_single_tp_plan(), execution_result)

        self.assertTrue(position["single_tp"])
        self.assertEqual(position["tp_price"], 106)
        self.assertEqual(position["tp_order_id"], "tp_solo")
        self.assertIsNone(position["tp1_price"])
        self.assertIsNone(position["tp2_price"])
        # execution_result never carries "tp1_order"/"tp2_order" keys in
        # single_tp mode - _accepted_order_id(None) returns "" (its own
        # established behavior, not a None), same as any other missing order.
        self.assertFalse(position["tp1_order_id"])
        self.assertFalse(position["tp2_order_id"])
        self.assertIsNone(position["sl_order_id"])
        self.assertEqual(position["stage"], DCA_PENDING)


class IsDcaCandidateTests(unittest.TestCase):
    def _position(self, **overrides):
        base = {"stage": DCA_PENDING, "dca_applied": False, "dca_price": 96}
        base.update(overrides)
        return base

    def test_eligible_by_default(self):
        with patch.object(config, "DCA_ENABLED", True):
            self.assertTrue(PositionManager._is_dca_candidate(self._position()))

    def test_ineligible_when_disabled(self):
        with patch.object(config, "DCA_ENABLED", False):
            self.assertFalse(PositionManager._is_dca_candidate(self._position()))

    def test_ineligible_once_already_applied(self):
        with patch.object(config, "DCA_ENABLED", True):
            self.assertFalse(PositionManager._is_dca_candidate(self._position(dca_applied=True)))

    def test_ineligible_wrong_stage(self):
        with patch.object(config, "DCA_ENABLED", True):
            self.assertFalse(PositionManager._is_dca_candidate(self._position(stage=TP1_PENDING)))

    def test_ineligible_without_a_dca_price(self):
        with patch.object(config, "DCA_ENABLED", True):
            self.assertFalse(PositionManager._is_dca_candidate(self._position(dca_price=None)))


class DcaPriceReachedInRangeTests(unittest.TestCase):
    def test_buy_reached_when_the_candles_low_touches_or_crosses(self):
        position = {"side": "BUY", "dca_price": 96}
        self.assertTrue(PositionManager._dca_price_reached_in_range(
            position, [{"high": 100, "low": 96}]))
        self.assertTrue(PositionManager._dca_price_reached_in_range(
            position, [{"high": 100, "low": 95}]))
        self.assertFalse(PositionManager._dca_price_reached_in_range(
            position, [{"high": 100, "low": 97}]))

    def test_sell_reached_when_the_candles_high_touches_or_crosses(self):
        position = {"side": "SELL", "dca_price": 104}
        self.assertTrue(PositionManager._dca_price_reached_in_range(
            position, [{"high": 104, "low": 100}]))
        self.assertTrue(PositionManager._dca_price_reached_in_range(
            position, [{"high": 105, "low": 100}]))
        self.assertFalse(PositionManager._dca_price_reached_in_range(
            position, [{"high": 103, "low": 100}]))

    def test_uses_the_last_candle_in_the_list(self):
        position = {"side": "BUY", "dca_price": 96}
        candles = [{"high": 100, "low": 95}, {"high": 100, "low": 97}]
        self.assertFalse(PositionManager._dca_price_reached_in_range(position, candles))

    def test_none_or_empty_candles_is_never_reached(self):
        position = {"side": "BUY", "dca_price": 96}
        self.assertFalse(PositionManager._dca_price_reached_in_range(position, None))
        self.assertFalse(PositionManager._dca_price_reached_in_range(position, []))

    def test_candle_missing_high_or_low_is_never_reached(self):
        position = {"side": "BUY", "dca_price": 96}
        self.assertFalse(PositionManager._dca_price_reached_in_range(
            position, [{"high": 100}]))
        self.assertFalse(PositionManager._dca_price_reached_in_range(
            position, [{"low": 95}]))


class DcaBreakevenPriceReachedInRangeTests(unittest.TestCase):
    def test_buy_reached_when_the_candles_high_touches_or_crosses(self):
        position = {"side": "BUY", "breakeven_price": 98.02}
        self.assertTrue(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 98.02, "low": 97.5}]))
        self.assertTrue(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 99.0, "low": 97.5}]))
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 98.0, "low": 97.5}]))

    def test_sell_reached_when_the_candles_low_touches_or_crosses(self):
        position = {"side": "SELL", "breakeven_price": 97.98}
        self.assertTrue(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 98.5, "low": 97.98}]))
        self.assertTrue(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 98.5, "low": 97.0}]))
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 98.5, "low": 98.0}]))

    def test_none_or_empty_candles_is_never_reached(self):
        position = {"side": "BUY", "breakeven_price": 98.02}
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(position, None))
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(position, []))

    def test_candle_missing_high_or_low_is_never_reached(self):
        position = {"side": "BUY", "breakeven_price": 98.02}
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"high": 99.0}]))
        self.assertFalse(PositionManager._dca_breakeven_price_reached_in_range(
            position, [{"low": 97.5}]))


class ExecuteDcaShadowTests(unittest.TestCase):
    def setUp(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - these tests predate that
        # feature and don't mock signal_engine.direction_still_confirmed;
        # pinned off so a real .env flip to True can't silently reduce
        # dca_quantity out from under them - see ExecuteDcaPressureCheckTests
        # for that feature's own coverage.
        patcher = patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _manager_with_dca_pending(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": True})
        return manager

    def test_blends_entry_and_transitions_to_dca_active(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertTrue(position["dca_applied"])
        self.assertEqual(position["entry_price"], 98.0)
        self.assertEqual(position["sl_price"], 94.0)
        self.assertEqual(position["tp_price"], 106.0)
        self.assertEqual(position["quantity"], 2.0)
        self.assertIsNone(position["tp_order_id"])  # no real order in shadow

        # dca_fill_price passed through is the PLANNED dca_price, not a
        # separately-fetched real fill - same "trust the planned price"
        # convention execution.enter_trade already uses for the original
        # entry (see _execute_dca's own docstring).
        args, _ = build_plan.call_args
        self.assertEqual(args[2], 96)

    def test_breakeven_price_is_recomputed_from_the_new_blended_entry(self):
        # Real gap this closes: left stale at the ORIGINAL (pre-DCA)
        # entry's breakeven, config.STRUCTURE_STOP_MANAGEMENT_ENABLED's
        # trailing_stop_locked_profit flag would compare a post-DCA
        # trail against the wrong reference price.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            manager._execute_dca(position)

        self.assertAlmostEqual(position["breakeven_price"], 98.0 * 1.0002)

    def test_no_exchange_calls_happen_in_shadow_mode(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_stop_loss") as sl_order, \
             patch.object(exchange, "place_take_profit_full") as tp_order:
            manager._execute_dca(position)

        market_order.assert_not_called()
        sl_order.assert_not_called()
        tp_order.assert_not_called()

    def test_plan_computation_failure_leaves_position_unchanged_for_retry(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=None):
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        self.assertEqual(position["stage"], DCA_PENDING)
        self.assertFalse(position["dca_applied"])

    def test_never_places_a_trail_order_in_shadow_even_when_flag_enabled(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - placement is
        # strictly live-only (inside the `not shadow` branch); a shadow
        # position always keeps dca_trail_order_id at None regardless of
        # the flag.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(config, "DCA_BREAKEVEN_TRAILING_STOP_ENABLED", True), \
             patch.object(exchange, "place_trailing_stop_loss") as trail_order:
            manager._execute_dca(position)

        trail_order.assert_not_called()
        self.assertIsNone(position["dca_trail_order_id"])


class ExecuteDcaLiveTests(unittest.TestCase):
    def setUp(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - see ExecuteDcaShadowTests's
        # identical setUp for why.
        patcher = patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _manager_with_dca_pending(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        return manager

    def test_places_dca_order_cancels_old_tps_and_places_new_sl_and_tp(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]
        original_entry, original_quantity, original_atr = (
            position["entry_price"], position["quantity"], position.get("atr")
        )

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}) as market_order, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}) as tp_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}) as sl_order:
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)  # dca_quantity
        # The real fill price (96, from avgPrice) - not the planned
        # dca_price trigger - is what gets passed into build_dca_plan.
        build_plan.assert_called_once_with(
            original_entry, original_quantity, 96.0, 1.0, "BUY", None,
            atr=original_atr, buffer_atr_multiple=None,
        )
        self.assertEqual(cancel.call_count, 2)  # old TP1 + old TP2
        # Both the price/side args AND the clientAlgoId tag (see
        # _adopt_position's DCA_ACTIVE recovery, the reason this tag
        # exists at all) - checked separately since the tag's timestamp
        # suffix is non-deterministic.
        tp_order.assert_called_once()
        self.assertEqual(tp_order.call_args.args, ("BTCUSDT", "BUY", 106.0))
        self.assertTrue(tp_order.call_args.kwargs["client_algo_id"].startswith("dcaTP"))
        sl_order.assert_called_once()
        self.assertEqual(sl_order.call_args.args, ("BTCUSDT", "BUY", 94.0))
        self.assertTrue(sl_order.call_args.kwargs["client_algo_id"].startswith("dcaSL"))
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertEqual(position["tp_order_id"], "tp_new")
        self.assertEqual(position["sl_order_id"], "sl_new")

    def test_places_dormant_trail_order_when_flag_enabled(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - placed strictly
        # after the real SL succeeds, own separate try/except, tagged
        # with the dcaTrail prefix _adopt_position's restart recovery
        # looks for.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(config, "DCA_BREAKEVEN_TRAILING_STOP_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_TRAILING_CALLBACK_RATE", 0.2), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.15), \
             patch.object(exchange, "place_trailing_stop_loss", return_value={"algoId": "trail_new"}) as trail_order:
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        trail_order.assert_called_once()
        # quantity (args[2]) is plan["quantity"] - the post-DCA blended
        # size, matching _DCA_RESULT_PLAN's "quantity": 2.0.
        self.assertEqual(trail_order.call_args.args, ("BTCUSDT", "BUY", 2.0, 98.0 * 1.0015, 0.2))
        self.assertTrue(trail_order.call_args.kwargs["client_algo_id"].startswith("dcaTrail"))
        self.assertEqual(position["dca_trail_order_id"], "trail_new")

    def test_flag_disabled_never_attempts_trail_placement(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(config, "DCA_BREAKEVEN_TRAILING_STOP_ENABLED", False), \
             patch.object(exchange, "place_trailing_stop_loss") as trail_order:
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        trail_order.assert_not_called()
        self.assertIsNone(position["dca_trail_order_id"])

    def test_trail_placement_failure_is_best_effort_and_does_not_affect_sl_tp(self):
        # Own, separate try/except from the real-SL placement above - a
        # rejection here must never be mistaken for the atomic "first
        # real SL placement failed" case and trigger an emergency
        # market-close the position doesn't need.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all, \
             patch.object(config, "DCA_BREAKEVEN_TRAILING_STOP_ENABLED", True), \
             patch.object(exchange, "place_trailing_stop_loss", side_effect=Exception("boom")):
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        self.assertEqual(position["stage"], DCA_ACTIVE)
        self.assertEqual(position["sl_order_id"], "sl_new")
        self.assertEqual(position["tp_order_id"], "tp_new")
        self.assertIsNone(position["dca_trail_order_id"])
        close_market.assert_not_called()
        cancel_all.assert_not_called()

    def test_cancels_the_real_exchange_tp_orders_not_stale_local_ids(self):
        # Same ground-truth discipline _replace_sl_order already applies
        # to the SL side - cancel whatever is REALLY on the exchange, not
        # a possibly-stale locally-tracked id.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]
        position["tp1_order_id"] = "stale_tp1"
        position["tp2_order_id"] = "stale_tp2"
        real_tp1 = {"type": "TAKE_PROFIT_MARKET", "closePosition": False, "algoId": "real_tp1"}
        real_tp2 = {"type": "TAKE_PROFIT_MARKET", "closePosition": True, "algoId": "real_tp2"}

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[real_tp1, real_tp2]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}):
            manager._execute_dca(position)

        cancel.assert_any_call("BTCUSDT", "real_tp1")
        cancel.assert_any_call("BTCUSDT", "real_tp2")

    def test_sl_placement_failure_closes_position_at_market(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", side_effect=Exception("boom")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager._execute_dca(position)

        self.assertEqual(outcome, "DCA_SL_PLACEMENT_FAILED")
        close_market.assert_called_once_with("BTCUSDT", "BUY", 2.0)
        cancel_all.assert_called_once_with("BTCUSDT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_dca_order_error_leaves_position_unchanged_for_retry(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(exchange, "place_market_order", side_effect=Exception("boom")), \
             patch.object(risk_manager, "build_dca_plan") as build_plan:
            outcome = manager._execute_dca(position)

        self.assertIsNone(outcome)
        self.assertEqual(position["stage"], DCA_PENDING)
        build_plan.assert_not_called()


class ExecuteDcaPressureCheckTests(unittest.TestCase):
    """config.DCA_PRESSURE_CHECK_ENABLED - at the instant DCA fires,
    reuses signal_engine.direction_still_confirmed (against the
    position's own side) to decide whether this fire commits the normal
    DCA_SIZE_MULTIPLIER/DCA_STRUCTURE_STOP_ATR_BUFFER (confirmed - order
    flow still favors the original side) or a reduced size at a tighter
    stop (not confirmed - order flow has turned against it too). Never
    delays the fire itself - see config.py's comment for why."""

    def setUp(self):
        for name, value in (
            ("DCA_PRESSURE_SIZE_MULTIPLIER", 0.5),
            ("DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER", 0.25),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_pending(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": True})
        return manager

    def test_master_flag_off_never_calls_the_confirmation_check(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", False), \
             patch.object(signal_engine, "direction_still_confirmed") as confirmed_check, \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            manager._execute_dca(position)

        confirmed_check.assert_not_called()
        self.assertIsNone(position["dca_pressure_confirmed"])
        build_plan.assert_called_once_with(
            100, 1.0, 96, 1.0, "BUY", None, atr=1.0, buffer_atr_multiple=None,
        )

    def test_confirmed_keeps_the_full_size_and_normal_buffer(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            manager._execute_dca(position, htf_candles=["htf"], cvd_snapshot={}, current_price=96)

        self.assertTrue(position["dca_pressure_confirmed"])
        build_plan.assert_called_once_with(
            100, 1.0, 96, 1.0, "BUY", None, atr=1.0, buffer_atr_multiple=None,
        )

    def test_not_confirmed_reduces_size_and_tightens_the_buffer(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(False, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            manager._execute_dca(position, htf_candles=["htf"], cvd_snapshot={}, current_price=96)

        self.assertFalse(position["dca_pressure_confirmed"])
        # position["quantity"] (1.0) * DCA_PRESSURE_SIZE_MULTIPLIER (0.5),
        # not the plan's own dca_quantity (also 1.0 here, but a different
        # source - see config.py's comment on why these differ).
        build_plan.assert_called_once_with(
            100, 1.0, 96, 0.5, "BUY", None, atr=1.0, buffer_atr_multiple=0.25,
        )

    def test_not_confirmed_uses_the_reduced_quantity_for_the_real_dca_order(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(False, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}) as market_order, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}):
            manager._execute_dca(position, htf_candles=["htf"], cvd_snapshot={}, current_price=96)

        market_order.assert_called_once_with("BTCUSDT", "BUY", 0.5)  # reduced, not the plan's 1.0

    # config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED - real motivation
    # (2026-08-22): a BUY DCA'd right into the bottom of a real BTC
    # flash-crash. _dca_plan() defaults side="BUY", so a BEARISH crash is
    # the risk side these tests exercise.

    def test_crash_mode_forces_not_confirmed_even_when_the_normal_check_says_confirmed(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            manager._execute_dca(
                position, htf_candles=["htf"], cvd_snapshot={}, current_price=96,
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertFalse(position["dca_pressure_confirmed"])
        build_plan.assert_called_once_with(
            100, 1.0, 96, 0.5, "BUY", None, atr=1.0, buffer_atr_multiple=0.25,
        )

    def test_crash_mode_on_the_aligned_side_does_not_force(self):
        # A BUY during a BULLISH crash (a violent rally) is the side that
        # benefits, not the one this exists to protect.
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager._execute_dca(
                position, htf_candles=["htf"], cvd_snapshot={}, current_price=96,
                crash_snapshot={"available": True, "active": True, "direction": "BULLISH"},
            )

        self.assertTrue(position["dca_pressure_confirmed"])

    def test_crash_mode_inactive_does_not_force(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager._execute_dca(
                position, htf_candles=["htf"], cvd_snapshot={}, current_price=96,
                crash_snapshot={"available": True, "active": False, "direction": None},
            )

        self.assertTrue(position["dca_pressure_confirmed"])

    def test_crash_force_flag_off_does_not_force(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", False), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager._execute_dca(
                position, htf_candles=["htf"], cvd_snapshot={}, current_price=96,
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertTrue(position["dca_pressure_confirmed"])

    def test_crash_master_flag_off_does_not_force(self):
        manager = self._manager_with_dca_pending()
        position = manager.positions["BTCUSDT"]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_ENABLED", False), \
             patch.object(config, "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", True), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager._execute_dca(
                position, htf_candles=["htf"], cvd_snapshot={}, current_price=96,
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertTrue(position["dca_pressure_confirmed"])


class PollShadowDcaPendingTests(unittest.TestCase):
    def setUp(self):
        # PROFIT_PROTECTION_ENABLED/EARLY_BREAKEVEN_ENABLED now correctly
        # apply during DCA_PENDING too (see _is_profit_protection_
        # candidate/_is_early_breakeven_candidate's stage check - real
        # gap fixed 2026-08-17) - off by default here so these tests stay
        # isolated to the TP1-vs-DCA race itself, same discipline
        # PollLiveTests already uses. TryEarlyPromotionsShadowTests below
        # covers the promotion behavior directly.
        self.pp_patcher = patch.object(config, "PROFIT_PROTECTION_ENABLED", False)
        self.eb_patcher = patch.object(config, "EARLY_BREAKEVEN_ENABLED", False)
        self.pp_patcher.start()
        self.eb_patcher.start()
        self.addCleanup(self.pp_patcher.stop)
        self.addCleanup(self.eb_patcher.stop)

    def _manager_with_dca_pending(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": True})
        return manager

    def test_dca_fires_when_candle_touches_the_dca_price(self):
        manager = self._manager_with_dca_pending()
        candle = _candle(high=100, low=95)  # touches dca_price=96

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_dca_fire_passes_htf_candles_cvd_snapshot_and_candle_close_through(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - shadow mode has no real mark
        # price, so it uses the candle's own close, same convention
        # _dca_breakeven_confirmation's shadow call site already uses.
        manager = self._manager_with_dca_pending()
        candle = _candle(high=100, low=95, close=97)  # touches dca_price=96
        cvd_snapshot = {"available": True, "cvd_score": 0.5}

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})) as confirmed_check:
            manager.poll_shadow("BTCUSDT", candle, htf_candles=["htf"], cvd_snapshot=cvd_snapshot)

        confirmed_check.assert_called_once_with("BUY", ["htf"], None, cvd_snapshot, 97)

    def test_tp1_fires_when_dca_price_not_reached(self):
        # Real profit lock now, not flat breakeven - same 100.6 math as
        # PollShadowTests.test_tp1_pending_tp1_hit_moves_to_breakeven_and_
        # stays_open (entry=100, sl=98 via _dca_plan's underlying _plan()).
        manager = self._manager_with_dca_pending()
        candle = _candle(high=103, low=99)  # clears TP1=102, never reaches dca_price=96

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertEqual(position["sl_price"], 100.6)
        self.assertTrue(position["early_breakeven_profit_locked"])

    def test_dca_checked_before_tp1_when_both_touch_the_same_candle(self):
        # Deliberately conservative, same bias poll_shadow's own docstring
        # already applies to SL-vs-target ambiguity - a huge-range candle
        # touching both the DCA level and TP1 is treated as the adverse
        # DCA touch happening first.
        manager = self._manager_with_dca_pending()
        candle = _candle(high=103, low=95)  # clears both TP1=102 and dca_price=96

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager.poll_shadow("BTCUSDT", candle)

        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_neither_touched_leaves_stage_unchanged(self):
        manager = self._manager_with_dca_pending()
        candle = _candle(high=99, low=97)  # inside both levels

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)

    def _manager_with_single_tp_dca_pending(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_single_tp_plan(), {"shadow": True})
        return manager

    def test_single_tp_touch_closes_as_shadow_static_tp_hit(self):
        manager = self._manager_with_single_tp_dca_pending()
        candle = _candle(high=107, low=99)  # clears tp_price=106

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertEqual(outcome, "SHADOW_STATIC_TP_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_single_tp_not_touched_stays_open(self):
        manager = self._manager_with_single_tp_dca_pending()
        candle = _candle(high=103, low=99)  # clears neither tp_price=106 nor dca_price=96

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)

    def test_single_tp_dca_still_fires_when_dca_price_reached_first(self):
        manager = self._manager_with_single_tp_dca_pending()
        candle = _candle(high=100, low=95)  # touches dca_price=96, never reaches tp_price=106

        with patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_profit_protection_can_promote_a_dca_pending_position_before_tp1_or_dca(self):
        # The actual fix, exercised end to end: a DCA-pending position
        # that moves favorably enough gets promoted to BREAKEVEN_ACTIVE
        # with a real SL for the first time, same as a TP1_PENDING
        # position always could - it never reaches the DCA touch or the
        # TP1 touch at all.
        manager = self._manager_with_dca_pending()
        # _dca_plan(): entry=100, tp1=102, dca_price=96. 101 is short of
        # TP1 and far from dca_price - only a profit-protection arm
        # explains a promotion here.
        candle = _candle(high=101, low=99.5, close=101)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", False), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 1), \
             patch.object(config, "LEVERAGE", 10):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["stage"], BREAKEVEN_ACTIVE)
        self.assertTrue(position["profit_protection_applied"])


class PollShadowDcaActiveTests(unittest.TestCase):
    def setUp(self):
        # config.DCA_BREAKEVEN_ENABLED defaults True - off here so these
        # SL/TP-status and structure-trailing tests stay isolated from it
        # (this fixture's breakeven_price is left at register_dca_pending's
        # pre-DCA value, not recomputed for entry=98, so it would otherwise
        # arm unexpectedly). DcaBreakevenTests below covers it directly.
        # config.DCA_TP_STATIC_ROI_ENABLED - same isolation, same reasoning
        # (PollLiveDcaBreakevenConfirmationTests's own note on this) - this
        # fixture's tp_price=106.0 is deliberately fixed, not left to
        # self-heal onto whatever the live ROI target happens to compute.
        for name, value in (
            ("DCA_BREAKEVEN_ENABLED", False),
            ("DCA_TP_STATIC_ROI_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_active(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": True})
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "entry_price": 98.0, "sl_price": 94.0,
            "tp_price": 106.0, "quantity": 2.0, "dca_applied": True,
            "dca_breakeven_applied": False,
        })
        return manager

    def test_sl_hit_closes_as_a_loss(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=95, low=93)  # touches sl_price=94

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertEqual(outcome, "SHADOW_DCA_SL_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_tp_hit_closes_as_a_win(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=107, low=99)  # touches tp_price=106

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertEqual(outcome, "SHADOW_DCA_TP_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_neither_touched_stays_open(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=100, low=96)

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_dca_breakeven_arms_when_price_reaches_breakeven(self):
        # entry=98, sl=94 (a real loss level) - a candle whose close
        # reaches breakeven_price=98.02 closes the gap this feature
        # exists for: nothing else would have moved this SL yet.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02
        candle = _candle(high=98.5, low=97.5, close=98.02)

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 98.02)

    def test_dca_breakeven_does_not_arm_below_breakeven(self):
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02
        candle = _candle(high=97.5, low=95, close=97.0)

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 94.0)

    def test_dca_breakeven_disabled_is_a_noop(self):
        manager = self._manager_with_dca_active()  # DCA_BREAKEVEN_ENABLED=False from setUp
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02
        candle = _candle(high=99, low=97.5, close=98.5)  # would otherwise arm

        outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 94.0)

    def test_profit_protection_can_arm_a_dca_active_position(self):
        # The actual fix, exercised end to end: entry=98, tp_price=106 (8
        # wide). ACTIVATION_PCT_OF_TP1=50 -> arms once price is 4 above
        # entry (102). close=103 clears that. LOCK_PCT_OF_TP1=25 -> flat
        # floor of 98+2=100; RETRACE_PCT=50 off a peak of 103 -> 98 +
        # (103-98)*0.5 = 100.5, the more favorable of the two.
        manager = self._manager_with_dca_active()
        candle = _candle(high=103, low=99, close=103)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["profit_protection_applied"])
        self.assertEqual(position["stage"], DCA_ACTIVE)  # no stage transition needed - already protected
        self.assertAlmostEqual(position["sl_price"], 100.5)

    def test_profit_protection_trails_further_once_armed(self):
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position.update({
            "profit_protection_applied": True,
            "profit_protection_profit_locked": True,
            "profit_protection_peak_price": 103.0,
            "sl_price": 100.5,
        })
        candle = _candle(high=105, low=104, close=105)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["profit_protection_peak_price"], 105.0)
        # retrace_price = 98 + (105-98)*0.5 = 101.5; flat lock = 100 - max is 101.5
        self.assertAlmostEqual(position["sl_price"], 101.5)

    def test_profit_protection_disabled_is_a_noop(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=103, low=99, close=103)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False):
            outcome = manager.poll_shadow("BTCUSDT", candle)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["profit_protection_applied"])
        self.assertEqual(position["sl_price"], 94.0)  # untouched

    def test_structure_trailing_can_replace_a_dca_active_sl(self):
        # breakeven_price set explicitly to what _execute_dca would
        # really compute for entry=98 (see ExecuteDcaShadowTests.test_
        # breakeven_price_is_recomputed_from_the_new_blended_entry) - a
        # candidate of 99.0 is a genuine locked profit against THAT
        # reference, but would wrongly read as a scratch against the
        # stale pre-DCA breakeven (100.02) this fixture used to leave in
        # place before that fix.
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position["breakeven_price"] = 98.02
        candle = _candle(high=103, low=99, close=102)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False), \
             patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 99.0},
             ):
            outcome = manager.poll_shadow("BTCUSDT", candle, candles=["candle"])

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["sl_price"], 99.0)
        self.assertTrue(position["trailing_stop_locked_profit"])

    def test_structure_trailing_does_not_loosen_an_already_better_sl(self):
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position["sl_price"] = 101.0  # already better than the swing below
        candle = _candle(high=104, low=101.5, close=103)  # stays clear of both sl_price and tp_price

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False), \
             patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 99.0},
             ):
            outcome = manager.poll_shadow("BTCUSDT", candle, candles=["candle"])

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], 101.0)  # untouched

    def test_structure_trailing_disabled_or_no_candles_is_a_noop(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=103, low=99, close=102)

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", False), \
             patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True):
            outcome = manager.poll_shadow("BTCUSDT", candle, candles=None)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], 94.0)  # untouched


class PollShadowDcaBreakevenConfirmationTests(unittest.TestCase):
    """Shadow-mode mirror of PollLiveDcaBreakevenConfirmationTests - same
    two-phase config.DCA_BREAKEVEN_CONFIRMATION_ENABLED / ..._WITHHOLD_
    ENABLED rollout, simulated instead of hitting real exchange calls."""

    def setUp(self):
        for name, value in (
            ("DCA_BREAKEVEN_ENABLED", True),
            ("PROFIT_PROTECTION_ENABLED", False),
            ("HTF_TREND_FRESHNESS_ENABLED", True),
            ("EFFICIENCY_RATIO_GATE_ENABLED", True),
            ("SIGNAL_MIN_CVD_SCORE", 0.15),
            ("EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_active(self):
        manager = PositionManager()
        manager.register_dca_pending(_dca_plan(), {"shadow": True})
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "entry_price": 98.0, "sl_price": 94.0,
            "tp_price": 106.0, "quantity": 2.0, "dca_applied": True,
            "dca_breakeven_applied": False, "dca_breakeven_direction_confirmed": None,
            "breakeven_price": 98.02,
        })
        return manager

    def _confirming_structure_mocks(self):
        return (
            patch.object(
                market_structure, "structure_state",
                return_value={"available": True, "trend": "BULLISH"},
            ),
            patch.object(market_structure, "exponential_moving_average", return_value=95.0),
            patch.object(
                market_structure, "analyze",
                return_value={"available": True, "efficiency_ratio": 0.5},
            ),
        )

    def test_master_flag_off_behaves_as_before(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=98.5, low=97.5, close=98.02)

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", False):
            outcome = manager.poll_shadow(
                "BTCUSDT", candle, candles=["ltf"], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 98.02)
        self.assertIsNone(position["dca_breakeven_direction_confirmed"])

    def test_confirmed_but_withhold_disabled_still_applies_breakeven(self):
        s1, s2, s3 = self._confirming_structure_mocks()
        manager = self._manager_with_dca_active()
        candle = _candle(high=98.5, low=97.5, close=98.02)

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", False), \
             s1, s2, s3:
            outcome = manager.poll_shadow(
                "BTCUSDT", candle, candles=["ltf"], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertTrue(position["dca_breakeven_direction_confirmed"])
        self.assertEqual(position["sl_price"], 98.02)

    def test_confirmed_and_withhold_enabled_skips_the_move(self):
        s1, s2, s3 = self._confirming_structure_mocks()
        manager = self._manager_with_dca_active()
        candle = _candle(high=98.5, low=97.5, close=98.02)

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", True), \
             s1, s2, s3:
            outcome = manager.poll_shadow(
                "BTCUSDT", candle, candles=["ltf"], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertTrue(position["dca_breakeven_direction_confirmed"])
        self.assertEqual(position["sl_price"], 94.0)  # unchanged - real (wide) SL untouched

    def test_not_confirmed_and_withhold_enabled_applies_breakeven_normally(self):
        manager = self._manager_with_dca_active()
        candle = _candle(high=98.5, low=97.5, close=98.02)

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "trend": "BEARISH"},
             ), \
             patch.object(market_structure, "exponential_moving_average", return_value=95.0), \
             patch.object(
                 market_structure, "analyze",
                 return_value={"available": True, "efficiency_ratio": 0.5},
             ):
            outcome = manager.poll_shadow(
                "BTCUSDT", candle, candles=["ltf"], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertFalse(position["dca_breakeven_direction_confirmed"])
        self.assertEqual(position["sl_price"], 98.02)


class PollLiveDcaPendingTests(unittest.TestCase):
    def setUp(self):
        # See PollShadowDcaPendingTests.setUp - same isolation, same
        # real gap it's guarding against.
        for name, value in (
            ("MAE_TRACKING_ENABLED", False),
            ("PROFIT_PROTECTION_ENABLED", False),
            ("EARLY_BREAKEVEN_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_pending(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        return manager

    def test_dca_fires_when_the_candles_low_reaches_dca_price(self):
        # config._dca_price_reached_in_range - the candle's low (not a
        # point mark price) is what decides the DCA trigger now.
        manager = self._manager_with_dca_pending()
        candles = [{"high": 100.0, "low": 95.0}]

        with patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_dca_does_not_fire_when_the_candles_low_never_reached_dca_price(self):
        # The mirror of the test above - mark price alone would have said
        # "not reached" either way here, but this confirms the candle's
        # own low (97, above dca_price=96) is genuinely what's decisive.
        manager = self._manager_with_dca_pending()
        candles = [{"high": 100.0, "low": 97.0}]

        with patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)

    def test_dca_fires_on_a_wick_the_old_point_price_check_would_have_missed(self):
        # The actual gap this fix closes: price wicked down through
        # dca_price and back up between two poll ticks. By the time this
        # poll runs, the mark price (99.0) has already recovered and is
        # nowhere near dca_price=96 - the OLD _dca_price_reached(position,
        # current_price=99.0) would have returned False here, exactly the
        # scenario that let ~29% of resolved trades' adverse excursions go
        # completely undetected. The candle's low (94.0) still remembers
        # the wick, so _dca_price_reached_in_range must fire anyway.
        manager = self._manager_with_dca_pending()
        candles = [{"high": 100.0, "low": 94.0}]

        with patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_dca_fire_passes_htf_candles_cvd_snapshot_and_current_price_through(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - poll_live must forward the
        # same htf_candles/cvd_snapshot/current_price it already has this
        # tick into _execute_dca rather than an extra fetch.
        manager = self._manager_with_dca_pending()
        cvd_snapshot = {"available": True, "cvd_score": 0.5}
        candles = [{"high": 100.0, "low": 95.0}]

        with patch.object(config, "DCA_PRESSURE_CHECK_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN), \
             patch.object(signal_engine, "direction_still_confirmed", return_value=(True, {})) as confirmed_check:
            manager.poll_live("BTCUSDT", candles=candles, htf_candles=["htf"], cvd_snapshot=cvd_snapshot)

        confirmed_check.assert_called_once_with("BUY", ["htf"], candles, cvd_snapshot, 95.0)

    def test_tp1_finished_promotes_normally_without_dca(self):
        manager = self._manager_with_dca_pending()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], BREAKEVEN_ACTIVE)

    def test_tp2_finished_direct_closes_as_a_win(self):
        manager = self._manager_with_dca_pending()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp2_1" else "NEW"

        with patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "TP2_HIT_DIRECT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def _manager_with_single_tp_dca_pending(self):
        manager = PositionManager()
        execution_result = {"shadow": False, "tp_order": {"algoId": "tp_solo"}}
        manager.register_dca_pending(_dca_single_tp_plan(), execution_result)
        return manager

    def test_single_tp_finished_closes_as_static_tp_hit(self):
        manager = self._manager_with_single_tp_dca_pending()

        with patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", return_value="FINISHED"), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "STATIC_TP_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_single_tp_not_finished_stays_open(self):
        manager = self._manager_with_single_tp_dca_pending()

        with patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_single_tp_dca_still_fires_normally(self):
        manager = self._manager_with_single_tp_dca_pending()
        candles = [{"high": 100.0, "low": 95.0}]

        with patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)


class PollLiveDcaRestingOrderTests(unittest.TestCase):
    """config.DCA_RESTING_ORDER_ENABLED - a position with a real resting
    DCA order polls exchange.get_order_status (a plain order, reports
    partial fills via executed_qty) instead of watching candle ranges."""

    def setUp(self):
        for name, value in (
            ("MAE_TRACKING_ENABLED", False),
            ("PROFIT_PROTECTION_ENABLED", False),
            ("EARLY_BREAKEVEN_ENABLED", False),
            ("DCA_PRESSURE_CHECK_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_resting_dca(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
            "dca_order": {"orderId": "dca_resting_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        return manager

    def test_unfilled_resting_order_falls_through_to_no_op(self):
        # No candle-range check at all for this position - a resting
        # order that simply hasn't filled yet must not fire.
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "NEW", "executed_qty": 0.0, "avg_price": 0.0},
             ) as status, \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        status.assert_called_once_with("BTCUSDT", "dca_resting_1")
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)

    def test_unknown_status_falls_through_to_no_op(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "UNKNOWN", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)
        # Transient lookup failure - not a real "gone" signal, must stay
        # tracked, not cleared.
        self.assertEqual(manager.positions["BTCUSDT"]["dca_order_id"], "dca_resting_1")

    # config.DCA_RESTING_ORDER_ENABLED - real bug found live (2026-08-26,
    # MEUSDT/UBUSDT): a genuinely gone resting order used to read
    # identically to "still resting" forever (only executed_qty was ever
    # checked, never status), silently disabling both this path and
    # poll_live's own candle-range fallback. Ground-truth self-heal: a
    # terminal, not-going-to-fill status clears dca_order_id so the
    # candle-range fallback takes over on the very next poll.

    def test_canceled_order_clears_dca_order_id_for_the_fallback(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "CANCELED", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_PENDING)
        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_expired_order_clears_dca_order_id(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "EXPIRED", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_rejected_order_clears_dca_order_id(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "REJECTED", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_not_found_order_clears_dca_order_id(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "NOT_FOUND", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_cleared_dca_order_id_lets_the_very_next_poll_use_the_candle_range_fallback(self):
        # The actual self-heal payoff: once cleared, the next poll's
        # `if config.DCA_RESTING_ORDER_ENABLED and position.get(
        # "dca_order_id")` check is False, so it now falls through to the
        # `elif self._dca_price_reached_in_range(...)` fallback instead of
        # calling get_order_status on a dead order forever.
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "CANCELED", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            manager.poll_live("BTCUSDT")

        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])
        candles = [{"high": 100.0, "low": 95.0}]  # reaches dca_price=96

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "get_order_status") as status, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            manager.poll_live("BTCUSDT", candles=candles)

        status.assert_not_called()  # never polls a dca_order_id that's now None
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_full_fill_executes_dca_with_the_real_fill_price_and_quantity(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=96.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "FILLED", "executed_qty": 1.0, "avg_price": 95.8},
             ), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            outcome = manager.poll_live("BTCUSDT")

        # No reactive market order - the fill already happened via the
        # resting order.
        market_order.assert_not_called()
        build_plan.assert_called_once()
        _, kwargs = build_plan.call_args
        args = build_plan.call_args.args
        self.assertEqual(args[2], 95.8)  # dca_fill_price = real avg_price
        self.assertEqual(args[3], 1.0)   # dca_quantity = real executed_qty
        # Unfilled remainder cleanup attempted regardless (harmless no-op
        # on a fully-filled order).
        cancel_order.assert_called_once_with("BTCUSDT", "dca_resting_1")
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)
        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_partial_fill_uses_the_real_partial_quantity_not_the_planned_one(self):
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=96.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "PARTIALLY_FILLED", "executed_qty": 0.4, "avg_price": 95.9},
             ), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN) as build_plan:
            outcome = manager.poll_live("BTCUSDT")

        market_order.assert_not_called()
        args = build_plan.call_args.args
        self.assertEqual(args[2], 95.9)
        self.assertEqual(args[3], 0.4)
        # The still-unfilled remainder must not keep resting once the
        # position has moved to the new post-DCA plan.
        cancel_order.assert_called_once_with("BTCUSDT", "dca_resting_1")
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_remainder_cancel_failure_does_not_raise(self):
        # A fully-filled order rejecting the cleanup cancel (already gone)
        # must not prevent the DCA from completing successfully.
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=96.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "FILLED", "executed_qty": 1.0, "avg_price": 95.8},
             ), \
             patch.object(exchange, "cancel_order", side_effect=RuntimeError("order does not exist")), \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT")

        market_order.assert_not_called()
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_sl_placement_failure_still_closes_the_position(self):
        # The atomic "SL must exist" discipline is inherited unchanged -
        # a resting-order fill is not a special case for this failure path.
        manager = self._manager_with_resting_dca()

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=96.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "FILLED", "executed_qty": 1.0, "avg_price": 95.8},
             ), \
             patch.object(exchange, "cancel_order"), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders"), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "DCA_SL_PLACEMENT_FAILED")
        close_market.assert_called_once()

    def test_flag_off_falls_back_to_candle_range_even_with_a_stored_order_id(self):
        # A position registered while the flag was on, then the flag
        # turned back off - must not poll order status, must use the
        # existing candle-range fallback unchanged.
        manager = self._manager_with_resting_dca()
        candles = [{"high": 100.0, "low": 95.0}]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", False), \
             patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "get_order_status") as status, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        status.assert_not_called()
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_no_stored_order_id_falls_back_to_candle_range_even_with_flag_on(self):
        # config.DCA_RESTING_ORDER_ENABLED on, but this position has no
        # dca_order_id (registered before the flag was turned on, or
        # placement failed) - must fall back, not crash on a missing id.
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"}, "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        candles = [{"high": 100.0, "low": 95.0}]

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=95.0), \
             patch.object(exchange, "get_order_status") as status, \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "96"}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_new"}), \
             patch.object(risk_manager, "build_dca_plan", return_value=_DCA_RESULT_PLAN):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        status.assert_not_called()
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], DCA_ACTIVE)

    def test_tp1_fills_first_cancels_the_still_resting_dca_order(self):
        # DCA is never needed once TP1 fills - a still-resting DCA-add
        # order must not keep resting past this point (mirrors
        # _execute_dca's own symmetric cleanup of TP1/TP2 when DCA fires
        # first instead).
        manager = self._manager_with_resting_dca()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "NEW", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], BREAKEVEN_ACTIVE)
        cancel_order.assert_called_once_with("BTCUSDT", "dca_resting_1")
        self.assertIsNone(manager.positions["BTCUSDT"]["dca_order_id"])

    def test_tp1_fill_cancel_failure_does_not_block_promotion(self):
        manager = self._manager_with_resting_dca()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(
                 exchange, "get_order_status",
                 return_value={"status": "NEW", "executed_qty": 0.0, "avg_price": 0.0},
             ), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "cancel_order", side_effect=RuntimeError("gone")), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], BREAKEVEN_ACTIVE)

    def test_tp1_pending_position_without_a_dca_order_id_is_unaffected(self):
        # _promote_to_breakeven is shared with plain TP1_PENDING positions
        # (register(), not register_dca_pending()) - .get("dca_order_id")
        # must be a safe no-op for a position dict that never has the key.
        manager = PositionManager()
        execution_result = {
            "shadow": False, "sl_order": {"algoId": "sl_1"},
            "tp1_order": {"algoId": "tp1_1"}, "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register(_plan(), execution_result)

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp1_1" else "NEW"

        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=101.0), \
             patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 0.5}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "cancel_order") as cancel_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live("BTCUSDT")

        cancel_order.assert_not_called()
        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["stage"], BREAKEVEN_ACTIVE)


class EnsureProtectionOrdersSingleTpDcaPendingTests(unittest.TestCase):
    """config.TP_STATIC_ROI_ENABLED - self-heal for a single-TP DCA_PENDING
    position's missing tp_order_id, mirroring the existing TP1/TP2 self-
    heal tests' shape."""

    def _manager_with_single_tp_dca_pending(self, tp_order_id=None):
        manager = PositionManager()
        execution_result = {"shadow": False, "tp_order": {"algoId": "tp_solo"} if tp_order_id else None}
        manager.register_dca_pending(_dca_single_tp_plan(), execution_result)
        manager.positions["BTCUSDT"]["tp_order_id"] = tp_order_id
        return manager

    def test_missing_tp_is_resynced_from_a_real_exchange_order(self):
        manager = self._manager_with_single_tp_dca_pending(tp_order_id=None)
        real_tp = {"type": "TAKE_PROFIT_MARKET", "closePosition": "true", "algoId": "real_tp"}

        with patch.object(exchange, "get_open_algo_orders", return_value=[real_tp]), \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._ensure_protection_orders(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["tp_order_id"], "real_tp")
        place.assert_not_called()

    def test_missing_tp_with_none_on_exchange_places_a_new_one(self):
        manager = self._manager_with_single_tp_dca_pending(tp_order_id=None)

        with patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_recovered"}) as place:
            outcome = manager._ensure_protection_orders(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["tp_order_id"], "tp_recovered")
        place.assert_called_once_with("BTCUSDT", "BUY", 106)

    def test_present_tp_order_id_is_a_noop(self):
        manager = self._manager_with_single_tp_dca_pending(tp_order_id="tp_solo")

        with patch.object(exchange, "get_open_algo_orders") as get_open, \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._ensure_protection_orders(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        get_open.assert_not_called()
        place.assert_not_called()

    def test_minus_2021_closes_the_whole_position_as_a_win(self):
        # Unlike TP1's -2021 fallback (_market_close_tp1: partial close +
        # promote), there's nothing left to promote once the single TP
        # accounts for the whole position - must close outright as
        # STATIC_TP_HIT, never a loss-side outcome.
        manager = self._manager_with_single_tp_dca_pending(tp_order_id=None)

        with patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_full",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager._ensure_protection_orders(manager.positions["BTCUSDT"])

        self.assertEqual(outcome, "STATIC_TP_HIT")
        market_close.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        cancel_all.assert_called_once_with("BTCUSDT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_market_close_failure_leaves_it_for_the_next_poll(self):
        manager = self._manager_with_single_tp_dca_pending(tp_order_id=None)

        with patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_full",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market", side_effect=RuntimeError("boom")):
            outcome = manager._ensure_protection_orders(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_poll_live_propagates_the_ensure_protection_orders_outcome(self):
        # Full integration: poll_live must return the outcome from
        # _ensure_protection_orders instead of continuing to process a
        # position that already closed this same tick.
        manager = self._manager_with_single_tp_dca_pending(tp_order_id=None)

        with patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(
                 exchange, "place_take_profit_full",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market"), \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "STATIC_TP_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))


class PollLiveDcaActiveTests(unittest.TestCase):
    def setUp(self):
        # PROFIT_PROTECTION_ENABLED and DCA_BREAKEVEN_ENABLED both apply to
        # DCA_ACTIVE (see _is_dca_profit_protection_candidate/
        # _is_dca_breakeven_candidate) and both default True - off here so
        # these SL/TP-status tests stay isolated and don't need to mock
        # exchange.get_mark_price. DcaActiveProfitProtectionTests/
        # DcaBreakevenTests below cover each directly.
        # config.DCA_TP_STATIC_ROI_ENABLED - same isolation, same
        # reasoning (PollShadowDcaActiveTests's own note on this).
        for name, value in (
            ("MAE_TRACKING_ENABLED", False),
            ("PROFIT_PROTECTION_ENABLED", False),
            ("DCA_BREAKEVEN_ENABLED", False),
            ("DCA_TP_STATIC_ROI_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_active(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "sl_order_id": "sl_new", "tp_order_id": "tp_new",
            "entry_price": 98.0, "sl_price": 94.0, "tp_price": 106.0, "quantity": 2.0,
            "dca_applied": True, "dca_breakeven_applied": False,
        })
        return manager

    def test_sl_finished_closes_as_dca_sl_hit(self):
        manager = self._manager_with_dca_active()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "sl_new" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "DCA_SL_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_tp_finished_closes_as_dca_tp_hit(self):
        manager = self._manager_with_dca_active()

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "tp_new" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "DCA_TP_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_neither_finished_stays_open(self):
        manager = self._manager_with_dca_active()

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_trail_finished_closes_as_dca_breakeven_trail_hit(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a resting native
        # trailing stop that fires is detected the same way sl_status/
        # tp_status already are, via _status_or_missing on the stored
        # order id, and sweeps all open orders exactly like the sibling
        # SL/TP-finished paths above.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["dca_trail_order_id"] = "trail_1"

        def status_side_effect(symbol, order_id):
            return "FINISHED" if order_id == "trail_1" else "NEW"

        with patch.object(exchange, "get_algo_order_status", side_effect=status_side_effect), \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            outcome = manager.poll_live("BTCUSDT")

        self.assertEqual(outcome, "DCA_BREAKEVEN_TRAIL_HIT")
        cancel_all.assert_called_once_with("BTCUSDT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))

    def test_no_trail_order_id_is_a_safe_no_op(self):
        # dca_trail_order_id stays None for every position that never had
        # the flag on (or whose placement failed) - _status_or_missing
        # already returns "MISSING" for a falsy id, so this must behave
        # identically to today (stays open, no new close reason).
        manager = self._manager_with_dca_active()
        self.assertIsNone(manager.positions["BTCUSDT"].get("dca_trail_order_id"))

        with patch.object(exchange, "get_algo_order_status", return_value="NEW"):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_dca_breakeven_arms_when_price_reaches_breakeven(self):
        # entry=98, sl=94 (a real loss level) - the candle's high recovers
        # to breakeven_price=98.02, the fix closes that gap by moving the
        # SL there so the trade can no longer close as a full loss.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02
        candles = [{"high": 98.02, "low": 97.5}]

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 98.02)
        self.assertEqual(position["sl_order_id"], "sl_be")
        self.assertEqual(position["sl_price"], 98.02)

    def test_dca_breakeven_arms_on_a_wick_the_old_point_price_check_would_have_missed(self):
        # The actual gap this fix closes: price wicked up to breakeven and
        # back down between two poll ticks. By the time this poll runs,
        # the mark price (97.5) has already dropped back below breakeven
        # (98.02) - the OLD _dca_breakeven_price_reached(position,
        # current_price=97.5) would have returned False here. The candle's
        # high (98.1) still remembers the touch, so the range check must
        # arm anyway.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02
        candles = [{"high": 98.1, "low": 97.5}]

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=97.5), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live("BTCUSDT", candles=candles)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 98.02)

    def test_dca_breakeven_does_not_arm_below_breakeven(self):
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=97.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 94.0)
        new_sl.assert_not_called()

    def test_dca_breakeven_does_not_re_arm_once_already_applied(self):
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position.update({"breakeven_price": 98.02, "dca_breakeven_applied": True, "sl_price": 98.02})

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        new_sl.assert_not_called()  # already armed - structure/profit-protection trailing takes over from here

    def test_dca_breakeven_does_not_arm_while_a_native_trail_order_is_resting(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a resting
        # dca_trail_order_id means the native order is already doing this
        # job server-side. Without _is_dca_breakeven_candidate's skip, the
        # poll-based path would replace the SL with a flat stop and (via
        # _replace_sl_order's own cleanup) cancel the trail order that was
        # just protecting the position better.
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position.update({"breakeven_price": 98.02, "dca_trail_order_id": "trail_1"})

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=99.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        self.assertFalse(position["dca_breakeven_applied"])
        new_sl.assert_not_called()
        self.assertEqual(position["dca_trail_order_id"], "trail_1")

    def test_profit_protection_arm_takes_priority_over_dca_breakeven_in_the_same_tick(self):
        # Mirrors _try_early_promotions' profit-protection-then-early-
        # breakeven ordering: if one tick's move already clears profit
        # protection's deeper threshold, that arm wins and returns first -
        # DCA breakeven's own arm never gets a chance to fire that tick.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["breakeven_price"] = 98.02

        with patch.object(config, "DCA_BREAKEVEN_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_pp"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["profit_protection_applied"])
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertEqual(position["sl_price"], 100.5)

    def test_profit_protection_can_arm_a_dca_active_position(self):
        # Same numbers as PollShadowDcaActiveTests' equivalent test:
        # entry=98, tp_price=106, ACTIVATION_PCT_OF_TP1=50 arms at 102,
        # mark price 103 clears it, lock/retrace both computed the same
        # way -> 100.5.
        manager = self._manager_with_dca_active()

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_pp"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["profit_protection_applied"])
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 100.5)
        self.assertEqual(position["sl_order_id"], "sl_pp")
        self.assertEqual(position["sl_price"], 100.5)

    def test_profit_protection_replace_retires_a_resting_dca_trail_order(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - _replace_sl_order's
        # centralized cleanup: a DIFFERENT mechanism (profit protection
        # here) just took over the SL, so a resting native trail order's
        # job is done and it must be cancelled + cleared, not left to
        # possibly conflict with a later third replace.
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["dca_trail_order_id"] = "trail_1"

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_pp"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["profit_protection_applied"])
        cancel.assert_any_call("BTCUSDT", "sl_new")
        cancel.assert_any_call("BTCUSDT", "trail_1")
        self.assertEqual(cancel.call_count, 2)
        self.assertIsNone(position["dca_trail_order_id"])

    def test_replace_sl_order_cleanup_is_a_no_op_without_a_trail_order(self):
        # No dca_trail_order_id at all (the normal case for every
        # position that never had the flag on) - the new cleanup must be
        # a true no-op, same behavior as before this feature existed.
        manager = self._manager_with_dca_active()
        self.assertIsNone(manager.positions["BTCUSDT"].get("dca_trail_order_id"))

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "get_mark_price", return_value=103.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_pp"}):
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        cancel.assert_called_once_with("BTCUSDT", "sl_new")

    def test_profit_protection_trails_once_armed(self):
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position.update({
            "profit_protection_applied": True,
            "profit_protection_profit_locked": True,
            "profit_protection_peak_price": 103.0,
            "sl_price": 100.5,
            "sl_order_id": "sl_pp",
        })

        with patch.object(config, "PROFIT_PROTECTION_ENABLED", True), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "get_mark_price", return_value=105.0), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_pp2"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT")

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        cancel.assert_called_once_with("BTCUSDT", "sl_pp")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 101.5)
        self.assertEqual(position["sl_price"], 101.5)

    def test_structure_trailing_can_replace_a_dca_active_sl(self):
        # Same breakeven_price reasoning as the shadow-mode equivalent -
        # a candidate of 99.0 only reads as a genuine locked profit
        # against the post-DCA breakeven (98.02), not the stale pre-DCA
        # one this fixture used to leave in place.
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position["breakeven_price"] = 98.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 99.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_struct"}) as new_sl:
            outcome = manager.poll_live("BTCUSDT", candles=["candle"])

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 99.0)
        self.assertEqual(position["sl_price"], 99.0)
        self.assertTrue(position["trailing_stop_locked_profit"])

    def test_structure_trailing_does_not_loosen_an_already_better_sl(self):
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position["sl_price"] = 101.0  # already better than the swing below

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 99.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_stop_loss") as new_sl:
            outcome = manager.poll_live("BTCUSDT", candles=["candle"])

        self.assertIsNone(outcome)
        new_sl.assert_not_called()
        self.assertEqual(manager.positions["BTCUSDT"]["sl_price"], 101.0)

    def test_minus_2021_during_structure_trailing_closes_as_dca_sl_hit(self):
        # Same real-world race as ReplaceSlOrderTests' equivalent, hit via
        # the structure-trailing path specifically (poll_live's own -2021
        # fallback for DCA_ACTIVE, not just the direct unit test).
        manager = self._manager_with_dca_active()
        position = manager.positions["BTCUSDT"]
        position["breakeven_price"] = 98.02

        with patch.object(config, "STRUCTURE_STOP_MANAGEMENT_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "last_swing_low": 99.0},
             ), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(
                 exchange, "place_stop_loss",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome = manager.poll_live("BTCUSDT", candles=["candle"])

        self.assertEqual(outcome, "DCA_SL_HIT")
        market_close.assert_called_once()
        self.assertFalse(manager.has_open_position("BTCUSDT"))


class ReplaceDcaTpOrderTests(unittest.TestCase):
    def _manager_with_dca_active(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "sl_order_id": "sl_new", "tp_order_id": "tp_old",
            "entry_price": 98.0, "sl_price": 94.0, "tp_price": 106.0, "quantity": 2.0,
            "dca_applied": True,
        })
        return manager

    def test_success_replaces_tp_order(self):
        manager = self._manager_with_dca_active()

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}) as place:
            outcome = manager._replace_dca_tp_order(manager.positions["BTCUSDT"], 110.0)

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertEqual(position["tp_order_id"], "tp_new")
        self.assertEqual(position["tp_price"], 110.0)
        cancel.assert_called_once_with("BTCUSDT", "tp_old")
        args, kwargs = place.call_args
        self.assertEqual(args[:3], ("BTCUSDT", "BUY", 110.0))
        self.assertTrue(kwargs["client_algo_id"].startswith("dcaTP"))

    def test_uses_the_real_exchange_order_not_a_stale_local_id(self):
        manager = self._manager_with_dca_active()
        real_tp = {"type": "TAKE_PROFIT_MARKET", "closePosition": True, "algoId": "real_tp_on_exchange"}

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[real_tp]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}):
            manager._replace_dca_tp_order(manager.positions["BTCUSDT"], 110.0)

        cancel.assert_called_once_with("BTCUSDT", "real_tp_on_exchange")

    def test_ground_truth_check_fails_retries_without_replacing(self):
        manager = self._manager_with_dca_active()

        with patch.object(exchange, "_fetch_open_position_detail", side_effect=RuntimeError("timeout")), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._replace_dca_tp_order(manager.positions["BTCUSDT"], 110.0)

        self.assertIsNone(outcome)
        cancel.assert_not_called()
        place.assert_not_called()

    def test_already_closed_leaves_it_for_the_next_poll(self):
        manager = self._manager_with_dca_active()

        with patch.object(exchange, "_fetch_open_position_detail", return_value=None):
            outcome = manager._replace_dca_tp_order(manager.positions["BTCUSDT"], 110.0)

        self.assertIsNone(outcome)
        self.assertTrue(manager.has_open_position("BTCUSDT"))

    def test_minus_2021_closes_as_a_real_win_not_a_loss(self):
        # Unlike _replace_sl_order's -2021 handling (which defaults to a
        # LOSS-side outcome), the new target already being behind price is
        # GOOD news on the TP side - must close as DCA_TP_HIT, not
        # DCA_SL_HIT/BREAKEVEN_TRIGGER_MARKET_CLOSE.
        manager = self._manager_with_dca_active()

        with patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(
                 exchange, "place_take_profit_full",
                 side_effect=Exception("APIError(code=-2021): Order would immediately trigger."),
             ), \
             patch.object(exchange, "close_position_market") as market_close, \
             patch.object(exchange, "cancel_all_open_orders"):
            outcome = manager._replace_dca_tp_order(manager.positions["BTCUSDT"], 110.0)

        self.assertEqual(outcome, "DCA_TP_HIT")
        market_close.assert_called_once()
        self.assertFalse(manager.has_open_position("BTCUSDT"))


class MigrateDcaTargetIfNeededTests(unittest.TestCase):
    """config.DCA_TP_STATIC_ROI_ENABLED - lets an ALREADY-open DCA_ACTIVE
    position pick up a live config change instead of running out whatever
    target was computed once at DCA-fire time forever."""

    def _manager_with_dca_active(self, shadow=False):
        manager = PositionManager()
        execution_result = {
            "shadow": shadow,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "shadow": shadow, "sl_order_id": "sl_new", "tp_order_id": "tp_old",
            "entry_price": 100.0, "sl_price": 94.0, "tp_price": 106.0, "quantity": 2.0,
            "dca_applied": True,
        })
        return manager

    def test_flag_off_is_a_noop(self):
        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", False), \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._migrate_dca_target_if_needed(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        self.assertEqual(manager.positions["BTCUSDT"]["tp_price"], 106.0)
        place.assert_not_called()

    def test_non_dca_active_stage_is_a_noop(self):
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["stage"] = "TP1_PENDING"

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            outcome = manager._migrate_dca_target_if_needed(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)

    def test_already_matching_target_is_a_noop(self):
        manager = self._manager_with_dca_active()
        manager.positions["BTCUSDT"]["tp_price"] = 105.0  # already the static-ROI target

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._migrate_dca_target_if_needed(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        place.assert_not_called()

    def test_live_position_gets_replaced_via_exchange(self):
        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": "tp_new"}):
            outcome = manager._migrate_dca_target_if_needed(manager.positions["BTCUSDT"])

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertAlmostEqual(position["tp_price"], 105.0)  # 100 * (1 + 0.5/10)
        cancel.assert_called_once_with("BTCUSDT", "tp_old")

    def test_shadow_position_updates_tp_price_directly_without_exchange_calls(self):
        manager = self._manager_with_dca_active(shadow=True)

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(exchange, "place_take_profit_full") as place:
            outcome = manager._migrate_dca_target_if_needed(
                manager.positions["BTCUSDT"], current_price=101.0,
            )

        self.assertIsNone(outcome)
        self.assertAlmostEqual(manager.positions["BTCUSDT"]["tp_price"], 105.0)
        place.assert_not_called()

    def test_shadow_position_closes_immediately_if_price_already_passed_the_new_target(self):
        manager = self._manager_with_dca_active(shadow=True)

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            outcome = manager._migrate_dca_target_if_needed(
                manager.positions["BTCUSDT"], current_price=106.0,  # already past the new 105 target
            )

        self.assertEqual(outcome, "SHADOW_DCA_TP_HIT")
        self.assertFalse(manager.has_open_position("BTCUSDT"))


class PollLiveDcaBreakevenConfirmationTests(unittest.TestCase):
    """config.DCA_BREAKEVEN_CONFIRMATION_ENABLED / ..._WITHHOLD_ENABLED -
    two-phase rollout: the master flag alone only computes+journals the
    verdict onto the position (the breakeven move still applies
    unconditionally underneath, byte-identical to today); WITHHOLD
    additionally lets a confirmed verdict skip the move."""

    def setUp(self):
        # config.DCA_TP_STATIC_ROI_ENABLED - isolated off here too (see
        # PollShadowDcaActiveTests's own note) - these tests are about the
        # breakeven-confirmation mechanism, not the TP target, and must
        # not have the fixture's fixed tp_price self-heal out from under
        # them mid-test.
        for name, value in (
            ("MAE_TRACKING_ENABLED", False),
            ("PROFIT_PROTECTION_ENABLED", False),
            ("DCA_BREAKEVEN_ENABLED", True),
            ("DCA_TP_STATIC_ROI_ENABLED", False),
            ("HTF_TREND_FRESHNESS_ENABLED", True),
            ("EFFICIENCY_RATIO_GATE_ENABLED", True),
            ("SIGNAL_MIN_CVD_SCORE", 0.15),
            ("EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _manager_with_dca_active(self):
        manager = PositionManager()
        execution_result = {
            "shadow": False,
            "tp1_order": {"algoId": "tp1_1"},
            "tp2_order": {"algoId": "tp2_1"},
        }
        manager.register_dca_pending(_dca_plan(), execution_result)
        position = manager.positions["BTCUSDT"]
        position.update({
            "stage": DCA_ACTIVE, "sl_order_id": "sl_new", "tp_order_id": "tp_new",
            "entry_price": 98.0, "sl_price": 94.0, "tp_price": 106.0, "quantity": 2.0,
            "dca_applied": True, "dca_breakeven_applied": False,
            "dca_breakeven_direction_confirmed": None, "breakeven_price": 98.02,
        })
        return manager

    def _confirming_structure_mocks(self):
        # BUY side: BULLISH HTF trend, EMA below current price, efficiency
        # above the chop threshold - all 4 checks pass (cvd_snapshot is
        # supplied per-test via poll_live's own kwarg, not mocked here).
        return (
            patch.object(
                market_structure, "structure_state",
                return_value={"available": True, "trend": "BULLISH"},
            ),
            patch.object(market_structure, "exponential_moving_average", return_value=95.0),
            patch.object(
                market_structure, "analyze",
                return_value={"available": True, "efficiency_ratio": 0.5},
            ),
        )

    def test_master_flag_off_ignores_htf_and_cvd_and_behaves_as_before(self):
        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", False), \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order"), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}):
            outcome = manager.poll_live(
                "BTCUSDT", candles=[{"high": 98.5, "low": 97.5}], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertIsNone(position["dca_breakeven_direction_confirmed"])

    def test_confirmed_but_withhold_disabled_still_applies_breakeven(self):
        s1, s2, s3 = self._confirming_structure_mocks()

        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", False), \
             s1, s2, s3, \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}) as new_sl:
            outcome = manager.poll_live(
                "BTCUSDT", candles=[{"high": 98.5, "low": 97.5}], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertTrue(position["dca_breakeven_direction_confirmed"])
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 98.02)

    def test_confirmed_and_withhold_enabled_skips_the_move(self):
        s1, s2, s3 = self._confirming_structure_mocks()

        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", True), \
             s1, s2, s3, \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "place_stop_loss") as new_sl, \
             patch.object(exchange, "cancel_algo_order") as cancel:
            outcome = manager.poll_live(
                "BTCUSDT", candles=[{"high": 98.5, "low": 97.5}], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertFalse(position["dca_breakeven_applied"])
        self.assertTrue(position["dca_breakeven_direction_confirmed"])
        self.assertEqual(position["sl_price"], 94.0)  # unchanged - real (wide) SL untouched
        new_sl.assert_not_called()
        cancel.assert_not_called()

    def test_not_confirmed_and_withhold_enabled_applies_breakeven_normally(self):
        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", True), \
             patch.object(
                 market_structure, "structure_state",
                 return_value={"available": True, "trend": "BEARISH"},  # disagrees with BUY
             ), \
             patch.object(market_structure, "exponential_moving_average", return_value=95.0), \
             patch.object(
                 market_structure, "analyze",
                 return_value={"available": True, "efficiency_ratio": 0.5},
             ), \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}) as new_sl:
            outcome = manager.poll_live(
                "BTCUSDT", candles=[{"high": 98.5, "low": 97.5}], htf_candles=["htf"],
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertFalse(position["dca_breakeven_direction_confirmed"])
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 98.02)

    def test_missing_htf_candles_fails_safe_and_applies_breakeven(self):
        # No htf_candles supplied (e.g. a symbol main.py's feed doesn't
        # have HTF history for yet) - direction_still_confirmed's own
        # fail-safe kicks in, same real-world effect as "not confirmed".
        manager = self._manager_with_dca_active()

        with patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_ENABLED", True), \
             patch.object(config, "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", True), \
             patch.object(exchange, "get_mark_price", return_value=98.02), \
             patch.object(exchange, "get_algo_order_status", return_value="NEW"), \
             patch.object(exchange, "_fetch_open_position_detail", return_value={"quantity": 2.0}), \
             patch.object(exchange, "get_open_algo_orders", return_value=[]), \
             patch.object(exchange, "cancel_algo_order") as cancel, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": "sl_be"}) as new_sl:
            outcome = manager.poll_live(
                "BTCUSDT", candles=[{"high": 98.5, "low": 97.5}], htf_candles=None,
                cvd_snapshot={"available": True, "cvd_score": 0.5},
            )

        self.assertIsNone(outcome)
        position = manager.positions["BTCUSDT"]
        self.assertTrue(position["dca_breakeven_applied"])
        self.assertFalse(position["dca_breakeven_direction_confirmed"])
        cancel.assert_called_once_with("BTCUSDT", "sl_new")
        new_sl.assert_called_once_with("BTCUSDT", "BUY", 98.02)


if __name__ == "__main__":
    unittest.main()
