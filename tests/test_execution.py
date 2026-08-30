import unittest
from unittest.mock import patch

import config
import exchange
import execution
import risk_manager


def _plan():
    return {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "entry_price": 100,
        "sl_price": 98,
        "tp1_price": 102,
        "tp2_price": 104,
        "breakeven_price": 100.02,
        "quantity": 1.0,
        "tp1_quantity": 0.5,
        "tp2_quantity": 0.5,
    }


class EnterTradeShadowModeTests(unittest.TestCase):
    def test_shadow_mode_places_no_real_orders(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss:
            result = execution.enter_trade(_plan())

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        self.assertIsNone(result["entry_order"])
        market_order.assert_not_called()
        stop_loss.assert_not_called()


class EnterTradeLiveModeTests(unittest.TestCase):
    def test_live_mode_places_entry_then_sl_tp1_tp2(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "status": "FILLED", "avgPrice": "100"}) as market_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}) as stop_loss, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp2:
            result = execution.enter_trade(_plan())

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        stop_loss.assert_called_once_with("BTCUSDT", "BUY", 98)
        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.5, 102)
        tp2.assert_called_once_with("BTCUSDT", "BUY", 104)
        self.assertEqual(result["real_entry_price"], 100.0)

    def test_real_entry_price_reflects_actual_slippage_not_the_plan(self):
        # SL/TP1/TP2 stay exactly as planned (real structure levels, not
        # shifted for slippage) - only the reported real_entry_price
        # differs, for position_manager._resolve_real_entry to pick up.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100.12"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp2:
            result = execution.enter_trade(_plan())

        self.assertEqual(result["real_entry_price"], 100.12)
        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.5, 102)  # unshifted
        tp2.assert_called_once_with("BTCUSDT", "BUY", 104)  # unshifted

    def test_live_mode_entry_failure_returns_not_ok(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", side_effect=RuntimeError("boom")):
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])

    def test_leverage_failure_aborts_before_any_entry_attempt(self):
        # Some symbols cap out below config.LEVERAGE - proceeding anyway
        # used to place a doomed entry order and fail a second time with
        # an unrelated-looking error. Must abort cleanly with no entry
        # order attempted at all.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=False), \
             patch.object(exchange, "place_market_order") as market_order:
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("leverage", result["error"])
        market_order.assert_not_called()

    def test_sl_placement_failure_closes_the_just_opened_position(self):
        # A real position now exists on the exchange (entry filled) - if
        # SL can't be attached, it must be closed immediately rather than
        # left both naked and untracked (main.py only registers on ok=True).
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("SL rejected")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full") as tp2:
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("SL placement failed", result["error"])
        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        tp1.assert_not_called()
        tp2.assert_not_called()

    def test_sl_failure_with_4130_cancels_the_stray_conflicting_order(self):
        # Real bug found live (STGUSDT/DEXEUSDT, 2026-08-08): -4130 means a
        # conflicting closePosition stop/TP was already sitting on this
        # symbol before this entry started. Left in place, that same stray
        # order survives the market-close untouched (closePosition orders
        # aren't cancelled just because the position went flat) and blocks
        # every future entry on this symbol with the identical error,
        # forever. Must be cleared as part of this same recovery.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("APIError(code=-4130): An open stop or take profit order with GTE and closePosition in the direction is existing.")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])
        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        cancel_all.assert_called_once_with("BTCUSDT")

    def test_sl_failure_without_4130_does_not_touch_other_orders(self):
        # A plain SL rejection unrelated to a conflicting order has nothing
        # to clean up - calling cancel_all_open_orders here would be a
        # no-op at best and a surprising side effect at worst.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("SL rejected")), \
             patch.object(exchange, "close_position_market") as close_market, \
             patch.object(exchange, "cancel_all_open_orders") as cancel_all:
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])
        close_market.assert_called_once()
        cancel_all.assert_not_called()

    def test_sl_placement_failure_survives_a_failed_close_attempt_too(self):
        # Even the worst case (can't attach SL AND can't close the
        # position) must not raise out of enter_trade - it has to return
        # a normal not-ok result so the caller doesn't crash.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("SL rejected")), \
             patch.object(exchange, "close_position_market", side_effect=RuntimeError("close also failed")):
            result = execution.enter_trade(_plan())

        self.assertFalse(result["ok"])

    def test_tp1_failure_does_not_abort_the_trade(self):
        # SL is already attached at this point - the position is safe.
        # A TP1 failure is degraded, not dangerous, so the trade must
        # still be reported ok=True and get tracked.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}), \
             patch.object(exchange, "place_take_profit_partial", side_effect=RuntimeError("TP1 rejected")), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            result = execution.enter_trade(_plan())

        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["sl_order"])
        self.assertIsNone(result["tp1_order"])
        self.assertIsNotNone(result["tp2_order"])

    def test_tp2_failure_does_not_abort_the_trade(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", side_effect=RuntimeError("TP2 rejected")):
            result = execution.enter_trade(_plan())

        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["sl_order"])
        self.assertIsNotNone(result["tp1_order"])
        self.assertIsNone(result["tp2_order"])


class EnterTradeShadowOnlyTriggerTests(unittest.TestCase):
    """config.SHADOW_ONLY_TRIGGERS - forces an individual trigger into
    shadow while EXECUTION_MODE stays LIVE (real motivation, 2026-08-29:
    CVD_DIVERGENCE went from zero live track record to 60% of trade
    volume overnight after a data-seeding bug fix - forced into shadow for
    further evaluation without risking more capital)."""

    def test_matching_trigger_forces_shadow_even_though_mode_is_live(self):
        plan = dict(_plan(), signal_trigger="CVD_DIVERGENCE")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "place_market_order") as market_order:
            result = execution.enter_trade(plan)

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        market_order.assert_not_called()

    def test_non_matching_trigger_is_unaffected_and_trades_live(self):
        plan = dict(_plan(), signal_trigger="STRUCTURE_BREAK")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "status": "FILLED", "avgPrice": "100"}) as market_order, \
             patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            result = execution.enter_trade(plan)

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        market_order.assert_called_once()


class EnterTradeLimitShadowModeTests(unittest.TestCase):
    def test_shadow_mode_places_no_real_orders(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_limit(_plan())

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        self.assertIsNone(result["entry_order"])
        limit_order.assert_not_called()


class EnterTradeLimitLiveModeTests(unittest.TestCase):
    """config.LIMIT_ENTRY_MODE_ENABLED - structurally different from
    enter_trade's LIVE path: nothing is filled yet at placement time, so
    no SL/TP1/TP2 must ever be placed here (position_manager.poll_pending_entry
    is where that happens, once a real fill is detected)."""

    def test_live_mode_places_only_the_limit_entry_order(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}) as limit_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss, \
             patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full") as tp2:
            result = execution.enter_trade_limit(_plan())

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        self.assertIsNotNone(result["entry_order"])
        limit_order.assert_called_once_with("BTCUSDT", "BUY", 1.0, 100)
        stop_loss.assert_not_called()
        tp1.assert_not_called()
        tp2.assert_not_called()

    def test_leverage_failure_aborts_before_any_entry_attempt(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=False), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_limit(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("leverage", result["error"])
        limit_order.assert_not_called()

    def test_entry_order_failure_returns_not_ok(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", side_effect=RuntimeError("boom")):
            result = execution.enter_trade_limit(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])


class EnterTradeLimitShadowOnlyTriggerTests(unittest.TestCase):
    def test_matching_trigger_forces_shadow_even_though_mode_is_live(self):
        plan = dict(_plan(), signal_trigger="CVD_DIVERGENCE")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_limit(plan)

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        limit_order.assert_not_called()

    def test_non_matching_trigger_is_unaffected_and_trades_live(self):
        plan = dict(_plan(), signal_trigger="STRUCTURE_BREAK")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}) as limit_order:
            result = execution.enter_trade_limit(plan)

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        limit_order.assert_called_once()


class EnterTradeRetracementShadowModeTests(unittest.TestCase):
    def test_shadow_mode_places_no_real_orders(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_retracement(_plan())

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        self.assertIsNone(result["entry_order"])
        self.assertAlmostEqual(result["retracement_price"], 99.8)  # 100 - 0.1*(100-98)
        limit_order.assert_not_called()


class EnterTradeRetracementLiveModeTests(unittest.TestCase):
    """config.RETRACEMENT_ENTRY_ENABLED - structurally identical to
    enter_trade_limit's LIVE path (nothing is filled yet, no SL/TP1/TP2
    here - position_manager.poll_retracement_pending resolves the fill
    or its bounded market fallback later), just resting at a computed
    retracement price instead of the raw plan["entry_price"]."""

    def test_live_mode_places_only_the_retracement_limit_order(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}) as limit_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss, \
             patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full") as tp2:
            result = execution.enter_trade_retracement(_plan())

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        self.assertIsNotNone(result["entry_order"])
        self.assertAlmostEqual(result["retracement_price"], 99.8)
        limit_order.assert_called_once_with("BTCUSDT", "BUY", 1.0, 99.8)
        stop_loss.assert_not_called()
        tp1.assert_not_called()
        tp2.assert_not_called()

    def test_leverage_failure_aborts_before_any_entry_attempt(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=False), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_retracement(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("leverage", result["error"])
        limit_order.assert_not_called()

    def test_entry_order_failure_returns_not_ok(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", side_effect=RuntimeError("boom")):
            result = execution.enter_trade_retracement(_plan())

        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])

    def test_forwards_fair_value_gaps_and_liquidity_pools_from_the_plan(self):
        # config.RETRACEMENT_STRUCTURE_TARGET_ENABLED - risk_manager.
        # compute_retracement_price needs these to consider a real
        # structural level; this is the only call site that ever supplies
        # them for a real order.
        plan = _plan()
        plan["fair_value_gaps"] = [{"top": 99, "bottom": 98.5}]
        plan["liquidity_pools"] = [{"type": "SELL_SIDE", "price": 99.2, "touches": 2}]

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}), \
             patch.object(
                 risk_manager, "compute_retracement_price", return_value=99.2,
             ) as compute_price:
            execution.enter_trade_retracement(plan)

        compute_price.assert_called_once_with(
            100, 98, "BUY",
            fvgs=plan["fair_value_gaps"], pools=plan["liquidity_pools"],
            prefer_deeper=False,
        )


class EnterTradeRetracementDepthAwareTests(unittest.TestCase):
    """config.RETRACEMENT_DEPTH_AWARE_ENABLED - a weak depth_imbalance at
    entry routes to a deeper structural level (risk_manager.compute_
    retracement_price's own prefer_deeper) and a longer resting timeout
    instead of today's shallow/base-timeout default."""

    def test_weak_depth_imbalance_routes_deeper_with_longer_timeout(self):
        plan = dict(_plan(), depth_imbalance=0.10)  # signed BUY: 0.10 < 0.30

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_ENABLED", True), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_MIN_IMBALANCE", 0.30), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS", 600), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}), \
             patch.object(risk_manager, "compute_retracement_price", return_value=99.0) as compute_price:
            result = execution.enter_trade_retracement(plan)

        compute_price.assert_called_once_with(
            100, 98, "BUY", fvgs=None, pools=None, prefer_deeper=True,
        )
        self.assertEqual(result["retracement_timeout_seconds"], 600)
        self.assertTrue(result["used_deep_retracement"])

    def test_strong_depth_imbalance_stays_shallow(self):
        plan = dict(_plan(), depth_imbalance=0.40)  # signed BUY: 0.40 >= 0.30

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_ENABLED", True), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_MIN_IMBALANCE", 0.30), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS", 600), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}), \
             patch.object(risk_manager, "compute_retracement_price", return_value=99.8) as compute_price:
            result = execution.enter_trade_retracement(plan)

        compute_price.assert_called_once_with(
            100, 98, "BUY", fvgs=None, pools=None, prefer_deeper=False,
        )
        self.assertEqual(result["retracement_timeout_seconds"], 300)
        self.assertFalse(result["used_deep_retracement"])

    def test_missing_depth_imbalance_fails_open_to_shallow(self):
        # No depth_imbalance key at all - unavailable data must never
        # independently trigger different order placement.
        plan = _plan()

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_ENABLED", True), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS", 600), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}), \
             patch.object(risk_manager, "compute_retracement_price", return_value=99.8) as compute_price:
            result = execution.enter_trade_retracement(plan)

        compute_price.assert_called_once_with(
            100, 98, "BUY", fvgs=None, pools=None, prefer_deeper=False,
        )
        self.assertEqual(result["retracement_timeout_seconds"], 300)
        self.assertFalse(result["used_deep_retracement"])

    def test_flag_disabled_ignores_depth_imbalance_entirely(self):
        plan = dict(_plan(), depth_imbalance=0.10)  # would be "weak" if the flag were on

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_ENABLED", False), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS", 600), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}), \
             patch.object(risk_manager, "compute_retracement_price", return_value=99.8) as compute_price:
            result = execution.enter_trade_retracement(plan)

        compute_price.assert_called_once_with(
            100, 98, "BUY", fvgs=None, pools=None, prefer_deeper=False,
        )
        self.assertEqual(result["retracement_timeout_seconds"], 300)
        self.assertFalse(result["used_deep_retracement"])

    def test_shadow_mode_also_carries_the_deep_timeout(self):
        # Real requirement: shadow trades are exactly what this mechanism
        # needs to be evaluated on before it's ever trusted live - if the
        # shadow branch silently reverted to the base timeout, it could
        # never be evidence-checked pre-live.
        plan = dict(_plan(), depth_imbalance=0.10)

        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_ENABLED", True), \
             patch.object(config, "RETRACEMENT_DEPTH_AWARE_MIN_IMBALANCE", 0.30), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300), \
             patch.object(config, "RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS", 600), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_retracement(plan)

        self.assertTrue(result["shadow"])
        self.assertEqual(result["retracement_timeout_seconds"], 600)
        self.assertTrue(result["used_deep_retracement"])
        limit_order.assert_not_called()


class EnterTradeRetracementShadowOnlyTriggerTests(unittest.TestCase):
    def test_matching_trigger_forces_shadow_even_though_mode_is_live(self):
        plan = dict(_plan(), signal_trigger="CVD_DIVERGENCE")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(exchange, "place_limit_order") as limit_order:
            result = execution.enter_trade_retracement(plan)

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        limit_order.assert_not_called()

    def test_non_matching_trigger_is_unaffected_and_trades_live(self):
        plan = dict(_plan(), signal_trigger="STRUCTURE_BREAK")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 1, "status": "NEW"}) as limit_order:
            result = execution.enter_trade_retracement(plan)

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        limit_order.assert_called_once()


def _dca_plan():
    return dict(_plan(), dca_price=96)


def _dca_single_tp_plan():
    return dict(
        _plan(), dca_price=96, single_tp=True, tp_price=104,
        tp1_price=None, tp2_price=None, tp1_quantity=None, tp2_quantity=None,
    )


class EnterTradeDcaPendingShadowModeTests(unittest.TestCase):
    def test_shadow_mode_places_no_real_orders(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(exchange, "place_market_order") as market_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss:
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        self.assertIsNone(result["entry_order"])
        market_order.assert_not_called()
        stop_loss.assert_not_called()


class EnterTradeDcaPendingLiveModeTests(unittest.TestCase):
    def test_live_mode_places_entry_then_tp1_tp2_but_never_an_sl(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}) as market_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp2:
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        self.assertNotIn("sl_order", result)
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.5, 102)
        tp2.assert_called_once_with("BTCUSDT", "BUY", 104)
        stop_loss.assert_not_called()
        self.assertEqual(result["real_entry_price"], 100.0)

    def test_real_entry_price_reflects_actual_slippage_not_the_plan(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "99.95"}), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertEqual(result["real_entry_price"], 99.95)

    def test_entry_order_failure_returns_not_ok(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", side_effect=RuntimeError("boom")):
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])

    def test_leverage_failure_aborts_before_any_entry_attempt(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=False), \
             patch.object(exchange, "place_market_order") as market_order:
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertFalse(result["ok"])
        self.assertIn("leverage", result["error"])
        market_order.assert_not_called()

    def test_tp1_placement_failure_does_not_abort_the_trade(self):
        # No SL exists here to make this dangerous the way enter_trade's
        # own SL failure is - a missing TP is just a degraded outcome,
        # same "best-effort" treatment enter_trade already gives TP1/TP2.
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_take_profit_partial", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            result = execution.enter_trade_dca_pending(_dca_plan())

        self.assertTrue(result["ok"])
        self.assertIsNone(result["tp1_order"])


class EnterTradeDcaPendingShadowOnlyTriggerTests(unittest.TestCase):
    def test_matching_trigger_forces_shadow_even_though_mode_is_live(self):
        plan = dict(_dca_plan(), signal_trigger="CVD_DIVERGENCE")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "place_market_order") as market_order:
            result = execution.enter_trade_dca_pending(plan)

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        market_order.assert_not_called()

    def test_non_matching_trigger_is_unaffected_and_trades_live(self):
        plan = dict(_dca_plan(), signal_trigger="STRUCTURE_BREAK")

        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(config, "SHADOW_ONLY_TRIGGERS", ["CVD_DIVERGENCE"]), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}) as market_order, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            result = execution.enter_trade_dca_pending(plan)

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        market_order.assert_called_once()


class EnterTradeDcaPendingSingleTpTests(unittest.TestCase):
    """config.TP_STATIC_ROI_ENABLED - plan["single_tp"] routes to ONE
    full-position TP order instead of TP1(partial)+TP2(remainder)."""

    def test_shadow_mode_places_no_real_orders(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
             patch.object(exchange, "place_market_order") as market_order:
            result = execution.enter_trade_dca_pending(_dca_single_tp_plan())

        self.assertTrue(result["ok"])
        self.assertTrue(result["shadow"])
        market_order.assert_not_called()

    def test_live_mode_places_entry_then_a_single_tp_but_never_tp1_or_sl(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}) as market_order, \
             patch.object(exchange, "place_stop_loss") as stop_loss, \
             patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp_full:
            result = execution.enter_trade_dca_pending(_dca_single_tp_plan())

        self.assertTrue(result["ok"])
        self.assertFalse(result["shadow"])
        market_order.assert_called_once_with("BTCUSDT", "BUY", 1.0)
        tp_full.assert_called_once_with("BTCUSDT", "BUY", 104)
        tp1.assert_not_called()
        stop_loss.assert_not_called()
        self.assertIsNone(result["tp1_order"])
        self.assertIsNone(result["tp2_order"])
        self.assertEqual(result["tp_order"], {"algoId": 4})
        self.assertEqual(result["real_entry_price"], 100.0)

    def test_tp_placement_failure_does_not_abort_the_trade(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"), \
             patch.object(exchange, "setup_leverage", return_value=True), \
             patch.object(exchange, "place_market_order", return_value={"orderId": 1, "avgPrice": "100"}), \
             patch.object(exchange, "place_take_profit_full", side_effect=RuntimeError("rejected")):
            result = execution.enter_trade_dca_pending(_dca_single_tp_plan())

        self.assertTrue(result["ok"])
        self.assertIsNone(result["tp_order"])


class PlaceProtectionOrdersTests(unittest.TestCase):
    """SL+TP1+TP2 placement, extracted out of enter_trade's tail so
    config.RETRACEMENT_ENTRY_ENABLED's settle path (position_manager.
    _finalize_retracement_entry) can place the exact same protection for
    a fill that already happened, without a second entry order. enter_trade
    itself already exercises this end-to-end (EnterTradeLiveModeTests) -
    these cover it directly, including a settled quantity that differs
    from the plan's own (the whole reason this needed extracting)."""

    def test_places_sl_then_tp1_tp2(self):
        with patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}) as sl, \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp2:
            sl_order, tp1_order, tp2_order, error = execution.place_protection_orders(
                "BTCUSDT", "BUY", _plan()
            )

        self.assertIsNone(error)
        sl.assert_called_once_with("BTCUSDT", "BUY", 98)
        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.5, 102)
        tp2.assert_called_once_with("BTCUSDT", "BUY", 104)
        self.assertEqual(sl_order, {"algoId": 2})
        self.assertEqual(tp1_order, {"algoId": 3})
        self.assertEqual(tp2_order, {"algoId": 4})

    def test_uses_whatever_quantity_the_plan_carries_not_a_fixed_one(self):
        # The real motivation for extracting this at all: a retracement
        # settle's plan can carry a real fill quantity that differs from
        # the originally-planned one (a blended limit+market-fallback
        # fill) - this must size the SL-failure market-close off THAT
        # quantity, not silently reuse whatever the original plan said.
        settled = dict(_plan(), quantity=1.7)

        with patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market") as close_market:
            execution.place_protection_orders("BTCUSDT", "BUY", settled)

        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.7)

    def test_sl_failure_closes_at_market_and_returns_an_error(self):
        with patch.object(exchange, "place_stop_loss", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "close_position_market") as close_market:
            sl_order, tp1_order, tp2_order, error = execution.place_protection_orders(
                "BTCUSDT", "BUY", _plan()
            )

        self.assertIsNone(sl_order)
        self.assertIsNone(tp1_order)
        self.assertIsNone(tp2_order)
        self.assertIn("rejected", error)
        close_market.assert_called_once_with("BTCUSDT", "BUY", 1.0)

    def test_tp1_failure_does_not_abort(self):
        with patch.object(exchange, "place_stop_loss", return_value={"algoId": 2}), \
             patch.object(exchange, "place_take_profit_partial", side_effect=RuntimeError("rejected")), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}):
            sl_order, tp1_order, tp2_order, error = execution.place_protection_orders(
                "BTCUSDT", "BUY", _plan()
            )

        self.assertIsNone(error)
        self.assertIsNone(tp1_order)
        self.assertEqual(tp2_order, {"algoId": 4})


class PlaceDcaProtectionOrdersTests(unittest.TestCase):
    """TP1+TP2 or a single TP, no SL - extracted out of enter_trade_dca_
    pending's tail for the same reason as place_protection_orders above."""

    def test_dual_tp_plan_places_tp1_and_tp2(self):
        with patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}) as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp2:
            tp1_order, tp2_order, tp_order, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_plan()
            )

        tp1.assert_called_once_with("BTCUSDT", "BUY", 0.5, 102)
        tp2.assert_called_once_with("BTCUSDT", "BUY", 104)
        self.assertEqual(tp1_order, {"algoId": 3})
        self.assertEqual(tp2_order, {"algoId": 4})
        self.assertIsNone(tp_order)
        # config.DCA_RESTING_ORDER_ENABLED defaults False - no resting
        # order placed unless a test explicitly opts in (see
        # PlaceDcaRestingOrderTests below).
        self.assertIsNone(dca_order)

    def test_single_tp_plan_places_only_the_full_tp(self):
        with patch.object(exchange, "place_take_profit_partial") as tp1, \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}) as tp_full:
            tp1_order, tp2_order, tp_order, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_single_tp_plan()
            )

        tp1.assert_not_called()
        tp_full.assert_called_once_with("BTCUSDT", "BUY", 104)
        self.assertIsNone(tp1_order)
        self.assertIsNone(tp2_order)
        self.assertEqual(tp_order, {"algoId": 4})
        self.assertIsNone(dca_order)

    def test_tp_failure_does_not_raise(self):
        with patch.object(exchange, "place_take_profit_full", side_effect=RuntimeError("rejected")):
            tp1_order, tp2_order, tp_order, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_single_tp_plan()
            )

        self.assertIsNone(tp_order)


class PlaceDcaRestingOrderTests(unittest.TestCase):
    """config.DCA_RESTING_ORDER_ENABLED - the resting DCA-add LIMIT order
    place_dca_protection_orders places alongside TP1/TP2 (or the single
    TP), always sized at DCA_PRESSURE_SIZE_MULTIPLIER (the conservative
    branch), tagged for restart recovery. See config.py's own comment for
    the full evidence/rationale."""

    def test_disabled_by_default_places_no_resting_order(self):
        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", False), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}), \
             patch.object(exchange, "place_limit_order") as limit_order:
            _, _, _, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_plan()
            )

        limit_order.assert_not_called()
        self.assertIsNone(dca_order)

    def test_enabled_places_resting_order_at_conservative_size_and_dca_price(self):
        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(config, "DCA_PRESSURE_SIZE_MULTIPLIER", 0.5), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 9}) as limit_order:
            _, _, _, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_plan()
            )

        limit_order.assert_called_once()
        args, kwargs = limit_order.call_args
        self.assertEqual(args[0], "BTCUSDT")
        self.assertEqual(args[1], "BUY")
        self.assertAlmostEqual(args[2], 0.5)  # 1.0 (quantity) * 0.5 multiplier
        self.assertEqual(args[3], 96)  # dca_price
        self.assertTrue(
            kwargs["client_order_id"].startswith(execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX)
        )
        self.assertEqual(dca_order, {"orderId": 9})

    def test_enabled_works_for_single_tp_plan_too(self):
        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}), \
             patch.object(exchange, "place_limit_order", return_value={"orderId": 9}) as limit_order:
            _, _, _, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_single_tp_plan()
            )

        limit_order.assert_called_once()
        self.assertEqual(dca_order, {"orderId": 9})

    def test_placement_failure_is_best_effort_not_raised(self):
        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}), \
             patch.object(exchange, "place_limit_order", side_effect=RuntimeError("rejected")):
            tp1_order, tp2_order, tp_order, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_plan()
            )

        # TP1/TP2 still placed - a failed resting order is degraded, not
        # a naked-position risk (same treatment as a failed TP).
        self.assertEqual(tp1_order, {"algoId": 3})
        self.assertEqual(tp2_order, {"algoId": 4})
        self.assertIsNone(dca_order)

    def test_zero_quantity_after_multiplier_places_nothing(self):
        with patch.object(config, "DCA_RESTING_ORDER_ENABLED", True), \
             patch.object(config, "DCA_PRESSURE_SIZE_MULTIPLIER", 0), \
             patch.object(exchange, "place_take_profit_partial", return_value={"algoId": 3}), \
             patch.object(exchange, "place_take_profit_full", return_value={"algoId": 4}), \
             patch.object(exchange, "place_limit_order") as limit_order:
            _, _, _, dca_order = execution.place_dca_protection_orders(
                "BTCUSDT", "BUY", _dca_plan()
            )

        limit_order.assert_not_called()
        self.assertIsNone(dca_order)


if __name__ == "__main__":
    unittest.main()
