import unittest
from unittest.mock import patch

import config
import exchange
import execution


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


if __name__ == "__main__":
    unittest.main()
