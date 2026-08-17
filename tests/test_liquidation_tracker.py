import unittest
from unittest.mock import patch

import config
from liquidation_tracker import LiquidationEngine, _order_payload


class LiquidationEngineTests(unittest.TestCase):
    def test_sell_side_forced_order_counts_as_long_liquidation(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000)

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 300):
            snapshot = engine.snapshot("BTCUSDT", now=1010)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["long_liquidation_notional"], 10000)
        self.assertEqual(snapshot["short_liquidation_notional"], 0)
        self.assertEqual(snapshot["net_liquidation_notional"], 10000)

    def test_buy_side_forced_order_counts_as_short_liquidation(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "BUY", 5000, timestamp=1000)

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 300):
            snapshot = engine.snapshot("BTCUSDT", now=1010)

        self.assertEqual(snapshot["short_liquidation_notional"], 5000)
        self.assertEqual(snapshot["net_liquidation_notional"], -5000)

    def test_events_outside_the_window_are_excluded(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000)

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 60):
            snapshot = engine.snapshot("BTCUSDT", now=2000)

        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["net_liquidation_notional"], 0.0)

    def test_snapshot_unavailable_with_no_events(self):
        engine = LiquidationEngine()
        snapshot = engine.snapshot("DOESNOTEXIST")
        self.assertFalse(snapshot["available"])

    def test_invalid_side_or_non_positive_notional_is_ignored(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "HOLD", 10000, timestamp=1000)
        engine.record_liquidation("BTCUSDT", "SELL", 0, timestamp=1000)
        engine.record_liquidation("BTCUSDT", "SELL", None, timestamp=1000)

        snapshot = engine.snapshot("BTCUSDT", now=1010)
        self.assertFalse(snapshot["available"])

    def test_reset_clears_a_single_symbol_only(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000)
        engine.record_liquidation("ETHUSDT", "SELL", 10000, timestamp=1000)

        engine.reset("BTCUSDT")

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 300):
            self.assertFalse(engine.snapshot("BTCUSDT", now=1010)["available"])
            self.assertTrue(engine.snapshot("ETHUSDT", now=1010)["available"])


class OrderPayloadParsingTests(unittest.TestCase):
    def test_extracts_order_from_combined_stream_wrapper(self):
        message = {"stream": "!forceOrder@arr", "data": {"e": "forceOrder", "o": {"s": "BTCUSDT"}}}
        self.assertEqual(_order_payload(message), {"s": "BTCUSDT"})

    def test_extracts_order_from_unwrapped_message(self):
        message = {"e": "forceOrder", "o": {"s": "BTCUSDT"}}
        self.assertEqual(_order_payload(message), {"s": "BTCUSDT"})

    def test_returns_none_for_malformed_message(self):
        self.assertIsNone(_order_payload("not a dict"))
        self.assertIsNone(_order_payload({"data": {"o": "not a dict"}}))
        self.assertIsNone(_order_payload({}))


class HandleMessageTests(unittest.TestCase):
    def test_handle_message_records_a_liquidation(self):
        engine = LiquidationEngine()
        engine.handle_message({
            "data": {"o": {
                "s": "BTCUSDT", "S": "SELL", "ap": "100", "z": "2", "T": 1000000,
            }}
        })

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 300):
            snapshot = engine.snapshot("BTCUSDT", now=1010)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["long_liquidation_notional"], 200)

    def test_handle_message_never_raises_on_garbage_input(self):
        engine = LiquidationEngine()
        try:
            engine.handle_message(None)
            engine.handle_message({"data": {}})
            engine.handle_message({"data": {"o": {"s": "", "S": "SELL"}}})
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"handle_message raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
