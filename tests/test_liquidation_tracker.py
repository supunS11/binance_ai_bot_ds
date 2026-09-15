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

    # config.LIQUIDATION_HEATMAP_ENABLED (2026-09-15) - price is now
    # retained alongside the existing fields, purely for liquidation_
    # heatmap.py's historical clustering; this engine's own real-time
    # snapshot() must stay byte-identical regardless.

    def test_record_liquidation_stores_price(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000, price=50000)

        self.assertEqual(list(engine._events["BTCUSDT"])[0], (1000.0, "SELL", 10000.0, 50000.0))

    def test_record_liquidation_without_price_defaults_to_none(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000)

        self.assertEqual(list(engine._events["BTCUSDT"])[0], (1000.0, "SELL", 10000.0, None))

    def test_record_liquidation_returns_true_when_accepted(self):
        engine = LiquidationEngine()
        self.assertTrue(engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000))

    def test_record_liquidation_returns_false_when_invalid(self):
        engine = LiquidationEngine()
        self.assertFalse(engine.record_liquidation("BTCUSDT", "HOLD", 10000, timestamp=1000))
        self.assertFalse(engine.record_liquidation("BTCUSDT", "SELL", 0, timestamp=1000))

    def test_snapshot_output_unchanged_by_the_new_price_field(self):
        engine = LiquidationEngine()
        engine.record_liquidation("BTCUSDT", "SELL", 10000, timestamp=1000, price=50000)

        with patch.object(config, "LIQUIDATION_WINDOW_SECONDS", 300):
            snapshot = engine.snapshot("BTCUSDT", now=1010)

        self.assertEqual(
            snapshot,
            {
                "available": True, "symbol": "BTCUSDT", "sample_count": 1,
                "long_liquidation_notional": 10000, "short_liquidation_notional": 0,
                "net_liquidation_notional": 10000,
            },
        )


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
            self.assertIsNone(engine.handle_message(None))
            self.assertIsNone(engine.handle_message({"data": {}}))
            self.assertIsNone(engine.handle_message({"data": {"o": {"s": "", "S": "SELL"}}}))
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"handle_message raised unexpectedly: {exc}")

    def test_handle_message_returns_the_parsed_tuple_with_price(self):
        engine = LiquidationEngine()
        result = engine.handle_message({
            "data": {"o": {
                "s": "BTCUSDT", "S": "SELL", "ap": "100", "z": "2", "T": 1000000,
            }}
        })

        self.assertEqual(result, ("BTCUSDT", "SELL", 200.0, 1000.0, 100.0))

    def test_handle_message_returns_none_when_the_event_is_dropped(self):
        # A real symbol/side but non-positive notional (price*quantity=0) -
        # record_liquidation itself rejects it, so nothing should be
        # journaled for it either.
        engine = LiquidationEngine()
        result = engine.handle_message({
            "data": {"o": {"s": "BTCUSDT", "S": "SELL", "ap": "0", "z": "2", "T": 1000000}}
        })

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
