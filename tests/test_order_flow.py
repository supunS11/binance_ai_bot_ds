import unittest
from unittest.mock import patch

import config
from order_flow import CVDEngine


class CVDEngineTests(unittest.TestCase):
    def test_buyer_maker_trade_is_signed_negative(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=True, timestamp=1000)

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            snapshot = engine.snapshot("BTCUSDT", now=1001)

        self.assertTrue(snapshot["available"])
        self.assertLess(snapshot["ratio_1m"], 0)

    def test_aggressive_buy_trade_is_signed_positive(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            snapshot = engine.snapshot("BTCUSDT", now=1001)

        self.assertGreater(snapshot["ratio_1m"], 0)

    def test_mixed_trades_produce_proportional_ratio(self):
        engine = CVDEngine()
        # 3x buy notional vs 1x sell notional -> ratio should be +0.5
        engine.record_trade("ETHUSDT", price=100, quantity=3, is_buyer_maker=False, timestamp=1000)
        engine.record_trade("ETHUSDT", price=100, quantity=1, is_buyer_maker=True, timestamp=1000)

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            snapshot = engine.snapshot("ETHUSDT", now=1001)

        self.assertAlmostEqual(snapshot["ratio_1m"], 0.5)

    def test_snapshot_unavailable_with_no_trades(self):
        engine = CVDEngine()
        snapshot = engine.snapshot("DOESNOTEXIST")

        self.assertFalse(snapshot["available"])

    def test_thin_window_below_min_notional_is_excluded(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=1, quantity=1, is_buyer_maker=False, timestamp=1000)

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 1_000_000):
            snapshot = engine.snapshot("BTCUSDT", now=1001)

        self.assertIsNone(snapshot["ratio_1m"])
        self.assertFalse(snapshot["available"])

    def test_old_trades_are_pruned_outside_the_max_window(self):
        engine = CVDEngine()

        with patch.object(config, "ORDER_FLOW_MAX_WINDOW_SECONDS", 60):
            engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
            engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1200)

        with engine.lock:
            series = list(engine._trades["BTCUSDT"])

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0][0], 1200)

    def test_zero_or_negative_price_or_quantity_is_ignored(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=0, quantity=1, is_buyer_maker=False)
        engine.record_trade("BTCUSDT", price=100, quantity=0, is_buyer_maker=False)

        snapshot = engine.snapshot("BTCUSDT")

        self.assertFalse(snapshot["available"])

    def test_reset_clears_a_single_symbol_only(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
        engine.record_trade("ETHUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)

        engine.reset("BTCUSDT")

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            self.assertFalse(engine.snapshot("BTCUSDT", now=1001)["available"])
            self.assertTrue(engine.snapshot("ETHUSDT", now=1001)["available"])


class CVDHistoryTests(unittest.TestCase):
    """Backs config.CVD_DIVERGENCE_TRIGGER_ENABLED - a persistent, per-
    candle-close cumulative CVD series (independent of the recent-window
    trade deque, which is deliberately pruned too aggressively to span
    multiple swings - see ORDER_FLOW_MAX_WINDOW_SECONDS)."""

    def test_finalize_candle_snapshots_the_running_cumulative_total(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=2, is_buyer_maker=False, timestamp=1000)  # +200
        engine.finalize_candle("BTCUSDT", open_time=1000)
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=True, timestamp=1100)  # -100
        engine.finalize_candle("BTCUSDT", open_time=2000)

        history = engine.cvd_history("BTCUSDT")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"open_time": 1000, "cumulative_cvd": 200})
        self.assertEqual(history[1], {"open_time": 2000, "cumulative_cvd": 100})

    def test_cvd_history_is_bounded_by_cvd_history_maxlen(self):
        # 12, not e.g. 3: finalize_candle floors CVD_HISTORY_MAXLEN at 10
        # (same defensive floor CandleStore.__init__ already uses), so a
        # value below that wouldn't actually exercise the real maxlen.
        engine = CVDEngine()

        with patch.object(config, "CVD_HISTORY_MAXLEN", 12):
            for i in range(15):
                engine.finalize_candle("BTCUSDT", open_time=i)

        history = engine.cvd_history("BTCUSDT")

        self.assertEqual(len(history), 12)
        self.assertEqual([point["open_time"] for point in history], list(range(3, 15)))

    def test_cvd_history_for_unknown_symbol_is_empty(self):
        engine = CVDEngine()
        self.assertEqual(engine.cvd_history("NOPE"), [])

    def test_snapshot_includes_history_even_when_unavailable(self):
        engine = CVDEngine()
        engine.finalize_candle("BTCUSDT", open_time=1000)

        snapshot = engine.snapshot("BTCUSDT")

        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["history"], [{"open_time": 1000, "cumulative_cvd": 0.0}])

    def test_snapshot_includes_history_when_available(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
        engine.finalize_candle("BTCUSDT", open_time=1000)

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            snapshot = engine.snapshot("BTCUSDT", now=1001)

        self.assertEqual(snapshot["history"], [{"open_time": 1000, "cumulative_cvd": 100}])

    def test_reset_clears_history_and_cumulative_for_a_single_symbol(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
        engine.finalize_candle("BTCUSDT", open_time=1000)
        engine.record_trade("ETHUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
        engine.finalize_candle("ETHUSDT", open_time=1000)

        engine.reset("BTCUSDT")

        self.assertEqual(engine.cvd_history("BTCUSDT"), [])
        self.assertEqual(len(engine.cvd_history("ETHUSDT")), 1)

    def test_reset_all_clears_every_symbols_history(self):
        engine = CVDEngine()
        engine.record_trade("BTCUSDT", price=100, quantity=1, is_buyer_maker=False, timestamp=1000)
        engine.finalize_candle("BTCUSDT", open_time=1000)

        engine.reset()

        self.assertEqual(engine.cvd_history("BTCUSDT"), [])


if __name__ == "__main__":
    unittest.main()
