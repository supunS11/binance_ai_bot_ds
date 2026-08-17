import unittest
from unittest.mock import patch

import config
from orderbook import DepthImbalanceEngine


class DepthImbalanceEngineTests(unittest.TestCase):
    def test_bid_heavy_book_has_positive_imbalance(self):
        engine = DepthImbalanceEngine()
        engine.record_depth(
            "BTCUSDT",
            bids=[["100", "10"]],
            asks=[["101", "1"]],
            timestamp=1000,
        )

        snapshot = engine.snapshot("BTCUSDT", now=1000)

        self.assertTrue(snapshot["available"])
        self.assertGreater(snapshot["depth_imbalance"], 0)

    def test_ask_heavy_book_has_negative_imbalance(self):
        engine = DepthImbalanceEngine()
        engine.record_depth(
            "BTCUSDT",
            bids=[["100", "1"]],
            asks=[["101", "10"]],
            timestamp=1000,
        )

        snapshot = engine.snapshot("BTCUSDT", now=1000)

        self.assertLess(snapshot["depth_imbalance"], 0)

    def test_balanced_book_is_near_zero_imbalance(self):
        engine = DepthImbalanceEngine()
        engine.record_depth(
            "BTCUSDT",
            bids=[["100", "5"]],
            asks=[["101", "5"]],
            timestamp=1000,
        )

        snapshot = engine.snapshot("BTCUSDT", now=1000)

        self.assertAlmostEqual(snapshot["depth_imbalance"], 0, delta=0.02)

    def test_repeated_samples_are_ema_smoothed_not_overwritten(self):
        engine = DepthImbalanceEngine()
        engine.record_depth("BTCUSDT", bids=[["100", "1"]], asks=[["101", "1"]], timestamp=1000)
        first = engine.snapshot("BTCUSDT", now=1000)["depth_imbalance"]

        engine.record_depth("BTCUSDT", bids=[["100", "100"]], asks=[["101", "1"]], timestamp=1001)
        second = engine.snapshot("BTCUSDT", now=1001)["depth_imbalance"]

        # A single new heavily-skewed sample should move the EMA, but not
        # jump all the way to the new sample's raw imbalance.
        self.assertGreater(second, first)
        self.assertLess(second, 0.99)

    def test_missing_symbol_is_unavailable(self):
        engine = DepthImbalanceEngine()
        snapshot = engine.snapshot("NOPE")

        self.assertFalse(snapshot["available"])

    def test_empty_bids_or_asks_are_ignored(self):
        engine = DepthImbalanceEngine()
        engine.record_depth("BTCUSDT", bids=[], asks=[["101", "1"]], timestamp=1000)

        snapshot = engine.snapshot("BTCUSDT", now=1000)

        self.assertFalse(snapshot["available"])

    def test_stale_data_marked_unavailable_by_watchdog_threshold(self):
        engine = DepthImbalanceEngine()
        engine.record_depth("BTCUSDT", bids=[["100", "1"]], asks=[["101", "1"]], timestamp=1000)

        with patch.object(config, "WS_STALE_SECONDS", 10):
            snapshot = engine.snapshot("BTCUSDT", now=1100)

        self.assertFalse(snapshot["available"])

    def test_reset_clears_a_single_symbol_only(self):
        engine = DepthImbalanceEngine()
        engine.record_depth("BTCUSDT", bids=[["100", "1"]], asks=[["101", "1"]], timestamp=1000)
        engine.record_depth("ETHUSDT", bids=[["100", "1"]], asks=[["101", "1"]], timestamp=1000)

        engine.reset("BTCUSDT")

        self.assertFalse(engine.snapshot("BTCUSDT", now=1000)["available"])
        self.assertTrue(engine.snapshot("ETHUSDT", now=1000)["available"])


if __name__ == "__main__":
    unittest.main()
