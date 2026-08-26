import unittest
from unittest.mock import patch

import config
from volume_profile import VolumeProfileEngine


class RecordTradeTests(unittest.TestCase):
    def setUp(self):
        self.engine = VolumeProfileEngine()

    def test_invalid_price_or_quantity_is_ignored(self):
        self.engine.record_trade("BTCUSDT", 0, 1.0, timestamp=1000)
        self.engine.record_trade("BTCUSDT", 100, 0, timestamp=1000)
        self.engine.record_trade("BTCUSDT", "not-a-number", 1.0, timestamp=1000)

        snapshot = self.engine.snapshot("BTCUSDT", now=1000)

        self.assertFalse(snapshot["available"])

    def test_samples_older_than_lookback_are_pruned_on_record(self):
        with patch.object(config, "VOLUME_PROFILE_LOOKBACK_SECONDS", 100):
            self.engine.record_trade("BTCUSDT", 100, 1.0, timestamp=1000)
            self.engine.record_trade("BTCUSDT", 100, 1.0, timestamp=1150)

            series = self.engine._trades["BTCUSDT"]

        self.assertEqual(len(series), 1)

    def test_maxlen_caps_retained_samples(self):
        with patch.object(config, "VOLUME_PROFILE_MAX_SAMPLES", 3):
            for i in range(5):
                self.engine.record_trade("BTCUSDT", 100, 1.0, timestamp=1000 + i)

            series = self.engine._trades["BTCUSDT"]

        self.assertEqual(len(series), 3)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.engine = VolumeProfileEngine()
        for name, value in (
            ("VOLUME_PROFILE_LOOKBACK_SECONDS", 14400),
            ("VOLUME_PROFILE_BUCKET_PCT", 0.05),
            ("VOLUME_PROFILE_VALUE_AREA_PCT", 70.0),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_no_samples_is_unavailable(self):
        snapshot = self.engine.snapshot("BTCUSDT", now=1000)

        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["poc_price"])
        self.assertEqual(snapshot["sample_count"], 0)

    def test_single_dominant_price_level_is_the_poc(self):
        # A heavy concentration of volume at 100, light noise elsewhere.
        for i in range(20):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000 + i)

        self.engine.record_trade("BTCUSDT", 90, 0.1, timestamp=1021)
        self.engine.record_trade("BTCUSDT", 110, 0.1, timestamp=1022)

        snapshot = self.engine.snapshot("BTCUSDT", now=1100)

        self.assertTrue(snapshot["available"])
        self.assertAlmostEqual(snapshot["poc_price"], 100, delta=0.2)
        self.assertEqual(snapshot["sample_count"], 22)

    def test_value_area_bounds_the_poc(self):
        for i in range(20):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000 + i)

        snapshot = self.engine.snapshot("BTCUSDT", now=1100)

        self.assertLessEqual(snapshot["value_area_low"], snapshot["poc_price"])
        self.assertGreaterEqual(snapshot["value_area_high"], snapshot["poc_price"])

    def test_latest_price_inside_value_area(self):
        for i in range(20):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000 + i)

        snapshot = self.engine.snapshot("BTCUSDT", now=1100)

        self.assertEqual(snapshot["position"], "INSIDE_VALUE_AREA")

    def test_latest_price_above_value_area(self):
        for i in range(20):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000 + i)

        # Last trade is a sharp, low-volume spike far above the built-up
        # value area.
        self.engine.record_trade("BTCUSDT", 130, 0.01, timestamp=1021)

        snapshot = self.engine.snapshot("BTCUSDT", now=1100)

        self.assertEqual(snapshot["position"], "ABOVE_VALUE_AREA")

    def test_latest_price_below_value_area(self):
        for i in range(20):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000 + i)

        self.engine.record_trade("BTCUSDT", 70, 0.01, timestamp=1021)

        snapshot = self.engine.snapshot("BTCUSDT", now=1100)

        self.assertEqual(snapshot["position"], "BELOW_VALUE_AREA")

    def test_samples_outside_the_lookback_window_are_excluded(self):
        with patch.object(config, "VOLUME_PROFILE_LOOKBACK_SECONDS", 100):
            self.engine.record_trade("BTCUSDT", 100, 10.0, timestamp=1000)

            snapshot = self.engine.snapshot("BTCUSDT", now=1300)

        self.assertFalse(snapshot["available"])

    def test_symbol_is_case_insensitive(self):
        self.engine.record_trade("btcusdt", 100, 10.0, timestamp=1000)

        snapshot = self.engine.snapshot("BTCUSDT", now=1010)

        self.assertTrue(snapshot["available"])


class ResetTests(unittest.TestCase):
    def test_reset_clears_a_single_symbol(self):
        engine = VolumeProfileEngine()
        engine.record_trade("BTCUSDT", 100, 1.0, timestamp=1000)
        engine.record_trade("ETHUSDT", 100, 1.0, timestamp=1000)

        engine.reset("BTCUSDT")

        self.assertFalse(engine.snapshot("BTCUSDT", now=1000)["available"])
        self.assertTrue(engine.snapshot("ETHUSDT", now=1000)["available"])

    def test_reset_with_no_symbol_clears_everything(self):
        engine = VolumeProfileEngine()
        engine.record_trade("BTCUSDT", 100, 1.0, timestamp=1000)

        engine.reset()

        self.assertFalse(engine.snapshot("BTCUSDT", now=1000)["available"])


if __name__ == "__main__":
    unittest.main()
