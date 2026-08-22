import unittest
from unittest.mock import patch

import config
from crash_detector import CrashDetector


class CrashDetectorTests(unittest.TestCase):
    def _detector(self):
        return CrashDetector()

    def test_empty_detector_is_unavailable_and_inactive(self):
        detector = self._detector()

        snapshot = detector.snapshot(now=1000)

        self.assertFalse(snapshot["available"])
        self.assertFalse(snapshot["active"])
        self.assertIsNone(snapshot["direction"])

    def test_single_sample_is_available_but_never_active(self):
        detector = self._detector()
        detector.record_price(100, timestamp=1000)

        snapshot = detector.snapshot(now=1000)

        self.assertTrue(snapshot["available"])
        self.assertFalse(snapshot["active"])

    def test_drop_past_threshold_activates_bearish(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(98.4, timestamp=1060)  # -1.6%, past threshold

            snapshot = detector.snapshot(now=1060)

        self.assertTrue(snapshot["active"])
        self.assertEqual(snapshot["direction"], "BEARISH")
        self.assertAlmostEqual(snapshot["pct_move"], 1.6, delta=0.01)

    def test_rise_past_threshold_activates_bullish(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(101.6, timestamp=1060)  # +1.6%, past threshold

            snapshot = detector.snapshot(now=1060)

        self.assertTrue(snapshot["active"])
        self.assertEqual(snapshot["direction"], "BULLISH")

    def test_move_under_threshold_stays_inactive(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5):
            detector.record_price(100, timestamp=1000)
            detector.record_price(99.0, timestamp=1060)  # -1.0%, under threshold

            snapshot = detector.snapshot(now=1060)

        self.assertFalse(snapshot["active"])

    def test_samples_outside_the_window_are_pruned(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5):
            # An old high well outside the window shouldn't count toward
            # drawdown once it's aged out.
            detector.record_price(200, timestamp=0)
            detector.record_price(100, timestamp=1000)
            detector.record_price(100.5, timestamp=1060)

            snapshot = detector.snapshot(now=1060)

        self.assertFalse(snapshot["active"])

    def test_stays_active_through_the_cooldown_window(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(98.0, timestamp=1060)

        # Well within the cooldown window, price now flat/no new samples -
        # still active.
        snapshot = detector.snapshot(now=1060 + 300)

        self.assertTrue(snapshot["active"])
        self.assertEqual(snapshot["direction"], "BEARISH")

    def test_clears_once_the_cooldown_expires(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(98.0, timestamp=1060)

        snapshot = detector.snapshot(now=1060 + 601)

        self.assertFalse(snapshot["active"])
        self.assertIsNone(snapshot["direction"])

    def test_repeat_triggers_extend_the_cooldown(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(98.0, timestamp=1060)
            first_active_until = detector.snapshot(now=1060)["active_until"]

            # A later record that still crosses threshold should push the
            # cooldown further out, not leave it pinned to the first trigger.
            detector.record_price(96.0, timestamp=1120)
            second_active_until = detector.snapshot(now=1120)["active_until"]

        self.assertGreater(second_active_until, first_active_until)

    def test_reset_clears_all_state(self):
        detector = self._detector()

        with patch.object(config, "CRASH_DETECTOR_WINDOW_SECONDS", 180), \
             patch.object(config, "CRASH_DETECTOR_MOVE_PCT", 1.5), \
             patch.object(config, "CRASH_DETECTOR_COOLDOWN_SECONDS", 600):
            detector.record_price(100, timestamp=1000)
            detector.record_price(98.0, timestamp=1060)

        detector.reset()
        snapshot = detector.snapshot(now=1060)

        self.assertFalse(snapshot["available"])
        self.assertFalse(snapshot["active"])

    def test_invalid_price_is_ignored(self):
        detector = self._detector()
        detector.record_price(None, timestamp=1000)
        detector.record_price(-5, timestamp=1001)
        detector.record_price("not-a-number", timestamp=1002)

        snapshot = detector.snapshot(now=1002)

        self.assertFalse(snapshot["available"])


if __name__ == "__main__":
    unittest.main()
