import unittest
from unittest.mock import patch

import config
import market_structure as ms
from oi_divergence import detect_divergence, diagnostic_candidates


def _history(*pairs):
    """pairs: (timestamp_seconds, oi_value)."""
    return list(pairs)


class DetectDivergenceTests(unittest.TestCase):
    """swing.open_time is milliseconds (Binance kline "t"); oi_history
    timestamps are real Unix seconds (time.time()) - open_time/1000 in
    every test below mirrors the conversion detect_divergence itself
    does, so history entries land at exactly the swing's real time."""

    def test_bearish_divergence_price_higher_high_oi_declines(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),  # price made a new high
        ]
        history = _history((1000, 50000), (2000, 40000))  # OI fell

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 5):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 105.0)
        self.assertEqual(result["index"], 1)
        self.assertEqual(result["open_time"], 2_000_000)

    def test_bullish_divergence_price_lower_low_oi_declines(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "LOW"),
            ms.SwingPoint(1, 2_000_000, 95.0, "LOW"),  # price made a new low
        ]
        history = _history((1000, 50000), (2000, 40000))  # OI fell

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 5):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 95.0)
        self.assertEqual(result["index"], 1)

    def test_no_divergence_when_oi_rises_with_price(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 60000))  # OI rose - confirms the breakout

        result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_decline_below_min_delta_pct_does_not_count(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 49000))  # only 2% decline

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 5):
            result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_uses_the_nearest_oi_sample_at_or_before_the_swing_time(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        # No sample exactly at 1000s/2000s - nearest-at-or-before should
        # pick 900->50000 and 1900->40000.
        history = _history((900, 50000), (1900, 40000), (2500, 1000))

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 5):
            result = detect_divergence(swings, history)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")

    def test_swing_before_any_oi_data_existed_is_skipped(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1500, 50000), (2000, 40000))  # nothing at/before 1000s

        result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_only_one_swing_of_a_kind_cannot_diverge(self):
        swings = [ms.SwingPoint(0, 1_000_000, 100.0, "HIGH")]
        history = _history((1000, 50000))

        self.assertIsNone(detect_divergence(swings, history))

    def test_empty_swings_or_history_returns_none(self):
        self.assertIsNone(detect_divergence([], _history((1000, 1))))
        self.assertIsNone(detect_divergence([ms.SwingPoint(0, 1000, 1, "HIGH")], []))

    def test_returns_the_more_recent_divergence_when_both_qualify(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),  # bearish divergence, index 1
            ms.SwingPoint(2, 3_000_000, 90.0, "LOW"),
            ms.SwingPoint(3, 4_000_000, 85.0, "LOW"),    # bullish divergence, index 3 (more recent)
        ]
        history = _history(
            (1000, 50000), (2000, 40000),  # OI fell on the high -> bearish
            (3000, 30000), (4000, 20000),  # OI fell on the low -> bullish
        )

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 5):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["index"], 3)

    def test_min_delta_pct_defaults_from_config_when_not_passed(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 49000))  # 2% decline

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 1):
            self.assertIsNotNone(detect_divergence(swings, history))

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 10):
            self.assertIsNone(detect_divergence(swings, history))

    def test_explicit_min_delta_pct_overrides_config(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 49000))  # 2% decline

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 0):
            self.assertIsNone(detect_divergence(swings, history, min_delta_pct=10))

    def test_zero_or_negative_baseline_oi_is_not_divided_by(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 0), (2000, 0))

        self.assertIsNone(detect_divergence(swings, history))


class DiagnosticCandidatesTests(unittest.TestCase):
    """config.OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED - exposes every
    real structural candidate regardless of outcome."""

    def test_structural_candidate_with_data_and_a_real_delta(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 40000))  # OI fell 20%

        result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["structural_direction"], "BEARISH")
        self.assertTrue(result[0]["oi_data_found"])
        self.assertAlmostEqual(result[0]["delta_pct"], 20.0)

    def test_reports_even_when_below_min_delta_pct(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 50000), (2000, 49000))  # only 2% decline

        with patch.object(config, "OI_DIVERGENCE_MIN_DELTA_PCT", 50):
            self.assertIsNone(detect_divergence(swings, history))
            result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["delta_pct"], 2.0)

    def test_structural_candidate_but_no_oi_data_before_the_swing(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1500, 50000), (2000, 40000))  # nothing at/before 1000s

        result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["oi_data_found"])
        self.assertIsNone(result[0]["delta_pct"])

    def test_zero_baseline_oi_reports_data_not_found(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
        ]
        history = _history((1000, 0), (2000, 0))

        result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["oi_data_found"])
        self.assertIsNone(result[0]["delta_pct"])

    def test_no_structural_candidate_when_price_did_not_make_a_new_extreme(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 95.0, "HIGH"),  # lower high, not new
        ]
        history = _history((1000, 50000), (2000, 40000))

        result = diagnostic_candidates(swings, history)

        self.assertEqual(result, [])

    def test_both_directions_reported_when_both_have_candidates(self):
        swings = [
            ms.SwingPoint(0, 1_000_000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2_000_000, 105.0, "HIGH"),
            ms.SwingPoint(2, 3_000_000, 90.0, "LOW"),
            ms.SwingPoint(3, 4_000_000, 85.0, "LOW"),
        ]
        history = _history((1000, 50000), (2000, 40000), (3000, 30000), (4000, 20000))

        result = diagnostic_candidates(swings, history)
        directions = {c["structural_direction"] for c in result}

        self.assertEqual(len(result), 2)
        self.assertEqual(directions, {"BULLISH", "BEARISH"})

    def test_empty_swings_or_history_returns_empty_list(self):
        self.assertEqual(diagnostic_candidates([], _history((1000, 1))), [])
        self.assertEqual(diagnostic_candidates([ms.SwingPoint(0, 1000, 1, "HIGH")], []), [])

    def test_only_one_swing_of_a_kind_cannot_be_a_candidate(self):
        swings = [ms.SwingPoint(0, 1_000_000, 100.0, "HIGH")]

        result = diagnostic_candidates(swings, _history((1000, 50000)))

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
