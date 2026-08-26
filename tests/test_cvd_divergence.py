import unittest
from unittest.mock import patch

import config
import market_structure as ms
from cvd_divergence import detect_divergence, diagnostic_candidates


def _history(*pairs):
    return [{"open_time": t, "cumulative_cvd": cvd} for t, cvd in pairs]


class DetectDivergenceTests(unittest.TestCase):
    def test_bullish_divergence_price_lower_low_cvd_higher_low(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),  # price made a new low
        ]
        history = _history((1000, -500), (2000, -100))  # CVD's low rose (less selling)

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 100):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 95.0)
        self.assertEqual(result["index"], 1)
        self.assertEqual(result["open_time"], 2000)

    def test_bearish_divergence_price_higher_high_cvd_lower_high(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "HIGH"),
            ms.SwingPoint(1, 2000, 105.0, "HIGH"),  # price made a new high
        ]
        history = _history((1000, 500), (2000, 100))  # CVD's high fell (less buying)

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 100):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 105.0)
        self.assertEqual(result["index"], 1)

    def test_no_divergence_when_price_and_cvd_agree(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        # CVD also fell (confirms the lower low, no absorption)
        history = _history((1000, -100), (2000, -500))

        result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_delta_below_min_delta_usdt_does_not_count(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500), (2000, -450))  # only $50 of absorption

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 100):
            result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_missing_cvd_sample_for_a_swing_time_is_skipped(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500))  # no sample at open_time 2000

        result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_only_one_swing_of_a_kind_cannot_diverge(self):
        swings = [ms.SwingPoint(0, 1000, 100.0, "LOW")]
        history = _history((1000, -500))

        result = detect_divergence(swings, history)

        self.assertIsNone(result)

    def test_empty_swings_or_history_returns_none(self):
        self.assertIsNone(detect_divergence([], _history((1000, 1))))
        self.assertIsNone(detect_divergence([ms.SwingPoint(0, 1000, 1, "LOW")], []))

    def test_returns_the_more_recent_divergence_when_both_qualify(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),   # bullish divergence, index 1
            ms.SwingPoint(2, 3000, 100.0, "HIGH"),
            ms.SwingPoint(3, 4000, 105.0, "HIGH"),  # bearish divergence, index 3 (more recent)
        ]
        history = _history(
            (1000, -500), (2000, -100),  # CVD rose on the low -> bullish
            (3000, 500), (4000, 100),    # CVD fell on the high -> bearish
        )

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 100):
            result = detect_divergence(swings, history)

        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["index"], 3)

    def test_min_delta_usdt_defaults_from_config_when_not_passed(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500), (2000, -450))  # $50 delta

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 25):
            self.assertIsNotNone(detect_divergence(swings, history))

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 1000):
            self.assertIsNone(detect_divergence(swings, history))

    def test_explicit_min_delta_usdt_overrides_config(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500), (2000, -450))  # $50 delta

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 0):
            self.assertIsNone(detect_divergence(swings, history, min_delta_usdt=1000))


class DiagnosticCandidatesTests(unittest.TestCase):
    """config.CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED - exposes every
    real structural candidate regardless of outcome, so detect_divergence's
    own silence (always None or a confirmed result) can actually be
    explained."""

    def test_structural_candidate_with_data_and_a_real_delta(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500), (2000, -450))  # $50 of absorption

        result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["structural_direction"], "BULLISH")
        self.assertTrue(result[0]["cvd_data_found"])
        self.assertAlmostEqual(result[0]["delta_usdt"], 50)

    def test_reports_even_when_below_min_delta_usdt(self):
        # detect_divergence itself would return None here (too small a
        # delta) - the diagnostic must still report the real candidate and
        # its real delta, that's the entire point.
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500), (2000, -450))

        with patch.object(config, "CVD_DIVERGENCE_MIN_DELTA_USDT", 10000):
            self.assertIsNone(detect_divergence(swings, history))
            result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["delta_usdt"], 50)

    def test_structural_candidate_but_no_cvd_data_at_the_swing_times(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
        ]
        history = _history((1000, -500))  # no sample at open_time 2000

        result = diagnostic_candidates(swings, history)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["cvd_data_found"])
        self.assertIsNone(result[0]["delta_usdt"])

    def test_no_structural_candidate_when_price_did_not_make_a_new_extreme(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 105.0, "LOW"),  # higher low, not a new low
        ]
        history = _history((1000, -500), (2000, -450))

        result = diagnostic_candidates(swings, history)

        self.assertEqual(result, [])

    def test_both_directions_reported_when_both_have_candidates(self):
        swings = [
            ms.SwingPoint(0, 1000, 100.0, "LOW"),
            ms.SwingPoint(1, 2000, 95.0, "LOW"),
            ms.SwingPoint(2, 3000, 100.0, "HIGH"),
            ms.SwingPoint(3, 4000, 105.0, "HIGH"),
        ]
        history = _history((1000, -500), (2000, -450), (3000, 500), (4000, 450))

        result = diagnostic_candidates(swings, history)
        directions = {c["structural_direction"] for c in result}

        self.assertEqual(len(result), 2)
        self.assertEqual(directions, {"BULLISH", "BEARISH"})

    def test_empty_swings_or_history_returns_empty_list(self):
        self.assertEqual(diagnostic_candidates([], _history((1000, 1))), [])
        self.assertEqual(diagnostic_candidates([ms.SwingPoint(0, 1000, 1, "LOW")], []), [])

    def test_only_one_swing_of_a_kind_cannot_be_a_candidate(self):
        swings = [ms.SwingPoint(0, 1000, 100.0, "LOW")]

        result = diagnostic_candidates(swings, _history((1000, -500)))

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
