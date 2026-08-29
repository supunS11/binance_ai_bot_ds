import unittest
from unittest.mock import patch

import config
from open_interest import OpenInterestEngine


class OpenInterestEngineTests(unittest.TestCase):
    def test_snapshot_unavailable_with_no_samples(self):
        engine = OpenInterestEngine()
        snapshot = engine.snapshot("DOESNOTEXIST")

        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["oi_change_pct"])

    def test_single_sample_has_no_change_pct_yet(self):
        engine = OpenInterestEngine()
        engine.record("BTCUSDT", 1000, timestamp=1000)

        snapshot = engine.snapshot("BTCUSDT", now=1001)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["oi_value"], 1000)
        self.assertIsNone(snapshot["oi_change_pct"])

    def test_rising_oi_produces_positive_change_pct(self):
        engine = OpenInterestEngine()

        with patch.object(config, "OI_LOOKBACK_SECONDS", 900):
            engine.record("BTCUSDT", 1000, timestamp=1000)
            engine.record("BTCUSDT", 1100, timestamp=1500)

            snapshot = engine.snapshot("BTCUSDT", now=1500)

        self.assertAlmostEqual(snapshot["oi_change_pct"], 10.0)

    def test_falling_oi_produces_negative_change_pct(self):
        engine = OpenInterestEngine()

        with patch.object(config, "OI_LOOKBACK_SECONDS", 900):
            engine.record("BTCUSDT", 1000, timestamp=1000)
            engine.record("BTCUSDT", 900, timestamp=1500)

            snapshot = engine.snapshot("BTCUSDT", now=1500)

        self.assertAlmostEqual(snapshot["oi_change_pct"], -10.0)

    def test_samples_older_than_lookback_are_not_used_as_baseline(self):
        engine = OpenInterestEngine()

        with patch.object(config, "OI_LOOKBACK_SECONDS", 100):
            engine.record("BTCUSDT", 500, timestamp=0)      # far outside lookback
            engine.record("BTCUSDT", 1000, timestamp=1000)  # baseline (in-window)
            engine.record("BTCUSDT", 1100, timestamp=1050)

            snapshot = engine.snapshot("BTCUSDT", now=1050)

        # Baseline should be 1000 (in-window), not 500 (stale) -> +10%, not +120%.
        self.assertAlmostEqual(snapshot["oi_change_pct"], 10.0)

    def test_ignores_none_and_negative_values(self):
        engine = OpenInterestEngine()
        engine.record("BTCUSDT", None, timestamp=1000)
        engine.record("BTCUSDT", -5, timestamp=1001)

        snapshot = engine.snapshot("BTCUSDT")
        self.assertFalse(snapshot["available"])

    def test_reset_clears_a_single_symbol_only(self):
        engine = OpenInterestEngine()
        engine.record("BTCUSDT", 1000, timestamp=1000)
        engine.record("ETHUSDT", 500, timestamp=1000)

        engine.reset("BTCUSDT")

        self.assertFalse(engine.snapshot("BTCUSDT")["available"])
        self.assertTrue(engine.snapshot("ETHUSDT")["available"])


class HistoryTests(unittest.TestCase):
    """Backs OI_DIVERGENCE_TRIGGER_ENABLED (oi_divergence.py), which needs
    OI's value AT specific past timestamps - not just snapshot()'s own
    recent-window change-pct read."""

    def test_history_returns_raw_samples_in_order(self):
        engine = OpenInterestEngine()
        engine.record("BTCUSDT", 1000, timestamp=100)
        engine.record("BTCUSDT", 1100, timestamp=200)

        self.assertEqual(engine.history("BTCUSDT"), [(100, 1000.0), (200, 1100.0)])

    def test_history_for_unknown_symbol_is_empty(self):
        engine = OpenInterestEngine()
        self.assertEqual(engine.history("NOPE"), [])

    def test_history_is_bounded_by_oi_history_max_samples(self):
        engine = OpenInterestEngine()

        with patch.object(config, "OI_HISTORY_MAX_SAMPLES", 3):
            for i in range(5):
                engine.record("BTCUSDT", 1000 + i, timestamp=i)

        self.assertEqual(len(engine.history("BTCUSDT")), 3)
        self.assertEqual([ts for ts, _ in engine.history("BTCUSDT")], [2, 3, 4])

    def test_snapshot_includes_history(self):
        engine = OpenInterestEngine()
        engine.record("BTCUSDT", 1000, timestamp=100)

        snapshot = engine.snapshot("BTCUSDT", now=101)

        self.assertEqual(snapshot["history"], [(100, 1000.0)])

    def test_snapshot_includes_empty_history_when_unavailable(self):
        engine = OpenInterestEngine()
        self.assertEqual(engine.snapshot("DOESNOTEXIST")["history"], [])


class SeedFromHistoryTests(unittest.TestCase):
    """Backs OI_DIVERGENCE_TRIGGER_ENABLED - real incident (2026-08-29):
    0/3,068,258 OI_DIVERGENCE_DIAGNOSTIC log lines ever found matching
    data, because _samples used to start empty on every restart while the
    swing points it's compared against come from REST-seeded, restart-
    surviving candle history. seed_from_history backfills _samples the
    same way at startup, from exchange.get_open_interest_history's
    return shape."""

    def test_seeds_history_from_timestamp_oi_value_tuples(self):
        engine = OpenInterestEngine()

        engine.seed_from_history("BTCUSDT", [(100.0, 1000.0), (200.0, 1100.0)])

        self.assertEqual(engine.history("BTCUSDT"), [(100.0, 1000.0), (200.0, 1100.0)])

    def test_out_of_order_rows_are_sorted_ascending(self):
        """External API response, not this module's own append path -
        _oi_at_or_before's correctness depends on ascending order, so
        this can't just trust the input's given order."""
        engine = OpenInterestEngine()

        engine.seed_from_history("BTCUSDT", [(200.0, 1100.0), (100.0, 1000.0)])

        self.assertEqual(engine.history("BTCUSDT"), [(100.0, 1000.0), (200.0, 1100.0)])

    def test_empty_or_none_history_rows_is_a_no_op(self):
        engine = OpenInterestEngine()

        engine.seed_from_history("BTCUSDT", [])
        engine.seed_from_history("BTCUSDT", None)

        self.assertEqual(engine.history("BTCUSDT"), [])

    def test_seeded_history_is_bounded_by_oi_history_max_samples(self):
        engine = OpenInterestEngine()

        with patch.object(config, "OI_HISTORY_MAX_SAMPLES", 3):
            engine.seed_from_history(
                "BTCUSDT", [(float(i), 1000.0 + i) for i in range(5)]
            )

        self.assertEqual(len(engine.history("BTCUSDT")), 3)
        self.assertEqual([ts for ts, _ in engine.history("BTCUSDT")], [2.0, 3.0, 4.0])

    def test_a_later_live_sample_appends_after_the_seeded_history(self):
        engine = OpenInterestEngine()
        engine.seed_from_history("BTCUSDT", [(100.0, 1000.0)])

        engine.record("BTCUSDT", 1200.0, timestamp=200.0)

        self.assertEqual(
            engine.history("BTCUSDT"), [(100.0, 1000.0), (200.0, 1200.0)]
        )


if __name__ == "__main__":
    unittest.main()
