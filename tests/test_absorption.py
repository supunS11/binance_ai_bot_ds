import unittest
from unittest.mock import patch

import config
from absorption import compute


def _cvd_snapshot(ratio_1m=None, notional_1m=None):
    return {"ratio_1m": ratio_1m, "notional_1m": notional_1m}


class ComputeTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("ORDER_FLOW_MIN_NOTIONAL_USDT", 5000),
            ("ABSORPTION_MIN_CVD_RATIO", 0.5),
            ("ABSORPTION_MAX_PRICE_MOVE_PCT", 0.05),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_one_sided_selling_that_did_not_move_price_is_bullish_absorption(self):
        # Aggressive selling (ratio negative) that didn't push price down -
        # buyers absorbed it.
        snapshot = _cvd_snapshot(ratio_1m=-0.8, notional_1m=10000)

        result = compute(snapshot, price_change_pct=0.01)

        self.assertEqual(result, "BUY")

    def test_one_sided_buying_that_did_not_move_price_is_bearish_absorption(self):
        snapshot = _cvd_snapshot(ratio_1m=0.8, notional_1m=10000)

        result = compute(snapshot, price_change_pct=-0.01)

        self.assertEqual(result, "SELL")

    def test_ratio_below_min_ratio_is_not_absorption(self):
        snapshot = _cvd_snapshot(ratio_1m=0.3, notional_1m=10000)

        result = compute(snapshot, price_change_pct=0.0)

        self.assertIsNone(result)

    def test_notional_below_floor_is_not_trustworthy(self):
        snapshot = _cvd_snapshot(ratio_1m=0.9, notional_1m=100)

        result = compute(snapshot, price_change_pct=0.0)

        self.assertIsNone(result)

    def test_price_moved_too_much_is_not_absorption(self):
        # One-sided flow that DID move price - the level yielded, not
        # absorbed it.
        snapshot = _cvd_snapshot(ratio_1m=0.9, notional_1m=10000)

        result = compute(snapshot, price_change_pct=0.5)

        self.assertIsNone(result)

    def test_price_moved_the_expected_direction_but_still_too_much(self):
        snapshot = _cvd_snapshot(ratio_1m=-0.9, notional_1m=10000)

        result = compute(snapshot, price_change_pct=-0.5)

        self.assertIsNone(result)

    def test_missing_price_change_pct_returns_none(self):
        snapshot = _cvd_snapshot(ratio_1m=0.9, notional_1m=10000)

        result = compute(snapshot, price_change_pct=None)

        self.assertIsNone(result)

    def test_missing_ratio_or_notional_returns_none(self):
        self.assertIsNone(compute({"ratio_1m": None, "notional_1m": 10000}, 0.0))
        self.assertIsNone(compute({"ratio_1m": 0.9, "notional_1m": None}, 0.0))

    def test_empty_cvd_snapshot_returns_none(self):
        self.assertIsNone(compute({}, 0.0))
        self.assertIsNone(compute(None, 0.0))

    def test_boundary_values_are_inclusive(self):
        # Exactly at both floors/ceilings should still count.
        snapshot = _cvd_snapshot(ratio_1m=-0.5, notional_1m=5000)

        result = compute(snapshot, price_change_pct=0.05)

        self.assertEqual(result, "BUY")


if __name__ == "__main__":
    unittest.main()
