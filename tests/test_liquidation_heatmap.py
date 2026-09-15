import unittest
from unittest.mock import patch

import config
import liquidation_heatmap
import liquidation_journal


class ClusterEventsTests(unittest.TestCase):
    def setUp(self):
        self.min_notional_patcher = patch.object(config, "LIQUIDATION_HEATMAP_CLUSTER_MIN_NOTIONAL_USDT", 0)
        self.min_notional_patcher.start()

    def tearDown(self):
        self.min_notional_patcher.stop()

    def test_two_real_sell_side_liquidations_form_a_sell_side_cluster(self):
        events = [("SELL", 100.0, 5000.0), ("SELL", 100.05, 5000.0)]
        pools = liquidation_heatmap.cluster_events(events, tolerance_pct=0.001)

        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["type"], "SELL_SIDE")
        self.assertEqual(pools[0]["touches"], 2)

    def test_two_real_buy_side_liquidations_form_a_buy_side_cluster(self):
        events = [("BUY", 50.0, 5000.0), ("BUY", 50.02, 5000.0)]
        pools = liquidation_heatmap.cluster_events(events, tolerance_pct=0.001)

        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["type"], "BUY_SIDE")

    def test_single_event_never_forms_a_cluster(self):
        events = [("SELL", 100.0, 5000.0)]
        self.assertEqual(liquidation_heatmap.cluster_events(events, tolerance_pct=0.001), [])

    def test_events_far_apart_do_not_cluster(self):
        events = [("SELL", 100.0, 5000.0), ("SELL", 150.0, 5000.0)]
        self.assertEqual(liquidation_heatmap.cluster_events(events, tolerance_pct=0.001), [])

    def test_cluster_price_is_notional_weighted_not_a_plain_mean(self):
        # A $2M forced close should pull the level toward itself much more
        # than a $500 one - a plain mean would land at 100.025.
        events = [("SELL", 100.0, 500.0), ("SELL", 100.05, 2_000_000.0)]
        pools = liquidation_heatmap.cluster_events(events, tolerance_pct=0.001)

        self.assertEqual(len(pools), 1)
        self.assertAlmostEqual(pools[0]["price"], 100.0499875, places=5)
        self.assertGreater(pools[0]["price"], 100.025)  # closer to the $2M event

    def test_cluster_below_min_notional_floor_is_dropped(self):
        events = [("SELL", 100.0, 100.0), ("SELL", 100.05, 100.0)]

        with patch.object(config, "LIQUIDATION_HEATMAP_CLUSTER_MIN_NOTIONAL_USDT", 20000):
            pools = liquidation_heatmap.cluster_events(events, tolerance_pct=0.001)

        self.assertEqual(pools, [])

    def test_buy_and_sell_side_events_cluster_independently(self):
        events = [
            ("SELL", 100.0, 5000.0), ("SELL", 100.05, 5000.0),
            ("BUY", 100.02, 5000.0), ("BUY", 100.06, 5000.0),
        ]
        pools = liquidation_heatmap.cluster_events(events, tolerance_pct=0.001)

        types = sorted(pool["type"] for pool in pools)
        self.assertEqual(types, ["BUY_SIDE", "SELL_SIDE"])


class RecomputeAllTests(unittest.TestCase):
    def setUp(self):
        liquidation_heatmap.reset_cache()
        self.min_notional_patcher = patch.object(config, "LIQUIDATION_HEATMAP_CLUSTER_MIN_NOTIONAL_USDT", 0)
        self.min_notional_patcher.start()

    def tearDown(self):
        liquidation_heatmap.reset_cache()
        self.min_notional_patcher.stop()

    def test_get_liquidation_pools_is_empty_before_any_recompute(self):
        self.assertEqual(liquidation_heatmap.get_liquidation_pools("BTCUSDT"), [])

    def test_recompute_all_populates_the_cache_from_the_journal(self):
        events = [
            {"timestamp": 1000.0, "exchange": "BINANCE", "symbol": "BTCUSDT", "side": "SELL", "price": 100.0, "notional": 5000.0},
            {"timestamp": 1001.0, "exchange": "BYBIT", "symbol": "BTCUSDT", "side": "SELL", "price": 100.05, "notional": 5000.0},
        ]

        with patch.object(liquidation_journal, "load_events", return_value=events):
            liquidation_heatmap.recompute_all(now=2000.0)

        pools = liquidation_heatmap.get_liquidation_pools("BTCUSDT")
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["type"], "SELL_SIDE")

    def test_recompute_all_merges_venues_not_separates_them(self):
        # Same real cluster, split across two exchanges - must still merge
        # into one, matching the "a cascade is invisible on one venue
        # alone" rationale, not stay as two separate 1-touch (sub-floor)
        # candidates.
        events = [
            {"timestamp": 1000.0, "exchange": "BINANCE", "symbol": "BTCUSDT", "side": "SELL", "price": 100.0, "notional": 5000.0},
            {"timestamp": 1001.0, "exchange": "OKX", "symbol": "BTCUSDT", "side": "SELL", "price": 100.02, "notional": 5000.0},
        ]

        with patch.object(liquidation_journal, "load_events", return_value=events):
            liquidation_heatmap.recompute_all(now=2000.0)

        self.assertEqual(len(liquidation_heatmap.get_liquidation_pools("BTCUSDT")), 1)

    def test_recompute_all_never_raises_and_keeps_the_previous_cache_on_failure(self):
        good_events = [
            {"timestamp": 1000.0, "exchange": "BINANCE", "symbol": "BTCUSDT", "side": "SELL", "price": 100.0, "notional": 5000.0},
            {"timestamp": 1001.0, "exchange": "BINANCE", "symbol": "BTCUSDT", "side": "SELL", "price": 100.02, "notional": 5000.0},
        ]
        with patch.object(liquidation_journal, "load_events", return_value=good_events):
            liquidation_heatmap.recompute_all(now=2000.0)

        self.assertEqual(len(liquidation_heatmap.get_liquidation_pools("BTCUSDT")), 1)

        with patch.object(liquidation_journal, "load_events", side_effect=RuntimeError("boom")):
            liquidation_heatmap.recompute_all(now=2000.0)  # must not raise

        self.assertEqual(len(liquidation_heatmap.get_liquidation_pools("BTCUSDT")), 1)


if __name__ == "__main__":
    unittest.main()
