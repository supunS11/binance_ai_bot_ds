import unittest
from unittest.mock import patch

import pandas as pd

import config
from backtest_feed import BacktestFeed


class BacktestFeedCandleTests(unittest.TestCase):
    def test_seed_ltf_marks_candles_closed(self):
        feed = BacktestFeed()
        df = pd.DataFrame([
            {"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
            {"time": 2000, "open": 1.5, "high": 2.5, "low": 1, "close": 2, "volume": 12},
        ])

        feed.seed_ltf("BTCUSDT", df)
        candles = feed.candles.get("BTCUSDT")

        self.assertEqual(len(candles), 2)
        self.assertTrue(all(c["closed"] for c in candles))
        self.assertEqual(candles[-1]["close"], 2)

    def test_seed_htf_is_independent_of_ltf(self):
        feed = BacktestFeed()
        ltf_df = pd.DataFrame([{"time": 1000, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}])
        htf_df = pd.DataFrame([{"time": 0, "open": 5, "high": 5, "low": 5, "close": 5, "volume": 5}])

        feed.seed_ltf("BTCUSDT", ltf_df)
        feed.seed_htf("BTCUSDT", htf_df)

        self.assertEqual(len(feed.candles.get("BTCUSDT")), 1)
        self.assertEqual(len(feed.htf_candles.get("BTCUSDT")), 1)
        self.assertEqual(feed.htf_candles.get("BTCUSDT")[0]["close"], 5)

    def test_push_ltf_candle_advances_the_rolling_window(self):
        feed = BacktestFeed(ltf_history_limit=10)

        feed.push_ltf_candle("BTCUSDT", {
            "open_time": 1000, "open": 1, "high": 1, "low": 1, "close": 1,
            "volume": 1, "closed": True,
        })
        feed.push_ltf_candle("BTCUSDT", {
            "open_time": 2000, "open": 1, "high": 1, "low": 1, "close": 2,
            "volume": 1, "closed": True,
        })

        candles = feed.candles.get("BTCUSDT")
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[-1]["close"], 2)
        self.assertEqual(feed.candles.latest("BTCUSDT")["open_time"], 2000)

    def test_push_htf_candle_updates_htf_store_only(self):
        feed = BacktestFeed()

        feed.push_htf_candle("BTCUSDT", {
            "open_time": 0, "open": 1, "high": 1, "low": 1, "close": 1,
            "volume": 1, "closed": True,
        })

        self.assertEqual(len(feed.htf_candles.get("BTCUSDT")), 1)
        self.assertEqual(feed.candles.get("BTCUSDT"), [])


class BacktestFeedCvdTests(unittest.TestCase):
    def test_replay_trades_feeds_the_real_cvd_engine(self):
        feed = BacktestFeed()
        # Buyer-maker=False (aggressor bought) -> positive/buy pressure;
        # matches order_flow.CVDEngine.record_trade's own documented sign
        # convention (see order_flow.py). Notional pinned well above
        # ORDER_FLOW_MIN_NOTIONAL_USDT explicitly (rather than relying on
        # whatever the live .env happens to have it set to) so this test
        # is deterministic regardless of ambient config.
        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 100):
            feed.replay_trades("BTCUSDT", [
                (100.0, 10_000.0, 1.0, False),
                (101.0, 10_000.0, 1.0, False),
            ])

            snapshot = feed.cvd.snapshot("BTCUSDT", now=102.0)

        self.assertTrue(snapshot["available"])
        self.assertGreater(snapshot["cvd_score"], 0)

    def test_replay_trades_is_chronological_order_dependent(self):
        feed = BacktestFeed()

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 100):
            feed.replay_trades("BTCUSDT", [
                (100.0, 10_000.0, 5.0, True),   # sell pressure
                (100.5, 10_000.0, 5.0, False),  # buy pressure, same size -> net ~0
            ])

            snapshot = feed.cvd.snapshot("BTCUSDT", now=101.0)

        self.assertTrue(snapshot["available"])
        self.assertAlmostEqual(snapshot["cvd_score"], 0.0, places=4)


class BacktestFeedStubSourceTests(unittest.TestCase):
    def test_depth_open_interest_liquidations_report_unavailable(self):
        feed = BacktestFeed()

        self.assertEqual(feed.depth.snapshot("BTCUSDT"), {"available": False})
        self.assertEqual(feed.open_interest.snapshot("BTCUSDT"), {"available": False})
        self.assertEqual(feed.liquidations.snapshot("BTCUSDT"), {"available": False})

    def test_stub_sources_accept_any_extra_args(self):
        """evaluate() calls .snapshot(symbol) with no extra args today, but
        the real engines accept optional kwargs (e.g. `now=`) - the stub
        must not blow up if a future caller passes any."""
        feed = BacktestFeed()

        self.assertEqual(feed.depth.snapshot("BTCUSDT", now=123), {"available": False})


class BacktestFeedVolumeFundingTests(unittest.TestCase):
    def test_set_quote_volume_and_funding_rate_are_plain_dict_reads(self):
        feed = BacktestFeed()

        feed.set_quote_volume("BTCUSDT", 42_000_000.0)
        feed.set_funding_rate("BTCUSDT", 0.0001)

        self.assertEqual(feed.volumes.get("BTCUSDT"), 42_000_000.0)
        self.assertEqual(feed.funding_rates.get("BTCUSDT"), 0.0001)
        self.assertIsNone(feed.volumes.get("ETHUSDT"))


if __name__ == "__main__":
    unittest.main()
