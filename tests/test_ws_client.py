import unittest
from unittest.mock import patch

import pandas as pd

import config
from ws_client import CandleStore, RealtimeMarketData


class CandleStoreTests(unittest.TestCase):
    def test_seed_loads_history_as_closed_candles(self):
        store = CandleStore(maxlen=50)
        df = pd.DataFrame([
            {"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
            {"time": 2000, "open": 1.5, "high": 2.5, "low": 1, "close": 2, "volume": 12},
        ])

        store.seed("btcusdt", df)
        candles = store.get("BTCUSDT")

        self.assertEqual(len(candles), 2)
        self.assertTrue(all(c["closed"] for c in candles))
        self.assertEqual(candles[-1]["close"], 2)

    def test_update_with_same_open_time_replaces_forming_candle(self):
        store = CandleStore(maxlen=50)
        store.update("BTCUSDT", {
            "open_time": 1000, "open": 1, "high": 1, "low": 1, "close": 1,
            "volume": 1, "closed": False,
        })
        store.update("BTCUSDT", {
            "open_time": 1000, "open": 1, "high": 1.2, "low": 0.9, "close": 1.1,
            "volume": 3, "closed": False,
        })

        candles = store.get("BTCUSDT")

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0]["close"], 1.1)
        self.assertFalse(candles[0]["closed"])

    def test_update_with_new_open_time_appends_new_candle(self):
        store = CandleStore(maxlen=50)
        store.update("BTCUSDT", {
            "open_time": 1000, "open": 1, "high": 1, "low": 1, "close": 1,
            "volume": 1, "closed": True,
        })
        store.update("BTCUSDT", {
            "open_time": 2000, "open": 1, "high": 1, "low": 1, "close": 1.5,
            "volume": 1, "closed": False,
        })

        candles = store.get("BTCUSDT")

        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0]["closed"], True)
        self.assertEqual(candles[-1]["closed"], False)

    def test_latest_returns_none_for_unknown_symbol(self):
        store = CandleStore(maxlen=50)
        self.assertIsNone(store.latest("NOPE"))


class OiPollLoopTests(unittest.TestCase):
    """Real event (2026-08-11): a -1003 citing "6000 requests per minute"
    (not the weight-budget ban message) hit on a run with 0 open
    positions, where this loop's un-paced full-watchlist burst every
    OI_POLL_INTERVAL_SECONDS was the most likely contributor found while
    investigating. These lock in that the sweep is now paced call-by-call
    across the poll window instead of fired as one tight burst."""

    def test_waits_between_each_symbol_not_after_the_whole_sweep(self):
        feed = RealtimeMarketData(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

        with patch("ws_client.get_open_interest", return_value=100.0), \
             patch.object(config, "OI_POLL_INTERVAL_SECONDS", 30), \
             patch.object(feed.stop_event, "wait", side_effect=[False, False, True]) as wait_mock:
            feed._oi_poll_loop(feed.generation)

        # One wait() call per symbol (3), each for interval/len(symbols) -
        # not a single wait(interval) after the whole sweep.
        self.assertEqual(wait_mock.call_count, 3)
        for call in wait_mock.call_args_list:
            self.assertAlmostEqual(call.args[0], 10.0)  # 30 / 3 symbols

    def test_records_open_interest_for_every_symbol_in_the_sweep(self):
        feed = RealtimeMarketData(["BTCUSDT", "ETHUSDT"])

        with patch("ws_client.get_open_interest", return_value=250.0), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._oi_poll_loop(feed.generation)

        self.assertEqual(feed.open_interest.snapshot("BTCUSDT")["oi_value"], 250.0)
        self.assertEqual(feed.open_interest.snapshot("ETHUSDT")["oi_value"], 250.0)

    def test_empty_symbol_list_waits_the_full_interval_without_dividing_by_zero(self):
        feed = RealtimeMarketData([])

        with patch.object(config, "OI_POLL_INTERVAL_SECONDS", 20), \
             patch.object(feed.stop_event, "wait", return_value=True) as wait_mock:
            feed._oi_poll_loop(feed.generation)

        wait_mock.assert_called_once_with(20.0)

    def test_stop_mid_sweep_does_not_call_open_interest_for_remaining_symbols(self):
        feed = RealtimeMarketData(["BTCUSDT", "ETHUSDT"])

        with patch("ws_client.get_open_interest", return_value=1.0) as oi_mock, \
             patch.object(feed.stop_event, "wait", side_effect=[True]):
            feed._oi_poll_loop(feed.generation)

        oi_mock.assert_called_once_with("BTCUSDT")


class VolumePollLoopTests(unittest.TestCase):
    """Backs config.MIN_24H_QUOTE_VOLUME_USDT, the signal-time liquidity
    floor that replaces watchlist-selection-time filtering when
    SCAN_SYMBOLS is pinned to a broad/unfiltered universe (2026-08-11)."""

    def test_start_volume_poll_is_a_noop_when_the_liquidity_floor_is_disabled(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 0):
            feed._start_volume_poll()

        self.assertIsNone(feed.volume_poll_thread)

    def test_volume_poll_loop_populates_feed_volumes(self):
        feed = RealtimeMarketData(["BTCUSDT", "ETHUSDT"])
        volumes = {"BTCUSDT": 5_000_000, "ETHUSDT": 9_000_000}

        with patch("ws_client.get_24h_quote_volumes", return_value=volumes), \
             patch.object(feed.stop_event, "wait", side_effect=[True]):
            feed._volume_poll_loop(feed.generation)

        self.assertEqual(feed.volumes, volumes)

    def test_empty_response_does_not_clear_existing_volumes(self):
        feed = RealtimeMarketData(["BTCUSDT"])
        feed.volumes = {"BTCUSDT": 5_000_000}

        with patch("ws_client.get_24h_quote_volumes", return_value={}), \
             patch.object(feed.stop_event, "wait", side_effect=[True]):
            feed._volume_poll_loop(feed.generation)

        self.assertEqual(feed.volumes, {"BTCUSDT": 5_000_000})


class FundingPollLoopTests(unittest.TestCase):
    """Backs config.FUNDING_RATE_ENABLED - a bulk (one call, every symbol)
    snapshot, same shape as VolumePollLoopTests above."""

    def test_start_funding_poll_is_a_noop_when_disabled(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        with patch.object(config, "FUNDING_RATE_ENABLED", False):
            feed._start_funding_poll()

        self.assertIsNone(feed.funding_poll_thread)

    def test_funding_poll_loop_populates_feed_funding_rates(self):
        feed = RealtimeMarketData(["BTCUSDT", "ETHUSDT"])
        rates = {"BTCUSDT": 0.0001, "ETHUSDT": -0.0002}

        with patch("ws_client.get_funding_rates", return_value=rates), \
             patch.object(feed.stop_event, "wait", side_effect=[True]):
            feed._funding_poll_loop(feed.generation)

        self.assertEqual(feed.funding_rates, rates)

    def test_empty_response_does_not_clear_existing_funding_rates(self):
        feed = RealtimeMarketData(["BTCUSDT"])
        feed.funding_rates = {"BTCUSDT": 0.0001}

        with patch("ws_client.get_funding_rates", return_value={}), \
             patch.object(feed.stop_event, "wait", side_effect=[True]):
            feed._funding_poll_loop(feed.generation)

        self.assertEqual(feed.funding_rates, {"BTCUSDT": 0.0001})


class RealtimeMarketDataMessageHandlingTests(unittest.TestCase):
    """These exercise the pure message-parsing/routing logic without ever
    opening a real socket (start()/connect() are never called)."""

    def _feed(self):
        return RealtimeMarketData(["BTCUSDT"])

    def test_handle_kline_updates_candle_store(self):
        feed = self._feed()
        feed._handle_kline({
            "s": "BTCUSDT",
            "k": {
                "t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.8",
                "v": "10", "x": False,
            },
        })

        latest = feed.candles.latest("BTCUSDT")

        self.assertIsNotNone(latest)
        self.assertEqual(latest["close"], 1.8)
        self.assertFalse(latest["closed"])

    def test_handle_kline_missing_fields_is_ignored_not_raised(self):
        feed = self._feed()
        feed._handle_kline({"s": "BTCUSDT", "k": {}})

        self.assertIsNone(feed.candles.latest("BTCUSDT"))

    def test_handle_kline_does_not_finalize_cvd_candle_while_still_forming(self):
        feed = self._feed()
        feed._handle_kline({
            "s": "BTCUSDT",
            "k": {"t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.8", "v": "10", "x": False},
        })

        self.assertEqual(feed.cvd.cvd_history("BTCUSDT"), [])

    def test_handle_kline_finalizes_cvd_candle_on_close(self):
        feed = self._feed()
        feed._handle_kline({
            "s": "BTCUSDT",
            "k": {"t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.8", "v": "10", "x": True},
        })

        history = feed.cvd.cvd_history("BTCUSDT")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["open_time"], 1000)

    def test_handle_kline_does_not_finalize_cvd_candle_for_the_htf_stream(self):
        feed = self._feed()

        with patch.object(config, "HTF_KLINE_INTERVAL", "1h"), \
             patch.object(config, "WS_KLINE_INTERVAL", "5m"):
            feed._handle_kline({
                "s": "BTCUSDT",
                "k": {"t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.8", "v": "10", "x": True, "i": "1h"},
            })

        self.assertEqual(feed.cvd.cvd_history("BTCUSDT"), [])

    def test_handle_agg_trade_feeds_cvd_engine(self):
        feed = self._feed()
        feed._handle_agg_trade({"s": "BTCUSDT", "p": "100", "q": "1", "m": False, "T": 1000000})

        with patch.object(config, "ORDER_FLOW_MIN_NOTIONAL_USDT", 0):
            snapshot = feed.cvd.snapshot("BTCUSDT", now=1001)

        self.assertTrue(snapshot["available"])
        self.assertGreater(snapshot["ratio_1m"], 0)

    def test_handle_depth_message_feeds_orderbook_engine(self):
        feed = self._feed()
        feed._handle_depth_message({
            "s": "BTCUSDT",
            "b": [["100", "10"]],
            "a": [["101", "1"]],
        })

        snapshot = feed.depth.snapshot("BTCUSDT")

        self.assertTrue(snapshot["available"])
        self.assertGreater(snapshot["depth_imbalance"], 0)

    def test_worker_active_false_after_stop_event_set(self):
        feed = self._feed()
        generation = feed.generation
        feed.stop_event.set()

        self.assertFalse(feed._worker_active(generation))

    def test_worker_active_false_for_stale_generation(self):
        feed = self._feed()
        generation = feed.generation
        feed.generation += 1

        self.assertFalse(feed._worker_active(generation))


if __name__ == "__main__":
    unittest.main()
