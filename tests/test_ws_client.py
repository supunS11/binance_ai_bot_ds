import json
import unittest
from unittest.mock import Mock, patch

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


class WatchdogLoopTests(unittest.TestCase):
    """Real incident (2026-08-28 20:07 - 2026-08-29 03:06): the depth
    websocket stream hung for ~7 hours with zero detection, because
    _watchdog_loop only ever read last_market_message_at - last_depth_
    message_at was written on every depth message but never checked
    anywhere. These lock in that both streams are now watched, and that
    the reset reason correctly names only the stream(s) that are actually
    stale."""

    def _feed(self):
        return RealtimeMarketData(["BTCUSDT"])

    def test_both_streams_fresh_does_not_reset(self):
        feed = self._feed()
        feed.running = True
        feed.last_market_message_at = 1_000.0
        feed.last_depth_message_at = 1_000.0

        with patch.object(config, "WS_WATCHDOG_INTERVAL_SECONDS", 5), \
             patch.object(config, "WS_STALE_SECONDS", 45), \
             patch.object(feed, "reset_connection") as reset_mock, \
             patch("ws_client.time.time", return_value=1_010.0), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._watchdog_loop()

        reset_mock.assert_not_called()

    def test_depth_stale_market_fresh_names_only_depth_in_reason(self):
        feed = self._feed()
        feed.running = True
        feed.last_market_message_at = 1_000.0
        feed.last_depth_message_at = 900.0  # 100s old

        with patch.object(config, "WS_WATCHDOG_INTERVAL_SECONDS", 5), \
             patch.object(config, "WS_STALE_SECONDS", 45), \
             patch.object(feed, "reset_connection") as reset_mock, \
             patch("ws_client.time.time", return_value=1_000.0), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._watchdog_loop()

        reset_mock.assert_called_once()
        reason = reset_mock.call_args.args[0]
        self.assertIn("depth=", reason)
        self.assertNotIn("market=", reason)

    def test_market_stale_depth_fresh_names_only_market_in_reason(self):
        feed = self._feed()
        feed.running = True
        feed.last_market_message_at = 900.0  # 100s old
        feed.last_depth_message_at = 1_000.0

        with patch.object(config, "WS_WATCHDOG_INTERVAL_SECONDS", 5), \
             patch.object(config, "WS_STALE_SECONDS", 45), \
             patch.object(feed, "reset_connection") as reset_mock, \
             patch("ws_client.time.time", return_value=1_000.0), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._watchdog_loop()

        reset_mock.assert_called_once()
        reason = reset_mock.call_args.args[0]
        self.assertIn("market=", reason)
        self.assertNotIn("depth=", reason)

    def test_both_streams_stale_names_both_in_reason(self):
        feed = self._feed()
        feed.running = True
        feed.last_market_message_at = 900.0
        feed.last_depth_message_at = 900.0

        with patch.object(config, "WS_WATCHDOG_INTERVAL_SECONDS", 5), \
             patch.object(config, "WS_STALE_SECONDS", 45), \
             patch.object(feed, "reset_connection") as reset_mock, \
             patch("ws_client.time.time", return_value=1_000.0), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._watchdog_loop()

        reset_mock.assert_called_once()
        reason = reset_mock.call_args.args[0]
        self.assertIn("market=", reason)
        self.assertIn("depth=", reason)


class ResetConnectionTests(unittest.TestCase):
    """Same 2026-08-28 incident: _reset_connection only ever closed/
    restarted the market stream, but always bumped the single shared
    generation counter both stream types key off - so any watchdog reset
    silently and permanently killed the depth stream, since _start_depth_
    streams() was never called to bring it back. These lock in that a
    reset now closes and restarts both."""

    def _feed(self):
        return RealtimeMarketData(["BTCUSDT"])

    def test_reset_restarts_both_market_and_depth_streams(self):
        feed = self._feed()

        with patch.object(feed, "_start_market_streams") as start_market, \
             patch.object(feed, "_start_depth_streams") as start_depth, \
             patch.object(feed.stop_event, "wait", return_value=False):
            feed._reset_connection("test reason")

        start_market.assert_called_once()
        start_depth.assert_called_once()

    def test_reset_closes_existing_market_and_depth_sockets(self):
        feed = self._feed()
        market_socket = Mock()
        depth_socket = Mock()
        feed.market_websockets = {1: market_socket}
        feed.depth_websockets = {2: depth_socket}

        with patch.object(feed, "_start_market_streams"), \
             patch.object(feed, "_start_depth_streams"), \
             patch.object(feed.stop_event, "wait", return_value=False):
            feed._reset_connection("test reason")

        market_socket.close.assert_called_once()
        depth_socket.close.assert_called_once()
        self.assertEqual(feed.market_websockets, {})
        self.assertEqual(feed.depth_websockets, {})

    def test_reset_refreshes_both_last_message_timestamps(self):
        feed = self._feed()
        feed.last_market_message_at = 0.0
        feed.last_depth_message_at = 0.0

        with patch.object(feed, "_start_market_streams"), \
             patch.object(feed, "_start_depth_streams"), \
             patch.object(feed.stop_event, "wait", return_value=False):
            feed._reset_connection("test reason")

        self.assertGreater(feed.last_market_message_at, 0.0)
        self.assertGreater(feed.last_depth_message_at, 0.0)


class SeedHistoryTests(unittest.TestCase):
    """Real incident (2026-08-29): CVD_DIVERGENCE_TRIGGER_ENABLED/OI_
    DIVERGENCE_TRIGGER_ENABLED are both live but never once found matching
    data (0/1,457,496 and 0/3,068,258 diagnostic log lines respectively),
    because CVDEngine/OpenInterestEngine's in-memory history used to start
    empty on every restart while self.candles (compared against for swing
    points) is REST-seeded and restart-surviving. _seed_history now seeds
    both the same way candles already are - gated behind the same flags
    that gate the trigger/diagnostic itself, so no REST budget is spent
    seeding data nobody reads when either is off."""

    def _feed(self):
        return RealtimeMarketData(["BTCUSDT"])

    def test_cvd_is_seeded_from_the_same_ltf_dataframe_as_candles_when_enabled(self):
        feed = self._feed()
        ltf_sentinel = pd.DataFrame([{"time": 1, "tbqav": "1", "qav": "1"}])

        with patch("ws_client.get_klines", return_value=ltf_sentinel) as klines_mock, \
             patch.object(feed.candles, "seed"), \
             patch.object(feed.htf_candles, "seed"), \
             patch.object(feed.cvd, "seed_from_klines") as cvd_seed_mock, \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", False):
            feed._seed_history()

        cvd_seed_mock.assert_called_once_with("BTCUSDT", ltf_sentinel)
        # Only the existing 2 calls (LTF + HTF candles) - CVD reuses the
        # LTF one rather than triggering a 3rd get_klines call of its own.
        self.assertEqual(klines_mock.call_count, 2)

    def test_cvd_seeding_is_skipped_when_the_trigger_is_disabled(self):
        feed = self._feed()

        with patch("ws_client.get_klines", return_value=pd.DataFrame()), \
             patch.object(feed.candles, "seed"), \
             patch.object(feed.htf_candles, "seed"), \
             patch.object(feed.cvd, "seed_from_klines") as cvd_seed_mock, \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", False), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", False):
            feed._seed_history()

        cvd_seed_mock.assert_not_called()

    def test_oi_is_seeded_from_get_open_interest_history_when_enabled(self):
        feed = self._feed()
        history_sentinel = [(100.0, 1000.0)]

        with patch("ws_client.get_klines", return_value=pd.DataFrame()), \
             patch.object(feed.candles, "seed"), \
             patch.object(feed.htf_candles, "seed"), \
             patch("ws_client.get_open_interest_history", return_value=history_sentinel) as oi_hist_mock, \
             patch.object(feed.open_interest, "seed_from_history") as oi_seed_mock, \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", False), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True):
            feed._seed_history()

        oi_hist_mock.assert_called_once_with("BTCUSDT")
        oi_seed_mock.assert_called_once_with("BTCUSDT", history_sentinel)

    def test_oi_seeding_is_skipped_when_the_trigger_is_disabled(self):
        feed = self._feed()

        with patch("ws_client.get_klines", return_value=pd.DataFrame()), \
             patch.object(feed.candles, "seed"), \
             patch.object(feed.htf_candles, "seed"), \
             patch("ws_client.get_open_interest_history") as oi_hist_mock, \
             patch.object(feed.open_interest, "seed_from_history") as oi_seed_mock, \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", False), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", False):
            feed._seed_history()

        oi_hist_mock.assert_not_called()
        oi_seed_mock.assert_not_called()


class LiquidationStreamUrlTests(unittest.TestCase):
    """Real bug found live (2026-08-29): the liquidation stream used a
    bare, unrouted wss://fstream.binance.com/ws/!forceOrder@arr URL.
    Binance's WebSocket routing change (deadline 2026-04-23, already
    passed) means an unrouted connection now only receives /public-
    category data, and forceOrder is a /market-category stream (confirmed
    against Binance's own docs) - so this silently delivered zero
    liquidation events from that date forward (confirmed live: 0/191
    resolved trades ever had liquidation data, 0 messages on a real 90s
    connection to the old URL, real messages within seconds on the routed
    one). Locks in that the stream now connects through the same routed
    /market/stream?streams= base every other market-category stream
    (kline/aggTrade) already uses, instead of its own hardcoded URL."""

    def test_connects_to_the_routed_market_stream_url_not_the_legacy_bare_one(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        def _raise_and_stop(*args, **kwargs):
            feed.stop_event.set()
            raise RuntimeError("boom")

        with patch("websockets.sync.client.connect", side_effect=_raise_and_stop) as connect_mock:
            feed._liquidation_stream_loop(feed.generation)

        connect_mock.assert_called_once()
        url = connect_mock.call_args.args[0]
        self.assertEqual(url, "wss://fstream.binance.com/market/stream?streams=!forceOrder@arr")
        self.assertNotIn("/ws/!forceOrder", url)


class CrossExchangeLiquidationStreamUrlTests(unittest.TestCase):
    """Confirms the new Bybit/OKX liquidation loops connect to their real,
    live-verified (2026-08-29) URLs, same test shape as the existing
    Binance LiquidationStreamUrlTests above."""

    def test_bybit_connects_to_the_confirmed_mainnet_url(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        def _raise_and_stop(*args, **kwargs):
            feed.stop_event.set()
            raise RuntimeError("boom")

        with patch("websockets.sync.client.connect", side_effect=_raise_and_stop) as connect_mock:
            feed._liquidation_stream_loop_bybit(feed.generation)

        connect_mock.assert_called_once()
        self.assertEqual(connect_mock.call_args.args[0], "wss://stream.bybit.com/v5/public/linear")

    def test_okx_connects_to_the_confirmed_url(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        def _raise_and_stop(*args, **kwargs):
            feed.stop_event.set()
            raise RuntimeError("boom")

        with patch("websockets.sync.client.connect", side_effect=_raise_and_stop) as connect_mock:
            feed._liquidation_stream_loop_okx(feed.generation)

        connect_mock.assert_called_once()
        self.assertEqual(connect_mock.call_args.args[0], "wss://ws.okx.com:8443/ws/v5/public")


class ReconnectBackoffWaitTests(unittest.TestCase):
    """Real segfault found live (2026-09-01): several realtime socket
    threads (market, depth, and both liquidation streams all share this
    exact reconnect shape) reconnected within the same few seconds after
    a shared network blip, and the process crashed with a genuine OS-
    level segfault shortly after - not a catchable Python exception.
    Jittering the reconnect backoff spreads retries apart instead of
    letting a shared failure cluster them into the same instant."""

    def test_waits_at_least_the_base_delay(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        with patch.object(feed.stop_event, "wait") as wait_mock, \
             patch("random.uniform", return_value=0.0):
            feed._reconnect_backoff_wait()

        wait_mock.assert_called_once_with(3.0)

    def test_adds_jitter_within_the_documented_spread(self):
        feed = RealtimeMarketData(["BTCUSDT"])

        with patch.object(feed.stop_event, "wait") as wait_mock, \
             patch("random.uniform", return_value=1.5) as uniform_mock:
            feed._reconnect_backoff_wait()

        uniform_mock.assert_called_once_with(0, 2)
        wait_mock.assert_called_once_with(4.5)

    def test_all_reconnect_loops_use_the_shared_jittered_backoff(self):
        # Locks in that market/depth/liquidation (Binance+Bybit+OKX) all
        # route through the same helper, not their own independent flat
        # wait(3) - consistency matters here since any of them clustering
        # with the others is what triggers the underlying race. The first
        # connect() failure leaves stop_event unset (backoff must fire and
        # the loop must retry); the second failure sets stop_event so the
        # loop exits instead of spinning forever.
        feed = RealtimeMarketData(["BTCUSDT"])
        loops = [
            (feed._market_stream_loop, True),
            (feed._depth_stream_loop, True),
            (feed._liquidation_stream_loop, False),
            (feed._liquidation_stream_loop_bybit, False),
            (feed._liquidation_stream_loop_okx, False),
        ]

        for loop, takes_symbols in loops:
            feed.stop_event.clear()
            calls = {"n": 0}

            def _raise_then_stop(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] >= 2:
                    feed.stop_event.set()
                raise RuntimeError("boom")

            with patch("websockets.sync.client.connect", side_effect=_raise_then_stop), \
                 patch.object(feed, "_reconnect_backoff_wait") as backoff_mock:
                if takes_symbols:
                    loop(["BTCUSDT"], feed.generation)
                else:
                    loop(feed.generation)

            backoff_mock.assert_called_once()

        feed.stop_event.clear()


class BybitLiquidationSubscribeTests(unittest.TestCase):
    """Real live finding (2026-08-29): Bybit rejects the ENTIRE subscribe
    batch if even one symbol has no liquidation handler (~5% of a real
    watchlist) - not a per-connection topic-count problem. These lock in
    the strip-and-retry loop that works around it."""

    def _feed(self, symbols):
        return RealtimeMarketData(symbols)

    def test_immediate_success_sends_once(self):
        feed = self._feed(["BTCUSDT", "ETHUSDT"])
        websocket = Mock()
        websocket.recv.return_value = json.dumps({"success": True, "ret_msg": "", "op": "subscribe"})

        result = feed._bybit_liquidation_subscribe(websocket, feed.generation)

        self.assertTrue(result)
        self.assertEqual(websocket.send.call_count, 1)
        sent = json.loads(websocket.send.call_args.args[0])
        self.assertEqual(sent["args"], ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"])

    def test_rejected_symbol_is_excluded_and_retried_until_success(self):
        feed = self._feed(["BTCUSDT", "1000000BOBUSDT", "ETHUSDT"])
        websocket = Mock()
        websocket.recv.side_effect = [
            json.dumps({
                "success": False,
                "ret_msg": "error:handler not found,topic:allLiquidation.1000000BOBUSDT",
            }),
            json.dumps({"success": True, "ret_msg": ""}),
        ]

        result = feed._bybit_liquidation_subscribe(websocket, feed.generation)

        self.assertTrue(result)
        self.assertEqual(websocket.send.call_count, 2)
        second_sent = json.loads(websocket.send.call_args_list[1].args[0])
        self.assertEqual(second_sent["args"], ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"])

    def test_every_symbol_rejected_gives_up_and_returns_false(self):
        feed = self._feed(["BTCUSDT", "ETHUSDT"])
        websocket = Mock()
        websocket.recv.side_effect = [
            json.dumps({"success": False, "ret_msg": "error:handler not found,topic:allLiquidation.BTCUSDT"}),
            json.dumps({"success": False, "ret_msg": "error:handler not found,topic:allLiquidation.ETHUSDT"}),
        ]

        result = feed._bybit_liquidation_subscribe(websocket, feed.generation)

        self.assertFalse(result)
        self.assertEqual(websocket.send.call_count, 2)

    def test_stop_requested_mid_retry_returns_false_without_further_sends(self):
        feed = self._feed(["BTCUSDT", "ETHUSDT"])
        websocket = Mock()

        def _reject_and_stop(*args, **kwargs):
            feed.stop_event.set()
            return json.dumps({"success": False, "ret_msg": "error:handler not found,topic:allLiquidation.BTCUSDT"})

        websocket.recv.side_effect = _reject_and_stop

        result = feed._bybit_liquidation_subscribe(websocket, feed.generation)

        self.assertFalse(result)
        self.assertEqual(websocket.send.call_count, 1)

    def test_unrecognized_rejection_reply_gives_up_immediately(self):
        feed = self._feed(["BTCUSDT"])
        websocket = Mock()
        websocket.recv.return_value = json.dumps({"success": False, "ret_msg": "some other error"})

        result = feed._bybit_liquidation_subscribe(websocket, feed.generation)

        self.assertFalse(result)
        self.assertEqual(websocket.send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
