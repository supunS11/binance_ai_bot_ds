"""Real-time Binance USDT-M futures market-data client.

Connection pattern adapted from v7/v8's proven
`market_intelligence.py::MarketFlowMonitor` (raw `websockets` client against
Binance's combined-stream gateway, chunked across sockets, with a watchdog
that force-reconnects on staleness) - generalized here to also carry the
kline stream needed for real-time market-structure detection (Phase 2), and
wired to feed `order_flow.CVDEngine` / `orderbook.DepthImbalanceEngine`
directly instead of computing its own scoring.

This module owns transport only: connecting, reconnecting, and routing
parsed messages to the candle store / CVD engine / depth engine. It has no
signal logic of its own.
"""
from collections import deque
import json
import random
import threading
import time

import config
from logger import log_error, log_info, log_warning
from exchange import (
    get_klines, get_open_interest, get_open_interest_history,
    get_24h_quote_volumes, get_funding_rates,
)
from order_flow import CVDEngine
from orderbook import DepthImbalanceEngine
from open_interest import OpenInterestEngine
from liquidation_tracker import LiquidationEngine, LIQUIDATION_STREAM_NAME
from crash_detector import CrashDetector
from volume_profile import VolumeProfileEngine
import cross_exchange_oi
import cross_exchange_liquidation


FUTURES_MARKET_STREAM_BASE = "wss://fstream.binance.com/market/stream?streams="
FUTURES_PUBLIC_STREAM_BASE = "wss://fstream.binance.com/public/stream?streams="
TESTNET_MARKET_STREAM_BASE = "wss://stream.binancefuture.com/market/stream?streams="
TESTNET_PUBLIC_STREAM_BASE = "wss://stream.binancefuture.com/public/stream?streams="


def _market_stream_base():
    return TESTNET_MARKET_STREAM_BASE if config.BINANCE_TESTNET else FUTURES_MARKET_STREAM_BASE


def _public_stream_base():
    return TESTNET_PUBLIC_STREAM_BASE if config.BINANCE_TESTNET else FUTURES_PUBLIC_STREAM_BASE


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _message_data(message):
    if not isinstance(message, dict):
        return {}

    data = message.get("data")
    return data if isinstance(data, dict) else message


class CandleStore:
    """Per-symbol rolling candle buffer built from the kline websocket
    stream. The last item may still be "forming" (closed=False) -
    deliberately, so structure detection (Phase 2) can react to a live,
    in-progress candle instead of only ever seeing what already closed."""

    def __init__(self, maxlen=200):
        self.maxlen = max(int(maxlen), 10)
        self.lock = threading.RLock()
        self._candles = {}

    def seed(self, symbol, df):
        symbol = symbol.upper()

        if df is None or df.empty:
            return

        candles = deque(maxlen=self.maxlen)

        for _, row in df.iterrows():
            candles.append({
                "open_time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "closed": True,
            })

        with self.lock:
            # A live update may have already arrived before the REST seed
            # finished - keep it if it's newer than the seeded history.
            existing = self._candles.get(symbol)

            if existing and existing[-1]["open_time"] > candles[-1]["open_time"]:
                candles.append(existing[-1])

            self._candles[symbol] = candles

    def update(self, symbol, candle):
        symbol = symbol.upper()

        with self.lock:
            candles = self._candles.get(symbol)

            if candles is None:
                candles = deque(maxlen=self.maxlen)
                self._candles[symbol] = candles

            if candles and candles[-1]["open_time"] == candle["open_time"]:
                candles[-1] = candle
            else:
                candles.append(candle)

    def get(self, symbol):
        symbol = symbol.upper()

        with self.lock:
            candles = self._candles.get(symbol)
            return list(candles) if candles else []

    def latest(self, symbol):
        candles = self.get(symbol)
        return candles[-1] if candles else None


class RealtimeMarketData:
    def __init__(self, symbols, shutdown_event=None):
        self.enabled = bool(config.WS_ENABLED)
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        self.shutdown_event = shutdown_event
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        self.candles = CandleStore(maxlen=config.WS_KLINE_HISTORY_LIMIT)
        self.htf_candles = CandleStore(maxlen=config.HTF_KLINE_HISTORY_LIMIT)
        self.cvd = CVDEngine()
        self.depth = DepthImbalanceEngine()
        self.open_interest = OpenInterestEngine()
        # Reuses OpenInterestEngine as-is (fully generic - it doesn't know
        # or care the value came from Binance) for two more venues - see
        # config.CROSS_EXCHANGE_OI_TRACKING_ENABLED.
        self.open_interest_bybit = OpenInterestEngine()
        self.open_interest_okx = OpenInterestEngine()
        self.volume_profile = VolumeProfileEngine()
        self.liquidations = LiquidationEngine()
        # Reuses LiquidationEngine as-is (exchange-agnostic - only ever
        # sees normalized tuples via record_liquidation) for two more
        # venues - see config.CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED.
        self.liquidations_bybit = LiquidationEngine()
        self.liquidations_okx = LiquidationEngine()
        self.crash_detector = CrashDetector()
        # 24h quote volume per symbol - the data behind the signal-time
        # liquidity floor (config.MIN_24H_QUOTE_VOLUME_USDT). Plain dict,
        # not a dedicated engine class: it's a single bulk snapshot
        # refreshed wholesale, not per-symbol accumulated state like OI.
        self.volumes = {}
        # Current funding rate per symbol - same bulk-snapshot shape as
        # volumes above, see config.FUNDING_RATE_ENABLED.
        self.funding_rates = {}

        self.running = False
        self.resetting = False
        self.generation = 0
        self.last_market_message_at = 0.0
        self.last_depth_message_at = 0.0
        self.last_restart_at = 0.0
        self.market_threads = []
        self.depth_threads = []
        self.market_websockets = {}
        self.depth_websockets = {}
        self.watchdog_thread = None
        self.oi_poll_thread = None
        self.oi_poll_thread_bybit = None
        self.oi_poll_thread_okx = None
        self.liquidation_thread = None
        self.liquidation_thread_bybit = None
        self.liquidation_thread_okx = None
        self.volume_poll_thread = None
        self.funding_poll_thread = None
        self.liquidation_websocket = None
        self.liquidation_websocket_bybit = None
        self.liquidation_websocket_okx = None

    # =========================
    # LIFECYCLE
    # =========================
    def start(self):
        if not self.enabled or not self.symbols:
            log_info("Realtime market data websocket disabled")
            return

        try:
            from websockets.sync.client import connect  # noqa: F401
        except ImportError:
            log_error(
                "Realtime market data websocket unavailable | "
                "`websockets` package not installed"
            )
            return

        self._seed_history()
        self._start_watchdog()

        with self.lock:
            if self.running:
                return

            self.running = True
            self.generation += 1
            self.last_market_message_at = time.time()
            self.last_depth_message_at = time.time()

        self._start_market_streams()
        self._start_depth_streams()
        self._start_oi_poll()
        self._start_oi_poll_bybit()
        self._start_oi_poll_okx()
        self._start_liquidation_stream()
        self._start_liquidation_stream_bybit()
        self._start_liquidation_stream_okx()
        self._start_volume_poll()
        self._start_funding_poll()

        log_info(
            f"Realtime market data websocket started | SYMBOLS={len(self.symbols)} | "
            f"MARKET_SOCKETS={len(self.market_threads)} | "
            f"DEPTH_SOCKETS={len(self.depth_threads)}"
        )

    def stop(self):
        self.stop_event.set()

        with self.lock:
            self.running = False
            self.generation += 1
            market_sockets = list(self.market_websockets.values())
            depth_sockets = list(self.depth_websockets.values())
            liquidation_sockets = [
                socket for socket in (
                    self.liquidation_websocket,
                    self.liquidation_websocket_bybit,
                    self.liquidation_websocket_okx,
                ) if socket is not None
            ]
            self.market_websockets = {}
            self.depth_websockets = {}
            self.liquidation_websocket = None
            self.liquidation_websocket_bybit = None
            self.liquidation_websocket_okx = None

        self._close_websockets(market_sockets + depth_sockets + liquidation_sockets)

    def _seed_history(self):
        for symbol in self.symbols:
            ltf_df = get_klines(
                symbol,
                config.WS_KLINE_INTERVAL,
                limit=config.WS_KLINE_HISTORY_LIMIT,
            )
            self.candles.seed(symbol, ltf_df)

            if config.CVD_DIVERGENCE_TRIGGER_ENABLED:
                # Reuses ltf_df above - no separate REST call needed, see
                # CVDEngine.seed_from_klines's own comment.
                self.cvd.seed_from_klines(symbol, ltf_df)

            htf_df = get_klines(
                symbol,
                config.HTF_KLINE_INTERVAL,
                limit=config.HTF_KLINE_HISTORY_LIMIT,
            )
            self.htf_candles.seed(symbol, htf_df)

            if config.OI_DIVERGENCE_TRIGGER_ENABLED:
                self.open_interest.seed_from_history(
                    symbol, get_open_interest_history(symbol)
                )

    def _worker_active(self, generation):
        if self.stop_event.is_set():
            return False

        if self.shutdown_event is not None and self.shutdown_event.is_set():
            return False

        with self.lock:
            return generation == self.generation

    def _reconnect_backoff_wait(self):
        """Real segfault found live (2026-09-01): a shared network blip
        made several of this bot's realtime socket threads (market,
        depth, and both liquidation streams all share this exact reconnect
        shape) fail and reconnect within the same few seconds - the
        process crashed with a genuine OS-level segfault
        (`kernel: python[...]: segfault ... in python3.12`) shortly after,
        not a catchable Python exception. websockets.sync's C-level
        socket/SSL teardown isn't proven safe under many threads
        reconnecting in lockstep, and this bot runs 10+ of these threads
        concurrently (market + depth chunks, plus 3 liquidation feeds).
        Jittering the backoff spreads reconnect attempts apart instead of
        letting a shared failure cluster them into the same instant -
        reduces how often threads hit that risky window together. This
        can't guarantee eliminating the underlying race (that's inside a
        third-party C extension, not something fixable from Python), only
        make the trigger condition less likely."""
        self.stop_event.wait(3 + random.uniform(0, 2))

    @staticmethod
    def _close_websockets(websockets):
        for websocket in websockets:
            try:
                websocket.close()
            except Exception as exc:
                log_warning(f"Realtime market data websocket close warning: {exc}")

    # =========================
    # WATCHDOG
    # =========================
    def _start_watchdog(self):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return

        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="realtime-market-data-watchdog",
            daemon=True,
        )
        self.watchdog_thread.start()

    def _watchdog_loop(self):
        interval = max(float(config.WS_WATCHDOG_INTERVAL_SECONDS), 5)

        while not self.stop_event.wait(interval):
            if self.shutdown_event is not None and self.shutdown_event.is_set():
                return

            stale_seconds = max(float(config.WS_STALE_SECONDS), 15)
            now = time.time()

            with self.lock:
                market_age = (
                    now - self.last_market_message_at
                    if self.last_market_message_at
                    else stale_seconds + 1
                )
                depth_age = (
                    now - self.last_depth_message_at
                    if self.last_depth_message_at
                    else stale_seconds + 1
                )
                running = self.running
                resetting = self.resetting

            market_stale = market_age >= stale_seconds
            depth_stale = depth_age >= stale_seconds

            if not resetting and (not running or market_stale or depth_stale):
                reason_parts = ["watchdog stale"]

                if market_stale:
                    reason_parts.append(f"market={round(market_age, 1)}s")

                if depth_stale:
                    reason_parts.append(f"depth={round(depth_age, 1)}s")

                if not running:
                    reason_parts.append("not running")

                self.reset_connection(" ".join(reason_parts))

    def reset_connection(self, reason):
        now = time.time()
        cooldown = max(float(config.WS_RESTART_COOLDOWN_SECONDS), 5)

        with self.lock:
            if self.resetting or now - self.last_restart_at < cooldown:
                return

            self.resetting = True
            self.last_restart_at = now

        threading.Thread(
            target=self._reset_connection,
            args=(reason,),
            name="realtime-market-data-reset",
            daemon=True,
        ).start()

    def _reset_connection(self, reason):
        log_warning(f"Realtime market data websocket resetting | REASON={reason}")

        try:
            with self.lock:
                self.running = False
                self.generation += 1
                market_sockets = list(self.market_websockets.values())
                depth_sockets = list(self.depth_websockets.values())
                self.market_websockets = {}
                self.depth_websockets = {}

            self._close_websockets(market_sockets + depth_sockets)

            if self.stop_event.wait(2):
                return

            with self.lock:
                self.running = True
                self.last_market_message_at = time.time()
                self.last_depth_message_at = time.time()

            self._start_market_streams()
            self._start_depth_streams()

            log_info(
                f"Realtime market data websocket restored | "
                f"MARKET_SOCKETS={len(self.market_threads)} | "
                f"DEPTH_SOCKETS={len(self.depth_threads)}"
            )

        except Exception as exc:
            log_error(f"Realtime market data websocket reset error: {exc}")
        finally:
            with self.lock:
                self.resetting = False

    # =========================
    # KLINE + AGGTRADE (combined per symbol - both light streams)
    # =========================
    def _start_market_streams(self):
        chunk_size = max(int(config.WS_STREAMS_PER_SOCKET), 1)

        with self.lock:
            generation = self.generation

        threads = []

        for start in range(0, len(self.symbols), chunk_size):
            symbols = self.symbols[start:start + chunk_size]
            thread = threading.Thread(
                target=self._market_stream_loop,
                args=(symbols, generation),
                name=f"realtime-market-{(start // chunk_size) + 1}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with self.lock:
            if generation == self.generation:
                self.market_threads = threads

    def _market_stream_names(self, symbols):
        ltf_interval = config.WS_KLINE_INTERVAL
        htf_interval = config.HTF_KLINE_INTERVAL
        names = []

        for symbol in symbols:
            lowered = symbol.lower()
            names.append(f"{lowered}@kline_{ltf_interval}")

            if htf_interval != ltf_interval:
                names.append(f"{lowered}@kline_{htf_interval}")

            names.append(f"{lowered}@aggTrade")

        return names

    def _market_stream_loop(self, symbols, generation):
        from websockets.sync.client import connect

        streams = "/".join(self._market_stream_names(symbols))
        url = f"{_market_stream_base()}{streams}"
        worker_id = threading.get_ident()

        while self._worker_active(generation):
            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.market_websockets[worker_id] = websocket

                    while self._worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self._handle_market_message(json.loads(message))

            except Exception as exc:
                if self._worker_active(generation):
                    log_warning(f"Realtime market websocket reconnecting: {exc}")
                    self._reconnect_backoff_wait()
            finally:
                with self.lock:
                    self.market_websockets.pop(worker_id, None)

    def _handle_market_message(self, message):
        if self.stop_event.is_set():
            return

        data = _message_data(message)

        if not isinstance(data, dict):
            return

        event_type = data.get("e")

        with self.lock:
            self.last_market_message_at = time.time()

        if event_type == "kline":
            self._handle_kline(data)
        elif event_type == "aggTrade":
            self._handle_agg_trade(data)

    def _handle_kline(self, data):
        kline = data.get("k") or {}
        symbol = str(data.get("s") or kline.get("s") or "").upper()

        if not symbol or "t" not in kline:
            return

        try:
            candle = {
                "open_time": int(kline["t"]),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "closed": bool(kline.get("x")),
            }
        except (TypeError, ValueError, KeyError):
            return

        interval = kline.get("i")

        if interval == config.HTF_KLINE_INTERVAL and interval != config.WS_KLINE_INTERVAL:
            self.htf_candles.update(symbol, candle)
        else:
            self.candles.update(symbol, candle)

            # Only the LTF stream feeds CVD divergence (cvd_divergence.py
            # compares CVD against LTF swing points, same timeframe every
            # other trigger uses) - the HTF stream never calls this.
            if candle["closed"]:
                self.cvd.finalize_candle(symbol, candle["open_time"])

    def _handle_agg_trade(self, data):
        symbol = str(data.get("s") or "").upper()

        if not symbol:
            return

        timestamp = _safe_float(data.get("T") or data.get("E"), time.time() * 1000) / 1000
        self.cvd.record_trade(
            symbol,
            data.get("p"),
            data.get("q"),
            bool(data.get("m")),
            timestamp=timestamp,
        )

        if config.VOLUME_PROFILE_TRACKING_ENABLED:
            self.volume_profile.record_trade(symbol, data.get("p"), data.get("q"), timestamp=timestamp)

        # config.CRASH_DETECTOR_REFERENCE_SYMBOL - single-symbol check,
        # negligible added cost on this otherwise-hot per-tick path (see
        # crash_detector.py for why this needs the raw trade stream
        # instead of the much coarser LTF kline).
        if symbol == config.CRASH_DETECTOR_REFERENCE_SYMBOL:
            self.crash_detector.record_price(data.get("p"), timestamp=timestamp)

    # =========================
    # DEPTH (heavier - own socket group, only for the watchlist tier)
    # =========================
    def _start_depth_streams(self):
        chunk_size = max(int(config.WS_DEPTH_STREAMS_PER_SOCKET), 1)

        with self.lock:
            generation = self.generation

        threads = []

        for start in range(0, len(self.symbols), chunk_size):
            symbols = self.symbols[start:start + chunk_size]
            thread = threading.Thread(
                target=self._depth_stream_loop,
                args=(symbols, generation),
                name=f"realtime-depth-{(start // chunk_size) + 1}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with self.lock:
            if generation == self.generation:
                self.depth_threads = threads

    def _depth_stream_names(self, symbols):
        levels = config.WS_DEPTH_LEVELS
        speed = config.WS_DEPTH_SPEED_MS
        return [f"{symbol.lower()}@depth{levels}@{speed}" for symbol in symbols]

    def _depth_stream_loop(self, symbols, generation):
        from websockets.sync.client import connect

        streams = "/".join(self._depth_stream_names(symbols))
        url = f"{_public_stream_base()}{streams}"
        worker_id = threading.get_ident()

        while self._worker_active(generation):
            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.depth_websockets[worker_id] = websocket

                    while self._worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self._handle_depth_message(json.loads(message))

            except Exception as exc:
                if self._worker_active(generation):
                    log_warning(f"Realtime depth websocket reconnecting: {exc}")
                    self._reconnect_backoff_wait()
            finally:
                with self.lock:
                    self.depth_websockets.pop(worker_id, None)

    def _handle_depth_message(self, message):
        if self.stop_event.is_set():
            return

        data = _message_data(message)

        if not isinstance(data, dict):
            return

        symbol = str(data.get("s") or "").upper()
        bids = data.get("b") or []
        asks = data.get("a") or []

        if not symbol or not bids or not asks:
            return

        with self.lock:
            self.last_depth_message_at = time.time()

        self.depth.record_depth(symbol, bids, asks)

    # =========================
    # OPEN INTEREST (REST-polled - no public OI websocket stream exists)
    # =========================
    def _start_oi_poll(self):
        if not config.OI_CONFIRMATION_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._oi_poll_loop,
            args=(generation,),
            name="realtime-oi-poll",
            daemon=True,
        )
        thread.start()
        self.oi_poll_thread = thread

    def _oi_poll_loop(self, generation):
        """Spreads the per-symbol REST calls evenly across the poll window
        instead of firing them all back-to-back and then sleeping the
        remainder - a tight burst of N rapid-fire calls is exactly the
        shape of traffic most likely to trip a raw-request-frequency
        limit (distinct from the weight-budget one - see
        exchange._rate_limit_public_request), even though the total
        weight per sweep is unchanged either way. Real event (2026-08-11):
        a -1003 citing "6000 requests per minute", not a weight-budget
        ban, on a run with 0 open positions - this loop's un-paced 40-
        symbol burst every OI_POLL_INTERVAL_SECONDS was the most likely
        contributor found while investigating."""
        interval = max(float(config.OI_POLL_INTERVAL_SECONDS), 5)

        while self._worker_active(generation):
            if not self.symbols:
                if self.stop_event.wait(interval):
                    return
                continue

            gap = interval / len(self.symbols)

            for symbol in self.symbols:
                if not self._worker_active(generation):
                    return

                self.open_interest.record(symbol, get_open_interest(symbol))

                if self.stop_event.wait(gap):
                    return

    # =========================
    # CROSS-EXCHANGE OPEN INTEREST (Bybit, OKX - informational only, see
    # config.CROSS_EXCHANGE_OI_TRACKING_ENABLED). Same spread-across-the-
    # window pacing as _oi_poll_loop above, on separate threads so a slow
    # or unavailable venue never delays Binance's own OI poll.
    # =========================
    def _start_oi_poll_bybit(self):
        if not config.CROSS_EXCHANGE_OI_TRACKING_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._oi_poll_loop_bybit,
            args=(generation,),
            name="realtime-oi-poll-bybit",
            daemon=True,
        )
        thread.start()
        self.oi_poll_thread_bybit = thread

    def _oi_poll_loop_bybit(self, generation):
        interval = max(float(config.CROSS_EXCHANGE_OI_POLL_INTERVAL_SECONDS), 5)

        while self._worker_active(generation):
            if not self.symbols:
                if self.stop_event.wait(interval):
                    return
                continue

            gap = interval / len(self.symbols)

            for symbol in self.symbols:
                if not self._worker_active(generation):
                    return

                self.open_interest_bybit.record(symbol, cross_exchange_oi.get_open_interest_bybit(symbol))

                if self.stop_event.wait(gap):
                    return

    def _start_oi_poll_okx(self):
        if not config.CROSS_EXCHANGE_OI_TRACKING_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._oi_poll_loop_okx,
            args=(generation,),
            name="realtime-oi-poll-okx",
            daemon=True,
        )
        thread.start()
        self.oi_poll_thread_okx = thread

    def _oi_poll_loop_okx(self, generation):
        interval = max(float(config.CROSS_EXCHANGE_OI_POLL_INTERVAL_SECONDS), 5)

        while self._worker_active(generation):
            if not self.symbols:
                if self.stop_event.wait(interval):
                    return
                continue

            gap = interval / len(self.symbols)

            for symbol in self.symbols:
                if not self._worker_active(generation):
                    return

                self.open_interest_okx.record(symbol, cross_exchange_oi.get_open_interest_okx(symbol))

                if self.stop_event.wait(gap):
                    return

    # =========================
    # 24H QUOTE VOLUME (single bulk REST call, not per-symbol - backs the
    # signal-time liquidity floor, config.MIN_24H_QUOTE_VOLUME_USDT)
    # =========================
    def _start_volume_poll(self):
        if float(config.MIN_24H_QUOTE_VOLUME_USDT) <= 0:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._volume_poll_loop,
            args=(generation,),
            name="realtime-volume-poll",
            daemon=True,
        )
        thread.start()
        self.volume_poll_thread = thread

    def _volume_poll_loop(self, generation):
        """Refreshes self.volumes from a single bulk call covering every
        symbol (get_24h_quote_volumes, already weight-throttled) - cheap
        enough that no per-symbol pacing is needed, unlike _oi_poll_loop.
        Real motivation (2026-08-11): SCAN_SYMBOLS was widened back to the
        full 500+ symbol universe (including known illiquid/vanity
        tickers) for broader coverage - this is the data behind the
        signal-time quality filter that replaces what watchlist selection
        used to provide."""
        interval = max(float(config.VOLUME_POLL_INTERVAL_SECONDS), 30)

        while self._worker_active(generation):
            volumes = get_24h_quote_volumes()

            if volumes:
                with self.lock:
                    self.volumes = volumes

            if self.stop_event.wait(interval):
                return

    # =========================
    # FUNDING RATE (single bulk REST call, not per-symbol - see
    # config.FUNDING_RATE_ENABLED)
    # =========================
    def _start_funding_poll(self):
        if not config.FUNDING_RATE_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._funding_poll_loop,
            args=(generation,),
            name="realtime-funding-poll",
            daemon=True,
        )
        thread.start()
        self.funding_poll_thread = thread

    def _funding_poll_loop(self, generation):
        """Refreshes self.funding_rates from a single bulk call covering
        every symbol (get_funding_rates, already weight-throttled) - same
        shape as _volume_poll_loop, no per-symbol pacing needed."""
        interval = max(float(config.FUNDING_POLL_INTERVAL_SECONDS), 30)

        while self._worker_active(generation):
            funding_rates = get_funding_rates()

            if funding_rates:
                with self.lock:
                    self.funding_rates = funding_rates

            if self.stop_event.wait(interval):
                return

    # =========================
    # LIQUIDATIONS (combined `!forceOrder@arr` stream - every symbol on
    # one connection, no chunking needed)
    # =========================
    def _start_liquidation_stream(self):
        if not config.LIQUIDATION_CONFIRMATION_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._liquidation_stream_loop,
            args=(generation,),
            name="realtime-liquidation-stream",
            daemon=True,
        )
        thread.start()
        self.liquidation_thread = thread

    def _liquidation_stream_loop(self, generation):
        from websockets.sync.client import connect

        url = f"{_market_stream_base()}{LIQUIDATION_STREAM_NAME}"

        while self._worker_active(generation):
            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.liquidation_websocket = websocket

                    while self._worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self.liquidations.handle_message(json.loads(message))

            except Exception as exc:
                if self._worker_active(generation):
                    log_warning(f"Realtime liquidation websocket reconnecting: {exc}")
                    self._reconnect_backoff_wait()
            finally:
                with self.lock:
                    if self.liquidation_websocket is not None:
                        self.liquidation_websocket = None

    # =========================
    # CROSS-EXCHANGE LIQUIDATIONS (Bybit + OKX - see config.CROSS_EXCHANGE_
    # LIQUIDATION_TRACKING_ENABLED)
    # =========================
    def _start_liquidation_stream_bybit(self):
        if not config.CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._liquidation_stream_loop_bybit,
            args=(generation,),
            name="realtime-liquidation-stream-bybit",
            daemon=True,
        )
        thread.start()
        self.liquidation_thread_bybit = thread

    def _bybit_liquidation_subscribe(self, websocket, generation):
        """Sends the subscribe frame, stripping any symbol Bybit rejects
        and retrying, until it accepts the remainder or symbols run out.
        Confirmed live (2026-08-29): Bybit fails the ENTIRE batch if even
        one symbol has no liquidation handler (~5% of this bot's watchlist,
        e.g. BTCDOMUSDT/1000SHIBUSDT - not a per-connection topic-count
        problem, all ~400 symbols fit in one connection fine once the
        unsupported ones are excluded). Re-discovers the same unsupported
        symbols from scratch on every reconnect rather than persisting a
        cooldown set across reconnects - reconnects are infrequent and
        each retry round-trip is sub-second, so the rediscovery cost is
        negligible; simpler than adding OI-style unavailable-symbol state
        for a "permanently unsupported" condition that isn't really a
        transient rate-limit/network failure. Returns True once
        subscribed, False if the connection should be abandoned."""
        symbols = list(self.symbols)

        while symbols:
            if not self._worker_active(generation):
                return False

            websocket.send(json.dumps(cross_exchange_liquidation.bybit_subscribe_frame(symbols)))

            try:
                reply = json.loads(websocket.recv(timeout=10))
            except TimeoutError:
                log_warning("Bybit liquidation subscribe timed out waiting for ack")
                return False

            if reply.get("success"):
                return True

            rejected = cross_exchange_liquidation.bybit_parse_rejected_symbol(reply)

            if rejected is None:
                log_warning(f"Bybit liquidation subscribe failed, unrecognized reply: {reply}")
                return False

            symbols = [symbol for symbol in symbols if symbol != rejected]

        return False

    def _liquidation_stream_loop_bybit(self, generation):
        from websockets.sync.client import connect

        while self._worker_active(generation):
            try:
                with connect(
                    cross_exchange_liquidation.BYBIT_WS_URL,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.liquidation_websocket_bybit = websocket

                    if not self._bybit_liquidation_subscribe(websocket, generation):
                        continue

                    while self._worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        parsed = cross_exchange_liquidation.parse_bybit_liquidation(json.loads(message))

                        if parsed is not None:
                            self.liquidations_bybit.record_liquidation(*parsed)

            except Exception as exc:
                if self._worker_active(generation):
                    log_warning(f"Realtime Bybit liquidation websocket reconnecting: {exc}")
                    self._reconnect_backoff_wait()
            finally:
                with self.lock:
                    if self.liquidation_websocket_bybit is not None:
                        self.liquidation_websocket_bybit = None

    def _start_liquidation_stream_okx(self):
        if not config.CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED:
            return

        with self.lock:
            generation = self.generation

        thread = threading.Thread(
            target=self._liquidation_stream_loop_okx,
            args=(generation,),
            name="realtime-liquidation-stream-okx",
            daemon=True,
        )
        thread.start()
        self.liquidation_thread_okx = thread

    def _liquidation_stream_loop_okx(self, generation):
        from websockets.sync.client import connect

        while self._worker_active(generation):
            try:
                with connect(
                    cross_exchange_liquidation.OKX_WS_URL,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.liquidation_websocket_okx = websocket

                    websocket.send(json.dumps(cross_exchange_liquidation.okx_subscribe_frame()))

                    while self._worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        for parsed in cross_exchange_liquidation.parse_okx_liquidation(json.loads(message)):
                            self.liquidations_okx.record_liquidation(*parsed)

            except Exception as exc:
                if self._worker_active(generation):
                    log_warning(f"Realtime OKX liquidation websocket reconnecting: {exc}")
                    self._reconnect_backoff_wait()
            finally:
                with self.lock:
                    if self.liquidation_websocket_okx is not None:
                        self.liquidation_websocket_okx = None
