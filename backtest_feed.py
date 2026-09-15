"""Backtest replacement for ws_client.RealtimeMarketData - duck-types the
exact interface main._evaluate_symbol/_poll_positions read from a live feed
(`.candles`, `.htf_candles`, `.cvd`, `.depth`, `.open_interest`,
`.open_interest_bybit`, `.open_interest_okx`, `.volume_profile`,
`.liquidations`, `.liquidations_bybit`, `.liquidations_okx`,
`.crash_detector`, `.trend_candles`, `.htf_trend_candles`, `.volumes`,
`.funding_rates`), but built from pre-fetched historical data and
advanced one closed candle at a time instead of a live websocket.

Candles and CVD reuse the real, unmodified `ws_client.CandleStore` and
`order_flow.CVDEngine` classes - the same code the live bot runs, just fed
historical instead of streaming data, so a backtest validates the actual
production candle/CVD logic rather than a reimplementation of it.

Depth imbalance, open interest, and (same-exchange and cross-exchange)
liquidations have no faithful historical-replay source (see exchange.py's
HISTORICAL DATA section docstring) - signal_engine.evaluate() already
treats all of them as optional/informational (never gates a signal on their
own), so a stub "unavailable" snapshot is an honest degrade, not a
workaround: a backtest without them is provably *more permissive* on those
specific checks, never silently wrong. This also means config.
LIQUIDATION_HEATMAP_SL_TP_ENABLED/LIQUIDATION_HEATMAP_SWEEP_ENABLED are
no-ops in a backtest - signal["liquidation_pools"] has no historical-replay
source either and is always [] here, same shape as the rest of this list.
"""
import config
from order_flow import CVDEngine
from ws_client import CandleStore


class _UnavailableSnapshotSource:
    """Stand-in for DepthImbalanceEngine/OpenInterestEngine/LiquidationEngine/
    CrashDetector - always reports no data, the same shape those real
    engines return before they've ever recorded anything live.
    signal_engine.evaluate()/position_manager.py already treat
    `"available": False` as a pure no-op for all of them. `symbol` is
    optional (not just extra *args) because CrashDetector.snapshot() is
    the one caller that takes none at all (main.py's `feed.crash_
    detector.snapshot()`, no symbol - crash detection is market-wide, off
    a single reference symbol, not per-symbol like the others)."""

    def snapshot(self, symbol=None, *args, **kwargs):
        return {"available": False}


class BacktestFeed:
    def __init__(self, ltf_history_limit=None, htf_history_limit=None):
        self.candles = CandleStore(maxlen=ltf_history_limit or config.WS_KLINE_HISTORY_LIMIT)
        self.htf_candles = CandleStore(maxlen=htf_history_limit or config.HTF_KLINE_HISTORY_LIMIT)
        self.cvd = CVDEngine()
        self.depth = _UnavailableSnapshotSource()
        self.open_interest = _UnavailableSnapshotSource()
        # Cross-exchange OI and volume-profile have no faithful
        # historical-replay source either (same rationale as depth/OI/
        # liquidations above) - same stub, same honest degrade.
        self.open_interest_bybit = _UnavailableSnapshotSource()
        self.open_interest_okx = _UnavailableSnapshotSource()
        self.volume_profile = _UnavailableSnapshotSource()
        self.liquidations = _UnavailableSnapshotSource()
        # config.CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED - same no-
        # faithful-replay-source rationale as open_interest_bybit/_okx
        # above. Real gap found live (2026-09-15): main._evaluate_symbol
        # reads feed.liquidations_bybit/feed.liquidations_okx
        # unconditionally (main.py, right after feed.liquidations) - this
        # class never grew these two attributes when cross_exchange_
        # liquidation.py was added, so every backtest run crashed with
        # AttributeError before ever reaching signal_engine.evaluate().
        self.liquidations_bybit = _UnavailableSnapshotSource()
        self.liquidations_okx = _UnavailableSnapshotSource()
        # config.CRASH_DETECTOR_ENABLED - real gap found live (2026-09-15,
        # same audit as liquidations_bybit/_okx above): main._evaluate_symbol
        # AND main._poll_positions both read feed.crash_detector.snapshot()
        # unconditionally. CrashDetector itself only needs real-time trade
        # prices off CRASH_DETECTOR_REFERENCE_SYMBOL (BTCUSDT by default) -
        # in principle replayable - but only if that symbol's own aggTrades
        # were fetched, which _run_symbol_backtest only does for whichever
        # symbol is currently being backtested. Stubbed unavailable rather
        # than half-wiring a reference-symbol-only replay path: same
        # "strictly more permissive, never silently wrong" shape as every
        # other stub here (CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED simply never
        # fires in a backtest).
        self.crash_detector = _UnavailableSnapshotSource()
        # config.EMA_TREND_MIXED_REJECT_ENABLED - a real, live reject gate
        # (not informational), but its inputs ARE faithfully replayable -
        # same real klines as .candles/.htf_candles, just a separate,
        # deeper CandleStore (ws_client.py's own __init__ keeps these
        # distinct so EMA_TREND_HISTORY_LIMIT can be tuned independently
        # of WS_KLINE_HISTORY_LIMIT/HTF_KLINE_HISTORY_LIMIT - see that
        # flag's own comment). Real gap found live (2026-09-15, same audit
        # as liquidations_bybit/_okx/crash_detector above): main.
        # _evaluate_symbol reads feed.trend_candles/feed.htf_trend_candles
        # unconditionally. Updated in lockstep with the main stores below
        # (push_ltf_candle/push_htf_candle), mirroring ws_client._handle_
        # kline exactly - this is a real replay, not a stub, so EMA_TREND_
        # MIXED_REJECT_ENABLED behaves faithfully in a backtest. One
        # honest fidelity gap: seeded from the same priming window
        # backtest.py fetches for the shallower main stores (sized off
        # WS_KLINE_HISTORY_LIMIT/HTF_KLINE_HISTORY_LIMIT, not the deeper
        # EMA_TREND_HISTORY_LIMIT), so early in a long backtest this
        # buffer is real but shallower than live's - never wrong data,
        # just a shorter warm-up.
        self.trend_candles = CandleStore(maxlen=config.EMA_TREND_HISTORY_LIMIT)
        self.htf_trend_candles = CandleStore(maxlen=config.EMA_TREND_HISTORY_LIMIT)
        # Plain dicts - matches how main._evaluate_symbol reads a live feed's
        # own .volumes/.funding_rates (feed.volumes.get(symbol), not a
        # snapshot() call), see main.py:163/165.
        self.volumes = {}
        self.funding_rates = {}

    def seed_ltf(self, symbol, klines_df):
        """Bulk-prime history before replay starts, same as ws_client's own
        startup seed - every seeded candle is `closed=True` (CandleStore.seed
        already enforces this)."""
        self.candles.seed(symbol, klines_df)
        self.trend_candles.seed(symbol, klines_df)

    def seed_htf(self, symbol, klines_df):
        self.htf_candles.seed(symbol, klines_df)
        self.htf_trend_candles.seed(symbol, klines_df)

    def push_ltf_candle(self, symbol, candle):
        """One replay step forward - `candle` must already be closed=True;
        v1 only replays closed candles (see backtest.py's module docstring
        for the documented SIGNAL_CONFIRM_TICKS fidelity tradeoff this
        implies)."""
        self.candles.update(symbol, candle)
        self.trend_candles.update(symbol, candle)

    def push_htf_candle(self, symbol, candle):
        self.htf_candles.update(symbol, candle)
        self.htf_trend_candles.update(symbol, candle)

    def replay_trades(self, symbol, trades):
        """`trades`: iterable of (timestamp_seconds, price, quantity,
        is_buyer_maker) tuples, in chronological order - the same shape
        order_flow.CVDEngine.record_trade() already takes, just historical
        instead of live. Call with exactly the trades due up to the current
        replay point before evaluating that point; CVDEngine has no
        "already recorded" guard, so replaying the same trade twice would
        double-count it."""
        for timestamp, price, quantity, is_buyer_maker in trades:
            self.cvd.record_trade(symbol, price, quantity, is_buyer_maker, timestamp=timestamp)

    def set_quote_volume(self, symbol, value):
        self.volumes[symbol] = value

    def set_funding_rate(self, symbol, value):
        self.funding_rates[symbol] = value
