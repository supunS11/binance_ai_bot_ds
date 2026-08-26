"""Backtest replacement for ws_client.RealtimeMarketData - duck-types the
exact interface main._evaluate_symbol/_poll_positions read from a live feed
(`.candles`, `.htf_candles`, `.cvd`, `.depth`, `.open_interest`,
`.open_interest_bybit`, `.open_interest_okx`, `.volume_profile`,
`.liquidations`, `.volumes`, `.funding_rates`), but built from pre-fetched
historical data and advanced one closed candle at a time instead of a live
websocket.

Candles and CVD reuse the real, unmodified `ws_client.CandleStore` and
`order_flow.CVDEngine` classes - the same code the live bot runs, just fed
historical instead of streaming data, so a backtest validates the actual
production candle/CVD logic rather than a reimplementation of it.

Depth imbalance, open interest, and liquidations have no faithful
historical-replay source (see exchange.py's HISTORICAL DATA section
docstring) - signal_engine.evaluate() already treats all three as optional/
informational (never gates a signal on their own), so a stub "unavailable"
snapshot is an honest degrade, not a workaround: a backtest without them is
provably *more permissive* on those specific checks, never silently wrong.
"""
import config
from order_flow import CVDEngine
from ws_client import CandleStore


class _UnavailableSnapshotSource:
    """Stand-in for DepthImbalanceEngine/OpenInterestEngine/LiquidationEngine
    - always reports no data, the same shape those real engines return
    before they've ever recorded anything live. signal_engine.evaluate()
    already treats `"available": False` as a pure no-op for all three."""

    def snapshot(self, symbol, *args, **kwargs):
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

    def seed_htf(self, symbol, klines_df):
        self.htf_candles.seed(symbol, klines_df)

    def push_ltf_candle(self, symbol, candle):
        """One replay step forward - `candle` must already be closed=True;
        v1 only replays closed candles (see backtest.py's module docstring
        for the documented SIGNAL_CONFIRM_TICKS fidelity tradeoff this
        implies)."""
        self.candles.update(symbol, candle)

    def push_htf_candle(self, symbol, candle):
        self.htf_candles.update(symbol, candle)

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
