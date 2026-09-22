"""Real-time liquidity sweep / stop-hunt detector.

A sweep is price wicking through a marked liquidity pool (equal highs/lows
from market_structure.find_liquidity_pools) and rejecting back inside
before/at candle close - the classic "run the stops, then reverse"
pattern.
"""
import config


def detect_sweep(
    candles, pools, require_closed_candle=None,
    min_wick_atr_multiple=None, atr=None, select_nearest=None,
):
    """By default (require_closed_candle=False) this checks the *current,
    possibly still-forming* candle - real-time, catching the sweep as it
    happens rather than after the candle closes.

    When require_closed_candle is True (config.REQUIRE_CLOSE_CONFIRMED_BREAK
    - the same flag market_structure.live_break_check uses, reused here
    rather than a new one, since it's the same principle applied
    uniformly), this instead checks the most recently CLOSED candle. Real
    motivation (2026-08-13, two live trades traced against actual Binance
    price history): both entered on a still-forming candle's wick-and-
    reject that hadn't actually held by the time the candle finished
    forming - price resumed the original move and hit SL within minutes.
    Deliberately does NOT just test `candles[-1]["closed"]` - see
    live_break_check's docstring for why that would almost never fire.

    min_wick_atr_multiple (config.LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE)
    requires the penetration to be a real one. wick_size was computed on
    every sweep from the beginning and never read by anything, so until
    2026-09-22 a one-tick poke through a pool scored identically to a
    genuine stop run. Needs `atr` to scale with the symbol's own
    volatility; fails open (no requirement) when either is missing or zero,
    same convention as every other read in this engine.

    select_nearest (config.LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED)
    changes WHICH swept pool is returned when a candle sweeps more than
    one. The original behaviour returns the first in list order, and
    find_liquidity_pools emits every BUY_SIDE pool before every SELL_SIDE
    one - so on a candle that swept both sides the BEARISH read won purely
    by enumeration order, never on merit. Default False keeps that
    behaviour byte-for-byte."""
    if not candles or not pools:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if min_wick_atr_multiple is None:
        min_wick_atr_multiple = float(config.LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE)

    if select_nearest is None:
        select_nearest = config.LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED

    if require_closed_candle:
        closed_candles = [c for c in candles if c.get("closed")]

        if not closed_candles:
            return None

        latest = closed_candles[-1]
    else:
        latest = candles[-1]

    high = latest["high"]
    low = latest["low"]
    close = latest["close"]
    # Fails open on a missing/zero ATR or a zero threshold - the sweep is
    # still a sweep, we just cannot judge its size.
    min_wick = max(float(min_wick_atr_multiple), 0) * max(float(atr or 0), 0)
    hits = []

    for pool in pools:
        level = pool["price"]

        if pool["type"] == "BUY_SIDE" and high > level and close < level:
            # Swept buy-side liquidity (stops above a high) and rejected
            # back down - bearish signal.
            hits.append({
                "direction": "BEARISH",
                "level": level,
                "wick_size": high - level,
                "pool": pool,
                "open_time": latest["open_time"],
            })
        elif pool["type"] == "SELL_SIDE" and low < level and close > level:
            # Swept sell-side liquidity (stops below a low) and rejected
            # back up - bullish signal.
            hits.append({
                "direction": "BULLISH",
                "level": level,
                "wick_size": level - low,
                "pool": pool,
                "open_time": latest["open_time"],
            })

    if min_wick > 0:
        hits = [h for h in hits if h["wick_size"] >= min_wick]

    if not hits:
        return None

    if select_nearest:
        return min(hits, key=lambda h: abs(h["level"] - close))

    return hits[0]


def detect_liquidation_confirmed_sweep(sweep, liquidation_snapshot, min_notional_usdt=None):
    """Promotes a plain liquidity sweep (detect_sweep above) into a
    stricter, distinct trigger by additionally requiring a REAL clustered
    forced-liquidation event backing it - not just the informational-only
    liquidation_aligned/liquidation_cluster fields signal_engine already
    journals for every trigger, but a genuine gating condition specific to
    this one: the swept level actually forced real positions closed in
    the sweep's direction, not just a wick that happened to tag a pool.
    Distinct from LIQUIDITY_SWEEP - a sweep alone is sufficient there;
    this only fires on the strict subset of sweeps that also have real
    liquidation flow behind them, so it can only ever be MORE selective,
    never a relaxation.

    Alignment check mirrors signal_engine.py's own liquidation_aligned
    formula exactly (net_liquidation_notional > 0 for a BULLISH sweep -
    forced-SELL long liquidations below market, consistent with a genuine
    stop-run flush; < 0 for BEARISH), not a new definition."""
    if sweep is None or not liquidation_snapshot or not liquidation_snapshot.get("available"):
        return None

    min_notional = max(
        float(
            config.LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT
            if min_notional_usdt is None else min_notional_usdt
        ),
        0,
    )
    total_notional = (
        liquidation_snapshot.get("long_liquidation_notional", 0)
        + liquidation_snapshot.get("short_liquidation_notional", 0)
    )

    if total_notional < min_notional:
        return None

    net = liquidation_snapshot.get("net_liquidation_notional")

    if net is None:
        return None

    aligned = net > 0 if sweep["direction"] == "BULLISH" else net < 0

    if not aligned:
        return None

    return dict(sweep)
