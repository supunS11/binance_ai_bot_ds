"""CVD/order-flow divergence: price's swing structure compared against
the CVD line's value at those same swing points - absorption when price
makes a new extreme but the aggressor flow behind it doesn't confirm it.

Genuinely different data than signal_engine's existing SIGNAL_MIN_CVD_SCORE
gate: that's a recent-window (1m/5m/15m) snapshot, blind to price's own
swing history. This compares CVD's value at two specific points in time
the same way price structure itself is compared (the same raw fractal
swing points STRUCTURE_BREAK/find_liquidity_pools already use) - the
classic technical-analysis sense of "divergence".
"""
import config


def detect_divergence(swings, cvd_history, min_delta_usdt=None):
    """swings: market_structure.find_swing_points() output (unfiltered
    fractal swings, NOT zigzagged). cvd_history: order_flow.CVDEngine.
    cvd_history(symbol) output - a list of {"open_time", "cumulative_cvd"}
    dicts, one per LTF candle close.

    Bullish divergence: the most recent swing LOW is lower than the one
    before it (price still making new lows), but CVD's value at that low
    is HIGHER than at the prior low by at least min_delta_usdt - selling
    pressure did not follow price down (absorption) -> BUY.

    Bearish divergence: the mirror, on swing HIGHs -> SELL.

    Returns the more recently formed of the two (by swing index) if both
    happen to qualify at once, or None."""
    if not swings or not cvd_history:
        return None

    min_delta = max(
        float(config.CVD_DIVERGENCE_MIN_DELTA_USDT if min_delta_usdt is None else min_delta_usdt),
        0,
    )
    cvd_by_time = {point["open_time"]: point["cumulative_cvd"] for point in cvd_history}

    highs = sorted((s for s in swings if s.kind == "HIGH"), key=lambda s: s.index)
    lows = sorted((s for s in swings if s.kind == "LOW"), key=lambda s: s.index)

    candidates = []

    if len(highs) >= 2:
        prev_high, last_high = highs[-2], highs[-1]
        prev_cvd = cvd_by_time.get(prev_high.open_time)
        last_cvd = cvd_by_time.get(last_high.open_time)

        if (
            prev_cvd is not None and last_cvd is not None
            and last_high.price > prev_high.price
            and (prev_cvd - last_cvd) >= min_delta
        ):
            candidates.append({
                "direction": "BEARISH",
                "level": last_high.price,
                "index": last_high.index,
                "open_time": last_high.open_time,
            })

    if len(lows) >= 2:
        prev_low, last_low = lows[-2], lows[-1]
        prev_cvd = cvd_by_time.get(prev_low.open_time)
        last_cvd = cvd_by_time.get(last_low.open_time)

        if (
            prev_cvd is not None and last_cvd is not None
            and last_low.price < prev_low.price
            and (last_cvd - prev_cvd) >= min_delta
        ):
            candidates.append({
                "direction": "BULLISH",
                "level": last_low.price,
                "index": last_low.index,
                "open_time": last_low.open_time,
            })

    if not candidates:
        return None

    return max(candidates, key=lambda c: c["index"])
