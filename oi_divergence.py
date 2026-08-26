"""Open-interest divergence: price's swing structure compared against
OI's value at those same points in time - a new price extreme that isn't
backed by expanding OI (fresh positioning) is weaker evidence than one
where OI genuinely built up alongside it. Same divergence concept as
cvd_divergence.py, applied to a different metric with a different
alignment strategy: OI is a periodically-POLLED series
(config.OI_POLL_INTERVAL_SECONDS), not one sample per candle close, so
matching a swing to an OI reading uses the nearest sample AT OR BEFORE
that time instead of an exact-timestamp dict lookup.

Real unit mismatch to get right: a candle's open_time (Binance kline "t")
is milliseconds since epoch; open_interest.OpenInterestEngine records its
own samples against time.time() - SECONDS since epoch. Every open_time
here is converted to seconds (/1000) before comparing against oi_history
timestamps - comparing them raw would silently misalign every lookup
(a millisecond timestamp is ~1000x any real second-based one, so the
"at or before" search would effectively always return the newest sample
regardless of which swing it was supposedly matched to).
"""
import config


def _oi_at_or_before(oi_history, target_time_seconds):
    """oi_history: open_interest.OpenInterestEngine.history(symbol)
    output - (timestamp, oi_value) tuples in ascending time order (the
    engine's own append-only order). Returns the value of the latest
    sample at or before target_time_seconds, or None if every sample is
    newer (no OI data existed yet at that point)."""
    value = None

    for timestamp, oi_value in oi_history:
        if timestamp > target_time_seconds:
            break

        value = oi_value

    return value


def detect_divergence(swings, oi_history, min_delta_pct=None):
    """swings: market_structure.find_swing_points() output (unfiltered
    fractal swings). oi_history: open_interest.OpenInterestEngine.
    history(symbol) output.

    Both directions use the SAME underlying condition - OI declining
    across the compared swing pair - unlike cvd_divergence.py's
    direction-specific comparison: a falling OI means positions are being
    closed regardless of which way price just moved, so it can't be
    backing either a fresh breakout up OR down.

    Bearish divergence: the most recent swing HIGH is higher than the one
    before it (price still making new highs), but OI fell by at least
    min_delta_pct over that same span - the rally isn't backed by fresh
    long positioning (likely short-covering only) -> SELL.

    Bullish divergence: the mirror, on swing LOWs -> BUY (the decline
    isn't backed by fresh short positioning).

    Returns the more recently formed of the two (by swing index) if both
    happen to qualify at once, or None."""
    if not swings or not oi_history:
        return None

    min_delta = max(
        float(config.OI_DIVERGENCE_MIN_DELTA_PCT if min_delta_pct is None else min_delta_pct),
        0,
    )

    highs = sorted((s for s in swings if s.kind == "HIGH"), key=lambda s: s.index)
    lows = sorted((s for s in swings if s.kind == "LOW"), key=lambda s: s.index)

    candidates = []

    if len(highs) >= 2:
        prev_high, last_high = highs[-2], highs[-1]
        prev_oi = _oi_at_or_before(oi_history, prev_high.open_time / 1000)
        last_oi = _oi_at_or_before(oi_history, last_high.open_time / 1000)

        if (
            prev_oi is not None and last_oi is not None and prev_oi > 0
            and last_high.price > prev_high.price
            and ((prev_oi - last_oi) / prev_oi * 100) >= min_delta
        ):
            candidates.append({
                "direction": "BEARISH",
                "level": last_high.price,
                "index": last_high.index,
                "open_time": last_high.open_time,
            })

    if len(lows) >= 2:
        prev_low, last_low = lows[-2], lows[-1]
        prev_oi = _oi_at_or_before(oi_history, prev_low.open_time / 1000)
        last_oi = _oi_at_or_before(oi_history, last_low.open_time / 1000)

        if (
            prev_oi is not None and last_oi is not None and prev_oi > 0
            and last_low.price < prev_low.price
            and ((prev_oi - last_oi) / prev_oi * 100) >= min_delta
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


def diagnostic_candidates(swings, oi_history):
    """Read-only diagnostic - same shape and motivation as cvd_divergence.
    diagnostic_candidates (see config.OI_DIVERGENCE_DIAGNOSTIC_LOGGING_
    ENABLED's own comment): detect_divergence only ever reports a
    CONFIRMED divergence, so it can't explain its own silence on its own.
    Exposes every real structural candidate (price making a new extreme)
    regardless of whether OI data was found at both swing points or
    whether the delta clears min_delta_pct. Kept separate from
    detect_divergence so the real trigger path is untouched; mirrors its
    comparison logic exactly.

    Returns a list of 0-2 dicts: {"structural_direction": "BEARISH"/
    "BULLISH", "oi_data_found": bool, "delta_pct": float or None}."""
    if not swings or not oi_history:
        return []

    highs = sorted((s for s in swings if s.kind == "HIGH"), key=lambda s: s.index)
    lows = sorted((s for s in swings if s.kind == "LOW"), key=lambda s: s.index)

    out = []

    if len(highs) >= 2:
        prev_high, last_high = highs[-2], highs[-1]

        if last_high.price > prev_high.price:
            prev_oi = _oi_at_or_before(oi_history, prev_high.open_time / 1000)
            last_oi = _oi_at_or_before(oi_history, last_high.open_time / 1000)
            data_found = prev_oi is not None and last_oi is not None and prev_oi > 0
            out.append({
                "structural_direction": "BEARISH",
                "oi_data_found": data_found,
                "delta_pct": ((prev_oi - last_oi) / prev_oi * 100) if data_found else None,
            })

    if len(lows) >= 2:
        prev_low, last_low = lows[-2], lows[-1]

        if last_low.price < prev_low.price:
            prev_oi = _oi_at_or_before(oi_history, prev_low.open_time / 1000)
            last_oi = _oi_at_or_before(oi_history, last_low.open_time / 1000)
            data_found = prev_oi is not None and last_oi is not None and prev_oi > 0
            out.append({
                "structural_direction": "BULLISH",
                "oi_data_found": data_found,
                "delta_pct": ((prev_oi - last_oi) / prev_oi * 100) if data_found else None,
            })

    return out
