"""ICT/SMC market-structure engine: swing points, BOS/CHoCH, order blocks,
fair value gaps, liquidity pools, and premium/discount + OTE zones.

Every function here is pure - a candle list in, structure out - so it runs
identically against closed history and a live/forming last candle. That's
what makes `live_break_check` meaningful: it re-evaluates against the
*current* candle on every websocket tick, not only once a candle closes.

Terminology (standard ICT/SMC):
- BOS (break of structure): price breaks a swing point in the direction of
  the prevailing trend - continuation.
- CHoCH (change of character): price breaks a swing point *against* the
  prevailing trend - the first warning of a reversal.
- Order block: the last opposite-colour candle before an impulsive move
  that caused a structure break - presumed origin of the move.
- FVG (fair value gap): a 3-candle imbalance where candle 1's wick and
  candle 3's wick don't overlap.
- Buy-side / sell-side liquidity: clusters of equal highs / equal lows,
  where breakout-buy orders and long stops respectively tend to sit.
- Premium / discount: the top half / bottom half of the current dealing
  range, split at its midpoint; OTE is a deeper retracement zone within
  that half used to time entries.
"""
from dataclasses import dataclass

import config


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class SwingPoint:
    index: int
    open_time: int
    price: float
    kind: str  # "HIGH" or "LOW"


def find_swing_points(candles, left=None, right=None):
    """Fractal swing points: a HIGH at i where candles[i]'s high is the max
    over the window [i-left, i+right]; symmetric for LOW. The most recent
    `right` candles never produce a swing yet - there's nothing after them
    to compare against - which is the correct causal behaviour for live
    data (a swing is only confirmed once price has moved away from it)."""
    left = int(config.SWING_LEFT if left is None else left)
    right = int(config.SWING_RIGHT if right is None else right)
    swings = []
    n = len(candles)

    for i in range(left, n - right):
        window = candles[i - left:i + right + 1]
        high = candles[i]["high"]
        low = candles[i]["low"]

        if high == max(c["high"] for c in window):
            swings.append(SwingPoint(i, candles[i]["open_time"], high, "HIGH"))

        if low == min(c["low"] for c in window):
            swings.append(SwingPoint(i, candles[i]["open_time"], low, "LOW"))

    return swings


def _zigzag(swings):
    """Collapse consecutive same-kind swings, keeping the more extreme
    one, so the sequence alternates HIGH/LOW in time order."""
    filtered = []

    for swing in sorted(swings, key=lambda s: s.index):
        if filtered and filtered[-1].kind == swing.kind:
            if swing.kind == "HIGH" and swing.price >= filtered[-1].price:
                filtered[-1] = swing
            elif swing.kind == "LOW" and swing.price <= filtered[-1].price:
                filtered[-1] = swing
        else:
            filtered.append(swing)

    return filtered


def _classify_swings(swings):
    """Walk an already-alternating (post-zigzag) swing sequence and
    classify the current trend plus the most recent BOS/CHoCH event. A
    trend break is defined as the newest confirmed pivot exceeding the
    *previous* confirmed pivot in the same direction (a higher high, or a
    lower low). Split out from structure_state so the classification logic
    can be unit-tested against hand-built swing sequences directly.

    `events` (every BOS/CHoCH found along the way, not just the last one)
    backs ORDER_BLOCK_RETEST_TRIGGER_ENABLED's find_order_blocks below -
    an origin block from several swings ago is still a valid, unmitigated
    retest target, not just the very latest event structure_state's
    original last_event-only shape exposed."""
    if len(swings) < 2:
        return {"available": False}

    trend = None
    last_high = None
    last_low = None
    last_event = None
    events = []

    for swing in swings:
        if swing.kind == "HIGH":
            if last_high is not None and swing.price > last_high.price:
                event_type = (
                    "CHoCH" if trend == "BEARISH" else "BOS"
                )
                trend = "BULLISH"
                last_event = {
                    "type": event_type,
                    "direction": "BULLISH",
                    "index": swing.index,
                    "price": swing.price,
                }
                events.append(last_event)
            last_high = swing
        else:
            if last_low is not None and swing.price < last_low.price:
                event_type = (
                    "CHoCH" if trend == "BULLISH" else "BOS"
                )
                trend = "BEARISH"
                last_event = {
                    "type": event_type,
                    "direction": "BEARISH",
                    "index": swing.index,
                    "price": swing.price,
                }
                events.append(last_event)
            last_low = swing

    return {
        "available": True,
        "trend": trend,
        "last_event": last_event,
        "events": events,
        "last_swing_high": last_high.price if last_high else None,
        "last_swing_low": last_low.price if last_low else None,
        "swings": swings,
    }


def structure_state(candles, left=None, right=None):
    swings = _zigzag(find_swing_points(candles, left, right))
    return _classify_swings(swings)


def live_break_check(candles, structure, require_closed_candle=None):
    """Has a candle broken the last confirmed swing level?

    By default (require_closed_candle=False) this checks the *current,
    possibly still-forming* candle - the real-time advantage over waiting
    for a candle to close before reacting.

    When require_closed_candle is True (config.REQUIRE_CLOSE_CONFIRMED_BREAK),
    it instead checks the most recently CLOSED candle. This deliberately
    does NOT just test `candles[-1]["closed"]` - the live buffer typically
    replaces a just-closed candle with a new forming one within moments (see
    ws_client.CandleStore.update), so a naive "is the current last candle
    closed" check would rarely be true at an arbitrary eval tick and would
    almost never fire. Scanning back to the last candle with closed=True
    instead gives a stable answer regardless of exactly when between
    candles this function happens to be called."""
    if not structure.get("available") or not candles:
        return {"broken": False}

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if require_closed_candle:
        closed_candles = [c for c in candles if c.get("closed")]

        if not closed_candles:
            return {"broken": False}

        latest = closed_candles[-1]
    else:
        latest = candles[-1]

    close = latest["close"]
    last_high = structure.get("last_swing_high")
    last_low = structure.get("last_swing_low")

    broke_up = last_high is not None and close > last_high
    broke_down = last_low is not None and close < last_low

    return {
        "broken": bool(broke_up or broke_down),
        "direction": "BULLISH" if broke_up else "BEARISH" if broke_down else None,
        "level": last_high if broke_up else last_low if broke_down else None,
        "candle_closed": latest["closed"],
        "open_time": latest["open_time"],
    }


def find_order_block(candles, index, direction):
    """The order block for a bullish break is the last bearish (red)
    candle before the impulsive move up; for a bearish break, the last
    bullish (green) candle before the move down."""
    index = min(max(index, 0), len(candles) - 1)

    for i in range(index, max(index - 10, -1), -1):
        candle = candles[i]
        is_bullish_candle = candle["close"] > candle["open"]

        if direction == "BULLISH" and not is_bullish_candle:
            return {
                "index": i,
                "high": candle["high"],
                "low": candle["low"],
                "open_time": candle["open_time"],
            }

        if direction == "BEARISH" and is_bullish_candle:
            return {
                "index": i,
                "high": candle["high"],
                "low": candle["low"],
                "open_time": candle["open_time"],
            }

    return None


def find_structure_events(candles, left=None, right=None):
    """Every BOS/CHoCH event across the full swing sequence - structure_state
    only exposes the single most recent one (last_event). Needed to
    enumerate historical order blocks below (ORDER_BLOCK_RETEST_TRIGGER_
    ENABLED): an origin block from several confirmed breaks ago is still
    a valid, unmitigated retest target, not just the latest one. Recomputes
    the same zigzag+classify walk structure_state does rather than
    threading a new parameter through it - same "only pay for it when the
    flag needing it is on" convention as LIQUIDITY_SWEEP_TRIGGER_ENABLED's
    own pools/swings recompute."""
    return _classify_swings(_zigzag(find_swing_points(candles, left, right))).get("events", [])


def find_order_blocks(candles, left=None, right=None, max_events=None):
    """Every historical order block whose origin was a REAL confirmed
    structure break (BOS/CHoCH) - not a heuristic guess at "impulsive
    candle", the same real definition find_order_block already uses for
    the live REQUIRE_ORDER_BLOCK_OR_FVG gate, just enumerated across the
    most recent max_events past breaks instead of only the very latest.
    Mirrors find_fair_value_gaps's flat-list shape (direction/high/low/
    index/open_time) so find_order_block_retest below can scan it the
    same way find_fvg_retest scans fair_value_gaps."""
    max_events = int(
        config.ORDER_BLOCK_RETEST_LOOKBACK_EVENTS if max_events is None else max_events
    )
    events = find_structure_events(candles, left, right)
    blocks = []

    for event in events[-max_events:] if max_events > 0 else []:
        block = find_order_block(candles, event["index"], event["direction"])

        if block is not None:
            blocks.append({
                "direction": event["direction"],
                "high": block["high"],
                "low": block["low"],
                "index": block["index"],
                "open_time": block["open_time"],
            })

    return blocks


def find_order_block_retest(candles, blocks=None, max_age_candles=None, require_closed_candle=None):
    """A fresh rejection wick back into a previously-formed, UNMITIGATED
    order block - the retest counterpart to find_fvg_retest, but for
    order blocks instead of fair value gaps (deliberately deferred when
    OB_FVG_RETEST_TRIGGER_ENABLED was first built - see that setting's
    config.py comment: it needed exactly this forward-scanning variant of
    find_order_block, more engineering than the flat FVG list needed at
    the time). "Unmitigated" mirrors find_fvg_retest exactly: no candle
    strictly between the block's origin index and the tested candle has
    already CLOSED fully through it (a wick through doesn't invalidate
    it - only a close past the far edge does).

    By default (require_closed_candle=False) this tests the current,
    possibly still-forming candle. When True (config.
    REQUIRE_CLOSE_CONFIRMED_BREAK, reused here as the same principle
    applied uniformly), scans back to the most recently CLOSED candle
    instead. Returns the most recently formed qualifying block's retest
    (direction/level/block/open_time - the candle actually tested), or
    None."""
    if len(candles) < 2:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if require_closed_candle:
        closed_candles = [(i, c) for i, c in enumerate(candles) if c.get("closed")]

        if not closed_candles:
            return None

        latest_index, latest = closed_candles[-1]
    else:
        latest_index = len(candles) - 1
        latest = candles[latest_index]

    blocks = find_order_blocks(candles) if blocks is None else blocks
    max_age = int(
        config.ORDER_BLOCK_RETEST_MAX_AGE_CANDLES if max_age_candles is None else max_age_candles
    )

    for block in sorted(blocks, key=lambda b: b["index"], reverse=True):
        if block["index"] >= latest_index or (latest_index - block["index"]) > max_age:
            continue

        high, low = block["high"], block["low"]
        mitigated = any(
            (block["direction"] == "BULLISH" and candles[i]["close"] < low)
            or (block["direction"] == "BEARISH" and candles[i]["close"] > high)
            for i in range(block["index"] + 1, latest_index)
        )

        if mitigated:
            continue

        if block["direction"] == "BULLISH" and latest["low"] <= high and latest["close"] > low:
            return {"direction": "BULLISH", "level": low, "block": block, "open_time": latest["open_time"]}

        if block["direction"] == "BEARISH" and latest["high"] >= low and latest["close"] < high:
            return {"direction": "BEARISH", "level": high, "block": block, "open_time": latest["open_time"]}

    return None


def detect_ema_pullback(candles, ema_value, require_closed_candle=None):
    """A pullback to the EMA within an established trend, followed by a
    same-candle reclaim - the classic trend-continuation entry, well
    suited to smooth, high-liquidity trending symbols (majors) that
    rarely produce the deep OTE retracement or CVD/depth imbalance every
    other trigger's downstream gate was tuned around (see config.
    EMA_PULLBACK_TRIGGER_ENABLED for the real evidence: BTC/ETH/BNB/SOL
    produced ZERO signal-related log activity across a full session with
    all 8 other triggers live).

    Bullish: the tested candle's LOW touches or pierces ema_value (a
    real pullback TO it, not just proximity) but its CLOSE reclaims back
    above - the dip held, the trend likely continues up.
    Bearish: the mirror - HIGH touches/pierces, CLOSE stays below.

    ema_value is the CURRENT ema (market_structure.
    exponential_moving_average) - a slow-moving rolling average, so
    using "now"'s value as a stand-in for "the EMA at the tested
    candle's close" is a reasonable approximation, the same cost/
    precision tradeoff every other trigger's shared-computation hoisting
    in signal_engine.py already makes.

    By default (require_closed_candle=False) tests the current, possibly
    still-forming candle. When True (config.REQUIRE_CLOSE_CONFIRMED_BREAK,
    reused here - the same principle applied uniformly across every
    trigger), scans back to the most recently CLOSED candle instead -
    same real motivation as every other close-confirmed trigger this
    session (a wick-and-reclaim read on a still-forming candle can flip
    before the candle actually finishes). Returns {"direction", "level",
    "open_time"} or None."""
    if not candles or ema_value is None:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if require_closed_candle:
        closed_candles = [c for c in candles if c.get("closed")]

        if not closed_candles:
            return None

        latest = closed_candles[-1]
    else:
        latest = candles[-1]

    high, low, close = latest["high"], latest["low"], latest["close"]

    if low <= ema_value and close > ema_value:
        return {"direction": "BULLISH", "level": ema_value, "open_time": latest["open_time"]}

    if high >= ema_value and close < ema_value:
        return {"direction": "BEARISH", "level": ema_value, "open_time": latest["open_time"]}

    return None


def find_fair_value_gaps(candles, lookback=None):
    lookback = int(config.FVG_LOOKBACK_CANDLES if lookback is None else lookback)
    gaps = []
    start = max(len(candles) - lookback, 2)

    for i in range(start, len(candles)):
        first = candles[i - 2]
        third = candles[i]

        if third["low"] > first["high"]:
            gaps.append({
                "type": "BULLISH",
                "top": third["low"],
                "bottom": first["high"],
                "index": i,
            })
        elif third["high"] < first["low"]:
            gaps.append({
                "type": "BEARISH",
                "top": first["low"],
                "bottom": third["high"],
                "index": i,
            })

    return gaps


def find_liquidity_pools(swings, tolerance_pct=None):
    """Cluster equal highs into BUY_SIDE liquidity (above the market -
    short stops + breakout buyers) and equal lows into SELL_SIDE liquidity
    (below the market - long stops), requiring at least 2 touches."""
    tolerance_pct = float(
        config.LIQUIDITY_POOL_TOLERANCE_PCT if tolerance_pct is None else tolerance_pct
    )
    pools = []

    for kind, label in (("HIGH", "BUY_SIDE"), ("LOW", "SELL_SIDE")):
        points = sorted(
            (s for s in swings if s.kind == kind),
            key=lambda s: s.price,
        )
        cluster = []

        for point in points:
            if cluster and abs(point.price - cluster[-1].price) / cluster[-1].price <= tolerance_pct:
                cluster.append(point)
                continue

            if len(cluster) >= 2:
                pools.append({
                    "type": label,
                    "price": sum(p.price for p in cluster) / len(cluster),
                    "touches": len(cluster),
                })

            cluster = [point]

        if len(cluster) >= 2:
            pools.append({
                "type": label,
                "price": sum(p.price for p in cluster) / len(cluster),
                "touches": len(cluster),
            })

    return pools


def find_fvg_retest(candles, fvgs=None, max_age_candles=None, require_closed_candle=None):
    """A fresh rejection wick into an UNMITIGATED fair value gap - the
    classic OB/FVG "retest" entry, independent of any live structure break
    right now. "Unmitigated" means no candle strictly between the gap's
    formation index and the tested candle has already CLOSED fully through
    the zone (a wick through doesn't invalidate it - only a close past the
    far edge does).

    By default (require_closed_candle=False) this tests the current,
    possibly still-forming candle. When require_closed_candle is True
    (config.REQUIRE_CLOSE_CONFIRMED_BREAK - the same flag live_break_check
    uses, reused here as the same principle applied uniformly), this scans
    back to the most recently CLOSED candle instead - same real motivation
    as detect_sweep's identical change: a live-candle wick-and-reject read
    that hadn't actually held by the time the candle finished forming.
    Returns the most recently formed qualifying gap (direction/level/gap/
    open_time - the candle actually tested), or None."""
    if len(candles) < 3:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if require_closed_candle:
        closed_candles = [(i, c) for i, c in enumerate(candles) if c.get("closed")]

        if not closed_candles:
            return None

        latest_index, latest = closed_candles[-1]
    else:
        latest_index = len(candles) - 1
        latest = candles[latest_index]

    fvgs = find_fair_value_gaps(candles) if fvgs is None else fvgs
    max_age = int(
        config.OB_FVG_RETEST_MAX_AGE_CANDLES if max_age_candles is None else max_age_candles
    )

    for gap in sorted(fvgs, key=lambda g: g["index"], reverse=True):
        if gap["index"] >= latest_index or (latest_index - gap["index"]) > max_age:
            continue

        top, bottom = gap["top"], gap["bottom"]
        mitigated = any(
            (gap["type"] == "BULLISH" and candles[i]["close"] < bottom)
            or (gap["type"] == "BEARISH" and candles[i]["close"] > top)
            for i in range(gap["index"] + 1, latest_index)
        )

        if mitigated:
            continue

        if gap["type"] == "BULLISH" and latest["low"] <= top and latest["close"] > bottom:
            return {"direction": "BULLISH", "level": bottom, "gap": gap, "open_time": latest["open_time"]}

        if gap["type"] == "BEARISH" and latest["high"] >= bottom and latest["close"] < top:
            return {"direction": "BEARISH", "level": top, "gap": gap, "open_time": latest["open_time"]}

    return None


def premium_discount_zone(candles, lookback=None):
    lookback = int(
        config.PREMIUM_DISCOUNT_LOOKBACK_CANDLES if lookback is None else lookback
    )
    window = candles[-lookback:] if len(candles) > lookback else candles

    if not window:
        return {"available": False}

    high = max(c["high"] for c in window)
    low = min(c["low"] for c in window)

    if high <= low:
        return {"available": False}

    midpoint = (high + low) / 2
    range_size = high - low
    ote_min = float(config.OTE_RETRACEMENT_MIN)
    ote_max = float(config.OTE_RETRACEMENT_MAX)

    return {
        "available": True,
        "range_high": high,
        "range_low": low,
        "midpoint": midpoint,
        # Bullish OTE: a deep pullback down into discount before continuing
        # up - measured as a retracement from the range high.
        "bullish_ote_zone": (
            high - range_size * ote_max,
            high - range_size * ote_min,
        ),
        # Bearish OTE: a deep pullback up into premium before continuing
        # down - measured as a retracement from the range low.
        "bearish_ote_zone": (
            low + range_size * ote_min,
            low + range_size * ote_max,
        ),
    }


def zone_for_price(zone, price):
    if not zone.get("available"):
        return None

    return "DISCOUNT" if price < zone["midpoint"] else "PREMIUM"


def in_ote(zone, price, direction):
    if not zone.get("available"):
        return False

    low, high = (
        zone["bullish_ote_zone"] if direction == "BULLISH" else zone["bearish_ote_zone"]
    )
    return low <= price <= high


def average_true_range(candles, period=None):
    period = int(config.ATR_PERIOD if period is None else period)

    if len(candles) < period + 1:
        return 0.0

    window = candles[-(period + 1):]
    true_ranges = []

    for i in range(1, len(window)):
        high = window[i]["high"]
        low = window[i]["low"]
        prev_close = window[i - 1]["close"]
        true_ranges.append(max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        ))

    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def exponential_moving_average(candles, period=None):
    """Standard EMA of closes - the same direction-confirmation concept
    as v7's EMA_WRONG_SIDE guard: a structure break that immediately sits
    on the wrong side of recent average price is a weaker signal than one
    that breaks and holds. Returns None with too little history rather
    than a misleading value seeded from a short window."""
    period = int(config.EMA_CONFIRMATION_PERIOD if period is None else period)

    if len(candles) < period:
        return None

    closes = [c["close"] for c in candles[-period * 3:]] if len(candles) > period * 3 else [c["close"] for c in candles]
    seed = sum(closes[:period]) / period
    multiplier = 2 / (period + 1)
    ema = seed

    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema

    return ema


def efficiency_ratio(candles, period=None):
    """Kaufman's Efficiency Ratio: net directional movement over the
    window divided by total path length (sum of each candle's absolute
    move). 1.0 means price moved in a straight line (strongly trending);
    near 0 means it round-tripped back and forth without going anywhere
    (chop). A structure break inside a low-ER market is weaker evidence
    than the same break inside a genuinely trending one - a "break" in a
    dead/choppy market is often just noise finding a random level, not
    real conviction. Returns None with too little history or a flat
    market (zero path length) rather than a misleading value."""
    period = int(config.CHOP_FILTER_LOOKBACK_CANDLES if period is None else period)

    if len(candles) < period + 1:
        return None

    window = candles[-(period + 1):]
    net_move = abs(window[-1]["close"] - window[0]["close"])
    path_length = sum(
        abs(window[i]["close"] - window[i - 1]["close"]) for i in range(1, len(window))
    )

    if path_length <= 0:
        return None

    return net_move / path_length


def price_correlation(candles_a, candles_b, period=None):
    """Pearson correlation between two symbols' closes over the same
    trailing window - how much of this symbol's move is just riding a
    reference symbol's (typically BTC) move, versus genuinely independent
    structure. -1..1, same convention as everywhere else this session.
    Returns None with too little overlapping history or a symbol whose
    price never moved in the window (zero variance) rather than a
    misleading value."""
    period = int(config.CORRELATION_LOOKBACK_CANDLES if period is None else period)

    if len(candles_a) < period or len(candles_b) < period:
        return None

    closes_a = [c["close"] for c in candles_a[-period:]]
    closes_b = [c["close"] for c in candles_b[-period:]]
    mean_a = sum(closes_a) / period
    mean_b = sum(closes_b) / period

    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(closes_a, closes_b))
    variance_a = sum((a - mean_a) ** 2 for a in closes_a)
    variance_b = sum((b - mean_b) ** 2 for b in closes_b)
    denominator = (variance_a * variance_b) ** 0.5

    if denominator <= 0:
        return None

    return covariance / denominator


def price_return(candles, period=None):
    """Simple return from the start to the end of the trailing window -
    used to check whether a reference symbol (typically BTC) is itself
    moving in the same direction as a signal, not just correlated in
    magnitude (price_correlation above is symmetric and direction-blind
    on its own)."""
    period = int(config.CORRELATION_LOOKBACK_CANDLES if period is None else period)

    if len(candles) < period:
        return None

    window = candles[-period:]
    start = window[0]["close"]

    if start == 0:
        return None

    return (window[-1]["close"] - start) / start


def analyze(candles):
    """Consolidated structure snapshot for a candle list - what
    signal_engine.py actually consumes."""
    structure = structure_state(candles)

    if not structure.get("available"):
        return {"available": False}

    swings = structure["swings"]
    zone = premium_discount_zone(candles)
    live_break = live_break_check(candles, structure)
    pools = find_liquidity_pools(swings)
    fvgs = find_fair_value_gaps(candles)
    atr = average_true_range(candles)
    efficiency = efficiency_ratio(candles)

    return {
        "available": True,
        "trend": structure["trend"],
        "last_event": structure["last_event"],
        "last_swing_high": structure["last_swing_high"],
        "last_swing_low": structure["last_swing_low"],
        "zone": zone,
        "live_break": live_break,
        "efficiency_ratio": efficiency,
        "liquidity_pools": pools,
        "fair_value_gaps": fvgs,
        "atr": atr,
    }
