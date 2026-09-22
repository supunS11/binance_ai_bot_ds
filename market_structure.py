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


def _walk_previous_pivot(swings):
    """The ORIGINAL classification, behaviour unchanged: a trend break is
    the newest confirmed pivot exceeding the *previous* confirmed pivot of
    the same kind (a higher high, or a lower low).

    Kept as its own named function - and still the default while
    config.STRUCTURE_PROTECTED_LEVEL_ENABLED is off - so every assertion
    written against this shape stays reachable and byte-identical. See
    _walk_protected_level below for what this rule actually measures and
    how far it is from the BOS/CHoCH it names."""
    trend = None
    last_high = None
    last_low = None
    last_event = None
    events = []

    for swing in swings:
        if swing.kind == "HIGH":
            if last_high is not None and swing.price > last_high.price:
                last_event = {
                    "type": "CHoCH" if trend == "BEARISH" else "BOS",
                    "direction": "BULLISH",
                    "index": swing.index,
                    "price": swing.price,
                }
                trend = "BULLISH"
                events.append(last_event)
            last_high = swing
        else:
            if last_low is not None and swing.price < last_low.price:
                last_event = {
                    "type": "CHoCH" if trend == "BULLISH" else "BOS",
                    "direction": "BEARISH",
                    "index": swing.index,
                    "price": swing.price,
                }
                trend = "BEARISH"
                events.append(last_event)
            last_low = swing

    return {"trend": trend, "last_event": last_event, "events": events}


def _walk_protected_level(swings):
    """Standard market structure: a break must exceed the PROTECTED extreme
    of the current leg, not merely the previous pivot of the same kind.

    2026-09-22 ground-up detector audit, 110 symbols x 30 days of 1h
    klines, 14,125 evaluations, measured against _walk_previous_pivot:

      events not exceeding the structural extreme    55.5%
        of which CHoCH                               71.9%
        of which BOS                                 36.4%
      total events                     76,178 -> 41,960   (-44.9%)
      CHoCH                            40,990 -> 14,258   (-65.2%)
      BOS share of all events            46.2% -> 66.0%

    The BOS/CHoCH ratio inverting is the tell. Real markets continue more
    often than they reverse; the previous-pivot rule reports the opposite
    because any minor up-tick after a lower high flips the trend label and
    manufactures a "change of character" that never happened. Concretely:
    highs of 100, 95, 97 with an ascending low between fire a bullish BOS
    at 97 while 100 stands unbroken and untested.

    The OPPOSING extreme resets when the trend actually flips, so the next
    break on that side measures continuation from the NEW leg rather than
    from a level the market has already left behind.

    Also returns the running extremes themselves. live_break_check reads
    them to classify a break as structural or minor, which is why this walk
    runs regardless of which classification is selected - the same "compute
    it either way so the forward dataset builds" convention
    signal_engine.py's ema_trend_bucket already uses."""
    trend = None
    structural_high = None
    structural_low = None
    last_event = None
    events = []

    for swing in swings:
        if swing.kind == "HIGH":
            if structural_high is not None and swing.price > structural_high:
                last_event = {
                    "type": "CHoCH" if trend == "BEARISH" else "BOS",
                    "direction": "BULLISH",
                    "index": swing.index,
                    "price": swing.price,
                }

                if trend == "BEARISH":
                    structural_low = None

                trend = "BULLISH"
                events.append(last_event)
            structural_high = (
                swing.price if structural_high is None
                else max(structural_high, swing.price)
            )
        else:
            if structural_low is not None and swing.price < structural_low:
                last_event = {
                    "type": "CHoCH" if trend == "BULLISH" else "BOS",
                    "direction": "BEARISH",
                    "index": swing.index,
                    "price": swing.price,
                }

                if trend == "BULLISH":
                    structural_high = None

                trend = "BEARISH"
                events.append(last_event)
            structural_low = (
                swing.price if structural_low is None
                else min(structural_low, swing.price)
            )

    return {
        "trend": trend,
        "last_event": last_event,
        "events": events,
        "structural_high": structural_high,
        "structural_low": structural_low,
    }


def _classify_swings(swings, structural=None):
    """Walk an already-alternating (post-zigzag) swing sequence and
    classify the current trend plus the most recent BOS/CHoCH event. Split
    out from structure_state so the classification logic can be unit-tested
    against hand-built swing sequences directly.

    `structural` (config.STRUCTURE_PROTECTED_LEVEL_ENABLED) selects WHICH
    rule supplies trend/last_event/events - see the two walks above. Both
    always run: the protected-level extremes are exposed either way so
    live_break_check can journal how a break classifies before anything
    gates on it. Two passes over a list of tens of swings costs nothing
    worth measuring.

    `events` (every BOS/CHoCH found along the way, not just the last one)
    backs ORDER_BLOCK_RETEST_TRIGGER_ENABLED's find_order_blocks below -
    an origin block from several swings ago is still a valid, unmitigated
    retest target, not just the very latest event structure_state's
    original last_event-only shape exposed.

    last_swing_high/last_swing_low are deliberately NOT the structural
    extremes and never become them - they are the most recent swing of each
    kind, which is the correct semantic for their two consumers:
    position_manager._structure_stop_candidate (a trailing stop ratchets to
    the latest confirmed swing, not to the leg's extreme) and CHOCH_RETEST's
    retracement level in signal_engine.py. A regression test pins them
    identical across both values of `structural`."""
    if len(swings) < 2:
        return {"available": False}

    if structural is None:
        structural = config.STRUCTURE_PROTECTED_LEVEL_ENABLED

    previous = _walk_previous_pivot(swings)
    protected = _walk_protected_level(swings)
    chosen = protected if structural else previous

    last_high = None
    last_low = None

    for swing in swings:
        if swing.kind == "HIGH":
            last_high = swing
        else:
            last_low = swing

    return {
        "available": True,
        "trend": chosen["trend"],
        "last_event": chosen["last_event"],
        "events": chosen["events"],
        "last_swing_high": last_high.price if last_high else None,
        "last_swing_low": last_low.price if last_low else None,
        "structural_high": protected["structural_high"],
        "structural_low": protected["structural_low"],
        "swings": swings,
    }


def structure_state(candles, left=None, right=None, structural=None):
    swings = _zigzag(find_swing_points(candles, left, right))
    return _classify_swings(swings, structural=structural)


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

    # `structural` is DESCRIPTIVE ONLY - it rejects nothing, changes neither
    # `broken` nor `direction`, and exists so the evidence journal can
    # measure whether minor breaks behave differently before any flag gates
    # on them (same journal-first discipline as ema_trend_bucket).
    #
    # This function breaks the most recent swing of each kind. That is not a
    # break of STRUCTURE whenever the protected extreme still stands above
    # (or below) it. 31.4% of ALL bars report a break here, which is far too
    # often for a genuine break of structure.
    #
    # RATE, and why two numbers exist for it. The audit's Phase 1c measured
    # 76.5% minor (110 symbols, 14,125 evaluations) against the BUFFER-WIDE
    # extreme - the highest high and lowest low anywhere in the 200-candle
    # window. This field instead uses _walk_protected_level's leg-local
    # extreme, which RESETS on a trend flip, so it is a lower bar to clear.
    # On identical buffers (20 majors, 876 breaks) the two disagree on 24.0%
    # of breaks: buffer-wide 71.9%, leg-local 48.9%. The leg-local number is
    # the one that describes THIS field, and the leg-local definition is the
    # correct one - standard market structure does not require price to
    # exceed an extreme from a leg the market has already reversed out of.
    # Phase 1c's buffer-wide version was a measurement proxy, not a spec.
    #
    # A missing extreme reads False ("cannot confirm this is structural").
    # That is fail-CLOSED, unlike every gate in this engine - deliberate,
    # and safe only because nothing rejects on it. Whatever eventually does
    # must pick its own fail-open behaviour explicitly.
    structural_high = structure.get("structural_high")
    structural_low = structure.get("structural_low")
    structural_break = (
        (broke_up and structural_high is not None and close > structural_high)
        or (broke_down and structural_low is not None and close < structural_low)
    )

    return {
        "broken": bool(broke_up or broke_down),
        "direction": "BULLISH" if broke_up else "BEARISH" if broke_down else None,
        "level": last_high if broke_up else last_low if broke_down else None,
        "structural": bool(structural_break),
        "candle_closed": latest["closed"],
        "open_time": latest["open_time"],
    }


def find_order_block(candles, index, direction, lookback=None):
    """The order block for a bullish break is the last bearish (red)
    candle before the impulsive move up; for a bearish break, the last
    bullish (green) candle before the move down.

    lookback (config.ORDER_BLOCK_SCAN_LOOKBACK_CANDLES) is how far back to
    scan for that candle. It was a hardcoded 10 until 2026-09-22 - a magic
    number in a structural detector, tied to no other lookback in the
    project. The default is still 10, so this is a rename of the constant,
    not a change to it."""
    index = min(max(index, 0), len(candles) - 1)
    lookback = int(
        config.ORDER_BLOCK_SCAN_LOOKBACK_CANDLES if lookback is None else lookback
    )

    for i in range(index, max(index - lookback, -1), -1):
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


def find_structure_events(candles, left=None, right=None, structural=None):
    """Every BOS/CHoCH event across the full swing sequence - structure_state
    only exposes the single most recent one (last_event). Needed to
    enumerate historical order blocks below (ORDER_BLOCK_RETEST_TRIGGER_
    ENABLED): an origin block from several confirmed breaks ago is still
    a valid, unmitigated retest target, not just the latest one. Recomputes
    the same zigzag+classify walk structure_state does rather than
    threading a new parameter through it - same "only pay for it when the
    flag needing it is on" convention as LIQUIDITY_SWEEP_TRIGGER_ENABLED's
    own pools/swings recompute."""
    return _classify_swings(
        _zigzag(find_swing_points(candles, left, right)), structural=structural
    ).get("events", [])


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


def find_order_block_retest(
    candles, blocks=None, max_age_candles=None, require_closed_candle=None,
    min_close_through_pct=None,
):
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
    None.

    min_close_through_pct (config.ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT)
    raises how far the retest candle's CLOSE has to reclaim back out of the
    block, measured from the far edge (0.0) toward the near edge (1.0) -
    the same parameter, geometry and rationale find_fvg_retest below
    already carries as OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT.

    2026-09-22: until now this function had no such requirement at all. Its
    condition was `latest["low"] <= high and latest["close"] > low`, which
    accepts a close ANYWHERE INSIDE the block - price sitting in the zone,
    not rejecting from it, despite the "fresh rejection wick" this
    docstring opens with. That is the identical pre-fix condition
    find_fvg_retest carried until 2026-08-22; these two are explicit
    counterparts sharing mitigation and closed-candle logic, and only one
    got the fix. Code default is 0.0 (exactly the old behaviour) so the
    live .env opts in - see the config.py comment for the detection-count
    sizing behind the 0.5 chosen there."""
    if len(candles) < 2:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if min_close_through_pct is None:
        min_close_through_pct = float(config.ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT)

    min_close_through_pct = min(max(min_close_through_pct, 0.0), 1.0)

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

        block_range = high - low
        # Near edge = the side price approached the block FROM. A BULLISH
        # order block is a demand zone sitting below price, so price comes
        # back down INTO it and the near edge is `high`; a BEARISH block is
        # supply above price, approached from below, near edge `low`. The
        # close must reclaim min_close_through_pct of the way from the far
        # edge back toward that near edge - same geometry as find_fvg_retest.
        #
        # max(block_range, 0) rather than a `continue` guard: a degenerate
        # zero-range block (high == low) then yields required == the far
        # edge at ANY pct, which is byte-identical to the pre-2026-09-22
        # condition, so the code default of 0.0 stays provably inert
        # instead of silently dropping blocks it used to accept.
        bullish_required_close = low + max(block_range, 0) * min_close_through_pct
        bearish_required_close = high - max(block_range, 0) * min_close_through_pct

        if (
            block["direction"] == "BULLISH"
            and latest["low"] <= high
            and latest["close"] > bullish_required_close
        ):
            return {"direction": "BULLISH", "level": low, "block": block, "open_time": latest["open_time"]}

        if (
            block["direction"] == "BEARISH"
            and latest["high"] >= low
            and latest["close"] < bearish_required_close
        ):
            return {"direction": "BEARISH", "level": high, "block": block, "open_time": latest["open_time"]}

    return None


def detect_level_pullback(candles, level, require_closed_candle=None):
    """Same-candle wick-to-level-and-reclaim pattern - the tested candle's
    LOW touches/pierces `level` but its CLOSE reclaims back above it
    (BULLISH), or the HIGH touches/pierces and CLOSE stays below
    (BEARISH). Generalises detect_ema_pullback's own logic to any scalar
    price level, not just the EMA - used directly by detect_ema_pullback
    below (passing ema_value as `level`) and by CHOCH_RETEST_TRIGGER_
    ENABLED (signal_engine.py, passing the active swing level the CHoCH
    is retracing to - 2026-09-15, Grok independent review finding: the
    age gate alone never confirmed price actually retraced, unlike every
    other retest-named trigger).

    By default (require_closed_candle=False) tests the current, possibly
    still-forming candle. When True (config.REQUIRE_CLOSE_CONFIRMED_BREAK,
    reused here - the same principle applied uniformly across every
    trigger), scans back to the most recently CLOSED candle instead -
    same real motivation as every other close-confirmed trigger this
    session (a wick-and-reclaim read on a still-forming candle can flip
    before the candle actually finishes). Returns {"direction", "level",
    "open_time"} or None."""
    if not candles or level is None:
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

    if low <= level and close > level:
        return {"direction": "BULLISH", "level": level, "open_time": latest["open_time"]}

    if high >= level and close < level:
        return {"direction": "BEARISH", "level": level, "open_time": latest["open_time"]}

    return None


def detect_ema_pullback(candles, ema_value, require_closed_candle=None, require_trend=None):
    """A pullback to the EMA within an established trend, followed by a
    same-candle reclaim - the classic trend-continuation entry, well
    suited to smooth, high-liquidity trending symbols (majors) that
    rarely produce the deep OTE retracement or CVD/depth imbalance every
    other trigger's downstream gate was tuned around (see config.
    EMA_PULLBACK_TRIGGER_ENABLED for the real evidence: BTC/ETH/BNB/SOL
    produced ZERO signal-related log activity across a full session with
    all 8 other triggers live).

    ema_value is the CURRENT ema (market_structure.
    exponential_moving_average) - a slow-moving rolling average, so
    using "now"'s value as a stand-in for "the EMA at the tested
    candle's close" is a reasonable approximation, the same cost/
    precision tradeoff every other trigger's shared-computation hoisting
    in signal_engine.py already makes.

    require_trend (config.EMA_PULLBACK_REQUIRE_TREND_ENABLED) supplies the
    "within an established trend" half of the description above, which this
    detector did not implement until 2026-09-22. Without it the function
    fires on ANY wick through the EMA that closes back across it, including
    chop straddling the EMA - and the trend requirement was delegated
    entirely to downstream gates reading a different timeframe (4h) and a
    different EMA pair (50/200), so nothing checked the slope of the EMA
    this trigger is actually built on. Measured live: 30.6% of detections
    fired against their own EMA's slope.

    A flat EMA fails BOTH directions - that is the absence of a trend, not
    a tie. Fails open when the prior EMA is unavailable (too little
    history), same convention as every other read here. Default off keeps
    the original thin-wrapper behaviour exactly."""
    result = detect_level_pullback(
        candles, ema_value, require_closed_candle=require_closed_candle
    )

    if result is None:
        return None

    if require_trend is None:
        require_trend = config.EMA_PULLBACK_REQUIRE_TREND_ENABLED

    if not require_trend or ema_value is None:
        return result

    prior = ema_prior_value(
        candles, candles_back=int(config.EMA_PULLBACK_TREND_LOOKBACK_CANDLES)
    )

    if prior is None:
        return result

    agrees = ema_value > prior if result["direction"] == "BULLISH" else ema_value < prior

    return result if agrees else None


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


def find_liquidity_pools(
    swings, tolerance_pct=None, anchored=None, min_touch_separation=None,
):
    """Cluster equal highs into BUY_SIDE liquidity (above the market -
    short stops + breakout buyers) and equal lows into SELL_SIDE liquidity
    (below the market - long stops), requiring at least 2 touches.

    anchored (config.LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED) decides
    what each candidate point is compared against. The original behaviour
    compares it to cluster[-1], the PREVIOUS point - single-linkage
    chaining, so a run of points each within tolerance of its neighbour
    merges into one pool spanning far more than the tolerance, and the
    reported price (the cluster mean) can sit at no real swing at all.
    Measured live: 8.0% of pools exceeded the tolerance, up to 3.4x it.

    Anchored compares each point to cluster[0] instead, so a pool can never
    span more than the tolerance it was built with. Default False keeps the
    chaining behaviour byte-for-byte - see the config flag's own comment.

    min_touch_separation (config.LIQUIDITY_POOL_MIN_TOUCH_SEPARATION_CANDLES)
    adds the TIME axis this clustering never had. A pool is supposed to be
    multiple SEPARATE VISITS to a level - that is what makes stops
    accumulate there - but the price-only clustering happily turns a run of
    consecutive fractal swings inside one impulse into a "pool with N
    touches". Measured 2026-09-22: 9.5% of pools have every touch on
    CONSECUTIVE candles, 13.4% span <= 3 candles, 32.9% span <= 10.

    Deliberately a SPAN test (first touch to last), not a pairwise one: a
    level tagged five times in one impulse and then revisited two days later
    IS a real pool, and a pairwise rule would discard it. Default 0 = no
    requirement, byte-identical to before."""
    tolerance_pct = float(
        config.LIQUIDITY_POOL_TOLERANCE_PCT if tolerance_pct is None else tolerance_pct
    )

    if anchored is None:
        anchored = config.LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED

    if min_touch_separation is None:
        min_touch_separation = config.LIQUIDITY_POOL_MIN_TOUCH_SEPARATION_CANDLES

    min_touch_separation = max(int(min_touch_separation), 0)
    pools = []

    def _flush(cluster, label):
        """A cluster becomes a pool only if it has the touches AND spans the
        time. Shared by the mid-loop and end-of-loop flushes so the two can
        never drift apart."""
        if len(cluster) < 2:
            return

        if min_touch_separation > 0:
            indices = [p.index for p in cluster]

            if max(indices) - min(indices) < min_touch_separation:
                return

        pools.append({
            "type": label,
            "price": sum(p.price for p in cluster) / len(cluster),
            "touches": len(cluster),
        })

    for kind, label in (("HIGH", "BUY_SIDE"), ("LOW", "SELL_SIDE")):
        points = sorted(
            (s for s in swings if s.kind == kind),
            key=lambda s: s.price,
        )
        cluster = []

        for point in points:
            # Anchored compares against the cluster's FIRST member so the
            # cluster can never span more than tolerance_pct; chained
            # compares against the previous point, which is what lets a run
            # of near-neighbours drift arbitrarily far.
            reference = None

            if cluster:
                reference = cluster[0] if anchored else cluster[-1]

            if reference is not None and abs(point.price - reference.price) / reference.price <= tolerance_pct:
                cluster.append(point)
                continue

            _flush(cluster, label)
            cluster = [point]

        _flush(cluster, label)

    return pools


def _ob_edges_as_pool(order_block, near):
    if order_block["direction"] == "BULLISH":  # demand zone, below price
        return {
            "type": "SELL_SIDE",
            "price": order_block["high"] if near else order_block["low"],
            "touches": 0,
        }

    return {  # BEARISH - supply zone, above price
        "type": "BUY_SIDE",
        "price": order_block["low"] if near else order_block["high"],
        "touches": 0,
    }


def _fvg_edges_as_pool(gap, near):
    if gap["type"] == "BULLISH":  # gap below price, support
        return {
            "type": "SELL_SIDE",
            "price": gap["top"] if near else gap["bottom"],
            "touches": 0,
        }

    return {  # BEARISH - gap above price, resistance
        "type": "BUY_SIDE",
        "price": gap["bottom"] if near else gap["top"],
        "touches": 0,
    }


def find_structure_candidates(candles, left=None, right=None, for_stop=False):
    """Real liquidity pools PLUS order-block/FVG zone edges, normalised to
    find_liquidity_pools' own {"type","price","touches"} shape so every
    existing pool-consuming function in risk_manager.py (_find_structure_
    target, _find_dca_level, _next_pool_beyond_stop) works unchanged
    regardless of which candidate produced the level. touches=0 (never
    produced by a real liquidity pool, which requires >=2) keeps a
    synthetic OB/FVG edge distinguishable without colliding with the
    None-means-fallback convention tp1_source relies on.

    for_stop=False (target/TP usage, default): OB/FVG zones contribute
    their NEAR edge (first realistic touch point) - matches
    _find_structure_target's own nearest-qualifying-candidate bias.
    for_stop=True (SL usage): zones contribute their FAR edge instead - a
    stop must clear the whole zone, not rest at its first-touch edge."""
    # list(...) copies rather than mutates find_liquidity_pools' own
    # returned list in place via the .append() calls below - it may be a
    # cached/shared object, not necessarily a fresh list per call.
    pools = list(find_liquidity_pools(find_swing_points(candles, left, right)))
    near = not for_stop

    for block in find_order_blocks(candles, left, right):
        pools.append(_ob_edges_as_pool(block, near))

    for gap in find_fair_value_gaps(candles):
        pools.append(_fvg_edges_as_pool(gap, near))

    return pools


def find_fvg_retest(
    candles, fvgs=None, max_age_candles=None, require_closed_candle=None,
    min_close_through_pct=None,
):
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
    open_time - the candle actually tested), or None. tested_index is
    that same tested candle's index within `candles` - the SAME index
    this function's own age-gate check above (OB_FVG_RETEST_MAX_AGE_
    CANDLES) already uses, so a caller computing "how old is this gap"
    for its own purposes (signal_engine.py's setup_age_candles) measures
    the exact same age the gate enforced, not a naive len(candles)-1 that
    can silently differ by however many still-forming candles sit past
    the last CLOSED one when require_closed_candle is True.

    min_close_through_pct (config.OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT)
    raises how far the retest candle's CLOSE has to reclaim back out of
    the gap, measured from the far edge (0.0, the close-anywhere-past-the-
    far-edge behavior this function originally shipped with) toward the
    near edge (1.0, a full close back outside the gap). Real motivation
    (2026-08-22): live OB_FVG_RETEST trades were still averaging ~0.68R
    max adverse excursion even on trades that went on to WIN (28% of wins
    still ran 1R+ against the position first) - and reading this
    function's own qualifying condition showed why: the original
    close > bottom (BULLISH) / close < top (BEARISH) check accepts a
    retest candle that closes barely off the gap's far edge, deep inside
    the zone, as equally valid as one that closes back near the near edge
    - no confirmation the rejection actually has any strength behind it.
    Defaults to the midpoint (0.5), not the strictest 1.0: a full close
    back outside the gap would reject a lot of genuine retests along with
    the weak ones, and no outcome data yet exists isolating exactly how
    much depth is enough - the journal never captured close-position-
    within-gap before now, so this is a reasoned, not yet outcome-
    validated, starting point (revisit once trades post-dating this
    change have resolved)."""
    if len(candles) < 3:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    if min_close_through_pct is None:
        min_close_through_pct = float(config.OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT)

    min_close_through_pct = min(max(min_close_through_pct, 0.0), 1.0)

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

        gap_range = top - bottom
        # Near edge = the side price approached the gap FROM (BULLISH came
        # from above -> near edge is top; BEARISH came from below -> near
        # edge is bottom). The close must reclaim min_close_through_pct of
        # the distance from the far edge toward that near edge.
        bullish_required_close = bottom + gap_range * min_close_through_pct
        bearish_required_close = top - gap_range * min_close_through_pct

        if (
            gap["type"] == "BULLISH"
            and latest["low"] <= top
            and latest["close"] > bullish_required_close
        ):
            return {
                "direction": "BULLISH", "level": bottom, "gap": gap,
                "open_time": latest["open_time"], "tested_index": latest_index,
            }

        if (
            gap["type"] == "BEARISH"
            and latest["high"] >= bottom
            and latest["close"] < bearish_required_close
        ):
            return {
                "direction": "BEARISH", "level": top, "gap": gap,
                "open_time": latest["open_time"], "tested_index": latest_index,
            }

    return None


def find_break_ote_retest(
    candles, swings=None, max_age_candles=None, require_closed_candle=None,
):
    """config.BREAK_OTE_RETEST_TRIGGER_ENABLED - structure broke recently,
    and price has NOW retraced into the OTE band OF THE BROKEN LEG.

    This is the two-candle pattern signal_engine.py's NOT_IN_OTE gate has
    always CLAIMED to enforce - "structure break, then retrace to OTE" - but
    never did. That gate tests both halves simultaneously on the break candle
    itself, which cannot happen: a break fires at an extreme (measured median
    retracement depth 0.158) while OTE demands 0.705-0.79. It blocks 98.4% of
    STRUCTURE_BREAK detections as a result. See OTE_GATE_EXEMPT_TRIGGERS in
    config.py for that measurement; this function is the other half of the
    fix - the pattern implemented on the candle where it can actually occur.

    Two deliberate differences from the gate:

      * The leg, not the window. The OTE band is measured on the impulse leg
        that broke (swing low -> break high, or the mirror), which is what
        ICT actually anchors a retracement to. The gate measures it on
        PREMIUM_DISCOUNT_LOOKBACK_CANDLES of arbitrary range, so its band
        moves with unrelated price action.

      * The break must be REAL. The leg comes from _walk_protected_level,
        not from structure_state's last_event, so this trigger anchors to a
        break that cleared the protected extreme regardless of whether
        config.STRUCTURE_PROTECTED_LEVEL_ENABLED is on. It is correct from
        day one and does not silently change meaning when that flag flips.

    Returns {direction, level, leg, open_time, tested_index,
    setup_age_candles} or None. Direction is the BREAK's direction - a
    bullish break that has retraced is a BUY back into discount, not a
    fade."""
    if len(candles) < 3:
        return None

    if require_closed_candle is None:
        require_closed_candle = config.REQUIRE_CLOSE_CONFIRMED_BREAK

    max_age = int(
        config.BREAK_OTE_RETEST_MAX_AGE_CANDLES
        if max_age_candles is None else max_age_candles
    )

    if require_closed_candle:
        closed_candles = [(i, c) for i, c in enumerate(candles) if c.get("closed")]

        if not closed_candles:
            return None

        tested_index, latest = closed_candles[-1]
    else:
        tested_index = len(candles) - 1
        latest = candles[tested_index]

    if swings is None:
        swings = _zigzag(find_swing_points(candles))

    if len(swings) < 2:
        return None

    event = _walk_protected_level(swings)["last_event"]

    if event is None:
        return None

    age = tested_index - event["index"]

    if age <= 0 or age > max_age:
        return None

    # The leg is the move that produced the break: from the last opposing
    # swing BEFORE the breaking pivot, to the breaking pivot itself.
    opposing_kind = "LOW" if event["direction"] == "BULLISH" else "HIGH"
    opposing = [
        s for s in swings
        if s.kind == opposing_kind and s.index < event["index"]
    ]

    if not opposing:
        return None

    if event["direction"] == "BULLISH":
        leg_high, leg_low = event["price"], opposing[-1].price
    else:
        leg_high, leg_low = opposing[-1].price, event["price"]

    leg = leg_high - leg_low

    if leg <= 0:
        return None

    ote_min = float(config.OTE_RETRACEMENT_MIN)
    ote_max = float(config.OTE_RETRACEMENT_MAX)
    close = latest["close"]

    if event["direction"] == "BULLISH":
        # Retraced DOWN from the break high into the band, and the leg is
        # still intact (a close back under its origin invalidates it).
        if close <= leg_low:
            return None

        band_low, band_high = leg_high - leg * ote_max, leg_high - leg * ote_min
    else:
        if close >= leg_high:
            return None

        band_low, band_high = leg_low + leg * ote_min, leg_low + leg * ote_max

    if not band_low <= close <= band_high:
        return None

    return {
        "direction": event["direction"],
        # The invalidation edge, same role structure_level plays for every
        # other trigger: the level the setup is wrong below (BULLISH) or
        # above (BEARISH).
        "level": leg_low if event["direction"] == "BULLISH" else leg_high,
        "leg": {"high": leg_high, "low": leg_low, "index": event["index"]},
        "open_time": latest["open_time"],
        "tested_index": tested_index,
        "setup_age_candles": age,
    }


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


def zone_direction(candles, lookback=None):
    """Which way the range itself is MOVING, as opposed to where price sits
    inside it (zone_for_price above).

    premium_discount_zone answers "is this cheap relative to the last N
    candles" - but it has no idea whether that whole range is sliding down.
    Cheap inside a falling market is not cheap, it is just the newest price
    on the way down, which is the mechanism behind the long-standing "buys
    keep entering at the top of the move" complaint. This measures the
    missing axis: the midpoint of the recent half of the window against the
    midpoint of the older half.

    Deliberately reads a SHORTER window than the zone it accompanies (see
    config.ZONE_DIRECTION_LOOKBACK_CANDLES for the measured reason) - the
    pairing that works is a long range for "where am I" and a short one for
    "where is it going". Returns None on too little history or an exact
    tie, so every caller fails open - same shape as htf_trend_live and
    signal_engine._ema_regime."""
    lookback = int(
        config.ZONE_DIRECTION_LOOKBACK_CANDLES if lookback is None else lookback
    )
    window = candles[-lookback:] if len(candles) > lookback else candles
    half = len(window) // 2

    if half < 2:
        return None

    def _midpoint(part):
        return (max(c["high"] for c in part) + min(c["low"] for c in part)) / 2

    older = _midpoint(window[:half])
    recent = _midpoint(window[half:])

    if recent > older:
        return "BULLISH"

    if recent < older:
        return "BEARISH"

    return None


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


def ema_prior_value(candles, period=None, candles_back=3):
    """EMA value as of `candles_back` candles ago - same math as
    exponential_moving_average, anchored earlier, so a caller can measure
    the EMA's own slope (config.HTF_TREND_LIVE_STRENGTH_REJECT_ENABLED)
    instead of only its current level. A separate function (not just a
    second call to exponential_moving_average with sliced candles) so
    it's independently mockable in tests, and so a too-short candles list
    fails closed with its own None rather than a misleadingly-short EMA
    window."""
    if candles_back <= 0 or len(candles) <= candles_back:
        return None

    return exponential_moving_average(candles[:-candles_back], period=period)


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


def price_hold_consistency(candles, side, lookback=None):
    """Fraction (0-1) of the most recent `lookback` real closed candles
    that closed in `side`'s own favorable direction (close > open for
    BUY, close < open for SELL) - config.OB_FVG_RETEST_PRICE_WEAK_REJECT_
    ENABLED. A real, replayable proxy for "was recent price action
    actually sustained in this direction, or just a fleeting flip":
    historical order-book depth isn't archived anywhere, so a book-based
    version of this same question can't be checked against real past
    trades the way DEPTH_TREND_UNSTABLE's own depth_consistency_pct is -
    this uses real closed-candle price action instead, which is archived
    indefinitely (see that flag's own config.py comment for the real
    evidence). Returns None with too little history rather than a
    misleadingly short window."""
    lookback = int(config.OB_FVG_RETEST_PRICE_HOLD_LOOKBACK_MINUTES if lookback is None else lookback)

    if lookback <= 0 or len(candles) < lookback:
        return None

    window = candles[-lookback:]
    favorable = sum(
        1 for candle in window
        if ((candle["close"] > candle["open"]) if side == "BUY" else (candle["close"] < candle["open"]))
    )

    return favorable / len(window)


def oi_percentile(history):
    """Percentile rank (0-1) of the most recent OI reading within its own
    trailing history - config.MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED (see
    that flag's own config.py comment for the real evidence). A different
    question from OI_RISING (direction/momentum over OI_RISING's own
    much shorter in-memory window): this is absolute level/regime -
    "is OI currently crowded relative to where it's recently sat" -
    over a much longer real trailing window that OpenInterestEngine's
    short-lived in-memory history doesn't carry, hence the separate
    on-demand exchange.get_open_interest_history call in main.py.
    history is that function's own ascending (timestamp, oi_value) tuple
    list, or None/too-short on failure. Requires at least 30 real samples
    (same floor used throughout this gate's own evidence check) - fewer
    than that makes a percentile rank too noisy to mean anything."""
    if not history or len(history) < 30:
        return None

    values = [value for _, value in history]
    current = values[-1]
    below_or_equal = sum(1 for value in values if value <= current)

    return below_or_equal / len(values)


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
        # The protected extremes of the current leg - distinct from the two
        # fields above, which are the most recent swing of each kind. See
        # _walk_protected_level. Exposed regardless of
        # config.STRUCTURE_PROTECTED_LEVEL_ENABLED.
        "structural_high": structure["structural_high"],
        "structural_low": structure["structural_low"],
        # The zigzagged swing list this snapshot was built from. Already
        # computed above for find_liquidity_pools, so exposing it is free and
        # lets find_break_ote_retest reuse it rather than redoing the
        # fractal scan. Flag-independent - the swings themselves do not
        # depend on which classification walk is selected.
        "swings": swings,
        "zone": zone,
        "live_break": live_break,
        "efficiency_ratio": efficiency,
        "liquidity_pools": pools,
        "fair_value_gaps": fvgs,
        "atr": atr,
    }
