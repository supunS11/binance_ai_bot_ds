"""Turns a signal_engine candidate into concrete SL / TP1 / TP2 prices and
a position size.

The stop comes first - beyond the structure level that triggered the
entry, plus a small ATR buffer - and everything else derives from it:
position size is solved from the risk distance (the "1% rule", via
risk_management.calculate_position_size).

TP1/TP2 target real liquidity pools when one exists with enough room,
matching v7's find_structure_take_profit (price is drawn toward real
liquidity, not an arbitrary multiple of the stop - that's the actual
premise of SMC). The R-multiple (2R/4R) is a floor, not the primary
target: it sets the *minimum* acceptable distance a structure target must
clear, and is used directly as a fallback only when no real level exists
out that far - same shape as v7's ENTRY_MIN_TP_ROOM_ROI/STRUCTURE_TP_MIN_ROI
fallback.
"""
import config
from risk_management import calculate_position_size, get_position_risk_budget


def _find_structure_target(pools, entry_price, side, min_r_multiple, max_r_multiple, risk_distance):
    """Nearest real liquidity-pool price in the trade's favorable
    direction that clears `min_r_multiple` R of room *and* stays within
    `max_r_multiple` R, or None if no pool qualifies. The max bound
    matters: without it, the "nearest qualifying pool" can still be
    absurdly far away if nothing closer exists (a real case seen live -
    a ~20R target that's realistically never going to be reached, which
    defeats the point of TP1 as an achievable first partial). BUY targets
    BUY_SIDE pools (resistance above entry - where breakout-buy stops
    sit); SELL targets SELL_SIDE pools (support below entry - where long
    stops sit) - the same pools liquidity_sweep.py already uses, just on
    the opposite side of price from where a sweep would be found."""
    if risk_distance <= 0:
        return None

    min_distance = min_r_multiple * risk_distance
    max_distance = max_r_multiple * risk_distance if max_r_multiple else None
    pool_type = "BUY_SIDE" if side == "BUY" else "SELL_SIDE"
    candidates = []

    for pool in pools or []:
        price = pool.get("price")

        if price is None or pool.get("type") != pool_type:
            continue

        distance = (price - entry_price) if side == "BUY" else (entry_price - price)

        if distance < min_distance:
            continue

        if max_distance is not None and distance > max_distance:
            continue

        candidates.append((distance, price))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1]


def _resolve_target(pools, entry_price, side, min_r_multiple, max_r_multiple, risk_distance):
    structure_price = _find_structure_target(
        pools, entry_price, side, min_r_multiple, max_r_multiple, risk_distance
    )

    if structure_price is not None:
        return structure_price

    distance = min_r_multiple * risk_distance
    return entry_price + distance if side == "BUY" else entry_price - distance


def nearest_favorable_structure_r(pools, entry_price, side, risk_distance):
    """Distance (in R) to the closest real liquidity-pool level in the
    trade's favorable direction - unlike _find_structure_target (which
    only considers pools that already clear TP1_R_MULTIPLE/TP2_R_MULTIPLE
    of room), this reports whatever real level is actually nearest, even
    one too close to ever qualify as a TP itself.

    Informational only, journaled by signal_journal.py so journal_analysis.py
    can test a real gap this codebase's target selection doesn't otherwise
    answer: TP1/TP2 are drawn TO a real structure level, but nothing checks
    whether a CLOSER one sits in the way first and could cap or reverse the
    move before that target is reached. Same "log it before gating on it"
    rollout as ema_aligned/oi_rising originally used - see
    config.OI_RISING_REJECT_ENABLED for what happens once evidence exists.

    None if no real pool exists in that direction, or risk_distance isn't
    usable (mirrors _find_structure_target's own None-on-failure shape)."""
    if risk_distance <= 0:
        return None

    price = _find_structure_target(pools, entry_price, side, 0, None, risk_distance)

    if price is None:
        return None

    return abs(price - entry_price) / risk_distance


def _find_dca_level(pools, entry_price, side):
    """Nearest real liquidity-pool price in the ADVERSE direction from
    entry - the mirror image of _find_structure_target's favorable-side
    search (used for TP1/TP2). BUY looks for a SELL_SIDE pool (support
    below entry, where a long-side liquidity sweep would sit - the
    "buy it cheaper" level); SELL looks for a BUY_SIDE pool (resistance
    above entry). None if no such pool exists at all - the caller
    supplies an ATR-based fallback."""
    pool_type = "SELL_SIDE" if side == "BUY" else "BUY_SIDE"
    candidates = []

    for pool in pools or []:
        price = pool.get("price")

        if price is None or pool.get("type") != pool_type:
            continue

        distance = (entry_price - price) if side == "BUY" else (price - entry_price)

        if distance <= 0:
            continue

        candidates.append((distance, price))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1]


def _dca_fallback_distance(atr):
    """No real structure level exists in the adverse direction within
    reach - fall back to a fixed ATR multiple rather than leave the
    position with no computable DCA/post-DCA-SL level at all (this
    project's whole no-SL-before-DCA design depends on one always being
    available). Same "2x" shape config.py's other ATR-driven fallbacks
    use as a starting, uncalibrated value."""
    return float(atr or 0) * 2


def compute_dca_price(entry_price, side, pools, atr=0):
    """Where the single DCA fires if price moves against the position
    before TP1 - the next real structure level in the ADVERSE direction
    from entry (see _find_dca_level), falling back to a fixed ATR
    distance when none exists. Same _apply_min_stop_distance floor
    compute_stop_loss uses - a level pathologically close to entry would
    trigger the DCA on ordinary noise instead of a real adverse move."""
    level = _find_dca_level(pools, entry_price, side)

    if level is None:
        distance = _dca_fallback_distance(atr)
        level = entry_price - distance if side == "BUY" else entry_price + distance

    return _apply_min_stop_distance(level, entry_price, side, atr=atr)


def compute_dca_sl_price(dca_fill_price, side, pools, atr=0):
    """The first real SL this position ever gets, placed the moment DCA
    fires - anchored to the next real structure level BEYOND the DCA fill
    itself (one level further in the adverse direction than dca_price
    was from the original entry), same ATR-buffer/min-distance shape
    compute_stop_loss uses for the ordinary (here, never-placed) stop."""
    level = _find_dca_level(pools, dca_fill_price, side)

    if level is None:
        distance = _dca_fallback_distance(atr)
        level = dca_fill_price - distance if side == "BUY" else dca_fill_price + distance

    buffer = float(atr or 0) * max(float(config.DCA_STRUCTURE_STOP_ATR_BUFFER), 0)
    sl_price = level - buffer if side == "BUY" else level + buffer
    return _apply_min_stop_distance(sl_price, dca_fill_price, side, atr=atr)


def price_at_roi_pct(entry_price, side, roi_pct):
    """Price at which unrealized ROI (at LEVERAGE) equals roi_pct% of
    margin - same underlying relationship _roi_pct/_price_at_tp1_roi_
    fraction already use elsewhere (profit protection), but against an
    ABSOLUTE target instead of a fraction of TP1's own ROI. Built for
    config.DCA_TP_STATIC_ROI_ENABLED - see compute_dca_target. None if
    entry_price/LEVERAGE aren't usable, same as its sibling."""
    if entry_price <= 0:
        return None

    leverage = max(float(config.LEVERAGE), 0)

    if leverage <= 0:
        return None

    distance = (max(float(roi_pct), 0) / 100) / leverage * entry_price
    return entry_price + distance if side == "BUY" else entry_price - distance


def compute_dca_target(new_entry_price, sl_price, side, pools):
    """The single post-DCA take-profit target that replaces TP1+TP2 once
    a DCA has fired.

    config.DCA_TP_STATIC_ROI_ENABLED - operator-requested alternative
    (2026-08-19, no evidence yet either way - default False, same "new
    mechanism earns a live default only after real data" convention as
    DCA_BREAKEVEN_CONFIRMATION_ENABLED): a fixed ROI% target
    (config.DCA_TP_TARGET_ROI_PCT) computed directly from the blended
    post-DCA entry price, completely independent of structure/liquidity
    pools or the R-multiple floor/cap below - "static" in the sense that
    once a DCA fires, this target is fully determined by the fill price
    and a config number alone, nothing else. Off (default): unchanged
    real-liquidity-first, R-multiple-floor fallback shape compute_targets
    already uses (config.DCA_TP_R_MULTIPLE/DCA_TP_MAX_R_MULTIPLE instead
    of TP1/TP2's own multiples), scaled against the NEW (post-DCA) risk
    distance."""
    if config.DCA_TP_STATIC_ROI_ENABLED:
        return price_at_roi_pct(new_entry_price, side, config.DCA_TP_TARGET_ROI_PCT)

    risk_distance = abs(new_entry_price - sl_price)

    if risk_distance <= 0:
        return None

    tp_min = max(float(config.DCA_TP_R_MULTIPLE), 0)
    tp_max = max(float(config.DCA_TP_MAX_R_MULTIPLE), tp_min)
    return _resolve_target(pools, new_entry_price, side, tp_min, tp_max, risk_distance)


def build_dca_plan(original_entry_price, original_quantity, dca_fill_price, dca_quantity, side, pools, atr=0):
    """Everything position_manager._execute_dca needs once a real DCA
    fill is confirmed: the blended entry price (quantity-weighted average
    of the two fills - the real average cost basis regardless of
    DCA_SIZE_MULTIPLIER), the first real SL this position ever gets, and
    the single TP that replaces TP1+TP2. Returns None if any leg can't be
    computed (mirrors build_trade_plan's own None-on-failure shape) -
    callers must treat that as "DCA fill happened but can't be protected
    yet, retry or escalate", never as "skip protection"."""
    total_quantity = original_quantity + dca_quantity

    if total_quantity <= 0:
        return None

    new_entry_price = (
        original_entry_price * original_quantity + dca_fill_price * dca_quantity
    ) / total_quantity

    sl_price = compute_dca_sl_price(dca_fill_price, side, pools, atr=atr)

    if sl_price is None or sl_price <= 0:
        return None

    if side == "BUY" and sl_price >= new_entry_price:
        return None

    if side == "SELL" and sl_price <= new_entry_price:
        return None

    tp_price = compute_dca_target(new_entry_price, sl_price, side, pools)

    if tp_price is None:
        return None

    return {
        "entry_price": new_entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "quantity": total_quantity,
        "risk_distance": abs(new_entry_price - sl_price),
    }


def _apply_min_stop_distance(sl_price, entry_price, side, atr=0):
    """Structure can occasionally land pathologically close to entry - a
    fast/noisy market, or a tight fractal window finding a swing point
    right next to current price. Left alone, that produces a stop that's
    essentially inside normal noise (gets hit immediately) and, because
    position size is solved from the stop distance, an oversized position
    to match. Widen the stop out to a minimum distance from entry rather
    than let one through at a size ordinary noise will trigger.

    The floor is the WIDER of two measures, not just MIN_STOP_DISTANCE_PCT
    alone: a flat percentage of price can't be "enough" for every symbol
    at once - real evidence (2026-08-13, both a direct log-distance trace
    of 24 real trades and three consecutive journal_analysis.py pulls)
    showed MIN_STOP_DISTANCE_PCT (0.6%) getting hit on ~40% of trades, with
    every floor-clamped trade resolving as a loss or scratch - 0.6% is
    inside normal 1h noise for the volatile small/mid-cap symbols this
    watchlist trades most often. MIN_STOP_DISTANCE_ATR_MULTIPLE scales the
    floor with each symbol's own measured volatility instead."""
    entry_price = float(entry_price or 0)

    if entry_price <= 0:
        return sl_price

    min_pct = max(float(config.MIN_STOP_DISTANCE_PCT), 0) / 100
    min_atr_multiple = max(float(config.MIN_STOP_DISTANCE_ATR_MULTIPLE), 0)
    min_distance = max(entry_price * min_pct, float(atr or 0) * min_atr_multiple)

    if min_distance <= 0 or abs(entry_price - sl_price) >= min_distance:
        return sl_price

    return entry_price - min_distance if side == "BUY" else entry_price + min_distance


def compute_stop_loss(signal, side):
    level = signal.get("structure_level")

    if level is None:
        return None

    atr = signal.get("atr") or 0
    buffer = atr * max(float(config.STRUCTURE_STOP_ATR_BUFFER), 0)
    sl_price = level - buffer if side == "BUY" else level + buffer

    return _apply_min_stop_distance(sl_price, signal.get("entry_price"), side, atr=atr)


def compute_targets(entry_price, sl_price, side, pools=None):
    risk_distance = abs(entry_price - sl_price)

    if risk_distance <= 0:
        return None, None

    tp1_min = max(float(config.TP1_R_MULTIPLE), 0)
    tp2_min = max(float(config.TP2_R_MULTIPLE), tp1_min)
    tp1_max = max(float(config.TP1_MAX_R_MULTIPLE), tp1_min)
    tp2_max = max(float(config.TP2_MAX_R_MULTIPLE), tp2_min)

    tp1_price = _resolve_target(pools, entry_price, side, tp1_min, tp1_max, risk_distance)
    tp1_actual_multiple = abs(tp1_price - entry_price) / risk_distance

    # TP2 must clear whichever is further out: the configured floor, or
    # at least 1R beyond wherever TP1 actually landed - guarantees TP2
    # always sits meaningfully beyond TP1 regardless of whether either
    # came from a real structure level or the fallback. Capped at its own
    # max the same way TP1 is, for the same reason.
    tp2_min_multiple = min(max(tp2_min, tp1_actual_multiple + 1.0), tp2_max)
    tp2_price = _resolve_target(pools, entry_price, side, tp2_min_multiple, tp2_max, risk_distance)

    return tp1_price, tp2_price


def compute_breakeven_price(entry_price, side):
    """A hair beyond flat entry, not exactly at it - so the breakeven stop
    still covers exchange fees/slippage instead of being a guaranteed
    zero-sum exit."""
    buffer_pct = max(float(config.BREAKEVEN_BUFFER_PCT), 0) / 100

    if side == "BUY":
        return entry_price * (1 + buffer_pct)

    return entry_price * (1 - buffer_pct)


def compute_early_breakeven_price(entry_price, side, risk_distance):
    """Where the stop goes on an EARLY breakeven promotion specifically
    (see config.EARLY_BREAKEVEN_LOCK_R_MULTIPLE) - distinct from
    compute_breakeven_price, which stays fee-buffer-only for the TP1-
    triggered promotion path. A lock multiple of 0 preserves the original
    flat-breakeven behavior exactly (falls through to
    compute_breakeven_price); above 0, the stop moves into real locked
    profit instead of a scratch, at the cost of a tighter stop that a
    genuine TP1/TP2 runner could dip through on its way to target."""
    lock_multiple = max(float(config.EARLY_BREAKEVEN_LOCK_R_MULTIPLE), 0)

    if lock_multiple <= 0:
        return compute_breakeven_price(entry_price, side)

    lock_distance = risk_distance * lock_multiple

    if side == "BUY":
        return entry_price + lock_distance

    return entry_price - lock_distance


def _price_at_tp1_roi_fraction(entry_price, side, tp1_price, fraction_pct):
    """Price at which unrealized ROI (at LEVERAGE) equals fraction_pct%
    of what TP1 itself would pay out in ROI. Shared scaling used by both
    the profit-protection arm trigger (PROFIT_PROTECTION_ACTIVATION_PCT_
    OF_TP1) and its trailing floor (PROFIT_PROTECTION_LOCK_PCT_OF_TP1) -
    same TP1-relative math, different fraction. None if TP1's own ROI
    can't be computed (missing/degenerate tp1_price, zero entry_price,
    or LEVERAGE<=0) - callers must treat None as "not available yet"."""
    if entry_price <= 0 or tp1_price is None:
        return None

    tp1_roi_pct = _roi_pct(entry_price, abs(tp1_price - entry_price))

    if tp1_roi_pct <= 0:
        return None

    leverage = max(float(config.LEVERAGE), 0)

    if leverage <= 0:
        return None

    fraction = max(float(fraction_pct), 0) / 100
    target_roi_pct = tp1_roi_pct * fraction
    distance = (target_roi_pct / 100) / leverage * entry_price
    return entry_price + distance if side == "BUY" else entry_price - distance


def compute_profit_protection_lock_price(entry_price, side, tp1_price):
    """config.PROFIT_PROTECTION_ENABLED - the price at which unrealized
    ROI reaches PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1% of what TP1
    itself would pay out in ROI. This is the ARM trigger only
    (position_manager checks "has price reached this yet") - once armed,
    the stop itself trails via compute_profit_protection_trailing_floor
    instead of jumping straight to this same price (2026-08-15: locking
    the SL exactly at the trigger price left ~zero room between the stop
    and current price the instant it fired, so ordinary noise on the very
    next tick closed the trade immediately - explicit operator report,
    fixed by separating "enough profit to arm" from "where the stop
    actually sits" the same way v7's evaluate_route_profit_protection
    separates trigger_roi from lock_roi/retrace_pct).

    Real operator concern (2026-08-18): when TP1's own ROI clears
    PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT, waiting for the normal
    (higher) PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1 fraction of it means
    a large amount of unrealized profit sits unprotected in absolute ROI
    terms - PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI (lower) is
    used instead in that case. Only the ARM fraction is tiered this way;
    compute_profit_protection_trailing_floor's LOCK_PCT_OF_TP1/RETRACE_PCT
    (what happens once already armed) are unaffected."""
    if entry_price > 0 and tp1_price is not None:
        tp1_roi_pct = _roi_pct(entry_price, abs(tp1_price - entry_price))
        threshold = max(float(config.PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT), 0)
        activation_pct = (
            config.PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI
            if tp1_roi_pct > threshold
            else config.PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1
        )
    else:
        activation_pct = config.PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1

    return _price_at_tp1_roi_fraction(entry_price, side, tp1_price, activation_pct)


def compute_profit_protection_trailing_floor(entry_price, side, tp1_price, peak_price):
    """Where the SL actually gets set once profit protection has armed -
    never below the PROFIT_PROTECTION_LOCK_PCT_OF_TP1 price (the
    worst-case guaranteed lock), otherwise PROFIT_PROTECTION_RETRACE_PCT
    of the gain from entry to the best price reached since arming
    (peak_price) is allowed to give back before the stop catches up.
    Ratchet-only in intent - callers (position_manager) still compare
    the result against the current sl_price via _more_favorable and skip
    a replace that isn't actually an improvement, since peak_price can
    only move in the position's favor but a shrinking TP1-relative floor
    on its own would not guarantee that.

    None only when the arm-time lock price itself can't be computed
    (same missing-data cases as compute_profit_protection_lock_price);
    a missing/invalid peak_price just falls back to that lock price."""
    lock_price = _price_at_tp1_roi_fraction(
        entry_price, side, tp1_price, config.PROFIT_PROTECTION_LOCK_PCT_OF_TP1
    )

    if lock_price is None or peak_price is None or entry_price <= 0:
        return lock_price

    peak_price = float(peak_price)
    retrace_pct = min(max(float(config.PROFIT_PROTECTION_RETRACE_PCT), 0), 100)
    retained_distance = abs(peak_price - entry_price) * (1 - retrace_pct / 100)
    retrace_price = (
        entry_price + retained_distance if side == "BUY" else entry_price - retained_distance
    )

    return max(lock_price, retrace_price) if side == "BUY" else min(lock_price, retrace_price)


def _entry_extension_r(signal, entry_price, side, risk_distance):
    """How far entry_price has already run beyond the structure level that
    triggered the setup, expressed in R (risk_distance-relative) - None
    when there's nothing to measure against (no structure_level, or a
    degenerate zero risk_distance). Shared by _entry_too_extended (the
    hard MAX_ENTRY_EXTENSION_R reject) and build_trade_plan's
    entry_extension_r output (used by main.py to route a moderately-
    extended-but-not-rejected entry to a limit order instead of a market
    order - see config.ENTRY_ROUTING_EXTENSION_THRESHOLD_R)."""
    if risk_distance <= 0:
        return None

    structure_level = signal.get("structure_level")

    if structure_level is None:
        return None

    extension = (
        entry_price - structure_level if side == "BUY" else structure_level - entry_price
    )
    return extension / risk_distance


def _entry_too_extended(extension_r):
    """See config.MAX_ENTRY_EXTENSION_R - rejects an entry that's already
    run too far beyond the structure level that triggered it, instead of
    market-chasing whatever price exists once confirmation finishes."""
    max_extension_r = max(float(config.MAX_ENTRY_EXTENSION_R), 0)

    if max_extension_r <= 0 or extension_r is None:
        return False

    return extension_r > max_extension_r


def _roi_pct(entry_price, price_distance):
    """ROI% (of margin, at config.LEVERAGE) for a given absolute price
    distance from entry - quantity cancels out of the ratio (loss-at-SL/
    profit-at-price and margin-used both scale with it identically), so
    this only ever needs the distance and entry price. Shared by
    _stop_roi_too_high (the loss side, MAX_SL_ROI_PCT) and
    compute_profit_protection_lock_price (the profit side,
    PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1) - same relationship, either
    direction."""
    if entry_price <= 0:
        return 0.0

    return (price_distance / entry_price) * max(float(config.LEVERAGE), 0) * 100


def _stop_roi_too_high(risk_distance, entry_price):
    """See config.MAX_SL_ROI_PCT - rejects an entry whose stop, once
    LEVERAGE is applied, would lose more than this % of the margin
    actually at risk if hit. Rejects outright rather than shrinking the
    position, per explicit operator choice: a stop this wide relative to
    leverage is treated as not worth taking at any size, not just a
    smaller one."""
    max_roi_pct = max(float(config.MAX_SL_ROI_PCT), 0)

    if max_roi_pct <= 0 or entry_price <= 0:
        return False

    return _roi_pct(entry_price, risk_distance) > max_roi_pct


def _confluence_size_multiplier(signal):
    """Scales risk per trade by how much of sweep/EMA/OI/liquidation
    confluence agrees with this signal (signal_engine's confluence_ratio),
    instead of gating entry on any of them individually - every signal
    that reaches build_trade_plan still trades, only the size adapts.
    Linear between CONFLUENCE_SIZING_MIN_MULTIPLIER (no confluence) and
    CONFLUENCE_SIZING_MAX_MULTIPLIER (full confluence). See
    config.CONFLUENCE_SIZING_ENABLED for the rationale."""
    if not config.CONFLUENCE_SIZING_ENABLED:
        return 1.0

    ratio = signal.get("confluence_ratio")

    if ratio is None:
        return 1.0

    min_mult = float(config.CONFLUENCE_SIZING_MIN_MULTIPLIER)
    max_mult = float(config.CONFLUENCE_SIZING_MAX_MULTIPLIER)
    return min_mult + (max_mult - min_mult) * ratio


def build_trade_plan(signal, balance):
    """`signal` is a dict from signal_engine.evaluate() where
    signal["signal"] is "BUY" or "SELL" (never call this for a rejected
    candidate). Returns (plan_dict_or_None, status_reason)."""
    side = signal["signal"]
    symbol = signal["symbol"]
    entry_price = float(signal["entry_price"])

    if entry_price <= 0:
        return None, "INVALID_ENTRY_PRICE"

    sl_price = compute_stop_loss(signal, side)

    if sl_price is None or sl_price <= 0:
        return None, "SL_UNAVAILABLE"

    if side == "BUY" and sl_price >= entry_price:
        return None, "SL_ON_WRONG_SIDE"

    if side == "SELL" and sl_price <= entry_price:
        return None, "SL_ON_WRONG_SIDE"

    risk_distance = abs(entry_price - sl_price)
    extension_r = _entry_extension_r(signal, entry_price, side, risk_distance)

    if _entry_too_extended(extension_r):
        return None, "ENTRY_TOO_EXTENDED"

    if _stop_roi_too_high(risk_distance, entry_price):
        return None, "SL_ROI_TOO_HIGH"

    # config.TP_STATIC_ROI_ENABLED - operator-requested alternative to the
    # TP1(partial)+TP2(remainder) ladder below: ONE fixed ROI% target that
    # closes the whole position at once, computed the same way DCA_TP_
    # STATIC_ROI_ENABLED's post-DCA target already is (price_at_roi_pct).
    # Skips compute_targets/TP1_CLOSE_PCT entirely - there is no partial
    # close or breakeven-on-TP1-fill step in this mode, so tp1_price/
    # tp2_price/tp1_quantity/tp2_quantity are deliberately left None
    # rather than populated with values nothing will use.
    single_tp = bool(config.TP_STATIC_ROI_ENABLED)

    if single_tp:
        tp_price = price_at_roi_pct(entry_price, side, config.TP_TARGET_ROI_PCT)

        if tp_price is None:
            return None, "TARGETS_UNAVAILABLE"

        tp1_price = tp2_price = None
    else:
        tp_price = None
        tp1_price, tp2_price = compute_targets(
            entry_price, sl_price, side, pools=signal.get("liquidity_pools")
        )

        if tp1_price is None:
            return None, "TARGETS_UNAVAILABLE"

    nearest_favorable_sr_r = nearest_favorable_structure_r(
        signal.get("liquidity_pools"), entry_price, side, risk_distance
    )

    size_multiplier = _confluence_size_multiplier(signal)

    if config.RISK_BASED_POSITION_SIZING_ENABLED:
        risk_budget = get_position_risk_budget(balance) * size_multiplier
        quantity = calculate_position_size(
            balance, entry_price, sl_price, symbol, risk_budget_override=risk_budget
        )
    else:
        margin = max(float(config.MARGIN_PER_TRADE), 0) * size_multiplier
        quantity = calculate_position_size(
            balance, entry_price, sl_price, symbol, margin_override=margin
        )

    if quantity <= 0:
        return None, "POSITION_SIZE_ZERO"

    if single_tp:
        tp1_quantity = tp2_quantity = None
    else:
        tp1_close_pct = min(max(float(config.TP1_CLOSE_PCT), 0), 100)
        tp1_quantity = round(quantity * tp1_close_pct / 100, 8)
        tp2_quantity = round(quantity - tp1_quantity, 8)

        if tp1_quantity <= 0 or tp2_quantity <= 0:
            return None, "TP_SPLIT_INVALID"

    # config.DCA_ENABLED - the level that triggers this project's single
    # DCA if price moves against the position before TP1, computed once
    # here at plan time (same as sl_price/tp1_price/tp2_price above) and
    # stored on the position by position_manager.register(). sl_price
    # itself is still computed above and still drives sizing/entry-
    # extension/SL-ROI the same as always - it just never becomes a real
    # resting order until either TP1 fills (existing breakeven-promotion
    # path) or DCA fires (position_manager._execute_dca, which places the
    # first real SL via risk_manager.compute_dca_sl_price). See
    # config.py's DCA section for the full rationale and accepted risk.
    dca_price = (
        compute_dca_price(entry_price, side, signal.get("liquidity_pools"), atr=signal.get("atr"))
        if config.DCA_ENABLED else None
    )
    dca_quantity = round(quantity * max(float(config.DCA_SIZE_MULTIPLIER), 0), 8)

    return {
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        # config.TP_STATIC_ROI_ENABLED - the single-TP shape used instead
        # of tp1_price/tp2_price/tp1_quantity/tp2_quantity above (all None
        # in that case). See position_manager.register_dca_pending for
        # where this actually branches into a differently-shaped position.
        "tp_price": tp_price,
        "single_tp": single_tp,
        "breakeven_price": compute_breakeven_price(entry_price, side),
        "quantity": quantity,
        "tp1_quantity": tp1_quantity,
        "tp2_quantity": tp2_quantity,
        "risk_distance": risk_distance,
        "size_multiplier": size_multiplier,
        "confluence_ratio": signal.get("confluence_ratio"),
        # How far entry_price already ran from the structure level, in R -
        # main.py uses this (against config.ENTRY_ROUTING_EXTENSION_THRESHOLD_R)
        # to route a moderately-extended entry to a limit order instead of
        # a market order, rather than an all-or-nothing switch. Guaranteed
        # non-None here: build_trade_plan already returned SL_UNAVAILABLE
        # above if structure_level/risk_distance weren't available.
        "entry_extension_r": extension_r,
        "nearest_favorable_sr_r": nearest_favorable_sr_r,
        "dca_price": dca_price,
        "dca_quantity": dca_quantity,
        # Carried through so position_manager._execute_dca can recompute
        # a fresh ATR buffer for the post-DCA SL without re-fetching
        # candles/re-running market_structure.analyze() itself.
        "atr": signal.get("atr"),
    }, "OK"
