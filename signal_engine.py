"""Combines HTF bias, LTF structure (CHoCH/BOS into an order block or FVG,
inside the OTE zone), liquidity-sweep confluence, and real-time order-flow
confirmation (CVD + depth imbalance) into a single entry signal.

This replaces v7/v8's strategy.py trend/confirm/entry cascade. The
difference isn't the concepts (structure + confirmation is the same
shape) - it's that every input here is live/event-driven off the websocket
feed, evaluated against the current forming candle, instead of only ever
looking at what already closed.

`evaluate()` always returns a dict with at least `signal` (None or
"BUY"/"SELL") and `reason` - callers that want to see *why* a candidate
was rejected (for the shadow journal) get that for free.
"""
import config
import cvd_divergence
import liquidity_sweep
import market_structure
import oi_divergence


_BULLISH_TO_SIDE = {"BULLISH": "BUY", "BEARISH": "SELL"}


def _reject(reason, **extra):
    return {"signal": None, "reason": reason, **extra}


# Plain reference to the function above, captured before _evaluate_direction
# (inside evaluate()) locally shadows the name `_reject` with a trigger-
# tagging wrapper - see its definition for why.
_reject_plain = _reject


def long_short_favorable(side, long_short_ratio):
    """Contrarian reading of the global long/short account ratio - don't
    buy into an already long-crowded market or short into an already
    short-crowded one. Called from main.py once the on-demand
    long_short_ratio fetch resolves (see config.LONG_SHORT_RATIO_ENABLED
    for why that value can't be computed inside evaluate() itself).
    Informational only, NOT a gate - same treatment as
    efficiency_favorable/funding_favorable above."""
    if long_short_ratio is None:
        return None

    threshold = max(float(config.LONG_SHORT_RATIO_CROWD_THRESHOLD), 0.0001)
    return long_short_ratio < threshold if side == "BUY" else long_short_ratio > 1 / threshold


def evaluate(
    symbol, htf_candles, ltf_candles, cvd_snapshot, depth_snapshot,
    oi_snapshot=None, liquidation_snapshot=None, quote_volume_usdt=None,
    btc_candles=None, funding_rate=None,
):
    if not htf_candles or not ltf_candles:
        return _reject("INSUFFICIENT_CANDLES")

    # Liquidity floor - independent of watchlist selection, see
    # config.MIN_24H_QUOTE_VOLUME_USDT. A symbol with no volume data yet
    # (poll hasn't completed, or the ticker endpoint has nothing for it)
    # is let through rather than blocked - never gate on data we don't
    # actually have.
    min_volume = float(config.MIN_24H_QUOTE_VOLUME_USDT)

    if min_volume > 0 and quote_volume_usdt is not None and quote_volume_usdt < min_volume:
        return _reject("QUOTE_VOLUME_TOO_LOW")

    htf_structure = market_structure.structure_state(htf_candles)

    if not htf_structure.get("available"):
        return _reject("HTF_STRUCTURE_UNAVAILABLE")

    # config.HTF_TREND_FRESHNESS_ENABLED - a second, faster-updating HTF
    # read alongside htf_structure's swing-confirmed trend (see its
    # config.py comment for the real evidence: a stale swing-confirmed
    # bias can persist for many hours after real price has already moved
    # against it, since a swing needs 16 real hours to confirm on the 4h
    # HTF). None with too little HTF history yet - same as every other
    # optional confirmation, absence never blocks a signal on its own.
    htf_trend_ema = None

    if config.HTF_TREND_FRESHNESS_ENABLED:
        htf_trend_ema = market_structure.exponential_moving_average(
            htf_candles, period=config.HTF_TREND_EMA_PERIOD
        )

    zone = market_structure.premium_discount_zone(htf_candles)

    if not zone.get("available"):
        return _reject("ZONE_UNAVAILABLE")

    ltf_analysis = market_structure.analyze(ltf_candles)

    if not ltf_analysis.get("available"):
        return _reject("LTF_STRUCTURE_UNAVAILABLE")

    live_break = ltf_analysis["live_break"]
    latest_price = ltf_candles[-1]["close"]

    # config.LIQUIDITY_SWEEP_TRIGGER_ENABLED - a second, alternative entry
    # trigger for symbols whose price rarely produces a clean structure
    # break but does sweep organized liquidity (see liquidity_sweep.py).
    # Hoisted here (instead of computed later, its original position -
    # see the guarded recompute inside _evaluate_direction below) when
    # EITHER that flag OR config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_
    # ENABLED is on (the latter needs a real `sweep` to additionally
    # confirm against real liquidation flow - see liquidity_sweep.
    # detect_liquidation_confirmed_sweep) - zero added cost across the
    # watchlist every eval tick when BOTH are off (the default) - pools/
    # sweep are computed exactly once either way, never twice, regardless
    # of how many candidates/directions end up being evaluated (see the
    # `nonlocal` note below).
    pools = None
    sweep = None

    if config.LIQUIDITY_SWEEP_TRIGGER_ENABLED or config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED:
        pools = market_structure.find_liquidity_pools(
            market_structure.find_swing_points(ltf_candles)
        )
        sweep = liquidity_sweep.detect_sweep(ltf_candles, pools)

    # config.OB_FVG_RETEST_TRIGGER_ENABLED - a fourth, alternative entry
    # trigger: a fresh rejection wick into an unmitigated FVG, independent
    # of any live break right now. Reuses ltf_analysis's already-computed
    # fair_value_gaps (zero extra cost) rather than recomputing them.
    fvg_retest = None

    if config.OB_FVG_RETEST_TRIGGER_ENABLED:
        fvg_retest = market_structure.find_fvg_retest(
            ltf_candles, fvgs=ltf_analysis["fair_value_gaps"]
        )

    # config.CVD_DIVERGENCE_TRIGGER_ENABLED - a fifth, alternative entry
    # trigger: price's swing structure vs the CVD line at those same swing
    # points (see cvd_divergence.py). Needs its own swing computation, not
    # reused from pools/sweep above (only computed when
    # LIQUIDITY_SWEEP_TRIGGER_ENABLED is on) - same zero-added-cost-when-
    # off shape as every other optional trigger.
    divergence = None

    if config.CVD_DIVERGENCE_TRIGGER_ENABLED:
        divergence_swings = market_structure.find_swing_points(ltf_candles)
        divergence = cvd_divergence.detect_divergence(
            divergence_swings, cvd_snapshot.get("history") or []
        )

    # config.ORDER_BLOCK_RETEST_TRIGGER_ENABLED - a sixth, alternative
    # entry trigger: a fresh rejection wick back into a previously-formed,
    # unmitigated order block (see market_structure.find_order_block_retest).
    order_block_retest = None

    if config.ORDER_BLOCK_RETEST_TRIGGER_ENABLED:
        order_block_retest = market_structure.find_order_block_retest(ltf_candles)

    # config.OI_DIVERGENCE_TRIGGER_ENABLED - a seventh, alternative entry
    # trigger: price's swing structure vs open interest's value at those
    # same swing points (see oi_divergence.py). Reuses the OI history
    # oi_snapshot already carries (OpenInterestEngine.snapshot()'s
    # "history" key) rather than a separate fetch.
    oi_divergence_result = None

    if config.OI_DIVERGENCE_TRIGGER_ENABLED:
        oi_divergence_swings = market_structure.find_swing_points(ltf_candles)
        oi_divergence_result = oi_divergence.detect_divergence(
            oi_divergence_swings, (oi_snapshot or {}).get("history") or []
        )

    # config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED - an eighth,
    # alternative entry trigger: a plain LIQUIDITY_SWEEP additionally
    # confirmed by a real clustered forced-liquidation event (see
    # liquidity_sweep.detect_liquidation_confirmed_sweep). `sweep` is
    # already hoisted above whenever this flag is on.
    liquidation_confirmed_sweep = None

    if config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED:
        liquidation_confirmed_sweep = liquidity_sweep.detect_liquidation_confirmed_sweep(
            sweep, liquidation_snapshot
        )

    # ema_value/btc_correlation/btc_return hoisted here (same gating as
    # before) because neither depends on direction - only the derived
    # ema_aligned/btc_aligned booleans do, computed per-direction inside
    # _evaluate_direction from these shared values. Avoids redundantly
    # recomputing an O(60-candle) EMA sum / O(20-candle) correlation on
    # the (uncommon) tick where both directions get attempted below.
    ema_value = None

    if config.EMA_CONFIRMATION_ENABLED:
        ema_value = market_structure.exponential_moving_average(ltf_candles)

    # config.EMA_PULLBACK_TRIGGER_ENABLED - a ninth, alternative entry
    # trigger: a pullback to the EMA followed by a same-candle reclaim
    # (see market_structure.detect_ema_pullback). Reuses ema_value,
    # already computed just above whenever EMA_CONFIRMATION_ENABLED is
    # on - zero extra cost, no second EMA calculation.
    ema_pullback = None

    if config.EMA_PULLBACK_TRIGGER_ENABLED:
        ema_pullback = market_structure.detect_ema_pullback(ltf_candles, ema_value)

    btc_correlation = None
    btc_return = None

    if (
        config.BTC_CORRELATION_ENABLED
        and btc_candles
        and symbol.upper() != config.CORRELATION_REFERENCE_SYMBOL
    ):
        btc_correlation = market_structure.price_correlation(ltf_candles, btc_candles)
        btc_return = market_structure.price_return(btc_candles)

    # Candidate list - same detection conditions and same priority order
    # as before (STRUCTURE_BREAK > OB_FVG_RETEST > LIQUIDITY_SWEEP >
    # CHOCH_RETEST > CVD_DIVERGENCE > ORDER_BLOCK_RETEST > OI_DIVERGENCE >
    # LIQUIDATION_SWEEP_CONFIRMED > EMA_PULLBACK), but no longer short-
    # circuited into if/elif: every
    # currently-qualifying trigger becomes a candidate, and the selection
    # logic below decides which one actually wins (see
    # config.TRIGGER_QUALITY_RANKING_ENABLED). Building the full list
    # unconditionally costs nothing extra worth caring about - pools/
    # sweep/fvg_retest are already computed unconditionally above
    # whenever their flags are on, and the CHoCH condition is a few dict
    # lookups either way.
    candidates = []

    if live_break.get("broken"):
        candidates.append({
            "signal_trigger": "STRUCTURE_BREAK",
            "direction": live_break["direction"],
            "structure_level": live_break.get("level"),
            "trigger_candle_open_time": live_break.get("open_time"),
        })

    if config.OB_FVG_RETEST_TRIGGER_ENABLED and fvg_retest is not None:
        candidates.append({
            "signal_trigger": "OB_FVG_RETEST",
            "direction": fvg_retest["direction"],
            "structure_level": fvg_retest.get("level"),
            # The candle find_fvg_retest actually tested (see its
            # require_closed_candle behavior) - not blindly
            # ltf_candles[-1], which may be a different, still-forming
            # candle by now.
            "trigger_candle_open_time": fvg_retest.get("open_time"),
        })

    if config.LIQUIDITY_SWEEP_TRIGGER_ENABLED and sweep is not None:
        candidates.append({
            "signal_trigger": "LIQUIDITY_SWEEP",
            "direction": sweep["direction"],
            "structure_level": sweep.get("level"),
            # The candle detect_sweep actually tested (see its
            # require_closed_candle behavior) - position_manager.
            # resolve_break_confirmations already handles this generically
            # whether it's a real value or None (e.g. if detect_sweep ever
            # runs with require_closed_candle=False).
            "trigger_candle_open_time": sweep.get("open_time"),
        })

    if (
        config.CHOCH_RETEST_TRIGGER_ENABLED
        and ltf_analysis.get("last_event")
        and ltf_analysis["last_event"]["type"] == "CHoCH"
        and (len(ltf_candles) - 1 - ltf_analysis["last_event"]["index"])
            <= max(int(config.CHOCH_TRIGGER_MAX_AGE_CANDLES), 0)
    ):
        choch_direction = ltf_analysis["last_event"]["direction"]
        candidates.append({
            "signal_trigger": "CHOCH_RETEST",
            "direction": choch_direction,
            # Deliberately NOT last_event["price"] - that's the price of
            # the NEW pivot that caused the event (e.g. a swing HIGH for
            # a bullish reversal), not the level that was broken. The
            # current retracement level is last_swing_low/last_swing_high,
            # the same fields STRUCTURE_BREAK already derives from.
            "structure_level": (
                ltf_analysis["last_swing_low"] if choch_direction == "BULLISH"
                else ltf_analysis["last_swing_high"]
            ),
            "trigger_candle_open_time": None,
        })

    if (
        config.CVD_DIVERGENCE_TRIGGER_ENABLED
        and divergence is not None
        and (len(ltf_candles) - 1 - divergence["index"])
            <= max(int(config.ORDER_FLOW_DIVERGENCE_LOOKBACK), 0)
    ):
        candidates.append({
            "signal_trigger": "CVD_DIVERGENCE",
            "direction": divergence["direction"],
            "structure_level": divergence.get("level"),
            # Deliberately None, same as CHOCH_RETEST above and for the
            # same reason - divergence.get("open_time") is the OLD swing
            # candle's open_time (already closed well before this tick),
            # not a candle being entered on right now. Passing it through
            # would make position_manager.resolve_break_confirmations
            # compare that swing candle's close back against its own
            # swing price (its own low/high) - trivially true almost
            # every time, a meaningless "confirmation".
            "trigger_candle_open_time": None,
        })

    if config.ORDER_BLOCK_RETEST_TRIGGER_ENABLED and order_block_retest is not None:
        candidates.append({
            "signal_trigger": "ORDER_BLOCK_RETEST",
            "direction": order_block_retest["direction"],
            "structure_level": order_block_retest.get("level"),
            # The candle find_order_block_retest actually tested (see its
            # require_closed_candle behavior) - same shape as OB_FVG_RETEST
            # above. Max-age gating already happened inside
            # find_order_block_retest itself (ORDER_BLOCK_RETEST_MAX_AGE_
            # CANDLES), not repeated here.
            "trigger_candle_open_time": order_block_retest.get("open_time"),
        })

    if (
        config.OI_DIVERGENCE_TRIGGER_ENABLED
        and oi_divergence_result is not None
        and (len(ltf_candles) - 1 - oi_divergence_result["index"])
            <= max(int(config.OI_DIVERGENCE_TRIGGER_MAX_AGE_CANDLES), 0)
    ):
        candidates.append({
            "signal_trigger": "OI_DIVERGENCE",
            "direction": oi_divergence_result["direction"],
            "structure_level": oi_divergence_result.get("level"),
            # Deliberately None, same reasoning as CVD_DIVERGENCE above -
            # this is the OLD swing candle's open_time, not a candle being
            # entered on right now.
            "trigger_candle_open_time": None,
        })

    if (
        config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED
        and liquidation_confirmed_sweep is not None
    ):
        candidates.append({
            "signal_trigger": "LIQUIDATION_SWEEP_CONFIRMED",
            "direction": liquidation_confirmed_sweep["direction"],
            "structure_level": liquidation_confirmed_sweep.get("level"),
            # Same shape as LIQUIDITY_SWEEP above - liquidation_confirmed_
            # sweep is a copy of the same close-confirmed sweep dict, just
            # additionally gated on real liquidation flow.
            "trigger_candle_open_time": liquidation_confirmed_sweep.get("open_time"),
        })

    if config.EMA_PULLBACK_TRIGGER_ENABLED and ema_pullback is not None:
        candidates.append({
            "signal_trigger": "EMA_PULLBACK",
            "direction": ema_pullback["direction"],
            "structure_level": ema_pullback.get("level"),
            # The candle detect_ema_pullback actually tested (see its
            # require_closed_candle behavior) - same shape as
            # LIQUIDITY_SWEEP/OB_FVG_RETEST/ORDER_BLOCK_RETEST above.
            "trigger_candle_open_time": ema_pullback.get("open_time"),
        })

    if not candidates:
        return _reject("NO_LIVE_STRUCTURE_BREAK")

    def _evaluate_direction(direction, trigger):
        """Everything below this point depends only on `direction`/`side`
        for every gate except NOT_IN_OTE (see config.OTE_GATE_STRUCTURE_
        BREAK_ONLY_ENABLED) - which is why the caller now dedupes on
        (direction, trigger), not direction alone. `structure_level`/
        `trigger_candle_open_time`/`signal_trigger` are left as None
        placeholders in the success dict - the caller overlays the
        winning candidate's real values afterward."""
        nonlocal pools, sweep

        # Which trigger(s) actually attempted this direction - a rejection
        # inside this function currently can't be told apart by trigger at
        # all (gates run once per direction, shared across every candidate
        # that direction has), so main.py's reject tally has never been
        # able to answer "how often does CVD_DIVERGENCE specifically die
        # at CVD_NOT_CONFIRMED" - only "how often does CVD_NOT_CONFIRMED
        # fire". Shadows the module-level _reject for the rest of this
        # function only, so every existing `return _reject(...)` call
        # below picks this up with zero changes to each call site. Purely
        # additive (a new "triggers" key, `reason` itself untouched) -
        # deliberately not baked into the reason string, so this can't
        # affect any existing test or the main reject-reason tally's
        # existing keys/counts.
        _triggers_for_direction = sorted({
            candidate["signal_trigger"] for candidate in candidates
            if candidate["direction"] == direction
        })

        def _reject(reason, **extra):
            return _reject_plain(reason, triggers=_triggers_for_direction, **extra)

        side = _BULLISH_TO_SIDE.get(direction)

        if side is None:
            return _reject("UNKNOWN_BREAK_DIRECTION")

        # config.TRIGGER_GATE_PROFILES - this project's founding difference
        # from binance_ai_bot_smc: each trigger only runs the gates that
        # actually fit its own detection logic, by default (see config.py's
        # PER-TRIGGER GATE MATCHING section for the full per-gate
        # reasoning). Every `"GATE_NAME" in applicable_gates` check below
        # replaces what used to be a bespoke `..._applies = not (...)`
        # expression per gate.
        applicable_gates = config.trigger_gate_profiles().get(trigger, frozenset())

        if "AGAINST_HTF_BIAS" in applicable_gates:
            htf_side = _BULLISH_TO_SIDE.get(htf_structure.get("trend"))

            if htf_side and side != htf_side:
                return _reject(f"AGAINST_HTF_BIAS htf={htf_structure.get('trend')} ltf={direction}")

        if "HTF_TREND_STALE" in applicable_gates and htf_trend_ema is not None:
            if side == "BUY" and latest_price < htf_trend_ema:
                return _reject("HTF_TREND_STALE")

            if side == "SELL" and latest_price > htf_trend_ema:
                return _reject("HTF_TREND_STALE")

        # config.EFFICIENCY_RATIO_GATE_ENABLED - genuine chop (price
        # round-tripping, no real directional move) is a failure mode
        # neither AGAINST_HTF_BIAS nor HTF_TREND_STALE can catch:
        # structure_state()'s swing-confirmed trend can only ever report
        # BULLISH or BEARISH, never "no real trend right now", so it
        # freezes on a stale read through genuine chop instead of
        # recognizing it - and chop doesn't move price decisively away
        # from the EMA either, so HTF_TREND_STALE reads "still fine" the
        # whole time. Checked here (before zone/OTE/OB-FVG/CVD/depth)
        # since it's a market-regime fact independent of where within a
        # move the entry sits - fetched early on purpose, ltf_analysis
        # already has it for free from the top of evaluate().
        efficiency_ratio = ltf_analysis.get("efficiency_ratio")

        if (
            "MARKET_CHOPPY" in applicable_gates
            and config.EFFICIENCY_RATIO_GATE_ENABLED
            and efficiency_ratio is not None
            and efficiency_ratio < config.EFFICIENCY_RATIO_CHOP_THRESHOLD
        ):
            return _reject("MARKET_CHOPPY")

        # NOT_IN_DISCOUNT/NOT_IN_PREMIUM - universal, applies to every
        # trigger (see config.py's TRIGGER_GATE_PROFILES docstring): a
        # reversal-at-extreme signal naturally already sits on the
        # matching side of this same HTF range, so this gate's geometry
        # agrees with reversal triggers rather than conflicting with them,
        # unlike NOT_IN_OTE below.
        price_zone = market_structure.zone_for_price(zone, latest_price)

        if side == "BUY" and price_zone != "DISCOUNT":
            return _reject(f"NOT_IN_DISCOUNT price_zone={price_zone}")

        if side == "SELL" and price_zone != "PREMIUM":
            return _reject(f"NOT_IN_PREMIUM price_zone={price_zone}")

        # NOT_IN_OTE checks whether CURRENT PRICE sits within a Fibonacci
        # retracement band of the OVERALL HTF range - the classic "break,
        # then retrace to OTE" setup, a real fit for STRUCTURE_BREAK
        # specifically (see config.py's TRIGGER_GATE_PROFILES section -
        # every other trigger anchors to its own, different zone with no
        # structural reason to also sit in this band).
        if (
            "NOT_IN_OTE" in applicable_gates
            and not market_structure.in_ote(zone, latest_price, direction)
        ):
            return _reject("NOT_IN_OTE")

        # How deep into the range this entry's retracement actually is,
        # using the SAME measure OTE_RETRACEMENT_MIN/MAX are expressed in
        # (0 = at the range extreme, 1 = fully retraced to the opposite
        # extreme) - purely diagnostic, not a second gate (in_ote above
        # already enforces the MIN/MAX band). Real motivation (2026-08-14,
        # operator feedback): signals were seen firing in discount/premium
        # while the underlying move hadn't actually finished - journaling
        # the real depth lets journal_analysis.py test whether shallower
        # retracements within the qualifying band lose more, instead of
        # guessing how much further to tighten OTE_RETRACEMENT_MIN.
        zone_range = zone["range_high"] - zone["range_low"]
        zone_retracement_pct = (
            (zone["range_high"] - latest_price) / zone_range if direction == "BULLISH"
            else (latest_price - zone["range_low"]) / zone_range
        )

        order_block = market_structure.find_order_block(
            ltf_candles, len(ltf_candles) - 1, direction
        )
        fvgs = ltf_analysis["fair_value_gaps"]
        matching_fvg = next(
            (gap for gap in reversed(fvgs) if gap["type"] == direction), None
        )

        # NO_ORDER_BLOCK_OR_FVG is tautologically always-satisfied for
        # OB_FVG_RETEST/ORDER_BLOCK_RETEST (the trigger IS "retesting an
        # OB/FVG") - config.TRIGGER_GATE_PROFILES excludes it from their
        # profile, so this check is skipped rather than trivially passed.
        if (
            "NO_ORDER_BLOCK_OR_FVG" in applicable_gates
            and config.REQUIRE_ORDER_BLOCK_OR_FVG
            and not order_block and not matching_fvg
        ):
            return _reject("NO_ORDER_BLOCK_OR_FVG")

        # Informational only, not gating: an EMA is a lagging/smoothed
        # indicator by construction, so requiring price to already be on
        # its correct side can delay entry on a sharp move until price
        # has run further - a real cost against this bot's real-time
        # premise, and one not yet backed by evidence it's worth paying.
        # ema_value itself is hoisted above (direction-independent) -
        # only the alignment derived from it varies per direction.
        ema_aligned = None

        if ema_value is not None:
            ema_aligned = (
                latest_price > ema_value if side == "BUY" else latest_price < ema_value
            )

        # Open Interest: informational only, NOT a gate - see
        # config.OI_CONFIRMATION_ENABLED for rationale. Rising OI during
        # this break points at fresh positioning, not just the other
        # side closing.
        oi_snapshot_ = oi_snapshot or {}
        oi_change_pct = None
        oi_rising = None

        if config.OI_CONFIRMATION_ENABLED and oi_snapshot_.get("available"):
            oi_change_pct = oi_snapshot_.get("oi_change_pct")

            if oi_change_pct is not None:
                oi_rising = oi_change_pct > 0

        # OI_RISING - a real gate, unlike every other field in this
        # function (see config.OI_RISING_REJECT_ENABLED for the evidence).
        # Universal across triggers (same as NOT_IN_DISCOUNT/DEPTH_OPPOSING
        # below), not read from applicable_gates - no per-trigger split was
        # found to differentiate on. Reason string deliberately carries no
        # continuous value (same convention as CVD_NOT_CONFIRMED/
        # DEPTH_OPPOSING below) so main.py's reject-reason tally aggregates
        # it into one count instead of one per oi_change_pct value.
        if config.OI_RISING_REJECT_ENABLED and oi_rising:
            return _reject("OI_RISING")

        # Liquidation clustering: informational only, NOT a gate - see
        # config.LIQUIDATION_CONFIRMATION_ENABLED for rationale. A
        # BULLISH break aligns with long-liquidation flow (forced SELL
        # orders as the swept low was hit); a BEARISH break aligns with
        # short-liquidation flow (forced BUY orders as the swept high
        # was hit).
        liquidation_snapshot_ = liquidation_snapshot or {}
        liquidation_notional_net = None
        liquidation_cluster = None
        liquidation_aligned = None

        if config.LIQUIDATION_CONFIRMATION_ENABLED and liquidation_snapshot_.get("available"):
            liquidation_notional_net = liquidation_snapshot_.get("net_liquidation_notional")
            total_notional = (
                liquidation_snapshot_.get("long_liquidation_notional", 0)
                + liquidation_snapshot_.get("short_liquidation_notional", 0)
            )
            liquidation_cluster = total_notional >= config.LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT

            if liquidation_notional_net is not None:
                liquidation_aligned = (
                    liquidation_notional_net > 0 if direction == "BULLISH"
                    else liquidation_notional_net < 0
                )

        if not cvd_snapshot.get("available"):
            return _reject("ORDER_FLOW_DATA_UNAVAILABLE")

        cvd_score = cvd_snapshot.get("cvd_score")

        if cvd_score is None:
            return _reject("ORDER_FLOW_SCORE_UNAVAILABLE")

        min_cvd = config.SIGNAL_MIN_CVD_SCORE

        # CVD_NOT_CONFIRMED - excluded from CVD_DIVERGENCE's profile (see
        # config.py's TRIGGER_GATE_PROFILES section): only skips the
        # threshold comparison just below, never the ORDER_FLOW_DATA_
        # UNAVAILABLE/ORDER_FLOW_SCORE_UNAVAILABLE checks above (those are
        # "is there any data at all", not a direction-agreement
        # comparison, and apply the same regardless of trigger).
        #
        # Reason deliberately doesn't embed the raw score/imbalance value
        # (same for DEPTH_OPPOSING and QUOTE_VOLUME_TOO_LOW below) - a
        # continuous value baked into the reason string means every
        # rejection gets its own distinct key, so main.py's reject-reason
        # tally (a Counter keyed on this string) can never aggregate them
        # into one meaningful count. The per-symbol detail that used to
        # live in the number now lives in the tally's symbol sample
        # instead.
        if "CVD_NOT_CONFIRMED" in applicable_gates:
            if side == "BUY" and cvd_score < min_cvd:
                return _reject("CVD_NOT_CONFIRMED")

            if side == "SELL" and cvd_score > -min_cvd:
                return _reject("CVD_NOT_CONFIRMED")

        depth_imbalance = None

        if depth_snapshot.get("available"):
            depth_imbalance = depth_snapshot.get("depth_imbalance", 0)
            min_depth = config.SIGNAL_MIN_DEPTH_IMBALANCE

            if side == "BUY" and depth_imbalance < -min_depth:
                return _reject("DEPTH_OPPOSING")

            if side == "SELL" and depth_imbalance > min_depth:
                return _reject("DEPTH_OPPOSING")

        if pools is None:
            # Not already computed above - either
            # LIQUIDITY_SWEEP_TRIGGER_ENABLED is off, or it's on but the
            # structure break fired first and this candidate never
            # needed sweep as a trigger. Still worth computing now for
            # the sweep_confluence informational field below. Assigned
            # via `nonlocal` so this only ever runs once total, even if
            # _evaluate_direction runs a second time for the other
            # direction.
            pools = market_structure.find_liquidity_pools(
                market_structure.find_swing_points(ltf_candles)
            )
            sweep = liquidity_sweep.detect_sweep(ltf_candles, pools)

        sweep_confluence = bool(sweep and sweep["direction"] == direction)

        # BTC correlation: informational only, NOT a gate - see
        # config.BTC_CORRELATION_ENABLED. Most alts move because BTC
        # moves, not from their own structure; skipped entirely when
        # evaluating BTC itself (self-correlation is meaningless).
        # btc_correlation/btc_return are hoisted above (direction-
        # independent) - only the alignment derived from them varies per
        # direction.
        btc_aligned = None

        if btc_return is not None:
            btc_aligned = (btc_return > 0) if side == "BUY" else (btc_return < 0)

        # Funding rate: informational only, NOT a gate - see
        # config.FUNDING_RATE_ENABLED. Strongly positive means longs are
        # paying heavily to stay long (a crowded trade, more prone to a
        # squeeze/reversal); strongly negative is the mirror image.
        funding_rate_ = funding_rate if config.FUNDING_RATE_ENABLED else None

        # Boolean "favorable" readings - informational only, journaled
        # but deliberately NOT added to confluence_fields/confluence_ratio
        # below (see config.EFFICIENCY_RATIO_CHOP_THRESHOLD: that sizing
        # mechanism is disabled today on real negative evidence from its
        # existing 5 fields - mixing new, unvalidated ones into it would
        # contaminate any future read of either). Kept independently
        # named/journaled so journal_analysis.py can evaluate each on its
        # own merits before any decision to fold it into sizing. Note
        # efficiency_ratio itself is now ALSO a real gate above
        # (config.EFFICIENCY_RATIO_GATE_ENABLED, checked earlier in this
        # function) - this boolean is a separate, informational-only
        # reading of the same underlying value, unaffected by that.
        efficiency_favorable = (
            efficiency_ratio > config.EFFICIENCY_RATIO_CHOP_THRESHOLD
            if efficiency_ratio is not None else None
        )
        funding_favorable = None

        if funding_rate_ is not None:
            adverse = config.FUNDING_RATE_ADVERSE_THRESHOLD
            funding_favorable = (
                funding_rate_ <= adverse if side == "BUY" else funding_rate_ >= -adverse
            )

        # Confluence score: how many of the informational fields above
        # agree with this signal's direction, out of how many were
        # actually available to check (a field that's None - e.g.
        # liquidation, still silent for most alts - is excluded from the
        # denominator rather than counted against the signal). Feeds
        # risk_manager's confluence-weighted position sizing instead of
        # gating entry on any of these individually - every signal that
        # reaches here still trades, only the size adapts. See
        # config.CONFLUENCE_SIZING_ENABLED.
        confluence_fields = [
            sweep_confluence, ema_aligned, oi_rising, liquidation_aligned, btc_aligned,
        ]
        confluence_available = [value for value in confluence_fields if value is not None]
        confluence_total = len(confluence_available)
        confluence_score = sum(1 for value in confluence_available if value)
        confluence_ratio = confluence_score / confluence_total if confluence_total else None

        return {
            "signal": side,
            "reason": "OK",
            "symbol": symbol,
            "entry_price": latest_price,
            "htf_trend": htf_structure.get("trend"),
            # Placeholders - the candidate (not the direction) determines
            # these; the caller overlays the winning candidate's real
            # values once a direction's pipeline has actually passed.
            "structure_level": None,
            "trigger_candle_open_time": None,
            "signal_trigger": None,
            "quote_volume_usdt": quote_volume_usdt,
            "order_block": order_block,
            "fvg": matching_fvg,
            "sweep_confluence": sweep_confluence,
            "cvd_score": cvd_score,
            "depth_imbalance": depth_imbalance,
            "atr": ltf_analysis.get("atr"),
            "premium_discount_zone": price_zone,
            "zone_retracement_pct": zone_retracement_pct,
            "liquidity_pools": pools,
            "ema_value": ema_value,
            "ema_aligned": ema_aligned,
            "oi_change_pct": oi_change_pct,
            "oi_rising": oi_rising,
            "liquidation_notional_net": liquidation_notional_net,
            "liquidation_cluster": liquidation_cluster,
            "liquidation_aligned": liquidation_aligned,
            "efficiency_ratio": efficiency_ratio,
            "efficiency_favorable": efficiency_favorable,
            "btc_correlation": btc_correlation,
            "btc_aligned": btc_aligned,
            "funding_rate": funding_rate_,
            "funding_favorable": funding_favorable,
            # long_short_ratio/long_short_favorable are NOT set here - the
            # raw ratio is only fetched on-demand in main.py, after a
            # candidate has already passed every check in this function (see
            # config.LONG_SHORT_RATIO_ENABLED - no bulk endpoint exists for
            # it). main.py calls long_short_favorable() below once it has the
            # real value.
            "confluence_score": confluence_score,
            "confluence_total": confluence_total,
            "confluence_ratio": confluence_ratio,
        }

    # Selection: unchanged single-candidate path when
    # TRIGGER_QUALITY_RANKING_ENABLED is off (byte-identical to the old
    # if/elif chain - only ever attempts candidates[0], the same one the
    # old code would have picked). When on, attempt every candidate.
    # Deduped via pipeline_results keyed on (direction, trigger), not
    # direction alone - most gates still give every same-direction
    # candidate an identical verdict (see config.OTE_GATE_STRUCTURE_BREAK_
    # ONLY_ENABLED's comment for the one that no longer does), but once
    # ANY gate can vary by trigger, two same-direction candidates can no
    # longer be assumed interchangeable. Real cost: up to len(candidates)
    # gate-cascade runs per tick instead of at most 2 - negligible at this
    # bot's scale (candidate counts per tick are rarely more than 2-3, and
    # nonlocal pools/sweep still only ever compute once regardless - see
    # their own guard below).
    selected = candidates if config.TRIGGER_QUALITY_RANKING_ENABLED else [candidates[0]]
    pipeline_results = {}
    passing = []

    for candidate in selected:
        direction = candidate["direction"]
        trigger = candidate["signal_trigger"]
        key = (direction, trigger)

        if key not in pipeline_results:
            pipeline_results[key] = _evaluate_direction(direction, trigger)

        result = pipeline_results[key]

        if result["signal"] is not None:
            passing.append((candidate, result))

    if not passing:
        # Top-priority ATTEMPTED candidate's own reason - preserves
        # today's reject-tally behavior exactly in the (common)
        # single-candidate case, and avoids inventing a new composite
        # reject string that would fragment main.py's reject-reason tally.
        first = selected[0]
        return pipeline_results[(first["direction"], first["signal_trigger"])]

    if len(passing) == 1:
        winner_candidate, winner_result = passing[0]
    else:
        # Ranking: prefer whichever same-direction candidate's
        # structure_level sits closest to current price (least
        # already-chased - same philosophy as risk_manager's
        # MAX_ENTRY_EXTENSION_R), but only override the fixed-priority
        # default (passing[0] - candidates was built in priority order)
        # when the edge is real, not infinitesimal - see
        # config.TRIGGER_QUALITY_EDGE_ATR_MULTIPLE for why (prevents
        # ordinary tick-to-tick price noise from flipping the winner and
        # resetting main.py's SignalStabilityTracker streak every time).
        # A candidate's structure_level CAN legitimately be None (e.g.
        # CHOCH_RETEST when only one side of last_swing_high/low has
        # formed yet) - score it as worst-possible (inf) rather than
        # crashing the abs() subtraction, so it's never preferred over a
        # scoreable alternative but can still win if it's the only
        # passing candidate at all (the len(passing)==1 branch above).
        def _score(candidate):
            level = candidate["structure_level"]
            return abs(latest_price - level) if level is not None else float("inf")

        default_candidate, default_result = passing[0]
        best_candidate, best_result = min(passing, key=lambda pair: _score(pair[0]))

        if best_candidate is default_candidate:
            winner_candidate, winner_result = default_candidate, default_result
        else:
            atr = ltf_analysis.get("atr") or 0
            edge = atr * max(float(config.TRIGGER_QUALITY_EDGE_ATR_MULTIPLE), 0)
            default_score = _score(default_candidate)
            best_score = _score(best_candidate)

            # edge=0 disables the hysteresis entirely (always take the
            # best-scored candidate) - best_score < default_score is
            # already guaranteed here (best_candidate is not
            # default_candidate, by construction of the min() above), so
            # this difference is always >= 0.
            if (default_score - best_score) >= edge:
                winner_candidate, winner_result = best_candidate, best_result
            else:
                winner_candidate, winner_result = default_candidate, default_result

    result = dict(winner_result)
    result["structure_level"] = winner_candidate["structure_level"]
    result["trigger_candle_open_time"] = winner_candidate["trigger_candle_open_time"]
    result["signal_trigger"] = winner_candidate["signal_trigger"]
    return result
