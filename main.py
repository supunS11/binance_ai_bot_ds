"""Orchestrator: real-time data -> structure/order-flow signal -> risk plan
-> entry -> TP1/TP2/SL -> breakeven -> close, end to end.

Defaults to SHADOW execution (config.EXECUTION_MODE) - signals are still
fully evaluated, sized, and journaled, and open "positions" are tracked
and resolved against live price action, but no real order is placed until
EXECUTION_MODE is explicitly switched to LIVE in .env. This is the
evidence-gate from the original plan: review shadow signal quality first.
"""
from collections import Counter
import threading
import time

import config
import exchange
import execution
import risk_manager
import signal_engine
import signal_journal
from logger import log_error, log_info, log_warning
from position_manager import PENDING_LIMIT_FILL, RETRACEMENT_PENDING, PositionManager
from ws_client import RealtimeMarketData


shutdown_event = threading.Event()


def _select_symbols():
    if config.SCAN_SYMBOLS:
        return list(config.SCAN_SYMBOLS)

    supported = exchange.get_supported_symbols()

    if not supported:
        log_error("No supported symbols resolved from exchange info; aborting")
        return []

    volumes = exchange.get_24h_quote_volumes()

    if volumes:
        ranked = sorted(supported, key=lambda symbol: volumes.get(symbol, 0), reverse=True)
    else:
        ranked = sorted(supported)

    return ranked[: max(config.WATCHLIST_SIZE, 1)]


def _current_balance():
    if config.EXECUTION_MODE == "LIVE":
        return exchange.get_balance()

    return config.SHADOW_ACCOUNT_BALANCE_USDT


_MAX_REJECT_SAMPLE_SYMBOLS = 5


def _tally_reject(reject_counts, reject_symbols, symbol, reason):
    """Records one rejection: always a count, plus (capped) which symbols
    actually triggered it. The cap matters - NO_LIVE_STRUCTURE_BREAK alone
    can fire for hundreds of symbols per heartbeat window, so this is a
    representative sample for eyeballing "which symbols", not a full list."""
    if reject_counts is None:
        return

    reject_counts[reason] += 1

    if reject_symbols is None:
        return

    sample = reject_symbols.setdefault(reason, [])

    if symbol not in sample and len(sample) < _MAX_REJECT_SAMPLE_SYMBOLS:
        sample.append(symbol)


class SignalStabilityTracker:
    """Requires a signal to keep qualifying for config.SIGNAL_CONFIRM_TICKS
    consecutive evaluations - not just the single instant it first appears
    - before it's acted on. See config.SIGNAL_CONFIRM_TICKS for the real
    incident (IOTXUSDT) that motivated this: CVD flipped pass/fail within
    16 seconds, and the resulting entry sat flat for 90+ minutes before
    losing. A symbol's streak resets whenever it stops qualifying, flips
    to the opposite side, or successfully enters a trade (so stale streak
    state can never leak into a later, unrelated setup on the same symbol
    once it's eligible again after that position closes and its cooldown
    passes)."""

    def __init__(self):
        self._streaks = {}  # symbol -> (side, trigger, consecutive_qualifying_ticks)

    def confirm(self, symbol, side, trigger=None):
        """Call once per eval tick for a symbol with a currently-qualifying
        signal. Returns True once `side` has qualified for the required
        number of consecutive calls.

        `trigger` (signal_engine's signal_trigger, e.g. "STRUCTURE_BREAK")
        raises the bar for every trigger, STRUCTURE_BREAK included - see
        config.STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS (a smaller bump, kept
        below config.EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS so the one
        trigger with a real track record isn't slowed down as much as the
        newer/unproven ones). The streak is keyed on (side, trigger), not
        just side: a candidate
        that qualified via one trigger type for a few ticks then flips to
        a different trigger type shouldn't inherit that count - it's a
        different setup. Known, accepted corner case: this also resets the
        streak when a setup is genuinely STRENGTHENING (e.g. only
        LIQUIDITY_SWEEP true on tick 1, STRUCTURE_BREAK also becomes true
        and wins on priority on tick 2) even though two triggers now
        agree - same simplicity tradeoff as the rest of this mechanism."""
        required = max(int(config.SIGNAL_CONFIRM_TICKS), 1)

        if trigger == "STRUCTURE_BREAK":
            required += max(int(config.STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS), 0)
        elif trigger is not None:
            required += max(int(config.EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS), 0)

        existing_side, existing_trigger, count = self._streaks.get(symbol, (None, None, 0))
        count = count + 1 if (existing_side == side and existing_trigger == trigger) else 1
        self._streaks[symbol] = (side, trigger, count)
        return count >= required

    def reset(self, symbol):
        self._streaks.pop(symbol, None)


def _evaluate_symbol(
    feed, symbol, positions, balance,
    reject_counts=None, reject_symbols=None, stability=None,
    reject_trigger_counts=None, reject_trigger_symbols=None,
):
    """`reject_counts`/`reject_symbols` (optional) tally why candidates
    don't convert to a trade - signal_engine.evaluate()'s `reason`, plus a
    couple of pre-signal/post-signal cases of our own - and which symbols
    triggered each one. Real gap found live (2026-08-11): with 0 open
    positions and a slow 1h/4h timeframe, "no entries" was completely
    unexplainable from the logs - every rejection reason was silently
    discarded, so there was no way to tell "working as intended, genuinely
    no qualifying setups yet" apart from "something is over-restrictive".
    This makes that visible via the heartbeat instead of guessing.
    Deliberately excludes the has_open_position/cooldown/max_positions
    early-exits above - those are routine operational skips, not signal-
    quality rejections, and would just dilute the tally.

    `reject_trigger_counts`/`reject_trigger_symbols` (optional, separate
    from the pair above - never mixed into the same tally) - a second,
    finer-grained breakdown answering "which TRIGGER dies at this gate",
    not just "how often does this gate fire". Real gap: gates run once per
    DIRECTION, shared across every candidate sharing it (see signal_engine.
    _evaluate_direction's own docstring), so a plain rejection has never
    been attributable to one specific trigger before - see signal_engine.
    evaluate()'s new `triggers` field on a reject dict. Kept as an entirely
    separate tally/heartbeat line rather than folded into reject_counts'
    existing keys, so the established top-8 heartbeat summary's format and
    counts are completely unaffected by this."""
    if positions.has_open_position(symbol):
        return

    if positions.is_in_cooldown(symbol):
        return

    # Real (non-shadow) positions only - a shadow trade (config.
    # SHADOW_ONLY_TRIGGERS or EXECUTION_MODE=SHADOW) never touches the
    # exchange, so it shouldn't compete with real trades for real capital
    # capacity. Real positions can still fill every MAX_TOTAL_POSITIONS
    # slot regardless of how many shadow trades are also being tracked.
    if positions.real_open_count() >= config.MAX_TOTAL_POSITIONS:
        return

    ltf_candles = feed.candles.get(symbol)
    htf_candles = feed.htf_candles.get(symbol)

    if not ltf_candles or not htf_candles:
        _tally_reject(reject_counts, reject_symbols, symbol, "NO_CANDLE_DATA")
        return

    cvd_snapshot = feed.cvd.snapshot(symbol)
    depth_snapshot = feed.depth.snapshot(symbol)
    oi_snapshot = feed.open_interest.snapshot(symbol)
    oi_snapshot_bybit = feed.open_interest_bybit.snapshot(symbol)
    oi_snapshot_okx = feed.open_interest_okx.snapshot(symbol)
    volume_profile_snapshot = feed.volume_profile.snapshot(symbol)
    liquidation_snapshot = feed.liquidations.snapshot(symbol)
    liquidation_snapshot_bybit = feed.liquidations_bybit.snapshot(symbol)
    liquidation_snapshot_okx = feed.liquidations_okx.snapshot(symbol)
    quote_volume_usdt = feed.volumes.get(symbol)
    btc_candles = feed.candles.get(config.CORRELATION_REFERENCE_SYMBOL)
    funding_rate = feed.funding_rates.get(symbol)
    crash_snapshot = feed.crash_detector.snapshot()

    result = signal_engine.evaluate(
        symbol, htf_candles, ltf_candles, cvd_snapshot, depth_snapshot,
        oi_snapshot=oi_snapshot, liquidation_snapshot=liquidation_snapshot,
        quote_volume_usdt=quote_volume_usdt, btc_candles=btc_candles,
        funding_rate=funding_rate, crash_snapshot=crash_snapshot,
        oi_snapshot_bybit=oi_snapshot_bybit, oi_snapshot_okx=oi_snapshot_okx,
        volume_profile_snapshot=volume_profile_snapshot,
        liquidation_snapshot_bybit=liquidation_snapshot_bybit,
        liquidation_snapshot_okx=liquidation_snapshot_okx,
    )

    if not result.get("signal"):
        if stability is not None:
            stability.reset(symbol)

        reason = result.get("reason") or "UNKNOWN"
        _tally_reject(reject_counts, reject_symbols, symbol, reason)

        triggers = result.get("triggers")

        if triggers:
            _tally_reject(
                reject_trigger_counts, reject_trigger_symbols, symbol,
                f"{reason} | triggers={','.join(triggers)}",
            )

        return

    if stability is not None and not stability.confirm(
        symbol, result["signal"], result.get("signal_trigger")
    ):
        _tally_reject(reject_counts, reject_symbols, symbol, "SIGNAL_NOT_YET_STABLE")
        return

    # Long/short ratio: fetched on-demand here, not polled across the
    # whole watchlist like the fields above - see
    # config.LONG_SHORT_RATIO_ENABLED for why (no bulk endpoint exists for
    # it). Only reached for a candidate that's already passed every other
    # check, so this is at most a handful of REST calls per hour, not one
    # per symbol per poll cycle.
    if config.LONG_SHORT_RATIO_ENABLED:
        result["long_short_ratio"] = exchange.get_long_short_ratio(symbol)
        result["long_short_favorable"] = signal_engine.long_short_favorable(
            result["signal"], result["long_short_ratio"]
        )

    plan, status = risk_manager.build_trade_plan(result, balance)

    if status != "OK":
        _tally_reject(reject_counts, reject_symbols, symbol, f"PLAN_REJECTED:{status}")
        # trigger/entry_price/structure_level (all already computed by
        # signal_engine, zero extra cost) - real gap found live (2026-08-16):
        # ENTRY_TOO_EXTENDED was ~99% of everything reaching this point for
        # some symbols (GALAUSDT/PNUTUSDT, real-price-traced), repeating for
        # an hour+ straight with nothing in the log to tell apart "price
        # genuinely keeps running away" (a real, working reject - see
        # HUMAUSDT's trace the same day) from "structure_level itself is
        # stale/fixed while price just chops nearby" (a real bug candidate).
        # structure_level logged here across repeated occurrences for the
        # same symbol is the direct tell: constant -> stale reference;
        # moving with price -> genuinely still-extending trend.
        log_info(
            f"{symbol} signal found but plan rejected | REASON={status} "
            f"trigger={result.get('signal_trigger')} entry~={result.get('entry_price')} "
            f"structure_level={result.get('structure_level')}"
        )
        return

    # Carried through to the position so resolve_break_confirmations() can
    # later check, once this exact candle actually finishes, whether price
    # held beyond the level it broke or snapped back inside first (just a
    # wick, not a real break) - see PositionManager.resolve_break_confirmations.
    # Sourced from the signal itself (the candle signal_engine actually
    # evaluated the break against), NOT blindly ltf_candles[-1] - with
    # REQUIRE_CLOSE_CONFIRMED_BREAK enabled that's the last CLOSED candle,
    # which may already differ from whatever candle is forming by the time
    # this line runs.
    plan["structure_level"] = result.get("structure_level")
    plan["trigger_candle_open_time"] = result.get("trigger_candle_open_time")
    # config.SHADOW_ONLY_TRIGGERS - execution._is_shadow_mode reads this
    # off plan, not result, to decide per-trigger shadow routing.
    plan["signal_trigger"] = result.get("signal_trigger")

    # config.TP_STATIC_ROI_ENABLED - see position_manager's heartbeat log
    # comment for why single-TP plans need their own display.
    tp_note = (
        f"TP={plan['tp_price']}" if plan.get("single_tp")
        else f"TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
    )
    log_info(
        f"{symbol} SIGNAL {result['signal']} | entry~={plan['entry_price']} "
        f"SL={plan['sl_price']} {tp_note} | "
        f"cvd={result.get('cvd_score')} sweep={result.get('sweep_confluence')} "
        f"htf_trend={result.get('htf_trend')}"
    )

    # config.LIMIT_ENTRY_MODE_ENABLED is a per-signal ROUTING switch, not
    # "always place a limit order" - a signal still close to the
    # structure level (low entry_extension_r) gets a market order, since
    # a guaranteed fill beats limit fill-uncertainty when the chase cost
    # is minimal anyway. Only a signal that's already moderately extended
    # (above the threshold, but under the hard MAX_ENTRY_EXTENSION_R
    # reject risk_manager already applied) gets routed to a resting limit
    # instead. entry_extension_r is always a real float here (build_trade_plan
    # only returns "OK" once it's computable), so the None branch below
    # is a defensive fallback (market), not an expected path.
    # config.RETRACEMENT_ENTRY_ENABLED takes priority over everything
    # below - it applies uniformly regardless of DCA_ENABLED/entry_
    # extension_r, and once its fill (or bounded market fallback) is
    # resolved it hands off into DCA_PENDING or TP1_PENDING via the exact
    # same register_dca_pending()/register() paths those branches use
    # directly - see position_manager.RETRACEMENT_PENDING/
    # _finalize_retracement_entry.
    #
    # config.DCA_ENABLED takes priority over LIMIT_ENTRY_MODE_ENABLED's own
    # market-vs-limit routing below - this project's DCA design (market
    # entry, no SL, TP1/TP2 live) was never specified alongside a resting
    # limit entry, and layering a third pending state on top of
    # PENDING_LIMIT_FILL/DCA_PENDING would add real complexity nothing
    # asked for. A DCA-enabled signal always enters at market.
    use_retracement = config.RETRACEMENT_ENTRY_ENABLED
    use_limit = (
        not use_retracement
        and not config.DCA_ENABLED
        and config.LIMIT_ENTRY_MODE_ENABLED
        and plan.get("entry_extension_r") is not None
        and plan["entry_extension_r"] > config.ENTRY_ROUTING_EXTENSION_THRESHOLD_R
    )

    if use_retracement:
        execution_result = execution.enter_trade_retracement(plan)
    elif config.DCA_ENABLED:
        execution_result = execution.enter_trade_dca_pending(plan)
    elif use_limit:
        execution_result = execution.enter_trade_limit(plan)
    else:
        execution_result = execution.enter_trade(plan)

    if not execution_result.get("ok"):
        log_warning(f"{symbol} entry failed | {execution_result.get('error')}")
        positions.mark_entry_failure(symbol)
        return

    trade_id = signal_journal.append_signal(result, plan, execution_result)

    if use_retracement:
        positions.register_retracement_pending(plan, execution_result, trade_id=trade_id)
    elif config.DCA_ENABLED:
        positions.register_dca_pending(plan, execution_result, trade_id=trade_id)
    elif use_limit:
        positions.register_pending_entry(plan, execution_result, trade_id=trade_id)
    else:
        positions.register(plan, execution_result, trade_id=trade_id)

    if stability is not None:
        # Clears the streak now that it's been acted on - without this, a
        # stale entry would sit in the tracker until this symbol becomes
        # eligible again (position closed + cooldown passed), and could
        # then wrongly count toward a completely unrelated future setup.
        stability.reset(symbol)


def _poll_positions(feed, positions):
    # Outcome journaling happens inside PositionManager._close() itself
    # (it's the only place that still has the trade_id after a position
    # is popped from tracking), not here.
    for symbol in list(positions.positions.keys()):
        position = positions.positions.get(symbol)

        if not position:
            continue

        if position["stage"] == RETRACEMENT_PENDING:
            latest_candle = feed.candles.latest(symbol)

            if position["shadow"]:
                positions.poll_shadow_retracement_pending(symbol, latest_candle)
            else:
                positions.poll_retracement_pending(symbol, latest_candle)

            continue

        if position["stage"] == PENDING_LIMIT_FILL:
            latest_candle = feed.candles.latest(symbol)

            if position["shadow"]:
                positions.poll_shadow_pending_entry(symbol, latest_candle)
            else:
                positions.poll_pending_entry(symbol, latest_candle)

            continue

        # htf_candles/cvd_snapshot: cheap in-memory reads off the same
        # feed _evaluate_symbol already uses every eval tick (no new REST
        # calls) - only actually consumed by config.DCA_BREAKEVEN_
        # CONFIRMATION_ENABLED's check (see PositionManager.
        # _dca_breakeven_confirmation), a no-op for every other stage.
        # Fetched unconditionally rather than gated on position["stage"]
        # here, same convention as candles= above.
        htf_candles = feed.htf_candles.get(symbol)
        cvd_snapshot = feed.cvd.snapshot(symbol)
        # config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED - same cheap
        # in-memory read as htf_candles/cvd_snapshot above, only actually
        # consumed by PositionManager._execute_dca, a no-op otherwise.
        crash_snapshot = feed.crash_detector.snapshot()

        if position["shadow"]:
            latest_candle = feed.candles.latest(symbol)
            positions.poll_shadow(
                symbol, latest_candle, candles=feed.candles.get(symbol),
                htf_candles=htf_candles, cvd_snapshot=cvd_snapshot,
                crash_snapshot=crash_snapshot,
            )
        else:
            positions.poll_live(
                symbol, candles=feed.candles.get(symbol),
                htf_candles=htf_candles, cvd_snapshot=cvd_snapshot,
                crash_snapshot=crash_snapshot,
            )

    # Catches a real (non-shadow) position closed OUTSIDE the bot
    # entirely (manual close, ADL, liquidation) - poll_live's own checks
    # structurally can't see this (see PositionManager.
    # reconcile_closed_positions's own docstring). Must run AFTER the
    # loop above so a genuine TP/SL fill this same tick gets its real
    # specific outcome from poll_live first, not the generic one here.
    positions.reconcile_closed_positions()

    # Full-fidelity snapshot for the next restart - see position_manager.
    # STATE_PATH's own comment for why this beats reconstructing from bare
    # exchange order shape. Once per poll cycle (not per-symbol/per-
    # mutation) is enough - worst-case loss on a crash is one cycle's
    # worth of updates, which the existing exchange-shape reconciliation
    # already covers safely as a fallback for whatever that window missed.
    positions.save_state()


def _resolve_break_confirmations(feed, positions):
    positions.resolve_break_confirmations(feed.candles)


def _reject_summary_line(counts, symbols_by_reason, top_n=8):
    """Shared formatting for both reject-tally heartbeat lines (the
    original reason-only one and the newer trigger-tagged one) - same
    "top N by count, with a capped sample of symbols" shape either way."""
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    parts = []

    for reason, count in top:
        sample = (symbols_by_reason or {}).get(reason, [])
        more = ",..." if sample and count > len(sample) else ""
        symbols_note = f"[{','.join(sample)}{more}]" if sample else ""
        parts.append(f"{reason}={count}{symbols_note}")

    return " ".join(parts)


def _log_heartbeat(
    feed, symbols, positions, reject_counts=None, reject_symbols=None,
    reject_trigger_counts=None, reject_trigger_symbols=None,
    oi_rising_block_total=None,
):
    # OPEN_POSITIONS is real (non-shadow) only - a shadow trade (config.
    # SHADOW_ONLY_TRIGGERS or EXECUTION_MODE=SHADOW) never touches the
    # exchange, so counting it here reads as "N real trades" when it
    # isn't. SHADOW_POSITIONS is the separate count for those.
    # real_open_count() is also what MAX_TOTAL_POSITIONS capacity checks
    # against in _evaluate_symbol - shadow trades don't compete with real
    # ones for capacity. positions.open_count() (both combined) still
    # exists but nothing reads it live-behaviorally anymore.
    log_info(
        f"Heartbeat | WATCHING={len(symbols)} | OPEN_POSITIONS={positions.real_open_count()} "
        f"| SHADOW_POSITIONS={positions.shadow_open_count()} | MODE={config.EXECUTION_MODE}"
    )

    if reject_counts:
        log_info(f"  REJECTED since last heartbeat | {_reject_summary_line(reject_counts, reject_symbols)}")

    # See _evaluate_symbol's docstring on reject_trigger_counts - entirely
    # separate from the line above, never mixed into it.
    if reject_trigger_counts:
        log_info(
            f"  REJECTED BY TRIGGER since last heartbeat | "
            f"{_reject_summary_line(reject_trigger_counts, reject_trigger_symbols)}"
        )

    # config.OI_RISING_REJECT_ENABLED - a dedicated, always-visible counter
    # for this one gate specifically, requested after real doubt about how
    # often it's actually firing: the REJECTED line above already tallies
    # OI_RISING correctly (it's just signal_engine's normal reject reason),
    # but _reject_summary_line only prints the top 8 reasons by count, and
    # candidates never even reach this gate until AGAINST_HTF_BIAS/zone/OTE/
    # order-block have already thinned the pool - in the real deployment
    # this was built for, OI_RISING never once cracked the top 8 against
    # reasons running in the hundreds, so its true count was unobservable
    # from the log even though it was being tallied correctly the whole
    # time. Read directly off reject_counts (not top-8-limited) BEFORE the
    # caller clears it for the next window, so nothing is lost.
    # oi_rising_block_total is a running total the caller never resets -
    # "since bot start", not "since last heartbeat" - so a rare block isn't
    # invisible just because it happened in a quiet window.
    oi_rising_since_heartbeat = (reject_counts or {}).get("OI_RISING", 0)

    if oi_rising_since_heartbeat or oi_rising_block_total:
        log_info(
            f"  OI_RISING gate | blocked {oi_rising_since_heartbeat} since last heartbeat | "
            f"{oi_rising_block_total or 0} total since bot start"
        )

    for symbol in list(positions.positions.keys()):
        position = positions.positions[symbol]

        # config.RETRACEMENT_ENTRY_ENABLED - this stage's position dict
        # keeps the whole plan nested under "plan" instead of flattening
        # entry_price/sl_price/tp1_price/tp2_price/tp_price/single_tp onto
        # itself (see register_retracement_pending's own docstring for
        # why) - logged from there instead, distinctly, rather than
        # forcing the general case below to branch on a shape it doesn't
        # otherwise need to know about.
        if position["stage"] == RETRACEMENT_PENDING:
            plan = position["plan"]
            log_info(
                f"  OPEN {symbol} {position['side']} stage={position['stage']} "
                f"retracement={position['retracement_price']} trigger={plan['entry_price']} "
                f"sl={plan['sl_price']}"
            )
            continue

        filled_note = (
            f" filled={position['filled_quantity']}"
            if position["stage"] == PENDING_LIMIT_FILL else ""
        )
        # config.TP_STATIC_ROI_ENABLED/DCA_TP_STATIC_ROI_ENABLED - a
        # single-TP position (DCA_ACTIVE always, or a static-ROI
        # DCA_PENDING one) has tp_price, not tp1_price/tp2_price (both
        # None in that shape - see risk_manager.build_trade_plan/
        # position_manager.register_dca_pending).
        tp_note = (
            f"tp={position['tp_price']}" if position.get("single_tp")
            else f"tp1={position['tp1_price']} tp2={position['tp2_price']}"
        )
        log_info(
            f"  OPEN {symbol} {position['side']} stage={position['stage']}{filled_note} "
            f"entry={position['entry_price']} sl={position['sl_price']} {tp_note}"
        )


def _refresh_watchlist(feed, positions, current_symbols):
    """config.WATCHLIST_REFRESH_SECONDS - real gap found live (2026-08-15):
    this setting existed in config but nothing ever called it -
    _select_symbols() only ran once, at startup, so a symbol that was
    top-400-by-volume at boot but later drifted below
    MIN_24H_QUOTE_VOLUME_USDT just sat there as a permanently-dead
    QUOTE_VOLUME_TOO_LOW slot until the next full process restart (real
    heartbeat data: ~17-20% of the watchlist most heartbeats). Re-ranks
    and swaps in whatever currently qualifies instead.

    No-op for a pinned config.SCAN_SYMBOLS list - there's nothing to
    re-rank. Symbols with an open position (any stage, including a still-
    pending limit fill) are always kept even if they'd otherwise drop out
    on volume - _poll_positions/_resolve_break_confirmations still need
    their candle feed regardless of whether they're still scan candidates.

    RealtimeMarketData has no live add/remove-symbol path (its symbol set
    is fixed at construction), so this is a soft-restart - stop the feed,
    build a fresh one, start it - not an incremental update. That means a
    real cost every time the symbol set actually changes: the new feed's
    start() re-seeds REST kline history for every symbol in the new list
    (same as a full process restart's cost, just recurring), and the
    caller's single-threaded main loop blocks for that duration with no
    position polling or signal evaluation happening. Skipped entirely
    when the computed set is unchanged from last time to avoid paying
    that cost for no reason."""
    if config.SCAN_SYMBOLS:
        return current_symbols, feed

    fresh = _select_symbols()

    if not fresh:
        log_warning("Watchlist refresh: symbol selection returned nothing, keeping the current list")
        return current_symbols, feed

    open_symbols = list(positions.positions.keys())
    merged = list(dict.fromkeys(list(fresh) + [s for s in open_symbols if s not in fresh]))

    if set(merged) == set(current_symbols):
        return current_symbols, feed

    log_info(
        f"Refreshing watchlist | {len(current_symbols)} -> {len(merged)} symbols "
        f"({len(open_symbols)} held open for existing positions)"
    )

    feed.stop()
    new_feed = RealtimeMarketData(merged, shutdown_event=shutdown_event)
    new_feed.start()
    return merged, new_feed


def main():
    exchange.sync_client_time()

    symbols = _select_symbols()

    if not symbols:
        log_error("No symbols selected for the watchlist; nothing to watch")
        return

    log_info(f"Watching {len(symbols)} symbols in {config.EXECUTION_MODE} mode: {', '.join(symbols)}")

    feed = RealtimeMarketData(symbols, shutdown_event=shutdown_event)
    feed.start()

    positions = PositionManager()

    if config.EXECUTION_MODE == "LIVE":
        positions.reconcile_on_startup(feed)
        positions.reconcile_pending_entries_on_startup()

    eval_interval = max(config.SIGNAL_EVAL_INTERVAL_SECONDS, 1)
    heartbeat_every = max(int(30 / eval_interval), 1)
    # Position polling makes several private REST calls per open position
    # (SL/TP1/TP2 order-status lookups, etc.) - paced on its own cadence
    # instead of every single signal-eval tick, so a large
    # MAX_TOTAL_POSITIONS doesn't multiply REST traffic by the (much
    # faster) eval interval. Real bug found live (2026-08-11):
    # POSITION_POLL_INTERVAL_SECONDS existed in config but was never
    # actually wired to anything - positions were polled on every eval
    # tick regardless, which (combined with the private REST layer's
    # throttle being a no-op - see exchange._private_rest_call) directly
    # contributed to a real Binance IP ban (-1003 Way too many requests).
    poll_every_ticks = max(round(config.POSITION_POLL_INTERVAL_SECONDS / eval_interval), 1)
    # config.WATCHLIST_REFRESH_SECONDS<=0 disables the refresh entirely
    # (same "0 disables" convention as MIN_STOP_DISTANCE_PCT etc.) rather
    # than falling through to max(...,1) and refreshing every tick, which
    # would tear the feed down constantly.
    watchlist_refresh_every_ticks = (
        max(round(config.WATCHLIST_REFRESH_SECONDS / eval_interval), 1)
        if config.WATCHLIST_REFRESH_SECONDS > 0 else None
    )
    tick = 0
    reject_counts = Counter()
    reject_symbols = {}
    reject_trigger_counts = Counter()
    reject_trigger_symbols = {}
    # Never cleared (unlike reject_counts/reject_trigger_counts above) -
    # see _log_heartbeat's own comment on oi_rising_block_total for why
    # this gate specifically gets a running, never-truncated total.
    oi_rising_block_total = 0
    stability = SignalStabilityTracker()
    # Balance barely changes tick-to-tick, so it's refreshed on the same
    # cadence as position polling rather than every eval tick - otherwise
    # a faster SIGNAL_EVAL_INTERVAL_SECONDS would scale this REST call
    # 1:1 with scan speed for no benefit (real issue found 2026-08-12).
    balance = _current_balance()

    try:
        while not shutdown_event.is_set():
            time.sleep(eval_interval)
            tick += 1

            if tick % poll_every_ticks == 0:
                balance = _current_balance()
                _poll_positions(feed, positions)
                _resolve_break_confirmations(feed, positions)

            if watchlist_refresh_every_ticks and tick % watchlist_refresh_every_ticks == 0:
                symbols, feed = _refresh_watchlist(feed, positions, symbols)

            for symbol in symbols:
                _evaluate_symbol(
                    feed, symbol, positions, balance, reject_counts, reject_symbols, stability,
                    reject_trigger_counts, reject_trigger_symbols,
                )

            if tick % heartbeat_every == 0:
                oi_rising_block_total += reject_counts.get("OI_RISING", 0)
                _log_heartbeat(
                    feed, symbols, positions, reject_counts, reject_symbols,
                    reject_trigger_counts, reject_trigger_symbols,
                    oi_rising_block_total=oi_rising_block_total,
                )
                reject_counts.clear()
                reject_symbols.clear()
                reject_trigger_counts.clear()
                reject_trigger_symbols.clear()

    except KeyboardInterrupt:
        log_warning("Shutdown requested (KeyboardInterrupt)")

    except Exception as exc:
        # Real evidence (2026-08-25, investigating why CVD_DIVERGENCE/
        # OI_DIVERGENCE never fire): no systemd/cron/supervisor restarts
        # this process on the VPS - confirmed directly. Of 45 restarts
        # found in the current log, only 5 followed a deliberate
        # KeyboardInterrupt and 12 had no diagnostic marker anywhere in
        # the log right before they happened. Root cause: this except
        # block previously only caught KeyboardInterrupt, so any OTHER
        # unhandled exception here propagated to the interpreter's
        # default excepthook, which prints to stderr only - never
        # reaching logs/bot.log (logger.py's logging.basicConfig writes
        # to that file only, no stderr capture). This doesn't fix
        # whatever is actually crashing the process, but ensures the
        # next crash leaves a real, diagnosable traceback in the log
        # instead of another silent restart. Re-raises unchanged - same
        # "process exits" behavior as today, just now with the reason
        # captured first.
        log_error(f"Fatal error in main loop, exiting: {exc!r}", exc_info=True)
        raise

    finally:
        shutdown_event.set()
        feed.stop()


if __name__ == "__main__":
    main()
