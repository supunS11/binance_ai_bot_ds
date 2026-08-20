"""Joins signal_journal.csv's signal rows to their outcome rows by
trade_id and breaks down win/breakeven/loss rates by the diagnostic
fields captured at signal time - the actual mechanism for answering "why
did these SL hit" with evidence instead of a guess.

Run directly on the machine with the real journal:
    python journal_analysis.py [hours_back]

`hours_back` restricts this to trades that CLOSED within that window -
use it to look only at trades resolved after a deploy/fix, since a fixed
number of stale pre-fix rows doesn't get diluted out just because more
trades pile up elsewhere. Omit it to include the whole journal.

Only trades logged after the trade_id correlation fix (this file's
counterpart change in signal_journal.py/position_manager.py) can be
joined - older rows without a trade_id are silently skipped, not guessed
at via symbol/time proximity.
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

from signal_journal import JOURNAL_PATH


LOSS_OUTCOMES = {
    "SL_HIT", "SHADOW_SL_HIT", "TP1_THEN_POSITION_ALREADY_CLOSED",
    # config.LIMIT_ENTRY_MODE_ENABLED - a resting limit filled, but SL
    # placement then failed and the filled quantity was emergency-closed
    # at market (position_manager._apply_pending_fill). Same conservative
    # treatment as TP1_THEN_POSITION_ALREADY_CLOSED above: an emergency
    # exit, counted as a loss rather than left unclassified.
    "LIMIT_FILL_SL_PLACEMENT_FAILED",
    # config.DCA_ENABLED - the ordinary post-DCA stop hit (position_
    # manager.poll_live/poll_shadow's DCA_ACTIVE branch), plus the same
    # "SL placement itself failed, emergency-closed at market" treatment
    # as LIMIT_FILL_SL_PLACEMENT_FAILED above, just for the DCA path
    # (position_manager._execute_dca).
    "DCA_SL_HIT", "SHADOW_DCA_SL_HIT", "DCA_SL_PLACEMENT_FAILED",
    # config.RETRACEMENT_ENTRY_ENABLED - the retracement (or its market
    # fallback) filled, but SL placement then failed and the filled
    # quantity was emergency-closed at market (position_manager.
    # _finalize_retracement_entry) - same "emergency exit, counted as a
    # loss rather than left unclassified" treatment as LIMIT_FILL_SL_
    # PLACEMENT_FAILED/DCA_SL_PLACEMENT_FAILED above.
    "RETRACEMENT_SL_PLACEMENT_FAILED",
}
# LIMIT_INVALIDATED_UNFILLED/LIMIT_EXPIRED_UNFILLED (config.LIMIT_ENTRY_MODE_ENABLED)
# and RETRACEMENT_INVALIDATED_UNFILLED/SHADOW_RETRACEMENT_INVALIDATED_UNFILLED
# (config.RETRACEMENT_ENTRY_ENABLED) are deliberately NOT in any of the
# three sets below - an entry that never filled has zero real P&L, so it
# falls through to classify()'s UNKNOWN rather than pollute WIN/LOSS/
# BREAKEVEN stats. They still count in `resolved` (outcome is non-empty)
# so fill-rate can be measured via the outcome breakdown even though
# they're not a win/loss/breakeven. RETRACEMENT_ENTRY_ENABLED's own market
# fallback means these can ONLY happen via a genuine pre-fill invalidation
# (price already reached the stop before the limit ever touched) - never
# a plain unfilled expiry, unlike the LIMIT_ENTRY_MODE_ENABLED pair.
BREAKEVEN_OUTCOMES = {
    "BREAKEVEN_STOP_HIT", "SHADOW_BREAKEVEN_STOP_HIT",
    "BREAKEVEN_TRIGGER_MARKET_CLOSE",
}
WIN_OUTCOMES = {
    "TP2_HIT", "TP2_HIT_DIRECT", "SHADOW_TP2_HIT",
    # A genuine small realized win from EARLY_BREAKEVEN_LOCK_R_MULTIPLE > 0
    # (position_manager only ever writes this outcome when a real profit
    # was actually locked at promotion time - see
    # position["early_breakeven_profit_locked"]) - distinct from
    # BREAKEVEN_STOP_HIT, which is a true zero-sum scratch.
    "EARLY_BREAKEVEN_PROFIT_HIT", "SHADOW_EARLY_BREAKEVEN_PROFIT_HIT",
    # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - a genuine profit lock
    # from the post-TP1 structure-based trailing stop (position_manager
    # only ever writes this when the trailed level actually sat above
    # breakeven - see position["trailing_stop_locked_profit"]). Takes
    # precedence over EARLY_BREAKEVEN_PROFIT_HIT when both flags could
    # apply (_breakeven_stop_outcome).
    "TRAILING_STOP_PROFIT_HIT", "SHADOW_TRAILING_STOP_PROFIT_HIT",
    # config.PROFIT_PROTECTION_ENABLED - a genuine profit lock from the
    # profit-protection trailing floor (position_manager._trail_profit_
    # protection_if_improved), same "real win" category as TRAILING_STOP_
    # PROFIT_HIT above - just a different trailing mechanism reaching the
    # same outcome. Found missing 2026-08-17 via a real-log cross-check:
    # 5 PROFIT_PROTECTION_HIT closes in one 40h window were silently
    # falling to UNKNOWN instead of WIN.
    "PROFIT_PROTECTION_HIT", "SHADOW_PROFIT_PROTECTION_HIT",
    # config.DCA_ENABLED - the single post-DCA TP hit (position_manager's
    # DCA_ACTIVE branch) - a genuine win on the blended, averaged-in
    # position, same category as TP2_HIT above just for the single-TP
    # lifecycle that replaces TP1+TP2 once a DCA has fired.
    "DCA_TP_HIT", "SHADOW_DCA_TP_HIT",
    # config.TP_STATIC_ROI_ENABLED - a single, full-position TP fill for a
    # DCA_PENDING position that never needed to DCA (see position_manager.
    # poll_live/poll_shadow's DCA_PENDING single_tp branch) - a genuine
    # win, same category as TP2_HIT/DCA_TP_HIT above, just for this
    # narrower pre-DCA single-TP lifecycle.
    "STATIC_TP_HIT", "SHADOW_STATIC_TP_HIT",
}


def classify(outcome):
    if outcome in LOSS_OUTCOMES:
        return "LOSS"
    if outcome in BREAKEVEN_OUTCOMES:
        return "BREAKEVEN"
    if outcome in WIN_OUTCOMES:
        return "WIN"
    return "UNKNOWN"


def load_trades(journal_path=None):
    """Returns {trade_id: merged_row}. A signal row supplies every
    signal-time field; its later outcome row (same trade_id) fills in
    `outcome` plus whatever outcome-time-only fields it carries (symbol,
    mae_r_multiple, mfe_r_multiple). Real bug found live (2026-08-09):
    an earlier version of this only ever merged `outcome` from that row,
    so a trade with no matching signal row at all (e.g. one adopted via
    startup reconciliation after a restart) showed "unknown" for every
    single field - including symbol - even though the outcome row had it
    the whole time. Blank fields never overwrite an already-populated one
    on either side, since both rows only ever carry their own half of the
    data (signal-time fields are always blank on the outcome row, and
    vice versa).

    Reads the target file AND every signal_journal.bak_*.csv sibling
    (signal_journal.py's _ensure_header rotates the whole file to one of
    these any time FIELDNAMES changes shape - confirmed live 2026-08-17:
    ~30 rotations since Aug 8 left ~95% of real trade history sitting in
    backup files this function never used to look at, so every evidence-
    based read this project has ever done was drawn from whatever sliver
    of data happened to land after the MOST RECENT rotation). trade_id
    embeds a millisecond timestamp at creation (_make_trade_id), so rows
    from different files can never collide - a plain merge across all of
    them is safe. Each file is parsed with its OWN header (not the current
    FIELDNAMES), so an old file missing a field that was added later just
    leaves it blank for those rows, same as any other missing field."""
    path = Path(journal_path) if journal_path else JOURNAL_PATH
    matching_files = sorted(path.parent.glob(f"{path.stem}*.csv"))

    if not matching_files:
        return {}

    trades = {}

    for file_path in matching_files:
        with open(file_path, newline="") as handle:
            for row in csv.DictReader(handle):
                trade_id = row.get("trade_id")

                if not trade_id:
                    continue

                existing = trades.setdefault(trade_id, {})
                existing.update({k: v for k, v in row.items() if v != ""})

    return trades


def _bucket_volume(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < 3_000_000:
        return "<3M (below the liquidity floor)"
    if value < 10_000_000:
        return "3M-10M"
    if value < 50_000_000:
        return "10M-50M"
    return ">=50M"


def _bucket_cvd(value):
    try:
        value = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"

    if value < 0.3:
        return "weak (<0.3)"
    if value < 0.6:
        return "moderate (0.3-0.6)"
    return "strong (>=0.6)"


def _bucket_efficiency(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < 0.3:
        return "choppy (<0.3)"
    if value < 0.6:
        return "moderate (0.3-0.6)"
    return "trending (>=0.6)"


def _bucket_correlation(value):
    try:
        value = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"

    if value < 0.3:
        return "weak (<0.3)"
    if value < 0.6:
        return "moderate (0.3-0.6)"
    return "strong (>=0.6)"


def _bucket_funding_rate(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < -0.0005:
        return "<-0.05% (crowded short)"
    if value > 0.0005:
        return ">0.05% (crowded long)"
    return "-0.05% to 0.05% (neutral)"


def _bucket_extension_r(value):
    """How far entry_price had already run past structure_level, in R, at
    signal time (see signal_journal.py's entry_extension_r comment) - the
    hypothesis this buckets is "wrong from the first tick" losses chasing
    an already-exhausted move, distinct from (and not covered by) any of
    the existing informational fields."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < 0:
        return "<0R (entry at/before the level)"
    if value < 0.2:
        return "0-0.2R (tight)"
    if value < 0.5:
        return "0.2-0.5R (extended - limit-routed if LIMIT_ENTRY_MODE_ENABLED)"
    return ">=0.5R (should be rare - normally rejected outright)"


def _bucket_nearest_sr_r(value):
    """How close (in R) the nearest REAL structure level in the favorable
    direction sits - see signal_journal.py's nearest_favorable_sr_r
    comment. Bucketed against TP1's own R floor (TP1_R_MULTIPLE=1.0
    default): anything below 1R is a level TP1 itself already runs past,
    i.e. a potential obstacle in the way rather than the target itself."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown (no real pool in that direction)"

    if value < 0.5:
        return "<0.5R (very close - likely caps the move early)"
    if value < 1.0:
        return "0.5-1R (inside TP1's own floor)"
    if value < 2.0:
        return "1-2R"
    return ">=2R (clear past TP1)"


def _bucket_setup_age(value):
    """How many candles old the underlying setup (CHoCH/FVG/order block/
    divergence) was at entry - see signal_journal.py's setup_age_candles
    comment. 0 always means STRUCTURE_BREAK/EMA_PULLBACK (fresh by
    construction); LIQUIDITY_SWEEP/LIQUIDATION_SWEEP_CONFIRMED report
    "unknown" (no formation index available for a liquidity pool) rather
    than a misleading 0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown (no formation index available for this trigger)"

    if value <= 0:
        return "0 (fresh - STRUCTURE_BREAK/EMA_PULLBACK)"
    if value <= 3:
        return "1-3 candles old"
    if value <= 8:
        return "4-8 candles old"
    return ">8 candles old"


def _bucket_zone_retracement(value):
    """How deep into the range this entry's retracement actually was, in
    the same 0..1 measure OTE_RETRACEMENT_MIN/MAX are expressed in - the
    diagnostic for testing whether shallower qualifying retracements lose
    more than deeper ones, instead of guessing how far to tighten
    OTE_RETRACEMENT_MIN (see config.py comment on that setting)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < 0.705:
        return "<0.705 (shallow - pre-tightening band)"
    if value < 0.75:
        return "0.705-0.75"
    if value < 0.79:
        return "0.75-0.79"
    return ">=0.79 (deep)"


def _bucket_long_short_ratio(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if value < 0.8:
        return "<0.8 (short-heavy)"
    if value > 1.2:
        return ">1.2 (long-heavy)"
    return "0.8-1.2 (balanced)"


_MAE_MFE_SANITY_BOUND = 1000  # R-multiples - see the outlier guard below


def _average_by_outcome(resolved, label, key, sanity_bound=_MAE_MFE_SANITY_BOUND):
    """Average of a numeric field (mae_r_multiple/mfe_r_multiple),
    grouped by WIN/BREAKEVEN/LOSS - the diagnostic that tells apart a
    LOSS that went wrong from the first tick (near-zero average MFE) from
    one that was solidly in profit before fully reversing (a large
    average MFE) - both look identical as a plain loss count.

    Real bug found live (2026-08-09): a since-fixed position_manager.py
    bug could re-derive the R-multiple's denominator from a moved
    sl_price instead of the original risk distance, producing values in
    the billions. Those rows are already permanently in the journal (this
    file's own schema didn't change, so no fresh-file migration clears
    them) - a plain average has no defense against an outlier that size,
    so anything beyond sanity_bound is excluded and reported, rather than
    silently included or silently dropped."""
    lines = [f"\nAverage {label} by outcome:"]
    sums = defaultdict(float)
    counts = defaultdict(int)
    excluded = 0

    for trade in resolved.values():
        try:
            value = float(trade.get(key))
        except (TypeError, ValueError):
            continue

        if abs(value) > sanity_bound:
            excluded += 1
            continue

        classification = classify(trade.get("outcome", ""))
        sums[classification] += value
        counts[classification] += 1

    for classification in ("WIN", "BREAKEVEN", "LOSS"):
        if counts[classification]:
            lines.append(
                f"  {classification}: avg={sums[classification] / counts[classification]:.3f}R "
                f"n={counts[classification]}"
            )

    if excluded:
        lines.append(
            f"  ({excluded} value(s) excluded as implausible outliers - "
            f"beyond {sanity_bound}R, almost certainly corrupted data)"
        )

    if len(lines) == 1:
        lines.append("  (no trades with this field recorded yet)")

    return lines


_LOSS_MFE_BUCKETS = [
    (0.0, 0.2, "0.0-0.2R (near-zero - wrong from the first tick)"),
    (0.2, 0.4, "0.2-0.4R"),
    (0.4, 0.6, "0.4-0.6R"),
    (0.6, 0.8, "0.6-0.8R"),
    (0.8, 1.0, "0.8-1.0R"),
    (1.0, float("inf"), "1.0R+ (ran deep in profit before fully reversing)"),
]


def _loss_mfe_distribution(resolved, sanity_bound=_MAE_MFE_SANITY_BOUND):
    """Bucketed distribution of MFE specifically for LOSS trades - an
    average alone can't tell apart "most losses went wrong immediately"
    from "a few trades ran deep in profit and reversed hard, skewing the
    mean" - literally the same lesson the corrupted-MAE/MFE bug taught:
    never trust a single aggregate without checking what's actually
    behind it. A consistent mid-range distribution argues for a trade-
    management fix (tighter partial/trail); a cluster near zero argues
    entries are still the real problem regardless of what the average
    said."""
    lines = ["\nLOSS trades - MFE distribution (near-zero = wrong from the "
              "first tick; large = was right, then fully reversed):"]
    counts = [0] * len(_LOSS_MFE_BUCKETS)
    total = 0

    for trade in resolved.values():
        if classify(trade.get("outcome", "")) != "LOSS":
            continue

        try:
            value = float(trade.get("mfe_r_multiple"))
        except (TypeError, ValueError):
            continue

        if value < 0 or value > sanity_bound:
            continue

        for index, (low, high, _label) in enumerate(_LOSS_MFE_BUCKETS):
            if low <= value < high:
                counts[index] += 1
                total += 1
                break

    if total == 0:
        lines.append("  (no LOSS trades with mfe_r_multiple recorded yet)")
        return lines

    for (low, high, label), count in zip(_LOSS_MFE_BUCKETS, counts):
        if count:
            lines.append(f"  {label}: n={count} ({count / total * 100:.0f}%)")

    return lines


_NEAR_ZERO_MFE_THRESHOLD = _LOSS_MFE_BUCKETS[0][1]  # matches the first bucket above


def _near_zero_mfe_losses(resolved, threshold=_NEAR_ZERO_MFE_THRESHOLD, sanity_bound=_MAE_MFE_SANITY_BOUND):
    """The LOSS trades that never even briefly moved in the trade's favor
    - the "wrong from the first tick" cohort _loss_mfe_distribution above
    already surfaces as a single aggregate count. Isolating them lets
    every other breakdown dimension (side, trigger, HTF trend, CVD
    strength...) be reused against JUST this cohort instead of the full
    resolved population, which dilutes the signal by mixing in every WIN/
    BREAKEVEN trade and the LOSS trades that ran deep in profit before
    reversing (a real, but completely different, trade-management
    problem - not a directional one)."""
    losses = {}

    for trade_id, trade in resolved.items():
        if classify(trade.get("outcome", "")) != "LOSS":
            continue

        try:
            mfe = float(trade.get("mfe_r_multiple"))
        except (TypeError, ValueError):
            continue

        if 0 <= mfe < threshold and mfe <= sanity_bound:
            losses[trade_id] = trade

    return losses


def _breakdown_lines(resolved, label, key_fn):
    lines = [f"\nBy {label}:"]
    buckets = defaultdict(lambda: defaultdict(int))

    for trade in resolved.values():
        buckets[key_fn(trade)][classify(trade.get("outcome", ""))] += 1

    for key, counts in sorted(buckets.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(counts.values())
        loss_rate = (counts["LOSS"] / total * 100) if total else 0
        lines.append(
            f"  {key}: n={total} WIN={counts['WIN']} BREAKEVEN={counts['BREAKEVEN']} "
            f"LOSS={counts['LOSS']} loss_rate={loss_rate:.0f}%"
        )

    return lines


def _closed_at(trade):
    """The outcome row's timestamp (the trade's actual close time) - since
    load_trades() merges non-blank fields from both rows and the outcome
    row is always written later, "timestamp" in the merged record ends up
    being the close time, not the original signal time."""
    try:
        return float(trade.get("timestamp", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def summarize(journal_path=None, since_timestamp=None):
    """`since_timestamp` (unix seconds) restricts this to trades that
    CLOSED after that point - the way to look only at trades resolved
    after a fix/deploy, instead of a fixed-size batch of stale pre-fix
    rows quietly dragging down an average forever (real case found live,
    2026-08-09: a since-fixed MAE/MFE bug's corrupted rows don't get
    diluted out just because more trades pile up in a different outcome
    bucket - most of them need to be excluded outright)."""
    trades = load_trades(journal_path)
    resolved = {
        tid: t for tid, t in trades.items()
        if t.get("outcome") and (since_timestamp is None or _closed_at(t) >= since_timestamp)
    }

    if not resolved:
        return (
            "No resolved (closed) trades with a matching signal row yet"
            + (" in this window" if since_timestamp else "")
            + ". This needs trades opened after the trade_id correlation "
            "fix - let more run, then check again."
        )

    lines = [f"Resolved trades: {len(resolved)}"]
    overall = defaultdict(int)

    for trade in resolved.values():
        overall[classify(trade.get("outcome", ""))] += 1

    lines.append(
        f"  WIN={overall['WIN']} BREAKEVEN={overall['BREAKEVEN']} "
        f"LOSS={overall['LOSS']} UNKNOWN={overall['UNKNOWN']}"
    )

    # Raw outcome breakdown - the direct read on config.LIMIT_ENTRY_MODE_ENABLED's
    # actual fill rate (LIMIT_EXPIRED_UNFILLED/LIMIT_INVALIDATED_UNFILLED
    # vs. everything that did fill), not just the WIN/LOSS/BREAKEVEN roll-up above.
    lines += _breakdown_lines(resolved, "outcome", lambda t: t.get("outcome", "unknown") or "unknown")
    lines += _breakdown_lines(resolved, "side (BUY/SELL)", lambda t: t.get("side", "unknown") or "unknown")
    lines += _breakdown_lines(resolved, "CVD score strength", lambda t: _bucket_cvd(t.get("cvd_score")))
    lines += _breakdown_lines(resolved, "entry trigger", lambda t: t.get("signal_trigger", "unknown") or "unknown")
    lines += _breakdown_lines(resolved, "entry extension (chase distance from the level)", lambda t: _bucket_extension_r(t.get("entry_extension_r")))
    lines += _breakdown_lines(resolved, "zone retracement depth", lambda t: _bucket_zone_retracement(t.get("zone_retracement_pct")))
    lines += _breakdown_lines(resolved, "nearest favorable S/R distance (informational)", lambda t: _bucket_nearest_sr_r(t.get("nearest_favorable_sr_r")))
    lines += _breakdown_lines(resolved, "setup age at entry (candles since the underlying CHoCH/FVG/block/divergence)", lambda t: _bucket_setup_age(t.get("setup_age_candles")))
    lines += _breakdown_lines(resolved, "sweep confluence", lambda t: t.get("sweep_confluence", "unknown") or "False")
    lines += _breakdown_lines(resolved, "EMA aligned (informational)", lambda t: t.get("ema_aligned", "unknown") or "False")
    lines += _breakdown_lines(resolved, "OI rising (informational)", lambda t: t.get("oi_rising", "unknown") or "False")
    lines += _breakdown_lines(resolved, "liquidation cluster (informational)", lambda t: t.get("liquidation_cluster", "unknown") or "False")
    lines += _breakdown_lines(resolved, "liquidation aligned (informational)", lambda t: t.get("liquidation_aligned", "unknown") or "False")
    lines += _breakdown_lines(resolved, "confluence score (drives position sizing)", lambda t: t.get("confluence_score", "unknown") or "0")
    lines += _breakdown_lines(resolved, "24h quote volume (liquidity floor)", lambda t: _bucket_volume(t.get("quote_volume_usdt")))
    lines += _breakdown_lines(resolved, "efficiency ratio (chop vs trend)", lambda t: _bucket_efficiency(t.get("efficiency_ratio")))
    lines += _breakdown_lines(resolved, "efficiency favorable (informational)", lambda t: t.get("efficiency_favorable", "unknown") or "False")
    lines += _breakdown_lines(resolved, "BTC correlation strength", lambda t: _bucket_correlation(t.get("btc_correlation")))
    lines += _breakdown_lines(resolved, "BTC aligned (informational)", lambda t: t.get("btc_aligned", "unknown") or "False")
    lines += _breakdown_lines(resolved, "funding rate", lambda t: _bucket_funding_rate(t.get("funding_rate")))
    lines += _breakdown_lines(resolved, "funding favorable (informational)", lambda t: t.get("funding_favorable", "unknown") or "False")
    lines += _breakdown_lines(resolved, "long/short account ratio", lambda t: _bucket_long_short_ratio(t.get("long_short_ratio")))
    lines += _breakdown_lines(resolved, "long/short favorable (informational)", lambda t: t.get("long_short_favorable", "unknown") or "False")
    lines += _breakdown_lines(resolved, "early breakeven applied (new 1R profit-lock trigger)", lambda t: t.get("early_breakeven_applied", "unknown") or "False")
    lines += _breakdown_lines(resolved, "break confirmed by candle close (wick vs real break)", lambda t: t.get("break_confirmed_by_close", "unknown") or "False")
    lines += _breakdown_lines(resolved, "HTF trend", lambda t: t.get("htf_trend", "unknown") or "unknown")
    lines += _breakdown_lines(resolved, "order block present", lambda t: t.get("order_block_present", "unknown") or "False")
    lines += _breakdown_lines(resolved, "FVG present", lambda t: t.get("fvg_present", "unknown") or "False")
    lines += _breakdown_lines(resolved, "symbol", lambda t: t.get("symbol", "unknown"))

    lines += _average_by_outcome(resolved, "MAE (adverse excursion)", "mae_r_multiple")
    lines += _average_by_outcome(resolved, "MFE (favorable excursion)", "mfe_r_multiple")
    lines += _loss_mfe_distribution(resolved)

    near_zero_mfe_losses = _near_zero_mfe_losses(resolved)

    if near_zero_mfe_losses:
        lines.append(
            f"\n--- Near-zero-MFE LOSS trades only (n={len(near_zero_mfe_losses)}) - "
            "what characterizes a genuinely wrong-direction/timing entry, "
            "isolated from the rest of the LOSS population ---"
        )
        lines += _breakdown_lines(near_zero_mfe_losses, "side (BUY/SELL)", lambda t: t.get("side", "unknown") or "unknown")
        lines += _breakdown_lines(near_zero_mfe_losses, "entry trigger", lambda t: t.get("signal_trigger", "unknown") or "unknown")
        lines += _breakdown_lines(near_zero_mfe_losses, "entry extension (chase distance from the level)", lambda t: _bucket_extension_r(t.get("entry_extension_r")))
        lines += _breakdown_lines(near_zero_mfe_losses, "zone retracement depth", lambda t: _bucket_zone_retracement(t.get("zone_retracement_pct")))
        lines += _breakdown_lines(near_zero_mfe_losses, "HTF trend", lambda t: t.get("htf_trend", "unknown") or "unknown")
        lines += _breakdown_lines(near_zero_mfe_losses, "CVD score strength", lambda t: _bucket_cvd(t.get("cvd_score")))
        lines += _breakdown_lines(near_zero_mfe_losses, "sweep confluence", lambda t: t.get("sweep_confluence", "unknown") or "False")
        lines += _breakdown_lines(near_zero_mfe_losses, "24h quote volume (liquidity floor)", lambda t: _bucket_volume(t.get("quote_volume_usdt")))
        lines += _breakdown_lines(near_zero_mfe_losses, "efficiency ratio (chop vs trend)", lambda t: _bucket_efficiency(t.get("efficiency_ratio")))
        lines += _breakdown_lines(near_zero_mfe_losses, "symbol", lambda t: t.get("symbol", "unknown"))

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    hours_back = float(sys.argv[1]) if len(sys.argv) > 1 else None
    since = (time.time() - hours_back * 3600) if hours_back else None
    print(summarize(since_timestamp=since))
