"""Append-only CSV journal of every generated entry signal and its
eventual outcome - the same "always write real evidence, never trust
memory" convention as v7/v8's signal_journal.py.

Only signals that actually pass signal_engine (a real BUY/SELL, not every
rejected NO_SIGNAL evaluation - that would be enormous at tick frequency)
get a row here.

Each signal gets a `trade_id`; its eventual outcome is appended as a
separate row carrying the same `trade_id` (append-only, crash-safe - never
mutates the original row). journal_analysis.py joins the two by that id -
without it, a symbol with several signals close together (a known real
pattern - see the repeated PIPPINUSDT/MANAUSDT re-entries) can't be told
apart, which makes "why did this SL hit" unanswerable.
"""
import csv
from pathlib import Path
import time

import config
from logger import log_warning


JOURNAL_PATH = Path(__file__).resolve().parent / "data" / "signal_journal.csv"

FIELDNAMES = [
    "timestamp", "trade_id", "symbol", "side", "entry_price", "sl_price",
    "tp1_price", "tp2_price", "quantity", "risk_distance_pct",
    "structure_level", "entry_extension_r", "nearest_favorable_sr_r", "setup_age_candles", "signal_trigger", "atr", "ema_value", "ema_alignment_value", "ema_aligned", "htf_trend", "htf_trend_live", "htf_trend_swing_age_hours", "premium_discount_zone",
    "zone_retracement_pct",
    "order_block_present", "fvg_present", "cvd_score", "depth_imbalance",
    "sweep_confluence", "oi_change_pct", "oi_rising",
    "oi_change_pct_bybit", "oi_change_pct_okx", "cross_exchange_oi_agree",
    "liquidation_notional_net",
    "liquidation_cluster", "liquidation_aligned",
    "liquidation_notional_net_bybit", "liquidation_notional_net_okx", "cross_exchange_liquidation_agree",
    "efficiency_ratio", "efficiency_favorable",
    "btc_correlation", "btc_aligned", "absorption_signal", "absorption_aligned",
    "vp_poc_price", "vp_value_area_high", "vp_value_area_low", "vp_position",
    "funding_rate", "funding_favorable", "long_short_ratio",
    "long_short_favorable", "confluence_score", "confluence_total",
    "confluence_ratio", "quote_volume_usdt", "size_multiplier", "tp1_r_multiple", "tp2_r_multiple",
    "execution_mode", "mae_r_multiple", "mfe_r_multiple", "early_breakeven_applied",
    "break_confirmed_by_close", "dca_applied", "dca_breakeven_direction_confirmed",
    "dca_pressure_confirmed", "retracement_fill_type", "retracement_fill_lag_seconds",
    "outcome",
]


def _existing_header(path):
    try:
        with open(path, newline="") as handle:
            return next(csv.reader(handle), None)
    except (OSError, StopIteration):
        return None


def _ensure_header():
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        return

    # A schema change (new diagnostic fields, most recently ema_value)
    # while a journal already exists leaves the OLD header sitting above
    # NEW-shaped rows - csv.DictReader then keys everything off the stale
    # header, so trade_id and every newer field silently stop resolving
    # for every row (confirmed live: journal_analysis.py reported zero
    # resolved trades despite real matching data being in the file).
    # Back the mismatched file up - matching this project's existing
    # signal_journal.bak_* convention - and start a fresh, correctly
    # headered file rather than let analysis silently break again on the
    # next field added.
    existing = _existing_header(JOURNAL_PATH)

    if existing is not None and existing != FIELDNAMES:
        backup_path = JOURNAL_PATH.with_name(
            f"signal_journal.bak_{int(time.time())}.csv"
        )
        JOURNAL_PATH.rename(backup_path)
        log_warning(
            f"signal_journal.csv header didn't match the current schema - "
            f"backed up to {backup_path.name} and started a fresh file"
        )

        with open(JOURNAL_PATH, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()


def _append_row(row):
    _ensure_header()

    with open(JOURNAL_PATH, "a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writerow(row)


def _make_trade_id(symbol):
    return f"{symbol}_{int(time.time() * 1000)}"


def append_signal(signal, plan, execution_result=None):
    """Returns the trade_id so the caller can pass it to position_manager,
    which threads it through to append_outcome when the trade closes.

    execution_result: the real per-trade outcome from execution.py's
    enter_trade*/execution_result dict (has a real "shadow" bool) - when
    provided, execution_mode reflects what ACTUALLY happened for this
    trade (config.SHADOW_ONLY_TRIGGERS can force shadow per-trigger while
    config.EXECUTION_MODE stays LIVE), not just the global config echoed a
    second time. Optional and defaults to the old config-echo behavior so
    existing callers that don't have it yet are unaffected."""
    trade_id = _make_trade_id(signal.get("symbol") or "UNKNOWN")
    entry_price = plan.get("entry_price") or 0
    risk_distance = plan.get("risk_distance") or 0

    row = {field: "" for field in FIELDNAMES}
    row.update({
        "timestamp": time.time(),
        "trade_id": trade_id,
        "symbol": signal.get("symbol"),
        "side": signal.get("signal"),
        "entry_price": entry_price,
        "sl_price": plan.get("sl_price"),
        "tp1_price": plan.get("tp1_price"),
        "tp2_price": plan.get("tp2_price"),
        "quantity": plan.get("quantity"),
        "risk_distance_pct": (
            round(risk_distance / entry_price * 100, 4) if entry_price else ""
        ),
        "structure_level": signal.get("structure_level"),
        # How far entry_price had already run past structure_level, in R,
        # at the moment this trade was taken (risk_manager.build_trade_plan's
        # entry_extension_r - already computed and used for market/limit
        # routing and the MAX_ENTRY_EXTENSION_R reject, but never
        # journaled before now). Added specifically to test whether
        # "wrong from the first tick" LOSS trades (see
        # journal_analysis.py's near-zero-MFE cohort breakdown) are
        # systematically entering on an already-exhausted move - none of
        # the other informational fields (CVD, sweep, EMA, OI) discriminate
        # that cohort from the rest, so this is the next real hypothesis
        # worth having evidence for.
        "entry_extension_r": plan.get("entry_extension_r"),
        # Distance (in R) to the nearest REAL liquidity-pool/structure level
        # in the trade's favorable direction, with no minimum-room floor -
        # unlike tp1_price/tp2_price (drawn to a level that already clears
        # TP1_R_MULTIPLE/TP2_R_MULTIPLE), this reports whatever's actually
        # closest, even one too tight to ever become a TP. Added to test a
        # gap the current target selection doesn't check: whether a closer
        # support/resistance level sits in the way of TP1 and caps/reverses
        # the move before it gets there (risk_manager.
        # nearest_favorable_structure_r). Informational only for now - see
        # config.OI_RISING_REJECT_ENABLED for the precedent of promoting a
        # field like this to a real gate once evidence supports it.
        "nearest_favorable_sr_r": plan.get("nearest_favorable_sr_r"),
        # How many candles old the underlying setup (CHoCH/FVG/order
        # block/divergence) actually was at entry - distinct from
        # entry_extension_r (that's price distance from the level, not
        # time). STRUCTURE_BREAK/EMA_PULLBACK are always 0 (react to the
        # live candle); the retest/divergence triggers can be genuinely
        # old, up to their own *_MAX_AGE_CANDLES/*_LOOKBACK cap - never
        # journaled before now (signal_engine.evaluate() computed it
        # internally for the age-gate checks but discarded it). Built
        # specifically to test whether a stale setup at entry correlates
        # with worse outcomes - the real, previously-unanswerable half of
        # the "entries feel late" complaint (2026-08-19).
        "setup_age_candles": signal.get("setup_age_candles"),
        "signal_trigger": signal.get("signal_trigger"),
        "atr": signal.get("atr"),
        "ema_value": signal.get("ema_value"),
        "ema_alignment_value": signal.get("ema_alignment_value"),
        "ema_aligned": signal.get("ema_aligned"),
        "htf_trend": signal.get("htf_trend"),
        "htf_trend_live": signal.get("htf_trend_live"),
        "htf_trend_swing_age_hours": signal.get("htf_trend_swing_age_hours"),
        "premium_discount_zone": signal.get("premium_discount_zone"),
        "zone_retracement_pct": signal.get("zone_retracement_pct"),
        "order_block_present": bool(signal.get("order_block")),
        "fvg_present": bool(signal.get("fvg")),
        "cvd_score": signal.get("cvd_score"),
        "depth_imbalance": signal.get("depth_imbalance"),
        "sweep_confluence": signal.get("sweep_confluence"),
        "oi_change_pct": signal.get("oi_change_pct"),
        "oi_rising": signal.get("oi_rising"),
        "oi_change_pct_bybit": signal.get("oi_change_pct_bybit"),
        "oi_change_pct_okx": signal.get("oi_change_pct_okx"),
        "cross_exchange_oi_agree": signal.get("cross_exchange_oi_agree"),
        "liquidation_notional_net": signal.get("liquidation_notional_net"),
        "liquidation_cluster": signal.get("liquidation_cluster"),
        "liquidation_aligned": signal.get("liquidation_aligned"),
        "liquidation_notional_net_bybit": signal.get("liquidation_notional_net_bybit"),
        "liquidation_notional_net_okx": signal.get("liquidation_notional_net_okx"),
        "cross_exchange_liquidation_agree": signal.get("cross_exchange_liquidation_agree"),
        "efficiency_ratio": signal.get("efficiency_ratio"),
        "efficiency_favorable": signal.get("efficiency_favorable"),
        "btc_correlation": signal.get("btc_correlation"),
        "btc_aligned": signal.get("btc_aligned"),
        "absorption_signal": signal.get("absorption_signal"),
        "absorption_aligned": signal.get("absorption_aligned"),
        "vp_poc_price": signal.get("vp_poc_price"),
        "vp_value_area_high": signal.get("vp_value_area_high"),
        "vp_value_area_low": signal.get("vp_value_area_low"),
        "vp_position": signal.get("vp_position"),
        "funding_rate": signal.get("funding_rate"),
        "funding_favorable": signal.get("funding_favorable"),
        "long_short_ratio": signal.get("long_short_ratio"),
        "long_short_favorable": signal.get("long_short_favorable"),
        "confluence_score": signal.get("confluence_score"),
        "confluence_total": signal.get("confluence_total"),
        "confluence_ratio": signal.get("confluence_ratio"),
        "quote_volume_usdt": signal.get("quote_volume_usdt"),
        "size_multiplier": plan.get("size_multiplier"),
        "tp1_r_multiple": config.TP1_R_MULTIPLE,
        "tp2_r_multiple": config.TP2_R_MULTIPLE,
        "execution_mode": (
            ("SHADOW" if execution_result.get("shadow") else "LIVE")
            if execution_result is not None else config.EXECUTION_MODE
        ),
    })
    _append_row(row)
    return trade_id


def append_retracement_settle(symbol, trade_id, entry_price, fill_type, fill_lag_seconds):
    """config.RETRACEMENT_ENTRY_ENABLED - a second, partial row for the
    same trade_id, appended once a retracement-pending signal actually
    settles into a real position (position_manager._finalize_retracement_
    entry). Same "append-only, never mutate the original row" convention
    append_outcome already uses (see JOURNAL_PATH's own module docstring)
    - trade_id ties it back to append_signal's row via the join every
    reader already does (journal_analysis.load_trades/signal_journal
    consumers merge same-trade_id rows, later non-blank fields winning).

    Real gap this closes (2026-08-25 investigation): the ORIGINAL signal
    row's entry_price is the planned/trigger price at signal time, never
    updated after a retracement fills at a different real price - every
    downstream read of entry_price for a retracement-settled trade was
    silently stale. This row supplies the REAL settled entry_price
    (overwriting the stale one on any reader that merges by trade_id,
    same mechanism DCA/outcome rows already rely on) plus the two new
    diagnostic fields: whether the resting limit filled on its own
    (`fill_type="LIMIT"`) or needed the market fallback for some/all of
    the quantity (`fill_type="MARKET_FALLBACK"`), and how many seconds
    elapsed between placing the resting order and this resolution -
    previously only reconstructible (partially - roughly half the time)
    by cross-referencing log lines after the fact."""
    row = {field: "" for field in FIELDNAMES}
    row["timestamp"] = time.time()
    row["trade_id"] = trade_id or ""
    row["symbol"] = symbol
    row["entry_price"] = entry_price
    row["retracement_fill_type"] = fill_type
    row["retracement_fill_lag_seconds"] = fill_lag_seconds
    _append_row(row)


def append_outcome(
    symbol, outcome, trade_id=None, mae_r_multiple=None, mfe_r_multiple=None,
    early_breakeven_applied=None, break_confirmed_by_close=None, dca_applied=None,
    dca_breakeven_direction_confirmed=None, dca_pressure_confirmed=None,
):
    row = {field: "" for field in FIELDNAMES}
    row["timestamp"] = time.time()
    row["trade_id"] = trade_id or ""
    row["symbol"] = symbol
    row["outcome"] = outcome

    if mae_r_multiple is not None:
        row["mae_r_multiple"] = mae_r_multiple

    if mfe_r_multiple is not None:
        row["mfe_r_multiple"] = mfe_r_multiple

    if early_breakeven_applied is not None:
        row["early_breakeven_applied"] = early_breakeven_applied

    if break_confirmed_by_close is not None:
        row["break_confirmed_by_close"] = break_confirmed_by_close

    # config.DCA_ENABLED - was this trade averaged into before it
    # resolved? Lets journal_analysis.py eventually compare DCA'd vs
    # never-DCA'd outcomes directly, once enough of each accumulate.
    if dca_applied is not None:
        row["dca_applied"] = dca_applied

    # config.DCA_BREAKEVEN_CONFIRMATION_ENABLED - was the trend/order-flow
    # picture still strongly confirmed the moment a DCA_ACTIVE position
    # reached breakeven? None (blank) means the check never ran at all
    # (either it never reached breakeven, or the master flag was off);
    # True/False means it ran. Lets journal_analysis.py eventually compare
    # outcomes where this was True (would-be-withheld) against the same
    # position's real outcome, before DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_
    # ENABLED is ever turned on for real.
    if dca_breakeven_direction_confirmed is not None:
        row["dca_breakeven_direction_confirmed"] = dca_breakeven_direction_confirmed

    # config.DCA_PRESSURE_CHECK_ENABLED - was order flow still confirmed
    # in the position's own favor at the instant DCA actually fired? None
    # (blank) means either the trade never DCA'd or the master flag was
    # off; True/False means the check ran. Lets journal_analysis.py
    # eventually compare not-confirmed (reduced-size/tighter-stop) fires
    # against confirmed (unchanged) ones once enough of each exist.
    if dca_pressure_confirmed is not None:
        row["dca_pressure_confirmed"] = dca_pressure_confirmed

    _append_row(row)
