"""Tracks open positions from entry through TP1 -> breakeven -> TP2/SL,
mirroring v7's multi_tp.py state machine: once TP1's algo order genuinely
triggers (status `FINISHED` - not just missing, which can also mean
cancelled because something else closed the position first), cancel the
original SL and replace it with one at breakeven for the remaining
quantity.

In SHADOW mode there is nothing to poll on the exchange, so outcomes are
simulated against the live candle stream instead - shadow trades still
produce real win/loss evidence about signal quality before anything is
placed for real.
"""
import json
import time
from pathlib import Path

import config
import exchange
import execution
import market_structure
import risk_manager
import signal_engine
import signal_journal
from logger import log_error, log_info, log_warning

# Full-fidelity snapshot of self.positions, refreshed roughly every
# POSITION_POLL_INTERVAL_SECONDS (see main._poll_positions's own call to
# save_state) - see reconcile_on_startup/load_state for why this exists:
# reconstructing position state from bare exchange order shape alone is
# fundamentally ambiguous for some real cases (DCA_ACTIVE vs an ordinary
# post-TP1 BREAKEVEN_ACTIVE position look identical on the exchange - see
# _recover_dca_active_position's own docstring) and lossy for all of them
# (structure_level, risk_distance, MAE/MFE tracking,
# profit-protection arm state etc. have no exchange-side representation
# at all). A restart now prefers this file wholesale over guessing,
# falling back to the existing exchange-shape reconciliation only for a
# symbol this file doesn't know about (first run, deleted/corrupted
# file, or a real position the bot never registered itself).
STATE_PATH = Path(__file__).resolve().parent / "data" / "position_state.json"


TP1_PENDING = "TP1_PENDING"
BREAKEVEN_ACTIVE = "BREAKEVEN_ACTIVE"
# config.DCA_ENABLED - entered with TP1/TP2 live but deliberately NO SL
# (see execution.enter_trade_dca_pending/config.py's DCA section). Two
# outcomes race every poll tick while in this stage: TP1 fills first (the
# existing breakeven-promotion path takes over, same as any other
# position - DCA never fires), or price reaches position["dca_price"]
# first (_execute_dca fires - see its docstring).
DCA_PENDING = "DCA_PENDING"
# A DCA has fired: TP1/TP2 were cancelled and replaced with a single TP,
# and the first real SL this position has ever had is now live, both
# anchored to the new blended entry price. Only a WIN/LOSS resolution
# happens from here - no further TP1->TP2 split, no further promotion
# step (there's nothing left to promote FROM; the SL already reflects the
# post-DCA structure level, not a placeholder).
DCA_ACTIVE = "DCA_ACTIVE"
# config.LIMIT_ENTRY_MODE_ENABLED - a resting GTC LIMIT entry that hasn't
# (fully) filled yet. Lives in the same `positions` dict as every other
# stage (not a separate manager) so has_open_position()/open_count()/
# cooldown accounting all work for it with no extra bookkeeping - a
# pending entry reserves its MAX_TOTAL_POSITIONS slot the instant it's
# placed, same as capital is nominally committed the moment it rests on
# the book.
PENDING_LIMIT_FILL = "PENDING_LIMIT_FILL"
# config.RETRACEMENT_ENTRY_ENABLED - a resting GTC LIMIT at a small
# pullback toward the stop from the planned entry price, placed instead
# of a synchronous market/DCA-pending entry. Same MAX_TOTAL_POSITIONS-
# slot-reservation shape as PENDING_LIMIT_FILL above, but resolves
# differently: it ALWAYS ends in a real position (a bounded market
# fallback for whatever didn't fill by config.RETRACEMENT_ENTRY_TIMEOUT_
# SECONDS, never a walk-away) and, once resolved, hands off into
# DCA_PENDING or TP1_PENDING via the normal register_dca_pending()/
# register() - see register_retracement_pending/_finalize_retracement_entry.
RETRACEMENT_PENDING = "RETRACEMENT_PENDING"
# Tags the real SL/TP orders _execute_dca places, so a restart's
# _adopt_position can tell a genuine DCA_ACTIVE position apart from an
# ordinary post-TP1 BREAKEVEN_ACTIVE one - both are otherwise the exact
# same shape on the exchange (one full-position STOP_MARKET + one full-
# position TAKE_PROFIT_MARKET, no partial TP). See exchange.
# place_stop_loss's own docstring for the mechanism.
_DCA_SL_CLIENT_ALGO_ID_PREFIX = "dcaSL"
_DCA_TP_CLIENT_ALGO_ID_PREFIX = "dcaTP"
# config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - tags the dormant/active
# native TRAILING_STOP_MARKET order _execute_dca places alongside the
# real SL, so a restart's _adopt_position can recognize it and repopulate
# position["dca_trail_order_id"] instead of losing track of it (same
# mechanism as the two prefixes above).
_DCA_TRAIL_CLIENT_ALGO_ID_PREFIX = "dcaTrail"


def _order_type(order):
    """The algo-order list endpoint returns the order type under
    `orderType`, not `type` (confirmed against v7's proven-working
    find_matching_open_algo_order) - checking `type` alone silently
    matches nothing, ever."""
    return str((order or {}).get("orderType") or (order or {}).get("type") or "").upper()


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _structure_stop_candidate(position, candles):
    """config.STRUCTURE_STOP_MANAGEMENT_ENABLED - the most recent
    CONFIRMED swing point in the position's favor (last_swing_low for
    BUY, last_swing_high for SELL), clamped so it can never sit worse
    than flat breakeven. None if candles is falsy, structure isn't
    available yet, or no swing has formed in the favorable direction -
    callers fall back to the fixed-distance calculation in that case.
    Shared by the early-breakeven lock and the post-TP1 trailing stop -
    same mechanism, two trigger points. Does NOT itself check
    config.STRUCTURE_STOP_MANAGEMENT_ENABLED - callers gate on that so
    this stays a pure "what would structure say" query, independent of
    whether the feature is currently enabled."""
    if not candles:
        return None

    structure = market_structure.structure_state(candles)

    if not structure.get("available"):
        return None

    side = position["side"]
    swing = (
        structure.get("last_swing_low") if side == "BUY"
        else structure.get("last_swing_high")
    )

    if swing is None:
        return None

    breakeven = risk_manager.compute_breakeven_price(position["entry_price"], side)
    return max(swing, breakeven) if side == "BUY" else min(swing, breakeven)


def _more_favorable(side, price, reference):
    """True if `price` sits strictly further in the position's favor
    than `reference` (BUY: higher is better; SELL: lower is better).
    Used both as the trailing stop's ratchet-only gate (never loosen)
    and to classify whether a given stop level actually locks in more
    than flat breakeven."""
    return price > reference if side == "BUY" else price < reference


def _resolve_real_entry(plan, execution_result, side):
    """execution_result["real_entry_price"] (see exchange.
    resolve_market_fill_price) - the real average fill Binance reports
    for the entry's MARKET order, used instead of plan["entry_price"]
    (the signal-time estimate the order was computed from) when it's
    available. Real evidence (2026-08-18, 30 live entries checked
    against actual account trade history): slippage up to +0.25% on this
    bot's own fills - not huge, but a real fraction of risk_distance
    floors as low as MIN_STOP_DISTANCE_PCT=0.6%, so it was showing up as
    noise in every MAE/MFE and R-multiple stat derived from risk_distance.

    Only entry_price/breakeven_price/risk_distance are corrected here -
    NOT sl_price/tp1_price/tp2_price, which stay exactly as risk_manager
    computed them (real structure levels or R-multiples of the ORIGINAL
    planned risk_distance) - shifting a real liquidity-pool price to
    chase half a percent of slippage on the entry would be wrong, that
    level doesn't move just because the fill did.

    Known, accepted residual gap: config.TP_STATIC_ROI_ENABLED's TP1 is
    NOT a structure level - it's a pure function of entry_price
    (risk_manager.price_at_roi_pct) - so ordinary market-order slippage
    technically makes it stale too, same slippage this function's own
    docstring already describes for the general case. NOT corrected here,
    because
    by the time this runs the real TP1 order has ALREADY been placed
    (enter_trade/enter_trade_dca_pending place it synchronously, before
    the real fill price is even resolved) - correcting the tracked value
    alone without also replacing the real resting order would make the
    bot's own bookkeeping disagree with the exchange, worse than the
    small (~0.25%, per the evidence above) staleness itself. See
    _resolve_tp1_price below - only safe to use where protection orders
    are placed AFTER the real entry_price is already known, e.g.
    _finalize_retracement_entry's settle path.

    Shadow mode (no real order, no real_entry_price) and any resolution
    failure both fall through to the exact old behavior (the planned
    price) - this is a bookkeeping accuracy improvement, never a reason
    to fail or alter an entry."""
    entry_price = execution_result.get("real_entry_price") or plan["entry_price"]

    if entry_price == plan["entry_price"]:
        return entry_price, plan["breakeven_price"], plan.get("risk_distance") or abs(
            plan["entry_price"] - plan["sl_price"]
        )

    risk_distance = abs(entry_price - plan["sl_price"])

    if risk_distance <= 0:
        # Slippage carried the real fill past/onto the planned SL itself -
        # pathological, but real_distance from the (still valid) planned
        # entry is more honest here than a zero/negative distance.
        risk_distance = plan.get("risk_distance") or abs(plan["entry_price"] - plan["sl_price"])

    breakeven_price = risk_manager.compute_breakeven_price(entry_price, side)
    return entry_price, breakeven_price, risk_distance


def _resolve_tp1_price(plan, entry_price, side):
    """plan["tp1_price"] as risk_manager originally computed it, UNLESS
    config.TP_STATIC_ROI_ENABLED was on for this plan (plan["tp1_static_
    roi_pct"] is then the ROI% used, not None) - a static-ROI TP1 is a
    pure function of entry_price, not a real structure level, so unlike
    sl_price/tp2_price (deliberately left alone by _resolve_real_entry
    above) it must be recomputed against whatever entry_price actually
    turned out to be, or the position's real target silently drifts by
    however much the real fill differed from the planned one.

    Real bug found live (2026-08-21, SLXUSDT under config.RETRACEMENT_
    ENTRY_ENABLED): a limit fill landed noticeably better than the
    planned trigger price (that's the whole point of retracement entry),
    but TP1 stayed computed off the STALE trigger price - closer than the
    intended TP_TARGET_ROI_PCT, not the 10% ROI the setting promised.

    ONLY safe to call from a settle path that places protection orders
    AFTER the real entry_price is already known - _finalize_retracement_
    entry (the caller this was built for) recomputes settled_plan["tp1_
    price"] with this BEFORE execution.place_protection_orders/place_dca_
    protection_orders ever run, so the real exchange order and the
    tracked value always agree. NOT called from register()/register_dca_
    pending() themselves - enter_trade/enter_trade_dca_pending place the
    real TP1 order synchronously, before real_entry_price is even
    resolved, so correcting only the tracked value there would make the
    bot's own bookkeeping disagree with what's actually resting on the
    exchange - see _resolve_real_entry's own docstring for that residual,
    accepted gap (ordinary slippage only, much smaller than retracement's
    deliberate price difference).

    Falls back to the original plan value if the recompute itself fails
    (entry_price<=0/LEVERAGE<=0) - same "never fail an entry over a
    bookkeeping accuracy improvement" principle _resolve_real_entry
    already follows."""
    roi_pct = plan.get("tp1_static_roi_pct")

    if roi_pct is None:
        return plan["tp1_price"]

    recomputed = risk_manager.price_at_roi_pct(entry_price, side, roi_pct)
    return plan["tp1_price"] if recomputed is None else recomputed


class PositionManager:
    def __init__(self):
        self.positions = {}
        self._closed_at = {}  # symbol -> timestamp of its most recent close

    def save_state(self, path=None):
        """Full snapshot of self.positions to disk - see STATE_PATH's own
        comment for why. Atomic (write to a temp file, then os.replace)
        so a crash mid-write can never leave a half-written/corrupt file
        for the next startup to trip over - load_state treats a corrupt
        file as "no state" (logs and falls back to guessing), never
        raises. Called once per poll cycle (main._poll_positions), not
        on every individual mutation - worst-case data loss on a crash
        is one POSITION_POLL_INTERVAL_SECONDS-ish window, which the
        existing exchange-shape reconciliation already covers safely for
        whatever that window missed."""
        path = path or STATE_PATH

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".json.tmp")

            with open(tmp_path, "w") as handle:
                json.dump(self.positions, handle)

            tmp_path.replace(path)
        except Exception as exc:
            log_warning(f"Failed to save position state (continuing): {exc}")

    @staticmethod
    def load_state(path=None):
        """The inverse of save_state - returns {} (not an error) for a
        missing, empty, or corrupt file, so a first-ever run or a lost/
        deleted state file just falls back to the existing exchange-
        shape reconciliation for every symbol, exactly as before this
        feature existed."""
        path = path or STATE_PATH

        try:
            with open(path) as handle:
                data = json.load(handle)

            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            log_warning(f"Failed to load saved position state (ignoring): {exc}")
            return {}

    def has_open_position(self, symbol):
        return symbol in self.positions

    def is_in_cooldown(self, symbol):
        """After ANY close (win, loss, or breakeven), skip that symbol for
        SYMBOL_REENTRY_COOLDOWN_SECONDS. Evidence (2026-08-08): several
        symbols hit SL repeatedly within seconds-to-minutes of the
        previous close, at nearly the same level - immediate re-entry
        into an actively chopping symbol instead of waiting for the
        picture to actually change."""
        closed_at = self._closed_at.get(symbol)

        if closed_at is None:
            return False

        cooldown = max(int(config.SYMBOL_REENTRY_COOLDOWN_SECONDS), 0)
        return (time.time() - closed_at) < cooldown

    def mark_entry_failure(self, symbol):
        """A failed entry (leverage rejected, SL placement failed, etc.)
        never reaches register(), so is_in_cooldown()'s normal trigger
        (_close()) never fires for it either - real bug found live
        (STGUSDT/DEXEUSDT, 2026-08-08): a symbol that fails entry for a
        persistent reason got retried on every single eval cycle with no
        backoff at all, each attempt opening and market-closing a real
        position. Route a failed attempt through the same cooldown a
        normal close gets, so a broken symbol backs off instead."""
        self._closed_at[symbol] = time.time()

    def open_count(self):
        return len(self.positions)

    def real_open_count(self):
        """config.SHADOW_ONLY_TRIGGERS - a per-trigger-forced-shadow trade
        (or the whole bot in EXECUTION_MODE=SHADOW) is fully evaluated,
        sized, and tracked here exactly like a real one, so open_count()
        alone conflates "how many real orders are actually on the
        exchange" with "how many setups are being watched at all". Real
        gap found live (2026-08-29): the heartbeat's own OPEN_POSITIONS
        read as "N real trades" when several of those N were shadow-only
        CVD_DIVERGENCE positions that never touched the exchange."""
        return sum(1 for position in self.positions.values() if not position["shadow"])

    def shadow_open_count(self):
        return sum(1 for position in self.positions.values() if position["shadow"])

    def register(self, plan, execution_result, trade_id=None):
        symbol = plan["symbol"]
        shadow = execution_result.get("shadow", True)
        entry_price, breakeven_price, risk_distance = _resolve_real_entry(
            plan, execution_result, plan["side"]
        )

        position = {
            "symbol": symbol,
            "trade_id": trade_id,
            "side": plan["side"],
            "entry_price": entry_price,
            "sl_price": plan["sl_price"],
            "tp1_price": plan["tp1_price"],
            "tp2_price": plan["tp2_price"],
            "breakeven_price": breakeven_price,
            "quantity": plan["quantity"],
            "tp1_quantity": plan["tp1_quantity"],
            "tp2_quantity": plan["tp2_quantity"],
            "sl_order_id": (
                exchange._accepted_order_id(execution_result.get("sl_order"))
                if not shadow else None
            ),
            "tp1_order_id": (
                exchange._accepted_order_id(execution_result.get("tp1_order"))
                if not shadow else None
            ),
            "tp2_order_id": (
                exchange._accepted_order_id(execution_result.get("tp2_order"))
                if not shadow else None
            ),
            "stage": TP1_PENDING,
            "shadow": shadow,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            # Set True only if EARLY_BREAKEVEN_LOCK_R_MULTIPLE was actually
            # > 0 at the moment of promotion - distinguishes a genuine
            # locked-profit stop hit (a real small win) from a flat
            # breakeven scratch, both of which land in BREAKEVEN_ACTIVE.
            # See _promote_to_breakeven and the EARLY_BREAKEVEN_PROFIT_HIT
            # outcome.
            "early_breakeven_profit_locked": False,
            # config.PROFIT_PROTECTION_ENABLED - mirrors the two keys
            # above but for the ROI-of-TP1 promotion path (mutually
            # exclusive with early_breakeven_applied/profit_locked at the
            # promotion moment - see _is_profit_protection_candidate).
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            # Best price reached since profit protection armed - None
            # until arming, seeded with the arm-time price, then updated
            # every poll while still trailing. See risk_manager.
            # compute_profit_protection_trailing_floor/
            # _trail_profit_protection_if_improved.
            "profit_protection_peak_price": None,
            # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - set True only if
            # a post-TP1 trailing stop replacement actually locked in more
            # than flat breakeven (see _trail_stop_if_improved and the
            # TRAILING_STOP_PROFIT_HIT outcome) - takes precedence over
            # early_breakeven_profit_locked in _breakeven_stop_outcome.
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            # Fixed at entry, never touched again - see
            # _mae_mfe_r_multiples for why this must never be re-derived
            # from position["sl_price"] later (that field legitimately
            # moves to the breakeven price once a trade is promoted).
            "risk_distance": risk_distance,
            # For resolve_break_confirmations() - was the structure break
            # that triggered this entry still holding once its candle
            # actually finished, or was it just a wick that snapped back
            # before the candle closed? None until that candle closes.
            "structure_level": plan.get("structure_level"),
            "trigger_candle_open_time": plan.get("trigger_candle_open_time"),
            "break_confirmed_by_close": None,
        }
        self.positions[symbol] = position
        return position

    def register_dca_pending(self, plan, execution_result, trade_id=None):
        """config.DCA_ENABLED - parallel to register(), but sl_order_id
        always starts None (execution.enter_trade_dca_pending places no
        SL at all, live or shadow) and stage starts DCA_PENDING instead
        of TP1_PENDING. dca_price/dca_quantity come straight from
        risk_manager.build_trade_plan's own dca_price/dca_quantity
        fields. dca_applied mirrors early_breakeven_applied's role: set
        True the moment _execute_dca actually fires, so it can never be
        attempted twice for the same position.

        config.DCA_RESTING_ORDER_ENABLED - dca_order_id is the resting
        LIMIT order's id (None when the flag is off, shadow, or
        placement failed - see execution.place_dca_protection_orders).
        poll_live's DCA_PENDING branch uses its presence to decide
        whether to poll real order status (exchange.get_order_status)
        instead of the candle-range check for this specific position.

        config.DCA_PROTECTIVE_FIRST_ENABLED - dca_protective_sl_order_id
        is the resting protective STOP_MARKET's id instead (None under
        the same conditions as dca_order_id above, plus whenever this
        flag is off) - a deliberately SEPARATE field from dca_order_id
        rather than reusing it, so the two mutually-exclusive mechanisms
        (see execution._place_dca_resting_or_protective_order) can never
        be confused with each other during rollout. Only one of the two
        fields is ever non-None for a given position."""
        symbol = plan["symbol"]
        shadow = execution_result.get("shadow", True)
        entry_price, breakeven_price, risk_distance = _resolve_real_entry(
            plan, execution_result, plan["side"]
        )
        # config.TP_STATIC_ROI_ENABLED - a single-tp position has tp_price/
        # tp_order_id instead of the tp1/tp2 pair (both None in that plan
        # shape - see risk_manager.build_trade_plan). Every DCA_PENDING
        # call site that reads these fields branches on this flag.
        single_tp = bool(plan.get("single_tp"))

        position = {
            "symbol": symbol,
            "trade_id": trade_id,
            "side": plan["side"],
            "entry_price": entry_price,
            "sl_price": plan["sl_price"],
            "tp1_price": plan["tp1_price"],
            "tp2_price": plan["tp2_price"],
            "tp_price": plan.get("tp_price"),
            "single_tp": single_tp,
            "breakeven_price": breakeven_price,
            "quantity": plan["quantity"],
            "tp1_quantity": plan["tp1_quantity"],
            "tp2_quantity": plan["tp2_quantity"],
            "sl_order_id": None,
            "tp1_order_id": (
                exchange._accepted_order_id(execution_result.get("tp1_order"))
                if not shadow else None
            ),
            "tp2_order_id": (
                exchange._accepted_order_id(execution_result.get("tp2_order"))
                if not shadow else None
            ),
            "tp_order_id": (
                exchange._accepted_order_id(execution_result.get("tp_order"))
                if not shadow else None
            ),
            "dca_order_id": (
                exchange._accepted_order_id(execution_result.get("dca_order"))
                if not shadow and not config.DCA_PROTECTIVE_FIRST_ENABLED else None
            ),
            "dca_protective_sl_order_id": (
                exchange._accepted_order_id(execution_result.get("dca_order"))
                if not shadow and config.DCA_PROTECTIVE_FIRST_ENABLED else None
            ),
            "dca_protective_stop_hit": None,
            "stage": DCA_PENDING,
            "shadow": shadow,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            "risk_distance": risk_distance,
            "structure_level": plan.get("structure_level"),
            "trigger_candle_open_time": plan.get("trigger_candle_open_time"),
            "break_confirmed_by_close": None,
            "dca_price": plan.get("dca_price"),
            "dca_quantity": plan.get("dca_quantity"),
            "dca_applied": False,
            "atr": plan.get("atr"),
        }
        self.positions[symbol] = position
        return position

    def register_pending_entry(self, plan, execution_result, trade_id=None):
        """A resting LIMIT entry has been placed but not (yet) filled -
        parallel to register(), but sl_order_id/tp1_order_id/tp2_order_id
        all start None (nothing is placed until a real fill exists, see
        poll_pending_entry) rather than being read from execution_result
        the way register() reads real SL/TP order ids off a synchronously-
        filled market order."""
        symbol = plan["symbol"]
        shadow = execution_result.get("shadow", True)

        position = {
            "symbol": symbol,
            "trade_id": trade_id,
            "side": plan["side"],
            "entry_price": plan["entry_price"],
            "sl_price": plan["sl_price"],
            "tp1_price": plan["tp1_price"],
            "tp2_price": plan["tp2_price"],
            "breakeven_price": plan["breakeven_price"],
            "quantity": plan["quantity"],
            "tp1_quantity": plan["tp1_quantity"],
            "tp2_quantity": plan["tp2_quantity"],
            "sl_order_id": None,
            "tp1_order_id": None,
            "tp2_order_id": None,
            "limit_order_id": (
                exchange._accepted_order_id(execution_result.get("entry_order"))
                if not shadow else None
            ),
            "limit_placed_at": time.time(),
            "filled_quantity": 0.0,
            "stage": PENDING_LIMIT_FILL,
            "shadow": shadow,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            # config.PROFIT_PROTECTION_ENABLED - mirrors the two keys
            # above but for the ROI-of-TP1 promotion path (mutually
            # exclusive with early_breakeven_applied/profit_locked at the
            # promotion moment - see _is_profit_protection_candidate).
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            # Best price reached since profit protection armed - None
            # until arming, seeded with the arm-time price, then updated
            # every poll while still trailing. See risk_manager.
            # compute_profit_protection_trailing_floor/
            # _trail_profit_protection_if_improved.
            "profit_protection_peak_price": None,
            # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - set True only if
            # a post-TP1 trailing stop replacement actually locked in more
            # than flat breakeven (see _trail_stop_if_improved and the
            # TRAILING_STOP_PROFIT_HIT outcome) - takes precedence over
            # early_breakeven_profit_locked in _breakeven_stop_outcome.
            "trailing_stop_locked_profit": False,
            # Tracking a not-yet-real fill's excursion is meaningless -
            # seeded with a real price only once poll_pending_entry sees
            # an actual fill (_apply_pending_fill).
            "mae_price": None,
            "mfe_price": None,
            "risk_distance": plan.get("risk_distance") or abs(plan["entry_price"] - plan["sl_price"]),
            "structure_level": plan.get("structure_level"),
            "trigger_candle_open_time": plan.get("trigger_candle_open_time"),
            "break_confirmed_by_close": None,
        }
        self.positions[symbol] = position
        return position

    def register_retracement_pending(self, plan, execution_result, trade_id=None):
        """config.RETRACEMENT_ENTRY_ENABLED - a resting retracement limit
        has been placed (or, in shadow mode, would have been) but not yet
        resolved. Unlike register()/register_dca_pending()/register_
        pending_entry(), which all flatten plan fields onto the position
        dict, this keeps the WHOLE plan nested under "plan" - this stage's
        only job is to hand off to one of those two once resolved (see
        _finalize_retracement_entry), and re-flattening a second time here
        would just be duplicated bookkeeping. "is_dca" is captured once,
        from config.DCA_ENABLED at the moment this fires, rather than
        re-read at resolution time - the routing decision that led here
        was already made in main.py against that same value, and a
        resting order can outlive a config change mid-flight.

        A resting, unfilled limit has no matching real exchange position,
        so a restart before it resolves can't be recovered by reconcile_
        on_startup (which only walks REAL open positions) -
        reconcile_pending_entries_on_startup covers this instead (cancels
        any stray resting LIMIT order found account-wide, re-evaluates
        fresh next tick), same as it already does for LIMIT_ENTRY_MODE_
        ENABLED's own resting orders."""
        symbol = plan["symbol"]
        shadow = execution_result.get("shadow", True)

        position = {
            "symbol": symbol,
            "trade_id": trade_id,
            "side": plan["side"],
            "plan": plan,
            "is_dca": bool(config.DCA_ENABLED),
            "retracement_price": execution_result.get("retracement_price"),
            # config.RETRACEMENT_DEPTH_AWARE_ENABLED - per-trade timeout
            # (longer for a weak-depth_imbalance entry routed deeper) -
            # falls back to the global default for any execution_result
            # that doesn't carry it, same convention as retracement_price
            # above.
            "retracement_timeout_seconds": execution_result.get(
                "retracement_timeout_seconds", config.RETRACEMENT_ENTRY_TIMEOUT_SECONDS
            ),
            "used_deep_retracement": execution_result.get("used_deep_retracement", False),
            "limit_order_id": (
                exchange._accepted_order_id(execution_result.get("entry_order"))
                if not shadow else None
            ),
            "limit_placed_at": time.time(),
            "stage": RETRACEMENT_PENDING,
            "shadow": shadow,
            "opened_at": time.time(),
        }
        self.positions[symbol] = position
        return position

    def reconcile_on_startup(self, feed=None):
        """Rebuild tracking for positions already open on the exchange
        when the process starts - crash, manual restart, or a redeploy.
        Without this, a restart makes the bot blind to real open
        positions: has_open_position() would wrongly say "no" for a
        symbol that already has one (risking a duplicate entry stacked on
        top of it), and the existing position would never get TP1 ->
        breakeven promotion, missing-order self-heal, or its outcome
        journaled - even though its real SL/TP1/TP2 orders keep working
        on Binance's side regardless of whether the bot remembers them.

        `feed` (RealtimeMarketData, already REST-seeded by the time
        main.py calls this - see feed.start()) is optional and used only
        to recover a genuine DCA_PENDING position's dca_price/atr from
        fresh candles - see _recover_dca_pending_position. Without it,
        such a position still gets protected (falls through to the
        existing emergency-stop path below), it just can't resume
        waiting for its DCA level.

        Tries save_state's persisted snapshot FIRST for each real
        position found (see STATE_PATH's own comment for why - it has
        full fidelity where exchange-shape reconciliation is ambiguous
        or lossy) via _try_restore_from_saved_state, falling back to the
        existing _adopt_position guessing only for a symbol that isn't
        in the saved state at all (first run, deleted/corrupted file, or
        a real position the bot never registered itself)."""
        live_positions = exchange.get_all_open_positions()
        adopted = 0
        restored = 0
        saved_state = self.load_state()

        for live_position in live_positions:
            symbol = live_position["symbol"]

            if symbol in self.positions:
                continue

            if self._try_restore_from_saved_state(symbol, live_position, saved_state):
                restored += 1
                adopted += 1
                continue

            # 2026-08-31: even when the full saved snapshot fails
            # _try_restore_from_saved_state's sanity check above (most
            # commonly a stale quantity - DCA fired on the real position
            # after the last periodic save_state() but before this
            # restart), its trade_id is still trustworthy - it never
            # changes when DCA fires, unlike quantity/entry_price. Side
            # must still match (same guard _try_restore_from_saved_state
            # itself uses) - a symbol that fully closed and reopened in
            # the OPPOSITE direction between the last save and this
            # restart must not inherit the old trade's identity. Real gap
            # this closes: without it, _adopt_position mints a brand-new
            # _RECOVERED_ trade_id, severing the link back to the original
            # signal (trigger, confluence, depth_imbalance) for the rest
            # of that trade's life - confirmed against real data (36 of
            # 315 trades, disproportionately the slow-resolving/DCA'd
            # ones).
            saved = saved_state.get(symbol)
            saved_trade_id = None

            if saved and saved.get("side") == live_position.get("side"):
                saved_trade_id = saved.get("trade_id")

            self._adopt_position(symbol, live_position, feed=feed, saved_trade_id=saved_trade_id)
            adopted += 1

        if live_positions:
            log_info(
                f"Startup reconciliation | {len(live_positions)} open "
                f"position(s) found on the exchange | {adopted} adopted "
                f"({restored} from saved state)"
            )

    def reconcile_pending_entries_on_startup(self):
        """A resting, not-yet-filled LIMIT entry has no recoverable plan
        after a restart - unlike a filled position (adopted + emergency-
        stopped by reconcile_on_startup/_adopt_position above), there's no
        structure_level/expiry/original-signal data to reconstruct here,
        only a guess. Cancel it outright and let the next eval tick
        re-evaluate that symbol fresh, matching this codebase's existing
        "close/cancel rather than guess" philosophy (_close_remainder_at_market,
        _market_close_tp1). Must run AFTER reconcile_on_startup() so a
        limit order that had already partially filled into a real
        position gets adopted (and protected) by the existing self-heal
        path first - this only ever touches still-resting orders with no
        exchange position attached.

        Covers BOTH config.LIMIT_ENTRY_MODE_ENABLED's and config.
        RETRACEMENT_ENTRY_ENABLED's resting orders - both place a plain
        (non-algo) LIMIT via exchange.place_limit_order, indistinguishable
        on this account-wide endpoint, and both get the exact same
        "no recoverable plan, cancel and re-evaluate fresh" treatment.
        This is what actually closes the "orphaned resting order" risk
        RETRACEMENT_ENTRY_ENABLED's own config.py comment describes -
        bounded to at most one poll cycle after a restart, not open-ended.

        config.DCA_RESTING_ORDER_ENABLED is the one deliberate exception
        to "cancel outright": unlike a pending ENTRY (no real position
        exists yet, nothing recoverable), a resting DCA-add order belongs
        to an already-open, real exchange position - blanket-cancelling
        it here would silently strip the exact protection this feature
        exists to provide, on every single restart (this bot restarts
        often - see config.DCA_RESTING_ORDER_ENABLED's own comment).
        Tagged orders (execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX) are
        recognized via their clientOrderId and re-attached to a matching,
        already-recovered DCA_PENDING position's dca_order_id instead -
        must run AFTER reconcile_on_startup() (already required above)
        so self.positions is populated by the time this checks it. A
        tagged order with NO matching tracked DCA_PENDING position
        (state lost/corrupt, or genuinely stale) still falls through to
        the same cancel-and-re-evaluate treatment as everything else -
        the exception is narrow, not a blanket exemption for the tag.

        Real bug found live (2026-08-26): the match condition used to
        require `not position.get("dca_order_id")` - but reconcile_on_
        startup's OWN preferred path (_try_restore_from_saved_state)
        restores the saved position dict verbatim, which already carries
        the real dca_order_id from before the restart. That's the common
        case for a DCA_PENDING position with a real resting order, not a
        rare one - so the "preserve" branch was failing on exactly the
        scenario this whole exception exists for, and silently cancelling
        a live resting DCA order on every restart. Confirmed live on
        MEUSDT/UBUSDT: the stale (now-cancelled) dca_order_id then also
        blocked poll_live's own candle-range fallback (an `elif`, only
        reached when dca_order_id is falsy) - DCA_PENDING has no real SL
        by design (register_dca_pending's own docstring), so both
        positions were left with neither a working DCA trigger nor a
        stop-loss until this fix. Now accepts a MATCHING dca_order_id
        too (confirms what saved-state already restored, doesn't need to
        "recover" anything) - only a genuine mismatch (a different,
        already-tracked order_id) still falls through to cancel."""
        if (
            not config.LIMIT_ENTRY_MODE_ENABLED
            and not config.RETRACEMENT_ENTRY_ENABLED
            and not config.DCA_RESTING_ORDER_ENABLED
        ):
            return

        cancelled = 0
        preserved = 0

        for order in exchange.get_all_open_orders():
            if str(order.get("type") or "").upper() != "LIMIT":
                continue

            symbol = order.get("symbol")
            order_id = order.get("orderId")

            if not symbol or not order_id:
                continue

            client_id = str(order.get("clientOrderId") or "")

            if client_id.startswith(execution.DCA_ADD_CLIENT_ORDER_ID_PREFIX):
                position = self.positions.get(symbol)

                if (
                    position
                    and position.get("stage") == DCA_PENDING
                    and (
                        not position.get("dca_order_id")
                        or position.get("dca_order_id") == order_id
                    )
                ):
                    position["dca_order_id"] = order_id
                    preserved += 1
                    log_info(
                        f"{symbol} recovered a resting DCA order on restart "
                        f"(order_id={order_id}) - preserved, not cancelled"
                    )
                    continue
                # No matching tracked DCA_PENDING position - genuinely
                # stray despite the tag (state lost/corrupt, or the
                # position resolved some other way before this ran) -
                # falls through to the same cancel below as everything
                # else.

            exchange.cancel_order(symbol, order_id)
            cancelled += 1
            log_warning(
                f"{symbol} cancelled a resting LIMIT order found on "
                f"restart (order_id={order_id}) - no recoverable pending-"
                f"entry plan survives a restart, re-evaluating fresh instead"
            )

        if cancelled or preserved:
            log_info(
                f"Startup reconciliation | {cancelled} resting limit "
                f"entry order(s) cancelled, {preserved} resting DCA "
                f"order(s) preserved"
            )

    def reconcile_stray_algo_orders_on_startup(self):
        """A TP1 partial (place_take_profit_partial) or DCA-breakeven
        trailing stop (place_trailing_stop_loss) is a reduceOnly, FIXED-
        quantity algo order - unlike SL/TP2 (place_stop_loss/place_take_
        profit_full, both closePosition=true), Binance does not auto-
        cancel these when the position they were sized for closes some
        other way. reconcile_closed_positions already cancels leftover
        orders for a symbol found closed while still tracked in self.
        positions - but a symbol whose position closed ENTIRELY while the
        bot was offline never re-enters self.positions at all (reconcile_
        on_startup only adopts symbols with a real currently-open
        exchange position), so nothing has ever cancelled its stray
        reduceOnly leg. Left alone it stays live indefinitely and, being
        a fixed quantity rather than closePosition=true, could fire
        unexpectedly against an unrelated future position opened on the
        same symbol later.

        Confirmed real, twice (RUNEUSDT, TWTUSDT - both closed entirely
        while untracked, both needed a manual cancel afterward because
        nothing in the bot's own reconciliation would ever have found
        them).

        Must run after reconcile_on_startup - self.positions must already
        reflect every symbol with a real open position by the time this
        runs."""
        cancelled = 0

        for order in exchange.get_all_open_algo_orders():
            symbol = order.get("symbol")
            algo_id = order.get("algoId")

            if not symbol or not algo_id or symbol in self.positions:
                continue

            exchange.cancel_algo_order(symbol, algo_id)
            cancelled += 1
            log_warning(
                f"{symbol} cancelled a stray algo order found on restart "
                f"(algoId={algo_id}, type={order.get('orderType')}) - no "
                f"tracked position exists for this symbol"
            )

        if cancelled:
            log_info(f"Startup reconciliation | {cancelled} stray algo order(s) cancelled")

    def reconcile_closed_positions(self):
        """Catches what poll_live structurally cannot: a tracked real
        (non-shadow) position closed OUTSIDE the bot entirely (manual
        close, ADL, liquidation). poll_live only ever detects a close by
        watching specific remembered order ids reach FINISHED - Binance
        auto-cancels/expires those same orders instead when something
        else closes the position first, so poll_live's checks silently
        fall through to None forever for exactly this case. Found live
        2026-08-29 (TRXUSDT/XPINUSDT: both closed by hand on the
        exchange, left an orphaned resting DCA order live indefinitely
        with nothing cancelling it).

        Must run AFTER the poll_live loop has already run for every
        symbol this tick (see main._poll_positions) - a position whose
        TP/SL genuinely finished THIS tick needs poll_live's own
        specific FINISHED check to claim it first and produce its real
        outcome (STATIC_TP_HIT, SL_HIT, DCA_TP_HIT, etc.); only a
        position still tracked after that loop gets checked here.
        Getting this ordering wrong would mislabel every ordinary TP/SL
        fill as CLOSED_EXTERNALLY.

        Fails OPEN: a failed/timed-out account-wide fetch is never
        treated as "no positions are open" (that would mass-close every
        real tracked position this tick on a transient API blip) - uses
        exchange._fetch_all_open_positions (raises) instead of
        get_all_open_positions (swallows to []) specifically so failure
        can be told apart from a genuinely empty account and skipped for
        this cycle instead."""
        try:
            real_open_symbols = {p["symbol"] for p in exchange._fetch_all_open_positions()}
        except Exception as exc:
            log_warning(
                f"reconcile_closed_positions: account-wide position fetch "
                f"failed, skipping this cycle: {exc}"
            )
            return

        for symbol, position in list(self.positions.items()):
            if position["shadow"]:
                continue

            if position["stage"] not in (TP1_PENDING, BREAKEVEN_ACTIVE, DCA_PENDING, DCA_ACTIVE):
                continue

            if symbol in real_open_symbols:
                continue

            log_warning(
                f"{symbol} tracked as {position['stage']} but has no real "
                f"open position on the exchange - closed externally (manual "
                f"close, ADL, or liquidation) - cleaning up leftover orders"
            )
            exchange.cancel_all_open_orders(symbol)
            self._close(symbol, "CLOSED_EXTERNALLY")

    def _try_restore_from_saved_state(self, symbol, live_position, saved_state):
        """Restores position_manager's own last-saved snapshot for
        `symbol` verbatim, if one exists and passes a cheap sanity check
        against the real exchange position - side must match, quantity
        must be close. That check exists for the real case that
        genuinely invalidates stale local state: a manual intervention
        on the exchange between the last save and this restart (see the
        XNYUSDT investigation, 2026-08-17) - a saved snapshot that no
        longer matches reality is worse than falling through to
        _adopt_position's own from-scratch reconciliation.

        Order ids/prices in the restored snapshot can be up to one
        POSITION_POLL_INTERVAL_SECONDS stale (save_state runs once per
        poll cycle, not on every mutation) - left to the existing self-
        heal/ground-truth-check machinery (_ensure_protection_orders,
        _replace_sl_order's own live check before acting) to catch and
        correct on the very next poll, same as it already does for a
        freshly-adopted position today."""
        saved = saved_state.get(symbol)

        if not saved:
            return False

        if saved.get("side") != live_position.get("side"):
            return False

        saved_quantity = saved.get("quantity")
        live_quantity = live_position.get("quantity")

        if not saved_quantity or not live_quantity or live_quantity <= 0:
            return False

        if abs(saved_quantity - live_quantity) / live_quantity > 0.01:
            log_warning(
                f"{symbol} saved position state quantity ({saved_quantity}) "
                f"doesn't match the real exchange quantity ({live_quantity}) "
                f"- likely a manual intervention since the last save, "
                f"falling back to reconciling from exchange state instead"
            )
            return False

        self.positions[symbol] = dict(saved)
        log_info(
            f"{symbol} restored from saved position state | side={saved.get('side')} "
            f"stage={saved.get('stage')} entry={saved.get('entry_price')}"
        )
        return True

    @staticmethod
    def _recover_dca_pending_position(
        symbol, side, entry_price, quantity, tp1_order, tp2_order, feed,
        saved_trade_id=None,
    ):
        """A real position with BOTH TP1 and TP2 resting but no SL at all,
        under config.DCA_ENABLED, is unambiguous - that's the only entry
        path DCA_ENABLED ever uses (see main._evaluate_symbol), and no
        other stage in this codebase produces that exact shape - so this
        is a genuine DCA_PENDING position, not an anomaly to protect
        defensively. Its exact original dca_price can't survive a
        restart (it only ever lived in memory - the whole DCA mechanism
        deliberately has no resting exchange order for it, see the DCA
        plan's accepted-risk note), but recomputing one fresh from
        CURRENT structure/ATR via the same deterministic risk_manager
        function signal time uses is honest, not a guess - if price
        already passed the new level while the bot was down, the DCA
        simply fires on the very next poll instead of being silently
        lost. Returns None (caller falls back to the pre-DCA emergency-
        stop path) when no candle history is available to compute it
        from - a symbol that fell out of the current watchlist, or a run
        with WS_ENABLED off."""
        candles = feed.candles.get(symbol) if feed is not None else []

        if not candles:
            return None

        pools = market_structure.find_liquidity_pools(market_structure.find_swing_points(candles))
        atr = market_structure.average_true_range(candles)
        dca_price = risk_manager.compute_dca_price(entry_price, side, pools, atr=atr)

        if dca_price is None or dca_price <= 0:
            return None

        def _trigger_price(order):
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        tp1_quantity = _safe_float(tp1_order.get("quantity") or tp1_order.get("origQty"))
        if tp1_quantity is None:
            tp1_quantity = round(quantity * min(max(float(config.TP1_CLOSE_PCT), 0), 100) / 100, 8)
        tp2_quantity = max(round(quantity - tp1_quantity, 8), 0)

        return {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            # Reference-only, never a real resting order in this stage -
            # same convention register_dca_pending's own sl_price follows.
            "sl_price": risk_manager._apply_min_stop_distance(entry_price, entry_price, side),
            "tp1_price": _trigger_price(tp1_order),
            "tp2_price": _trigger_price(tp2_order),
            "tp_price": None,
            "single_tp": False,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": tp1_quantity,
            "tp2_quantity": tp2_quantity,
            "sl_order_id": None,
            "tp1_order_id": exchange._accepted_order_id(tp1_order),
            "tp2_order_id": exchange._accepted_order_id(tp2_order),
            "tp_order_id": None,
            "stage": DCA_PENDING,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            # No original stop survives a restart to measure this from -
            # None (unknown) is honest, same policy the plain-adopt path
            # below already follows for a reconciled BREAKEVEN_ACTIVE.
            "risk_distance": None,
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
            "dca_price": dca_price,
            "dca_quantity": round(quantity * max(float(config.DCA_SIZE_MULTIPLIER), 0), 8),
            "dca_applied": False,
            "atr": atr,
        }

    @staticmethod
    def _recover_dca_pending_single_tp_position(
        symbol, side, entry_price, quantity, tp_order, feed,
        saved_trade_id=None,
    ):
        """config.TP_STATIC_ROI_ENABLED equivalent of
        _recover_dca_pending_position above - a real position with
        exactly ONE full-position TP resting and no SL at all, under
        config.DCA_ENABLED, is the single-TP DCA_PENDING shape (see
        risk_manager.build_trade_plan/register_dca_pending): no other
        stage in this codebase produces "one close_position=True TP,
        nothing else, no SL" while DCA_ENABLED is on. Same dca_price
        honesty policy as its dual-TP sibling - recomputed fresh from
        current structure/ATR, not recoverable from the restart itself.
        Returns None (caller falls back to the generic reconciliation
        path) when no candle history is available to compute it from."""
        candles = feed.candles.get(symbol) if feed is not None else []

        if not candles:
            return None

        pools = market_structure.find_liquidity_pools(market_structure.find_swing_points(candles))
        atr = market_structure.average_true_range(candles)
        dca_price = risk_manager.compute_dca_price(entry_price, side, pools, atr=atr)

        if dca_price is None or dca_price <= 0:
            return None

        def _trigger_price(order):
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        return {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            # Reference-only, never a real resting order in this stage -
            # same convention register_dca_pending's own sl_price follows.
            "sl_price": risk_manager._apply_min_stop_distance(entry_price, entry_price, side),
            "tp1_price": None,
            "tp2_price": None,
            "tp_price": _trigger_price(tp_order),
            "single_tp": True,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": None,
            "tp2_quantity": None,
            "sl_order_id": None,
            "tp1_order_id": None,
            "tp2_order_id": None,
            "tp_order_id": exchange._accepted_order_id(tp_order),
            "stage": DCA_PENDING,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            "risk_distance": None,
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
            "dca_price": dca_price,
            "dca_quantity": round(quantity * max(float(config.DCA_SIZE_MULTIPLIER), 0), 8),
            "dca_applied": False,
            "atr": atr,
        }

    @staticmethod
    def _recover_dca_protective_pending_position(
        symbol, side, entry_price, quantity, sl_order, tp1_order, tp2_order,
        saved_trade_id=None,
    ):
        """config.DCA_PROTECTIVE_FIRST_ENABLED equivalent of
        _recover_dca_pending_position above - a real position with a
        clientAlgoId-tagged (execution.DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_
        PREFIX) protective STOP_MARKET plus TP1+TP2 resting is unambiguous:
        no other stage in this codebase produces that exact shape. Unlike
        the legacy no-SL recovery above, dca_price doesn't need to be
        recomputed from current structure/ATR - it's read directly off
        the real, already-resting SL order's own trigger price, the same
        ground-truth-over-recomputation preference _recover_dca_active_
        position already uses for sl_price/tp_price. dca_quantity still
        needs a fresh value (nothing rests for it pre-escalation) - same
        DCA_SIZE_MULTIPLIER computation the legacy path uses, since
        escalation always fires at full size regardless of how a
        position was recovered."""
        dca_price = _safe_float(sl_order.get("triggerPrice") or sl_order.get("stopPrice"))

        if dca_price is None or dca_price <= 0:
            return None

        def _trigger_price(order):
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        tp1_quantity = _safe_float(tp1_order.get("quantity") or tp1_order.get("origQty"))
        if tp1_quantity is None:
            tp1_quantity = round(quantity * min(max(float(config.TP1_CLOSE_PCT), 0), 100) / 100, 8)
        tp2_quantity = max(round(quantity - tp1_quantity, 8), 0)

        return {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            "sl_price": risk_manager._apply_min_stop_distance(entry_price, entry_price, side),
            "tp1_price": _trigger_price(tp1_order),
            "tp2_price": _trigger_price(tp2_order),
            "tp_price": None,
            "single_tp": False,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": tp1_quantity,
            "tp2_quantity": tp2_quantity,
            "sl_order_id": None,
            "tp1_order_id": exchange._accepted_order_id(tp1_order),
            "tp2_order_id": exchange._accepted_order_id(tp2_order),
            "tp_order_id": None,
            "dca_order_id": None,
            "dca_protective_sl_order_id": exchange._accepted_order_id(sl_order),
            "dca_protective_stop_hit": None,
            "stage": DCA_PENDING,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            "risk_distance": None,
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
            "dca_price": dca_price,
            "dca_quantity": round(quantity * max(float(config.DCA_SIZE_MULTIPLIER), 0), 8),
            "dca_applied": False,
            "atr": None,
        }

    @staticmethod
    def _recover_dca_protective_pending_single_tp_position(
        symbol, side, entry_price, quantity, sl_order, tp_order,
        saved_trade_id=None,
    ):
        """Single-TP (config.TP_STATIC_ROI_ENABLED) sibling of
        _recover_dca_protective_pending_position above - same real-SL-
        trigger-price-over-recomputation approach, single-TP position
        shape instead of TP1+TP2."""
        dca_price = _safe_float(sl_order.get("triggerPrice") or sl_order.get("stopPrice"))

        if dca_price is None or dca_price <= 0:
            return None

        def _trigger_price(order):
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        return {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            "sl_price": risk_manager._apply_min_stop_distance(entry_price, entry_price, side),
            "tp1_price": None,
            "tp2_price": None,
            "tp_price": _trigger_price(tp_order),
            "single_tp": True,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": None,
            "tp2_quantity": None,
            "sl_order_id": None,
            "tp1_order_id": None,
            "tp2_order_id": None,
            "tp_order_id": exchange._accepted_order_id(tp_order),
            "dca_order_id": None,
            "dca_protective_sl_order_id": exchange._accepted_order_id(sl_order),
            "dca_protective_stop_hit": None,
            "stage": DCA_PENDING,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            "risk_distance": None,
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
            "dca_price": dca_price,
            "dca_quantity": round(quantity * max(float(config.DCA_SIZE_MULTIPLIER), 0), 8),
            "dca_applied": False,
            "atr": None,
        }

    @staticmethod
    def _recover_dca_active_position(
        symbol, side, entry_price, quantity, sl_order, tp_order,
        saved_trade_id=None,
    ):
        """sl_order's clientAlgoId carries the _DCA_SL_CLIENT_ALGO_ID_
        PREFIX tag _execute_dca stamped on it - unlike the DCA_PENDING
        recovery above, nothing needs to be recomputed here: the real
        sl_price/tp_price are read directly off the real resting orders,
        exactly as accurate as they were the moment DCA fired. Returns
        None (caller falls through to the ordinary BREAKEVEN_ACTIVE/
        TP1_PENDING adoption path) only if the tagged SL's trigger price
        can't be read at all - a real order existing with no readable
        price is the one case worth falling back rather than trusting."""
        def _trigger_price(order):
            if not order:
                return None
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        sl_price = _trigger_price(sl_order)

        if sl_price is None:
            return None

        tp_price = _trigger_price(tp_order)

        return {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp1_price": None,
            "tp2_price": None,
            "tp_price": tp_price,
            "single_tp": True,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": None,
            "tp2_quantity": None,
            "sl_order_id": exchange._accepted_order_id(sl_order),
            "tp1_order_id": None,
            "tp2_order_id": None,
            "tp_order_id": exchange._accepted_order_id(tp_order) if tp_order else None,
            "stage": DCA_ACTIVE,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            # Conservative on restart, same as the plain adoption path
            # below: we can't know from here whether profit protection
            # had already armed pre-restart, so this starts unarmed
            # rather than guessing. The real SL/TP stay exactly where
            # they already were regardless - this only affects whether a
            # fresh arm attempt happens on the next qualifying poll.
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": entry_price,
            "mfe_price": entry_price,
            # No original (pre-DCA) risk distance survives a restart -
            # same honesty policy _recover_dca_pending_position/the plain
            # BREAKEVEN_ACTIVE adoption path already follow.
            "risk_distance": None,
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
            "dca_applied": True,
            "dca_breakeven_applied": False,
            "dca_breakeven_direction_confirmed": None,
            # config.DCA_PRESSURE_CHECK_ENABLED - no record of what the
            # check found (if it even ran) survives a restart, same
            # honesty policy as dca_breakeven_direction_confirmed above.
            "dca_pressure_confirmed": None,
            "atr": None,
        }

    def _adopt_position(self, symbol, live_position, feed=None, saved_trade_id=None):
        side = live_position["side"]
        entry_price = live_position["entry_price"]
        quantity = live_position["quantity"]

        open_orders = exchange.get_open_algo_orders(symbol)
        sl_order = next((o for o in open_orders if _order_type(o) == "STOP_MARKET"), None)
        tp_orders = [o for o in open_orders if _order_type(o) == "TAKE_PROFIT_MARKET"]
        tp1_order = next(
            (o for o in tp_orders if str(o.get("closePosition")).lower() != "true"), None
        )
        tp2_order = next(
            (o for o in tp_orders if str(o.get("closePosition")).lower() == "true"), None
        )
        trail_order = next(
            (
                o for o in open_orders
                if _order_type(o) == "TRAILING_STOP_MARKET"
                and str(o.get("clientAlgoId") or "").startswith(_DCA_TRAIL_CLIENT_ALGO_ID_PREFIX)
            ),
            None,
        )

        def _trigger_price(order):
            if not order:
                return None
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        sl_price = _trigger_price(sl_order)

        if sl_order and str(sl_order.get("clientAlgoId") or "").startswith(_DCA_SL_CLIENT_ALGO_ID_PREFIX):
            recovered = self._recover_dca_active_position(
                symbol, side, entry_price, quantity, sl_order, tp2_order,
                saved_trade_id=saved_trade_id,
            )

            if recovered is not None:
                # Same clientAlgoId-based disambiguation _execute_dca's own
                # SL/TP tags already give this function - a resting dcaTrail
                # order restarts as-is (Binance keeps it live across a bot
                # restart), so recovery just has to notice it, not recreate it.
                recovered["dca_trail_order_id"] = (
                    exchange._accepted_order_id(trail_order) if trail_order else None
                )
                self.positions[symbol] = recovered
                log_info(
                    f"{symbol} adopted existing open position | side={side} "
                    f"entry={entry_price} sl={recovered['sl_price']} "
                    f"tp={recovered['tp_price']} stage=DCA_ACTIVE"
                )
                return

        # config.DCA_PROTECTIVE_FIRST_ENABLED - checked before the
        # sl_price-is-None branches below, since a protective-first
        # DCA_PENDING position now HAS a real SL-shaped order resting
        # (the whole point of this mechanism) - without this check first,
        # sl_price would read non-None and the position would silently
        # fall through into the ordinary post-TP1 recovery path further
        # down, losing its dca_price/pending semantics entirely. Same
        # clientAlgoId-tag disambiguation as the DCA_ACTIVE check above,
        # different prefix (execution.DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_
        # PREFIX, not _DCA_SL_CLIENT_ALGO_ID_PREFIX - a protective-first
        # SL was never through _execute_dca, so it's never tagged with
        # that one).
        if sl_order and str(sl_order.get("clientAlgoId") or "").startswith(
            execution.DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_PREFIX
        ):
            if tp1_order and tp2_order:
                recovered = self._recover_dca_protective_pending_position(
                    symbol, side, entry_price, quantity, sl_order, tp1_order, tp2_order,
                    saved_trade_id=saved_trade_id,
                )
            elif not tp1_order and tp2_order:
                recovered = self._recover_dca_protective_pending_single_tp_position(
                    symbol, side, entry_price, quantity, sl_order, tp2_order,
                    saved_trade_id=saved_trade_id,
                )
            else:
                recovered = None

            if recovered is not None:
                self.positions[symbol] = recovered
                log_info(
                    f"{symbol} adopted existing open position | side={side} "
                    f"entry={entry_price} stage=DCA_PENDING (protective stop "
                    f"@{recovered['dca_price']}) tp1={recovered['tp1_price']} "
                    f"tp2={recovered['tp2_price']} tp={recovered['tp_price']}"
                )
                return

        if sl_price is None and config.DCA_ENABLED and tp1_order and tp2_order:
            recovered = self._recover_dca_pending_position(
                symbol, side, entry_price, quantity, tp1_order, tp2_order, feed,
                saved_trade_id=saved_trade_id,
            )

            if recovered is not None:
                self.positions[symbol] = recovered
                log_info(
                    f"{symbol} adopted existing open position | side={side} "
                    f"entry={entry_price} stage=DCA_PENDING (no SL by design) "
                    f"dca_price={recovered['dca_price']} tp1={recovered['tp1_price']} "
                    f"tp2={recovered['tp2_price']}"
                )
                return

        # config.TP_STATIC_ROI_ENABLED - the single-TP DCA_PENDING shape:
        # exactly one full-position TP resting (tp2_order, per this
        # function's own close_position=True/False split above), no
        # partial TP1, no SL. Checked after the dual-TP case above (whose
        # own `tp1_order and tp2_order` condition is mutually exclusive
        # with `not tp1_order`) so order between them doesn't matter.
        if sl_price is None and config.DCA_ENABLED and not tp1_order and tp2_order:
            recovered = self._recover_dca_pending_single_tp_position(
                symbol, side, entry_price, quantity, tp2_order, feed,
                saved_trade_id=saved_trade_id,
            )

            if recovered is not None:
                self.positions[symbol] = recovered
                log_info(
                    f"{symbol} adopted existing open position | side={side} "
                    f"entry={entry_price} stage=DCA_PENDING (no SL by design, single TP) "
                    f"dca_price={recovered['dca_price']} tp={recovered['tp_price']}"
                )
                return

        if sl_price is None:
            # A real open position with no stop at all - treat as an
            # emergency: reconstruct a minimum-distance stop and place it
            # immediately rather than leave it unprotected until the next
            # opportunity to notice. Reached either because this isn't a
            # DCA_PENDING position at all, or it is one but no candle
            # history was available to recompute its dca_price above -
            # either way, protecting it now beats leaving it exposed.
            sl_price = risk_manager._apply_min_stop_distance(entry_price, entry_price, side)
            log_warning(
                f"{symbol} open position found with NO stop-loss during "
                f"startup reconciliation - placing an emergency stop at {sl_price}"
            )

        # Real trigger prices are used where an order actually exists;
        # anything missing gets a reconstructed target (current config's
        # R-multiples off the recovered SL) so _ensure_protection_orders
        # has a real price to place on the next poll - the same self-heal
        # path that already recovers a mid-session placement failure.
        fallback_tp1, fallback_tp2 = risk_manager.compute_targets(entry_price, sl_price, side)
        tp1_price = _trigger_price(tp1_order) or fallback_tp1
        tp2_price = _trigger_price(tp2_order) or fallback_tp2

        tp1_quantity = (
            _safe_float(tp1_order.get("quantity") or tp1_order.get("origQty"))
            if tp1_order else None
        )
        if tp1_quantity is None:
            tp1_quantity = round(quantity * min(max(float(config.TP1_CLOSE_PCT), 0), 100) / 100, 8)
        tp2_quantity = max(round(quantity - tp1_quantity, 8), 0)

        # Real bug found live (2026-08-10): "TP1 order absent" alone used
        # to be a reliable "already promoted" signal, but the early-1R
        # breakeven trigger moves the stop WITHOUT touching TP1 (it's
        # still a live, resting order - the trade just hasn't reached it
        # yet). A position early-promoted before a restart was getting
        # misclassified as still TP1_PENDING, which then computed
        # risk_distance from the already-moved (tiny) stop distance
        # instead of correctly marking it unrecoverable - and, worse, an
        # eventual stop-out on that misclassified position would log as a
        # full SL_HIT instead of the breakeven it actually was. Detect
        # "already protected" directly instead: a stop sitting on the
        # profit side of entry can only be a promoted stop (breakeven or
        # later) - an original risk-bearing stop is never there.
        sl_already_protects_profit = (
            (side == "BUY" and sl_price >= entry_price)
            or (side == "SELL" and sl_price <= entry_price)
        )
        stage = (
            BREAKEVEN_ACTIVE
            if (sl_already_protects_profit or (tp2_order and not tp1_order))
            else TP1_PENDING
        )

        position = {
            "symbol": symbol,
            "trade_id": saved_trade_id or f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "breakeven_price": risk_manager.compute_breakeven_price(entry_price, side),
            "quantity": quantity,
            "tp1_quantity": tp1_quantity,
            "tp2_quantity": tp2_quantity,
            "sl_order_id": exchange._accepted_order_id(sl_order) if sl_order else "",
            "tp1_order_id": exchange._accepted_order_id(tp1_order) if tp1_order else "",
            "tp2_order_id": exchange._accepted_order_id(tp2_order) if tp2_order else "",
            "stage": stage,
            "shadow": False,
            "opened_at": time.time(),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            # config.PROFIT_PROTECTION_ENABLED - mirrors the two keys
            # above but for the ROI-of-TP1 promotion path (mutually
            # exclusive with early_breakeven_applied/profit_locked at the
            # promotion moment - see _is_profit_protection_candidate).
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            # Best price reached since profit protection armed - None
            # until arming, seeded with the arm-time price, then updated
            # every poll while still trailing. See risk_manager.
            # compute_profit_protection_trailing_floor/
            # _trail_profit_protection_if_improved.
            "profit_protection_peak_price": None,
            # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - set True only if
            # a post-TP1 trailing stop replacement actually locked in more
            # than flat breakeven (see _trail_stop_if_improved and the
            # TRAILING_STOP_PROFIT_HIT outcome) - takes precedence over
            # early_breakeven_profit_locked in _breakeven_stop_outcome.
            "trailing_stop_locked_profit": False,
            # MAE/MFE tracking effectively restarts here too - there's no
            # way to recover the price path from before this restart.
            "mae_price": entry_price,
            "mfe_price": entry_price,
            # Real bug found live (2026-08-09): a position adopted while
            # already BREAKEVEN_ACTIVE has `sl_price` sitting at the
            # breakeven price, not the original stop - the true original
            # risk distance was lost the moment the exchange's SL order
            # got replaced, before this restart, and there is no way to
            # recover it now. Using abs(entry-sl_price) here for that case
            # would divide MAE/MFE by a near-zero breakeven-buffer
            # distance instead of the real risk, producing R-multiples in
            # the billions. None (unknown) is honest; a fabricated number
            # isn't. Only a TP1_PENDING adoption still has its real,
            # untouched original stop to compute this from.
            "risk_distance": abs(entry_price - sl_price) if stage == TP1_PENDING else None,
            # No original signal/candle to check this against - stays
            # unresolved forever for a reconciled position, same honesty
            # policy as risk_distance above.
            "structure_level": None,
            "trigger_candle_open_time": None,
            "break_confirmed_by_close": None,
        }

        if not sl_order:
            try:
                new_sl = exchange.place_stop_loss(symbol, side, sl_price)
                position["sl_order_id"] = exchange._accepted_order_id(new_sl)
            except Exception as exc:
                log_error(
                    f"{symbol} CRITICAL: failed to place emergency stop "
                    f"during startup reconciliation - manual intervention "
                    f"needed: {exc}"
                )

        self.positions[symbol] = position
        log_info(
            f"{symbol} adopted existing open position | side={side} "
            f"entry={entry_price} sl={sl_price} tp1={tp1_price} "
            f"tp2={tp2_price} stage={stage}"
        )

    def _close(self, symbol, outcome):
        position = self.positions.pop(symbol, None)

        if position:
            self._closed_at[symbol] = time.time()
            log_info(f"{symbol} position closed | OUTCOME={outcome}")
            mae_r, mfe_r = self._mae_mfe_r_multiples(position)
            signal_journal.append_outcome(
                symbol, outcome, position.get("trade_id"),
                mae_r_multiple=mae_r, mfe_r_multiple=mfe_r,
                early_breakeven_applied=position.get("early_breakeven_applied", False),
                break_confirmed_by_close=position.get("break_confirmed_by_close"),
                dca_applied=position.get("dca_applied", False),
                dca_breakeven_direction_confirmed=position.get("dca_breakeven_direction_confirmed"),
                dca_pressure_confirmed=position.get("dca_pressure_confirmed"),
                dca_protective_stop_hit=position.get("dca_protective_stop_hit"),
            )

        return outcome

    def resolve_break_confirmations(self, candle_store):
        """For every open position whose triggering LTF candle has since
        actually closed, records whether price held beyond the structure
        level it broke, or snapped back inside before that candle
        finished - i.e. was just a wick, not a real break. The entry
        itself fires the instant the live/forming candle breaks the level
        (that's the whole point of reacting in real time), so this is
        necessarily a look-back done a candle later, not a gate on entry.
        `candle_store` only needs a `.get(symbol)` returning the same
        candle-dict shape ws_client.CandleStore does."""
        for symbol, position in list(self.positions.items()):
            if position.get("break_confirmed_by_close") is not None:
                continue

            trigger_open_time = position.get("trigger_candle_open_time")
            structure_level = position.get("structure_level")

            if trigger_open_time is None or structure_level is None:
                continue

            trigger_candle = next(
                (c for c in candle_store.get(symbol) if c["open_time"] == trigger_open_time),
                None,
            )

            if not trigger_candle or not trigger_candle.get("closed"):
                continue

            side = position["side"]
            held = (
                trigger_candle["close"] > structure_level if side == "BUY"
                else trigger_candle["close"] < structure_level
            )
            position["break_confirmed_by_close"] = held

    @staticmethod
    def _update_mae_mfe(position, low_or_price, high_or_price=None):
        """Tracks the worst (MAE) and best (MFE) price seen over the life
        of a trade so far. Live mode passes a single mark-price sample
        (high_or_price defaults to the same value); shadow mode passes
        the current candle's actual low/high so a single candle's full
        range is captured, not just its close. See
        config.MAE_TRACKING_ENABLED for why this exists."""
        if not config.MAE_TRACKING_ENABLED:
            return

        if high_or_price is None:
            high_or_price = low_or_price

        if low_or_price is None or high_or_price is None:
            return

        side = position["side"]
        mae_price = position.get("mae_price", position["entry_price"])
        mfe_price = position.get("mfe_price", position["entry_price"])

        if side == "BUY":
            position["mae_price"] = min(mae_price, low_or_price)
            position["mfe_price"] = max(mfe_price, high_or_price)
        else:
            position["mae_price"] = max(mae_price, high_or_price)
            position["mfe_price"] = min(mfe_price, low_or_price)

    @staticmethod
    def _mae_mfe_r_multiples(position):
        """Both expressed as an R-multiple of the position's ORIGINAL risk
        distance (entry to its stop at the moment the trade opened) -
        comparable across symbols and volatility regimes, unlike a raw
        price distance. A LOSS with mfe_r_multiple near zero never moved
        favorably at all (wrong from the first tick); a LOSS with a large
        mfe_r_multiple was solidly in profit at some point before fully
        reversing - two very different problems that look identical as a
        plain outcome.

        Real bug found live (2026-08-09): this used to recompute
        abs(entry_price - position["sl_price"]) here directly - but
        sl_price is legitimately mutated over a trade's life (moved to
        the breakeven price on promotion, in both shadow mode's direct
        mutation and a startup-reconciliation adoption that picks up an
        already-breakeven real order). Once sl_price sits a few cents
        from entry instead of the original stop's real distance, dividing
        by it inflates the R-multiple into the billions. Must always use
        the fixed `risk_distance` captured once at position start
        (register()/_adopt_position()), never re-derive it from the
        current (possibly-moved) sl_price.

        config.DCA_ENABLED - a second bug of the same shape, found
        2026-08-31: _execute_dca legitimately overwrites entry_price/
        risk_distance with the new blended values once DCA fires, which
        silently changes what "1R" means mid-trade despite this
        function's own stated invariant above. original_entry_price/
        original_risk_distance (set once, in _execute_dca, before that
        overwrite) are preferred here when present so a DCA'd trade's
        MAE/MFE stay measured against the same original risk unit for
        its whole life, same as a trade that never DCA'd."""
        entry_price = position.get("original_entry_price") or position["entry_price"]
        risk_distance = position.get("original_risk_distance") or position.get("risk_distance")

        if risk_distance is None or risk_distance <= 0:
            return None, None

        mae_price = position.get("mae_price", entry_price)
        mfe_price = position.get("mfe_price", entry_price)
        mae_r = round(abs(entry_price - mae_price) / risk_distance, 4)
        mfe_r = round(abs(entry_price - mfe_price) / risk_distance, 4)
        return mae_r, mfe_r

    def _replace_sl_order(
        self, position, target_price, reason,
        close_if_not_open=True, not_open_outcome="TP1_THEN_POSITION_ALREADY_CLOSED",
    ):
        """Ground-truth cancel/replace of whatever SL is ACTUALLY sitting
        on the exchange (never a possibly-stale local id - a stale/wrong
        id cancels nothing, leaves the real order in place, and the
        placement below then fails with -4130 "already existing" every
        single poll forever). Shared by _promote_to_breakeven (the
        one-time TP1/early-breakeven promotion) and _trail_stop_if_improved
        (the repeated post-TP1 ratchet) - the atomic replace discipline
        must not be duplicated between them.

        Returns (outcome, replaced):
        - outcome: a close-outcome string if the position closed as a
          side effect of this call (only when close_if_not_open=True and
          it's already gone, or a -2021 forced a market-close of the
          remainder), else None.
        - replaced: True only if cancel+place both succeeded, in which
          case position["sl_order_id"]/["sl_price"] are already updated.

        Never touches position["stage"] - that responsibility stays with
        the caller (_promote_to_breakeven transitions to BREAKEVEN_ACTIVE
        on success; _trail_stop_if_improved never transitions at all,
        since it only ever runs once a position is already there)."""
        symbol = position["symbol"]

        try:
            live_position = exchange._fetch_open_position_detail(symbol)
        except Exception as exc:
            # Couldn't confirm ground truth this cycle (network/backoff) -
            # do nothing rather than guess; the next poll tries again.
            log_warning(f"{symbol} position-state check failed, retrying next poll: {exc}")
            return None, False

        if live_position is None:
            if close_if_not_open:
                # TP1 filling coincided with the position closing entirely
                # (e.g. the original SL also triggered) - nothing left to
                # promote. Stop retrying a doomed replacement.
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, not_open_outcome), False

            # A trailing attempt found the position already gone - do NOT
            # guess an outcome here (it would misclassify what's likely a
            # protected breakeven-or-better close as TP1_THEN_POSITION_ALREADY_CLOSED,
            # a LOSS in journal_analysis.py). The next poll's ordinary
            # sl_status/tp2_status check resolves and journals the real
            # outcome instead.
            log_warning(
                f"{symbol} position already closed by the time a SL "
                f"replacement was attempted - leaving it for the next "
                f"poll's own status check to resolve"
            )
            return None, False

        try:
            existing_sl = self._find_open_order(symbol, "STOP_MARKET", close_position=True)

            if existing_sl:
                exchange.cancel_algo_order(symbol, exchange._accepted_order_id(existing_sl))
            elif position["sl_order_id"]:
                exchange.cancel_algo_order(symbol, position["sl_order_id"])

            new_sl_order = exchange.place_stop_loss(
                symbol, position["side"], target_price
            )
            position["sl_order_id"] = exchange._accepted_order_id(new_sl_order)
            position["sl_price"] = target_price
            log_info(f"{symbol} {reason} | SL moved to {target_price}")

            # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a DIFFERENT
            # mechanism (profit protection or structure trailing; DCA
            # breakeven's own poll-based path never reaches here while a
            # real trail order exists - see _is_dca_breakeven_candidate)
            # has just taken over the SL. The native trailing stop's job
            # is done - retire it here, centrally, so every caller of this
            # function gets the cleanup for free and it can never later
            # conflict with a THIRD replace. No-op (.get() is falsy) for
            # every position that never had one.
            if position.get("dca_trail_order_id"):
                try:
                    exchange.cancel_algo_order(symbol, position["dca_trail_order_id"])
                except Exception as exc:
                    log_warning(
                        f"{symbol} dormant DCA trailing stop cleanup cancel "
                        f"failed (likely already gone): {exc}"
                    )
                position["dca_trail_order_id"] = None

            return None, True

        except Exception as exc:
            if "-2021" in str(exc):
                # The new level is already behind current price - Binance
                # refuses to place a stop that would fire instantly. The
                # remainder is effectively unprotected right now (the old
                # SL is already cancelled above), so close it at market
                # immediately instead of leaving it exposed and retrying
                # the same failing order.
                log_warning(
                    f"{symbol} new SL level already reached by price - "
                    f"closing remainder at market ({reason}, attempted "
                    f"target={target_price})"
                )
                return self._close_remainder_at_market(position), False

            log_error(f"{symbol} SL replacement error: {exc}")
            return None, False

    def _promote_to_breakeven(self, position, reason="TP1 filled", target_price=None):
        """Runs once TP1 is detected as filled (or, via `reason`, when a
        trade earns an early promotion before TP1 by reaching
        EARLY_BREAKEVEN_R_MULTIPLE in profit - see
        _is_early_breakeven_candidate). Returns an outcome string if the
        position closed as part of this call, otherwise None - the retry
        must never be silently repeated forever with no state change and
        no escape hatch, since that can leave a position genuinely
        unprotected between a failed cancel and a failed replace.

        `target_price` defaults to the flat (fee-buffer-only) breakeven
        price for a genuine TP1 fill; the early-breakeven caller passes a
        real locked-profit price instead (see
        _early_breakeven_lock_price).

        config.DCA_RESTING_ORDER_ENABLED - a genuine promotion means DCA
        will never be needed for this position (TP1/early-breakeven both
        only ever happen from favorable movement) - a still-resting DCA
        add order must not keep resting past this point, mirroring
        _execute_dca's own symmetric cleanup of TP1/TP2 when DCA fires
        first instead. .get() is safe/always None for a TP1_PENDING
        position (this function is shared by both stages), which never
        has this key at all."""
        target_price = position["breakeven_price"] if target_price is None else target_price

        dca_order_id = position.get("dca_order_id")

        if dca_order_id:
            try:
                exchange.cancel_order(position["symbol"], dca_order_id)
            except Exception as exc:
                log_warning(
                    f"{position['symbol']} resting DCA order cleanup cancel "
                    f"failed on promotion (likely already gone): {exc}"
                )
            position["dca_order_id"] = None

        if not config.MOVE_SL_TO_BREAKEVEN_AFTER_TP1:
            position["stage"] = BREAKEVEN_ACTIVE
            return None

        outcome, replaced = self._replace_sl_order(position, target_price, reason)

        if outcome is not None:
            return outcome

        if replaced:
            position["stage"] = BREAKEVEN_ACTIVE

        return None

    def _promote_to_breakeven_on_tp1_fill(self, position, candles=None):
        """A genuine TP1 fill used to promote the remainder to FLAT
        breakeven only (compute_breakeven_price - a fee-buffer, not a real
        profit) even though price has, by definition, already moved at
        least TP1_R_MULTIPLE R in the position's favor by this point -
        further than EARLY_BREAKEVEN_R_MULTIPLE's own trigger. Real gap
        found live (2026-08-18, operator observation): "closing from
        breakeven" was landing at a bare scratch instead of a small real
        profit. Reuses the same real-structure-first/EARLY_BREAKEVEN_LOCK_
        R_MULTIPLE lock _early_breakeven_lock_price already computes for
        the pre-TP1 trigger - same real evidence behind that number,
        applied here too instead of inventing a second one. Sets
        early_breakeven_profit_locked (not early_breakeven_applied - that
        flag means specifically "the pre-TP1 trigger fired", a different
        event) so an eventual stop-out on this ratcheted level is
        classified as EARLY_BREAKEVEN_PROFIT_HIT, not a plain scratch."""
        lock_price = self._early_breakeven_lock_price(position, candles)
        position["early_breakeven_profit_locked"] = _more_favorable(
            position["side"], lock_price, position["breakeven_price"]
        )
        return self._promote_to_breakeven(position, target_price=lock_price)

    def _close_remainder_at_market(self, position):
        symbol = position["symbol"]

        try:
            live_position = exchange._fetch_open_position_detail(symbol)
        except Exception as exc:
            log_error(f"{symbol} market-close position check error: {exc}")
            return None

        if live_position is None:
            exchange.cancel_all_open_orders(symbol)
            return self._close(symbol, "TP1_THEN_POSITION_ALREADY_CLOSED")

        try:
            exchange.close_position_market(
                symbol, position["side"], live_position["quantity"]
            )
            exchange.cancel_all_open_orders(symbol)

            outcome = (
                "TRAILING_STOP_PROFIT_HIT" if position.get("trailing_stop_locked_profit")
                else "PROFIT_PROTECTION_HIT" if position.get("profit_protection_profit_locked")
                else "EARLY_BREAKEVEN_PROFIT_HIT" if position.get("early_breakeven_profit_locked")
                # A DCA_ACTIVE position never has an EARLY_BREAKEVEN promotion
                # of its own (that path only exists pre-DCA), so this is
                # unambiguous - reached only when the -2021 fallback fires on
                # a DCA position's SL with no profit locked yet, i.e. a real
                # DCA stop-loss, not a leftover pre-DCA breakeven close.
                else "DCA_SL_HIT" if position.get("stage") == DCA_ACTIVE
                else "BREAKEVEN_TRIGGER_MARKET_CLOSE"
            )
            return self._close(symbol, outcome)

        except Exception as exc:
            log_error(f"{symbol} market-close-remainder error: {exc}")
            return None

    def _close_dca_remainder_as_tp_hit(self, position):
        """The TP-side mirror of _close_remainder_at_market's -2021
        handling - used only by _replace_dca_tp_order, when the new
        static-ROI target has already been reached/passed by the time the
        replacement order was attempted. Unlike _close_remainder_at_market
        (whose default assumes an SL-side close), this is unambiguously a
        real win - price already cleared the profit target - so it always
        closes as DCA_TP_HIT, never guessed from profit-lock flags."""
        symbol = position["symbol"]

        try:
            live_position = exchange._fetch_open_position_detail(symbol)
        except Exception as exc:
            log_error(f"{symbol} market-close position check error: {exc}")
            return None

        if live_position is None:
            exchange.cancel_all_open_orders(symbol)
            return self._close(symbol, "DCA_TP_HIT")

        try:
            exchange.close_position_market(
                symbol, position["side"], live_position["quantity"]
            )
            exchange.cancel_all_open_orders(symbol)
            return self._close(symbol, "DCA_TP_HIT")

        except Exception as exc:
            log_error(f"{symbol} market-close-remainder error: {exc}")
            return None

    def _replace_dca_tp_order(self, position, target_price):
        """Ground-truth cancel/replace of the resting DCA_ACTIVE TP order -
        same atomic-replace discipline as _replace_sl_order, mirrored for
        the TP side. Built for _migrate_dca_target_if_needed (config.
        DCA_TP_STATIC_ROI_ENABLED) - only ever called live (shadow
        positions never have a real order to replace, see
        _migrate_dca_target_if_needed's own shadow branch).

        Returns an outcome string if the position closed as a side effect
        of this call (already gone, or the new target was already reached
        by price - see _close_dca_remainder_as_tp_hit), else None. Never
        touches position["stage"] or anything else - purely a target-price
        update."""
        symbol = position["symbol"]

        try:
            live_position = exchange._fetch_open_position_detail(symbol)
        except Exception as exc:
            log_warning(f"{symbol} position-state check failed, retrying next poll: {exc}")
            return None

        if live_position is None:
            log_warning(
                f"{symbol} position already closed by the time a DCA TP "
                f"replacement was attempted - leaving it for the next "
                f"poll's own status check to resolve"
            )
            return None

        try:
            existing_tp = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=True)

            if existing_tp:
                exchange.cancel_algo_order(symbol, exchange._accepted_order_id(existing_tp))
            elif position.get("tp_order_id"):
                exchange.cancel_algo_order(symbol, position["tp_order_id"])

            new_tp_order = exchange.place_take_profit_full(
                symbol, position["side"], target_price,
                client_algo_id=f"{_DCA_TP_CLIENT_ALGO_ID_PREFIX}{int(time.time() * 1000)}",
            )
            position["tp_order_id"] = exchange._accepted_order_id(new_tp_order)
            position["tp_price"] = target_price
            log_info(f"{symbol} DCA TP migrated to static ROI target | TP moved to {target_price}")
            return None

        except Exception as exc:
            if "-2021" in str(exc):
                # The new target is already behind current price - Binance
                # refuses to place a take-profit that would fire instantly.
                # Unlike the SL-side equivalent, this is GOOD news (price
                # already reached the target) - close immediately as a
                # real win rather than leaving the position running on a
                # stale target and retrying the same failing order.
                log_warning(
                    f"{symbol} new DCA TP target already reached by price - "
                    f"closing remainder at market as a win (attempted "
                    f"target={target_price})"
                )
                return self._close_dca_remainder_as_tp_hit(position)

            log_error(f"{symbol} DCA TP replacement error: {exc}")
            return None

    def _migrate_dca_target_if_needed(self, position, current_price=None):
        """config.DCA_TP_STATIC_ROI_ENABLED - a DCA_ACTIVE position's
        tp_price is normally set once, at the moment DCA fires (risk_
        manager.compute_dca_target), and never touched again. This lets
        an ALREADY-open DCA_ACTIVE position pick up a live config change
        instead of running out its original target forever - real
        operator need (2026-08-19): flipping DCA_TP_STATIC_ROI_ENABLED,
        or editing DCA_TP_TARGET_ROI_PCT, should apply to positions
        already in flight, not just future DCA fires.

        Self-healing and idempotent: recomputes the CURRENT target every
        call and only replaces anything when it actually differs from
        position["tp_price"] - a no-op once a position is already on the
        right target (the same entry_price/side/ROI% always produce the
        exact same float, so nothing thrashes tick to tick).

        No-op entirely when DCA_TP_STATIC_ROI_ENABLED is off - the
        structure/R-multiple target's own inputs (pools) aren't available
        in poll_live/poll_shadow, so that path is only ever set at
        DCA-fire time, same as always.

        `current_price` (shadow only - poll_live goes through the real
        exchange's own -2021 rejection instead) lets a shadow position
        close immediately as a win if the new target was already passed,
        the same real-world effect _replace_dca_tp_order's -2021 handling
        produces live."""
        if not config.DCA_TP_STATIC_ROI_ENABLED or position["stage"] != DCA_ACTIVE:
            return None

        target = risk_manager.price_at_roi_pct(
            position["entry_price"], position["side"], config.DCA_TP_TARGET_ROI_PCT
        )

        if target is None or target == position.get("tp_price"):
            return None

        if not position["shadow"]:
            return self._replace_dca_tp_order(position, target)

        side = position["side"]

        if current_price is not None and (
            current_price >= target if side == "BUY" else current_price <= target
        ):
            return self._close(position["symbol"], "SHADOW_DCA_TP_HIT")

        position["tp_price"] = target
        log_info(
            f"{position['symbol']} [SHADOW] DCA TP migrated to static ROI target | TP -> {target}"
        )
        return None

    @staticmethod
    def _is_dca_candidate(position):
        """Cheap, no-network pre-check for config.DCA_ENABLED - only fetch
        a current price at all for a position that could actually use it:
        still waiting on the DCA-or-TP1 race, DCA not already fired."""
        return (
            config.DCA_ENABLED
            and position["stage"] == DCA_PENDING
            and not position.get("dca_applied")
            and position.get("dca_price") is not None
        )

    @staticmethod
    def _dca_price_reached_in_range(position, candles):
        """Has price reached position["dca_price"] at ANY point within
        the current (possibly still-forming) candle - not a single
        point-in-time sample. Real gap this closes (2026-08-24 price-path
        audit): the old point-price check (exchange.get_mark_price,
        sampled once every POSITION_POLL_INTERVAL_SECONDS) missed a touch
        whenever price crossed dca_price and reversed between two poll
        ticks - ~29% of resolved trades showed a large adverse excursion
        that never triggered DCA. candles[-1] is continuously updated by
        the kline websocket stream as new trades happen (ws_client.
        CandleStore's own "last item may still be forming" docstring), so
        its high/low reflects the full range covered since the candle
        opened - same range-check poll_shadow already uses for
        backtesting (touched_dca = low <= dca_price / high >= dca_price),
        now applied to live mode. None/empty candles or a candle missing
        high/low leaves this False - never fire on incomplete data."""
        if not candles:
            return False

        latest_candle = candles[-1]
        high, low = latest_candle.get("high"), latest_candle.get("low")

        if high is None or low is None:
            return False

        side = position["side"]
        dca_price = position["dca_price"]
        return low <= dca_price if side == "BUY" else high >= dca_price

    def _poll_dca_resting_order(
        self, position, candles=None, htf_candles=None, cvd_snapshot=None,
        current_price=None, crash_snapshot=None,
    ):
        """config.DCA_RESTING_ORDER_ENABLED - polls the resting DCA LIMIT
        order's real status instead of watching candle ranges for the
        trigger (_dca_price_reached_in_range stays the fallback for any
        position with no dca_order_id). A plain order, not algo, so
        exchange.get_order_status is the right primitive - same one
        poll_retracement_pending already uses, and the one that reports
        partial fills via executed_qty.

        Returns (fired, outcome): `fired` is False when nothing has
        changed yet (still resting, transient lookup failure, or the
        order is gone with zero fill) - the caller must fall through to
        the early-promotion checks exactly like the "not yet reached"
        case on the candle-range path. `fired` is True the moment ANY
        fill is detected, mirroring the candle-range path's own
        unconditional `return self._execute_dca(...)` - `outcome` is
        then whatever _execute_dca itself returned (None on success,
        since position["stage"] is DCA_ACTIVE by then and must not be
        treated by TP1_PENDING/DCA_PENDING-shaped checks further down
        this tick; a close-outcome string if the post-DCA SL placement
        failed)."""
        symbol = position["symbol"]
        order_id = position["dca_order_id"]
        order = exchange.get_order_status(symbol, order_id)

        if order["executed_qty"] <= 0:
            # Real bug found live (2026-08-26, MEUSDT/UBUSDT, root-caused
            # to reconcile_pending_entries_on_startup wrongly cancelling a
            # real resting DCA order on restart - now fixed there too):
            # a genuinely gone order used to read identically to "still
            # resting" here forever, since only executed_qty was ever
            # checked, never the order's own status. That silently
            # disabled BOTH this path (nothing left to ever fire) AND
            # poll_live's candle-range fallback (an `elif`, only reached
            # once dca_order_id is falsy) - DCA_PENDING carries no real
            # SL by design, so an affected position was left with
            # neither. Ground-truth self-heal: a terminal, not-going-to-
            # fill status clears dca_order_id so the very next poll falls
            # through to the fallback instead of waiting on a dead order
            # forever. "UNKNOWN" (a transient lookup failure - see
            # exchange.get_order_status's own docstring) deliberately
            # stays conservative here, same as "NEW"/"PARTIALLY_FILLED"
            # with nothing filled yet - only a real terminal status counts
            # as "gone", not a momentary API hiccup.
            if order["status"] in ("CANCELED", "EXPIRED", "REJECTED", "NOT_FOUND"):
                log_warning(
                    f"{symbol} resting DCA order {order_id} is gone "
                    f"(status={order['status']}) but was still tracked - "
                    f"clearing dca_order_id so the candle-range fallback "
                    f"takes over on the next poll"
                )
                position["dca_order_id"] = None

            return False, None

        outcome = self._execute_dca(
            position, candles=candles, htf_candles=htf_candles,
            cvd_snapshot=cvd_snapshot, current_price=current_price,
            crash_snapshot=crash_snapshot,
            resting_fill={"quantity": order["executed_qty"], "price": order["avg_price"]},
        )

        # Any unfilled remainder must not keep resting once the position
        # has already moved to the new post-DCA plan (or closed outright)
        # - best-effort, same treatment as every other post-fill cleanup
        # in this class; a harmless no-op if the order was already fully
        # filled or is already gone. Deliberately not a re-check-after-
        # cancel loop the way poll_retracement_pending's cancel-race
        # handling is - any fill that races this specific cancel is a
        # small, bounded imprecision (a slightly larger real add than the
        # blend math above already used), not the open-ended gap this
        # whole feature exists to close.
        try:
            exchange.cancel_order(symbol, order_id)
        except Exception as exc:
            log_warning(
                f"{symbol} DCA resting order cleanup cancel failed "
                f"(likely already fully filled or gone): {exc}"
            )

        return True, outcome

    def _poll_dca_protective_stop(self, position):
        """config.DCA_PROTECTIVE_FIRST_ENABLED - polls the resting
        protective STOP_MARKET's real status. An algo order (unlike
        _poll_dca_resting_order's plain LIMIT order), so exchange.
        get_algo_order_status is the right primitive.

        Returns (fired, outcome): `fired` is False when nothing has
        changed yet (still resting, or a transient lookup failure -
        get_algo_order_status's own docstring covers status meanings).
        `fired` is True once the stop has genuinely triggered - the
        position is closed via self._close(symbol, "DCA_PROTECTIVE_
        SL_HIT") exactly like every other SL-hit close in this file,
        after sweeping whatever TP1/TP2 (or single TP) is still resting
        (now irrelevant - the position is gone)."""
        symbol = position["symbol"]
        order_id = position["dca_protective_sl_order_id"]
        status = exchange.get_algo_order_status(symbol, order_id)

        if status == "FINISHED":
            exchange.cancel_all_open_orders(symbol)
            position["dca_protective_stop_hit"] = True
            return True, self._close(symbol, "DCA_PROTECTIVE_SL_HIT")

        if status in ("CANCELED", "EXPIRED", "REJECTED", "NOT_FOUND"):
            # Gone without ever firing - either _try_dca_protective_
            # escalation's own cancel (in which case dca_applied is
            # already True and this branch is moot - see poll_live's own
            # ordering), or an external cancel/expiry. Clear the id so
            # _ensure_protection_orders' self-heal re-places it on the
            # very next poll rather than leaving the position silently
            # unprotected forever - same ground-truth self-heal
            # discipline _poll_dca_resting_order already established for
            # the legacy mechanism.
            log_warning(
                f"{symbol} DCA protective stop {order_id} is gone "
                f"(status={status}) but was still tracked - clearing "
                f"dca_protective_sl_order_id so it gets re-placed next poll"
            )
            position["dca_protective_sl_order_id"] = None

        return False, None

    def _try_dca_protective_escalation(
        self, position, candles=None, htf_candles=None, cvd_snapshot=None,
        current_price=None, crash_snapshot=None,
    ):
        """config.DCA_PROTECTIVE_ESCALATION_ENABLED - the real-time "last
        look", only ever called once price has actually reached
        position["dca_price"] (caller's responsibility, via
        _dca_price_reached_in_range). Runs the same pressure check
        _execute_dca's own config.DCA_PRESSURE_CHECK_ENABLED branch
        would, but BEFORE any quantity is added rather than after - not
        confirmed means the resting protective stop simply stays armed
        and nothing else happens (the trade resolves via that stop,
        exactly as if escalation didn't exist). Confirmed means order
        flow genuinely still favors the original thesis, so this cancels
        the protective stop and fires a real, full-size DCA add via
        _execute_dca(pressure_confirmed_override=True) - inheriting that
        function's entire existing tail (build_dca_plan at full size,
        TP1/TP2 cancel, new TP+SL placement, and critically the atomic
        "SL placement failed -> market-close" safety net) completely
        unmodified.

        Ground-truth position check first (exchange._fetch_open_position_
        detail, same primitive _replace_sl_order already trusts for this
        exact class of race) - the resting protective stop may have
        already fired between this poll tick starting and now. Returns
        None when nothing closed this call (protective stop stays
        resting, or a transient check failure - retry next poll either
        way); a close-outcome string otherwise."""
        symbol = position["symbol"]
        confirmed, detail = self._dca_pressure_check(
            position, htf_candles, candles, cvd_snapshot, current_price, crash_snapshot,
        )
        position["dca_pressure_confirmed"] = confirmed

        if not confirmed:
            log_info(
                f"{symbol} DCA protective stop remains armed - escalation "
                f"declined | {detail}"
            )
            return None

        try:
            live_position = exchange._fetch_open_position_detail(symbol)
        except Exception as exc:
            log_warning(f"{symbol} DCA escalation position check failed, retrying next poll: {exc}")
            return None

        if live_position is None:
            exchange.cancel_all_open_orders(symbol)
            position["dca_protective_stop_hit"] = True
            return self._close(symbol, "DCA_PROTECTIVE_SL_HIT")

        try:
            existing = self._find_open_order(symbol, "STOP_MARKET", close_position=True)

            if existing:
                exchange.cancel_algo_order(symbol, exchange._accepted_order_id(existing))
            elif position.get("dca_protective_sl_order_id"):
                exchange.cancel_algo_order(symbol, position["dca_protective_sl_order_id"])
        except Exception as exc:
            log_warning(f"{symbol} DCA protective stop cancel failed (continuing): {exc}")

        position["dca_protective_sl_order_id"] = None

        return self._execute_dca(
            position, candles=candles, htf_candles=htf_candles,
            cvd_snapshot=cvd_snapshot, current_price=current_price,
            crash_snapshot=crash_snapshot, pressure_confirmed_override=True,
        )

    def _dca_pressure_check(
        self, position, htf_candles, candles, cvd_snapshot, current_price, crash_snapshot,
    ):
        """Shared by _execute_dca's own config.DCA_PRESSURE_CHECK_ENABLED
        branch and _try_dca_protective_escalation's real-time "last
        look" - same signal_engine.direction_still_confirmed call plus
        config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED forcing,
        factored out so both paths can never drift apart. Side-effect-
        free (does not touch `position`). Returns (confirmed, detail) -
        confirmed is direction_still_confirmed's own bool/None (None
        when the underlying data isn't available), detail is the
        diagnostic dict for logging."""
        side = position["side"]
        confirmed, detail = signal_engine.direction_still_confirmed(
            side, htf_candles, candles, cvd_snapshot, current_price
        )

        crash_snapshot_ = crash_snapshot or {}
        crash_forced = (
            config.CRASH_DETECTOR_ENABLED
            and config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED
            and crash_snapshot_.get("active")
            and (
                (side == "BUY" and crash_snapshot_.get("direction") == "BEARISH")
                or (side == "SELL" and crash_snapshot_.get("direction") == "BULLISH")
            )
        )

        if crash_forced and confirmed:
            confirmed = False
            detail = {
                **(detail or {}),
                "crash_detector_forced": True,
                "crash_direction": crash_snapshot_.get("direction"),
                "crash_move_pct": crash_snapshot_.get("pct_move"),
            }

        return confirmed, detail

    def _execute_dca(
        self, position, candles=None, htf_candles=None, cvd_snapshot=None, current_price=None,
        crash_snapshot=None, resting_fill=None, pressure_confirmed_override=None,
    ):
        """config.DCA_ENABLED - price reached position["dca_price"]
        before TP1 ever filled. Adds position["dca_quantity"] at that
        level, computes the blended entry / first-ever-real-SL / single-
        TP via risk_manager.build_dca_plan, cancels the original TP1+TP2,
        places the new single TP + the first real SL, and transitions to
        DCA_ACTIVE. Mirrors _apply_pending_fill's "SL must be atomic even
        when placed asynchronously" discipline: a failed post-DCA SL
        placement closes the position at market immediately rather than
        leave real (now-doubled) quantity unprotected.

        Live mode resolves the real average fill price for the DCA order
        (exchange.resolve_market_fill_price - same real-fill-price fix as
        the original entry, see _resolve_real_entry) and uses that instead
        of the planned dca_price trigger for the blended-entry/post-DCA-SL
        math - shadow mode has no real order to resolve, so it keeps
        trusting the planned trigger price exactly as before.

        config.DCA_PRESSURE_CHECK_ENABLED - `htf_candles`/`cvd_snapshot`/
        `current_price` (see config.py's comment for what this checks and
        why it adjusts size/stop rather than delaying the fire itself) are
        only used when that flag is on; a no-op without it, same
        convention as every other optional-data parameter in this class.

        config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED - extends the
        pressure check above rather than replacing it: if a market-wide
        crash is active in the direction this DCA is fighting (BUY DCA
        during a BEARISH crash, SELL DCA during a BULLISH one), pressure
        is forced not-confirmed regardless of what direction_still_
        confirmed found - a confirmed market-wide crash is stronger, more
        directly-evidenced signal than the per-symbol trend/CVD read.
        Deliberately does NOT skip the DCA itself (see crash_detector.py/
        config.py for why - this position has no real SL until this DCA
        places one).

        Returns a close-outcome string only if the post-DCA SL placement
        failed badly enough to force closing the position; otherwise
        None (including the ordinary case where DCA succeeded and the
        position lives on as DCA_ACTIVE).

        config.DCA_RESTING_ORDER_ENABLED - `resting_fill` (optional
        {"quantity", "price"}) is passed when poll_live already detected
        a real fill on the resting DCA LIMIT order (see poll_live's
        DCA_PENDING branch). When given, this skips the DCA_PRESSURE_
        CHECK_ENABLED block entirely - sizing already happened at
        placement time, always the conservative DCA_PRESSURE_SIZE_
        MULTIPLIER/DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER branch (see
        config.DCA_RESTING_ORDER_ENABLED's own comment for why) - and
        skips place_market_order below, since the fill already happened
        via the resting order. Everything after that (build_dca_plan,
        cancel TP1/TP2, place new TP/SL, atomic-SL-failure handling,
        DCA_ACTIVE transition) is unchanged - this is exactly why the
        fill is threaded through as a parameter rather than duplicating
        the function: the "SL must be atomic" safety discipline below is
        inherited for free either way.

        config.DCA_PROTECTIVE_FIRST_ENABLED - `pressure_confirmed_
        override` (optional bool) is given by _try_dca_protective_
        escalation once its own real-time pressure check already read
        True - skips the internal direction_still_confirmed recompute
        entirely and uses this value directly instead (escalation only
        ever calls with True, so this always takes the full DCA_SIZE_
        MULTIPLIER/DCA_STRUCTURE_STOP_ATR_BUFFER branch below, same as an
        ordinary confirmed DCA_PRESSURE_CHECK_ENABLED fire). Every
        existing call site passes nothing, preserving current behavior
        exactly - same "thread the decision through as a parameter,
        inherit the safety discipline for free" reasoning as
        `resting_fill` above."""
        symbol = position["symbol"]
        side = position["side"]
        shadow = position["shadow"]
        dca_fill_price = position["dca_price"]
        dca_quantity = position["dca_quantity"]
        buffer_atr_multiple = None
        pressure_confirmed = None
        pressure_detail = None

        if resting_fill is not None:
            dca_quantity = resting_fill["quantity"]
            dca_fill_price = resting_fill["price"]
            buffer_atr_multiple = max(float(config.DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER), 0)
        elif pressure_confirmed_override is not None:
            # Already checked in real time by _try_dca_protective_
            # escalation immediately before calling here - always True in
            # practice (escalation declines and never calls this function
            # at all when its own check reads False), so this always
            # takes the full-size/normal-buffer path below, same as an
            # ordinary confirmed DCA_PRESSURE_CHECK_ENABLED fire.
            pressure_confirmed = pressure_confirmed_override
        elif config.DCA_PRESSURE_CHECK_ENABLED:
            pressure_confirmed, pressure_detail = self._dca_pressure_check(
                position, htf_candles, candles, cvd_snapshot, current_price, crash_snapshot,
            )

            if not pressure_confirmed:
                dca_quantity = round(
                    position["quantity"] * max(float(config.DCA_PRESSURE_SIZE_MULTIPLIER), 0), 8
                )
                buffer_atr_multiple = max(float(config.DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER), 0)
                log_info(
                    f"{symbol}{' [SHADOW]' if shadow else ''} DCA pressure check: not "
                    f"confirmed - reduced size ({dca_quantity}) and tighter stop | {pressure_detail}"
                )

        position["dca_pressure_confirmed"] = pressure_confirmed

        if resting_fill is None and not shadow:
            try:
                dca_order = exchange.place_market_order(symbol, side, dca_quantity)
            except Exception as exc:
                log_error(f"{symbol} DCA order error: {exc}")
                return None  # retry next poll - not worse off than before

            dca_fill_price = exchange.resolve_market_fill_price(
                symbol, dca_order, position["dca_price"]
            )

        pools = (
            market_structure.find_liquidity_pools(market_structure.find_swing_points(candles))
            if candles else None
        )
        plan = risk_manager.build_dca_plan(
            position["entry_price"], position["quantity"],
            dca_fill_price, dca_quantity, side, pools,
            atr=position.get("atr"), buffer_atr_multiple=buffer_atr_multiple,
        )

        if plan is None:
            log_error(
                f"{symbol} DCA fill happened but a new SL/TP plan could not "
                f"be computed - will retry next poll"
            )
            return None

        if not shadow:
            try:
                existing_tp1 = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=False)

                if existing_tp1:
                    exchange.cancel_algo_order(symbol, exchange._accepted_order_id(existing_tp1))
                elif position["tp1_order_id"]:
                    exchange.cancel_algo_order(symbol, position["tp1_order_id"])

                existing_tp2 = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=True)

                if existing_tp2:
                    exchange.cancel_algo_order(symbol, exchange._accepted_order_id(existing_tp2))
                elif position["tp2_order_id"]:
                    exchange.cancel_algo_order(symbol, position["tp2_order_id"])
            except Exception as exc:
                log_warning(f"{symbol} DCA TP1/TP2 cancel error (continuing): {exc}")

            try:
                tp_order = exchange.place_take_profit_full(
                    symbol, side, plan["tp_price"],
                    client_algo_id=f"{_DCA_TP_CLIENT_ALGO_ID_PREFIX}{int(time.time() * 1000)}",
                )
                position["tp_order_id"] = exchange._accepted_order_id(tp_order)
            except Exception as exc:
                log_warning(f"{symbol} post-DCA single-TP placement failed: {exc}")
                position["tp_order_id"] = None

            try:
                sl_order = exchange.place_stop_loss(
                    symbol, side, plan["sl_price"],
                    client_algo_id=f"{_DCA_SL_CLIENT_ALGO_ID_PREFIX}{int(time.time() * 1000)}",
                )
                position["sl_order_id"] = exchange._accepted_order_id(sl_order)
            except Exception as exc:
                log_error(
                    f"{symbol} first real SL placement failed after DCA - "
                    f"closing position at market rather than leave it "
                    f"unprotected: {exc}"
                )
                try:
                    exchange.close_position_market(symbol, side, plan["quantity"])
                except Exception as close_exc:
                    log_error(
                        f"{symbol} CRITICAL: failed to close unprotected "
                        f"position after post-DCA SL failure - manual "
                        f"intervention needed: {close_exc}"
                    )
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_SL_PLACEMENT_FAILED")

            # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - see its own
            # config.py comment for the full evidence/rationale. Placed
            # strictly AFTER the real-SL try/except above has already
            # succeeded (reaching this line proves it did, since every
            # failure path there returns early) - deliberately its own,
            # separate try/except so a failure here can NEVER be mistaken
            # for the atomic "first real SL placement failed" case above
            # and trigger an emergency market-close the position doesn't
            # need. Best-effort/non-fatal, same treatment as the TP
            # placement above: a rejection just means dca_trail_order_id
            # stays None, and DCA_BREAKEVEN_ENABLED's existing poll-based
            # mechanism (_is_dca_breakeven_candidate) remains the fallback
            # exactly as it works today.
            position["dca_trail_order_id"] = None

            if config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED:
                breakeven_price = risk_manager.compute_breakeven_price(plan["entry_price"], side)

                try:
                    trail_order = exchange.place_trailing_stop_loss(
                        symbol, side, plan["quantity"], breakeven_price,
                        config.DCA_BREAKEVEN_TRAILING_CALLBACK_RATE,
                        client_algo_id=f"{_DCA_TRAIL_CLIENT_ALGO_ID_PREFIX}{int(time.time() * 1000)}",
                    )
                    position["dca_trail_order_id"] = exchange._accepted_order_id(trail_order)
                except Exception as exc:
                    log_warning(
                        f"{symbol} dormant DCA-breakeven trailing stop failed "
                        f"(falling back to poll-based breakeven move): {exc}"
                    )
        else:
            position["tp_order_id"] = None
            position["dca_trail_order_id"] = None

        # 2026-08-31: capture the ORIGINAL (pre-DCA) entry/risk/quantity
        # before they're overwritten below - needed both to keep
        # _mae_mfe_r_multiples accurate for a DCA'd trade (its own
        # docstring already requires "risk captured once at position
        # start", broken by the overwrite below) and for
        # config.DCA_MAX_ADVERSE_R_ENABLED. original_quantity specifically
        # exists for that check: real evidence (2026-08-31, BANKUSDT) - a
        # PRICE-distance version of this check (adverse price move /
        # original risk distance) understates real dollar risk once DCA
        # has grown the position, since the same price move past the DCA
        # fill now applies to MORE quantity than the original 1R was ever
        # sized for. The check compares real unrealized DOLLARS lost
        # (current blended entry/quantity) against original_risk_distance
        # * original_quantity - the actual dollar amount the ORIGINAL,
        # single-entry plan would have risked - not a price-only ratio.
        # Guarded so a hypothetical future re-fire can't clobber the real
        # original - this project's DCA is single-fire by design, but the
        # guard is free insurance.
        if position.get("original_risk_distance") is None:
            position["original_entry_price"] = position["entry_price"]
            position["original_risk_distance"] = position["risk_distance"]
            position["original_quantity"] = position["quantity"]

        position["entry_price"] = plan["entry_price"]
        position["sl_price"] = plan["sl_price"]
        position["tp_price"] = plan["tp_price"]
        position["quantity"] = plan["quantity"]
        position["risk_distance"] = plan["risk_distance"]
        # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - _trail_stop_if_
        # improved derives trailing_stop_locked_profit from position[
        # "breakeven_price"] directly (not recomputed each call, unlike
        # _structure_stop_candidate's own internal breakeven check) - left
        # stale at the ORIGINAL (pre-DCA) entry's breakeven, that flag
        # would misclassify a genuine post-DCA trail against the wrong
        # reference price. Recomputed here from the new blended entry.
        position["breakeven_price"] = risk_manager.compute_breakeven_price(
            plan["entry_price"], side
        )
        position["dca_applied"] = True
        # config.DCA_BREAKEVEN_ENABLED - DCA_ACTIVE always starts unarmed,
        # same reasoning as profit_protection_applied above (a DCA that
        # fires always comes from DCA_PENDING, never an already-promoted
        # position).
        position["dca_breakeven_applied"] = False
        # config.DCA_BREAKEVEN_CONFIRMATION_ENABLED - None until the
        # breakeven check actually runs at least once (distinct from
        # False, "ran and wasn't confirmed") - see _dca_breakeven_confirmation.
        position["dca_breakeven_direction_confirmed"] = None
        # config.TP_STATIC_ROI_ENABLED - DCA_ACTIVE is always single-TP
        # shaped regardless of what this position was in DCA_PENDING (TP1/
        # TP2 were just cancelled above either way) - not read by any
        # DCA_ACTIVE code path (those key off stage directly), set here
        # only for cross-stage consistency.
        position["single_tp"] = True
        # config.DCA_RESTING_ORDER_ENABLED - whatever was resting (fully
        # consumed, or its remainder already cancelled by poll_live's
        # DCA_PENDING branch before calling here) no longer applies once
        # DCA_ACTIVE - cleared so nothing downstream mistakes a stale id
        # for a still-live order.
        position["dca_order_id"] = None
        position["stage"] = DCA_ACTIVE
        log_info(
            f"{symbol}{' [SHADOW]' if shadow else ''} DCA fired | "
            f"new entry={plan['entry_price']} SL={plan['sl_price']} "
            f"TP={plan['tp_price']} qty={plan['quantity']}"
        )
        return None

    def _is_early_breakeven_candidate(self, position):
        """Cheap, no-network pre-check for early breakeven
        (config.EARLY_BREAKEVEN_ENABLED) - only fetch a current price at
        all (an extra REST call in poll_live) for a position that could
        actually qualify: TP1 still pending, not already promoted.

        Originally gated on confluence_ratio (protect low-confidence
        trades faster) - real evidence (2026-08-10, 61 resolved LOSS
        trades) didn't support that: confluence showed no correlation
        with outcome. What the same data DID show clearly: MFE for LOSS
        trades is bimodal, not a smooth spread - 38% near zero (wrong
        from the first tick, no fix here helps them) but 28% ran 1.0R+ in
        profit before fully reversing to a full loss, completely
        unprotected the whole way down since nothing moves the stop until
        TP1 formally triggers at 2R. This now targets that second,
        evidence-backed population directly - every trade still waiting
        on TP1, not just ones with low confluence."""
        if not config.EARLY_BREAKEVEN_ENABLED or position.get("early_breakeven_applied"):
            return False

        # config.DCA_ENABLED - a DCA-pending position is ALSO still
        # waiting on TP1 to genuinely fill (no real SL exists yet either
        # way), the same "hasn't been promoted yet" condition TP1_PENDING
        # represents - real gap found live (2026-08-17, operator
        # question): this used to check `!= TP1_PENDING` only, so while
        # DCA_ENABLED is True (every position starts in DCA_PENDING, not
        # TP1_PENDING) this could never arm at all, silently.
        if position["stage"] not in (TP1_PENDING, DCA_PENDING):
            return False

        # config.TP_STATIC_ROI_ENABLED - a single-TP DCA_PENDING position
        # is deliberately kept simple: no partial TP1, no early-arm
        # mechanisms layered on top, just the DCA-vs-single-TP race. Early
        # breakeven/profit protection promote toward BREAKEVEN_ACTIVE, a
        # stage built around the tp1-filled/tp2-still-open shape - not a
        # fit for a position that has neither.
        if position.get("single_tp"):
            return False

        return abs(position["entry_price"] - position["sl_price"]) > 0

    @staticmethod
    def _early_breakeven_price_reached(position, current_price):
        """Has price moved EARLY_BREAKEVEN_R_MULTIPLE R in the position's
        favor yet? Shared by poll_live (real mark price) and poll_shadow
        (simulated candle close)."""
        if current_price is None or current_price <= 0:
            return False

        side = position["side"]
        entry_price = position["entry_price"]
        risk_distance = abs(entry_price - position["sl_price"])
        trigger_distance = risk_distance * max(float(config.EARLY_BREAKEVEN_R_MULTIPLE), 0)
        favorable_move = (
            current_price - entry_price if side == "BUY" else entry_price - current_price
        )
        return favorable_move >= trigger_distance

    @staticmethod
    def _early_breakeven_lock_price(position, candles=None):
        """Where the stop goes on THIS early promotion. config.STRUCTURE_STOP_MANAGEMENT_ENABLED
        tries the most recent confirmed market-structure swing first (see
        _structure_stop_candidate); the fixed-distance calculation below
        is both the original behavior (feature off) and the fallback when
        no swing is available yet. Uses the fixed `risk_distance` captured
        once at position start (same discipline as _mae_mfe_r_multiples),
        not a re-derived value, since sl_price is still the original
        (unmoved) stop at this point anyway."""
        side = position["side"]

        if config.STRUCTURE_STOP_MANAGEMENT_ENABLED:
            candidate = _structure_stop_candidate(position, candles)

            if candidate is not None:
                return candidate

        fixed = risk_manager.compute_early_breakeven_price(
            position["entry_price"], side, position["risk_distance"]
        )
        breakeven = risk_manager.compute_breakeven_price(position["entry_price"], side)
        return max(fixed, breakeven) if side == "BUY" else min(fixed, breakeven)

    def _is_profit_protection_candidate(self, position):
        """Cheap, no-network pre-check for config.PROFIT_PROTECTION_ENABLED -
        same shape as _is_early_breakeven_candidate, a different metric
        (% of TP1's own ROI instead of an R-multiple of risk_distance).
        See config.py's comment on PROFIT_PROTECTION_ENABLED for the real
        motivation (TP1/TP2 can take a long time, and EARLY_BREAKEVEN's
        flat 0.5R/0.3R lock doesn't scale with how much TP1 itself would
        actually pay out on a given trade)."""
        if not config.PROFIT_PROTECTION_ENABLED or position.get("profit_protection_applied"):
            return False

        # config.DCA_ENABLED - see the identical note in
        # _is_early_breakeven_candidate.
        if position["stage"] not in (TP1_PENDING, DCA_PENDING):
            return False

        # config.TP_STATIC_ROI_ENABLED - see the identical note in
        # _is_early_breakeven_candidate.
        if position.get("single_tp"):
            return False

        # config.PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED - see
        # config.py's own comment for the full reasoning. TP1 here is a
        # small, fixed ROI% target, not the variable/potentially-large
        # one the activation tiers above were built around - the
        # post-TP1 remainder toward a real TP2 (_is_dca_profit_
        # protection_candidate) is a separate check, untouched by this.
        if config.PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED and config.TP_STATIC_ROI_ENABLED:
            return False

        return position.get("tp1_price") is not None

    def _is_dca_profit_protection_candidate(self, position):
        """config.PROFIT_PROTECTION_ENABLED, once a DCA has actually
        fired (stage DCA_ACTIVE) - a separate eligibility check from
        _is_profit_protection_candidate above, not a stage addition to
        it: DCA_ACTIVE has no tp1_price concept left (TP1/TP2 were
        cancelled and replaced by the single tp_price - see
        _execute_dca), and a DCA_ACTIVE position starts this stage with
        profit_protection_applied guaranteed False - a DCA that fires
        always came from DCA_PENDING, never from an already-armed
        promotion (that path goes to BREAKEVEN_ACTIVE instead, and DCA
        never fires from there) - so it needs its own fresh arm step
        here, not just a trail. See _is_tp2_profit_protection_candidate
        below for BREAKEVEN_ACTIVE's own equivalent (a genuine TP1 fill
        leaves the same flag False too, for the same underlying reason)."""
        if not config.PROFIT_PROTECTION_ENABLED or position.get("profit_protection_applied"):
            return False

        if position["stage"] != DCA_ACTIVE:
            return False

        return position.get("tp_price") is not None

    def _is_tp2_profit_protection_candidate(self, position):
        """config.PROFIT_PROTECTION_TP2_LEG_ENABLED - real gap found live
        (2026-08-23): a position that reached BREAKEVEN_ACTIVE via a
        GENUINE TP1 fill (_promote_to_breakeven_on_tp1_fill) starts that
        leg with profit_protection_applied still False, same as
        DCA_ACTIVE's own gap above - but unlike DCA_ACTIVE, nothing ever
        gave this leg a fresh-arm step; BREAKEVEN_ACTIVE's own poll logic
        only ever TRAILS an already-armed lock (see
        _trail_profit_protection_if_improved's `if position.get(
        "profit_protection_applied")` guard). TP2 is always real,
        structure-resolved - never a small fixed target the way TP1 can
        be - so it fits this mechanism's premise exactly. Deliberately
        doesn't distinguish a genuine TP1 fill from an early-breakeven/
        pre-TP1-profit-protection promotion into this same stage - the
        latter either already has profit_protection_applied True
        (excluded below) or arrived via early breakeven instead (still
        False, and still a legitimate BREAKEVEN_ACTIVE position worth
        protecting toward tp2_price either way)."""
        if not config.PROFIT_PROTECTION_ENABLED or not config.PROFIT_PROTECTION_TP2_LEG_ENABLED:
            return False

        if position.get("profit_protection_applied"):
            return False

        if position["stage"] != BREAKEVEN_ACTIVE:
            return False

        return position.get("tp2_price") is not None

    @staticmethod
    def _is_dca_breakeven_candidate(position):
        """config.DCA_BREAKEVEN_ENABLED - see its config.py comment for
        the gap this closes: between firing and either PROFIT_PROTECTION's
        much deeper ROI-of-TP threshold or a lagging confirmed structure
        swing, a DCA_ACTIVE position that merely recovers to breakeven has
        nothing protecting it. One-time arm, same shape as
        _is_dca_profit_protection_candidate - dca_breakeven_applied is set
        the moment it fires and never re-checked again for this position.

        config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a resting
        dca_trail_order_id means a native trailing stop is already doing
        this job (or waiting to), server-side - this poll-based mechanism
        becomes a pure fallback for whenever that flag is off or its
        placement failed for this specific position. Without this check,
        the poll-based path would detect the exact same "price reached
        breakeven" condition, replace the SL with a FLAT stop, and (via
        _replace_sl_order's own cleanup) cancel the trailing stop that was
        just protecting the position better - racing its own fix back
        down to the old behavior on literally the first tick both are
        true."""
        if not config.DCA_BREAKEVEN_ENABLED or position.get("dca_breakeven_applied"):
            return False

        if position.get("dca_trail_order_id"):
            return False

        return position["stage"] == DCA_ACTIVE

    @staticmethod
    def _dca_breakeven_price_reached_in_range(position, candles):
        """Has price reached position["breakeven_price"] at ANY point
        within the current (possibly still-forming) candle - not a
        single point-in-time sample. Same class of gap _dca_price_
        reached_in_range closed for the DCA entry trigger (2026-08-24):
        a point-sampled check can miss a brief touch of breakeven, and
        even when a later poll catches the recovery, by the time the SL
        replacement order actually reaches the exchange price may have
        already reversed back past breakeven - Binance rejects the
        placement with -2021 ("would immediately trigger") and
        _replace_sl_order's existing fallback closes the remainder at
        MARKET at whatever price it's actually at then, which can land
        the trade at a real loss despite the buffer. Catching the touch
        as early as possible (this candle's actual high/low, not a
        delayed point sample) narrows that race window. None/empty
        candles or a candle missing high/low leaves this False - never
        fire on incomplete data.

        poll_live only - candles is its sole source of candle data there.
        poll_shadow does the same high/low comparison inline instead,
        against its own already-guaranteed latest_candle parameter
        (candles is a separate, optional parameter in poll_shadow, not
        reliably in sync with latest_candle the way it is in poll_live)."""
        if not candles:
            return False

        latest_candle = candles[-1]
        high, low = latest_candle.get("high"), latest_candle.get("low")

        if high is None or low is None:
            return False

        side = position["side"]
        breakeven = position["breakeven_price"]
        return high >= breakeven if side == "BUY" else low <= breakeven

    @staticmethod
    def _dca_breakeven_confirmation(position, htf_candles, ltf_candles, cvd_snapshot, current_price):
        """config.DCA_BREAKEVEN_CONFIRMATION_ENABLED / ..._WITHHOLD_ENABLED
        - see signal_engine.direction_still_confirmed for what's actually
        checked. Always returns (withhold, confirmed, detail) so callers
        can journal confirmed/detail unconditionally without a None check
        of their own - confirmed/detail are only ever None when the
        master flag itself is off (the feature has never run at all, a
        real distinction from "ran and found it not confirmed").

        `withhold` is the one bit that actually changes behavior, kept
        separate from `confirmed` on purpose: the two-phase rollout this
        was built for runs with confirmed/detail journaled for a while
        before DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED is ever flipped
        on, so a caller must never infer "withhold" just because
        `confirmed` came back True."""
        if not config.DCA_BREAKEVEN_CONFIRMATION_ENABLED:
            return False, None, None

        confirmed, detail = signal_engine.direction_still_confirmed(
            position["side"], htf_candles, ltf_candles, cvd_snapshot, current_price
        )
        withhold = confirmed and config.DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED
        return withhold, confirmed, detail

    @staticmethod
    def _profit_protection_lock_price(position, target_price=None):
        """The ARM trigger price only - see risk_manager.
        compute_profit_protection_lock_price. Once armed, the stop no
        longer jumps straight to this price; _profit_protection_trailing_
        floor takes over (see its docstring and config.py's comment on
        PROFIT_PROTECTION_LOCK_PCT_OF_TP1 for why).

        `target_price` defaults to position["tp1_price"] (the pre-DCA
        case) - config.DCA_ENABLED callers pass position["tp_price"]
        instead once a DCA has fired (DCA_ACTIVE has no TP1 concept
        left), since PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1/LOCK_PCT_
        OF_TP1 are really just "% of what THE relevant target's own ROI
        would pay out" - same math, scaled against whichever single
        target actually applies to this position right now."""
        target_price = position["tp1_price"] if target_price is None else target_price
        return risk_manager.compute_profit_protection_lock_price(
            position["entry_price"], position["side"], target_price
        )

    @staticmethod
    def _profit_protection_trailing_floor(position, peak_price, target_price=None):
        """Where the stop actually gets set, both at arm time and on
        every subsequent trail - see risk_manager.
        compute_profit_protection_trailing_floor. `target_price` - see
        _profit_protection_lock_price's docstring."""
        target_price = position["tp1_price"] if target_price is None else target_price
        return risk_manager.compute_profit_protection_trailing_floor(
            position["entry_price"], position["side"], target_price, peak_price
        )

    def _trail_profit_protection_if_improved(self, position, current_price):
        """Continuous companion to the one-time arm above: once armed,
        keeps updating position["profit_protection_peak_price"] with the
        best price seen and re-derives the trailing floor from it every
        call, replacing the real SL only when that's strictly more
        favorable than the current one (ratchet-only, same discipline and
        the same shared _replace_sl_order helper as _trail_stop_if_
        improved - the two run independently and never loosen each
        other's work). Returns a close-outcome string only if the replace
        attempt forced a market-close of the remainder (-2021), otherwise
        None. No-op for a position that never armed, or without a usable
        current_price."""
        if not position.get("profit_protection_applied"):
            return None

        if current_price is None or current_price <= 0:
            return None

        side = position["side"]
        peak_price = position.get("profit_protection_peak_price")
        peak_price = current_price if peak_price is None else (
            max(peak_price, current_price) if side == "BUY" else min(peak_price, current_price)
        )
        position["profit_protection_peak_price"] = peak_price

        # config.PROFIT_PROTECTION_TP2_LEG_ENABLED - which target this
        # position actually armed against varies now (tp1_price for a
        # pre-TP1 arm, tp_price post-DCA, tp2_price for a genuine-TP1-fill
        # BREAKEVEN_ACTIVE arm - see every profit_protection_applied=True
        # site for where this gets set). Falls back to the pre-existing
        # DCA_ACTIVE-vs-tp1_price stage-based guess only for a position
        # restored from saved state written before this field existed.
        target_field = position.get("profit_protection_target")

        if target_field is None:
            target_field = "tp_price" if position["stage"] == DCA_ACTIVE else "tp1_price"

        target_price = position[target_field]
        candidate = self._profit_protection_trailing_floor(position, peak_price, target_price=target_price)

        if candidate is None or not _more_favorable(side, candidate, position["sl_price"]):
            return None

        return self._replace_sl_order(
            position, candidate, "Profit protection trailing stop", close_if_not_open=False
        )[0]

    def _profit_protection_price_reached(self, position, current_price, target_price=None):
        """Has price reached the profit-protection lock price yet? Shared
        by poll_live (real mark price) and poll_shadow (simulated candle
        close) - same pattern as _early_breakeven_price_reached.
        `target_price` - see _profit_protection_lock_price's docstring."""
        if current_price is None or current_price <= 0:
            return False

        lock_price = self._profit_protection_lock_price(position, target_price=target_price)

        if lock_price is None:
            return False

        side = position["side"]
        return current_price >= lock_price if side == "BUY" else current_price <= lock_price

    def _try_early_promotions(
        self, position, current_price, candles, profit_protection_candidate, early_breakeven_candidate,
    ):
        """Shared by poll_live's TP1_PENDING and DCA_PENDING branches -
        both are "still waiting on TP1 to genuinely fill, no real SL
        placed by this promotion path yet" states (see
        _is_early_breakeven_candidate/_is_profit_protection_candidate,
        both of which now accept either stage - config.DCA_ENABLED gap
        found live, 2026-08-17). Checked before EARLY_BREAKEVEN: mutually
        exclusive at the promotion moment (both check the same stage set
        and stop applying once promoted), so whichever fires here is
        simply whichever threshold was reached first in time,

        Returns True if a promotion happened this call (caller should
        stop processing this tick and return None), False otherwise."""
        if (
            profit_protection_candidate
            and self._profit_protection_price_reached(position, current_price)
        ):
            position["profit_protection_peak_price"] = current_price
            lock_price = self._profit_protection_trailing_floor(position, current_price)

            if lock_price is not None:
                position["profit_protection_applied"] = True
                position["profit_protection_profit_locked"] = True
                position["profit_protection_target"] = "tp1_price"
                self._promote_to_breakeven(
                    position,
                    reason="Profit protection (ROI-of-TP1 trailing arm)",
                    target_price=lock_price,
                )
                return True

        if (
            early_breakeven_candidate
            and self._early_breakeven_price_reached(position, current_price)
        ):
            lock_price = self._early_breakeven_lock_price(position, candles)
            position["early_breakeven_applied"] = True
            position["early_breakeven_profit_locked"] = _more_favorable(
                position["side"], lock_price, position["breakeven_price"]
            )
            self._promote_to_breakeven(
                position,
                reason="Early breakeven (profit-lock)",
                target_price=lock_price,
            )
            return True

        return False

    def poll_live(
        self, symbol, candles=None, htf_candles=None, cvd_snapshot=None, crash_snapshot=None,
    ):
        """Returns an outcome string if the position closed this call,
        otherwise None. `candles` (LTF history for the symbol) is only
        used by config.STRUCTURE_STOP_MANAGEMENT_ENABLED's structure-aware
        early-breakeven lock and post-TP1 trailing stop - both are no-ops
        when it's not supplied. `htf_candles`/`cvd_snapshot` are only used
        by config.DCA_BREAKEVEN_CONFIRMATION_ENABLED (see
        _dca_breakeven_confirmation) and config.DCA_PRESSURE_CHECK_ENABLED
        (see _execute_dca) - also a no-op without them, same convention.
        `crash_snapshot` - config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED,
        also only consumed by _execute_dca, same no-op-without-it shape."""
        position = self.positions.get(symbol)

        if not position or position["shadow"]:
            return None

        # config.TP_STATIC_ROI_ENABLED - the only case where this can now
        # return a real outcome: a single-TP DCA_PENDING position whose
        # missing TP order can't be placed because price already passed
        # it, closed at market immediately (see _market_close_static_tp).
        # Every other path through _ensure_protection_orders still
        # implicitly returns None, same as before this existed.
        protection_outcome = self._ensure_protection_orders(position)

        if protection_outcome is not None:
            return protection_outcome

        # One shared mark-price fetch, reused for MAE/MFE tracking and
        # both early-promotion checks below - no reason to pay for extra
        # REST calls when any of them need the same current price.
        early_breakeven_candidate = (
            position["stage"] in (TP1_PENDING, DCA_PENDING)
            and self._is_early_breakeven_candidate(position)
        )
        profit_protection_candidate = (
            position["stage"] in (TP1_PENDING, DCA_PENDING)
            and self._is_profit_protection_candidate(position)
        )
        dca_candidate = self._is_dca_candidate(position)
        dca_profit_protection_candidate = self._is_dca_profit_protection_candidate(position)
        dca_breakeven_candidate = self._is_dca_breakeven_candidate(position)
        current_price = None

        if (
            config.MAE_TRACKING_ENABLED or early_breakeven_candidate
            or profit_protection_candidate or dca_candidate or dca_profit_protection_candidate
            or dca_breakeven_candidate
        ):
            current_price = exchange.get_mark_price(symbol)
            self._update_mae_mfe(position, current_price)

        if position["stage"] == DCA_PENDING:
            # DCA (adverse) checked before the early-promotion (favorable)
            # checks - same conservative "adverse event wins ties" bias
            # poll_shadow's own docstring already applies.
            #
            # config.DCA_RESTING_ORDER_ENABLED - a position with a real
            # resting DCA order polls that order's own status
            # (_poll_dca_resting_order) instead of watching candle ranges
            # for the trigger; a position without one (flag off, or
            # registered before it was turned on) falls back to the
            # existing candle-range check against the current candle's
            # full high/low range (not current_price, a single point-in-
            # time sample) - see _dca_price_reached_in_range's own
            # docstring for the real gap that closes.
            if dca_candidate:
                # config.DCA_PROTECTIVE_FIRST_ENABLED - a protective-first
                # position has a real resting stop, not a resting add-in
                # order, so it's checked first and separately: poll the
                # stop's own status, and only if config.
                # DCA_PROTECTIVE_ESCALATION_ENABLED is also on AND price
                # has actually reached dca_price, run the real-time
                # pressure check to decide whether to escalate. See
                # config.DCA_PROTECTIVE_FIRST_ENABLED's own comment for
                # the full rationale.
                if config.DCA_PROTECTIVE_FIRST_ENABLED and position.get("dca_protective_sl_order_id"):
                    fired, outcome = self._poll_dca_protective_stop(position)
                    if fired:
                        return outcome
                    if (
                        config.DCA_PROTECTIVE_ESCALATION_ENABLED
                        and self._dca_price_reached_in_range(position, candles)
                    ):
                        outcome = self._try_dca_protective_escalation(
                            position, candles=candles, htf_candles=htf_candles,
                            cvd_snapshot=cvd_snapshot, current_price=current_price,
                            crash_snapshot=crash_snapshot,
                        )
                        if outcome is not None or position.get("dca_applied"):
                            return outcome
                elif config.DCA_RESTING_ORDER_ENABLED and position.get("dca_order_id"):
                    fired, outcome = self._poll_dca_resting_order(
                        position, candles=candles, htf_candles=htf_candles,
                        cvd_snapshot=cvd_snapshot, current_price=current_price,
                        crash_snapshot=crash_snapshot,
                    )
                    if fired:
                        return outcome
                elif self._dca_price_reached_in_range(position, candles):
                    return self._execute_dca(
                        position, candles=candles, htf_candles=htf_candles,
                        cvd_snapshot=cvd_snapshot, current_price=current_price,
                        crash_snapshot=crash_snapshot,
                    )

            if self._try_early_promotions(
                position, current_price, candles, profit_protection_candidate, early_breakeven_candidate,
            ):
                return None

            # config.TP_STATIC_ROI_ENABLED - one full-position TP instead
            # of TP1(partial)+TP2(remainder): a fill closes the WHOLE
            # position at once, no promotion, no tp2 to wait on.
            if position.get("single_tp"):
                tp_status = self._status_or_missing(symbol, position["tp_order_id"])

                if tp_status == "FINISHED":
                    exchange.cancel_all_open_orders(symbol)
                    return self._close(symbol, "STATIC_TP_HIT")

                return None

            tp1_status = self._status_or_missing(symbol, position["tp1_order_id"])

            if tp1_status == "FINISHED":
                return self._promote_to_breakeven_on_tp1_fill(position, candles)

            tp2_status = self._status_or_missing(symbol, position["tp2_order_id"])

            if tp2_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "TP2_HIT_DIRECT")

            return None

        if position["stage"] == DCA_ACTIVE:
            # config.DCA_TP_STATIC_ROI_ENABLED - self-heals an ALREADY-open
            # DCA_ACTIVE position onto the current config's target instead
            # of running out whatever was computed once at DCA-fire time
            # forever (real operator need, 2026-08-19: flipping this flag,
            # or editing DCA_TP_TARGET_ROI_PCT, should apply to positions
            # already in flight too). Checked first, before profit
            # protection below, so target_price=position["tp_price"]
            # everywhere else in this branch already reflects the
            # migrated value this same tick.
            outcome = self._migrate_dca_target_if_needed(position)

            if outcome is not None:
                return outcome

            # config.DCA_MAX_ADVERSE_R_ENABLED - checked before the
            # favorable-side promotions below, same "adverse event wins
            # ties" bias this function's own DCA_PENDING branch already
            # documents for the DCA-fire-vs-early-promotion race.
            if (
                config.DCA_MAX_ADVERSE_R_ENABLED
                and current_price is not None
                and self._dca_max_adverse_loss_reached(position, current_price)
            ):
                outcome = self._close_dca_active_on_max_adverse_loss(position)

                if outcome is not None:
                    return outcome

            # config.PROFIT_PROTECTION_ENABLED - a fresh arm step, unlike
            # BREAKEVEN_ACTIVE below (which only ever TRAILS an already-
            # armed lock carried over from TP1_PENDING) - see
            # _is_dca_profit_protection_candidate's docstring for why
            # DCA_ACTIVE always starts unarmed.
            if (
                dca_profit_protection_candidate
                and self._profit_protection_price_reached(
                    position, current_price, target_price=position["tp_price"]
                )
            ):
                position["profit_protection_peak_price"] = current_price
                lock_price = self._profit_protection_trailing_floor(
                    position, current_price, target_price=position["tp_price"]
                )

                if lock_price is not None:
                    outcome, replaced = self._replace_sl_order(
                        position, lock_price, "Profit protection (ROI-of-TP arm, post-DCA)",
                    )

                    if outcome is not None:
                        return outcome

                    if replaced:
                        position["profit_protection_applied"] = True
                        position["profit_protection_profit_locked"] = True
                        position["profit_protection_target"] = "tp_price"

                    return None

            # config.DCA_BREAKEVEN_ENABLED - same "fresh arm" shape as
            # profit protection above, checked second (mirrors
            # _try_early_promotions' profit-protection-then-early-
            # breakeven ordering): if this tick's move already satisfied
            # profit protection's much deeper threshold, that arm already
            # fired and returned above - this only fires on its own for a
            # smaller recovery that reaches breakeven but not that far.
            if (
                dca_breakeven_candidate
                and self._dca_breakeven_price_reached_in_range(position, candles)
            ):
                withhold, confirmed, detail = self._dca_breakeven_confirmation(
                    position, htf_candles, candles, cvd_snapshot, current_price
                )
                position["dca_breakeven_direction_confirmed"] = confirmed

                if withhold:
                    log_info(
                        f"{symbol} DCA breakeven withheld - direction still confirmed | {detail}"
                    )
                    return None

                outcome, replaced = self._replace_sl_order(
                    position, position["breakeven_price"], "DCA breakeven (price reached)",
                )

                if outcome is not None:
                    return outcome

                if replaced:
                    position["dca_breakeven_applied"] = True

                return None

            sl_status = self._status_or_missing(symbol, position["sl_order_id"])

            if sl_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_SL_HIT")

            tp_status = self._status_or_missing(symbol, position.get("tp_order_id"))

            if tp_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_TP_HIT")

            # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - the dormant/
            # active native trailing stop itself fired. _status_or_missing
            # already returns "MISSING" for a None/falsy id, so this is a
            # safe no-op for every position without a real trail order -
            # no extra guard needed. Only ever activates at or past
            # breakeven and then only tightens further (see place_
            # trailing_stop_loss), so a close via this order is always
            # at-or-better-than breakeven by construction - see journal_
            # analysis.py's WIN_OUTCOMES for why this is win-shaped.
            trail_status = self._status_or_missing(symbol, position.get("dca_trail_order_id"))

            if trail_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_BREAKEVEN_TRAIL_HIT")

            if position.get("profit_protection_applied"):
                # Extra mark-price fetch, same "only pay for it when a
                # position could actually use it" discipline as
                # BREAKEVEN_ACTIVE below - only positions that armed
                # profit protection need this.
                outcome = self._trail_profit_protection_if_improved(
                    position, exchange.get_mark_price(symbol)
                )

                if outcome is not None:
                    return outcome

            # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - _structure_stop_
            # candidate and _trail_stop_if_improved are both already
            # generic over position["entry_price"]/["side"]/["sl_price"]/
            # ["breakeven_price"] - no DCA-specific logic needed here, as
            # long as _execute_dca kept those fields current (it does).
            # Same ratchet-only trail BREAKEVEN_ACTIVE already uses below.
            return self._trail_stop_if_improved(position, candles)

        if position["stage"] == TP1_PENDING:
            if self._try_early_promotions(
                position, current_price, candles, profit_protection_candidate, early_breakeven_candidate,
            ):
                return None

            tp1_status = self._status_or_missing(symbol, position["tp1_order_id"])

            if tp1_status == "FINISHED":
                return self._promote_to_breakeven_on_tp1_fill(position, candles)

            sl_status = self._status_or_missing(symbol, position["sl_order_id"])

            if sl_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "SL_HIT")

            tp2_status = self._status_or_missing(symbol, position["tp2_order_id"])

            if tp2_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "TP2_HIT_DIRECT")

            return None

        if position["stage"] == BREAKEVEN_ACTIVE:
            sl_status = self._status_or_missing(symbol, position["sl_order_id"])

            if sl_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, self._breakeven_stop_outcome(position, shadow=False))

            tp2_status = self._status_or_missing(symbol, position["tp2_order_id"])

            if tp2_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "TP2_HIT")

            # config.PROFIT_PROTECTION_TP2_LEG_ENABLED - a genuine TP1 fill
            # leaves profit_protection_applied False for the rest of this
            # leg's life (see _is_tp2_profit_protection_candidate) - this
            # is its one-time fresh arm, mirroring _try_early_promotions'
            # own "stop processing this tick once armed" convention (the
            # next poll's trail-continuation branch below picks it up from
            # here, same cadence as every other one-time arm in this
            # class).
            if self._is_tp2_profit_protection_candidate(position):
                current_price = exchange.get_mark_price(symbol)

                if self._profit_protection_price_reached(
                    position, current_price, target_price=position["tp2_price"]
                ):
                    lock_price = self._profit_protection_trailing_floor(
                        position, current_price, target_price=position["tp2_price"]
                    )

                    if lock_price is not None:
                        outcome, replaced = self._replace_sl_order(
                            position, lock_price, "Profit protection (TP2 leg)",
                        )

                        if outcome is not None:
                            return outcome

                        if replaced:
                            position["profit_protection_applied"] = True
                            position["profit_protection_profit_locked"] = True
                            position["profit_protection_peak_price"] = current_price
                            position["profit_protection_target"] = "tp2_price"

                        return None

            if position.get("profit_protection_applied"):
                # Extra mark-price fetch, same "only pay for it when a
                # position could actually use it" discipline as the
                # TP1_PENDING branch above - only positions that armed
                # profit protection need this.
                outcome = self._trail_profit_protection_if_improved(
                    position, exchange.get_mark_price(symbol)
                )

                if outcome is not None:
                    return outcome

            return self._trail_stop_if_improved(position, candles)

        return None

    def _trail_stop_if_improved(self, position, candles):
        """config.STRUCTURE_STOP_MANAGEMENT_ENABLED - ratchet-only:
        replaces the SL with the nearest confirmed structure swing only
        when it's strictly MORE favorable than the current sl_price,
        never loosens. Never transitions stage (stays BREAKEVEN_ACTIVE -
        only _promote_to_breakeven owns stage transitions) and never
        touches risk_distance (see _mae_mfe_r_multiples's invariant that
        it's fixed once at entry). Only called after this poll's own
        sl_status/tp2_status checks already confirmed neither fired -
        _replace_sl_order's internal ground-truth check is a rare true-
        race backstop, not the primary detection path. Returns a close-
        outcome string only if the replace attempt forced a market-close
        of the remainder (-2021), otherwise None."""
        if not config.STRUCTURE_STOP_MANAGEMENT_ENABLED or not candles:
            return None

        candidate = _structure_stop_candidate(position, candles)

        if candidate is None:
            return None

        side = position["side"]

        if not _more_favorable(side, candidate, position["sl_price"]):
            return None

        outcome, replaced = self._replace_sl_order(
            position, candidate, "Structure trailing stop", close_if_not_open=False
        )

        if replaced:
            position["trailing_stop_locked_profit"] = _more_favorable(
                side, candidate, position["breakeven_price"]
            )

        return outcome

    @staticmethod
    def _breakeven_stop_outcome(position, shadow):
        """4-way precedence for a BREAKEVEN_ACTIVE SL hit: a trailed
        profit takes priority over either early-promotion lock (profit_
        protection and early_breakeven are mutually exclusive with each
        other - only one of those two flags can ever be true, since only
        one promotion path can have actually fired for a given position -
        so their relative order below doesn't matter), which both take
        priority over a flat breakeven scratch - all land at the same
        code path (the SL order firing) and are told apart only by these
        flags."""
        prefix = "SHADOW_" if shadow else ""

        if position.get("trailing_stop_locked_profit"):
            return f"{prefix}TRAILING_STOP_PROFIT_HIT"

        if position.get("profit_protection_profit_locked"):
            return f"{prefix}PROFIT_PROTECTION_HIT"

        if position.get("early_breakeven_profit_locked"):
            return f"{prefix}EARLY_BREAKEVEN_PROFIT_HIT"

        return f"{prefix}BREAKEVEN_STOP_HIT"

    @staticmethod
    def _status_or_missing(symbol, order_id):
        """Never calls Binance with a blank id - that's a guaranteed
        -1102 on every single poll forever, for a leg that isn't even
        placed yet (or failed to place, see _ensure_protection_orders)."""
        if not order_id:
            return "MISSING"

        return exchange.get_algo_order_status(symbol, order_id)

    def _ensure_protection_orders(self, position):
        """SL is guaranteed atomic at entry - execution.py aborts the
        whole trade if it can't be placed. TP1/TP2 are best-effort there
        (a placement failure doesn't abort an otherwise-safe, SL-protected
        trade), which means either can legitimately be missing. Retry
        placing whichever is missing instead of leaving that leg
        permanently degraded (no profit-taking on it) with no way to
        recover."""
        symbol = position["symbol"]
        side = position["side"]

        if position["stage"] == DCA_ACTIVE:
            # Single-TP lifecycle post-DCA - tp2_order_id/tp1_order_id are
            # stale leftovers from before the DCA fired, not the real
            # order to self-heal here. SL is never self-healed anywhere
            # in this file (see this function's own docstring) - a
            # post-DCA SL placement failure is handled synchronously by
            # _execute_dca itself (close at market), same discipline as
            # every other "SL must be atomic" path.
            if not position.get("tp_order_id"):
                existing = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=True)

                if existing:
                    position["tp_order_id"] = exchange._accepted_order_id(existing)
                    log_info(f"{symbol} post-DCA TP order tracking re-synced from exchange")
                else:
                    try:
                        order = exchange.place_take_profit_full(symbol, side, position["tp_price"])
                        position["tp_order_id"] = exchange._accepted_order_id(order)

                        if position["tp_order_id"]:
                            log_info(f"{symbol} post-DCA TP order recovered")
                    except Exception as exc:
                        log_warning(f"{symbol} post-DCA TP recovery attempt failed: {exc}")

            return None

        if (
            position["stage"] == DCA_PENDING
            and config.DCA_PROTECTIVE_FIRST_ENABLED
            and not position.get("dca_applied")
        ):
            # config.DCA_PROTECTIVE_FIRST_ENABLED - genuine self-heal
            # (unlike execution._place_dca_resting_order, which has no
            # retry at all if its one placement attempt fails) plus the
            # migration path for a position that entered DCA_PENDING
            # before this flag was turned on: an old-style dca_order_id
            # add-in LIMIT order, if still resting, gets actively
            # cancelled here rather than left to fill unexpectedly once
            # protective-first is supposed to be in charge. Does not
            # return early - falls through into the ordinary TP self-heal
            # below exactly as today, single_tp or not.
            if position.get("dca_order_id"):
                try:
                    exchange.cancel_order(symbol, position["dca_order_id"])
                    log_info(f"{symbol} stale pre-migration DCA add-in order cancelled")
                except Exception as exc:
                    log_warning(f"{symbol} stale DCA add-in order cancel failed: {exc}")
                position["dca_order_id"] = None

            if not position.get("dca_protective_sl_order_id"):
                existing = self._find_open_order(symbol, "STOP_MARKET", close_position=True)

                if existing:
                    position["dca_protective_sl_order_id"] = exchange._accepted_order_id(existing)
                    log_info(f"{symbol} DCA protective stop tracking re-synced from exchange")
                else:
                    try:
                        order = exchange.place_stop_loss(
                            symbol, side, position["dca_price"],
                            client_algo_id=(
                                f"{execution.DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_PREFIX}"
                                f"{int(time.time() * 1000)}"
                            ),
                        )
                        position["dca_protective_sl_order_id"] = exchange._accepted_order_id(order)

                        if position["dca_protective_sl_order_id"]:
                            log_info(f"{symbol} DCA protective stop recovered")
                    except Exception as exc:
                        log_warning(f"{symbol} DCA protective stop recovery attempt failed: {exc}")

        if position["stage"] == DCA_PENDING and position.get("single_tp"):
            # config.TP_STATIC_ROI_ENABLED - same single-TP self-heal
            # shape as DCA_ACTIVE above, just for the pre-DCA stage. No
            # SL to self-heal here either (see this function's own
            # docstring) - a DCA_PENDING position's SL doesn't exist yet
            # by design, same as always.
            if not position.get("tp_order_id"):
                existing = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=True)

                if existing:
                    position["tp_order_id"] = exchange._accepted_order_id(existing)
                    log_info(f"{symbol} TP order tracking re-synced from exchange")
                else:
                    try:
                        order = exchange.place_take_profit_full(symbol, side, position["tp_price"])
                        position["tp_order_id"] = exchange._accepted_order_id(order)

                        if position["tp_order_id"]:
                            log_info(f"{symbol} TP order recovered")
                    except Exception as exc:
                        if "-2021" in str(exc):
                            # Price has already passed the (only) TP level
                            # entirely - unlike TP1's -2021 fallback below
                            # (partial close + promote), there's nothing
                            # left to promote once the WHOLE position
                            # closes, so this ends the trade outright.
                            log_warning(
                                f"{symbol} TP level already passed by price - "
                                "closing at market instead"
                            )
                            return self._market_close_static_tp(position)

                        log_warning(f"{symbol} TP recovery attempt failed: {exc}")

            return None

        if position["stage"] in (TP1_PENDING, DCA_PENDING) and not position["tp1_order_id"]:
            # Check the exchange for a real TP1-shaped order before
            # attempting to place one - if local tracking merely lost the
            # id (a reconciliation mismatch, for instance) while the real
            # order is still there, placing another one gets rejected
            # with -4130 ("already existing") on every single poll
            # forever. Re-sync from the real order instead of duplicating.
            existing = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=False)

            if existing:
                position["tp1_order_id"] = exchange._accepted_order_id(existing)
                log_info(f"{symbol} TP1 order tracking re-synced from exchange")
            else:
                try:
                    order = exchange.place_take_profit_partial(
                        symbol, side, position["tp1_quantity"], position["tp1_price"]
                    )
                    position["tp1_order_id"] = exchange._accepted_order_id(order)

                    if position["tp1_order_id"]:
                        log_info(f"{symbol} TP1 order recovered")
                except Exception as exc:
                    if "-2021" in str(exc):
                        # Price has already passed the TP1 level entirely -
                        # a conditional order there would fire instantly,
                        # which is exactly what "take profit" means here.
                        # Take it at market instead of leaving TP1
                        # permanently unplaceable and retried forever.
                        log_warning(
                            f"{symbol} TP1 level already passed by price - "
                            "closing TP1 quantity at market instead"
                        )
                        self._market_close_tp1(position)
                    else:
                        log_warning(f"{symbol} TP1 recovery attempt failed: {exc}")

        if not position["tp2_order_id"]:
            existing = self._find_open_order(symbol, "TAKE_PROFIT_MARKET", close_position=True)

            if existing:
                position["tp2_order_id"] = exchange._accepted_order_id(existing)
                log_info(f"{symbol} TP2 order tracking re-synced from exchange")
            else:
                try:
                    order = exchange.place_take_profit_full(
                        symbol, side, position["tp2_price"]
                    )
                    position["tp2_order_id"] = exchange._accepted_order_id(order)

                    if position["tp2_order_id"]:
                        log_info(f"{symbol} TP2 order recovered")
                except Exception as exc:
                    log_warning(f"{symbol} TP2 recovery attempt failed: {exc}")

    @staticmethod
    def _find_open_order(symbol, order_type, close_position):
        """Ground truth from the exchange: is there already a matching
        order sitting there, regardless of what local tracking thinks?
        Used before both placing a "missing" order (self-heal without
        creating a duplicate) and before cancelling a "known" order
        (cancel the real one, not a possibly-stale local id)."""
        for order in exchange.get_open_algo_orders(symbol):
            if _order_type(order) != order_type:
                continue

            is_close_position = str(order.get("closePosition")).lower() == "true"

            if is_close_position == close_position:
                return order

        return None

    def _market_close_tp1(self, position):
        """TP1's price was already passed by the market before the order
        could be placed - close that quantity at market (the position is
        still SL-protected throughout) and promote the remainder to
        breakeven, the same outcome a genuine TP1 fill would have led to."""
        symbol = position["symbol"]
        side = position["side"]

        try:
            exchange.close_position_market(symbol, side, position["tp1_quantity"])
        except Exception as exc:
            log_error(f"{symbol} TP1 market-close-instead error: {exc}")
            return

        log_info(f"{symbol} TP1 quantity closed at market (price already past TP1)")
        self._promote_to_breakeven_on_tp1_fill(position)

    def _market_close_static_tp(self, position):
        """config.TP_STATIC_ROI_ENABLED equivalent of _market_close_tp1,
        called only from _ensure_protection_orders's -2021 fallback for a
        single-TP DCA_PENDING position. Unlike TP1's partial close +
        promotion, this closes the WHOLE position at market and ends the
        trade outright as a real win (STATIC_TP_HIT) - there is nothing
        left to promote once a single, full-position TP accounts for
        everything. Returns the close outcome string so the caller
        (_ensure_protection_orders, and poll_live above it) can propagate
        it and stop processing this tick, or None if the market-close
        itself failed (left for the next poll to retry)."""
        symbol = position["symbol"]
        side = position["side"]

        try:
            exchange.close_position_market(symbol, side, position["quantity"])
        except Exception as exc:
            log_error(f"{symbol} static TP market-close-instead error: {exc}")
            return None

        exchange.cancel_all_open_orders(symbol)
        log_info(f"{symbol} static TP quantity closed at market (price already past target)")
        return self._close(symbol, "STATIC_TP_HIT")

    @staticmethod
    def _dca_max_adverse_loss_reached(position, current_price):
        """config.DCA_MAX_ADVERSE_R_ENABLED - see its own config.py
        comment for the real evidence. Compares real unrealized DOLLARS
        lost right now (current blended entry/quantity vs current_price)
        against original_risk_distance * original_quantity - the actual
        dollar amount the ORIGINAL, single-entry plan would have risked.

        Deliberately NOT a price-distance-only ratio (adverse price move
        / original_risk_distance) - real evidence (2026-08-31, BANKUSDT)
        showed that understates real dollar risk once DCA has grown the
        position: the same price move past the DCA fill now applies to
        MORE quantity than the original 1R was ever sized for, so price-
        distance-R and dollar-R diverge, sometimes by a full R or more.

        None/missing original_risk_distance or original_quantity (e.g. a
        restart-recovered DCA_ACTIVE position, which has no way to know
        its pre-DCA risk/size - see _recover_dca_active_position) never
        triggers - fail-open, same convention as every gate in this
        codebase."""
        original_risk = position.get("original_risk_distance")
        original_quantity = position.get("original_quantity")

        if not original_risk or original_risk <= 0 or not original_quantity or original_quantity <= 0:
            return False

        original_dollar_risk = original_risk * original_quantity
        side = position["side"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        unrealized_loss = (
            (entry_price - current_price) * quantity if side == "BUY"
            else (current_price - entry_price) * quantity
        )
        return (unrealized_loss / original_dollar_risk) >= max(float(config.DCA_MAX_ADVERSE_R_MULTIPLE), 0)

    def _close_dca_active_on_max_adverse_loss(self, position):
        """Same market-close-then-_close shape as _market_close_static_tp
        - the position's unrealized loss has reached config.
        DCA_MAX_ADVERSE_R_MULTIPLE times its original planned risk (see
        _dca_max_adverse_loss_reached), so it's closed now rather than
        left to run to the structural post-DCA SL, which is deliberately
        several original-R further away. Returns None (left for the next
        poll to retry) if the market-close itself failed."""
        symbol = position["symbol"]
        side = position["side"]

        try:
            exchange.close_position_market(symbol, side, position["quantity"])
        except Exception as exc:
            log_error(f"{symbol} max-adverse-loss market-close error: {exc}")
            return None

        exchange.cancel_all_open_orders(symbol)
        log_info(
            f"{symbol} closed at market - unrealized loss reached "
            f"{config.DCA_MAX_ADVERSE_R_MULTIPLE}R of original risk, not "
            f"waiting for the structural post-DCA SL"
        )
        return self._close(symbol, "DCA_MAX_ADVERSE_LOSS")

    def _try_early_promotions_shadow(self, position, latest_candle, candles):
        """Shadow-mode counterpart to _try_early_promotions - shared by
        poll_shadow's TP1_PENDING and DCA_PENDING branches (see
        _is_early_breakeven_candidate/_is_profit_protection_candidate,
        both of which now accept either stage - config.DCA_ENABLED gap
        found live, 2026-08-17). Returns True if a promotion happened
        this call."""
        symbol = position["symbol"]
        side = position["side"]

        if self._is_profit_protection_candidate(position) and self._profit_protection_price_reached(
            position, latest_candle["close"]
        ):
            arm_price = latest_candle["close"]
            position["profit_protection_peak_price"] = arm_price
            lock_price = self._profit_protection_trailing_floor(position, arm_price)

            if lock_price is not None:
                position["profit_protection_applied"] = True
                position["profit_protection_profit_locked"] = True
                position["profit_protection_target"] = "tp1_price"
                position["stage"] = BREAKEVEN_ACTIVE
                position["sl_price"] = lock_price
                log_info(
                    f"{symbol} [SHADOW] profit protection (ROI-of-TP1 trailing arm) | "
                    f"SL -> {position['sl_price']}"
                )
                return True

        if self._is_early_breakeven_candidate(position) and self._early_breakeven_price_reached(
            position, latest_candle["close"]
        ):
            lock_price = self._early_breakeven_lock_price(position, candles)
            position["early_breakeven_applied"] = True
            position["early_breakeven_profit_locked"] = _more_favorable(
                side, lock_price, position["breakeven_price"]
            )
            position["stage"] = BREAKEVEN_ACTIVE
            position["sl_price"] = lock_price
            log_info(
                f"{symbol} [SHADOW] early breakeven (profit-lock) | "
                f"SL -> {position['sl_price']}"
            )
            return True

        return False

    def poll_shadow(
        self, symbol, latest_candle, candles=None, htf_candles=None, cvd_snapshot=None,
        crash_snapshot=None,
    ):
        """Simulates the same TP1 -> breakeven -> TP2/SL sequence against
        live price action. When both the stop and a target fall inside the
        same candle's range, the SL side is assumed to have been touched
        first - a deliberately conservative simplification so shadow stats
        don't overstate win rate; it is not a substitute for real fills.
        `candles` (LTF history) is only used by config.STRUCTURE_STOP_MANAGEMENT_ENABLED's
        structure-aware early-breakeven lock and post-TP1 trailing - both
        are no-ops when it's not supplied. `htf_candles`/`cvd_snapshot`/
        `crash_snapshot` - see poll_live's own docstring, same
        DCA_BREAKEVEN_CONFIRMATION_ENABLED/DCA_PRESSURE_CHECK_ENABLED/
        CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED no-op convention."""
        position = self.positions.get(symbol)

        if not position or not position["shadow"] or not latest_candle:
            return None

        high = latest_candle["high"]
        low = latest_candle["low"]
        side = position["side"]

        self._update_mae_mfe(position, low, high)

        if position["stage"] == TP1_PENDING:
            hit_sl = (
                low <= position["sl_price"]
                if side == "BUY"
                else high >= position["sl_price"]
            )

            if hit_sl:
                return self._close(symbol, "SHADOW_SL_HIT")

            if self._try_early_promotions_shadow(position, latest_candle, candles):
                return None

            hit_tp1 = (
                high >= position["tp1_price"]
                if side == "BUY"
                else low <= position["tp1_price"]
            )

            if hit_tp1:
                lock_price = self._early_breakeven_lock_price(position, candles)
                position["early_breakeven_profit_locked"] = _more_favorable(
                    side, lock_price, position["breakeven_price"]
                )
                position["stage"] = BREAKEVEN_ACTIVE
                position["sl_price"] = lock_price
                log_info(f"{symbol} [SHADOW] TP1 would have filled | SL -> {lock_price}")

            return None

        if position["stage"] == DCA_PENDING:
            # DCA (adverse) checked before TP1 (favorable) - same
            # deliberately conservative "assume the worse-for-the-trade
            # touch happened first" bias this function's own docstring
            # already applies to SL-vs-target ambiguity within one candle.
            if self._is_dca_candidate(position):
                dca_price = position["dca_price"]
                touched_dca = low <= dca_price if side == "BUY" else high >= dca_price

                if touched_dca:
                    # config.DCA_PROTECTIVE_FIRST_ENABLED - shadow has no
                    # real resting orders, so there's nothing to poll; the
                    # touch itself IS the protective-stop-fires moment,
                    # unless escalation is on and the real-time pressure
                    # check reads confirmed. Same "adverse-touch wins the
                    # within-candle ambiguity" bias this function's own
                    # docstring already states, applied consistently here.
                    if config.DCA_PROTECTIVE_FIRST_ENABLED:
                        if config.DCA_PROTECTIVE_ESCALATION_ENABLED:
                            confirmed, detail = self._dca_pressure_check(
                                position, htf_candles, candles, cvd_snapshot,
                                latest_candle["close"], crash_snapshot,
                            )
                            position["dca_pressure_confirmed"] = confirmed

                            if confirmed:
                                return self._execute_dca(
                                    position, candles=candles, htf_candles=htf_candles,
                                    cvd_snapshot=cvd_snapshot, current_price=latest_candle["close"],
                                    crash_snapshot=crash_snapshot, pressure_confirmed_override=True,
                                )

                        position["dca_protective_stop_hit"] = True
                        return self._close(symbol, "SHADOW_DCA_PROTECTIVE_SL_HIT")

                    return self._execute_dca(
                        position, candles=candles, htf_candles=htf_candles,
                        cvd_snapshot=cvd_snapshot, current_price=latest_candle["close"],
                        crash_snapshot=crash_snapshot,
                    )

            if self._try_early_promotions_shadow(position, latest_candle, candles):
                return None

            # config.TP_STATIC_ROI_ENABLED - one full-position TP instead
            # of TP1(partial)+TP2(remainder): a touch closes the WHOLE
            # position at once, no promotion.
            if position.get("single_tp"):
                hit_tp = (
                    high >= position["tp_price"]
                    if side == "BUY"
                    else low <= position["tp_price"]
                )

                if hit_tp:
                    return self._close(symbol, "SHADOW_STATIC_TP_HIT")

                return None

            hit_tp1 = (
                high >= position["tp1_price"]
                if side == "BUY"
                else low <= position["tp1_price"]
            )

            if hit_tp1:
                lock_price = self._early_breakeven_lock_price(position, candles)
                position["early_breakeven_profit_locked"] = _more_favorable(
                    side, lock_price, position["breakeven_price"]
                )
                position["stage"] = BREAKEVEN_ACTIVE
                position["sl_price"] = lock_price
                log_info(f"{symbol} [SHADOW] TP1 would have filled | SL -> {lock_price}")

            return None

        if position["stage"] == DCA_ACTIVE:
            # config.DCA_TP_STATIC_ROI_ENABLED - see poll_live's identical
            # migration check for the full reasoning. Checked first here
            # too, so hit_tp below already reads the migrated tp_price
            # this same tick.
            migrate_outcome = self._migrate_dca_target_if_needed(
                position, current_price=latest_candle["close"]
            )

            if migrate_outcome is not None:
                return migrate_outcome

            # config.DCA_MAX_ADVERSE_R_ENABLED - see poll_live's identical
            # check for the full reasoning/evidence. Checked before hit_sl
            # below, same "adverse event wins ties" bias.
            if config.DCA_MAX_ADVERSE_R_ENABLED and self._dca_max_adverse_loss_reached(
                position, latest_candle["close"]
            ):
                return self._close(symbol, "SHADOW_DCA_MAX_ADVERSE_LOSS")

            hit_sl = low <= position["sl_price"] if side == "BUY" else high >= position["sl_price"]

            if hit_sl:
                return self._close(symbol, "SHADOW_DCA_SL_HIT")

            hit_tp = high >= position["tp_price"] if side == "BUY" else low <= position["tp_price"]

            if hit_tp:
                return self._close(symbol, "SHADOW_DCA_TP_HIT")

            # config.PROFIT_PROTECTION_ENABLED - fresh arm step (see
            # _is_dca_profit_protection_candidate's docstring for why
            # DCA_ACTIVE always starts unarmed, unlike BREAKEVEN_ACTIVE
            # below which only ever trails an already-armed lock).
            if self._is_dca_profit_protection_candidate(position) and self._profit_protection_price_reached(
                position, latest_candle["close"], target_price=position["tp_price"]
            ):
                arm_price = latest_candle["close"]
                position["profit_protection_peak_price"] = arm_price
                lock_price = self._profit_protection_trailing_floor(
                    position, arm_price, target_price=position["tp_price"]
                )

                if lock_price is not None:
                    position["profit_protection_applied"] = True
                    position["profit_protection_profit_locked"] = True
                    position["profit_protection_target"] = "tp_price"
                    position["sl_price"] = lock_price
                    log_info(
                        f"{symbol} [SHADOW] profit protection (ROI-of-TP arm, post-DCA) | "
                        f"SL -> {position['sl_price']}"
                    )
                    return None

            # config.DCA_BREAKEVEN_ENABLED - same "fresh arm" shape as
            # profit protection above, checked second: if this candle's
            # move already satisfied profit protection's much deeper
            # threshold, that arm already fired and returned above - this
            # only fires on its own for a smaller recovery.
            # Uses the high/low already extracted from latest_candle above
            # (same as this function's own DCA-entry touch check just
            # above) rather than _dca_breakeven_price_reached_in_range's
            # `candles` param - candles is a separate, optional parameter
            # here (only used by config.STRUCTURE_STOP_MANAGEMENT_ENABLED
            # elsewhere), not guaranteed to be populated/in sync with
            # latest_candle the way it is in poll_live.
            touched_breakeven = (
                high >= position["breakeven_price"] if side == "BUY"
                else low <= position["breakeven_price"]
            )

            if self._is_dca_breakeven_candidate(position) and touched_breakeven:
                withhold, confirmed, detail = self._dca_breakeven_confirmation(
                    position, htf_candles, candles, cvd_snapshot, latest_candle["close"]
                )
                position["dca_breakeven_direction_confirmed"] = confirmed

                if withhold:
                    log_info(
                        f"{symbol} [SHADOW] DCA breakeven withheld - direction still "
                        f"confirmed | {detail}"
                    )
                    return None

                position["dca_breakeven_applied"] = True
                position["sl_price"] = position["breakeven_price"]
                log_info(
                    f"{symbol} [SHADOW] DCA breakeven (price reached) | "
                    f"SL -> {position['sl_price']}"
                )
                return None

            if position.get("profit_protection_applied"):
                peak_source = high if side == "BUY" else low
                peak_price = position.get("profit_protection_peak_price")
                peak_price = peak_source if peak_price is None else (
                    max(peak_price, peak_source) if side == "BUY" else min(peak_price, peak_source)
                )
                position["profit_protection_peak_price"] = peak_price

                candidate = self._profit_protection_trailing_floor(
                    position, peak_price, target_price=position["tp_price"]
                )

                if candidate is not None and _more_favorable(side, candidate, position["sl_price"]):
                    position["sl_price"] = candidate
                    log_info(
                        f"{symbol} [SHADOW] profit protection trailing stop (post-DCA) | "
                        f"SL -> {candidate}"
                    )

            # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - _structure_stop_
            # candidate is already generic over position["entry_price"]/
            # ["side"] - _execute_dca keeps breakeven_price current too,
            # so trailing_stop_locked_profit below compares against the
            # post-DCA breakeven, not the stale pre-DCA one.
            if config.STRUCTURE_STOP_MANAGEMENT_ENABLED and candles:
                candidate = _structure_stop_candidate(position, candles)

                if candidate is not None and _more_favorable(side, candidate, position["sl_price"]):
                    position["sl_price"] = candidate
                    position["trailing_stop_locked_profit"] = _more_favorable(
                        side, candidate, position["breakeven_price"]
                    )
                    log_info(f"{symbol} [SHADOW] structure trailing stop (post-DCA) | SL -> {candidate}")

            return None

        if position["stage"] == BREAKEVEN_ACTIVE:
            hit_sl = (
                low <= position["sl_price"]
                if side == "BUY"
                else high >= position["sl_price"]
            )

            if hit_sl:
                return self._close(symbol, self._breakeven_stop_outcome(position, shadow=True))

            hit_tp2 = (
                high >= position["tp2_price"]
                if side == "BUY"
                else low <= position["tp2_price"]
            )

            if hit_tp2:
                return self._close(symbol, "SHADOW_TP2_HIT")

            # config.PROFIT_PROTECTION_TP2_LEG_ENABLED - shadow counterpart
            # to poll_live's own fresh-arm branch; see
            # _is_tp2_profit_protection_candidate. Uses latest_candle
            # close as the arm price, same convention _try_early_
            # promotions_shadow already uses for its own arm check.
            if self._is_tp2_profit_protection_candidate(position) and self._profit_protection_price_reached(
                position, latest_candle["close"], target_price=position["tp2_price"]
            ):
                arm_price = latest_candle["close"]
                lock_price = self._profit_protection_trailing_floor(
                    position, arm_price, target_price=position["tp2_price"]
                )

                if lock_price is not None:
                    position["profit_protection_applied"] = True
                    position["profit_protection_profit_locked"] = True
                    position["profit_protection_target"] = "tp2_price"
                    position["profit_protection_peak_price"] = arm_price
                    position["sl_price"] = lock_price
                    log_info(
                        f"{symbol} [SHADOW] profit protection (TP2 leg) | SL -> {lock_price}"
                    )

                    return None

            if position.get("profit_protection_applied"):
                peak_source = high if side == "BUY" else low
                peak_price = position.get("profit_protection_peak_price")
                peak_price = peak_source if peak_price is None else (
                    max(peak_price, peak_source) if side == "BUY" else min(peak_price, peak_source)
                )
                position["profit_protection_peak_price"] = peak_price

                # config.PROFIT_PROTECTION_TP2_LEG_ENABLED - see
                # _trail_profit_protection_if_improved's identical
                # target_field lookup (poll_live's own copy of this same
                # concern) for why this can no longer assume tp1_price.
                target_field = position.get("profit_protection_target", "tp1_price")
                candidate = self._profit_protection_trailing_floor(
                    position, peak_price, target_price=position[target_field]
                )

                if candidate is not None and _more_favorable(side, candidate, position["sl_price"]):
                    position["sl_price"] = candidate
                    log_info(
                        f"{symbol} [SHADOW] profit protection trailing stop | SL -> {candidate}"
                    )

            if config.STRUCTURE_STOP_MANAGEMENT_ENABLED and candles:
                candidate = _structure_stop_candidate(position, candles)

                if candidate is not None and _more_favorable(side, candidate, position["sl_price"]):
                    position["sl_price"] = candidate
                    position["trailing_stop_locked_profit"] = _more_favorable(
                        side, candidate, position["breakeven_price"]
                    )
                    log_info(f"{symbol} [SHADOW] structure trailing stop | SL -> {candidate}")

        return None

    # =========================
    # PENDING LIMIT ENTRY - config.LIMIT_ENTRY_MODE_ENABLED. A resting
    # GTC LIMIT order can fill (fully or partially) at any arbitrary
    # moment while unattended, unlike a market order's synchronous fill -
    # so unlike enter_trade()/register(), there is no single call where
    # "place entry, then immediately place SL" can happen atomically.
    # Protection is instead placed the moment a fill is FIRST detected on
    # poll (see _apply_pending_fill), keeping the same "never leave a real
    # filled quantity unprotected" invariant the rest of this file already
    # enforces, just detected asynchronously instead of synchronously.
    # =========================
    @staticmethod
    def _pending_entry_invalidated(position, latest_candle):
        """Cheap, no-extra-REST invalidation check for a still-unfilled
        (or partially-filled) resting limit: has price already reached the
        level where the eventual stop would sit, before the limit ever
        finished filling? If so the setup that justified this entry has
        already failed - don't keep it resting waiting to fill into a
        broken trade. Uses the candle's full range (not just its close),
        symmetric with poll_shadow's existing SL-touch check, so an
        intrabar wick isn't missed. `latest_candle` comes from the free
        websocket feed - no extra REST call."""
        if not latest_candle:
            return False

        side = position["side"]
        sl_price = position["sl_price"]

        if side == "BUY":
            return latest_candle["low"] <= sl_price

        return latest_candle["high"] >= sl_price

    def _settle_pending_entry_to_tp1_pending(self, position):
        """The resting limit is done filling (either genuinely FILLED, or
        cancelled with real quantity already on it) - place TP1 sized off
        whatever actually filled (not the originally planned quantity,
        which the real fill may differ from) and promote to the normal
        TP1_PENDING lifecycle that poll_live already drives."""
        symbol = position["symbol"]
        side = position["side"]
        filled_quantity = position["filled_quantity"]
        tp1_close_pct = min(max(float(config.TP1_CLOSE_PCT), 0), 100)
        tp1_quantity = round(filled_quantity * tp1_close_pct / 100, 8)
        tp2_quantity = round(filled_quantity - tp1_quantity, 8)
        position["quantity"] = filled_quantity
        position["tp1_quantity"] = tp1_quantity
        position["tp2_quantity"] = tp2_quantity

        if tp1_quantity > 0:
            try:
                tp1_order = exchange.place_take_profit_partial(
                    symbol, side, tp1_quantity, position["tp1_price"]
                )
                position["tp1_order_id"] = exchange._accepted_order_id(tp1_order)
            except Exception as exc:
                log_warning(
                    f"{symbol} TP1 placement failed after limit fill "
                    f"settled (SL is active): {exc}"
                )

        position["stage"] = TP1_PENDING
        log_info(
            f"{symbol} limit entry settled | qty={filled_quantity} "
            f"entry={position['entry_price']} sl={position['sl_price']} "
            f"tp1={position['tp1_price']} tp2={position['tp2_price']}"
        )

    def _apply_pending_fill(self, position, order, settle=False):
        """A resting limit's executed_qty grew since it was last checked.
        On the FIRST fill, place SL now (closePosition=true protects
        whatever quantity is actually open right now, and stays valid
        without resizing even if more of the resting remainder fills
        later) and TP2 now (same closePosition=true property) - mirrors
        execution.py's "SL is atomic, TP2/TP1 best-effort" discipline, just
        triggered by a poll detecting the fill instead of a synchronous
        return value. TP1 needs an exact quantity, so it's deferred until
        the fill is final (`settle=True`, or the order itself reports
        FILLED) rather than sized against a still-growing partial fill.
        Returns an outcome string only if the position had to be closed
        outright (SL placement failed); otherwise None, whether or not
        settlement happened this call."""
        symbol = position["symbol"]
        side = position["side"]
        first_fill = position["filled_quantity"] == 0
        position["filled_quantity"] = order["executed_qty"]

        if first_fill:
            entry_price = order["avg_price"] or position["entry_price"]
            position["entry_price"] = entry_price
            position["risk_distance"] = abs(entry_price - position["sl_price"])
            position["mae_price"] = entry_price
            position["mfe_price"] = entry_price

            try:
                sl_order = exchange.place_stop_loss(symbol, side, position["sl_price"])
                position["sl_order_id"] = exchange._accepted_order_id(sl_order)
            except Exception as exc:
                log_error(
                    f"{symbol} SL placement failed after limit entry "
                    f"filled - closing the filled quantity at market "
                    f"rather than leave it unprotected: {exc}"
                )
                try:
                    exchange.close_position_market(symbol, side, position["filled_quantity"])
                except Exception as close_exc:
                    log_error(
                        f"{symbol} CRITICAL: failed to close unprotected "
                        f"limit-fill position - manual intervention "
                        f"needed: {close_exc}"
                    )
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "LIMIT_FILL_SL_PLACEMENT_FAILED")

            try:
                tp2_order = exchange.place_take_profit_full(symbol, side, position["tp2_price"])
                position["tp2_order_id"] = exchange._accepted_order_id(tp2_order)
            except Exception as exc:
                log_warning(
                    f"{symbol} TP2 placement failed after limit fill (SL is active): {exc}"
                )

            log_info(
                f"{symbol} limit entry filling | filled={position['filled_quantity']} "
                f"avg_price={entry_price}"
            )

        if settle or order["status"] == "FILLED":
            self._settle_pending_entry_to_tp1_pending(position)

        return None

    def _drop_unfilled_pending_entry(self, position, invalidated, shadow=False):
        """Genuinely zero-fill cancel/expiry - nothing was ever protected,
        nothing to unwind on the exchange side beyond the cancel already
        issued by the caller. Invalidation gets the same cooldown
        treatment as a real SL hit (the level failed); expiry does not
        (nothing was wrong with the setup, price just didn't come back in
        time) - same distinction SYMBOL_REENTRY_COOLDOWN_SECONDS's
        docstring already draws for a real close."""
        symbol = position["symbol"]
        self.positions.pop(symbol, None)
        reason = "LIMIT_INVALIDATED_UNFILLED" if invalidated else "LIMIT_EXPIRED_UNFILLED"

        if invalidated:
            self._closed_at[symbol] = time.time()

        prefix = " [SHADOW]" if shadow else ""
        log_info(f"{symbol}{prefix} pending limit entry cancelled | {reason}")
        signal_journal.append_outcome(symbol, reason, position.get("trade_id"))
        return reason

    def poll_pending_entry(self, symbol, latest_candle):
        """Returns an outcome string if the pending entry resolved this
        call (closed outright on an SL-placement failure, or dropped
        unfilled on cancel), otherwise None - including when it silently
        settled into TP1_PENDING, which is not itself a closed-trade
        outcome. Only meaningful for stage == PENDING_LIMIT_FILL,
        non-shadow positions - see main.py's _poll_positions dispatch."""
        position = self.positions.get(symbol)

        if not position or position["shadow"] or position["stage"] != PENDING_LIMIT_FILL:
            return None

        order = exchange.get_order_status(symbol, position["limit_order_id"])

        if order["status"] == "UNKNOWN":
            return None  # transient fetch failure - retry next poll

        if order["executed_qty"] > position["filled_quantity"]:
            outcome = self._apply_pending_fill(position, order)

            if outcome:
                return outcome

            if position["stage"] != PENDING_LIMIT_FILL:
                return None  # settled into TP1_PENDING this call

        invalidated = self._pending_entry_invalidated(position, latest_candle)
        expired = (
            time.time() - position["limit_placed_at"]
        ) >= max(float(config.LIMIT_ENTRY_EXPIRY_SECONDS), 0)

        if not invalidated and not expired:
            return None

        return self._resolve_pending_entry_cancel(position, invalidated)

    def _resolve_pending_entry_cancel(self, position, invalidated):
        symbol = position["symbol"]
        exchange.cancel_order(symbol, position["limit_order_id"])

        # A fill can race a cancel - Binance doesn't guarantee atomicity
        # between them - so always re-check ground truth after cancelling
        # instead of assuming the cancel definitely beat any last-moment
        # fill. `settle=True` here because the remainder is now dead
        # regardless of what status the exchange reports for it.
        order = exchange.get_order_status(symbol, position["limit_order_id"])

        if order["status"] != "UNKNOWN" and order["executed_qty"] > position["filled_quantity"]:
            return self._apply_pending_fill(position, order, settle=True)

        if position["filled_quantity"] > 0:
            # Already protected via SL/TP2 from an earlier partial fill;
            # the remainder is now cancelled - settle what's real rather
            # than treat an already-real, protected quantity as unfilled.
            self._settle_pending_entry_to_tp1_pending(position)
            return None

        return self._drop_unfilled_pending_entry(position, invalidated)

    def poll_shadow_pending_entry(self, symbol, latest_candle):
        """SHADOW equivalent of poll_pending_entry, simulated against the
        live candle stream the same way poll_shadow already is."""
        position = self.positions.get(symbol)

        if (
            not position or not position["shadow"]
            or position["stage"] != PENDING_LIMIT_FILL or not latest_candle
        ):
            return None

        side = position["side"]
        entry_price = position["entry_price"]
        sl_price = position["sl_price"]
        low = latest_candle["low"]
        high = latest_candle["high"]
        touches_entry = low <= entry_price <= high
        touches_sl = low <= sl_price if side == "BUY" else high >= sl_price

        if touches_entry and touches_sl:
            # Same deliberately conservative simplification poll_shadow
            # already documents: assume the stop would have been touched
            # first, don't overstate a would-be fill's win rate.
            return self._drop_unfilled_pending_entry(position, invalidated=True, shadow=True)

        if touches_entry:
            position["filled_quantity"] = position["quantity"]
            position["mae_price"] = entry_price
            position["mfe_price"] = entry_price
            position["stage"] = TP1_PENDING
            log_info(f"{symbol} [SHADOW] limit entry would have filled | entry={entry_price}")
            return None

        if touches_sl:
            # Reached the stop level without ever touching the entry -
            # invalidated, never filled.
            return self._drop_unfilled_pending_entry(position, invalidated=True, shadow=True)

        expired = (
            time.time() - position["limit_placed_at"]
        ) >= max(float(config.LIMIT_ENTRY_EXPIRY_SECONDS), 0)

        if expired:
            return self._drop_unfilled_pending_entry(position, invalidated=False, shadow=True)

        return None

    @staticmethod
    def _retracement_entry_invalidated(position, latest_candle):
        """Same real-time invalidation check as _pending_entry_invalidated
        (has price already reached the level the eventual stop would sit
        at, before the retracement limit ever filled?), reading sl_price
        out of the nested plan - config.RETRACEMENT_ENTRY_ENABLED's
        position shape, unlike PENDING_LIMIT_FILL's, never flattens it
        onto the position dict directly."""
        if not latest_candle:
            return False

        side = position["side"]
        sl_price = position["plan"]["sl_price"]

        if side == "BUY":
            return latest_candle["low"] <= sl_price

        return latest_candle["high"] >= sl_price

    @staticmethod
    def _retracement_runaway(position, latest_candle):
        """config.RETRACEMENT_REJECT_ON_RUNAWAY_R - only meaningful for a
        used_deep_retracement position, and only checked by callers once
        already expired (never while still resting - matches exactly what
        was asked for). Candle CLOSE, not high/low - see that config's own
        comment for why."""
        if not latest_candle or not position.get("used_deep_retracement"):
            return False

        plan = position["plan"]
        side = position["side"]
        trigger_price = plan["entry_price"]
        risk_distance = plan.get("risk_distance") or abs(trigger_price - plan["sl_price"])

        if risk_distance <= 0:
            return False

        offset = config.RETRACEMENT_REJECT_ON_RUNAWAY_R * risk_distance

        if side == "BUY":
            return latest_candle["close"] >= trigger_price + offset

        return latest_candle["close"] <= trigger_price - offset

    def _resolve_retracement_market_fallback(self, position, filled_quantity, filled_avg_price):
        """Places a market order for whatever quantity the resting
        retracement limit did NOT fill (all of it, if none) - the
        guarantee that config.RETRACEMENT_ENTRY_ENABLED never skips a
        signal the way a plain limit-with-no-fallback would. A no-op
        (returns the fill exactly as given) when the limit already filled
        in full - remainder <= 0. Blends a genuine partial limit fill with
        the market fallback into one quantity-weighted entry price, the
        same blending math risk_manager.build_dca_plan already uses for a
        DCA fill - a partial fill isn't thrown away just because the
        remainder timed out. Returns (total_quantity, entry_price,
        used_fallback, error) - error is None on success; on a
        market-order failure the resting limit has already been
        cancelled by the caller, so any real partial fill IS on the
        exchange right now, unprotected, and the caller must retry
        resolution next poll rather than give up. `used_fallback` (
        config.DCA_RESTING_ORDER_ENABLED's sibling observability need,
        see signal_journal.append_retracement_settle) is True whenever
        this placed (or tried to place) a real market order for any
        remainder - False only for the genuine no-op "already fully
        filled by the resting limit alone" case."""
        plan = position["plan"]
        symbol = position["symbol"]
        side = position["side"]
        remainder = round(plan["quantity"] - filled_quantity, 8)

        if remainder <= 0:
            return filled_quantity, filled_avg_price, False, None

        try:
            market_order = exchange.place_market_order(symbol, side, remainder)
        except Exception as exc:
            return None, None, True, f"market fallback order error: {exc}"

        market_price = exchange.resolve_market_fill_price(symbol, market_order, plan["entry_price"])
        total_quantity = round(filled_quantity + remainder, 8)

        if filled_quantity > 0:
            entry_price = (
                filled_avg_price * filled_quantity + market_price * remainder
            ) / total_quantity
        else:
            entry_price = market_price

        return total_quantity, entry_price, True, None

    def _finalize_retracement_entry(self, position, entry_price, quantity, fill_type):
        """Common tail once a final entry_price/quantity is known - a
        genuine full limit fill, a blended limit+market-fallback fill, or
        (shadow mode) a simulated touch/fallback. Builds a settled plan
        (entry_price/quantity/tp1_quantity/tp2_quantity/breakeven_price/
        risk_distance recomputed from these real numbers; every REAL
        structure-anchored level - sl_price/tp2_price/tp_price/dca_price -
        left exactly as risk_manager originally computed it, same
        principle _resolve_real_entry already applies for ordinary
        slippage on a synchronous market entry. tp1_price is the one
        exception: under config.TP_STATIC_ROI_ENABLED it's a pure
        function of entry_price, not a structure level, so it's
        recomputed here via _resolve_tp1_price - safe to do here
        specifically because protection orders haven't been placed yet at
        this point, unlike the synchronous entry paths - see that
        function's own docstring for the real bug this closes), places
        protection orders (DCA-shaped or plain, per position["is_dca"]/
        plan["single_tp"]), and hands off to register_dca_pending()/
        register() so the position ends up in the exact same shape a
        direct entry at this real price would have produced from the
        start. Returns an outcome string only if a non-DCA settle's SL
        placement failed outright (closed already, at market - see
        execution.place_protection_orders); None otherwise, including the
        ordinary case where this settled into DCA_PENDING/TP1_PENDING,
        which is not itself a closed-trade outcome.

        `fill_type` ("LIMIT" or "MARKET_FALLBACK", see poll_retracement_
        pending/poll_shadow_retracement_pending) is journaled here
        alongside the real fill lag - see signal_journal.
        append_retracement_settle's own docstring for why this also
        corrects the original signal row's stale (pre-retracement)
        entry_price, not just adds new fields."""
        plan = position["plan"]
        symbol = position["symbol"]
        side = position["side"]
        shadow = position["shadow"]
        is_dca = position["is_dca"]
        trade_id = position.get("trade_id")

        settled_plan = dict(plan)
        settled_plan["entry_price"] = entry_price
        settled_plan["quantity"] = quantity
        settled_plan["breakeven_price"] = risk_manager.compute_breakeven_price(entry_price, side)
        risk_distance = abs(entry_price - plan["sl_price"])
        settled_plan["risk_distance"] = risk_distance if risk_distance > 0 else plan.get("risk_distance")

        if not plan.get("single_tp"):
            settled_plan["tp1_price"] = _resolve_tp1_price(plan, entry_price, side)
        elif plan.get("tp1_static_roi_pct") is not None:
            # config.TP2_ENABLED=False - a single_tp plan built from a
            # static-ROI TP1 (config.TP_STATIC_ROI_ENABLED) has that same
            # "pure function of entry_price" drift problem the non-
            # single_tp branch above already handles - a retracement fill
            # settles at a different price than the original signal
            # estimate, so tp_price needs the same real-price recompute,
            # not just tp1_price. _resolve_tp1_price itself doesn't care
            # about single_tp - it only reads plan["tp1_price"]/plan.get
            # ("tp1_static_roi_pct"), both still populated in this plan
            # shape, so it's safe to reuse verbatim here.
            settled_plan["tp_price"] = _resolve_tp1_price(plan, entry_price, side)

        # config.RETRACEMENT_ENTRY_ENABLED observability (2026-08-25) -
        # journaled here (entry_price/quantity/fill_type all already
        # known) rather than after protection orders, so a SL-placement
        # failure below still gets this recorded instead of losing it.
        fill_lag_seconds = round(time.time() - position["limit_placed_at"], 1)
        signal_journal.append_retracement_settle(
            symbol, trade_id, entry_price, fill_type, fill_lag_seconds,
            used_deep_retracement=position.get("used_deep_retracement", False),
        )

        if plan.get("single_tp"):
            settled_plan["tp1_quantity"] = None
            settled_plan["tp2_quantity"] = None
        else:
            tp1_close_pct = min(max(float(config.TP1_CLOSE_PCT), 0), 100)
            settled_plan["tp1_quantity"] = round(quantity * tp1_close_pct / 100, 8)
            settled_plan["tp2_quantity"] = round(quantity - settled_plan["tp1_quantity"], 8)

        if shadow:
            execution_result = {
                "ok": True, "shadow": True, "entry_order": None,
                "sl_order": None, "tp1_order": None, "tp2_order": None, "tp_order": None,
                "dca_order": None,
            }
        elif is_dca:
            tp1_order, tp2_order, tp_order, dca_order = execution.place_dca_protection_orders(
                symbol, side, settled_plan
            )
            execution_result = {
                "ok": True, "shadow": False, "entry_order": None,
                "tp1_order": tp1_order, "tp2_order": tp2_order, "tp_order": tp_order,
                "dca_order": dca_order,
                "real_entry_price": entry_price,
            }
        else:
            sl_order, tp1_order, tp2_order, error = execution.place_protection_orders(
                symbol, side, settled_plan
            )

            if error:
                self.positions.pop(symbol, None)
                self._closed_at[symbol] = time.time()
                log_error(f"{symbol} retracement settle failed | {error}")
                signal_journal.append_outcome(symbol, "RETRACEMENT_SL_PLACEMENT_FAILED", trade_id)
                return "RETRACEMENT_SL_PLACEMENT_FAILED"

            execution_result = {
                "ok": True, "shadow": False, "entry_order": None,
                "sl_order": sl_order, "tp1_order": tp1_order, "tp2_order": tp2_order,
                "real_entry_price": entry_price,
            }

        if is_dca:
            self.register_dca_pending(settled_plan, execution_result, trade_id=trade_id)
        else:
            self.register(settled_plan, execution_result, trade_id=trade_id)

        return None

    def _drop_unfilled_retracement_entry(
        self, position, shadow=False,
        outcome="RETRACEMENT_INVALIDATED_UNFILLED", reason="invalidated before filling",
    ):
        """Genuinely zero-fill invalidation before the retracement limit
        ever touched - nothing was protected, nothing to unwind beyond the
        cancel already issued by the caller. Gets the same cooldown
        treatment as a real SL hit (the level failed before entry even
        happened) - see _drop_unfilled_pending_entry's identical
        reasoning for its own invalidated branch. `outcome`/`reason`
        default to the original SL-side-invalidation case; also reused
        (2026-08-30) for the RETRACEMENT_REJECT_ON_RUNAWAY_R expiry case
        with its own distinct outcome tag - same "nothing to unwind"
        shape either way."""
        symbol = position["symbol"]
        self.positions.pop(symbol, None)
        self._closed_at[symbol] = time.time()
        prefix = " [SHADOW]" if shadow else ""
        log_info(f"{symbol}{prefix} retracement entry {reason} - never entered")
        signal_journal.append_outcome(symbol, outcome, position.get("trade_id"))
        return outcome

    def poll_retracement_pending(self, symbol, latest_candle):
        """Returns an outcome string if this call closed things out
        outright (a genuine pre-fill invalidation, or a non-DCA settle's
        SL placement failure); None otherwise, including the ordinary
        case where this call either kept the limit resting or settled the
        position into DCA_PENDING/TP1_PENDING. Only meaningful for
        stage == RETRACEMENT_PENDING, non-shadow positions - see main.py's
        _poll_positions dispatch."""
        position = self.positions.get(symbol)

        if not position or position["shadow"] or position["stage"] != RETRACEMENT_PENDING:
            return None

        plan = position["plan"]
        order = exchange.get_order_status(symbol, position["limit_order_id"])

        if order["status"] == "UNKNOWN":
            return None  # transient fetch failure - retry next poll

        filled_quantity = order["executed_qty"]
        fully_filled = filled_quantity >= plan["quantity"]
        invalidated = self._retracement_entry_invalidated(position, latest_candle)
        expired = (
            time.time() - position["limit_placed_at"]
        ) >= max(float(position.get("retracement_timeout_seconds", config.RETRACEMENT_ENTRY_TIMEOUT_SECONDS)), 0)

        if not fully_filled and not invalidated and not expired:
            return None  # keep resting

        exchange.cancel_order(symbol, position["limit_order_id"])

        # A fill can race the cancel - re-check ground truth after, same
        # discipline poll_pending_entry's own _resolve_pending_entry_cancel
        # already applies.
        order = exchange.get_order_status(symbol, position["limit_order_id"])

        if order["status"] != "UNKNOWN":
            filled_quantity = order["executed_qty"]

        filled_avg_price = order["avg_price"] if filled_quantity > 0 else None

        if filled_quantity <= 0 and invalidated:
            # Setup failed before ever filling - nothing to unwind, nothing
            # protected, and never falls back to market into a broken
            # setup.
            return self._drop_unfilled_retracement_entry(position)

        # config.RETRACEMENT_REJECT_ON_RUNAWAY_R (2026-08-30) - a
        # deep-routed limit that timed out with price already having run
        # favorably past the reject threshold: the setup already played
        # out without us, so don't chase a now-materially-worse entry.
        # `not fully_filled` - nothing to reject if it already filled in
        # full. `not invalidated` - the SL-side invalidation case above
        # (or its partial-fill fallthrough below) is a different, already
        # -handled failure mode.
        if (
            expired and not invalidated and not fully_filled
            and self._retracement_runaway(position, latest_candle)
        ):
            if filled_quantity <= 0:
                return self._drop_unfilled_retracement_entry(
                    position,
                    outcome="RETRACEMENT_EXPIRED_REJECTED_RUNAWAY",
                    reason="expired unfilled - price already ran favorably past the reject threshold, not chasing",
                )

            log_info(
                f"{symbol} retracement expired with a partial fill and price already "
                f"ran favorably past the reject threshold - keeping the partial, not "
                f"chasing the remainder"
            )
            return self._finalize_retracement_entry(
                position, filled_avg_price, filled_quantity, "PARTIAL_NO_CHASE"
            )

        # Every other resolution (full fill, partial fill + invalidated,
        # partial/zero fill + expired) ends here: market-fallback for
        # whatever didn't fill (a no-op if it's already fully filled),
        # then finalize with the real (possibly blended) quantity/price -
        # guarantees this signal still becomes a position exactly like a
        # direct entry would have.
        total_quantity, entry_price, used_fallback, error = self._resolve_retracement_market_fallback(
            position, filled_quantity, filled_avg_price
        )

        if error:
            log_error(f"{symbol} {error} - leaving unresolved for the next poll")
            return None

        fill_type = "MARKET_FALLBACK" if used_fallback else "LIMIT"
        return self._finalize_retracement_entry(position, entry_price, total_quantity, fill_type)

    def _close_shadow_retracement_invalidated(self, position):
        symbol = position["symbol"]
        self.positions.pop(symbol, None)
        self._closed_at[symbol] = time.time()
        log_info(f"{symbol} [SHADOW] retracement entry invalidated before filling - never entered")
        signal_journal.append_outcome(
            symbol, "SHADOW_RETRACEMENT_INVALIDATED_UNFILLED", position.get("trade_id")
        )
        return "SHADOW_RETRACEMENT_INVALIDATED_UNFILLED"

    def poll_shadow_retracement_pending(self, symbol, latest_candle):
        """SHADOW equivalent of poll_retracement_pending - no real orders
        exist, so "filled" is simulated against the candle's own range
        (same deliberately conservative "assume the adverse side touched
        first" simplification poll_shadow's own docstring already
        documents for a real SL-vs-target ambiguity) and the market
        fallback on expiry just uses the candle's own close as the
        simulated fill price for the full planned quantity - there is no
        real partial fill to blend against in shadow mode."""
        position = self.positions.get(symbol)

        if (
            not position or not position["shadow"]
            or position["stage"] != RETRACEMENT_PENDING or not latest_candle
        ):
            return None

        plan = position["plan"]
        side = position["side"]
        retracement_price = position["retracement_price"]
        sl_price = plan["sl_price"]
        low = latest_candle["low"]
        high = latest_candle["high"]
        touches_retracement = low <= retracement_price <= high
        touches_sl = low <= sl_price if side == "BUY" else high >= sl_price

        if touches_sl:
            # Whether or not it also touched the retracement level this
            # same candle - same conservative "assume the adverse side
            # first" bias as the live invalidation check.
            return self._close_shadow_retracement_invalidated(position)

        if touches_retracement:
            return self._finalize_retracement_entry(
                position, retracement_price, plan["quantity"], "LIMIT"
            )

        expired = (
            time.time() - position["limit_placed_at"]
        ) >= max(float(position.get("retracement_timeout_seconds", config.RETRACEMENT_ENTRY_TIMEOUT_SECONDS)), 0)

        if expired:
            if self._retracement_runaway(position, latest_candle):
                return self._drop_unfilled_retracement_entry(
                    position, shadow=True,
                    outcome="SHADOW_RETRACEMENT_EXPIRED_REJECTED_RUNAWAY",
                    reason="expired unfilled - price already ran favorably past the reject threshold, not chasing",
                )

            log_info(
                f"{symbol} [SHADOW] retracement entry expired unfilled - "
                f"market fallback @ {latest_candle['close']}"
            )
            return self._finalize_retracement_entry(
                position, latest_candle["close"], plan["quantity"], "MARKET_FALLBACK"
            )

        return None
