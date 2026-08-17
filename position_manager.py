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
import time

import config
import exchange
import market_structure
import risk_manager
import signal_journal
from logger import log_error, log_info, log_warning


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


class PositionManager:
    def __init__(self):
        self.positions = {}
        self._closed_at = {}  # symbol -> timestamp of its most recent close

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

    def register(self, plan, execution_result, trade_id=None):
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
            "confluence_ratio": plan.get("confluence_ratio"),
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
            "mae_price": plan["entry_price"],
            "mfe_price": plan["entry_price"],
            # Fixed at entry, never touched again - see
            # _mae_mfe_r_multiples for why this must never be re-derived
            # from position["sl_price"] later (that field legitimately
            # moves to the breakeven price once a trade is promoted).
            "risk_distance": plan.get("risk_distance") or abs(plan["entry_price"] - plan["sl_price"]),
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
        attempted twice for the same position."""
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
            "tp1_order_id": (
                exchange._accepted_order_id(execution_result.get("tp1_order"))
                if not shadow else None
            ),
            "tp2_order_id": (
                exchange._accepted_order_id(execution_result.get("tp2_order"))
                if not shadow else None
            ),
            "stage": DCA_PENDING,
            "shadow": shadow,
            "opened_at": time.time(),
            "confluence_ratio": plan.get("confluence_ratio"),
            "early_breakeven_applied": False,
            "early_breakeven_profit_locked": False,
            "profit_protection_applied": False,
            "profit_protection_profit_locked": False,
            "profit_protection_peak_price": None,
            "trailing_stop_locked_profit": False,
            "mae_price": plan["entry_price"],
            "mfe_price": plan["entry_price"],
            "risk_distance": plan.get("risk_distance") or abs(plan["entry_price"] - plan["sl_price"]),
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
            "confluence_ratio": plan.get("confluence_ratio"),
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

    def reconcile_on_startup(self):
        """Rebuild tracking for positions already open on the exchange
        when the process starts - crash, manual restart, or a redeploy.
        Without this, a restart makes the bot blind to real open
        positions: has_open_position() would wrongly say "no" for a
        symbol that already has one (risking a duplicate entry stacked on
        top of it), and the existing position would never get TP1 ->
        breakeven promotion, missing-order self-heal, or its outcome
        journaled - even though its real SL/TP1/TP2 orders keep working
        on Binance's side regardless of whether the bot remembers them."""
        live_positions = exchange.get_all_open_positions()
        adopted = 0

        for live_position in live_positions:
            symbol = live_position["symbol"]

            if symbol in self.positions:
                continue

            self._adopt_position(symbol, live_position)
            adopted += 1

        if live_positions:
            log_info(
                f"Startup reconciliation | {len(live_positions)} open "
                f"position(s) found on the exchange | {adopted} adopted"
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
        exchange position attached."""
        if not config.LIMIT_ENTRY_MODE_ENABLED:
            return

        cancelled = 0

        for order in exchange.get_all_open_orders():
            if str(order.get("type") or "").upper() != "LIMIT":
                continue

            symbol = order.get("symbol")
            order_id = order.get("orderId")

            if not symbol or not order_id:
                continue

            exchange.cancel_order(symbol, order_id)
            cancelled += 1
            log_warning(
                f"{symbol} cancelled a resting LIMIT order found on "
                f"restart (order_id={order_id}) - no recoverable pending-"
                f"entry plan survives a restart, re-evaluating fresh instead"
            )

        if cancelled:
            log_info(
                f"Startup reconciliation | {cancelled} resting limit "
                f"entry order(s) cancelled"
            )

    def _adopt_position(self, symbol, live_position):
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

        def _trigger_price(order):
            if not order:
                return None
            return _safe_float(order.get("triggerPrice") or order.get("stopPrice"))

        sl_price = _trigger_price(sl_order)

        if sl_price is None:
            # A real open position with no stop at all - treat as an
            # emergency: reconstruct a minimum-distance stop and place it
            # immediately rather than leave it unprotected until the next
            # opportunity to notice.
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
            "trade_id": f"{symbol}_RECOVERED_{int(time.time() * 1000)}",
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
            # No original signal to read confluence from - kept for
            # journaling only; early breakeven no longer depends on it
            # (see _is_early_breakeven_candidate).
            "confluence_ratio": None,
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
            # policy as confluence_ratio/risk_distance above.
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
        current (possibly-moved) sl_price."""
        entry_price = position["entry_price"]
        risk_distance = position.get("risk_distance")

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
                    "closing remainder at market"
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
        _early_breakeven_lock_price)."""
        target_price = position["breakeven_price"] if target_price is None else target_price

        if not config.MOVE_SL_TO_BREAKEVEN_AFTER_TP1:
            position["stage"] = BREAKEVEN_ACTIVE
            return None

        outcome, replaced = self._replace_sl_order(position, target_price, reason)

        if outcome is not None:
            return outcome

        if replaced:
            position["stage"] = BREAKEVEN_ACTIVE

        return None

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
                else "BREAKEVEN_TRIGGER_MARKET_CLOSE"
            )
            return self._close(symbol, outcome)

        except Exception as exc:
            log_error(f"{symbol} market-close-remainder error: {exc}")
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
    def _dca_price_reached(position, current_price):
        """Has price reached position["dca_price"] yet - the adverse-
        direction level computed once at plan time (risk_manager.
        compute_dca_price). Shared by poll_live (real mark price) and
        poll_shadow (simulated candle touch)."""
        if current_price is None or current_price <= 0:
            return False

        side = position["side"]
        dca_price = position["dca_price"]
        return current_price <= dca_price if side == "BUY" else current_price >= dca_price

    def _execute_dca(self, position, candles=None):
        """config.DCA_ENABLED - price reached position["dca_price"]
        before TP1 ever filled. Adds position["dca_quantity"] at that
        level (same "trust the planned price, don't complicate with a
        separately-fetched real fill price" convention execution.
        enter_trade already uses for the original entry - this codebase
        has never queried a market order's real average fill price
        anywhere), computes the blended entry / first-ever-real-SL /
        single-TP via risk_manager.build_dca_plan, cancels the original
        TP1+TP2, places the new single TP + the first real SL, and
        transitions to DCA_ACTIVE. Mirrors _apply_pending_fill's "SL must
        be atomic even when placed asynchronously" discipline: a failed
        post-DCA SL placement closes the position at market immediately
        rather than leave real (now-doubled) quantity unprotected.

        Returns a close-outcome string only if the post-DCA SL placement
        failed badly enough to force closing the position; otherwise
        None (including the ordinary case where DCA succeeded and the
        position lives on as DCA_ACTIVE)."""
        symbol = position["symbol"]
        side = position["side"]
        shadow = position["shadow"]
        dca_fill_price = position["dca_price"]

        if not shadow:
            try:
                exchange.place_market_order(symbol, side, position["dca_quantity"])
            except Exception as exc:
                log_error(f"{symbol} DCA order error: {exc}")
                return None  # retry next poll - not worse off than before

        pools = (
            market_structure.find_liquidity_pools(market_structure.find_swing_points(candles))
            if candles else None
        )
        plan = risk_manager.build_dca_plan(
            position["entry_price"], position["quantity"],
            dca_fill_price, position["dca_quantity"], side, pools,
            atr=position.get("atr"),
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
                tp_order = exchange.place_take_profit_full(symbol, side, plan["tp_price"])
                position["tp_order_id"] = exchange._accepted_order_id(tp_order)
            except Exception as exc:
                log_warning(f"{symbol} post-DCA single-TP placement failed: {exc}")
                position["tp_order_id"] = None

            try:
                sl_order = exchange.place_stop_loss(symbol, side, plan["sl_price"])
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
        else:
            position["tp_order_id"] = None

        position["entry_price"] = plan["entry_price"]
        position["sl_price"] = plan["sl_price"]
        position["tp_price"] = plan["tp_price"]
        position["quantity"] = plan["quantity"]
        position["risk_distance"] = plan["risk_distance"]
        position["dca_applied"] = True
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

        if position["stage"] != TP1_PENDING:
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

        if position["stage"] != TP1_PENDING:
            return False

        return position.get("tp1_price") is not None

    @staticmethod
    def _profit_protection_lock_price(position):
        """The ARM trigger price only - see risk_manager.
        compute_profit_protection_lock_price. Once armed, the stop no
        longer jumps straight to this price; _profit_protection_trailing_
        floor takes over (see its docstring and config.py's comment on
        PROFIT_PROTECTION_LOCK_PCT_OF_TP1 for why)."""
        return risk_manager.compute_profit_protection_lock_price(
            position["entry_price"], position["side"], position["tp1_price"]
        )

    @staticmethod
    def _profit_protection_trailing_floor(position, peak_price):
        """Where the stop actually gets set, both at arm time and on
        every subsequent trail - see risk_manager.
        compute_profit_protection_trailing_floor."""
        return risk_manager.compute_profit_protection_trailing_floor(
            position["entry_price"], position["side"], position["tp1_price"], peak_price
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

        candidate = self._profit_protection_trailing_floor(position, peak_price)

        if candidate is None or not _more_favorable(side, candidate, position["sl_price"]):
            return None

        return self._replace_sl_order(
            position, candidate, "Profit protection trailing stop", close_if_not_open=False
        )[0]

    def _profit_protection_price_reached(self, position, current_price):
        """Has price reached the profit-protection lock price yet? Shared
        by poll_live (real mark price) and poll_shadow (simulated candle
        close) - same pattern as _early_breakeven_price_reached."""
        if current_price is None or current_price <= 0:
            return False

        lock_price = self._profit_protection_lock_price(position)

        if lock_price is None:
            return False

        side = position["side"]
        return current_price >= lock_price if side == "BUY" else current_price <= lock_price

    def poll_live(self, symbol, candles=None):
        """Returns an outcome string if the position closed this call,
        otherwise None. `candles` (LTF history for the symbol) is only
        used by config.STRUCTURE_STOP_MANAGEMENT_ENABLED's structure-aware
        early-breakeven lock and post-TP1 trailing stop - both are no-ops
        when it's not supplied."""
        position = self.positions.get(symbol)

        if not position or position["shadow"]:
            return None

        self._ensure_protection_orders(position)

        # One shared mark-price fetch, reused for MAE/MFE tracking and
        # both early-promotion checks below - no reason to pay for extra
        # REST calls when any of them need the same current price.
        early_breakeven_candidate = (
            position["stage"] == TP1_PENDING and self._is_early_breakeven_candidate(position)
        )
        profit_protection_candidate = (
            position["stage"] == TP1_PENDING and self._is_profit_protection_candidate(position)
        )
        dca_candidate = self._is_dca_candidate(position)
        current_price = None

        if (
            config.MAE_TRACKING_ENABLED or early_breakeven_candidate
            or profit_protection_candidate or dca_candidate
        ):
            current_price = exchange.get_mark_price(symbol)
            self._update_mae_mfe(position, current_price)

        if position["stage"] == DCA_PENDING:
            if dca_candidate and self._dca_price_reached(position, current_price):
                return self._execute_dca(position, candles=candles)

            tp1_status = self._status_or_missing(symbol, position["tp1_order_id"])

            if tp1_status == "FINISHED":
                return self._promote_to_breakeven(position)

            tp2_status = self._status_or_missing(symbol, position["tp2_order_id"])

            if tp2_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "TP2_HIT_DIRECT")

            return None

        if position["stage"] == DCA_ACTIVE:
            sl_status = self._status_or_missing(symbol, position["sl_order_id"])

            if sl_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_SL_HIT")

            tp_status = self._status_or_missing(symbol, position.get("tp_order_id"))

            if tp_status == "FINISHED":
                exchange.cancel_all_open_orders(symbol)
                return self._close(symbol, "DCA_TP_HIT")

            return None

        if position["stage"] == TP1_PENDING:
            # Checked before EARLY_BREAKEVEN: mutually exclusive at the
            # promotion moment (both check stage==TP1_PENDING and stop
            # applying once promoted), so whichever fires here is simply
            # whichever threshold was reached first in time - order only
            # matters on the rare tick where both would qualify at once.
            if (
                profit_protection_candidate
                and self._profit_protection_price_reached(position, current_price)
            ):
                position["profit_protection_peak_price"] = current_price
                lock_price = self._profit_protection_trailing_floor(position, current_price)

                if lock_price is not None:
                    position["profit_protection_applied"] = True
                    position["profit_protection_profit_locked"] = True
                    self._promote_to_breakeven(
                        position,
                        reason="Profit protection (ROI-of-TP1 trailing arm)",
                        target_price=lock_price,
                    )
                    return None

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
                return None

            tp1_status = self._status_or_missing(symbol, position["tp1_order_id"])

            if tp1_status == "FINISHED":
                return self._promote_to_breakeven(position)

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

            return

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
        self._promote_to_breakeven(position)

    def poll_shadow(self, symbol, latest_candle, candles=None):
        """Simulates the same TP1 -> breakeven -> TP2/SL sequence against
        live price action. When both the stop and a target fall inside the
        same candle's range, the SL side is assumed to have been touched
        first - a deliberately conservative simplification so shadow stats
        don't overstate win rate; it is not a substitute for real fills.
        `candles` (LTF history) is only used by config.STRUCTURE_STOP_MANAGEMENT_ENABLED's
        structure-aware early-breakeven lock and post-TP1 trailing - both
        are no-ops when it's not supplied."""
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

            # Checked before EARLY_BREAKEVEN - see the identical note in
            # poll_live.
            if self._is_profit_protection_candidate(position) and self._profit_protection_price_reached(
                position, latest_candle["close"]
            ):
                arm_price = latest_candle["close"]
                position["profit_protection_peak_price"] = arm_price
                lock_price = self._profit_protection_trailing_floor(position, arm_price)

                if lock_price is not None:
                    position["profit_protection_applied"] = True
                    position["profit_protection_profit_locked"] = True
                    position["stage"] = BREAKEVEN_ACTIVE
                    position["sl_price"] = lock_price
                    log_info(
                        f"{symbol} [SHADOW] profit protection (ROI-of-TP1 trailing arm) | "
                        f"SL -> {position['sl_price']}"
                    )
                    return None

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
                return None

            hit_tp1 = (
                high >= position["tp1_price"]
                if side == "BUY"
                else low <= position["tp1_price"]
            )

            if hit_tp1:
                position["stage"] = BREAKEVEN_ACTIVE
                position["sl_price"] = position["breakeven_price"]
                log_info(f"{symbol} [SHADOW] TP1 would have filled | SL -> breakeven")

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
                    return self._execute_dca(position, candles=candles)

            hit_tp1 = (
                high >= position["tp1_price"]
                if side == "BUY"
                else low <= position["tp1_price"]
            )

            if hit_tp1:
                position["stage"] = BREAKEVEN_ACTIVE
                position["sl_price"] = position["breakeven_price"]
                log_info(f"{symbol} [SHADOW] TP1 would have filled | SL -> breakeven")

            return None

        if position["stage"] == DCA_ACTIVE:
            hit_sl = low <= position["sl_price"] if side == "BUY" else high >= position["sl_price"]

            if hit_sl:
                return self._close(symbol, "SHADOW_DCA_SL_HIT")

            hit_tp = high >= position["tp_price"] if side == "BUY" else low <= position["tp_price"]

            if hit_tp:
                return self._close(symbol, "SHADOW_DCA_TP_HIT")

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

            if position.get("profit_protection_applied"):
                peak_source = high if side == "BUY" else low
                peak_price = position.get("profit_protection_peak_price")
                peak_price = peak_source if peak_price is None else (
                    max(peak_price, peak_source) if side == "BUY" else min(peak_price, peak_source)
                )
                position["profit_protection_peak_price"] = peak_price

                candidate = self._profit_protection_trailing_floor(position, peak_price)

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
