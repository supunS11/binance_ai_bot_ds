"""Places the entry, then attaches SL + TP1 + TP2 using the same order
mechanics v7/v8 use: a full-position STOP_MARKET (closePosition=true) for
the stop, a reduce-only TAKE_PROFIT_MARKET for TP1's partial quantity, and
a full-position TAKE_PROFIT_MARKET (closePosition=true) for TP2 covering
whatever remains after TP1.

Defaults to SHADOW mode (config.EXECUTION_MODE) - real order placement
only happens once that's explicitly switched to LIVE, so the very first
run of this bot cannot place a real order by accident.
"""
import config
import exchange
import risk_manager
from logger import log_error, log_info, log_warning


def place_protection_orders(symbol, side, plan):
    """SL (atomic - a failure here closes plan["quantity"] at market
    immediately, matching enter_trade's own discipline) then TP1(partial)+
    TP2(full), both best-effort. Extracted out of enter_trade's tail so
    config.RETRACEMENT_ENTRY_ENABLED's settle path (position_manager.
    _finalize_retracement_entry, resolving a fill that already happened -
    real limit fill, or its market fallback) can place the exact same
    protection without a second entry order. Returns (sl_order, tp1_order,
    tp2_order, error) - error is None on success; on SL failure the
    position has already been closed at market and (for a stale-order
    -4130) all open orders cancelled, and callers must NOT register a
    position at all."""
    try:
        sl_order = exchange.place_stop_loss(symbol, side, plan["sl_price"])
    except Exception as exc:
        log_error(
            f"{symbol} SL placement failed after entry filled - closing "
            f"position at market rather than leave it unprotected: {exc}"
        )
        try:
            exchange.close_position_market(symbol, side, plan["quantity"])
        except Exception as close_exc:
            log_error(
                f"{symbol} CRITICAL: failed to close unprotected position "
                f"after SL placement failure - manual intervention needed: "
                f"{close_exc}"
            )

        if "-4130" in str(exc):
            # See enter_trade's own comment on this - a conflicting
            # closePosition stop/TP already sitting on this symbol before
            # this entry started, otherwise stuck there forever blocking
            # every future attempt.
            exchange.cancel_all_open_orders(symbol)

        return None, None, None, f"SL placement failed: {exc}"

    tp1_order = None
    tp2_order = None

    try:
        tp1_order = exchange.place_take_profit_partial(
            symbol, side, plan["tp1_quantity"], plan["tp1_price"]
        )
    except Exception as exc:
        log_warning(f"{symbol} TP1 placement failed (SL is active): {exc}")

    try:
        tp2_order = exchange.place_take_profit_full(symbol, side, plan["tp2_price"])
    except Exception as exc:
        log_warning(f"{symbol} TP2 placement failed (SL is active): {exc}")

    return sl_order, tp1_order, tp2_order, None


def place_dca_protection_orders(symbol, side, plan):
    """No SL (matches enter_trade_dca_pending's own no-SL-before-DCA
    design) - a single full-position TP (plan["single_tp"]) or TP1
    (partial) + TP2 (full), both best-effort. Extracted out of
    enter_trade_dca_pending's tail for the same reason place_protection_
    orders was: config.RETRACEMENT_ENTRY_ENABLED's settle path needs to
    place DCA-shaped protection for a fill that already happened, without
    placing a second entry order. Returns (tp1_order, tp2_order, tp_order) -
    exactly one of (tp_order) or (tp1_order, tp2_order) is ever non-None
    depending on plan["single_tp"], the other pair stays None."""
    if plan.get("single_tp"):
        tp_order = None

        try:
            tp_order = exchange.place_take_profit_full(symbol, side, plan["tp_price"])
        except Exception as exc:
            log_warning(f"{symbol} TP placement failed: {exc}")

        return None, None, tp_order

    tp1_order = None
    tp2_order = None

    try:
        tp1_order = exchange.place_take_profit_partial(
            symbol, side, plan["tp1_quantity"], plan["tp1_price"]
        )
    except Exception as exc:
        log_warning(f"{symbol} TP1 placement failed: {exc}")

    try:
        tp2_order = exchange.place_take_profit_full(symbol, side, plan["tp2_price"])
    except Exception as exc:
        log_warning(f"{symbol} TP2 placement failed: {exc}")

    return tp1_order, tp2_order, None


def enter_trade(plan):
    symbol = plan["symbol"]
    side = plan["side"]

    if config.EXECUTION_MODE != "LIVE":
        log_info(
            f"[SHADOW] {symbol} would enter {side} qty={plan['quantity']} "
            f"entry~={plan['entry_price']} SL={plan['sl_price']} "
            f"TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
        )
        return {
            "ok": True,
            "shadow": True,
            "entry_order": None,
            "sl_order": None,
            "tp1_order": None,
            "tp2_order": None,
        }

    # Leverage must actually be confirmed before risking an entry attempt -
    # some symbols cap out below config.LEVERAGE (Binance rejects with
    # -4028 for that symbol/notional bracket). Proceeding anyway used to
    # place a doomed entry order against whatever leverage happened to
    # already be active on the account, failing a second time with a
    # confusing, unrelated-looking -2027. Abort cleanly instead - no
    # entry order is ever attempted, so there's nothing to unwind.
    if not exchange.setup_leverage(symbol):
        error = f"leverage {config.LEVERAGE}x not available for {symbol}"
        log_error(f"{symbol} entry aborted | {error}")
        return {"ok": False, "shadow": False, "error": error}

    # Entry + SL are treated as one atomic unit: once the entry order
    # fills, a real position exists on the exchange, so a failure from
    # here on must never be allowed to leave it both naked (no stop) and
    # untracked (main.py only registers the position when this returns
    # ok=True) - that combination is how a position silently loses its
    # protection with the bot having no idea it exists.
    try:
        entry_order = exchange.place_market_order(symbol, side, plan["quantity"])
    except Exception as exc:
        log_error(f"{symbol} entry order error: {exc}")
        return {"ok": False, "shadow": False, "error": str(exc)}

    real_entry_price = exchange.resolve_market_fill_price(symbol, entry_order, plan["entry_price"])

    sl_order, tp1_order, tp2_order, error = place_protection_orders(symbol, side, plan)

    if error:
        return {"ok": False, "shadow": False, "error": error}

    log_info(
        f"{symbol} entered {side} qty={plan['quantity']} | "
        f"SL={plan['sl_price']} TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
    )

    return {
        "ok": True,
        "shadow": False,
        "entry_order": entry_order,
        "sl_order": sl_order,
        "tp1_order": tp1_order,
        "tp2_order": tp2_order,
        "real_entry_price": real_entry_price,
    }


def enter_trade_dca_pending(plan):
    """config.DCA_ENABLED - places the entry + TP1 + TP2 exactly like
    enter_trade, but deliberately places NO SL. The position stays
    unprotected by any resting exchange order until either TP1 fills (the
    existing breakeven-promotion path finally places one, same as today)
    or price reaches plan["dca_price"] and position_manager._execute_dca
    places the first real SL alongside a new single post-DCA TP. Callers
    register the result via positions.register_dca_pending(...), not
    positions.register(...). See config.py's DCA section for the full
    rationale and the real risk being accepted during this window - not
    something this function tries to mitigate on its own.

    config.TP_STATIC_ROI_ENABLED - plan["single_tp"] routes to ONE
    full-position take-profit (plan["tp_price"]) instead of TP1(partial)+
    TP2(remainder) - see risk_manager.build_trade_plan's own comment.
    Still no SL either way; the DCA-or-TP race is otherwise unchanged."""
    symbol = plan["symbol"]
    side = plan["side"]
    single_tp = plan.get("single_tp")

    if config.EXECUTION_MODE != "LIVE":
        if single_tp:
            log_info(
                f"[SHADOW] {symbol} would enter {side} qty={plan['quantity']} "
                f"entry~={plan['entry_price']} (NO SL - dca_price={plan['dca_price']}) "
                f"TP={plan['tp_price']}"
            )
        else:
            log_info(
                f"[SHADOW] {symbol} would enter {side} qty={plan['quantity']} "
                f"entry~={plan['entry_price']} (NO SL - dca_price={plan['dca_price']}) "
                f"TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
            )
        return {
            "ok": True,
            "shadow": True,
            "entry_order": None,
            "tp1_order": None,
            "tp2_order": None,
            "tp_order": None,
        }

    # Same abort-before-any-order-attempt discipline as enter_trade.
    if not exchange.setup_leverage(symbol):
        error = f"leverage {config.LEVERAGE}x not available for {symbol}"
        log_error(f"{symbol} entry aborted | {error}")
        return {"ok": False, "shadow": False, "error": error}

    try:
        entry_order = exchange.place_market_order(symbol, side, plan["quantity"])
    except Exception as exc:
        log_error(f"{symbol} entry order error: {exc}")
        return {"ok": False, "shadow": False, "error": str(exc)}

    real_entry_price = exchange.resolve_market_fill_price(symbol, entry_order, plan["entry_price"])

    # TP (single, or TP1/TP2) is best-effort, same as enter_trade - a
    # missing TP here is a degraded outcome (the trade still resolves via
    # DCA-or-TP1/TP either way), not a naked-position risk on its own, so
    # it doesn't trigger the emergency-close path enter_trade's SL step
    # does.
    tp1_order, tp2_order, tp_order = place_dca_protection_orders(symbol, side, plan)

    if single_tp:
        log_info(
            f"{symbol} entered {side} qty={plan['quantity']} | NO SL (DCA pending) "
            f"dca_price={plan['dca_price']} TP={plan['tp_price']}"
        )
    else:
        log_info(
            f"{symbol} entered {side} qty={plan['quantity']} | NO SL (DCA pending) "
            f"dca_price={plan['dca_price']} TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
        )

    return {
        "ok": True,
        "shadow": False,
        "entry_order": entry_order,
        "tp1_order": tp1_order,
        "tp2_order": tp2_order,
        "tp_order": tp_order,
        "real_entry_price": real_entry_price,
    }


def enter_trade_limit(plan):
    """config.LIMIT_ENTRY_MODE_ENABLED - places a resting GTC LIMIT entry
    at plan["entry_price"] instead of a market order. Deliberately a
    separate function from enter_trade rather than a branch inside it:
    the LIVE body here is structurally different, not just a different
    order type - there is no fill yet, so there is nothing to protect and
    no synchronous SL-after-fill step. Protection (SL/TP2/TP1) is placed
    later, asynchronously, once position_manager.poll_pending_entry
    actually detects a fill. Callers register the result via
    positions.register_pending_entry(...), not positions.register(...)."""
    symbol = plan["symbol"]
    side = plan["side"]

    if config.EXECUTION_MODE != "LIVE":
        log_info(
            f"[SHADOW] {symbol} would place a LIMIT {side} qty={plan['quantity']} "
            f"@ {plan['entry_price']} SL={plan['sl_price']} "
            f"TP1={plan['tp1_price']} TP2={plan['tp2_price']}"
        )
        return {"ok": True, "shadow": True, "entry_order": None}

    # Same abort-before-any-order-attempt discipline as enter_trade - no
    # entry order is ever attempted if the configured leverage isn't
    # actually available for this symbol.
    if not exchange.setup_leverage(symbol):
        error = f"leverage {config.LEVERAGE}x not available for {symbol}"
        log_error(f"{symbol} limit entry aborted | {error}")
        return {"ok": False, "shadow": False, "error": error}

    try:
        entry_order = exchange.place_limit_order(
            symbol, side, plan["quantity"], plan["entry_price"]
        )
    except Exception as exc:
        log_error(f"{symbol} limit entry order error: {exc}")
        return {"ok": False, "shadow": False, "error": str(exc)}

    log_info(
        f"{symbol} limit entry placed {side} qty={plan['quantity']} "
        f"@ {plan['entry_price']} | SL={plan['sl_price']} "
        f"TP1={plan['tp1_price']} TP2={plan['tp2_price']} "
        f"(pending fill, expires in {config.LIMIT_ENTRY_EXPIRY_SECONDS}s)"
    )

    return {"ok": True, "shadow": False, "entry_order": entry_order}


def enter_trade_retracement(plan):
    """config.RETRACEMENT_ENTRY_ENABLED - places a resting GTC LIMIT at a
    small pullback toward the stop (risk_manager.compute_retracement_
    price) instead of paying the trigger-instant price plan["entry_price"]
    itself. No protection is placed here - there is no fill yet, and
    unlike enter_trade_limit this doesn't even know yet whether the
    eventual position will be DCA-shaped or plain. position_manager.
    poll_retracement_pending/poll_shadow_retracement_pending resolve the
    fill (or the bounded market fallback via config.RETRACEMENT_ENTRY_
    TIMEOUT_SECONDS if it never comes) and place protection afterward via
    place_protection_orders/place_dca_protection_orders. Callers register
    the result via positions.register_retracement_pending(...), not
    register()/register_dca_pending() directly - see config.py's own
    comment for the full mechanism and the real evidence behind it."""
    symbol = plan["symbol"]
    side = plan["side"]
    retracement_price = risk_manager.compute_retracement_price(
        plan["entry_price"], plan["sl_price"], side
    )

    if config.EXECUTION_MODE != "LIVE":
        log_info(
            f"[SHADOW] {symbol} would place a RETRACEMENT limit {side} "
            f"qty={plan['quantity']} @ {retracement_price} "
            f"(trigger price was {plan['entry_price']})"
        )
        return {
            "ok": True, "shadow": True, "entry_order": None,
            "retracement_price": retracement_price,
        }

    # Same abort-before-any-order-attempt discipline as enter_trade.
    if not exchange.setup_leverage(symbol):
        error = f"leverage {config.LEVERAGE}x not available for {symbol}"
        log_error(f"{symbol} retracement entry aborted | {error}")
        return {"ok": False, "shadow": False, "error": error}

    try:
        entry_order = exchange.place_limit_order(symbol, side, plan["quantity"], retracement_price)
    except Exception as exc:
        log_error(f"{symbol} retracement entry order error: {exc}")
        return {"ok": False, "shadow": False, "error": str(exc)}

    log_info(
        f"{symbol} retracement entry placed {side} qty={plan['quantity']} "
        f"@ {retracement_price} (trigger was {plan['entry_price']}) | "
        f"expires in {config.RETRACEMENT_ENTRY_TIMEOUT_SECONDS}s -> market fallback"
    )

    return {
        "ok": True, "shadow": False, "entry_order": entry_order,
        "retracement_price": retracement_price,
    }
