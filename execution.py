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
from logger import log_error, log_info, log_warning


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
            # -4130 means a conflicting closePosition stop/TP was already
            # sitting on this symbol before this entry even started - real
            # bug found live (STGUSDT/DEXEUSDT, 2026-08-08): that order
            # survives the market-close above untouched (closePosition
            # orders aren't cancelled just because the position went flat),
            # so leaving it in place made every future entry attempt on
            # this symbol hit the identical -4130 forever, every eval
            # cycle, each one opening and immediately flattening a real
            # position for no reason. Clear it so the next attempt has a
            # clean slate. cancel_all_open_orders never raises (each of
            # its two endpoint calls is independently try/excepted).
            exchange.cancel_all_open_orders(symbol)

        return {"ok": False, "shadow": False, "error": f"SL placement failed: {exc}"}

    # TP1/TP2 are best-effort from here: the position already has real
    # downside protection via SL, so a failure here is degraded (missing
    # profit-taking) but not dangerous - the position still gets tracked.
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
    something this function tries to mitigate on its own."""
    symbol = plan["symbol"]
    side = plan["side"]

    if config.EXECUTION_MODE != "LIVE":
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

    # TP1/TP2 are best-effort, same as enter_trade - a missing TP here is
    # a degraded outcome (the trade still resolves via DCA-or-TP1 either
    # way), not a naked-position risk on its own, so it doesn't trigger
    # the emergency-close path enter_trade's SL step does.
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
