"""Places the entry, then attaches SL + TP1 + TP2 using the same order
mechanics v7/v8 use: a full-position STOP_MARKET (closePosition=true) for
the stop, a reduce-only TAKE_PROFIT_MARKET for TP1's partial quantity, and
a full-position TAKE_PROFIT_MARKET (closePosition=true) for TP2 covering
whatever remains after TP1.

Defaults to SHADOW mode (config.EXECUTION_MODE) - real order placement
only happens once that's explicitly switched to LIVE, so the very first
run of this bot cannot place a real order by accident. config.SHADOW_ONLY_
TRIGGERS can also force an individual trigger into shadow while the bot
stays globally LIVE - see _is_shadow_mode below.
"""
import time

import config
import exchange
import risk_manager
from logger import log_error, log_info, log_warning


def _is_shadow_mode(plan):
    """True if this specific plan should not place a real order - either
    the whole bot is in shadow mode, or its trigger is individually forced
    into shadow via config.SHADOW_ONLY_TRIGGERS (evidence-gate philosophy
    applied per-trigger instead of bot-wide - see that config's own
    comment). One-directional: can only add shadow behavior on top of
    LIVE, never force a trigger LIVE while EXECUTION_MODE is SHADOW."""
    return (
        config.EXECUTION_MODE != "LIVE"
        or plan.get("signal_trigger") in config.SHADOW_ONLY_TRIGGERS
    )

# config.DCA_RESTING_ORDER_ENABLED - tags the resting LIMIT order placed
# for the DCA add itself (a plain order, not algo - see exchange.
# place_limit_order's own client_order_id docstring), so position_manager.
# reconcile_pending_entries_on_startup can tell a legitimate DCA-add order
# for an already-open, tracked position apart from a genuinely orphaned
# pending-entry LIMIT order - otherwise indistinguishable on Binance's
# account-wide open-orders endpoint, and that routine's default behavior
# is to cancel every resting plain LIMIT order it finds. Lives here (not
# position_manager.py, alongside the analogous _DCA_SL_CLIENT_ALGO_ID_
# PREFIX/_DCA_TP_CLIENT_ALGO_ID_PREFIX) because this is where the order
# is actually placed, and position_manager already imports this module -
# importing the other way around would be circular.
DCA_ADD_CLIENT_ORDER_ID_PREFIX = "dcaAdd"
# config.DCA_PROTECTIVE_FIRST_ENABLED - tags the resting protective SL
# (an algo order, unlike DCA_ADD_CLIENT_ORDER_ID_PREFIX above - see
# exchange.place_stop_loss's own client_algo_id docstring) placed at
# dca_price instead of a quantity-adding order, so position_manager.
# _adopt_position can tell a protective-first DCA_PENDING position apart
# from an ordinary post-TP1 position on restart (both would otherwise
# look identical: one real SL + resting TP order(s)) - same
# clientAlgoId-tag disambiguation _DCA_SL_CLIENT_ALGO_ID_PREFIX already
# gives DCA_ACTIVE recovery.
DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_PREFIX = "dcaProtSL"


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


def _place_dca_resting_order(symbol, side, plan):
    """config.DCA_RESTING_ORDER_ENABLED - see its own config.py comment
    for the full rationale. Best-effort, same treatment as TP1/TP2/TP
    above: a failed resting-order placement leaves the position exactly
    as protected as it is today (candle-range detection + reactive
    market order, position_manager._dca_price_reached_in_range), not
    worse - never escalates to an emergency close the way a failed SL
    placement does elsewhere.

    Always sized at DCA_PRESSURE_SIZE_MULTIPLIER (the pressure check's
    own "not confirmed"/conservative branch), never the full
    DCA_SIZE_MULTIPLIER size - see config.DCA_RESTING_ORDER_ENABLED's
    comment for why this is the right default even on the "confirmed"
    fraction of fires. Returns None if the flag is off or placement
    failed; position_manager.register_dca_pending stores None for
    dca_order_id either way, which is exactly what makes it fall back to
    the existing candle-range detection path."""
    if not config.DCA_RESTING_ORDER_ENABLED:
        return None

    resting_quantity = round(
        plan["quantity"] * max(float(config.DCA_PRESSURE_SIZE_MULTIPLIER), 0), 8
    )

    if resting_quantity <= 0 or plan.get("dca_price") is None:
        return None

    try:
        return exchange.place_limit_order(
            symbol, side, resting_quantity, plan["dca_price"],
            client_order_id=f"{DCA_ADD_CLIENT_ORDER_ID_PREFIX}{int(time.time() * 1000)}",
        )
    except Exception as exc:
        log_warning(f"{symbol} DCA resting order placement failed: {exc}")
        return None


def _place_dca_protective_stop(symbol, side, plan):
    """config.DCA_PROTECTIVE_FIRST_ENABLED - see its own config.py
    comment for the full evidence/rationale. Quantity-neutral (closes
    whatever is currently open on this side, same exchange.place_stop_
    loss shape used everywhere else in this file) - unlike
    _place_dca_resting_order above, this never adds to the position by
    itself; escalating to a real add is position_manager._try_dca_
    protective_escalation's job, not this function's. Best-effort, same
    treatment as _place_dca_resting_order - a failed placement here
    doesn't block the entry (TP1/TP2 are already correct), it just means
    position_manager._ensure_protection_orders' self-heal retries next
    poll rather than leaving the position silently unprotected forever."""
    if not config.DCA_PROTECTIVE_FIRST_ENABLED or plan.get("dca_price") is None:
        return None

    try:
        return exchange.place_stop_loss(
            symbol, side, plan["dca_price"],
            client_algo_id=f"{DCA_PROTECTIVE_SL_CLIENT_ALGO_ID_PREFIX}{int(time.time() * 1000)}",
        )
    except Exception as exc:
        log_warning(f"{symbol} DCA protective stop placement failed: {exc}")
        return None


def _place_dca_resting_or_protective_order(symbol, side, plan):
    """config.DCA_PROTECTIVE_FIRST_ENABLED supersedes config.
    DCA_RESTING_ORDER_ENABLED when both are on - a position gets exactly
    one resting DCA-price order, never both. See each flag's own
    config.py comment."""
    if config.DCA_PROTECTIVE_FIRST_ENABLED:
        return _place_dca_protective_stop(symbol, side, plan)

    return _place_dca_resting_order(symbol, side, plan)


def place_dca_protection_orders(symbol, side, plan):
    """No SL by default (matches enter_trade_dca_pending's own no-SL-
    before-DCA design, UNLESS config.DCA_PROTECTIVE_FIRST_ENABLED is on -
    see that flag's own comment) - a single full-position TP
    (plan["single_tp"]) or TP1 (partial) + TP2 (full), both best-effort,
    plus a resting order at dca_price: either a quantity-ADDING LIMIT
    order (config.DCA_RESTING_ORDER_ENABLED, see _place_dca_resting_
    order) or a quantity-NEUTRAL protective stop (config.
    DCA_PROTECTIVE_FIRST_ENABLED, see _place_dca_protective_stop) -
    never both, see _place_dca_resting_or_protective_order. Extracted out
    of enter_trade_dca_pending's tail for the same reason place_protection_
    orders was: config.RETRACEMENT_ENTRY_ENABLED's settle path needs to
    place DCA-shaped protection for a fill that already happened, without
    placing a second entry order. Returns (tp1_order, tp2_order, tp_order,
    dca_order) - exactly one of (tp_order) or (tp1_order, tp2_order) is
    ever non-None depending on plan["single_tp"], the other pair stays
    None; dca_order is None whenever both flags are off, shadow, or
    placement failed."""
    if plan.get("single_tp"):
        tp_order = None

        try:
            tp_order = exchange.place_take_profit_full(symbol, side, plan["tp_price"])
        except Exception as exc:
            log_warning(f"{symbol} TP placement failed: {exc}")

        return None, None, tp_order, _place_dca_resting_or_protective_order(symbol, side, plan)

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

    return tp1_order, tp2_order, None, _place_dca_resting_or_protective_order(symbol, side, plan)


def enter_trade(plan):
    symbol = plan["symbol"]
    side = plan["side"]

    if _is_shadow_mode(plan):
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
    enter_trade, but deliberately places NO SL (UNLESS config.
    DCA_PROTECTIVE_FIRST_ENABLED is on - see that flag's own config.py
    comment - in which case place_dca_protection_orders rests a real,
    quantity-neutral protective stop at dca_price immediately, and this
    "no SL" framing no longer applies). Without that flag: the position
    stays unprotected by any resting exchange order until either TP1
    fills (the existing breakeven-promotion path finally places one, same
    as today) or price reaches plan["dca_price"] and position_manager.
    _execute_dca places the first real SL alongside a new single post-DCA
    TP. Callers register the result via positions.register_dca_pending(
    ...), not positions.register(...). See config.py's DCA section for
    the full rationale and the real risk being accepted during this
    window - not something this function tries to mitigate on its own.

    config.TP_STATIC_ROI_ENABLED - plan["single_tp"] routes to ONE
    full-position take-profit (plan["tp_price"]) instead of TP1(partial)+
    TP2(remainder) - see risk_manager.build_trade_plan's own comment.
    Still no SL either way (absent DCA_PROTECTIVE_FIRST_ENABLED); the
    DCA-or-TP race is otherwise unchanged."""
    symbol = plan["symbol"]
    side = plan["side"]
    single_tp = plan.get("single_tp")

    if _is_shadow_mode(plan):
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
            "dca_order": None,
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
    # does. dca_order (config.DCA_RESTING_ORDER_ENABLED) is the same
    # best-effort treatment - see _place_dca_resting_order.
    tp1_order, tp2_order, tp_order, dca_order = place_dca_protection_orders(symbol, side, plan)

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
        "dca_order": dca_order,
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

    if _is_shadow_mode(plan):
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

    # config.RETRACEMENT_DEPTH_AWARE_ENABLED - see config.py's own comment
    # for the real evidence behind the 0.30 threshold. depth_imbalance is
    # None (data unavailable) -> use_deep stays False -> today's shallow/
    # base-timeout behavior unchanged - missing data must never
    # independently trigger different order placement, same fail-open
    # convention as every gate in signal_engine.py.
    depth_imbalance = plan.get("depth_imbalance")
    signed_depth = (
        (depth_imbalance if side == "BUY" else -depth_imbalance)
        if depth_imbalance is not None else None
    )
    use_deep = (
        config.RETRACEMENT_DEPTH_AWARE_ENABLED
        and signed_depth is not None
        and signed_depth < config.RETRACEMENT_DEPTH_AWARE_MIN_IMBALANCE
    )
    timeout_seconds = (
        config.RETRACEMENT_ENTRY_TIMEOUT_DEEP_SECONDS if use_deep
        else config.RETRACEMENT_ENTRY_TIMEOUT_SECONDS
    )

    retracement_price = risk_manager.compute_retracement_price(
        plan["entry_price"], plan["sl_price"], side,
        fvgs=plan.get("fair_value_gaps"), pools=plan.get("liquidity_pools"),
        prefer_deeper=use_deep,
    )

    if _is_shadow_mode(plan):
        log_info(
            f"[SHADOW] {symbol} would place a RETRACEMENT limit {side} "
            f"qty={plan['quantity']} @ {retracement_price} "
            f"(trigger price was {plan['entry_price']})"
        )
        return {
            "ok": True, "shadow": True, "entry_order": None,
            "retracement_price": retracement_price,
            "retracement_timeout_seconds": timeout_seconds,
            "used_deep_retracement": use_deep,
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
        f"expires in {timeout_seconds}s -> market fallback"
    )

    return {
        "ok": True, "shadow": False, "entry_order": entry_order,
        "retracement_price": retracement_price,
        "retracement_timeout_seconds": timeout_seconds,
        "used_deep_retracement": use_deep,
    }
