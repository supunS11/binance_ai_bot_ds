import math

import config
from exchange import get_symbol_max_order_quantity, get_symbol_precision


MIN_NOTIONAL = 5.0


def get_position_risk_budget(balance):
    """Return the maximum planned dollar loss for one trade."""
    try:
        balance = max(float(balance or 0), 0)
        risk_pct = max(float(getattr(config, "POSITION_RISK_PCT", 0)), 0)
        risk_budget = balance * risk_pct / 100
        max_risk = max(float(getattr(config, "POSITION_RISK_MAX_USDT", 0)), 0)

        if max_risk > 0:
            risk_budget = min(risk_budget, max_risk)

        return max(risk_budget, 0)
    except Exception:
        return 0


def _round_quantity_down(quantity, precision):
    precision = max(int(precision or 0), 0)
    factor = 10 ** precision
    return math.floor(max(float(quantity or 0), 0) * factor) / factor


def calculate_position_size(
    balance,
    entry_price,
    sl_price,
    symbol,
    margin_override=None,
    risk_budget_override=None,
):
    """Size a position from the stop distance (the "1% rule"), falling back
    to a flat margin*leverage notional when risk-based sizing is off or the
    stop price isn't known yet."""
    try:
        margin = margin_override if margin_override is not None else config.MARGIN_PER_TRADE
        margin = max(float(margin or 0), 0)
        entry_price = float(entry_price or 0)

        if margin <= 0 or entry_price <= 0:
            return 0

        base_notional = margin * config.LEVERAGE
        quantity = base_notional / entry_price

        if getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False):
            stop_price = float(sl_price or 0)
            stop_distance = abs(entry_price - stop_price)
            risk_budget = (
                float(risk_budget_override)
                if risk_budget_override is not None
                else get_position_risk_budget(balance)
            )

            if stop_price <= 0 or stop_distance <= 0 or risk_budget <= 0:
                return 0

            quantity = min(quantity, risk_budget / stop_distance)

        max_notional = base_notional * 1.5
        max_qty = max_notional / entry_price
        quantity = min(quantity, max_qty)

        # Real exchange ceiling, independent of the margin-based sanity
        # cap above - a low-priced/volatile symbol (e.g. risk-based sizing
        # on XAIUSDT) can need a contract count that clears max_notional
        # comfortably but still exceeds Binance's own max order quantity.
        # Left unclamped here, the entry order silently gets cut down to
        # this same limit anyway at placement time (normalize_order_quantity)
        # while tp1_quantity/tp2_quantity (exact fractions of the
        # uncapped `quantity` returned from here) stay oversized and get
        # rejected later with -4005 - size against it up front so every
        # leg of the trade agrees on the same real position size.
        exchange_max_qty = get_symbol_max_order_quantity(symbol)

        if exchange_max_qty > 0:
            quantity = min(quantity, exchange_max_qty)

        precision = get_symbol_precision(symbol)
        quantity = _round_quantity_down(quantity, precision)

        if quantity * entry_price < MIN_NOTIONAL:
            return 0

        if quantity <= 0:
            return 0

        return quantity

    except Exception:
        return 0
