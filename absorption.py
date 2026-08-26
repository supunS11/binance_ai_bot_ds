"""Order-book absorption: real, one-sided aggressive volume that didn't
move price - someone (a large resting order, or several) absorbed it
without yielding the level.

Genuinely different from the two existing order-flow gates: cvd_score
(SIGNAL_MIN_CVD_SCORE) measures aggressor LEAN over a recent window,
blind to how much price actually moved for that volume; depth_imbalance
(SIGNAL_MIN_DEPTH_IMBALANCE) measures RESTING size at this instant, blind
to whether real flow is even testing it right now. This is the one thing
neither measures: did real volume meet real resistance. Uses data already
flowing for both of those (order_flow.CVDEngine's existing ratio_1m/
notional_1m, orderbook.DepthImbalanceEngine's short mid-price history) -
no new market-data subscription.

Informational only - see config.ABSORPTION_TRACKING_ENABLED's own comment
for why this isn't a gate yet (zero trade history exists on it by
construction; the same "diagnostic/journal first, gate later once real
evidence exists" convention every other field in this project follows).
"""
import config


def compute(cvd_snapshot, price_change_pct):
    """cvd_snapshot: order_flow.CVDEngine.snapshot() output.
    price_change_pct: orderbook.DepthImbalanceEngine.snapshot()'s
    "price_change_pct_1m" - None if not enough price history has been
    retained yet (tracking just (re)started).

    Requires real, one-sided aggressor volume (ratio_1m magnitude and
    notional_1m both clearing floors - the same "is this reading even
    trustworthy" question ORDER_FLOW_MIN_NOTIONAL_USDT already answers
    for cvd_score) AND price barely moving despite it. Direction is the
    OPPOSITE of the aggressor flow: aggressive SELLING that didn't push
    price down means buyers absorbed it - bullish information, not
    bearish. Returns "BUY"/"SELL"/None."""
    if not cvd_snapshot or price_change_pct is None:
        return None

    ratio = cvd_snapshot.get("ratio_1m")
    notional = cvd_snapshot.get("notional_1m")

    if ratio is None or notional is None:
        return None

    min_notional = max(float(config.ORDER_FLOW_MIN_NOTIONAL_USDT), 0)
    min_ratio = max(float(config.ABSORPTION_MIN_CVD_RATIO), 0)
    max_move_pct = max(float(config.ABSORPTION_MAX_PRICE_MOVE_PCT), 0)

    if notional < min_notional or abs(ratio) < min_ratio:
        return None

    if abs(price_change_pct) > max_move_pct:
        return None

    return "BUY" if ratio < 0 else "SELL"
