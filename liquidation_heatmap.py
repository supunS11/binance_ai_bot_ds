"""Real historical liquidation-density clustering - the free, real-data
alternative to a paid, modeled liquidation heatmap (see config.
LIQUIDATION_HEATMAP_ENABLED for the full rationale).

cluster_events mirrors market_structure.find_liquidity_pools' own equal-
price-tolerance clustering algorithm exactly, just fed real forced-
liquidation events (from liquidation_journal.load_events) instead of
swing points - so the output is the same {"type","price","touches"} shape
every existing pool-consuming function already understands
(risk_manager._find_dca_level/_find_structure_target/_next_pool_beyond_
stop, liquidity_sweep.detect_sweep), even though nothing wires this data
into any of them yet (see config.LIQUIDATION_HEATMAP_ENABLED - this ships
additive/journaled-only until real evidence justifies merging it in).

recompute_all/get_liquidation_pools is a periodically-refreshed cache
(see ws_client._liquidation_heatmap_recluster_loop), not computed per
eval tick - a multi-day, cross-symbol event log is too expensive to
rescan ~400 times per poll cycle.
"""
import threading
import time

import config
from logger import log_warning
import liquidation_journal


_lock = threading.RLock()
_cache = {}  # symbol -> [{"type","price","touches"}, ...]


def _side_to_type(side):
    # side="SELL" is a real forced LONG close (Binance/Bybit/OKX all
    # normalize to this convention already, see liquidation_tracker.py's
    # own docstring) - real evidence that long stops/liquidations
    # concentrate here, same concept market_structure.find_liquidity_
    # pools' own SELL_SIDE ("below the market - long stops") already
    # represents. side="BUY" (forced short close) mirrors to BUY_SIDE.
    return "SELL_SIDE" if side == "SELL" else "BUY_SIDE"


def cluster_events(events, tolerance_pct=None):
    """events: iterable of (side, price, notional) - real liquidation
    events for ONE symbol, any venue mix (see recompute_all - venues are
    deliberately merged before this is called, not clustered separately).

    Same walk-and-chain clustering as find_liquidity_pools: sort by
    price, chain a point into the running cluster while within
    tolerance_pct of the cluster's LAST-added price, flush once a point
    falls outside that band. Unlike find_liquidity_pools' plain
    arithmetic mean, `price` here is NOTIONAL-WEIGHTED - a real dollar
    weight exists for a liquidation event (unlike a swing point), so a
    $2M forced close should pull the level toward itself more than a
    $500 one. `touches` is the raw event count (not notional-weighted),
    matching find_liquidity_pools' own field exactly so it stays directly
    comparable to every existing touches consumer.

    A cluster only qualifies once it has >=2 events (same hardcoded
    floor find_liquidity_pools itself uses, not a new config knob) AND
    its combined notional clears config.LIQUIDATION_HEATMAP_CLUSTER_MIN_
    NOTIONAL_USDT."""
    tolerance_pct = float(
        config.LIQUIDATION_HEATMAP_CLUSTER_TOLERANCE_PCT if tolerance_pct is None else tolerance_pct
    )
    min_notional = max(float(config.LIQUIDATION_HEATMAP_CLUSTER_MIN_NOTIONAL_USDT), 0)
    pools = []

    for side, label in (("SELL", "SELL_SIDE"), ("BUY", "BUY_SIDE")):
        points = sorted(
            ((price, notional) for s, price, notional in events if s == side),
            key=lambda point: point[0],
        )
        cluster = []

        for point in points:
            if cluster and abs(point[0] - cluster[-1][0]) / cluster[-1][0] <= tolerance_pct:
                cluster.append(point)
                continue

            _flush(pools, label, cluster, min_notional)
            cluster = [point]

        _flush(pools, label, cluster, min_notional)

    return pools


def _flush(pools, label, cluster, min_notional):
    if len(cluster) < 2:
        return

    total_notional = sum(notional for _, notional in cluster)

    if total_notional < min_notional:
        return

    weighted_price = sum(price * notional for price, notional in cluster) / total_notional

    pools.append({"type": label, "price": weighted_price, "touches": len(cluster)})


def recompute_all(now=None):
    """One sequential read of the whole durable journal (liquidation_
    journal.load_events, since=now-LIQUIDATION_HEATMAP_RETENTION_SECONDS),
    grouped by symbol, reclustered in one pass, swaps the cache dict
    atomically. Venues are merged (not clustered per-exchange) - same
    "a cascade is invisible on one venue alone" rationale config.
    CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED already states, and the
    same source-agnostic merge market_structure.find_structure_candidates
    already does for pools/OB/FVG. Never raises - a failed recompute
    leaves the previous cache in place, same "don't let one bad cycle
    erase good data" discipline as every other periodic recompute in
    this codebase."""
    now = time.time() if now is None else now
    since = now - max(float(config.LIQUIDATION_HEATMAP_RETENTION_SECONDS), 0)

    try:
        events = liquidation_journal.load_events(since=since)
        by_symbol = {}

        for event in events:
            by_symbol.setdefault(event["symbol"], []).append(
                (event["side"], event["price"], event["notional"])
            )

        new_cache = {
            symbol: cluster_events(symbol_events)
            for symbol, symbol_events in by_symbol.items()
        }
    except Exception as exc:
        log_warning(f"liquidation heatmap recompute failed (keeping previous cache): {exc}")
        return

    with _lock:
        _cache.clear()
        _cache.update(new_cache)


def get_liquidation_pools(symbol):
    """O(1) cached read, safe every eval tick. [] until the first
    recompute_all() has run, or if this symbol has no qualifying real
    clusters yet."""
    with _lock:
        return list(_cache.get(symbol.upper(), []))


def reset_cache():
    """Test helper, mirrors liquidation_tracker.LiquidationEngine.reset()."""
    with _lock:
        _cache.clear()
