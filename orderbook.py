"""Real-time order-book depth imbalance + microprice from the depth stream.

Formula ported from v7/v8's market_intelligence.py::MarketFlowMonitor
(a proven-live component) and generalized so it isn't limited to that
module's fixed symbol list.
"""
import threading
import time
from collections import deque

import config


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value, minimum=-1.0, maximum=1.0):
    return min(max(float(value), minimum), maximum)


class DepthImbalanceEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self._book = {}  # symbol -> {imbalance, microprice_bps, samples, updated_at}

    def _state(self, symbol):
        state = self._book.get(symbol)

        if state is None:
            state = {
                "imbalance": 0.0,
                "microprice_bps": 0.0,
                "samples": 0,
                "updated_at": 0.0,
                "best_bid": 0.0,
                "best_ask": 0.0,
                # config.ABSORPTION_TRACKING_ENABLED - a short rolling
                # (timestamp, mid_price) series, pruned by ABSORPTION_
                # PRICE_HISTORY_SECONDS, independent of imbalance's own EMA
                # smoothing above. Nothing else in this codebase retains a
                # sub-candle price history - absorption.py needs "how much
                # did price actually move in the last ~60s", which candles
                # (1h) are far too coarse for.
                "mid_price_history": deque(),
            }
            self._book[symbol] = state

        return state

    def record_depth(self, symbol, bids, asks, timestamp=None):
        """`bids`/`asks` are lists of `[price, quantity]` pairs (top-of-book
        first), matching Binance's partial-depth/bookTicker payload shape."""
        symbol = symbol.upper()

        if not bids or not asks:
            return

        bid_notional = sum(
            _safe_float(level[0]) * _safe_float(level[1])
            for level in bids
            if len(level) >= 2
        )
        ask_notional = sum(
            _safe_float(level[0]) * _safe_float(level[1])
            for level in asks
            if len(level) >= 2
        )
        total_notional = bid_notional + ask_notional

        if total_notional <= 0:
            return

        imbalance = (bid_notional - ask_notional) / total_notional
        best_bid = _safe_float(bids[0][0])
        best_bid_qty = _safe_float(bids[0][1])
        best_ask = _safe_float(asks[0][0])
        best_ask_qty = _safe_float(asks[0][1])
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        quantity_total = best_bid_qty + best_ask_qty
        microprice = (
            ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) / quantity_total
            if quantity_total > 0
            else mid
        )
        microprice_bps = ((microprice - mid) / mid) * 10000 if mid else 0
        alpha = _clip(0.15, 0.01, 1.0)

        with self.lock:
            state = self._state(symbol)

            if state["samples"]:
                state["imbalance"] = (alpha * imbalance) + ((1 - alpha) * state["imbalance"])
                state["microprice_bps"] = (
                    (alpha * microprice_bps) + ((1 - alpha) * state["microprice_bps"])
                )
            else:
                state["imbalance"] = imbalance
                state["microprice_bps"] = microprice_bps

            state["samples"] += 1
            state["best_bid"] = best_bid
            state["best_ask"] = best_ask
            recorded_at = time.time() if timestamp is None else timestamp
            state["updated_at"] = recorded_at

            if mid:
                history = state["mid_price_history"]
                history.append((recorded_at, mid))
                cutoff = recorded_at - max(float(config.ABSORPTION_PRICE_HISTORY_SECONDS), 10)

                while history and history[0][0] < cutoff:
                    history.popleft()

    def _price_change_pct(self, state, now):
        """config.ABSORPTION_TRACKING_ENABLED - % change from the oldest
        retained mid-price sample at or before `now - ABSORPTION_WINDOW_
        SECONDS` to the current mid - same "latest sample at or before a
        target time" search as oi_divergence._oi_at_or_before, applied to
        a continuously-retained short series instead of a periodic poll.
        None (not False/0) whenever there isn't yet a real reference point
        old enough - a freshly (re)started engine must never report a
        manufactured 0% move as if it were a real reading. Assumes the
        caller already holds self.lock (only called from within
        snapshot() below)."""
        history = state["mid_price_history"]

        if not history:
            return None

        window_seconds = max(float(config.ABSORPTION_WINDOW_SECONDS), 1)
        cutoff = now - window_seconds
        current = history[-1][1]
        reference = None

        for sample_time, mid in history:
            if sample_time <= cutoff:
                reference = mid
            else:
                break

        if reference is None or reference <= 0:
            return None

        return (current - reference) / reference * 100

    def snapshot(self, symbol, now=None):
        symbol = symbol.upper()
        now = time.time() if now is None else now
        stale_seconds = max(float(config.WS_STALE_SECONDS), 5)

        with self.lock:
            state = self._book.get(symbol)

            if not state or not state["samples"]:
                return {"available": False, "symbol": symbol}

            age = now - state["updated_at"]
            fresh = age <= stale_seconds

            return {
                "available": fresh,
                "symbol": symbol,
                "depth_imbalance": round(state["imbalance"], 4),
                "microprice_bps": round(state["microprice_bps"], 4),
                "best_bid": state["best_bid"],
                "best_ask": state["best_ask"],
                "samples": state["samples"],
                "age_seconds": round(age, 1),
                "price_change_pct_1m": self._price_change_pct(state, now),
            }

    def reset(self, symbol=None):
        with self.lock:
            if symbol is None:
                self._book.clear()
            else:
                self._book.pop(symbol.upper(), None)
