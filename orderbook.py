"""Real-time order-book depth imbalance + microprice from the depth stream.

Formula ported from v7/v8's market_intelligence.py::MarketFlowMonitor
(a proven-live component) and generalized so it isn't limited to that
module's fixed symbol list.
"""
import threading
import time

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
            state["updated_at"] = time.time() if timestamp is None else timestamp

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
            }

    def reset(self, symbol=None):
        with self.lock:
            if symbol is None:
                self._book.clear()
            else:
                self._book.pop(symbol.upper(), None)
