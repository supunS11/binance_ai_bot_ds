"""Real-time forced-liquidation clustering, via Binance's combined
`!forceOrder@arr` stream (every symbol, one connection - no per-symbol
depth/book reconciliation needed). Ported from v7's proven
liquidation_shadow.py, simplified to this bot's direct-lock engine style
(order_flow.py/orderbook.py) instead of its queue+worker-thread version,
which existed for a heavier multi-symbol shadow monitor than this bot's
single evaluate-on-tick usage needs.

See config.LIQUIDATION_CONFIRMATION_ENABLED for the informational-only
rationale; the BULLISH/BEARISH alignment check lives in signal_engine.py,
next to where sweep direction is already known.
"""
import threading
import time
from collections import deque

import config


LIQUIDATION_STREAM_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if result == result else default  # filters NaN
    except (TypeError, ValueError):
        return default


def _order_payload(message):
    if not isinstance(message, dict):
        return None

    data = message.get("data")
    data = data if isinstance(data, dict) else message
    order = data.get("o")
    return order if isinstance(order, dict) else None


class LiquidationEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self._events = {}  # symbol -> deque[(timestamp, side, notional)]

    def _series(self, symbol):
        series = self._events.get(symbol)

        if series is None:
            series = deque(maxlen=max(int(config.LIQUIDATION_MAX_EVENTS_PER_SYMBOL), 1))
            self._events[symbol] = series

        return series

    def record_liquidation(self, symbol, side, notional, timestamp=None):
        side = str(side or "").upper()

        if side not in ("BUY", "SELL") or notional is None or notional <= 0:
            return

        symbol = symbol.upper()
        timestamp = time.time() if timestamp is None else float(timestamp)

        with self.lock:
            self._series(symbol).append((timestamp, side, float(notional)))

    def handle_message(self, message):
        """Parse one raw `!forceOrder@arr` websocket message and record it.
        Never raises - a malformed message is simply dropped."""
        order = _order_payload(message)

        if not order:
            return

        symbol = str(order.get("s") or "")

        if not symbol:
            return

        side = order.get("S")
        price = _safe_float(order.get("ap")) or _safe_float(order.get("p"))
        quantity = _safe_float(order.get("z")) or _safe_float(order.get("q"))
        notional = abs(price * quantity)
        timestamp = _safe_float(order.get("T") or order.get("E"), time.time() * 1000) / 1000

        self.record_liquidation(symbol, side, notional, timestamp=timestamp)

    def snapshot(self, symbol, now=None):
        """`net_liquidation_notional` is long-liquidation notional (forced
        SELL orders - longs force-closed) minus short-liquidation notional
        (forced BUY orders) - positive means longs are being liquidated,
        negative means shorts are."""
        symbol = symbol.upper()
        now = time.time() if now is None else now
        window = max(int(config.LIQUIDATION_WINDOW_SECONDS), 1)
        cutoff = now - window

        with self.lock:
            events = list(self._events.get(symbol, ()))

        recent = [event for event in events if event[0] >= cutoff]

        if not recent:
            return {
                "available": False,
                "symbol": symbol,
                "sample_count": 0,
                "long_liquidation_notional": 0.0,
                "short_liquidation_notional": 0.0,
                "net_liquidation_notional": 0.0,
            }

        # SELL-side forced orders close out longs; BUY-side close out shorts.
        long_notional = sum(notional for _, side, notional in recent if side == "SELL")
        short_notional = sum(notional for _, side, notional in recent if side == "BUY")

        return {
            "available": True,
            "symbol": symbol,
            "sample_count": len(recent),
            "long_liquidation_notional": round(long_notional, 2),
            "short_liquidation_notional": round(short_notional, 2),
            "net_liquidation_notional": round(long_notional - short_notional, 2),
        }

    def reset(self, symbol=None):
        with self.lock:
            if symbol is None:
                self._events.clear()
            else:
                self._events.pop(symbol.upper(), None)
