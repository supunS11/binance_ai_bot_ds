"""Cumulative volume delta (CVD) from the real-time aggTrade tape.

This is the order-flow confirmation layer: SMC structure (market_structure.py,
built in Phase 2) tells us *where* a setup is; CVD tells us whether real
aggressive buying/selling is actually behind it, which is what separates a
genuine order-block/FVG signal from a false one instead of just waiting for
a candle to close and hoping.
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


class CVDEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self._trades = {}  # symbol -> deque[(timestamp, signed_notional, notional)]
        self._cumulative = {}  # symbol -> running cumulative signed notional (the CVD line)
        self._cvd_history = {}  # symbol -> deque[{"open_time","cumulative_cvd"}], one per LTF candle close

    def _series(self, symbol):
        series = self._trades.get(symbol)

        if series is None:
            series = deque()
            self._trades[symbol] = series

        return series

    def record_trade(self, symbol, price, quantity, is_buyer_maker, timestamp=None):
        """`is_buyer_maker=True` means the resting order was a buy and the
        aggressor was a seller (Binance's aggTrade convention) - so a
        buyer-maker trade is signed negative (sell pressure) here."""
        price = _safe_float(price)
        quantity = _safe_float(quantity)

        if price <= 0 or quantity <= 0:
            return

        symbol = symbol.upper()
        timestamp = time.time() if timestamp is None else float(timestamp)
        notional = price * quantity
        signed_notional = -notional if is_buyer_maker else notional
        max_window = max(int(config.ORDER_FLOW_MAX_WINDOW_SECONDS), 60)

        with self.lock:
            series = self._series(symbol)
            series.append((timestamp, signed_notional, notional))
            cutoff = timestamp - max_window

            while series and series[0][0] < cutoff:
                series.popleft()

            self._cumulative[symbol] = self._cumulative.get(symbol, 0.0) + signed_notional

    def finalize_candle(self, symbol, open_time):
        """Snapshot the cumulative CVD line at a price-candle-aligned
        timestamp - called once per LTF candle close (see ws_client.
        _handle_kline's kline["x"] flag). This is what makes swing-point
        CVD divergence possible (cvd_divergence.py): the raw trade deque
        above is deliberately pruned to a short recent window
        (config.ORDER_FLOW_MAX_WINDOW_SECONDS) for the ratio/cvd_score
        snapshot below, and can't span the multiple candles/swings a real
        divergence comparison needs - this is a separate, much cheaper
        series (one float per candle, not per trade) retained far longer
        (config.CVD_HISTORY_MAXLEN)."""
        symbol = symbol.upper()

        with self.lock:
            history = self._cvd_history.get(symbol)

            if history is None:
                history = deque(maxlen=max(int(config.CVD_HISTORY_MAXLEN), 10))
                self._cvd_history[symbol] = history

            history.append({
                "open_time": open_time,
                "cumulative_cvd": self._cumulative.get(symbol, 0.0),
            })

    def cvd_history(self, symbol):
        symbol = symbol.upper()

        with self.lock:
            return list(self._cvd_history.get(symbol, ()))

    def snapshot(self, symbol, windows=(60, 300, 900), now=None):
        symbol = symbol.upper()
        now = time.time() if now is None else now
        min_notional = max(float(config.ORDER_FLOW_MIN_NOTIONAL_USDT), 0)

        with self.lock:
            series = list(self._trades.get(symbol, ()))

        if not series:
            return {"available": False, "symbol": symbol, "history": self.cvd_history(symbol)}

        ratios = {}
        notionals = {}

        for window in windows:
            cutoff = now - window
            selected = [trade for trade in series if trade[0] >= cutoff]
            total = sum(trade[2] for trade in selected)
            signed = sum(trade[1] for trade in selected)
            ratios[window] = (signed / total) if total >= min_notional else None
            notionals[window] = total

        weighted_parts = [
            (ratios[window], weight)
            for window, weight in zip(windows, (0.5, 0.3, 0.2))
            if ratios.get(window) is not None
        ]
        weight_total = sum(weight for _, weight in weighted_parts)
        cvd_score = (
            sum(value * weight for value, weight in weighted_parts) / weight_total
            if weight_total > 0
            else None
        )

        return {
            "available": cvd_score is not None,
            "symbol": symbol,
            "cvd_score": round(cvd_score, 4) if cvd_score is not None else None,
            "ratio_1m": ratios.get(windows[0]),
            "ratio_5m": ratios.get(windows[1]) if len(windows) > 1 else None,
            "ratio_15m": ratios.get(windows[2]) if len(windows) > 2 else None,
            "notional_1m": round(notionals.get(windows[0], 0), 2),
            "sample_count": len(series),
            "history": self.cvd_history(symbol),
        }

    def reset(self, symbol=None):
        with self.lock:
            if symbol is None:
                self._trades.clear()
                self._cumulative.clear()
                self._cvd_history.clear()
            else:
                symbol = symbol.upper()
                self._trades.pop(symbol, None)
                self._cumulative.pop(symbol, None)
                self._cvd_history.pop(symbol, None)
