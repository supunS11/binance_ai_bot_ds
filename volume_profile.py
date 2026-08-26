"""Intraday volume-profile zones from the real-time aggTrade tape - see
config.VOLUME_PROFILE_TRACKING_ENABLED for the informational-only
rationale.

Retains raw (timestamp, price, quantity) samples - the first thing in
this codebase to keep per-trade PRICE. order_flow.CVDEngine.record_trade
reduces straight to notional (price * quantity) and discards price
entirely, which is sufficient for a net order-flow read but not for a
price-bucketed histogram, which is what this module builds instead.
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


class VolumeProfileEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self._trades = {}  # symbol -> deque[(timestamp, price, quantity)]

    def _series(self, symbol):
        series = self._trades.get(symbol)

        if series is None:
            series = deque(maxlen=max(int(config.VOLUME_PROFILE_MAX_SAMPLES), 2))
            self._trades[symbol] = series

        return series

    def record_trade(self, symbol, price, quantity, timestamp=None):
        price = _safe_float(price)
        quantity = _safe_float(quantity)

        if price <= 0 or quantity <= 0:
            return

        symbol = symbol.upper()
        timestamp = time.time() if timestamp is None else float(timestamp)
        max_window = max(int(config.VOLUME_PROFILE_LOOKBACK_SECONDS), 60)

        with self.lock:
            series = self._series(symbol)
            series.append((timestamp, price, quantity))
            cutoff = timestamp - max_window

            while series and series[0][0] < cutoff:
                series.popleft()

    def snapshot(self, symbol, now=None):
        symbol = symbol.upper()
        now = time.time() if now is None else now
        lookback = max(int(config.VOLUME_PROFILE_LOOKBACK_SECONDS), 60)

        with self.lock:
            series = list(self._trades.get(symbol, ()))

        cutoff = now - lookback
        samples = [(ts, price, qty) for ts, price, qty in series if ts >= cutoff]

        if not samples:
            return {
                "available": False,
                "symbol": symbol,
                "poc_price": None,
                "value_area_high": None,
                "value_area_low": None,
                "position": None,
                "sample_count": 0,
            }

        latest_price = samples[-1][1]
        bucket_pct = max(float(config.VOLUME_PROFILE_BUCKET_PCT), 0.001)
        bucket_size = latest_price * bucket_pct / 100

        if bucket_size <= 0:
            return {
                "available": False,
                "symbol": symbol,
                "poc_price": None,
                "value_area_high": None,
                "value_area_low": None,
                "position": None,
                "sample_count": len(samples),
            }

        buckets = {}

        for _, price, qty in samples:
            bucket_key = round(price / bucket_size)
            buckets[bucket_key] = buckets.get(bucket_key, 0.0) + price * qty

        ordered_keys = sorted(buckets.keys())
        notionals = [buckets[key] for key in ordered_keys]
        total_notional = sum(notionals)
        poc_index = notionals.index(max(notionals))

        low = high = poc_index
        covered = notionals[poc_index]
        target = total_notional * max(float(config.VOLUME_PROFILE_VALUE_AREA_PCT), 0.0) / 100

        while covered < target and (low > 0 or high < len(ordered_keys) - 1):
            below = notionals[low - 1] if low > 0 else -1
            above = notionals[high + 1] if high < len(ordered_keys) - 1 else -1

            if below >= above:
                low -= 1
                covered += notionals[low]
            else:
                high += 1
                covered += notionals[high]

        poc_price = ordered_keys[poc_index] * bucket_size
        value_area_high = ordered_keys[high] * bucket_size
        value_area_low = ordered_keys[low] * bucket_size

        if latest_price > value_area_high:
            position = "ABOVE_VALUE_AREA"
        elif latest_price < value_area_low:
            position = "BELOW_VALUE_AREA"
        else:
            position = "INSIDE_VALUE_AREA"

        return {
            "available": True,
            "symbol": symbol,
            "poc_price": poc_price,
            "value_area_high": value_area_high,
            "value_area_low": value_area_low,
            "position": position,
            "sample_count": len(samples),
        }

    def reset(self, symbol=None):
        with self.lock:
            if symbol is None:
                self._trades.clear()
            else:
                self._trades.pop(symbol.upper(), None)
