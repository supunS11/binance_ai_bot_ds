"""Market-wide flash-crash detection off a single reference symbol's
real-time trade price (not the LTF kline - the whole real incident this
was built for fit inside part of one 1h candle, far too coarse to react
mid-move).

Real motivation (2026-08-22): a BTC flash-crash (~2.65% in ~4 minutes,
05:07-05:11 UTC) caught one open position on the wrong side mid-DCA -
it averaged in right at the bottom, then its real SL placement was
rejected by the exchange ("Order would immediately trigger" - price had
already blown through the level by the time the order was submitted),
forcing an emergency market-close that realized real slippage. Every
position on the OTHER side of the same move profited from it. Volume/
price were already clearly anomalous (15-20x normal) a full 2 minutes
before that damaging DCA fired - real, actionable lead time this module
exists to use, via config.CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED (new
entries) and config.CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED (forces
the existing DCA_PRESSURE_CHECK_ENABLED conservative path harder -
never skips the DCA itself, since DCA is what places a DCA_PENDING
position's first real SL in this project's design; skipping it would
leave the position naked, not safer).

Deliberately does its OWN transition logging (unlike CVDEngine/
DepthImbalanceEngine/OpenInterestEngine/LiquidationEngine, which are
pure data trackers with no logging of their own) - activation is a
rare, important, directly actionable event operators need to see
immediately, not a per-tick numeric update those other engines produce.
"""
import threading
import time
from collections import deque

import config
from logger import log_info, log_warning


class CrashDetector:
    def __init__(self):
        self.lock = threading.RLock()
        self._samples = deque()  # (timestamp, price), oldest first
        self._active = False
        self._direction = None
        self._pct_move = 0.0
        self._active_until = 0.0

    def record_price(self, price, timestamp=None):
        try:
            price = float(price)
        except (TypeError, ValueError):
            return

        if price <= 0:
            return

        timestamp = time.time() if timestamp is None else float(timestamp)
        window = max(float(config.CRASH_DETECTOR_WINDOW_SECONDS), 1)

        with self.lock:
            self._samples.append((timestamp, price))
            cutoff = timestamp - window

            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            self._evaluate(timestamp)

    def _evaluate(self, now):
        """Caller already holds self.lock. Drawdown-from-the-window's-own-
        high / runup-from-the-window's-own-low (not a naive first-to-last
        return) - reacts as the move is happening regardless of whether
        the extreme sits at the start, middle, or end of the window."""
        if len(self._samples) < 2:
            return

        prices = [p for _, p in self._samples]
        latest = prices[-1]
        high = max(prices)
        low = min(prices)

        drawdown_pct = ((high - latest) / high * 100) if high > 0 else 0.0
        runup_pct = ((latest - low) / low * 100) if low > 0 else 0.0
        threshold = max(float(config.CRASH_DETECTOR_MOVE_PCT), 0)

        if drawdown_pct >= threshold and drawdown_pct >= runup_pct:
            self._trigger("BEARISH", drawdown_pct, now)
        elif runup_pct >= threshold:
            self._trigger("BULLISH", runup_pct, now)

    def _trigger(self, direction, pct_move, now):
        """Caller already holds self.lock. Cooldown extends (not resets)
        on every repeat trigger while the move is still ongoing - a
        sustained crash keeps crash-mode continuously active instead of
        flickering off the moment one single window-check falls just
        under threshold."""
        cooldown = max(float(config.CRASH_DETECTOR_COOLDOWN_SECONDS), 0)
        is_new_activation = not self._active

        self._active = True
        self._pct_move = round(pct_move, 3)
        self._active_until = now + cooldown

        if is_new_activation or direction != self._direction:
            window = max(float(config.CRASH_DETECTOR_WINDOW_SECONDS), 1)
            log_warning(
                f"CRASH_DETECTOR active | direction={direction} "
                f"move={self._pct_move}% window={window}s "
                f"cooldown_until={round(self._active_until, 1)}"
            )

        self._direction = direction

    def snapshot(self, now=None):
        now = time.time() if now is None else now

        with self.lock:
            if self._active and now >= self._active_until:
                self._active = False
                self._direction = None
                self._pct_move = 0.0
                log_info("CRASH_DETECTOR cleared")

            return {
                "available": bool(self._samples),
                "active": self._active,
                "direction": self._direction,
                "pct_move": self._pct_move,
                "active_until": self._active_until if self._active else None,
                "window_seconds": max(float(config.CRASH_DETECTOR_WINDOW_SECONDS), 1),
                "sample_count": len(self._samples),
            }

    def reset(self):
        with self.lock:
            self._samples.clear()
            self._active = False
            self._direction = None
            self._pct_move = 0.0
            self._active_until = 0.0
