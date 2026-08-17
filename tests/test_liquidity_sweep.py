import unittest
from unittest.mock import patch

import config
import liquidity_sweep as ls


def _candle(high, low, close, open_time=0, closed=True):
    return {"open_time": open_time, "open": close, "high": high, "low": low, "close": close, "volume": 1, "closed": closed}


class DetectSweepTests(unittest.TestCase):
    def test_wick_above_buy_side_pool_then_reject_is_bearish(self):
        candles = [_candle(high=105, low=99, close=100)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools)

        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["direction"], "BEARISH")
        self.assertAlmostEqual(sweep["wick_size"], 2)

    def test_wick_below_sell_side_pool_then_reject_is_bullish(self):
        candles = [_candle(high=101, low=95, close=100)]
        pools = [{"type": "SELL_SIDE", "price": 97, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools)

        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["direction"], "BULLISH")

    def test_wick_through_without_reject_is_not_a_sweep(self):
        # closes beyond the level -> real breakout, not a stop-hunt reject
        candles = [_candle(high=106, low=99, close=105)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools)
        self.assertIsNone(sweep)

    def test_no_pools_returns_none(self):
        candles = [_candle(high=105, low=99, close=100)]
        self.assertIsNone(ls.detect_sweep(candles, []))

    def test_no_candles_returns_none(self):
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]
        self.assertIsNone(ls.detect_sweep([], pools))

    def test_sweep_result_includes_the_tested_candles_open_time(self):
        candles = [_candle(high=105, low=99, close=100, open_time=456)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools)

        self.assertEqual(sweep["open_time"], 456)


class RequireClosedCandleTests(unittest.TestCase):
    """config.REQUIRE_CLOSE_CONFIRMED_BREAK - real motivation (2026-08-13,
    two live trades traced against actual Binance price history): a sweep
    read against a still-forming candle can flip before the candle
    actually finishes, and both traced trades entered on exactly that kind
    of premature read before immediately reversing. Reuses the same flag
    market_structure.live_break_check already uses - the same principle,
    applied uniformly."""

    def test_forming_candles_sweep_is_ignored_when_required(self):
        candles = [_candle(high=105, low=99, close=100, closed=False)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools, require_closed_candle=True)

        self.assertIsNone(sweep)

    def test_fires_once_the_sweeping_candle_closes(self):
        candles = [
            _candle(high=101, low=99, close=100, open_time=0, closed=True),
            _candle(high=105, low=99, close=100, open_time=1, closed=True),
        ]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools, require_closed_candle=True)

        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["direction"], "BEARISH")
        self.assertEqual(sweep["open_time"], 1)

    def test_ignores_a_forming_candle_even_if_an_earlier_closed_one_exists(self):
        # The forming candle (open_time=1) sweeps the pool, but the last
        # CLOSED candle (open_time=0) never did - must not fire on the
        # forming one just because it's last in the list.
        candles = [
            _candle(high=101, low=99, close=100, open_time=0, closed=True),
            _candle(high=105, low=99, close=100, open_time=1, closed=False),
        ]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools, require_closed_candle=True)

        self.assertIsNone(sweep)

    def test_no_closed_candle_at_all_returns_none(self):
        candles = [_candle(high=105, low=99, close=100, closed=False)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        self.assertIsNone(ls.detect_sweep(candles, pools, require_closed_candle=True))

    def test_defaults_from_config(self):
        candles = [_candle(high=105, low=99, close=100, closed=False)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        with patch.object(config, "REQUIRE_CLOSE_CONFIRMED_BREAK", True):
            sweep = ls.detect_sweep(candles, pools)

        self.assertIsNone(sweep)

    def test_require_closed_candle_false_checks_the_forming_candle(self):
        candles = [_candle(high=105, low=99, close=100, closed=False)]
        pools = [{"type": "BUY_SIDE", "price": 103, "touches": 2}]

        sweep = ls.detect_sweep(candles, pools, require_closed_candle=False)

        self.assertIsNotNone(sweep)


_SWEEP = {"direction": "BULLISH", "level": 97, "wick_size": 2, "pool": {}, "open_time": 1}


def _liquidation_snapshot(long_notional=0, short_notional=0, available=True):
    return {
        "available": available,
        "long_liquidation_notional": long_notional,
        "short_liquidation_notional": short_notional,
        "net_liquidation_notional": long_notional - short_notional,
    }


class DetectLiquidationConfirmedSweepTests(unittest.TestCase):
    """config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED - promotes a
    plain sweep into a stricter trigger by additionally requiring a real
    clustered forced-liquidation event, aligned with the sweep's
    direction (same alignment formula signal_engine.py's own informational
    liquidation_aligned field already uses)."""

    def test_none_sweep_returns_none(self):
        self.assertIsNone(
            ls.detect_liquidation_confirmed_sweep(None, _liquidation_snapshot(long_notional=100000))
        )

    def test_unavailable_liquidation_snapshot_returns_none(self):
        result = ls.detect_liquidation_confirmed_sweep(
            _SWEEP, _liquidation_snapshot(long_notional=100000, available=False)
        )
        self.assertIsNone(result)

    def test_none_liquidation_snapshot_returns_none(self):
        self.assertIsNone(ls.detect_liquidation_confirmed_sweep(_SWEEP, None))

    def test_total_notional_below_min_notional_returns_none(self):
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = ls.detect_liquidation_confirmed_sweep(
                _SWEEP, _liquidation_snapshot(long_notional=10000)
            )
        self.assertIsNone(result)

    def test_bullish_sweep_requires_positive_net_long_liquidations(self):
        # BULLISH sweep + short liquidations dominating (net < 0) -> not aligned.
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = ls.detect_liquidation_confirmed_sweep(
                _SWEEP, _liquidation_snapshot(long_notional=10000, short_notional=90000)
            )
        self.assertIsNone(result)

    def test_bullish_sweep_with_aligned_long_liquidation_cluster_passes(self):
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = ls.detect_liquidation_confirmed_sweep(
                _SWEEP, _liquidation_snapshot(long_notional=90000, short_notional=10000)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 97)
        self.assertEqual(result["open_time"], 1)

    def test_bearish_sweep_requires_negative_net_short_liquidations(self):
        bearish_sweep = dict(_SWEEP, direction="BEARISH")

        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = ls.detect_liquidation_confirmed_sweep(
                bearish_sweep, _liquidation_snapshot(long_notional=90000, short_notional=10000)
            )
        self.assertIsNone(result)

    def test_bearish_sweep_with_aligned_short_liquidation_cluster_passes(self):
        bearish_sweep = dict(_SWEEP, direction="BEARISH")

        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = ls.detect_liquidation_confirmed_sweep(
                bearish_sweep, _liquidation_snapshot(long_notional=10000, short_notional=90000)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")

    def test_min_notional_defaults_from_config(self):
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 200000):
            result = ls.detect_liquidation_confirmed_sweep(
                _SWEEP, _liquidation_snapshot(long_notional=90000, short_notional=10000)
            )
        self.assertIsNone(result)

    def test_explicit_min_notional_overrides_config(self):
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 0):
            result = ls.detect_liquidation_confirmed_sweep(
                _SWEEP, _liquidation_snapshot(long_notional=90000, short_notional=10000),
                min_notional_usdt=200000,
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
