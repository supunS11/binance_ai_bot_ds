import unittest
from unittest.mock import patch

import config
import liquidity_sweep as ls

# config.LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE / _SELECT_NEAREST_POOL_ENABLED
# - pinned inert for the whole module. The live .env sets both, and every
# existing DetectSweepTests fixture was written against the original
# "first pool in list order, any wick size" behaviour. Same insulation
# pattern tests/test_risk_manager.py and tests/test_market_structure.py
# already use. The flags' own behaviour is covered by
# SweepWickThresholdTests / SweepPoolSelectionTests below, which enable
# them locally.
_wick_patcher = patch.object(config, "LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE", 0.0)
_nearest_patcher = patch.object(config, "LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED", False)


def setUpModule():
    _wick_patcher.start()
    _nearest_patcher.start()


def tearDownModule():
    _wick_patcher.stop()
    _nearest_patcher.stop()


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


class SweepWickThresholdTests(unittest.TestCase):
    """config.LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE (2026-09-22) -
    wick_size was computed on every sweep from the start and never read by
    anything, so a one-tick poke through a pool scored identically to a
    real stop run. Measured live: the p10 sweep penetrated by 0.052 ATR."""

    POOLS = [{"type": "SELL_SIDE", "price": 100.0, "touches": 2}]

    def _sweep(self, low, **kwargs):
        # BULLISH: wick below the pool, close back above it.
        return ls.detect_sweep(
            [_candle(high=105, low=low, close=101)], self.POOLS, **kwargs)

    def test_zero_threshold_is_byte_identical_to_today(self):
        # A 0.01 wick on a 10.0 ATR - pure noise, but accepted, because that
        # is exactly what the detector did before this parameter existed.
        self.assertIsNotNone(self._sweep(99.99, min_wick_atr_multiple=0.0, atr=10.0))

    def test_a_wick_below_the_threshold_is_dropped(self):
        # wick 0.01, requirement 0.10 * 10.0 = 1.0
        self.assertIsNone(self._sweep(99.99, min_wick_atr_multiple=0.10, atr=10.0))

    def test_a_wick_above_the_threshold_is_kept(self):
        # wick 2.0, requirement 1.0
        result = self._sweep(98.0, min_wick_atr_multiple=0.10, atr=10.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertAlmostEqual(result["wick_size"], 2.0)

    def test_exactly_at_the_threshold_is_kept(self):
        # wick 1.0, requirement 0.10 * 10.0 = 1.0 - the test is >=
        self.assertIsNotNone(self._sweep(99.0, min_wick_atr_multiple=0.10, atr=10.0))

    def test_missing_or_zero_atr_fails_open(self):
        # The sweep is still a sweep; we just cannot judge its size. Never
        # silently kill the trigger on missing data.
        for atr in (None, 0, 0.0):
            with self.subTest(atr=atr):
                self.assertIsNotNone(
                    self._sweep(99.99, min_wick_atr_multiple=0.10, atr=atr))

    def test_reads_the_config_value_when_not_passed(self):
        with patch.object(config, "LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE", 0.10):
            self.assertIsNone(self._sweep(99.99, atr=10.0))

        with patch.object(config, "LIQUIDITY_SWEEP_MIN_WICK_ATR_MULTIPLE", 0.0):
            self.assertIsNotNone(self._sweep(99.99, atr=10.0))


class SweepPoolSelectionTests(unittest.TestCase):
    """config.LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED (2026-09-22) -
    detect_sweep returned the first pool in list order, and
    find_liquidity_pools emits every BUY_SIDE pool before every SELL_SIDE
    one. On a candle sweeping both sides the BEARISH read therefore won by
    enumeration order alone. Measured live: 7.6% of sweeping candles swept
    more than one pool, 2.5% swept pools in conflicting directions."""

    def test_first_in_list_wins_by_default(self):
        pools = [{"type": "SELL_SIDE", "price": 90.0, "touches": 2},
                 {"type": "SELL_SIDE", "price": 99.0, "touches": 2}]
        # Candle wicks below both and closes above both.
        result = ls.detect_sweep(
            [_candle(high=105, low=89.0, close=100)], pools, select_nearest=False)

        self.assertAlmostEqual(result["level"], 90.0)

    def test_nearest_to_close_wins_when_enabled(self):
        pools = [{"type": "SELL_SIDE", "price": 90.0, "touches": 2},
                 {"type": "SELL_SIDE", "price": 99.0, "touches": 2}]
        result = ls.detect_sweep(
            [_candle(high=105, low=89.0, close=100)], pools, select_nearest=True)

        self.assertAlmostEqual(result["level"], 99.0)

    def test_conflicting_directions_no_longer_resolve_by_list_order(self):
        # BUY_SIDE is enumerated first, so today this is BEARISH regardless
        # of which level price is actually nearer.
        pools = [{"type": "BUY_SIDE", "price": 104.0, "touches": 2},
                 {"type": "SELL_SIDE", "price": 99.5, "touches": 2}]
        candle = _candle(high=106, low=99.0, close=100.0)

        first = ls.detect_sweep([candle], pools, select_nearest=False)
        nearest = ls.detect_sweep([candle], pools, select_nearest=True)

        self.assertEqual(first["direction"], "BEARISH")     # by enumeration
        self.assertEqual(nearest["direction"], "BULLISH")   # by distance
        self.assertAlmostEqual(nearest["level"], 99.5)

    def test_selection_runs_after_the_wick_filter(self):
        # Needs pools on OPPOSITE sides: for two same-side pools the nearer
        # one always has the LARGER wick, so the filter can never remove a
        # selection winner there.
        #
        # BUY_SIDE 100.5 - wick 0.1 (noise), but only 0.5 from the close.
        # SELL_SIDE 90.0 - wick 5.0 (real),  but 10.0 from the close.
        pools = [{"type": "BUY_SIDE", "price": 100.5, "touches": 2},
                 {"type": "SELL_SIDE", "price": 90.0, "touches": 2}]
        candle = _candle(high=100.6, low=85.0, close=100.0)

        # Unfiltered, the nearest pool wins and the trade is a SHORT.
        unfiltered = ls.detect_sweep([candle], pools, select_nearest=True)
        self.assertEqual(unfiltered["direction"], "BEARISH")
        self.assertAlmostEqual(unfiltered["level"], 100.5)

        # Filtered, that pool's 0.1 wick is noise and drops out - so the
        # surviving sweep is returned instead of nothing, and the trade
        # direction FLIPS. Worth knowing: the wick threshold is not purely
        # subtractive, it can change which side the signal takes.
        filtered = ls.detect_sweep(
            [candle], pools,
            select_nearest=True, min_wick_atr_multiple=0.10, atr=10.0,
        )
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered["direction"], "BULLISH")
        self.assertAlmostEqual(filtered["level"], 90.0)

    def test_no_matching_pool_is_still_none(self):
        pools = [{"type": "SELL_SIDE", "price": 50.0, "touches": 2}]
        for nearest in (False, True):
            with self.subTest(select_nearest=nearest):
                self.assertIsNone(ls.detect_sweep(
                    [_candle(high=105, low=99.0, close=100)], pools,
                    select_nearest=nearest))

    def test_reads_the_config_value_when_not_passed(self):
        pools = [{"type": "SELL_SIDE", "price": 90.0, "touches": 2},
                 {"type": "SELL_SIDE", "price": 99.0, "touches": 2}]
        candle = _candle(high=105, low=89.0, close=100)

        with patch.object(config, "LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED", True):
            self.assertAlmostEqual(ls.detect_sweep([candle], pools)["level"], 99.0)

        with patch.object(config, "LIQUIDITY_SWEEP_SELECT_NEAREST_POOL_ENABLED", False):
            self.assertAlmostEqual(ls.detect_sweep([candle], pools)["level"], 90.0)


if __name__ == "__main__":
    unittest.main()
