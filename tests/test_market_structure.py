import unittest
from unittest.mock import patch

import config
import market_structure as ms

# config.ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT - pinned inert (0.0) for
# the whole module. Every FindOrderBlockRetest* assertion below predates
# this flag and was written against the old "close anywhere inside the
# block" condition; the live .env sets 0.5, which would otherwise silently
# break them the moment it is deployed. Same insulation pattern
# tests/test_risk_manager.py already uses for SL_TP_USE_HTF_STRUCTURE, and
# the same "new flag, old tests never pin it, live .env eventually sets it"
# cycle that has caught this suite out before. The dedicated threshold
# tests (OrderBlockRetestCloseThroughTests) set it locally instead.
_ob_close_through_patcher = patch.object(
    config, "ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT", 0.0
)
# config.EMA_PULLBACK_REQUIRE_TREND_ENABLED (live .env: True) and
# config.LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED (live .env: True) -
# pinned inert for the same reason. Every DetectEmaPullbackTests fixture
# predates the trend condition and builds candles with no EMA history to
# slope, and every FindLiquidityPoolsTests fixture was written against
# chained clustering. Their own behaviour is covered by
# EmaPullbackTrendTests / AnchoredPoolClusteringTests, which enable the
# flags locally.
_ema_trend_patcher = patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", False)
_anchored_pools_patcher = patch.object(
    config, "LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED", False
)


def setUpModule():
    _ob_close_through_patcher.start()
    _ema_trend_patcher.start()
    _anchored_pools_patcher.start()


def tearDownModule():
    _ob_close_through_patcher.stop()
    _ema_trend_patcher.stop()
    _anchored_pools_patcher.stop()


def _candle(open_time, high, low, close=None, open_=None, closed=True):
    close = high if close is None else close
    open_ = low if open_ is None else open_
    return {
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "closed": closed,
    }


class FindSwingPointsTests(unittest.TestCase):
    def test_detects_a_single_high_and_low_fractal(self):
        # left=right=1: index i is a pivot if it's the extreme of [i-1,i+1]
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=14, low=5),  # pivot HIGH and pivot LOW
            _candle(2, high=10, low=9),
        ]
        swings = ms.find_swing_points(candles, left=1, right=1)
        kinds = {(s.index, s.kind, s.price) for s in swings}

        self.assertIn((1, "HIGH", 14), kinds)
        self.assertIn((1, "LOW", 5), kinds)

    def test_edges_without_full_window_produce_no_swings(self):
        candles = [_candle(0, high=10, low=9), _candle(1, high=12, low=8)]
        swings = ms.find_swing_points(candles, left=1, right=1)
        self.assertEqual(swings, [])


class ZigzagTests(unittest.TestCase):
    def test_consecutive_same_kind_highs_collapse_to_the_higher_one(self):
        swings = [
            ms.SwingPoint(0, 0, 10, "HIGH"),
            ms.SwingPoint(1, 1, 15, "HIGH"),
        ]
        collapsed = ms._zigzag(swings)

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0].price, 15)

    def test_consecutive_same_kind_lows_collapse_to_the_lower_one(self):
        swings = [
            ms.SwingPoint(0, 0, 10, "LOW"),
            ms.SwingPoint(1, 1, 5, "LOW"),
        ]
        collapsed = ms._zigzag(swings)

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0].price, 5)


class ClassifySwingsTests(unittest.TestCase):
    """Hand-built alternating swing sequences, so BOS/CHoCH classification
    is tested directly against known trend shapes rather than depending on
    precisely engineering fractal candle data."""

    def _swing(self, index, price, kind):
        return ms.SwingPoint(index, index, price, kind)

    def test_higher_high_after_higher_low_is_bullish_bos(self):
        swings = [
            self._swing(1, 8, "LOW"),
            self._swing(3, 12, "HIGH"),   # first high -> BOS (trend was None)
            self._swing(5, 9, "LOW"),     # higher low, no trend flip
            self._swing(7, 14, "HIGH"),   # higher high -> continuation BOS
        ]
        result = ms._classify_swings(swings)

        self.assertTrue(result["available"])
        self.assertEqual(result["trend"], "BULLISH")
        self.assertEqual(result["last_event"]["type"], "BOS")
        self.assertEqual(result["last_swing_high"], 14)
        self.assertEqual(result["last_swing_low"], 9)
        # events: every BOS/CHoCH found, not just last_event - backs
        # find_structure_events/find_order_blocks (ORDER_BLOCK_RETEST_
        # TRIGGER_ENABLED). Only 1 here: the first HIGH/LOW never produce
        # an event (nothing prior to compare against), only the final
        # HIGH(14) exceeding HIGH(12) does.
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][-1], result["last_event"])

    def test_lower_low_after_bullish_trend_is_choch(self):
        swings = [
            self._swing(1, 8, "LOW"),
            self._swing(3, 12, "HIGH"),  # first high: recorded, no trend yet
            self._swing(5, 9, "LOW"),    # higher low, no flip
            self._swing(7, 14, "HIGH"),  # higher high -> BOS, trend BULLISH
            self._swing(9, 5, "LOW"),    # lower low -> CHoCH into BEARISH
        ]
        result = ms._classify_swings(swings)

        self.assertEqual(result["trend"], "BEARISH")
        self.assertEqual(result["last_event"]["type"], "CHoCH")
        self.assertEqual(result["last_event"]["direction"], "BEARISH")

    def test_lower_low_after_bearish_trend_is_continuation_bos(self):
        swings = [
            self._swing(1, 12, "HIGH"),
            self._swing(3, 8, "LOW"),   # first low -> BOS, trend BEARISH
            self._swing(5, 10, "HIGH"), # lower high, no flip
            self._swing(7, 6, "LOW"),   # lower low -> continuation BOS
        ]
        result = ms._classify_swings(swings)

        self.assertEqual(result["trend"], "BEARISH")
        self.assertEqual(result["last_event"]["type"], "BOS")

    def test_fewer_than_two_swings_is_unavailable(self):
        result = ms._classify_swings([self._swing(1, 10, "HIGH")])
        self.assertFalse(result["available"])


class LiveBreakCheckTests(unittest.TestCase):
    def test_close_above_last_swing_high_is_a_bullish_break(self):
        candles = [_candle(0, high=10, low=9, close=10.5, closed=False)]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure, require_closed_candle=False)

        self.assertTrue(result["broken"])
        self.assertEqual(result["direction"], "BULLISH")
        self.assertFalse(result["candle_closed"])

    def test_close_below_last_swing_low_is_a_bearish_break(self):
        candles = [_candle(0, high=10, low=4, close=4.5)]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure)

        self.assertTrue(result["broken"])
        self.assertEqual(result["direction"], "BEARISH")

    def test_close_inside_range_is_no_break(self):
        candles = [_candle(0, high=8, low=6, close=7)]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure)

        self.assertFalse(result["broken"])

    def test_unavailable_structure_is_never_broken(self):
        result = ms.live_break_check([_candle(0, 10, 9)], {"available": False})
        self.assertFalse(result["broken"])

    def test_broken_result_includes_the_evaluated_candles_open_time(self):
        candles = [_candle(123, high=10, low=9, close=10.5, closed=False)]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure, require_closed_candle=False)

        self.assertEqual(result["open_time"], 123)

    def test_require_closed_candle_ignores_a_forming_candles_break(self):
        # The forming candle (open_time=1) wicks above the swing high, but
        # require_closed_candle=True must only look at the last CLOSED
        # candle (open_time=0), which never broke it.
        candles = [
            _candle(0, high=9.5, low=8, close=9, closed=True),
            _candle(1, high=11, low=9, close=10.5, closed=False),
        ]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure, require_closed_candle=True)

        self.assertFalse(result["broken"])

    def test_require_closed_candle_fires_once_the_breaking_candle_closes(self):
        candles = [
            _candle(0, high=9.5, low=8, close=9, closed=True),
            _candle(1, high=11, low=9.5, close=10.5, closed=True),
        ]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure, require_closed_candle=True)

        self.assertTrue(result["broken"])
        self.assertEqual(result["direction"], "BULLISH")
        self.assertTrue(result["candle_closed"])
        self.assertEqual(result["open_time"], 1)

    def test_require_closed_candle_with_no_closed_candle_yet_is_not_broken(self):
        candles = [_candle(0, high=11, low=9, close=10.5, closed=False)]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        result = ms.live_break_check(candles, structure, require_closed_candle=True)

        self.assertFalse(result["broken"])

    def test_require_closed_candle_defaults_from_config(self):
        candles = [
            _candle(0, high=9.5, low=8, close=9, closed=True),
            _candle(1, high=11, low=9.5, close=10.5, closed=False),
        ]
        structure = {"available": True, "last_swing_high": 10, "last_swing_low": 5}

        with patch.object(config, "REQUIRE_CLOSE_CONFIRMED_BREAK", True):
            result = ms.live_break_check(candles, structure)

        self.assertFalse(result["broken"])


class FindOrderBlockTests(unittest.TestCase):
    def test_bullish_break_uses_last_red_candle(self):
        candles = [
            _candle(0, high=10, low=9, open_=9.5, close=9.2),   # red
            _candle(1, high=11, low=9.5, open_=9.6, close=10.9),  # green (impulse)
        ]
        block = ms.find_order_block(candles, index=1, direction="BULLISH")

        self.assertIsNotNone(block)
        self.assertEqual(block["index"], 0)

    def test_bearish_break_uses_last_green_candle(self):
        candles = [
            _candle(0, high=10, low=9, open_=9.2, close=9.8),   # green
            _candle(1, high=9.5, low=8, open_=9.4, close=8.2),  # red (impulse down)
        ]
        block = ms.find_order_block(candles, index=1, direction="BEARISH")

        self.assertIsNotNone(block)
        self.assertEqual(block["index"], 0)


class FairValueGapTests(unittest.TestCase):
    def test_detects_bullish_gap(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # low(11) > candle0 high(10) -> bullish FVG
        ]
        gaps = ms.find_fair_value_gaps(candles, lookback=10)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["type"], "BULLISH")

    def test_detects_bearish_gap(self):
        candles = [
            _candle(0, high=12, low=11),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=10, low=9),  # high(10) < candle0 low(11) -> bearish FVG
        ]
        gaps = ms.find_fair_value_gaps(candles, lookback=10)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["type"], "BEARISH")

    def test_no_gap_when_candles_overlap(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=9.5),
            _candle(2, high=10.2, low=9.8),
        ]
        gaps = ms.find_fair_value_gaps(candles, lookback=10)
        self.assertEqual(gaps, [])


class FindFvgRetestTests(unittest.TestCase):
    def test_wick_and_reject_into_bullish_gap(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.8, low=10.5, open_=10.7, close=10.6),  # wick in, close above bottom
        ]
        result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 10)

    def test_wick_and_reject_into_bearish_gap(self):
        candles = [
            _candle(0, high=12, low=11),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=10, low=9),  # gap: top=11, bottom=10, index=2
            # wick in, close reclaims well past the midpoint (10.5) back
            # toward the near edge (bottom=10) - a strong rejection.
            _candle(3, high=10.5, low=10.3, open_=10.4, close=10.2),
        ]
        result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 11)

    def test_gap_mitigated_by_a_close_past_the_far_edge_is_excluded(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10, low=8, open_=9.8, close=9.5),  # closes below bottom -> mitigates
            _candle(4, high=10.8, low=10.5, open_=10.7, close=10.6),  # would otherwise retest
        ]
        result = ms.find_fvg_retest(candles)

        self.assertIsNone(result)

    def test_wick_through_without_a_close_past_the_edge_does_not_mitigate(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.5, low=9.5, open_=9.8, close=10.3),  # wicks below bottom, closes back above
            _candle(4, high=10.8, low=10.5, open_=10.7, close=10.6),  # retest candle
        ]
        result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")

    def test_gap_older_than_max_age_is_ignored(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            # Flat fillers (high=11, low=10, matching the gap's own bounds)
            # deliberately don't form any NEW gap of their own - a wider
            # filler range would create a second, more recent gap and
            # accidentally still satisfy the retest, defeating the point of
            # this test.
            _candle(3, high=11, low=10, open_=10.5, close=10.5),
            _candle(4, high=11, low=10, open_=10.5, close=10.5),
            _candle(5, high=10.8, low=10.5, open_=10.7, close=10.6),  # would retest, but age=3 > max_age=1
        ]
        result = ms.find_fvg_retest(candles, max_age_candles=1)

        self.assertIsNone(result)

    def test_no_gaps_is_none(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=9.5),
            _candle(2, high=10.2, low=9.8),
        ]
        self.assertIsNone(ms.find_fvg_retest(candles))

    def test_insufficient_candles_is_none(self):
        self.assertIsNone(ms.find_fvg_retest([_candle(0, high=10, low=9)]))

    def test_result_includes_the_tested_candles_open_time(self):
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.8, low=10.5, open_=10.7, close=10.6),
        ]
        result = ms.find_fvg_retest(candles)

        self.assertEqual(result["open_time"], 3)

    def test_result_includes_the_tested_candles_index(self):
        # Real bug found live (2026-08-21): signal_engine.py's
        # setup_age_candles used to recompute this as len(ltf_candles)-1
        # instead of reading it from here - silently wrong whenever
        # require_closed_candle left the tested candle short of the
        # buffer's own last index (see the require_closed_candle test
        # below for exactly that case).
        candles = [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.8, low=10.5, open_=10.7, close=10.6),
        ]
        result = ms.find_fvg_retest(candles)

        self.assertEqual(result["tested_index"], 3)


class FindFvgRetestMinCloseThroughPctTests(unittest.TestCase):
    """config.OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT - real motivation
    (2026-08-22): live OB_FVG_RETEST trades averaged ~0.68R max adverse
    excursion even on eventual wins, and the original condition accepted
    a close barely past the gap's far edge - deep inside the zone - as
    equally valid as a strong reclaim near the near edge."""

    def _bullish_gap_candles(self, retest_close):
        return [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.8, low=10.5, open_=10.7, close=retest_close),
        ]

    def _bearish_gap_candles(self, retest_close):
        return [
            _candle(0, high=12, low=11),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=10, low=9),  # gap: top=11, bottom=10, index=2
            _candle(3, high=10.5, low=10.3, open_=10.4, close=retest_close),
        ]

    def test_bullish_close_that_only_clears_the_far_edge_is_rejected_at_default(self):
        # close=10.05 is barely above bottom(10) - deep inside the gap,
        # well short of the midpoint(10.5) the default 0.5 requires.
        candles = self._bullish_gap_candles(retest_close=10.05)

        result = ms.find_fvg_retest(candles)

        self.assertIsNone(result)

    def test_bullish_close_that_clears_the_midpoint_is_accepted_at_default(self):
        candles = self._bullish_gap_candles(retest_close=10.51)

        result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")

    def test_bearish_close_that_only_clears_the_far_edge_is_rejected_at_default(self):
        # close=10.95 is barely below top(11) - deep inside the gap, well
        # short of the midpoint(10.5) the default 0.5 requires.
        candles = self._bearish_gap_candles(retest_close=10.95)

        result = ms.find_fvg_retest(candles)

        self.assertIsNone(result)

    def test_bearish_close_that_clears_the_midpoint_is_accepted_at_default(self):
        candles = self._bearish_gap_candles(retest_close=10.49)

        result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")

    def test_pct_zero_restores_the_original_close_anywhere_past_far_edge_behavior(self):
        candles = self._bullish_gap_candles(retest_close=10.05)

        result = ms.find_fvg_retest(candles, min_close_through_pct=0.0)

        self.assertIsNotNone(result)

    def test_pct_one_requires_a_full_close_back_outside_the_gap(self):
        # close=10.99 clears the midpoint easily but still hasn't fully
        # exited the gap (top=11) - only qualifies once pct reaches 1.0's
        # requirement is NOT met here, confirming pct=1.0 is stricter than
        # the 0.5 default already tested above.
        candles = self._bullish_gap_candles(retest_close=10.99)

        result = ms.find_fvg_retest(candles, min_close_through_pct=1.0)

        self.assertIsNone(result)

    def test_defaults_from_config(self):
        candles = self._bullish_gap_candles(retest_close=10.05)

        with patch.object(config, "OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT", 0.0):
            result = ms.find_fvg_retest(candles)

        self.assertIsNotNone(result)


class FindFvgRetestRequireClosedCandleTests(unittest.TestCase):
    """config.REQUIRE_CLOSE_CONFIRMED_BREAK - same real motivation as
    liquidity_sweep.detect_sweep's identical change (2026-08-13, two live
    trades traced against actual price history): a retest read against a
    still-forming candle can flip before the candle actually finishes."""

    def _gapped_candles(self, retest_closed):
        return [
            _candle(0, high=10, low=9),
            _candle(1, high=10.5, low=10.2),
            _candle(2, high=12, low=11),  # gap: bottom=10, top=11, index=2
            _candle(3, high=10.8, low=10.5, open_=10.7, close=10.6, closed=retest_closed),
        ]

    def test_forming_retest_candle_is_ignored_when_required(self):
        candles = self._gapped_candles(retest_closed=False)

        result = ms.find_fvg_retest(candles, require_closed_candle=True)

        self.assertIsNone(result)

    def test_fires_once_the_retest_candle_closes(self):
        candles = self._gapped_candles(retest_closed=True)

        result = ms.find_fvg_retest(candles, require_closed_candle=True)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["open_time"], 3)

    def test_tested_index_is_the_closed_candle_not_a_later_still_forming_one(self):
        # The real scenario behind the setup_age_candles bug: a still-
        # forming candle sits AFTER the one this function actually tested
        # (and gated the age of) - tested_index must point at the closed
        # one (3), not len(candles)-1 (4).
        candles = self._gapped_candles(retest_closed=True) + [
            _candle(4, high=10.9, low=10.6, open_=10.6, close=10.7, closed=False),
        ]

        result = ms.find_fvg_retest(candles, require_closed_candle=True)

        self.assertEqual(result["tested_index"], 3)

    def test_no_closed_candle_at_all_returns_none(self):
        candles = [_candle(0, high=10, low=9, closed=False)]

        self.assertIsNone(ms.find_fvg_retest(candles, require_closed_candle=True))

    def test_defaults_from_config(self):
        candles = self._gapped_candles(retest_closed=False)

        with patch.object(config, "REQUIRE_CLOSE_CONFIRMED_BREAK", True):
            result = ms.find_fvg_retest(candles)

        self.assertIsNone(result)

    def test_require_closed_candle_false_checks_the_forming_candle(self):
        candles = self._gapped_candles(retest_closed=False)

        result = ms.find_fvg_retest(candles, require_closed_candle=False)

        self.assertIsNotNone(result)


class FindStructureEventsTests(unittest.TestCase):
    """Backs ORDER_BLOCK_RETEST_TRIGGER_ENABLED's find_order_blocks below -
    structure_state/_classify_swings only expose the single most recent
    event (last_event); this needs every one across the full sequence."""

    def test_returns_every_bos_choch_not_just_the_last(self):
        with patch.object(ms, "find_swing_points", return_value=[
            ms.SwingPoint(1, 100, 8, "LOW"),
            ms.SwingPoint(3, 300, 12, "HIGH"),   # first high: recorded, no event yet
            ms.SwingPoint(5, 500, 9, "LOW"),     # higher low, no event
            ms.SwingPoint(7, 700, 14, "HIGH"),   # higher high -> BOS #1, trend BULLISH
            ms.SwingPoint(9, 900, 6, "LOW"),     # lower low -> CHoCH #2, trend BEARISH
            ms.SwingPoint(11, 1100, 10, "HIGH"), # not a new high (10 < 14), no event
            ms.SwingPoint(13, 1300, 4, "LOW"),   # lower low -> continuation BOS #3
        ]):
            events = ms.find_structure_events([])

        self.assertEqual([e["type"] for e in events], ["BOS", "CHoCH", "BOS"])
        self.assertEqual([e["index"] for e in events], [7, 9, 13])
        self.assertEqual(events[-1]["direction"], "BEARISH")

    def test_fewer_than_two_swings_returns_empty_list(self):
        with patch.object(ms, "find_swing_points", return_value=[ms.SwingPoint(1, 100, 8, "LOW")]):
            self.assertEqual(ms.find_structure_events([]), [])


class FindOrderBlocksTests(unittest.TestCase):
    def test_builds_a_block_for_each_recent_event(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),    # bearish - origin of event @1
            _candle(1, high=8, low=6, open_=6, close=8),     # bullish impulsive move
            _candle(2, high=6, low=5, open_=6, close=5),     # bearish - origin of event @3
            _candle(3, high=10, low=7, open_=7, close=10),   # bullish impulsive move
        ]
        events = [
            {"type": "BOS", "direction": "BULLISH", "index": 1, "price": 8},
            {"type": "BOS", "direction": "BULLISH", "index": 3, "price": 10},
        ]

        with patch.object(ms, "find_structure_events", return_value=events):
            blocks = ms.find_order_blocks(candles, max_events=5)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["direction"], "BULLISH")
        self.assertEqual(blocks[0]["index"], 0)
        self.assertEqual(blocks[1]["index"], 2)

    def test_max_events_limits_how_many_recent_breaks_are_considered(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),
            _candle(1, high=8, low=6, open_=6, close=8),
            _candle(2, high=6, low=5, open_=6, close=5),
            _candle(3, high=10, low=7, open_=7, close=10),
        ]
        events = [
            {"type": "BOS", "direction": "BULLISH", "index": 1, "price": 8},
            {"type": "BOS", "direction": "BULLISH", "index": 3, "price": 10},
        ]

        with patch.object(ms, "find_structure_events", return_value=events):
            blocks = ms.find_order_blocks(candles, max_events=1)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["index"], 2)

    def test_event_with_no_qualifying_order_block_is_skipped(self):
        candles = [_candle(0, high=5, low=4, open_=4, close=5)]  # only a bullish candle exists
        events = [{"type": "BOS", "direction": "BULLISH", "index": 0, "price": 5}]

        with patch.object(ms, "find_structure_events", return_value=events):
            blocks = ms.find_order_blocks(candles)

        self.assertEqual(blocks, [])

    def test_max_events_zero_or_negative_returns_no_blocks(self):
        with patch.object(ms, "find_structure_events", return_value=[
            {"type": "BOS", "direction": "BULLISH", "index": 0, "price": 5},
        ]):
            self.assertEqual(ms.find_order_blocks([_candle(0, high=5, low=4)], max_events=0), [])


class FindOrderBlockRetestTests(unittest.TestCase):
    def test_wick_and_reject_into_bullish_block(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),   # origin block: high=5, low=4
            _candle(1, high=8, low=6, open_=6, close=8),
            _candle(2, high=10, low=7, open_=7, close=9),
            _candle(3, high=5.5, low=4.2, open_=5.3, close=4.8),  # wick in, close above low(4)
        ]
        blocks = [{"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}]

        result = ms.find_order_block_retest(candles, blocks=blocks)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 4)

    def test_wick_and_reject_into_bearish_block(self):
        candles = [
            _candle(0, high=10, low=9, open_=9, close=10),  # origin block: high=10, low=9
            _candle(1, high=8, low=6),
            _candle(2, high=7, low=5),
            _candle(3, high=10.3, low=9.5, open_=9.8, close=9.6),  # wick in, close below high(10)
        ]
        blocks = [{"direction": "BEARISH", "high": 10, "low": 9, "index": 0, "open_time": 0}]

        result = ms.find_order_block_retest(candles, blocks=blocks)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 10)

    def test_block_mitigated_by_a_close_past_the_far_edge_is_excluded(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),
            _candle(1, high=6, low=3, open_=5, close=3.5),  # closes below low(4) -> mitigates
            _candle(2, high=5.5, low=4.2, open_=5.3, close=4.8),  # would otherwise retest
        ]
        blocks = [{"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}]

        result = ms.find_order_block_retest(candles, blocks=blocks)

        self.assertIsNone(result)

    def test_block_older_than_max_age_is_ignored(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),
            _candle(1, high=6, low=5),
            _candle(2, high=6, low=5),
            _candle(3, high=5.5, low=4.2, open_=5.3, close=4.8),
        ]
        blocks = [{"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}]

        result = ms.find_order_block_retest(candles, blocks=blocks, max_age_candles=1)

        self.assertIsNone(result)

    def test_no_blocks_is_none(self):
        candles = [_candle(0, high=10, low=9), _candle(1, high=10.5, low=9.5)]
        self.assertIsNone(ms.find_order_block_retest(candles, blocks=[]))

    def test_insufficient_candles_is_none(self):
        self.assertIsNone(ms.find_order_block_retest([_candle(0, high=10, low=9)]))

    def test_result_includes_the_tested_candles_open_time(self):
        candles = [
            _candle(0, high=5, low=4, open_=5, close=4),
            _candle(1, high=6, low=5),
            _candle(2, high=6, low=5),
            _candle(3, high=5.5, low=4.2, open_=5.3, close=4.8),
        ]
        blocks = [{"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}]

        result = ms.find_order_block_retest(candles, blocks=blocks)

        self.assertEqual(result["open_time"], 3)

    def test_computes_blocks_internally_when_not_provided(self):
        candles = [_candle(0, high=5, low=4), _candle(1, high=6, low=5)]

        with patch.object(ms, "find_order_blocks", return_value=[]):
            result = ms.find_order_block_retest(candles)

        self.assertIsNone(result)


class FindOrderBlockRetestRequireClosedCandleTests(unittest.TestCase):
    def _blocked_candles(self, retest_closed):
        return [
            _candle(0, high=5, low=4, open_=5, close=4),
            _candle(1, high=9, low=8),   # well above the block - not a retest
            _candle(2, high=9, low=8),   # well above the block - not a retest
            _candle(3, high=5.5, low=4.2, open_=5.3, close=4.8, closed=retest_closed),
        ]

    def _blocks(self):
        return [{"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}]

    def test_forming_retest_candle_is_ignored_when_required(self):
        result = ms.find_order_block_retest(
            self._blocked_candles(False), blocks=self._blocks(), require_closed_candle=True
        )
        self.assertIsNone(result)

    def test_fires_once_the_retest_candle_closes(self):
        result = ms.find_order_block_retest(
            self._blocked_candles(True), blocks=self._blocks(), require_closed_candle=True
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["open_time"], 3)

    def test_no_closed_candle_at_all_returns_none(self):
        candles = [
            _candle(0, high=10, low=9, closed=False),
            _candle(1, high=10, low=9, closed=False),
        ]
        result = ms.find_order_block_retest(candles, blocks=self._blocks(), require_closed_candle=True)
        self.assertIsNone(result)

    def test_defaults_from_config(self):
        with patch.object(config, "REQUIRE_CLOSE_CONFIRMED_BREAK", True):
            result = ms.find_order_block_retest(self._blocked_candles(False), blocks=self._blocks())

        self.assertIsNone(result)

    def test_require_closed_candle_false_checks_the_forming_candle(self):
        result = ms.find_order_block_retest(
            self._blocked_candles(False), blocks=self._blocks(), require_closed_candle=False
        )

        self.assertIsNotNone(result)


class OrderBlockRetestCloseThroughTests(unittest.TestCase):
    """config.ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT (2026-09-22) - the
    rejection-strength requirement ported from find_fvg_retest, which has
    carried the identical parameter since 2026-08-22.

    Block used throughout: high=5, low=4, so range=1 and the midpoint is
    4.5. For a BULLISH (demand) block the far edge is `low` and the near
    edge is `high`, so required_close = 4 + 1*pct.
    """

    BLOCK = {"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}

    def _candles(self, retest_close):
        return [
            _candle(0, high=5, low=4, open_=5, close=4),     # the origin block
            _candle(1, high=9, low=8),                        # away from the block
            _candle(2, high=9, low=8),
            _candle(3, high=5.5, low=4.05, open_=5.3, close=retest_close),
        ]

    def _retest(self, retest_close, pct):
        return ms.find_order_block_retest(
            self._candles(retest_close), blocks=[dict(self.BLOCK)],
            min_close_through_pct=pct,
        )

    def test_zero_is_byte_identical_to_the_pre_fix_condition(self):
        # The old condition was exactly `close > low`. At pct=0.0 a close a
        # hair above the far edge must still detect - this is the proof the
        # CODE DEFAULT is inert, so a deploy without the .env line cannot
        # change live behaviour.
        self.assertIsNotNone(self._retest(4.01, 0.0))
        # ...and a close at or below the far edge must still not detect.
        self.assertIsNone(self._retest(4.0, 0.0))
        self.assertIsNone(self._retest(3.99, 0.0))

    def test_the_code_default_is_zero_so_a_code_only_deploy_is_a_no_op(self):
        # This project deploys code and .env separately onto a live account,
        # so the CODE default must stay 0.0 - reading config.<flag> here
        # would just read whatever the live .env holds (0.5), which proves
        # nothing. Asserted against the source literal rather than by
        # reloading config: importlib.reload rebinds module attributes and
        # would silently break this module's own setUpModule patcher (and
        # any other active patch.object) for every test that follows.
        import pathlib
        import re

        source = pathlib.Path(config.__file__).read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r'ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT\s*=\s*env_float\(\s*'
            r'"ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT"\s*,\s*([0-9.]+)\s*,?\s*\)',
            source,
        )

        self.assertIsNotNone(
            match,
            "could not find the ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT env_float "
            "declaration in config.py - if it was renamed or restructured, update "
            "this guard rather than deleting it",
        )
        self.assertEqual(
            float(match.group(1)), 0.0,
            "the CODE default for ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT must stay "
            "0.0 so deploying code without the matching .env line cannot change "
            "behaviour on a live account",
        )

    def test_half_rejects_a_close_below_the_block_midpoint(self):
        # 4.3 sits deep inside the block - the "price is sitting in the
        # zone" case the audit identified. Detected at 0.0, rejected at 0.5.
        self.assertIsNotNone(self._retest(4.3, 0.0))
        self.assertIsNone(self._retest(4.3, 0.5))

    def test_half_accepts_a_close_above_the_block_midpoint(self):
        self.assertIsNotNone(self._retest(4.8, 0.5))

    def test_half_is_exclusive_at_the_midpoint_itself(self):
        # required_close = 4 + 1*0.5 = 4.5, and the test is strictly `>`.
        self.assertIsNone(self._retest(4.5, 0.5))
        self.assertIsNotNone(self._retest(4.51, 0.5))

    def test_one_requires_a_full_reclaim_past_the_near_edge(self):
        self.assertIsNone(self._retest(4.9, 1.0))
        self.assertIsNotNone(self._retest(5.1, 1.0))

    def test_bearish_blocks_mirror_the_geometry(self):
        # BEARISH supply block high=10, low=9, midpoint 9.5. Far edge is
        # `high`, near edge `low`, so required_close = 10 - 1*pct and the
        # test is strictly `<`.
        block = {"direction": "BEARISH", "high": 10, "low": 9, "index": 0, "open_time": 0}
        candles = [
            _candle(0, high=10, low=9, open_=9, close=10),
            _candle(1, high=8, low=6),
            _candle(2, high=7, low=5),
            _candle(3, high=10.3, low=9.5, open_=9.8, close=9.7),
        ]

        self.assertIsNotNone(
            ms.find_order_block_retest(candles, blocks=[dict(block)], min_close_through_pct=0.0)
        )
        # 9.7 is above the 9.5 midpoint - a weak rejection for a SHORT.
        self.assertIsNone(
            ms.find_order_block_retest(candles, blocks=[dict(block)], min_close_through_pct=0.5)
        )

    def test_zero_range_block_behaves_identically_at_every_pct(self):
        # A doji origin candle gives high == low, so there is no range to
        # measure a reclaim against. max(range, 0) makes required_close
        # collapse to the far edge at ANY pct rather than the block being
        # silently dropped - the behaviour it had before this parameter
        # existed.
        block = {"direction": "BULLISH", "high": 4, "low": 4, "index": 0, "open_time": 0}
        candles = [
            _candle(0, high=4, low=4, open_=4, close=4),
            _candle(1, high=9, low=8),
            _candle(2, high=9, low=8),
            _candle(3, high=5.5, low=3.9, open_=5.3, close=4.01),
        ]

        for pct in (0.0, 0.5, 1.0):
            with self.subTest(pct=pct):
                self.assertIsNotNone(
                    ms.find_order_block_retest(
                        candles, blocks=[dict(block)], min_close_through_pct=pct
                    )
                )

    def test_out_of_range_values_are_clamped(self):
        # An .env typo must not be able to invert the condition: a negative
        # pct would otherwise push required_close BELOW the far edge and
        # accept closes the unfixed code rejected.
        self.assertIsNone(self._retest(3.99, -1.0))
        self.assertIsNotNone(self._retest(4.01, -1.0))     # clamps to 0.0
        self.assertIsNone(self._retest(4.9, 5.0))          # clamps to 1.0
        self.assertIsNotNone(self._retest(5.1, 5.0))

    def test_reads_the_config_value_when_not_passed(self):
        with patch.object(config, "ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT", 0.5):
            self.assertIsNone(
                ms.find_order_block_retest(self._candles(4.3), blocks=[dict(self.BLOCK)])
            )

        with patch.object(config, "ORDER_BLOCK_RETEST_MIN_CLOSE_THROUGH_PCT", 0.0):
            self.assertIsNotNone(
                ms.find_order_block_retest(self._candles(4.3), blocks=[dict(self.BLOCK)])
            )


class EmaPullbackTrendTests(unittest.TestCase):
    """config.EMA_PULLBACK_REQUIRE_TREND_ENABLED (2026-09-22) - supplies the
    "within an established trend" half of detect_ema_pullback's own
    description, which the detector never implemented. Measured live, 30.6%
    of its detections fired against the slope of the very EMA they were
    pulling back to."""

    def _candles(self, closes, wick_close):
        """`closes` builds the EMA history; the final candle is the retest,
        wicking through `level` and closing back across it."""
        out = [_candle(i, c + 1, c - 1, close=c, open_=c) for i, c in enumerate(closes)]
        out.append(_candle(len(closes), high=wick_close + 3, low=wick_close - 3,
                           close=wick_close, open_=wick_close))
        return out

    def test_flag_off_is_identical_to_the_thin_wrapper(self):
        # A falling EMA with a BULLISH pullback - accepted today, because
        # there is no trend condition at all.
        candles = self._candles([100 - i for i in range(30)], wick_close=75)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", False):
            result = ms.detect_ema_pullback(candles, ema_value=74)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")

    def test_bullish_pullback_against_a_falling_ema_is_dropped(self):
        candles = self._candles([100 - i for i in range(30)], wick_close=75)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True):
            result = ms.detect_ema_pullback(candles, ema_value=74)

        self.assertIsNone(result)

    def test_bullish_pullback_with_a_rising_ema_is_kept(self):
        candles = self._candles([70 + i for i in range(30)], wick_close=101)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True):
            result = ms.detect_ema_pullback(candles, ema_value=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")

    def test_bearish_pullback_requires_a_falling_ema(self):
        falling = self._candles([100 - i for i in range(30)], wick_close=73)
        rising = self._candles([70 + i for i in range(30)], wick_close=99)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True):
            # BEARISH: wick ABOVE the level, close back below it.
            self.assertIsNotNone(ms.detect_ema_pullback(falling, ema_value=74))
            self.assertIsNone(ms.detect_ema_pullback(rising, ema_value=100))

    def test_a_flat_ema_rejects_both_directions(self):
        # Not a tie to be broken - a flat EMA is the ABSENCE of a trend, so
        # neither side qualifies as trend-continuation.
        candles = self._candles([100] * 30, wick_close=101)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True), \
             patch.object(ms, "ema_prior_value", return_value=100.0):
            self.assertIsNone(ms.detect_ema_pullback(candles, ema_value=100.0))

    def test_missing_prior_ema_fails_open(self):
        # Too little history to measure a slope must not silently kill the
        # trigger - same fail-open convention as every other read.
        candles = self._candles([70 + i for i in range(30)], wick_close=101)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True), \
             patch.object(ms, "ema_prior_value", return_value=None):
            self.assertIsNotNone(ms.detect_ema_pullback(candles, ema_value=100))

    def test_no_pullback_is_still_none_whatever_the_trend(self):
        candles = self._candles([70 + i for i in range(30)], wick_close=101)

        with patch.object(config, "EMA_PULLBACK_REQUIRE_TREND_ENABLED", True):
            # level far below the candle's low - never touched.
            self.assertIsNone(ms.detect_ema_pullback(candles, ema_value=10))


class AnchoredPoolClusteringTests(unittest.TestCase):
    """config.LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED (2026-09-22) -
    find_liquidity_pools compared each point to cluster[-1], the PREVIOUS
    point, so a chain of points each within tolerance of its neighbour
    merged into one pool spanning far more than the tolerance. Measured
    live: 8.0% of pools exceeded their own tolerance, up to 3.4x."""

    def _swings(self, prices):
        return [ms.SwingPoint(i, i, p, "HIGH") for i, p in enumerate(prices)]

    def test_a_chain_merges_when_chained_and_splits_when_anchored(self):
        # Each step is 0.8% - inside a 1% tolerance - but the run spans
        # 2.4%, well outside it.
        swings = self._swings([100.0, 100.8, 101.6, 102.4])

        chained = ms.find_liquidity_pools(swings, tolerance_pct=0.01, anchored=False)
        anchored = ms.find_liquidity_pools(swings, tolerance_pct=0.01, anchored=True)

        self.assertEqual(len(chained), 1)
        self.assertGreater(len(anchored), len(chained))
        # Nothing anchored may span more than the tolerance it was built with.
        for pool in anchored:
            self.assertLessEqual(pool["touches"], 2)

    def test_genuinely_equal_highs_still_cluster_when_anchored(self):
        swings = self._swings([100.0, 100.05, 100.09])

        anchored = ms.find_liquidity_pools(swings, tolerance_pct=0.01, anchored=True)

        self.assertEqual(len(anchored), 1)
        self.assertEqual(anchored[0]["touches"], 3)

    def test_default_reads_the_config_flag(self):
        swings = self._swings([100.0, 100.8, 101.6, 102.4])

        with patch.object(config, "LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED", False):
            self.assertEqual(len(ms.find_liquidity_pools(swings, tolerance_pct=0.01)), 1)

        with patch.object(config, "LIQUIDITY_POOL_ANCHORED_CLUSTERING_ENABLED", True):
            self.assertGreater(len(ms.find_liquidity_pools(swings, tolerance_pct=0.01)), 1)

    def test_a_single_point_never_forms_a_pool_either_way(self):
        for anchored in (False, True):
            with self.subTest(anchored=anchored):
                self.assertEqual(
                    ms.find_liquidity_pools(self._swings([100.0]), anchored=anchored), [])


class OrderBlockScanLookbackTests(unittest.TestCase):
    """config.ORDER_BLOCK_SCAN_LOOKBACK_CANDLES (2026-09-22) - was a
    hardcoded 10 inside find_order_block. Default 10 must reproduce that
    exactly; this exists to make the number visible, not to change it."""

    def _candles(self):
        # index 0 is the only bearish candle; everything after is bullish.
        out = [_candle(0, high=10, low=9, open_=10, close=9)]
        for i in range(1, 12):
            out.append(_candle(i, high=20 + i, low=19 + i, open_=19 + i, close=20 + i))
        return out

    def test_default_ten_finds_a_block_nine_candles_back(self):
        candles = self._candles()
        block = ms.find_order_block(candles, index=9, direction="BULLISH")

        self.assertIsNotNone(block)
        self.assertEqual(block["index"], 0)

    def test_a_shorter_lookback_misses_it(self):
        candles = self._candles()
        self.assertIsNone(
            ms.find_order_block(candles, index=9, direction="BULLISH", lookback=5))

    def test_reads_the_config_value_when_not_passed(self):
        candles = self._candles()

        with patch.object(config, "ORDER_BLOCK_SCAN_LOOKBACK_CANDLES", 5):
            self.assertIsNone(ms.find_order_block(candles, index=9, direction="BULLISH"))

        with patch.object(config, "ORDER_BLOCK_SCAN_LOOKBACK_CANDLES", 10):
            self.assertIsNotNone(ms.find_order_block(candles, index=9, direction="BULLISH"))


class DetectLevelPullbackTests(unittest.TestCase):
    def test_bullish_pullback_wicks_to_level_then_reclaims(self):
        candles = [_candle(0, high=103, low=99, open_=99.5, close=101)]

        result = ms.detect_level_pullback(candles, level=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 100)

    def test_bearish_pullback_wicks_to_level_then_reclaims(self):
        candles = [_candle(0, high=101, low=97, open_=100.5, close=99)]

        result = ms.detect_level_pullback(candles, level=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 100)

    def test_no_touch_of_the_level_is_not_a_pullback(self):
        candles = [_candle(0, high=110, low=105, open_=106, close=108)]

        self.assertIsNone(ms.detect_level_pullback(candles, level=100))

    def test_touch_without_a_reclaim_is_a_breakdown_not_a_pullback(self):
        # Wicks below the level but closes below it too - a real break,
        # not a held pullback.
        candles = [_candle(0, high=99, low=95, open_=98.5, close=97)]

        self.assertIsNone(ms.detect_level_pullback(candles, level=100))

    def test_none_level_is_none(self):
        candles = [_candle(0, high=103, low=99, close=101)]
        self.assertIsNone(ms.detect_level_pullback(candles, level=None))

    def test_empty_candles_is_none(self):
        self.assertIsNone(ms.detect_level_pullback([], level=100))

    def test_result_includes_the_tested_candles_open_time(self):
        candles = [_candle(5, high=103, low=99, open_=99.5, close=101)]

        result = ms.detect_level_pullback(candles, level=100)

        self.assertEqual(result["open_time"], 5)

    def test_require_closed_candle_true_scans_back_to_last_closed(self):
        candles = [
            _candle(0, high=106, low=104, close=105),  # well above the level - not a pullback
            _candle(1, high=103, low=99, open_=99.5, close=101, closed=False),
        ]
        result = ms.detect_level_pullback(candles, level=100, require_closed_candle=True)
        self.assertIsNone(result)


class DetectEmaPullbackTests(unittest.TestCase):
    def test_bullish_pullback_wicks_to_ema_then_reclaims(self):
        candles = [_candle(0, high=103, low=99, open_=99.5, close=101)]

        result = ms.detect_ema_pullback(candles, ema_value=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["level"], 100)

    def test_bearish_pullback_wicks_to_ema_then_reclaims(self):
        candles = [_candle(0, high=101, low=97, open_=100.5, close=99)]

        result = ms.detect_ema_pullback(candles, ema_value=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["level"], 100)

    def test_no_touch_of_the_ema_is_not_a_pullback(self):
        candles = [_candle(0, high=110, low=105, open_=106, close=108)]

        self.assertIsNone(ms.detect_ema_pullback(candles, ema_value=100))

    def test_touch_without_a_reclaim_is_a_breakdown_not_a_pullback(self):
        # Wicks below the EMA but closes below it too - a real break, not
        # a held pullback.
        candles = [_candle(0, high=99, low=95, open_=98.5, close=97)]

        self.assertIsNone(ms.detect_ema_pullback(candles, ema_value=100))

    def test_none_ema_value_is_none(self):
        candles = [_candle(0, high=103, low=99, close=101)]
        self.assertIsNone(ms.detect_ema_pullback(candles, ema_value=None))

    def test_empty_candles_is_none(self):
        self.assertIsNone(ms.detect_ema_pullback([], ema_value=100))

    def test_result_includes_the_tested_candles_open_time(self):
        candles = [_candle(5, high=103, low=99, open_=99.5, close=101)]

        result = ms.detect_ema_pullback(candles, ema_value=100)

        self.assertEqual(result["open_time"], 5)


class DetectEmaPullbackRequireClosedCandleTests(unittest.TestCase):
    """config.REQUIRE_CLOSE_CONFIRMED_BREAK - same real motivation as
    every other close-confirmed trigger: a wick-and-reclaim read on a
    still-forming candle can flip before the candle actually finishes."""

    def _candles(self, tested_closed):
        return [
            _candle(0, high=106, low=104, close=105),  # well above the EMA - not a pullback
            _candle(1, high=103, low=99, open_=99.5, close=101, closed=tested_closed),
        ]

    def test_forming_tested_candle_is_ignored_when_required(self):
        result = ms.detect_ema_pullback(
            self._candles(tested_closed=False), ema_value=100, require_closed_candle=True
        )
        self.assertIsNone(result)

    def test_fires_once_the_tested_candle_closes(self):
        result = ms.detect_ema_pullback(
            self._candles(tested_closed=True), ema_value=100, require_closed_candle=True
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["open_time"], 1)

    def test_no_closed_candle_at_all_returns_none(self):
        candles = [_candle(0, high=103, low=99, close=101, closed=False)]
        result = ms.detect_ema_pullback(candles, ema_value=100, require_closed_candle=True)
        self.assertIsNone(result)

    def test_defaults_from_config(self):
        with patch.object(config, "REQUIRE_CLOSE_CONFIRMED_BREAK", True):
            result = ms.detect_ema_pullback(self._candles(tested_closed=False), ema_value=100)

        self.assertIsNone(result)

    def test_require_closed_candle_false_checks_the_forming_candle(self):
        result = ms.detect_ema_pullback(
            self._candles(tested_closed=False), ema_value=100, require_closed_candle=False
        )
        self.assertIsNotNone(result)


class LiquidityPoolTests(unittest.TestCase):
    def test_two_equal_highs_form_a_buy_side_pool(self):
        swings = [
            ms.SwingPoint(0, 0, 100.0, "HIGH"),
            ms.SwingPoint(1, 1, 100.05, "HIGH"),
        ]
        pools = ms.find_liquidity_pools(swings, tolerance_pct=0.001)

        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["type"], "BUY_SIDE")
        self.assertEqual(pools[0]["touches"], 2)

    def test_two_equal_lows_form_a_sell_side_pool(self):
        swings = [
            ms.SwingPoint(0, 0, 50.0, "LOW"),
            ms.SwingPoint(1, 1, 50.02, "LOW"),
        ]
        pools = ms.find_liquidity_pools(swings, tolerance_pct=0.001)

        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["type"], "SELL_SIDE")

    def test_single_swing_never_forms_a_pool(self):
        swings = [ms.SwingPoint(0, 0, 100.0, "HIGH")]
        pools = ms.find_liquidity_pools(swings, tolerance_pct=0.001)
        self.assertEqual(pools, [])

    def test_swings_far_apart_do_not_cluster(self):
        swings = [
            ms.SwingPoint(0, 0, 100.0, "HIGH"),
            ms.SwingPoint(1, 1, 150.0, "HIGH"),
        ]
        pools = ms.find_liquidity_pools(swings, tolerance_pct=0.001)
        self.assertEqual(pools, [])


class FindStructureCandidatesTests(unittest.TestCase):
    def test_bullish_order_block_becomes_sell_side_near_edge_by_default(self):
        block = {"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}
        with patch.object(ms, "find_liquidity_pools", return_value=[]), \
             patch.object(ms, "find_order_blocks", return_value=[block]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[]):
            pools = ms.find_structure_candidates([_candle(0, high=5, low=4)])

        self.assertEqual(pools, [{"type": "SELL_SIDE", "price": 5, "touches": 0}])

    def test_bullish_order_block_uses_far_edge_for_stop(self):
        block = {"direction": "BULLISH", "high": 5, "low": 4, "index": 0, "open_time": 0}
        with patch.object(ms, "find_liquidity_pools", return_value=[]), \
             patch.object(ms, "find_order_blocks", return_value=[block]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[]):
            pools = ms.find_structure_candidates([_candle(0, high=5, low=4)], for_stop=True)

        self.assertEqual(pools, [{"type": "SELL_SIDE", "price": 4, "touches": 0}])

    def test_bearish_order_block_becomes_buy_side(self):
        block = {"direction": "BEARISH", "high": 10, "low": 9, "index": 0, "open_time": 0}
        with patch.object(ms, "find_liquidity_pools", return_value=[]), \
             patch.object(ms, "find_order_blocks", return_value=[block]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[]):
            near = ms.find_structure_candidates([_candle(0, high=10, low=9)])
            far = ms.find_structure_candidates([_candle(0, high=10, low=9)], for_stop=True)

        self.assertEqual(near, [{"type": "BUY_SIDE", "price": 9, "touches": 0}])
        self.assertEqual(far, [{"type": "BUY_SIDE", "price": 10, "touches": 0}])

    def test_bullish_fvg_becomes_sell_side(self):
        gap = {"type": "BULLISH", "top": 95, "bottom": 90, "index": 2}
        with patch.object(ms, "find_liquidity_pools", return_value=[]), \
             patch.object(ms, "find_order_blocks", return_value=[]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[gap]):
            near = ms.find_structure_candidates([_candle(0, high=10, low=9)])
            far = ms.find_structure_candidates([_candle(0, high=10, low=9)], for_stop=True)

        self.assertEqual(near, [{"type": "SELL_SIDE", "price": 95, "touches": 0}])
        self.assertEqual(far, [{"type": "SELL_SIDE", "price": 90, "touches": 0}])

    def test_bearish_fvg_becomes_buy_side(self):
        gap = {"type": "BEARISH", "top": 110, "bottom": 105, "index": 2}
        with patch.object(ms, "find_liquidity_pools", return_value=[]), \
             patch.object(ms, "find_order_blocks", return_value=[]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[gap]):
            near = ms.find_structure_candidates([_candle(0, high=10, low=9)])
            far = ms.find_structure_candidates([_candle(0, high=10, low=9)], for_stop=True)

        self.assertEqual(near, [{"type": "BUY_SIDE", "price": 105, "touches": 0}])
        self.assertEqual(far, [{"type": "BUY_SIDE", "price": 110, "touches": 0}])

    def test_merges_with_real_liquidity_pools(self):
        real_pool = {"type": "BUY_SIDE", "price": 120, "touches": 3}
        with patch.object(ms, "find_liquidity_pools", return_value=[real_pool]), \
             patch.object(ms, "find_order_blocks", return_value=[]), \
             patch.object(ms, "find_fair_value_gaps", return_value=[]):
            pools = ms.find_structure_candidates([_candle(0, high=10, low=9)])

        self.assertEqual(pools, [real_pool])


class PremiumDiscountZoneTests(unittest.TestCase):
    def _candles(self):
        return [_candle(i, high=110, low=90) for i in range(5)]

    def test_zone_bounds_and_midpoint(self):
        zone = ms.premium_discount_zone(self._candles(), lookback=10)

        self.assertTrue(zone["available"])
        self.assertEqual(zone["range_high"], 110)
        self.assertEqual(zone["range_low"], 90)
        self.assertEqual(zone["midpoint"], 100)

    def test_zone_for_price_above_and_below_midpoint(self):
        zone = ms.premium_discount_zone(self._candles(), lookback=10)

        self.assertEqual(ms.zone_for_price(zone, 105), "PREMIUM")
        self.assertEqual(ms.zone_for_price(zone, 95), "DISCOUNT")

    def test_empty_candles_are_unavailable(self):
        zone = ms.premium_discount_zone([], lookback=10)
        self.assertFalse(zone["available"])

    def test_in_ote_bullish_zone(self):
        zone = ms.premium_discount_zone(self._candles(), lookback=10)
        with patch.object(config, "OTE_RETRACEMENT_MIN", 0.6), patch.object(
            config, "OTE_RETRACEMENT_MAX", 0.8
        ):
            zone = ms.premium_discount_zone(self._candles(), lookback=10)
            # range 90-110, bullish OTE = high - range*[0.8, 0.6] = [94, 98]
            self.assertTrue(ms.in_ote(zone, 96, "BULLISH"))
            self.assertFalse(ms.in_ote(zone, 105, "BULLISH"))


class ZoneDirectionTests(unittest.TestCase):
    """config.ZONE_DIRECTION_REJECT_ENABLED - which way the range is MOVING,
    the companion to zone_for_price's "where in the range am I". Compares
    the midpoint of the recent half of the window against the older half."""

    def test_rising_range_is_bullish(self):
        candles = [
            _candle(0, high=110, low=90), _candle(1, high=110, low=90),
            _candle(2, high=120, low=100), _candle(3, high=120, low=100),
        ]
        # older half midpoint 100, recent half midpoint 110
        self.assertEqual(ms.zone_direction(candles, lookback=4), "BULLISH")

    def test_falling_range_is_bearish(self):
        candles = [
            _candle(0, high=120, low=100), _candle(1, high=120, low=100),
            _candle(2, high=110, low=90), _candle(3, high=110, low=90),
        ]
        self.assertEqual(ms.zone_direction(candles, lookback=4), "BEARISH")

    def test_flat_range_is_none_so_the_gate_fails_open(self):
        candles = [_candle(i, high=110, low=90) for i in range(4)]
        self.assertIsNone(ms.zone_direction(candles, lookback=4))

    def test_too_little_history_is_none(self):
        candles = [_candle(i, high=110, low=90) for i in range(3)]
        self.assertIsNone(ms.zone_direction(candles, lookback=10))

    def test_empty_candles_are_none(self):
        self.assertIsNone(ms.zone_direction([], lookback=10))

    def test_lookback_slices_to_the_most_recent_window(self):
        # Over all 8 candles the range is FALLING (mid 190 -> 105); over
        # just the last 4 it is RISING (mid 95 -> 115). The lookback must
        # decide which one is measured - this is the whole reason the
        # direction window is configured separately from the zone window.
        candles = (
            [_candle(i, high=200, low=180) for i in range(4)]
            + [_candle(i, high=100, low=90) for i in range(4, 6)]
            + [_candle(i, high=120, low=110) for i in range(6, 8)]
        )
        self.assertEqual(ms.zone_direction(candles, lookback=8), "BEARISH")
        self.assertEqual(ms.zone_direction(candles, lookback=4), "BULLISH")

    def test_lookback_defaults_to_the_configured_value(self):
        candles = (
            [_candle(i, high=200, low=180) for i in range(4)]
            + [_candle(i, high=100, low=90) for i in range(4, 6)]
            + [_candle(i, high=120, low=110) for i in range(6, 8)]
        )
        with patch.object(config, "ZONE_DIRECTION_LOOKBACK_CANDLES", 4):
            self.assertEqual(ms.zone_direction(candles), "BULLISH")

        with patch.object(config, "ZONE_DIRECTION_LOOKBACK_CANDLES", 8):
            self.assertEqual(ms.zone_direction(candles), "BEARISH")

    def test_direction_is_independent_of_where_price_sits_in_the_range(self):
        # The point of the measure: two windows with the SAME high/low
        # bounds overall can still be drifting opposite ways. zone_for_price
        # cannot tell these apart; zone_direction can.
        rising = [
            _candle(0, high=105, low=90), _candle(1, high=105, low=90),
            _candle(2, high=110, low=95), _candle(3, high=110, low=95),
        ]
        falling = [
            _candle(0, high=110, low=95), _candle(1, high=110, low=95),
            _candle(2, high=105, low=90), _candle(3, high=105, low=90),
        ]
        zone_r = ms.premium_discount_zone(rising, lookback=4)
        zone_f = ms.premium_discount_zone(falling, lookback=4)

        self.assertEqual(zone_r["range_high"], zone_f["range_high"])
        self.assertEqual(zone_r["range_low"], zone_f["range_low"])
        self.assertEqual(ms.zone_for_price(zone_r, 99), ms.zone_for_price(zone_f, 99))
        self.assertEqual(ms.zone_direction(rising, lookback=4), "BULLISH")
        self.assertEqual(ms.zone_direction(falling, lookback=4), "BEARISH")


class AverageTrueRangeTests(unittest.TestCase):
    def test_atr_is_positive_for_moving_candles(self):
        candles = [_candle(i, high=100 + i, low=95 + i, close=97 + i) for i in range(20)]
        atr = ms.average_true_range(candles, period=14)
        self.assertGreater(atr, 0)

    def test_atr_zero_with_too_few_candles(self):
        candles = [_candle(0, high=100, low=95, close=97)]
        atr = ms.average_true_range(candles, period=14)
        self.assertEqual(atr, 0.0)


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_none_with_too_few_candles(self):
        candles = [_candle(i, high=101, low=99, close=100) for i in range(5)]
        self.assertIsNone(ms.exponential_moving_average(candles, period=20))

    def test_flat_price_series_converges_to_that_price(self):
        candles = [_candle(i, high=101, low=99, close=100) for i in range(60)]
        ema = ms.exponential_moving_average(candles, period=20)
        self.assertAlmostEqual(ema, 100, places=6)

    def test_rising_prices_pull_ema_up_but_below_the_latest_close(self):
        candles = [_candle(i, high=i + 1, low=i - 1, close=float(i)) for i in range(1, 61)]
        ema = ms.exponential_moving_average(candles, period=20)
        self.assertLess(ema, candles[-1]["close"])
        self.assertGreater(ema, candles[0]["close"])


class EfficiencyRatioTests(unittest.TestCase):
    """Kaufman's Efficiency Ratio - backs the chop/volatility-regime
    filter (config.CHOP_FILTER_LOOKBACK_CANDLES), informational only."""

    def test_straight_line_trend_is_close_to_one(self):
        candles = [_candle(i, high=100 + i, low=99 + i, close=100 + i) for i in range(15)]
        er = ms.efficiency_ratio(candles, period=14)
        self.assertAlmostEqual(er, 1.0, places=6)

    def test_round_trip_chop_is_close_to_zero(self):
        # Up 10 then back down to the start - net move ~0, big path length.
        closes = list(range(100, 110)) + list(range(110, 99, -1))
        candles = [_candle(i, high=c + 1, low=c - 1, close=c) for i, c in enumerate(closes)]
        er = ms.efficiency_ratio(candles, period=len(candles) - 1)
        self.assertLess(er, 0.1)

    def test_none_with_too_few_candles(self):
        candles = [_candle(i, high=101, low=99, close=100) for i in range(5)]
        self.assertIsNone(ms.efficiency_ratio(candles, period=14))

    def test_none_with_zero_path_length(self):
        candles = [_candle(i, high=100, low=100, close=100) for i in range(15)]
        self.assertIsNone(ms.efficiency_ratio(candles, period=14))


class PriceHoldConsistencyTests(unittest.TestCase):
    """config.OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED - real evidence
    (2026-09-02, 43 resolved OB_FVG_RETEST trades). Real closed-candle
    price-action proxy for "was recent pressure sustained", since
    historical order-book depth isn't archived anywhere to check the
    same question the way depth_consistency_pct does."""

    def _candle(self, open_, close):
        return {"open": open_, "close": close}

    def test_none_with_too_few_candles(self):
        candles = [self._candle(100, 101) for _ in range(5)]
        self.assertIsNone(ms.price_hold_consistency(candles, "BUY", lookback=10))

    def test_all_favorable_buy_closes_is_1(self):
        candles = [self._candle(100, 101) for _ in range(10)]  # every close > open
        self.assertAlmostEqual(ms.price_hold_consistency(candles, "BUY", lookback=10), 1.0)

    def test_all_unfavorable_buy_closes_is_0(self):
        candles = [self._candle(101, 100) for _ in range(10)]  # every close < open
        self.assertAlmostEqual(ms.price_hold_consistency(candles, "BUY", lookback=10), 0.0)

    def test_sell_favors_closes_below_open(self):
        candles = [self._candle(101, 100) for _ in range(10)]
        self.assertAlmostEqual(ms.price_hold_consistency(candles, "SELL", lookback=10), 1.0)

    def test_mixed_candles_computes_the_real_fraction(self):
        candles = (
            [self._candle(100, 101) for _ in range(6)]  # favorable for BUY
            + [self._candle(101, 100) for _ in range(4)]  # unfavorable for BUY
        )
        self.assertAlmostEqual(ms.price_hold_consistency(candles, "BUY", lookback=10), 0.6)

    def test_only_the_most_recent_lookback_candles_are_used(self):
        # 10 unfavorable, then 5 favorable - lookback=5 should only see
        # the favorable tail.
        candles = (
            [self._candle(101, 100) for _ in range(10)]
            + [self._candle(100, 101) for _ in range(5)]
        )
        self.assertAlmostEqual(ms.price_hold_consistency(candles, "BUY", lookback=5), 1.0)

    def test_lookback_defaults_from_config(self):
        with patch.object(config, "OB_FVG_RETEST_PRICE_HOLD_LOOKBACK_MINUTES", 3):
            candles = [self._candle(100, 101) for _ in range(3)]
            self.assertAlmostEqual(ms.price_hold_consistency(candles, "BUY"), 1.0)


class OiPercentileTests(unittest.TestCase):
    """config.MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED - real evidence
    (2026-09-02, 185 resolved trades that already pass MARKET_CHOPPY,
    real futures_open_interest_hist data)."""

    def _history(self, values):
        return [(float(i), v) for i, v in enumerate(values)]

    def test_none_with_no_history(self):
        self.assertIsNone(ms.oi_percentile(None))
        self.assertIsNone(ms.oi_percentile([]))

    def test_none_below_30_samples(self):
        history = self._history([100.0] * 29)
        self.assertIsNone(ms.oi_percentile(history))

    def test_latest_value_at_the_low_end_is_near_0(self):
        # 30 samples, latest is the smallest -> only itself is <=.
        values = list(range(1, 30)) + [0]
        history = self._history(values)
        self.assertAlmostEqual(ms.oi_percentile(history), 1 / 30)

    def test_latest_value_at_the_high_end_is_1(self):
        # 30 samples, latest is the largest -> every value is <=.
        values = list(range(0, 29)) + [1000]
        history = self._history(values)
        self.assertAlmostEqual(ms.oi_percentile(history), 1.0)

    def test_latest_value_at_the_median_is_half(self):
        # 30 samples 0..29 with the median value (14) moved to the end:
        # 15 of the 30 values (0..14) are <= it.
        values = [v for v in range(30) if v != 14] + [14]
        history = self._history(values)
        self.assertAlmostEqual(ms.oi_percentile(history), 0.5)


class PriceCorrelationTests(unittest.TestCase):
    """Backs the BTC-correlation confluence field (config.BTC_CORRELATION_ENABLED)."""

    def test_identical_series_are_fully_correlated(self):
        candles_a = [_candle(i, high=101 + i, low=99 + i, close=100 + i) for i in range(20)]
        candles_b = [_candle(i, high=101 + i, low=99 + i, close=100 + i) for i in range(20)]
        self.assertAlmostEqual(ms.price_correlation(candles_a, candles_b, period=20), 1.0, places=6)

    def test_inverse_series_are_fully_anti_correlated(self):
        candles_a = [_candle(i, high=101 + i, low=99 + i, close=100 + i) for i in range(20)]
        candles_b = [_candle(i, high=101 - i, low=99 - i, close=100 - i) for i in range(20)]
        self.assertAlmostEqual(ms.price_correlation(candles_a, candles_b, period=20), -1.0, places=6)

    def test_none_with_too_few_candles(self):
        candles = [_candle(i, high=101, low=99, close=100) for i in range(5)]
        self.assertIsNone(ms.price_correlation(candles, candles, period=20))

    def test_none_when_one_series_has_zero_variance(self):
        candles_a = [_candle(i, high=101 + i, low=99 + i, close=100 + i) for i in range(20)]
        flat = [_candle(i, high=101, low=99, close=100) for i in range(20)]
        self.assertIsNone(ms.price_correlation(candles_a, flat, period=20))


class PriceReturnTests(unittest.TestCase):
    def test_positive_return_for_a_rising_series(self):
        candles = [_candle(0, high=101, low=99, close=100), _candle(1, high=111, low=109, close=110)]
        self.assertAlmostEqual(ms.price_return(candles, period=2), 0.10, places=6)

    def test_none_with_too_few_candles(self):
        candles = [_candle(0, high=101, low=99, close=100)]
        self.assertIsNone(ms.price_return(candles, period=2))


class AnalyzeTests(unittest.TestCase):
    def test_unavailable_with_too_few_candles(self):
        result = ms.analyze([_candle(0, 10, 9)])
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
