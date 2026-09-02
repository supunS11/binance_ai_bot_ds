import unittest
from unittest.mock import patch

import config
import market_structure as ms


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
