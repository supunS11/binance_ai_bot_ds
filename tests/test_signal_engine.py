import unittest
from unittest.mock import patch

import config
import cvd_divergence
import liquidity_sweep
import market_structure
import oi_divergence
import risk_manager
import signal_engine


def _ltf_candles(close):
    return [{
        "open_time": 0, "open": close - 0.5, "high": close + 0.5,
        "low": close - 1.0, "close": close, "volume": 1.0, "closed": False,
    }]


HTF_BULLISH = {"available": True, "trend": "BULLISH"}
HTF_BEARISH = {"available": True, "trend": "BEARISH"}
ZONE = {
    "available": True,
    "midpoint": 100,
    "range_high": 110,
    "range_low": 90,
    "bullish_ote_zone": (90, 95),
    "bearish_ote_zone": (105, 110),
}
LTF_BULLISH_BREAK = {
    "available": True,
    "live_break": {"broken": True, "direction": "BULLISH", "level": 90},
    "fair_value_gaps": [{"type": "BULLISH", "top": 95, "bottom": 90, "index": 0}],
    "atr": 1.0,
}
LTF_BEARISH_BREAK = {
    "available": True,
    "live_break": {"broken": True, "direction": "BEARISH", "level": 110},
    "fair_value_gaps": [{"type": "BEARISH", "top": 110, "bottom": 105, "index": 0}],
    "atr": 1.0,
}
OI_RISING = {"available": True, "oi_change_pct": 5.0}
LIQUIDATION_LONG_CLUSTER = {
    "available": True,
    "long_liquidation_notional": 80000,
    "short_liquidation_notional": 10000,
    "net_liquidation_notional": 70000,
}


class SignalEngineTests(unittest.TestCase):
    def _run(
        self,
        symbol="BTCUSDT",
        ltf_close=93.0,
        cvd=None,
        depth=None,
        htf_structure=None,
        zone=None,
        ltf_analysis=None,
        order_block=None,
        sweep_direction="BULLISH",
        sweep_level=None,
        sweep_open_time=None,
        fvg_retest_direction=None,
        fvg_retest_level=None,
        fvg_retest_open_time=0,
        divergence_direction=None,
        divergence_level=None,
        divergence_index=5,
        divergence_open_time=999,
        order_block_retest_direction=None,
        order_block_retest_level=None,
        order_block_retest_open_time=777,
        oi_divergence_direction=None,
        oi_divergence_level=None,
        oi_divergence_index=5,
        oi_divergence_open_time=888,
        liquidation_confirmed_sweep_direction=None,
        liquidation_confirmed_sweep_level=None,
        liquidation_confirmed_sweep_open_time=666,
        ema_pullback_direction=None,
        ema_pullback_level=None,
        ema_pullback_open_time=555,
        ema_value=85.0,
        htf_trend_ema=None,
        oi_snapshot=None,
        liquidation_snapshot=None,
        quote_volume_usdt=None,
        btc_candles="default",
        btc_correlation=0.5,
        btc_return=0.02,
        funding_rate=None,
        oi_rising_reject_enabled=False,
    ):
        cvd = {"available": True, "cvd_score": 0.5} if cvd is None else cvd
        depth = {"available": True, "depth_imbalance": 0.2} if depth is None else depth
        htf_structure = HTF_BULLISH if htf_structure is None else htf_structure
        zone = ZONE if zone is None else zone
        ltf_analysis = LTF_BULLISH_BREAK if ltf_analysis is None else ltf_analysis
        sweep = (
            {"direction": sweep_direction, "level": sweep_level, "open_time": sweep_open_time}
            if sweep_direction else None
        )
        fvg_retest = (
            {
                "direction": fvg_retest_direction, "level": fvg_retest_level, "gap": {},
                "open_time": fvg_retest_open_time,
            }
            if fvg_retest_direction else None
        )
        divergence = (
            {
                "direction": divergence_direction, "level": divergence_level,
                "index": divergence_index, "open_time": divergence_open_time,
            }
            if divergence_direction else None
        )
        order_block_retest = (
            {
                "direction": order_block_retest_direction, "level": order_block_retest_level,
                "open_time": order_block_retest_open_time,
            }
            if order_block_retest_direction else None
        )
        oi_divergence_result = (
            {
                "direction": oi_divergence_direction, "level": oi_divergence_level,
                "index": oi_divergence_index, "open_time": oi_divergence_open_time,
            }
            if oi_divergence_direction else None
        )
        liquidation_confirmed_sweep = (
            {
                "direction": liquidation_confirmed_sweep_direction,
                "level": liquidation_confirmed_sweep_level,
                "open_time": liquidation_confirmed_sweep_open_time,
            }
            if liquidation_confirmed_sweep_direction else None
        )
        ema_pullback = (
            {
                "direction": ema_pullback_direction, "level": ema_pullback_level,
                "open_time": ema_pullback_open_time,
            }
            if ema_pullback_direction else None
        )
        oi_snapshot = OI_RISING if oi_snapshot is None else oi_snapshot
        liquidation_snapshot = (
            LIQUIDATION_LONG_CLUSTER if liquidation_snapshot is None else liquidation_snapshot
        )
        btc_candles = ["btc_candle_placeholder"] if btc_candles == "default" else btc_candles

        # Two distinct callers share this one mocked function: the
        # existing informational LTF ema_value (called with just
        # ltf_candles, no period=) and the new HTF_TREND_FRESHNESS_ENABLED
        # gate's htf_trend_ema (always called with an explicit period= -
        # see signal_engine.py). Routing on whether period was passed
        # keeps them independently controllable - htf_trend_ema defaults
        # to None (gate is a no-op) so no existing test is affected
        # unless it explicitly opts in.
        def _ema_side_effect(candles, period=None):
            return htf_trend_ema if period is not None else ema_value

        # OI_RISING_REJECT_ENABLED defaults True in config.py (real gate,
        # see its own comment) but oi_snapshot's own default above
        # (OI_RISING) is rising - defaulting this kwarg to False keeps
        # every test above and below that doesn't care about this specific
        # gate exercising its old, pre-gate behavior; opt in explicitly
        # (oi_rising_reject_enabled=True) to test the gate itself.
        with patch.object(config, "OI_RISING_REJECT_ENABLED", oi_rising_reject_enabled), \
             patch.object(market_structure, "structure_state", return_value=htf_structure), \
             patch.object(market_structure, "premium_discount_zone", return_value=zone), \
             patch.object(market_structure, "analyze", return_value=ltf_analysis), \
             patch.object(market_structure, "find_order_block", return_value=order_block), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]), \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "find_fvg_retest", return_value=fvg_retest), \
             patch.object(market_structure, "find_order_block_retest", return_value=order_block_retest), \
             patch.object(market_structure, "detect_ema_pullback", return_value=ema_pullback), \
             patch.object(market_structure, "exponential_moving_average", side_effect=_ema_side_effect), \
             patch.object(market_structure, "price_correlation", return_value=btc_correlation), \
             patch.object(market_structure, "price_return", return_value=btc_return), \
             patch.object(liquidity_sweep, "detect_sweep", return_value=sweep), \
             patch.object(
                 liquidity_sweep, "detect_liquidation_confirmed_sweep",
                 return_value=liquidation_confirmed_sweep,
             ), \
             patch.object(cvd_divergence, "detect_divergence", return_value=divergence), \
             patch.object(oi_divergence, "detect_divergence", return_value=oi_divergence_result):
            return signal_engine.evaluate(
                symbol, ["htf_placeholder"], _ltf_candles(ltf_close), cvd, depth,
                oi_snapshot=oi_snapshot, liquidation_snapshot=liquidation_snapshot,
                quote_volume_usdt=quote_volume_usdt, btc_candles=btc_candles,
                funding_rate=funding_rate,
            )

    def test_full_buy_signal_when_everything_aligns(self):
        result = self._run()

        self.assertEqual(result["signal"], "BUY")
        self.assertTrue(result["sweep_confluence"])
        self.assertEqual(result["premium_discount_zone"], "DISCOUNT")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        # (range_high=110 - entry=93) / (range_high=110 - range_low=90)
        self.assertAlmostEqual(result["zone_retracement_pct"], 0.85)

    def test_full_sell_signal_when_everything_aligns(self):
        result = self._run(
            ltf_close=108.0,
            cvd={"available": True, "cvd_score": -0.5},
            depth={"available": True, "depth_imbalance": -0.2},
            htf_structure=HTF_BEARISH,
            ltf_analysis=LTF_BEARISH_BREAK,
            sweep_direction="BEARISH",
            ema_value=115.0,
        )

        self.assertEqual(result["signal"], "SELL")
        self.assertTrue(result["sweep_confluence"])
        self.assertEqual(result["premium_discount_zone"], "PREMIUM")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        # (entry=108 - range_low=90) / (range_high=110 - range_low=90)
        self.assertAlmostEqual(result["zone_retracement_pct"], 0.9)

    def test_no_signal_when_htf_structure_unavailable(self):
        result = self._run(htf_structure={"available": False})
        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "HTF_STRUCTURE_UNAVAILABLE")

    def test_no_signal_when_zone_unavailable(self):
        result = self._run(zone={"available": False})
        self.assertEqual(result["reason"], "ZONE_UNAVAILABLE")

    def test_no_signal_when_ltf_structure_unavailable(self):
        result = self._run(ltf_analysis={"available": False})
        self.assertEqual(result["reason"], "LTF_STRUCTURE_UNAVAILABLE")

    def test_no_signal_when_no_live_break(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}
        # sweep_direction=None: this test predates every alternative
        # trigger and asserts the "nothing at all fired" case - must not
        # depend on whichever trigger flags happen to be True in the
        # loaded .env (LIQUIDITY_SWEEP_TRIGGER_ENABLED is live-True today).
        result = self._run(ltf_analysis=analysis, sweep_direction=None)
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_no_signal_against_htf_bias(self):
        result = self._run(htf_structure=HTF_BEARISH)
        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    # config.HTF_TREND_FRESHNESS_ENABLED - a second, faster-updating HTF
    # read (a plain EMA on the HTF candles) that can catch a stale swing-
    # confirmed htf_trend AGAINST_HTF_BIAS alone would miss. Default
    # ltf_close=93.0; htf_trend_ema defaults to None in _run() (a no-op)
    # so every test above this point is unaffected unless it opts in.

    def test_buy_rejected_when_price_has_fallen_below_the_htf_trend_ema(self):
        # latest_price=93 < htf_trend_ema=95 - the swing-confirmed trend
        # says BULLISH (HTF_BULLISH default), but real price has already
        # slipped back below the faster-updating HTF average.
        result = self._run(htf_trend_ema=95.0)
        self.assertEqual(result["reason"], "HTF_TREND_STALE")

    def test_buy_allowed_when_price_is_still_above_the_htf_trend_ema(self):
        result = self._run(htf_trend_ema=90.0)
        self.assertEqual(result["signal"], "BUY")

    def test_sell_rejected_when_price_has_risen_above_the_htf_trend_ema(self):
        result = self._run(
            ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
            depth={"available": True, "depth_imbalance": -0.2},
            htf_structure=HTF_BEARISH, ltf_analysis=LTF_BEARISH_BREAK,
            ema_value=115.0, htf_trend_ema=105.0,
        )
        self.assertEqual(result["reason"], "HTF_TREND_STALE")

    def test_sell_allowed_when_price_is_still_below_the_htf_trend_ema(self):
        result = self._run(
            ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
            depth={"available": True, "depth_imbalance": -0.2},
            htf_structure=HTF_BEARISH, ltf_analysis=LTF_BEARISH_BREAK,
            ema_value=115.0, htf_trend_ema=115.0,
        )
        self.assertEqual(result["signal"], "SELL")

    def test_disabled_flag_skips_the_check_even_when_price_disagrees(self):
        with patch.object(config, "HTF_TREND_FRESHNESS_ENABLED", False):
            result = self._run(htf_trend_ema=95.0)

        self.assertEqual(result["signal"], "BUY")

    def test_no_htf_history_yet_does_not_block_the_signal(self):
        # htf_trend_ema=None (too little HTF history) - absence never
        # blocks a signal on its own, same convention as every other
        # optional confirmation.
        result = self._run(htf_trend_ema=None)
        self.assertEqual(result["signal"], "BUY")

    def test_market_choppy_rejects_below_the_threshold(self):
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3):
            result = self._run(ltf_analysis=analysis)

        self.assertEqual(result["reason"], "MARKET_CHOPPY")

    def test_market_choppy_allows_at_or_above_the_threshold(self):
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.3)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3):
            result = self._run(ltf_analysis=analysis)

        self.assertEqual(result["signal"], "BUY")

    def test_market_choppy_gate_disabled_lets_a_choppy_signal_through(self):
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3), \
             patch.object(config, "EFFICIENCY_RATIO_GATE_ENABLED", False):
            result = self._run(ltf_analysis=analysis)

        self.assertEqual(result["signal"], "BUY")
        # Still journaled/available as an informational reading even
        # though the gate itself is off.
        self.assertFalse(result["efficiency_favorable"])

    def test_no_efficiency_data_yet_does_not_block_the_signal(self):
        # LTF_BULLISH_BREAK carries no efficiency_ratio key - absence
        # never blocks a signal on its own, same convention as every
        # other optional confirmation.
        result = self._run()
        self.assertEqual(result["signal"], "BUY")

    def test_market_choppy_applies_to_sell_the_same_way(self):
        analysis = dict(LTF_BEARISH_BREAK, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
            )

        self.assertEqual(result["reason"], "MARKET_CHOPPY")

    def test_no_signal_when_price_not_in_discount_for_buy(self):
        result = self._run(ltf_close=105.0)
        self.assertIn("NOT_IN_DISCOUNT", result["reason"])

    def test_no_signal_when_not_in_ote(self):
        # 99 is still < midpoint(100) -> discount, but outside (90, 95) OTE.
        # OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED pinned False - this test
        # predates that flag and is about the base OTE check applying at
        # all, not trigger-scoping (see OteGateStructureBreakOnlyTests-
        # style tests above for that) - without pinning it, whatever the
        # real .env happens to have it set to would decide whether the
        # default sweep_direction="BULLISH" candidate this fixture also
        # produces skips OTE and wins instead of STRUCTURE_BREAK.
        with patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", False):
            result = self._run(ltf_close=99.0)

        self.assertEqual(result["reason"], "NOT_IN_OTE")

    def test_ote_gate_still_applies_to_structure_break_when_scoped(self):
        # sweep_direction=None - _run()'s own default ("BULLISH") would
        # otherwise also produce a LIQUIDITY_SWEEP candidate, which (once
        # the new flag is on) skips OTE and could win the ranking instead
        # of STRUCTURE_BREAK, defeating this test's isolation.
        with patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", True):
            result = self._run(ltf_close=99.0, sweep_direction=None)

        self.assertEqual(result["reason"], "NOT_IN_OTE")

    def test_ote_gate_skipped_for_non_structure_break_triggers_when_scoped(self):
        # live_break disabled so OB_FVG_RETEST is the only candidate -
        # otherwise STRUCTURE_BREAK would also fire here and, being
        # higher-priority, would win and still get gated on OTE itself.
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False})

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", True):
            result = self._run(
                ltf_close=99.0, ltf_analysis=analysis,
                fvg_retest_direction="BULLISH", fvg_retest_level=91.0,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")

    def test_ote_gate_applies_to_every_trigger_by_default(self):
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False})

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", False):
            result = self._run(
                ltf_close=99.0, ltf_analysis=analysis,
                fvg_retest_direction="BULLISH", fvg_retest_level=91.0,
            )

        self.assertEqual(result["reason"], "NOT_IN_OTE")

    def test_market_choppy_gate_still_applies_to_structure_break_when_scoped(self):
        # sweep_direction=None - _run()'s own default ("BULLISH") would
        # otherwise also produce a LIQUIDITY_SWEEP candidate; harmless here
        # (LIQUIDITY_SWEEP isn't in the exempt trigger set either) but kept
        # off anyway to isolate STRUCTURE_BREAK cleanly, same convention as
        # the OTE gate tests above.
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3), \
             patch.object(config, "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", True):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertEqual(result["reason"], "MARKET_CHOPPY")

    def test_market_choppy_gate_skipped_for_cvd_divergence_when_scoped(self):
        # live_break disabled + sweep_direction=None so CVD_DIVERGENCE is
        # the only candidate - otherwise STRUCTURE_BREAK/LIQUIDITY_SWEEP
        # would also fire, and being higher-priority, would win and still
        # get gated on MARKET_CHOPPY themselves.
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False}, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3), \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=91.0,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")

    def test_market_choppy_gate_applies_to_every_trigger_by_default(self):
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False}, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3), \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=91.0,
            )

        self.assertEqual(result["reason"], "MARKET_CHOPPY")

    def test_cvd_confirmation_gate_still_applies_to_structure_break_when_scoped(self):
        with patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15), \
             patch.object(config, "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED", True):
            result = self._run(cvd={"available": True, "cvd_score": 0.05}, sweep_direction=None)

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    def test_cvd_confirmation_gate_skipped_for_cvd_divergence_when_scoped(self):
        # live_break disabled + sweep_direction=None so CVD_DIVERGENCE is
        # the only candidate, same isolation as the MARKET_CHOPPY tests
        # above.
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False})

        with patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15), \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=91.0,
                cvd={"available": True, "cvd_score": 0.05},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")

    def test_cvd_confirmation_gate_applies_to_every_trigger_by_default(self):
        analysis = dict(LTF_BULLISH_BREAK, live_break={"broken": False})

        with patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15), \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=91.0,
                cvd={"available": True, "cvd_score": 0.05},
            )

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    def test_no_signal_without_order_block_or_fvg_when_required(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["fair_value_gaps"] = []

        with patch.object(config, "REQUIRE_ORDER_BLOCK_OR_FVG", True):
            result = self._run(ltf_analysis=analysis, order_block=None)

        self.assertEqual(result["reason"], "NO_ORDER_BLOCK_OR_FVG")

    def test_signal_allowed_without_ob_or_fvg_when_not_required(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["fair_value_gaps"] = []

        with patch.object(config, "REQUIRE_ORDER_BLOCK_OR_FVG", False):
            result = self._run(ltf_analysis=analysis, order_block=None)

        self.assertEqual(result["signal"], "BUY")

    def test_ema_wrong_side_is_logged_but_does_not_block_a_buy(self):
        # Informational only: price below EMA on a BUY is recorded as
        # ema_aligned=False, but must NOT reject the signal - an EMA is a
        # lagging indicator, so gating on it would delay real-time entries
        # on sharp moves without evidence it's actually worth the cost.
        result = self._run(ema_value=95.0)  # ltf_close defaults to 93 < 95

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["ema_aligned"])
        self.assertEqual(result["ema_value"], 95.0)

    def test_ema_aligned_true_for_sell_when_price_is_below_ema(self):
        result = self._run(
            ltf_close=108.0,
            cvd={"available": True, "cvd_score": -0.5},
            depth={"available": True, "depth_imbalance": -0.2},
            htf_structure=HTF_BEARISH,
            ltf_analysis=LTF_BEARISH_BREAK,
            sweep_direction="BEARISH",
            ema_value=115.0,  # 108 < 115 -> aligned for a SELL
        )

        self.assertEqual(result["signal"], "SELL")
        self.assertTrue(result["ema_aligned"])

    def test_ema_unavailable_does_not_block_the_signal(self):
        result = self._run(ema_value=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["ema_aligned"])
        self.assertIsNone(result["ema_value"])

    def test_ema_fields_are_none_when_confirmation_disabled(self):
        with patch.object(config, "EMA_CONFIRMATION_ENABLED", False):
            result = self._run(ema_value=95.0)  # would be misaligned if computed

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["ema_value"])
        self.assertIsNone(result["ema_aligned"])

    def test_oi_rising_is_recorded_when_the_reject_gate_is_off(self):
        result = self._run(oi_snapshot={"available": True, "oi_change_pct": 8.0})

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["oi_change_pct"], 8.0)
        self.assertTrue(result["oi_rising"])

    def test_oi_rising_rejects_when_the_reject_gate_is_on(self):
        result = self._run(
            oi_snapshot={"available": True, "oi_change_pct": 8.0},
            oi_rising_reject_enabled=True,
        )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "OI_RISING")

    def test_oi_falling_is_not_rejected_even_when_the_gate_is_on(self):
        result = self._run(
            oi_snapshot={"available": True, "oi_change_pct": -3.0},
            oi_rising_reject_enabled=True,
        )

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["oi_rising"])

    def test_oi_unavailable_is_not_rejected_even_when_the_gate_is_on(self):
        result = self._run(
            oi_snapshot={"available": False}, oi_rising_reject_enabled=True,
        )

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["oi_rising"])

    def test_oi_falling_is_recorded_but_does_not_block_a_signal(self):
        result = self._run(oi_snapshot={"available": True, "oi_change_pct": -3.0})

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["oi_rising"])

    def test_oi_unavailable_leaves_fields_none(self):
        result = self._run(oi_snapshot={"available": False})

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["oi_change_pct"])
        self.assertIsNone(result["oi_rising"])

    def test_oi_fields_are_none_when_confirmation_disabled(self):
        with patch.object(config, "OI_CONFIRMATION_ENABLED", False):
            result = self._run(oi_snapshot={"available": True, "oi_change_pct": 8.0})

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["oi_change_pct"])
        self.assertIsNone(result["oi_rising"])

    def test_liquidation_cluster_aligned_with_bullish_break_is_recorded(self):
        result = self._run(liquidation_snapshot={
            "available": True, "long_liquidation_notional": 80000,
            "short_liquidation_notional": 5000, "net_liquidation_notional": 75000,
        })

        self.assertEqual(result["signal"], "BUY")
        self.assertTrue(result["liquidation_cluster"])
        self.assertTrue(result["liquidation_aligned"])

    def test_liquidation_opposite_side_is_recorded_as_not_aligned(self):
        # Short-liquidation-dominant flow during a BULLISH break doesn't
        # match the "stops below the low got run" story - recorded as
        # ema_aligned=False equivalent, still not gating.
        result = self._run(liquidation_snapshot={
            "available": True, "long_liquidation_notional": 5000,
            "short_liquidation_notional": 80000, "net_liquidation_notional": -75000,
        })

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["liquidation_aligned"])

    def test_liquidation_below_cluster_threshold_is_not_a_cluster(self):
        with patch.object(config, "LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000):
            result = self._run(liquidation_snapshot={
                "available": True, "long_liquidation_notional": 1000,
                "short_liquidation_notional": 500, "net_liquidation_notional": 500,
            })

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["liquidation_cluster"])

    def test_liquidation_unavailable_leaves_fields_none(self):
        result = self._run(liquidation_snapshot={"available": False})

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["liquidation_notional_net"])
        self.assertIsNone(result["liquidation_cluster"])
        self.assertIsNone(result["liquidation_aligned"])

    def test_liquidation_fields_are_none_when_confirmation_disabled(self):
        with patch.object(config, "LIQUIDATION_CONFIRMATION_ENABLED", False):
            result = self._run(liquidation_snapshot={
                "available": True, "long_liquidation_notional": 80000,
                "short_liquidation_notional": 5000, "net_liquidation_notional": 75000,
            })

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["liquidation_notional_net"])
        self.assertIsNone(result["liquidation_cluster"])
        self.assertIsNone(result["liquidation_aligned"])

    def test_efficiency_ratio_is_read_from_ltf_analysis(self):
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.75)
        result = self._run(ltf_analysis=analysis)
        self.assertEqual(result["efficiency_ratio"], 0.75)

    def test_efficiency_ratio_defaults_to_none_when_absent(self):
        result = self._run()  # LTF_BULLISH_BREAK carries no efficiency_ratio key
        self.assertIsNone(result["efficiency_ratio"])

    def test_btc_correlation_and_alignment_are_recorded_for_a_non_reference_symbol(self):
        result = self._run(symbol="ETHUSDT", btc_correlation=0.8, btc_return=0.05)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["btc_correlation"], 0.8)
        self.assertTrue(result["btc_aligned"])  # BUY + BTC rising -> aligned

    def test_btc_misaligned_when_btc_moves_the_opposite_direction(self):
        result = self._run(symbol="ETHUSDT", btc_return=-0.05)
        self.assertFalse(result["btc_aligned"])

    def test_btc_correlation_skipped_for_the_reference_symbol_itself(self):
        # _run() defaults to symbol="BTCUSDT", the reference symbol -
        # self-correlation is meaningless, must not be computed.
        result = self._run()
        self.assertIsNone(result["btc_correlation"])
        self.assertIsNone(result["btc_aligned"])

    def test_btc_correlation_skipped_when_no_btc_candles_available(self):
        result = self._run(symbol="ETHUSDT", btc_candles=None)
        self.assertIsNone(result["btc_correlation"])
        self.assertIsNone(result["btc_aligned"])

    def test_btc_correlation_disabled_by_config(self):
        with patch.object(config, "BTC_CORRELATION_ENABLED", False):
            result = self._run(symbol="ETHUSDT")

        self.assertIsNone(result["btc_correlation"])
        self.assertIsNone(result["btc_aligned"])

    def test_btc_aligned_counts_toward_confluence_score(self):
        result = self._run(symbol="ETHUSDT", btc_return=0.05)  # aligned

        self.assertEqual(result["confluence_score"], 5)
        self.assertEqual(result["confluence_total"], 5)

    def test_funding_rate_is_carried_through(self):
        result = self._run(funding_rate=0.0003)
        self.assertEqual(result["funding_rate"], 0.0003)

    def test_funding_rate_is_none_when_disabled(self):
        with patch.object(config, "FUNDING_RATE_ENABLED", False):
            result = self._run(funding_rate=0.0003)

        self.assertIsNone(result["funding_rate"])

    def test_efficiency_favorable_true_above_the_chop_threshold(self):
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.5)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3):
            result = self._run(ltf_analysis=analysis)

        self.assertTrue(result["efficiency_favorable"])

    def test_efficiency_favorable_false_below_the_chop_threshold(self):
        # EFFICIENCY_RATIO_GATE_ENABLED off here on purpose - this test is
        # about the informational efficiency_favorable boolean specifically,
        # not the real MARKET_CHOPPY gate (see MarketChoppyGateTests) - an
        # efficiency_ratio this low would otherwise be rejected before ever
        # reaching the success dict this test reads.
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.1)

        with patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3), \
             patch.object(config, "EFFICIENCY_RATIO_GATE_ENABLED", False):
            result = self._run(ltf_analysis=analysis)

        self.assertFalse(result["efficiency_favorable"])

    def test_efficiency_favorable_is_none_when_efficiency_ratio_is_unavailable(self):
        result = self._run()  # LTF_BULLISH_BREAK carries no efficiency_ratio key
        self.assertIsNone(result["efficiency_favorable"])

    def test_funding_favorable_for_buy_below_the_adverse_threshold(self):
        with patch.object(config, "FUNDING_RATE_ADVERSE_THRESHOLD", 0.0005):
            result = self._run(funding_rate=0.0001)  # not crowded long

        self.assertTrue(result["funding_favorable"])

    def test_funding_unfavorable_for_buy_above_the_adverse_threshold(self):
        with patch.object(config, "FUNDING_RATE_ADVERSE_THRESHOLD", 0.0005):
            result = self._run(funding_rate=0.001)  # crowded long, adverse to a BUY

        self.assertFalse(result["funding_favorable"])

    def test_funding_favorable_is_mirrored_for_sell(self):
        with patch.object(config, "FUNDING_RATE_ADVERSE_THRESHOLD", 0.0005):
            # SELL favorable requires funding_rate >= -adverse (not
            # crowded-short, the squeeze-risk case for a SELL) - the
            # mirror image of the BUY case above, which requires
            # funding_rate <= +adverse (not crowded-long).
            favorable = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=LTF_BEARISH_BREAK,
                sweep_direction="BEARISH", ema_value=115.0, funding_rate=-0.0001,
            )
            unfavorable = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=LTF_BEARISH_BREAK,
                sweep_direction="BEARISH", ema_value=115.0, funding_rate=-0.001,
            )

        self.assertTrue(favorable["funding_favorable"])
        self.assertFalse(unfavorable["funding_favorable"])

    def test_funding_favorable_is_none_when_funding_rate_is_unavailable(self):
        result = self._run(funding_rate=None)
        self.assertIsNone(result["funding_favorable"])

    def test_funding_favorable_is_none_when_disabled(self):
        with patch.object(config, "FUNDING_RATE_ENABLED", False):
            result = self._run(funding_rate=0.0003)

        self.assertIsNone(result["funding_favorable"])

    def test_new_favorable_booleans_are_not_added_to_confluence(self):
        # Operator-confirmed decision: efficiency_favorable/funding_favorable/
        # long_short_favorable stay independently journaled, NOT mixed into
        # confluence_fields/confluence_ratio (that mechanism is disabled on
        # real negative evidence from its existing 5 fields - see
        # config.CONFLUENCE_SIZING_ENABLED). This locks the decision in so
        # a future edit can't silently reintroduce it.
        analysis = dict(LTF_BULLISH_BREAK, efficiency_ratio=0.9)
        result = self._run(ltf_analysis=analysis, funding_rate=0.0001)

        self.assertTrue(result["efficiency_favorable"])
        self.assertTrue(result["funding_favorable"])
        # Same confluence_total as test_confluence_score_full_agreement_
        # gives_ratio_one below (4: sweep/ema/oi/liquidation - default
        # symbol is the BTC reference symbol itself, so btc_aligned is
        # skipped) - unaffected by either new favorable boolean being true.
        self.assertEqual(result["confluence_total"], 4)


    def test_confluence_score_full_agreement_gives_ratio_one(self):
        # Defaults: sweep=BULLISH (matches direction), ema_value=85 <
        # ltf_close=93 (aligned for BUY), OI_RISING, LIQUIDATION_LONG_CLUSTER
        # (net positive, aligned for a BULLISH break) - all four agree.
        result = self._run()

        self.assertEqual(result["confluence_score"], 4)
        self.assertEqual(result["confluence_total"], 4)
        self.assertAlmostEqual(result["confluence_ratio"], 1.0)

    def test_confluence_score_counts_only_disagreeing_fields_against_it(self):
        result = self._run(
            ema_value=95.0,  # 93 < 95 -> misaligned for a BUY
            oi_snapshot={"available": True, "oi_change_pct": -3.0},  # falling
            liquidation_snapshot={
                "available": True, "long_liquidation_notional": 1000,
                "short_liquidation_notional": 9000, "net_liquidation_notional": -8000,
            },  # short-dominant during a BULLISH break -> not aligned
        )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confluence_score"], 1)  # only sweep agrees
        self.assertEqual(result["confluence_total"], 4)
        self.assertAlmostEqual(result["confluence_ratio"], 0.25)

    def test_unavailable_fields_are_excluded_from_the_denominator(self):
        result = self._run(
            ema_value=None,
            oi_snapshot={"available": False},
            liquidation_snapshot={"available": False},
        )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confluence_score"], 1)  # sweep only
        self.assertEqual(result["confluence_total"], 1)
        self.assertAlmostEqual(result["confluence_ratio"], 1.0)

    def test_disabled_confirmations_shrink_the_denominator_not_the_score(self):
        with patch.object(config, "EMA_CONFIRMATION_ENABLED", False), \
             patch.object(config, "OI_CONFIRMATION_ENABLED", False), \
             patch.object(config, "LIQUIDATION_CONFIRMATION_ENABLED", False):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confluence_score"], 1)
        self.assertEqual(result["confluence_total"], 1)
        self.assertAlmostEqual(result["confluence_ratio"], 1.0)

    def test_no_signal_when_order_flow_data_unavailable(self):
        result = self._run(cvd={"available": False})
        self.assertEqual(result["reason"], "ORDER_FLOW_DATA_UNAVAILABLE")

    def test_no_signal_when_cvd_score_missing(self):
        result = self._run(cvd={"available": True, "cvd_score": None})
        self.assertEqual(result["reason"], "ORDER_FLOW_SCORE_UNAVAILABLE")

    def test_no_signal_when_cvd_not_confirmed_for_buy(self):
        with patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15):
            result = self._run(cvd={"available": True, "cvd_score": 0.05})

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    def test_no_signal_when_depth_opposing_for_buy(self):
        with patch.object(config, "SIGNAL_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(depth={"available": True, "depth_imbalance": -0.5})

        self.assertIn("DEPTH_OPPOSING", result["reason"])

    def test_signal_produced_when_depth_data_unavailable(self):
        result = self._run(depth={"available": False})
        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["depth_imbalance"])

    def test_trigger_candle_open_time_comes_from_the_live_break(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = dict(LTF_BULLISH_BREAK["live_break"], open_time=456)
        result = self._run(ltf_analysis=analysis)

        self.assertEqual(result["trigger_candle_open_time"], 456)

    def test_no_signal_when_below_the_liquidity_floor(self):
        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 3_000_000):
            result = self._run(quote_volume_usdt=1_500_000)

        self.assertIsNone(result["signal"])
        self.assertIn("QUOTE_VOLUME_TOO_LOW", result["reason"])

    def test_signal_allowed_at_or_above_the_liquidity_floor(self):
        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 3_000_000):
            result = self._run(quote_volume_usdt=3_000_000)

        self.assertEqual(result["signal"], "BUY")

    def test_missing_volume_data_does_not_block_the_signal(self):
        # Never gate on data we don't actually have - a symbol whose
        # volume poll hasn't completed yet must not be silently excluded.
        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 3_000_000):
            result = self._run(quote_volume_usdt=None)

        self.assertEqual(result["signal"], "BUY")

    def test_liquidity_floor_disabled_allows_thin_symbols(self):
        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 0):
            result = self._run(quote_volume_usdt=100)

        self.assertEqual(result["signal"], "BUY")

    def test_quote_volume_usdt_is_carried_through_to_the_result(self):
        with patch.object(config, "MIN_24H_QUOTE_VOLUME_USDT", 0):
            result = self._run(quote_volume_usdt=42_000_000)

        self.assertEqual(result["quote_volume_usdt"], 42_000_000)

    def test_no_signal_without_ltf_candles(self):
        result = signal_engine.evaluate("BTCUSDT", ["htf"], [], {}, {})
        self.assertEqual(result["reason"], "INSUFFICIENT_CANDLES")

    # config.LIQUIDITY_SWEEP_TRIGGER_ENABLED - a second, alternative entry
    # trigger alongside a live LTF structure break, feeding the exact same
    # downstream pipeline (never a second independent pipeline - see
    # config.py's comment for why).

    def test_sweep_only_candidate_rejects_when_sweep_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run(ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_sweep_triggered_signal_when_no_break_but_sweep_trigger_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "LIQUIDITY_SWEEP")
        self.assertEqual(result["structure_level"], 89)
        self.assertIsNone(result["trigger_candle_open_time"])
        self.assertTrue(result["sweep_confluence"])  # sweep is its own trigger, necessarily aligned

    def test_sweep_trigger_candle_open_time_comes_from_the_sweep_itself(self):
        # Now that detect_sweep can be close-candle-gated, its own
        # open_time (whichever candle it actually tested) is the real
        # trigger_candle_open_time - not hardcoded None.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89,
                sweep_open_time=789,
            )

        self.assertEqual(result["trigger_candle_open_time"], 789)

    def test_structure_break_takes_priority_over_a_simultaneous_sweep(self):
        # LTF_BULLISH_BREAK (level=90) is still active; sweep direction
        # deliberately conflicts (BEARISH) to prove the break wins outright
        # and sweep_confluence stays independently honest about the
        # disagreement, rather than silently blending the two.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(sweep_direction="BEARISH", sweep_level=77)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result["structure_level"], 90)  # from the break, not sweep's 77
        self.assertFalse(result["sweep_confluence"])

    def test_no_signal_when_neither_break_nor_sweep_even_with_trigger_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_sweep_triggered_signal_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_sweep_triggered_signal_still_requires_order_block_or_fvg(self):
        analysis = dict(LTF_BULLISH_BREAK, fair_value_gaps=[])
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "REQUIRE_ORDER_BLOCK_OR_FVG", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89, order_block=None,
            )

        self.assertEqual(result["reason"], "NO_ORDER_BLOCK_OR_FVG")

    def test_sweep_triggered_signal_still_gated_by_cvd(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15):
            result = self._run(
                ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89,
                cvd={"available": True, "cvd_score": 0.05},
            )

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    def test_pools_and_sweep_computed_exactly_once_when_break_triggered_flag_off(self):
        with patch.object(market_structure, "structure_state", return_value=HTF_BULLISH), \
             patch.object(market_structure, "premium_discount_zone", return_value=ZONE), \
             patch.object(market_structure, "analyze", return_value=LTF_BULLISH_BREAK), \
             patch.object(market_structure, "find_order_block", return_value=None), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]) as mock_pools, \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "exponential_moving_average", return_value=85.0), \
             patch.object(market_structure, "price_correlation", return_value=0.5), \
             patch.object(market_structure, "price_return", return_value=0.02), \
             patch.object(liquidity_sweep, "detect_sweep", return_value={"direction": "BULLISH", "level": 89}) as mock_detect, \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False), \
             patch.object(config, "OI_RISING_REJECT_ENABLED", False):
            signal_engine.evaluate(
                "BTCUSDT", ["htf_placeholder"], _ltf_candles(93.0),
                {"available": True, "cvd_score": 0.5}, {"available": True, "depth_imbalance": 0.2},
                oi_snapshot=OI_RISING, liquidation_snapshot=LIQUIDATION_LONG_CLUSTER,
            )

        self.assertEqual(mock_pools.call_count, 1)
        self.assertEqual(mock_detect.call_count, 1)

    def test_pools_and_sweep_computed_exactly_once_when_break_triggered_flag_on(self):
        with patch.object(market_structure, "structure_state", return_value=HTF_BULLISH), \
             patch.object(market_structure, "premium_discount_zone", return_value=ZONE), \
             patch.object(market_structure, "analyze", return_value=LTF_BULLISH_BREAK), \
             patch.object(market_structure, "find_order_block", return_value=None), \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]) as mock_pools, \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "exponential_moving_average", return_value=85.0), \
             patch.object(market_structure, "price_correlation", return_value=0.5), \
             patch.object(market_structure, "price_return", return_value=0.02), \
             patch.object(liquidity_sweep, "detect_sweep", return_value={"direction": "BULLISH", "level": 89}) as mock_detect, \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_RISING_REJECT_ENABLED", False):
            signal_engine.evaluate(
                "BTCUSDT", ["htf_placeholder"], _ltf_candles(93.0),
                {"available": True, "cvd_score": 0.5}, {"available": True, "depth_imbalance": 0.2},
                oi_snapshot=OI_RISING, liquidation_snapshot=LIQUIDATION_LONG_CLUSTER,
            )

        self.assertEqual(mock_pools.call_count, 1)
        self.assertEqual(mock_detect.call_count, 1)

    # config.OB_FVG_RETEST_TRIGGER_ENABLED - a fresh rejection wick into an
    # unmitigated FVG, independent of any live break right now. Priority:
    # STRUCTURE_BREAK > OB_FVG_RETEST > LIQUIDITY_SWEEP > CHOCH_RETEST.

    def test_fvg_retest_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_fvg_retest_triggered_signal_when_no_break_but_trigger_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")
        self.assertEqual(result["structure_level"], 90)
        # Unlike LIQUIDITY_SWEEP/CHOCH_RETEST, this trigger's defining event
        # IS the current forming candle - trigger_candle_open_time is real,
        # not None, so resolve_break_confirmations can validate it.
        self.assertEqual(result["trigger_candle_open_time"], 0)

    def test_ob_fvg_retest_trigger_candle_open_time_comes_from_the_retest_itself(self):
        # find_fvg_retest can now be close-candle-gated, so its own
        # open_time (whichever candle it actually tested) may differ from
        # ltf_candles[-1] - confirm the real value flows through, not a
        # blind read of the candle list.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                fvg_retest_open_time=321,
            )

        self.assertEqual(result["trigger_candle_open_time"], 321)

    def test_structure_break_takes_priority_over_ob_fvg_retest(self):
        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True):
            result = self._run(fvg_retest_direction="BEARISH", fvg_retest_level=77)

        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result["structure_level"], 90)

    def test_ob_fvg_retest_takes_priority_over_liquidity_sweep(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis,
                sweep_direction="BULLISH", sweep_level=89,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
            )

        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")
        self.assertEqual(result["structure_level"], 90)

    def test_ob_fvg_retest_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_ob_fvg_retest_still_gated_by_cvd(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                cvd={"available": True, "cvd_score": 0.05},
            )

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    # config.CHOCH_RETEST_TRIGGER_ENABLED - an already-CONFIRMED reversal
    # (last_event type CHoCH), retested within CHOCH_TRIGGER_MAX_AGE_CANDLES.
    # Ranked last in priority: STRUCTURE_BREAK > OB_FVG_RETEST >
    # LIQUIDITY_SWEEP > CHOCH_RETEST.

    def _choch_analysis(self, direction, event_price, event_index=0, event_type="CHoCH"):
        analysis = dict(LTF_BULLISH_BREAK if direction == "BULLISH" else LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}
        analysis["last_event"] = {
            "type": event_type, "direction": direction, "index": event_index,
            "price": event_price,
        }
        analysis["last_swing_low"] = 88
        analysis["last_swing_high"] = 112
        return analysis

    def test_choch_retest_only_candidate_rejects_when_trigger_disabled(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", False):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_choch_retest_triggered_signal_when_recent_and_enabled(self):
        # event_price=95 deliberately does NOT equal last_swing_low(88) -
        # proves structure_level comes from the swing level, not the event.
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")
        self.assertEqual(result["structure_level"], 88)  # last_swing_low, NOT event_price(95)
        self.assertIsNone(result["trigger_candle_open_time"])

    def test_choch_retest_uses_swing_high_not_event_price_for_sell(self):
        analysis = self._choch_analysis("BEARISH", event_price=85)  # a LOW, deliberately wrong if used

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
            )

        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")
        self.assertEqual(result["structure_level"], 112)  # last_swing_high, NOT event_price(85)

    def test_choch_retest_ignored_when_event_too_old(self):
        # _ltf_candles() always produces a single candle at index 0, so
        # "age" (len(ltf_candles)-1 - event_index) is 0 unless event_index
        # is negative - simulating an event that happened well before the
        # start of this LTF buffer.
        analysis = self._choch_analysis("BULLISH", event_price=95, event_index=-10)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_TRIGGER_MAX_AGE_CANDLES", 5):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_choch_retest_ignored_when_last_event_is_bos_not_choch(self):
        analysis = self._choch_analysis("BULLISH", event_price=95, event_type="BOS")

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_liquidity_sweep_takes_priority_over_choch_retest(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89,
            )

        self.assertEqual(result["signal_trigger"], "LIQUIDITY_SWEEP")

    def test_choch_retest_still_respects_htf_bias(self):
        # AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED defaults True
        # in this project (CHOCH_RETEST is exempt by default - see
        # config.py's TRIGGER_GATE_PROFILES section) - pinned off here so
        # this test keeps proving the gate itself still works correctly
        # when that exemption isn't in play, same "_still_applies_when_
        # scoped_off" pattern the OTE/MARKET_CHOPPY/CVD_NOT_CONFIRMED
        # tests already use.
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(ltf_analysis=analysis, sweep_direction=None, htf_structure=HTF_BEARISH)

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_choch_retest_still_gated_by_cvd(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                cvd={"available": True, "cvd_score": 0.05},
            )

        self.assertIn("CVD_NOT_CONFIRMED", result["reason"])

    # Regression coverage for the structure_level bug caught during plan
    # review: feeding a trigger's signal into a REAL risk_manager call must
    # land the SL on the structurally correct side of entry, for both new
    # triggers and both directions - the exact check that would have caught
    # CHOCH_RETEST originally using last_event["price"] (a fresh HIGH/LOW,
    # the wrong side) instead of last_swing_low/last_swing_high.

    def test_choch_retest_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        # MAX_ENTRY_EXTENSION_R disabled: this test isolates SL-sidedness
        # only, not the (unrelated) entry-extension gate - a retracement
        # level several R away from entry is a realistic CHOCH_RETEST shape
        # but would otherwise trip ENTRY_TOO_EXTENDED for reasons that have
        # nothing to do with what this test is checking.
        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_choch_retest_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = self._choch_analysis("BEARISH", event_price=85)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    def test_ob_fvg_retest_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_ob_fvg_retest_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                fvg_retest_direction="BEARISH", fvg_retest_level=110,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    # config.CVD_DIVERGENCE_TRIGGER_ENABLED - price's swing structure vs
    # the CVD line at those same swing points (cvd_divergence.py). Ranked
    # last in priority: STRUCTURE_BREAK > OB_FVG_RETEST > LIQUIDITY_SWEEP
    # > CHOCH_RETEST > CVD_DIVERGENCE. detect_divergence() itself is
    # mocked here (see _run()'s divergence_* params) - its own logic is
    # covered directly in test_cvd_divergence.py; these tests only prove
    # signal_engine wires the candidate correctly (age gate, flag gate,
    # structure_level/trigger_candle_open_time, priority).

    def test_cvd_divergence_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_cvd_divergence_triggered_signal_when_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                divergence_open_time=555,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")
        self.assertEqual(result["structure_level"], 88)
        # Deliberately None, not divergence_open_time - see signal_engine.py's
        # comment: the divergence's open_time is the OLD swing candle, not
        # a candle being entered on now, so no close-confirmation applies.
        self.assertIsNone(result["trigger_candle_open_time"])

    def test_cvd_divergence_ignored_when_too_old(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        # _ltf_candles() always produces a single candle at index 0, so
        # "age" (len(ltf_candles)-1 - divergence_index) is 0 unless
        # divergence_index is negative.
        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "ORDER_FLOW_DIVERGENCE_LOOKBACK", 5):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                divergence_index=-10,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_choch_retest_takes_priority_over_cvd_divergence(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
            )

        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

    def test_cvd_divergence_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_cvd_divergence_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_cvd_divergence_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                divergence_direction="BEARISH", divergence_level=112,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    # config.ORDER_BLOCK_RETEST_TRIGGER_ENABLED - a fresh rejection wick
    # back into a previously-formed, unmitigated order block. Ranked
    # after CVD_DIVERGENCE: STRUCTURE_BREAK > OB_FVG_RETEST > LIQUIDITY_
    # SWEEP > CHOCH_RETEST > CVD_DIVERGENCE > ORDER_BLOCK_RETEST >
    # OI_DIVERGENCE > LIQUIDATION_SWEEP_CONFIRMED. find_order_block_retest
    # itself is mocked (see _run()'s order_block_retest_* params) - its
    # own logic is covered in test_market_structure.py.

    def test_order_block_retest_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_order_block_retest_triggered_signal_when_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
                order_block_retest_open_time=321,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "ORDER_BLOCK_RETEST")
        self.assertEqual(result["structure_level"], 88)
        self.assertEqual(result["trigger_candle_open_time"], 321)

    def test_cvd_divergence_takes_priority_over_order_block_retest(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True):
            # Same level on both candidates - a tied _score() always
            # favors the fixed-priority default regardless of whether
            # TRIGGER_QUALITY_RANKING_ENABLED happens to be True in the
            # loaded .env (min() keeps the first equal-scored candidate).
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
            )

        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")

    def test_order_block_retest_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_order_block_retest_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_order_block_retest_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                order_block_retest_direction="BEARISH", order_block_retest_level=112,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    # config.OI_DIVERGENCE_TRIGGER_ENABLED - price's swing structure vs
    # open interest at those same swing points. Ranked after
    # ORDER_BLOCK_RETEST. oi_divergence.detect_divergence itself is
    # mocked (see _run()'s oi_divergence_* params) - its own logic is
    # covered in test_oi_divergence.py.

    def test_oi_divergence_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_oi_divergence_triggered_signal_when_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OI_DIVERGENCE")
        self.assertEqual(result["structure_level"], 88)
        self.assertIsNone(result["trigger_candle_open_time"])

    def test_oi_divergence_ignored_when_too_old(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_MAX_AGE_CANDLES", 5):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
                oi_divergence_index=-10,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_order_block_retest_takes_priority_over_oi_divergence(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True):
            # Same level on both candidates - see the identical note on
            # test_cvd_divergence_takes_priority_over_order_block_retest.
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                order_block_retest_direction="BULLISH", order_block_retest_level=88,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
            )

        self.assertEqual(result["signal_trigger"], "ORDER_BLOCK_RETEST")

    def test_oi_divergence_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_oi_divergence_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_oi_divergence_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                oi_divergence_direction="BEARISH", oi_divergence_level=112,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    # config.LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED - a plain
    # LIQUIDITY_SWEEP additionally confirmed by a real clustered forced-
    # liquidation event. Ranked last of all 8 triggers.
    # liquidity_sweep.detect_liquidation_confirmed_sweep itself is mocked
    # (see _run()'s liquidation_confirmed_sweep_* params) - its own logic
    # is covered in test_liquidity_sweep.py.

    def test_liquidation_sweep_confirmed_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_liquidation_sweep_confirmed_triggered_signal_when_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
                liquidation_confirmed_sweep_open_time=654,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "LIQUIDATION_SWEEP_CONFIRMED")
        self.assertEqual(result["structure_level"], 88)
        self.assertEqual(result["trigger_candle_open_time"], 654)

    def test_oi_divergence_takes_priority_over_liquidation_sweep_confirmed(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True):
            # Same level on both candidates - see the identical note on
            # test_cvd_divergence_takes_priority_over_order_block_retest.
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                oi_divergence_direction="BULLISH", oi_divergence_level=88,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
            )

        self.assertEqual(result["signal_trigger"], "OI_DIVERGENCE")

    def test_liquidation_sweep_confirmed_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_liquidation_sweep_confirmed_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_liquidation_sweep_confirmed_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                liquidation_confirmed_sweep_direction="BEARISH",
                liquidation_confirmed_sweep_level=112,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])

    # config.EMA_PULLBACK_TRIGGER_ENABLED - a pullback to the EMA
    # followed by a same-candle reclaim. Ranked last of all 9 triggers.
    # market_structure.detect_ema_pullback itself is mocked (see _run()'s
    # ema_pullback_* params) - its own logic is covered in
    # test_market_structure.py.

    def test_ema_pullback_only_candidate_rejects_when_trigger_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                ema_pullback_direction="BULLISH", ema_pullback_level=88,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_ema_pullback_triggered_signal_when_enabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                ema_pullback_direction="BULLISH", ema_pullback_level=88,
                ema_pullback_open_time=321,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "EMA_PULLBACK")
        self.assertEqual(result["structure_level"], 88)
        self.assertEqual(result["trigger_candle_open_time"], 321)

    def test_liquidation_sweep_confirmed_takes_priority_over_ema_pullback(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True):
            # Same level on both candidates - a tied _score() always
            # favors the fixed-priority default regardless of whether
            # TRIGGER_QUALITY_RANKING_ENABLED happens to be True in the
            # loaded .env (min() keeps the first equal-scored candidate).
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                liquidation_confirmed_sweep_direction="BULLISH",
                liquidation_confirmed_sweep_level=88,
                ema_pullback_direction="BULLISH", ema_pullback_level=88,
            )

        self.assertEqual(result["signal_trigger"], "LIQUIDATION_SWEEP_CONFIRMED")

    def test_ema_pullback_still_respects_htf_bias(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                ema_pullback_direction="BULLISH", ema_pullback_level=88,
                htf_structure=HTF_BEARISH,
            )

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_ema_pullback_signal_produces_a_correctly_sided_stop_loss_for_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                ema_pullback_direction="BULLISH", ema_pullback_level=88,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertLess(plan["sl_price"], plan["entry_price"])

    def test_ema_pullback_signal_produces_a_correctly_sided_stop_loss_for_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                ema_pullback_direction="BEARISH", ema_pullback_level=112,
            )

            plan, status = risk_manager.build_trade_plan(result, balance=1000)

        self.assertEqual(status, "OK")
        self.assertGreater(plan["sl_price"], plan["entry_price"])


    # config.TRIGGER_QUALITY_RANKING_ENABLED - instead of the fixed
    # priority order (first candidate wins), gate every currently-
    # qualifying candidate and rank the survivors by objective quality
    # (distance from current price to the trigger level), overriding the
    # fixed-priority default only when the edge is real - see
    # config.TRIGGER_QUALITY_EDGE_ATR_MULTIPLE. Default OFF - every test
    # above this point already proves the disabled path is byte-identical
    # to the old if/elif chain (no changes were needed to any of them).

    def test_ranking_overrides_the_default_when_the_edge_is_real(self):
        # STRUCTURE_BREAK (level=90, distance=3 from entry=93) is the
        # fixed-priority default; LIQUIDITY_SWEEP (level=92, distance=1)
        # is closer by 2, well over the default 0.25*ATR(1.0) edge.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_EDGE_ATR_MULTIPLE", 0.25):
            result = self._run(sweep_direction="BULLISH", sweep_level=92)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "LIQUIDITY_SWEEP")
        self.assertEqual(result["structure_level"], 92)

    def test_ranking_sticks_with_the_default_when_the_edge_is_too_small(self):
        # LIQUIDITY_SWEEP (level=90.1, distance=2.9) is objectively closer
        # than STRUCTURE_BREAK (level=90, distance=3), but only by 0.1 -
        # under the default 0.25*ATR edge, so the fixed-priority default
        # must still win (the hysteresis margin exists precisely to
        # prevent this kind of marginal difference from deciding it).
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_EDGE_ATR_MULTIPLE", 0.25):
            result = self._run(sweep_direction="BULLISH", sweep_level=90.1)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result["structure_level"], 90)

    def test_zero_edge_multiple_always_takes_the_best_scored_candidate(self):
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_EDGE_ATR_MULTIPLE", 0):
            result = self._run(sweep_direction="BULLISH", sweep_level=90.1)

        self.assertEqual(result["signal_trigger"], "LIQUIDITY_SWEEP")

    def test_ranking_does_not_crash_when_a_candidates_structure_level_is_none(self):
        # A candidate's structure_level can legitimately be None (e.g.
        # CHOCH_RETEST when only one side of last_swing_high/low has
        # formed yet) - real bug found while verifying this session's
        # fixes: ranking's abs(latest_price - structure_level) crashed
        # with a TypeError instead of just deprioritizing it. The
        # structure-break default (a real level) must still win.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True):
            result = self._run(sweep_direction="BULLISH", sweep_level=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result["structure_level"], 90)

    def test_ranking_recovers_a_signal_the_fixed_priority_default_would_have_missed(self):
        # STRUCTURE_BREAK fires BEARISH here - against HTF_BULLISH bias,
        # so it's the only candidate the OLD fixed-priority chain would
        # ever attempt, and it fails AGAINST_HTF_BIAS outright. A
        # simultaneous LIQUIDITY_SWEEP fires BULLISH (agrees with HTF
        # bias) and would pass every gate. With ranking OFF this is a
        # missed setup (matches today's real behavior); with ranking ON,
        # the whole point of this feature, it's recovered.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": True, "direction": "BEARISH", "level": 110}

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", False):
            off_result = self._run(ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89)

        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True):
            on_result = self._run(ltf_analysis=analysis, sweep_direction="BULLISH", sweep_level=89)

        self.assertIsNone(off_result["signal"])
        self.assertIn("AGAINST_HTF_BIAS", off_result["reason"])

        self.assertEqual(on_result["signal"], "BUY")
        self.assertEqual(on_result["signal_trigger"], "LIQUIDITY_SWEEP")
        self.assertEqual(on_result["structure_level"], 89)

    def test_direction_pipeline_runs_once_per_distinct_trigger_not_once_per_direction(self):
        # STRUCTURE_BREAK and LIQUIDITY_SWEEP both fire BULLISH here -
        # ranking has real candidates to choose between. Since
        # OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED (and any future gate like
        # it) means two same-direction candidates can now get genuinely
        # different verdicts, the shared gate pipeline is deduped on
        # (direction, trigger), not direction alone - so it runs once per
        # distinct trigger sharing this direction (2 here), not once
        # total. find_liquidity_pools/detect_sweep stay at 1 regardless -
        # both are cached via `nonlocal` at the top of evaluate() itself,
        # computed once before any per-direction pipeline run happens.
        # LIQUIDATION_SWEEP_CONFIRMED explicitly disabled - real .env has
        # it on, and it would otherwise also fire BULLISH here (a real,
        # currently-enabled 9th trigger, not part of what this test is
        # isolating).
        with patch.object(market_structure, "structure_state", return_value=HTF_BULLISH), \
             patch.object(market_structure, "premium_discount_zone", return_value=ZONE), \
             patch.object(market_structure, "analyze", return_value=LTF_BULLISH_BREAK), \
             patch.object(market_structure, "find_order_block", return_value=None) as mock_ob, \
             patch.object(market_structure, "find_liquidity_pools", return_value=[]) as mock_pools, \
             patch.object(market_structure, "find_swing_points", return_value=[]), \
             patch.object(market_structure, "find_fvg_retest", return_value=None), \
             patch.object(market_structure, "exponential_moving_average", return_value=85.0), \
             patch.object(market_structure, "price_correlation", return_value=0.5), \
             patch.object(market_structure, "price_return", return_value=0.02), \
             patch.object(liquidity_sweep, "detect_sweep", return_value={"direction": "BULLISH", "level": 92}) as mock_detect, \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True), \
             patch.object(config, "OI_RISING_REJECT_ENABLED", False):
            result = signal_engine.evaluate(
                "BTCUSDT", ["htf_placeholder"], _ltf_candles(93.0),
                {"available": True, "cvd_score": 0.5}, {"available": True, "depth_imbalance": 0.2},
                oi_snapshot=OI_RISING, liquidation_snapshot=LIQUIDATION_LONG_CLUSTER,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(mock_pools.call_count, 1)
        self.assertEqual(mock_detect.call_count, 1)
        self.assertEqual(mock_ob.call_count, 2)

    def test_hysteresis_keeps_the_winner_stable_across_a_small_price_move(self):
        # Same two close-scoring candidates as the "sticks with default"
        # test above, evaluated at two slightly different entry prices
        # (ordinary tick-to-tick noise) - the winner must not flip, which
        # is exactly what would otherwise reset main.py's
        # SignalStabilityTracker streak every tick and starve the setup.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_RANKING_ENABLED", True), \
             patch.object(config, "TRIGGER_QUALITY_EDGE_ATR_MULTIPLE", 0.25):
            result_a = self._run(ltf_close=93.0, sweep_direction="BULLISH", sweep_level=90.1)
            result_b = self._run(ltf_close=93.05, sweep_direction="BULLISH", sweep_level=90.1)

        self.assertEqual(result_a["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result_b["signal_trigger"], "STRUCTURE_BREAK")


class RejectTriggerTaggingTests(unittest.TestCase):
    """A direction-level rejection (inside _evaluate_direction) now also
    carries which trigger(s) actually attempted that direction - purely
    additive (a new "triggers" key; `reason` itself is untouched, so this
    can never affect main.py's existing reject-reason tally or any test
    asserting on `reason` alone). Delegates to SignalEngineTests._run
    (verified self-independent) rather than subclassing it, which would
    otherwise re-run all of SignalEngineTests' own tests again under this
    class too."""

    def _run(self, **kwargs):
        return SignalEngineTests._run(self, **kwargs)

    def test_single_trigger_is_tagged_on_rejection(self):
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run(htf_structure=HTF_BEARISH)  # AGAINST_HTF_BIAS

        self.assertTrue(result["reason"].startswith("AGAINST_HTF_BIAS"))
        self.assertEqual(result["triggers"], ["STRUCTURE_BREAK"])

    def test_multiple_triggers_sharing_the_direction_are_all_tagged(self):
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                htf_structure=HTF_BEARISH,  # AGAINST_HTF_BIAS for both
                sweep_direction="BULLISH", sweep_level=90.0,
            )

        self.assertTrue(result["reason"].startswith("AGAINST_HTF_BIAS"))
        self.assertEqual(result["triggers"], ["LIQUIDITY_SWEEP", "STRUCTURE_BREAK"])

    def test_reason_string_itself_is_unaffected_by_tagging(self):
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run(htf_structure=HTF_BEARISH)

        self.assertEqual(result["reason"], "AGAINST_HTF_BIAS htf=BEARISH ltf=BULLISH")

    def test_early_rejection_before_any_candidate_has_no_triggers_key(self):
        # NO_LIVE_STRUCTURE_BREAK fires before any direction/candidate is
        # resolved - _evaluate_direction (and its trigger tagging) is
        # never reached at all.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run(ltf_analysis={
                "available": True, "live_break": {"broken": False},
                "fair_value_gaps": [], "atr": 1.0,
            })

        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")
        self.assertNotIn("triggers", result)

    def test_passing_signal_has_no_triggers_key(self):
        # triggers is a rejection-diagnostic field only - a real signal
        # already carries signal_trigger (the winning candidate's own
        # trigger), which is the meaningful field once something passes.
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")
        self.assertNotIn("triggers", result)


class LongShortFavorableTests(unittest.TestCase):
    """signal_engine.long_short_favorable - called from main.py once the
    on-demand long_short_ratio fetch resolves (see
    config.LONG_SHORT_RATIO_ENABLED for why the raw value can't be
    computed inside evaluate() itself)."""

    def test_none_ratio_is_none(self):
        self.assertIsNone(signal_engine.long_short_favorable("BUY", None))

    def test_buy_favorable_below_the_crowd_threshold(self):
        with patch.object(config, "LONG_SHORT_RATIO_CROWD_THRESHOLD", 2.0):
            self.assertTrue(signal_engine.long_short_favorable("BUY", 1.5))

    def test_buy_unfavorable_at_or_above_the_crowd_threshold(self):
        with patch.object(config, "LONG_SHORT_RATIO_CROWD_THRESHOLD", 2.0):
            self.assertFalse(signal_engine.long_short_favorable("BUY", 2.5))

    def test_sell_favorable_above_the_inverse_crowd_threshold(self):
        # threshold=2.0 -> SELL favorable above 1/2.0 = 0.5
        with patch.object(config, "LONG_SHORT_RATIO_CROWD_THRESHOLD", 2.0):
            self.assertTrue(signal_engine.long_short_favorable("SELL", 0.6))

    def test_sell_unfavorable_at_or_below_the_inverse_crowd_threshold(self):
        with patch.object(config, "LONG_SHORT_RATIO_CROWD_THRESHOLD", 2.0):
            self.assertFalse(signal_engine.long_short_favorable("SELL", 0.4))


if __name__ == "__main__":
    unittest.main()
