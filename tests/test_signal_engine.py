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
        fvg_retest_index=0,
        fvg_retest_tested_index=0,
        divergence_direction=None,
        divergence_level=None,
        divergence_index=5,
        divergence_open_time=999,
        order_block_retest_direction=None,
        order_block_retest_level=None,
        order_block_retest_open_time=777,
        order_block_retest_index=0,
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
        ema_alignment_value=85.0,
        htf_trend_ema=None,
        htf_trend_ema_prior=None,
        oi_snapshot=None,
        liquidation_snapshot=None,
        quote_volume_usdt=None,
        btc_candles="default",
        btc_correlation=0.5,
        btc_return=0.02,
        funding_rate=None,
        oi_rising_reject_enabled=False,
        crash_snapshot=None,
        oi_snapshot_bybit=None,
        oi_snapshot_okx=None,
        volume_profile_snapshot=None,
        htf_candles=None,
        htf_trend_ema_primary_enabled=False,
        liquidation_snapshot_bybit=None,
        liquidation_snapshot_okx=None,
    ):
        cvd = {"available": True, "cvd_score": 0.5} if cvd is None else cvd
        depth = {"available": True, "depth_imbalance": 0.2} if depth is None else depth
        htf_structure = HTF_BULLISH if htf_structure is None else htf_structure
        # A real (if inert - structure_state is always mocked below)
        # candle dict, not a bare placeholder string - config.
        # HTF_TREND_EMA_PRIMARY_ENABLED's htf_trend_live computation
        # indexes into this directly (htf_candles[-1]["close"]), unlike
        # structure_state's own swing detection which never touches it.
        htf_candles = (
            [{"open_time": 0, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1, "closed": True}]
            if htf_candles is None else htf_candles
        )
        zone = ZONE if zone is None else zone
        ltf_analysis = LTF_BULLISH_BREAK if ltf_analysis is None else ltf_analysis
        sweep = (
            {"direction": sweep_direction, "level": sweep_level, "open_time": sweep_open_time}
            if sweep_direction else None
        )
        fvg_retest = (
            {
                "direction": fvg_retest_direction, "level": fvg_retest_level,
                "gap": {"index": fvg_retest_index},
                "open_time": fvg_retest_open_time,
                # market_structure.find_fvg_retest's own age-gate/setup_age_
                # candles reference index - see its docstring for why this
                # is distinct from len(ltf_candles)-1.
                "tested_index": fvg_retest_tested_index,
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
                "block": {"index": order_block_retest_index},
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

        # Three distinct callers share this one mocked function: the
        # existing informational LTF ema_value (called with just
        # ltf_candles, no period=), the faster ema_alignment_value used
        # only for ema_aligned (always called with period=
        # config.EMA_ALIGNMENT_PERIOD), and the HTF_TREND_FRESHNESS_ENABLED
        # gate's htf_trend_ema (always called with period=
        # config.HTF_TREND_EMA_PERIOD - see signal_engine.py). Routing on
        # the exact period value keeps all three independently
        # controllable - htf_trend_ema defaults to None (gate is a no-op)
        # so no existing test is affected unless it explicitly opts in.
        def _ema_side_effect(candles, period=None):
            if period is None:
                return ema_value
            if period == config.EMA_ALIGNMENT_PERIOD:
                return ema_alignment_value
            return htf_trend_ema

        # config.HTF_TREND_LIVE_STRENGTH_REJECT_ENABLED's slope input -
        # mocked as its own separate function (not a second exponential_
        # moving_average call) precisely so it's independently
        # controllable here, unlike a second call with sliced candles
        # would be under _ema_side_effect above (keyed only on period).
        # Defaults to None (gate's slope check is a no-op) so no existing
        # test is affected unless it explicitly opts in.

        # OI_RISING_REJECT_ENABLED defaults True in config.py (real gate,
        # see its own comment) but oi_snapshot's own default above
        # (OI_RISING) is rising - defaulting this kwarg to False keeps
        # every test above and below that doesn't care about this specific
        # gate exercising its old, pre-gate behavior; opt in explicitly
        # (oi_rising_reject_enabled=True) to test the gate itself.
        #
        # HTF_TREND_EMA_PRIMARY_ENABLED defaults True in the real live
        # .env (explicit live test, see config.py's own comment) - without
        # pinning it False here, AGAINST_HTF_BIAS would silently switch to
        # reading htf_trend_live for every test in this suite, which stays
        # None unless a test explicitly sets up htf_trend_ema, turning the
        # gate into a silent no-op for tests that rely on it firing (e.g.
        # test_no_signal_against_htf_bias). Same insulation pattern as
        # oi_rising_reject_enabled above.
        with patch.object(config, "OI_RISING_REJECT_ENABLED", oi_rising_reject_enabled), \
             patch.object(config, "HTF_TREND_EMA_PRIMARY_ENABLED", htf_trend_ema_primary_enabled), \
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
             patch.object(market_structure, "ema_prior_value", return_value=htf_trend_ema_prior), \
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
                symbol, htf_candles, _ltf_candles(ltf_close), cvd, depth,
                oi_snapshot=oi_snapshot, liquidation_snapshot=liquidation_snapshot,
                quote_volume_usdt=quote_volume_usdt, btc_candles=btc_candles,
                funding_rate=funding_rate, crash_snapshot=crash_snapshot,
                oi_snapshot_bybit=oi_snapshot_bybit, oi_snapshot_okx=oi_snapshot_okx,
                volume_profile_snapshot=volume_profile_snapshot,
                liquidation_snapshot_bybit=liquidation_snapshot_bybit,
                liquidation_snapshot_okx=liquidation_snapshot_okx,
            )

    def test_full_buy_signal_when_everything_aligns(self):
        result = self._run()

        self.assertEqual(result["signal"], "BUY")
        self.assertTrue(result["sweep_confluence"])
        self.assertEqual(result["premium_discount_zone"], "DISCOUNT")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        # (range_high=110 - entry=93) / (range_high=110 - range_low=90)
        self.assertAlmostEqual(result["zone_retracement_pct"], 0.85)

    def test_success_dict_carries_the_fair_value_gaps_used_for_this_eval(self):
        # config.RETRACEMENT_STRUCTURE_TARGET_ENABLED -
        # risk_manager.build_trade_plan carries this onto the plan so
        # execution.enter_trade_retracement can consider a real structural
        # level instead of only a synthetic R-fraction. Zero extra cost -
        # the same ltf_analysis["fair_value_gaps"] already computed for
        # every eval tick regardless.
        result = self._run()

        self.assertEqual(result["fair_value_gaps"], LTF_BULLISH_BREAK["fair_value_gaps"])

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

    # config.HTF_TREND_SWING_AGE_REJECT_ENABLED - EXPLICIT LIVE TEST, no
    # resolved-trade evidence supporting it (see config.py's own comment).
    # market_structure.structure_state() itself is mocked via htf_structure
    # in this test suite - these tests inject a real last_event/htf_candles
    # pair since the age computation reads htf_candles directly, not
    # through the mock.

    def test_htf_trend_swing_age_is_computed_from_real_candles(self):
        htf_candles = [{"open_time": i * 14_400_000} for i in range(20)]
        htf_structure = dict(
            HTF_BULLISH, last_event={"index": 17, "type": "BOS", "direction": "BULLISH", "price": 100},
        )

        result = self._run(htf_structure=htf_structure, htf_candles=htf_candles)

        # (19 - 17) candles * 4h/candle = 8 hours
        self.assertAlmostEqual(result["htf_trend_swing_age_hours"], 8.0)

    def test_htf_trend_swing_age_is_none_without_a_last_event(self):
        # Default HTF_BULLISH carries no last_event - every existing test
        # in this suite relies on this staying a true no-op.
        result = self._run()
        self.assertIsNone(result["htf_trend_swing_age_hours"])

    def test_htf_trend_swing_too_old_rejects_when_gate_enabled(self):
        htf_candles = [{"open_time": i * 14_400_000} for i in range(20)]
        htf_structure = dict(
            HTF_BULLISH, last_event={"index": 0, "type": "BOS", "direction": "BULLISH", "price": 100},
        )

        with patch.object(config, "HTF_TREND_SWING_AGE_REJECT_ENABLED", True), \
             patch.object(config, "HTF_TREND_MAX_SWING_AGE_HOURS", 72.0):
            result = self._run(htf_structure=htf_structure, htf_candles=htf_candles)

        # (19 - 0) candles * 4h/candle = 76h > 72h threshold
        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "HTF_TREND_SWING_TOO_OLD")

    def test_htf_trend_swing_age_under_threshold_does_not_reject(self):
        htf_candles = [{"open_time": i * 14_400_000} for i in range(20)]
        htf_structure = dict(
            HTF_BULLISH, last_event={"index": 17, "type": "BOS", "direction": "BULLISH", "price": 100},
        )

        with patch.object(config, "HTF_TREND_SWING_AGE_REJECT_ENABLED", True), \
             patch.object(config, "HTF_TREND_MAX_SWING_AGE_HOURS", 72.0):
            result = self._run(htf_structure=htf_structure, htf_candles=htf_candles)

        self.assertEqual(result["signal"], "BUY")

    def test_htf_trend_swing_too_old_reject_disabled_by_default_flag(self):
        htf_candles = [{"open_time": i * 14_400_000} for i in range(20)]
        htf_structure = dict(
            HTF_BULLISH, last_event={"index": 0, "type": "BOS", "direction": "BULLISH", "price": 100},
        )

        with patch.object(config, "HTF_TREND_SWING_AGE_REJECT_ENABLED", False):
            result = self._run(htf_structure=htf_structure, htf_candles=htf_candles)

        self.assertEqual(result["signal"], "BUY")


    # config.HTF_TREND_EMA_PRIMARY_ENABLED - EXPLICIT LIVE TEST, zero
    # resolved-trade evidence (see config.py's own comment). Replaces
    # AGAINST_HTF_BIAS's source entirely rather than adding a new reject,
    # so these tests cover both the computed field and the source switch.

    def test_htf_trend_live_is_bullish_when_price_above_the_ema(self):
        # htf_trend_ema kept below the default ltf_close=93 so
        # HTF_TREND_STALE's own check (latest_price < htf_trend_ema for a
        # BUY) never fires here - this test is only about the computed
        # field, not gate interactions (those are covered separately).
        htf_candles = [{"open_time": 0, "close": 100}]

        result = self._run(htf_candles=htf_candles, htf_trend_ema=85.0)

        self.assertEqual(result["htf_trend_live"], "BULLISH")

    def test_htf_trend_live_is_bearish_when_price_below_the_ema(self):
        htf_candles = [{"open_time": 0, "close": 80}]

        result = self._run(htf_candles=htf_candles, htf_trend_ema=85.0)

        self.assertEqual(result["htf_trend_live"], "BEARISH")

    def test_htf_trend_live_is_none_without_ema_data(self):
        # Default htf_trend_ema=None (freshness disabled/insufficient
        # HTF history) - same no-op convention as every optional field.
        result = self._run()
        self.assertIsNone(result["htf_trend_live"])

    def test_against_htf_bias_uses_ema_trend_instead_of_swing_trend_when_enabled(self):
        # Swing says BEARISH (would reject the default BUY candidate
        # under the old mechanism) - EMA says BULLISH (price above it).
        htf_structure = dict(HTF_BULLISH, trend="BEARISH")
        htf_candles = [{"open_time": 0, "close": 100}]

        result = self._run(
            htf_structure=htf_structure, htf_candles=htf_candles, htf_trend_ema=95.0,
            htf_trend_ema_primary_enabled=True,
        )

        self.assertEqual(result["signal"], "BUY")

    def test_against_htf_bias_rejects_on_ema_trend_even_when_swing_trend_agrees(self):
        # Swing says BULLISH (would pass the default BUY candidate under
        # the old mechanism) - EMA says BEARISH (price below it).
        htf_candles = [{"open_time": 0, "close": 100}]

        result = self._run(
            htf_candles=htf_candles, htf_trend_ema=105.0, htf_trend_ema_primary_enabled=True,
        )

        self.assertIsNone(result["signal"])
        self.assertIn("AGAINST_HTF_BIAS", result["reason"])
        self.assertIn("htf=BEARISH", result["reason"])

    def test_against_htf_bias_still_uses_swing_trend_when_flag_disabled(self):
        # Same disagreement setup as the enabling test above, but the
        # flag stays off (default) - old swing-based behavior persists.
        htf_structure = dict(HTF_BULLISH, trend="BEARISH")
        htf_candles = [{"open_time": 0, "close": 100}]

        result = self._run(htf_structure=htf_structure, htf_candles=htf_candles, htf_trend_ema=95.0)

        self.assertIn("AGAINST_HTF_BIAS", result["reason"])

    def test_htf_trend_stale_is_a_noop_once_ema_primary_enabled(self):
        # Same setup as the existing test_buy_rejected_when_price_has_
        # fallen_below_the_htf_trend_ema (ltf_close=93 < htf_trend_ema=95,
        # which fires HTF_TREND_STALE under the old mechanism) - but the
        # HTF candle's own close (100) agrees with the default BUY, so
        # once the new mechanism is primary this must succeed cleanly,
        # not just avoid one specific rejection.
        htf_candles = [{"open_time": 0, "close": 100}]

        result = self._run(
            htf_candles=htf_candles, htf_trend_ema=95.0, htf_trend_ema_primary_enabled=True,
        )

        self.assertEqual(result["signal"], "BUY")

    def test_htf_trend_swing_too_old_is_a_noop_once_ema_primary_enabled(self):
        htf_candles = [{"open_time": i * 14_400_000, "close": 100} for i in range(20)]
        htf_structure = dict(
            HTF_BULLISH, last_event={"index": 0, "type": "BOS", "direction": "BULLISH", "price": 100},
        )

        with patch.object(config, "HTF_TREND_SWING_AGE_REJECT_ENABLED", True), \
             patch.object(config, "HTF_TREND_MAX_SWING_AGE_HOURS", 72.0):
            result = self._run(
                htf_structure=htf_structure, htf_candles=htf_candles, htf_trend_ema=95.0,
                htf_trend_ema_primary_enabled=True,
            )

        # Swing age here is still 76h > 72h (same as the dedicated
        # swing-age test), which would normally reject - must not here.
        self.assertEqual(result["signal"], "BUY")

    # config.HTF_TREND_LIVE_STRENGTH_REJECT_ENABLED - real evidence
    # (2026-09-01, see config.py's own comment). htf_trend_live above is
    # a pure binary above/below its own EMA - these tests cover the
    # strength refinement layered on top of it, only meaningful under
    # HTF_TREND_EMA_PRIMARY_ENABLED (htf_trend_live isn't the operative
    # AGAINST_HTF_BIAS source otherwise).

    def test_htf_trend_live_weak_distance_rejects_even_with_strong_slope(self):
        # Thresholds pinned explicitly (0.5%/0.3%) rather than relying on
        # config.py's own defaults (0.0%/0.0% as of 2026-09-01 - see that
        # flag's own comment for why the stricter tier didn't hold up
        # under corrected evidence) - this test is about the mechanism
        # correctly rejecting once BELOW a threshold, independent of
        # whatever value the mechanism ships with. distance = (100.3-
        # 100.0)/100.0*100 = 0.3% < 0.5%. slope = (100.0-95.0)/95.0*100
        # ~= 5.26%, comfortably clearing its own threshold - proves
        # distance is checked (and can reject) independently of slope.
        htf_candles = [{"open_time": 0, "close": 100.3}]

        with patch.object(config, "HTF_TREND_LIVE_MIN_DISTANCE_PCT", 0.5), \
             patch.object(config, "HTF_TREND_LIVE_MIN_SLOPE_PCT", 0.3):
            result = self._run(
                htf_candles=htf_candles, htf_trend_ema=100.0, htf_trend_ema_prior=95.0,
                htf_trend_ema_primary_enabled=True,
            )

        self.assertEqual(result["reason"], "HTF_TREND_LIVE_WEAK_DISTANCE")

    def test_htf_trend_live_weak_slope_rejects_even_with_strong_distance(self):
        # Thresholds pinned explicitly - see comment on the sibling test
        # above. distance = (105.0-100.0)/100.0*100 = 5%, comfortably
        # clearing its threshold. slope = (100.0-99.9)/99.9*100 ~= 0.1% <
        # 0.3% - proves slope is checked independently of distance.
        htf_candles = [{"open_time": 0, "close": 105.0}]

        with patch.object(config, "HTF_TREND_LIVE_MIN_DISTANCE_PCT", 0.5), \
             patch.object(config, "HTF_TREND_LIVE_MIN_SLOPE_PCT", 0.3):
            result = self._run(
                htf_candles=htf_candles, htf_trend_ema=100.0, htf_trend_ema_prior=99.9,
                htf_trend_ema_primary_enabled=True,
            )

        self.assertEqual(result["reason"], "HTF_TREND_LIVE_WEAK_SLOPE")

    def test_htf_trend_live_strong_distance_and_slope_passes(self):
        # Thresholds pinned explicitly - see comment on the first test in
        # this block.
        htf_candles = [{"open_time": 0, "close": 105.0}]

        with patch.object(config, "HTF_TREND_LIVE_MIN_DISTANCE_PCT", 0.5), \
             patch.object(config, "HTF_TREND_LIVE_MIN_SLOPE_PCT", 0.3):
            result = self._run(
                htf_candles=htf_candles, htf_trend_ema=100.0, htf_trend_ema_prior=95.0,
                htf_trend_ema_primary_enabled=True,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertAlmostEqual(result["htf_trend_live_distance_pct"], 5.0, places=4)
        self.assertAlmostEqual(result["htf_trend_live_slope_pct"], 5.263157, places=4)

    def test_htf_trend_live_strength_reject_disabled_lets_weak_reads_through(self):
        # Thresholds pinned explicitly - see comment on the first test in
        # this block.
        htf_candles = [{"open_time": 0, "close": 100.3}]

        with patch.object(config, "HTF_TREND_LIVE_STRENGTH_REJECT_ENABLED", False), \
             patch.object(config, "HTF_TREND_LIVE_MIN_DISTANCE_PCT", 0.5), \
             patch.object(config, "HTF_TREND_LIVE_MIN_SLOPE_PCT", 0.3):
            result = self._run(
                htf_candles=htf_candles, htf_trend_ema=100.0, htf_trend_ema_prior=95.0,
                htf_trend_ema_primary_enabled=True,
            )

        self.assertEqual(result["signal"], "BUY")

    def test_htf_trend_live_strength_reject_inert_when_ema_primary_disabled(self):
        # Same weak-distance shape as the dedicated rejection test above
        # (0.3% above the EMA), but htf_trend_ema_primary_enabled left at
        # its default False - swing-confirmed structure (default HTF_
        # BULLISH, agrees with the default BUY candidate) is the
        # operative bias instead, and this gate never even evaluates.
        # htf_trend_ema kept below the default ltf_close=93 - same trick
        # the first test in this block uses - so the unrelated HTF_TREND_
        # STALE gate (only live when EMA_PRIMARY is off) doesn't fire
        # instead and mask what this test is actually proving.
        htf_candles = [{"open_time": 0, "close": 80.24}]

        result = self._run(htf_candles=htf_candles, htf_trend_ema=80.0, htf_trend_ema_prior=76.0)

        self.assertEqual(result["signal"], "BUY")

    def test_htf_trend_live_strength_reject_fails_open_on_missing_ema_data(self):
        # Default htf_trend_ema=None (HTF_TREND_FRESHNESS_ENABLED off and
        # no explicit value here) - both distance and slope stay None,
        # same fail-open convention as every gate in this file.
        result = self._run(htf_trend_ema_primary_enabled=True)

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["htf_trend_live_distance_pct"])
        self.assertIsNone(result["htf_trend_live_slope_pct"])

    def test_htf_trend_live_strength_reject_exempt_for_reversal_triggers(self):
        # Same reversal-trigger exemption AGAINST_HTF_BIAS itself gets
        # (AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED defaults
        # True) - this new check is nested inside the same "AGAINST_HTF_
        # BIAS" applicable_gates block, not a separate profile entry, so
        # CVD_DIVERGENCE never reaches it even with weak distance/slope.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}
        htf_candles = [{"open_time": 0, "close": 100.3}]

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                htf_candles=htf_candles, htf_trend_ema=100.0, htf_trend_ema_prior=100.2,
                htf_trend_ema_primary_enabled=True,
            )

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
        # ema_aligned is driven by ema_alignment_value (EMA_ALIGNMENT_
        # PERIOD), NOT ema_value (EMA_CONFIRMATION_PERIOD) - these are
        # deliberately independent, see config.EMA_ALIGNMENT_PERIOD.
        result = self._run(ema_alignment_value=95.0)  # ltf_close defaults to 93 < 95

        self.assertEqual(result["signal"], "BUY")
        self.assertFalse(result["ema_aligned"])
        self.assertEqual(result["ema_alignment_value"], 95.0)

    def test_ema_aligned_true_for_sell_when_price_is_below_ema(self):
        result = self._run(
            ltf_close=108.0,
            cvd={"available": True, "cvd_score": -0.5},
            depth={"available": True, "depth_imbalance": -0.2},
            htf_structure=HTF_BEARISH,
            ltf_analysis=LTF_BEARISH_BREAK,
            sweep_direction="BEARISH",
            ema_alignment_value=115.0,  # 108 < 115 -> aligned for a SELL
        )

        self.assertEqual(result["signal"], "SELL")
        self.assertTrue(result["ema_aligned"])

    def test_ema_unavailable_does_not_block_the_signal(self):
        result = self._run(ema_value=None, ema_alignment_value=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["ema_aligned"])
        self.assertIsNone(result["ema_value"])
        self.assertIsNone(result["ema_alignment_value"])

    def test_ema_aligned_tracks_ema_alignment_value_not_ema_value(self):
        # Deliberately set the two EMAs to DISAGREE: ema_value=90 says
        # aligned (93 > 90), ema_alignment_value=99 says misaligned
        # (93 < 99). ema_aligned must follow ema_alignment_value only -
        # this is the actual fix for the EMA/BTC alignment timeframe
        # mismatch (config.EMA_ALIGNMENT_PERIOD).
        result = self._run(ema_value=90.0, ema_alignment_value=99.0)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["ema_value"], 90.0)
        self.assertEqual(result["ema_alignment_value"], 99.0)
        self.assertFalse(result["ema_aligned"])

    def test_ema_fields_are_none_when_confirmation_disabled(self):
        # Both ema_value and ema_alignment_value are gated by the same
        # EMA_CONFIRMATION_ENABLED toggle - see signal_engine.py.
        with patch.object(config, "EMA_CONFIRMATION_ENABLED", False):
            result = self._run(ema_value=95.0, ema_alignment_value=95.0)  # would be misaligned if computed

        self.assertEqual(result["signal"], "BUY")
        self.assertIsNone(result["ema_value"])
        self.assertIsNone(result["ema_alignment_value"])
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

    def test_crash_mode_blocks_a_buy_during_a_bearish_crash(self):
        # config.CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED - real motivation
        # (2026-08-22): a BUY DCA'd right into the bottom of a real BTC
        # flash-crash while every SELL-side position open at the same
        # moment profited from the identical move.
        with patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", True):
            result = self._run(
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "CRASH_MODE")

    def test_crash_mode_blocks_a_sell_during_a_bullish_crash(self):
        with patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", True):
            result = self._run(
                ltf_close=108.0,
                cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH,
                ltf_analysis=LTF_BEARISH_BREAK,
                sweep_direction="BEARISH",
                ema_value=115.0,
                crash_snapshot={"available": True, "active": True, "direction": "BULLISH"},
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "CRASH_MODE")

    def test_crash_mode_does_not_block_the_aligned_side(self):
        # A BUY during a BULLISH crash (a violent rally) is the side that
        # benefits from the move, not the one this gate exists to protect.
        with patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", True):
            result = self._run(
                crash_snapshot={"available": True, "active": True, "direction": "BULLISH"},
            )

        self.assertEqual(result["signal"], "BUY")

    def test_crash_mode_does_not_block_when_inactive(self):
        with patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", True):
            result = self._run(
                crash_snapshot={"available": True, "active": False, "direction": None},
            )

        self.assertEqual(result["signal"], "BUY")

    def test_crash_mode_does_not_block_when_the_block_flag_is_off(self):
        with patch.object(config, "CRASH_DETECTOR_ENABLED", True), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", False):
            result = self._run(
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertEqual(result["signal"], "BUY")

    def test_crash_mode_does_not_block_when_the_master_flag_is_off(self):
        with patch.object(config, "CRASH_DETECTOR_ENABLED", False), \
             patch.object(config, "CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", True):
            result = self._run(
                crash_snapshot={"available": True, "active": True, "direction": "BEARISH"},
            )

        self.assertEqual(result["signal"], "BUY")

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

    # config.ABSORPTION_TRACKING_ENABLED - informational only, same
    # treatment as btc_aligned above. absorption.compute() itself is
    # covered directly in test_absorption.py; these only prove
    # signal_engine wires real cvd_snapshot/depth_snapshot data through
    # to absorption_signal, and that alignment is compared against the
    # winning candidate's own side.

    def test_absorption_signal_and_aligned_computed_from_real_data(self):
        # Aggressive one-sided SELLING (ratio negative) that didn't move
        # price - buyers absorbed it -> "BUY" signal. BUY-side candidate
        # (default _run()) agrees -> aligned True.
        cvd = {"available": True, "cvd_score": 0.5, "ratio_1m": -0.8, "notional_1m": 10000}
        depth = {"available": True, "depth_imbalance": 0.2, "price_change_pct_1m": 0.01}

        with patch.object(config, "ABSORPTION_TRACKING_ENABLED", True), \
             patch.object(config, "ABSORPTION_MIN_CVD_RATIO", 0.5), \
             patch.object(config, "ABSORPTION_MAX_PRICE_MOVE_PCT", 0.05):
            result = self._run(cvd=cvd, depth=depth)

        self.assertEqual(result["absorption_signal"], "BUY")
        self.assertTrue(result["absorption_aligned"])

    def test_absorption_aligned_false_when_signal_disagrees_with_the_candidate_side(self):
        # Same absorption reading (bullish) but the winning candidate is
        # SELL - aligned must be False, not None (real data exists, it
        # just disagrees).
        cvd = {"available": True, "cvd_score": -0.5, "ratio_1m": -0.8, "notional_1m": 10000}
        depth = {"available": True, "depth_imbalance": -0.2, "price_change_pct_1m": 0.01}

        with patch.object(config, "ABSORPTION_TRACKING_ENABLED", True), \
             patch.object(config, "ABSORPTION_MIN_CVD_RATIO", 0.5), \
             patch.object(config, "ABSORPTION_MAX_PRICE_MOVE_PCT", 0.05):
            result = self._run(
                ltf_close=108.0, cvd=cvd, depth=depth, htf_structure=HTF_BEARISH,
                ltf_analysis=LTF_BEARISH_BREAK, sweep_direction="BEARISH", ema_value=115.0,
            )

        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["absorption_signal"], "BUY")
        self.assertFalse(result["absorption_aligned"])

    def test_absorption_signal_none_when_price_moved_too_much(self):
        # Real one-sided flow that DID move price - not absorption.
        cvd = {"available": True, "cvd_score": 0.5, "ratio_1m": -0.8, "notional_1m": 10000}
        depth = {"available": True, "depth_imbalance": 0.2, "price_change_pct_1m": 0.5}

        with patch.object(config, "ABSORPTION_TRACKING_ENABLED", True), \
             patch.object(config, "ABSORPTION_MIN_CVD_RATIO", 0.5), \
             patch.object(config, "ABSORPTION_MAX_PRICE_MOVE_PCT", 0.05):
            result = self._run(cvd=cvd, depth=depth)

        self.assertIsNone(result["absorption_signal"])
        self.assertIsNone(result["absorption_aligned"])

    def test_absorption_signal_none_when_price_change_data_unavailable(self):
        # depth_snapshot carries no price_change_pct_1m key at all - the
        # engine just started tracking, not enough history yet.
        cvd = {"available": True, "cvd_score": 0.5, "ratio_1m": -0.8, "notional_1m": 10000}
        depth = {"available": True, "depth_imbalance": 0.2}

        with patch.object(config, "ABSORPTION_TRACKING_ENABLED", True):
            result = self._run(cvd=cvd, depth=depth)

        self.assertIsNone(result["absorption_signal"])
        self.assertIsNone(result["absorption_aligned"])

    def test_absorption_disabled_by_config(self):
        cvd = {"available": True, "cvd_score": 0.5, "ratio_1m": -0.8, "notional_1m": 10000}
        depth = {"available": True, "depth_imbalance": 0.2, "price_change_pct_1m": 0.01}

        with patch.object(config, "ABSORPTION_TRACKING_ENABLED", False):
            result = self._run(cvd=cvd, depth=depth)

        self.assertIsNone(result["absorption_signal"])
        self.assertIsNone(result["absorption_aligned"])

    # config.DEPTH_TREND_TRACKING_ENABLED - informational only by default.
    # orderbook.DepthImbalanceEngine._depth_consistency_pct itself is
    # covered directly in test_orderbook.py; these only prove signal_engine
    # wires real depth_snapshot data through and compares it against the
    # winning candidate's own side.

    def test_depth_trend_aligned_true_when_book_favorable_and_consistent(self):
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.9}

        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6):
            result = self._run(depth=depth)

        self.assertEqual(result["depth_consistency_pct"], 0.9)
        self.assertTrue(result["depth_trend_aligned"])

    def test_depth_trend_aligned_false_when_consistency_below_threshold(self):
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.3}

        # REJECT_ENABLED pinned False - this test is about the
        # informational field reading False, not about the (separately
        # tested) live gate rejecting the signal outright.
        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED", False):
            result = self._run(depth=depth)

        self.assertFalse(result["depth_trend_aligned"])

    def test_depth_trend_aligned_none_when_consistency_data_unavailable(self):
        depth = {"available": True, "depth_imbalance": 0.2}  # no depth_consistency_pct key

        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True):
            result = self._run(depth=depth)

        self.assertIsNone(result["depth_consistency_pct"])
        self.assertIsNone(result["depth_trend_aligned"])

    def test_depth_trend_disabled_by_config(self):
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.9}

        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", False):
            result = self._run(depth=depth)

        self.assertIsNone(result["depth_consistency_pct"])
        self.assertIsNone(result["depth_trend_aligned"])

    # config.DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED - brand new,
    # unvalidated mechanism, default OFF. Catches the gap DEPTH_OPPOSING
    # can't: an instantaneous reading that passes but was actually just a
    # last-second flip, not a genuinely held book.

    def test_depth_trend_unstable_gate_is_a_noop_by_default(self):
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.1}

        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED", False):
            result = self._run(depth=depth)

        self.assertEqual(result["signal"], "BUY")

    def test_depth_trend_unstable_rejects_when_enabled_and_below_threshold(self):
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.1}

        with patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED", True):
            result = self._run(depth=depth)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "DEPTH_TREND_UNSTABLE")

    def test_depth_trend_unstable_exempt_for_cvd_divergence(self):
        # CVD_DIVERGENCE's whole thesis is that book pressure is CHANGING
        # right now - requiring it to have already been stable before the
        # change would punish the exact freshness that makes it a genuine
        # reversal. See _TREND_AGREEMENT_EXEMPT_TRIGGERS.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.1}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                depth=depth,
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")

    def test_depth_trend_unstable_still_applies_to_cvd_divergence_when_exemption_disabled(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}
        depth = {"available": True, "depth_imbalance": 0.2, "depth_consistency_pct": 0.1}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_TRACKING_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_PCT", 0.6), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_REJECT_ENABLED", True), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                depth=depth,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "DEPTH_TREND_UNSTABLE")

    # config.WHALE_TRADE_TRACKING_ENABLED - informational only by default.
    # The whale-window scan itself is covered directly in
    # test_order_flow.py; these only prove signal_engine wires real
    # cvd_snapshot data through and compares it against the winning
    # candidate's own side.

    def test_whale_aligned_true_when_notional_clears_floor_and_matches_side(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "BUY"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000):
            result = self._run(cvd=cvd)

        self.assertEqual(result["whale_notional"], 25000)
        self.assertEqual(result["whale_direction"], "BUY")
        self.assertTrue(result["whale_aligned"])

    def test_whale_aligned_false_when_direction_disagrees(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "SELL"}

        # REJECT_ENABLED pinned False - this test is about the
        # informational field reading False, not about the (separately
        # tested) live gate rejecting the signal outright.
        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000), \
             patch.object(config, "WHALE_AGAINST_REJECT_ENABLED", False):
            result = self._run(cvd=cvd)

        self.assertFalse(result["whale_aligned"])

    def test_whale_aligned_none_when_below_notional_floor(self):
        # A small recent trade isn't evidence either way - None, not False.
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 5000, "whale_direction": "BUY"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000):
            result = self._run(cvd=cvd)

        self.assertIsNone(result["whale_aligned"])

    def test_whale_disabled_by_config(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "BUY"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", False):
            result = self._run(cvd=cvd)

        self.assertIsNone(result["whale_notional"])
        self.assertIsNone(result["whale_direction"])
        self.assertIsNone(result["whale_aligned"])

    # config.WHALE_AGAINST_REJECT_ENABLED - brand new, unvalidated
    # mechanism, default OFF. Universal across triggers (same as
    # DEPTH_OPPOSING/CRASH_MODE, no trigger_gate_profiles() entry).

    def test_whale_against_gate_is_a_noop_by_default(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "SELL"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000), \
             patch.object(config, "WHALE_AGAINST_REJECT_ENABLED", False):
            result = self._run(cvd=cvd)

        self.assertEqual(result["signal"], "BUY")

    def test_whale_against_rejects_when_enabled_and_opposing(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "SELL"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000), \
             patch.object(config, "WHALE_AGAINST_REJECT_ENABLED", True):
            result = self._run(cvd=cvd)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "WHALE_AGAINST")

    def test_whale_against_does_not_reject_when_direction_agrees(self):
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "BUY"}

        with patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000), \
             patch.object(config, "WHALE_AGAINST_REJECT_ENABLED", True):
            result = self._run(cvd=cvd)

        self.assertEqual(result["signal"], "BUY")

    def test_whale_against_still_applies_to_cvd_divergence(self):
        # Unlike DEPTH_TREND_MIN_CONSISTENCY, WHALE_AGAINST does NOT get the
        # reversal-trigger exemption - a real institutional print dumping
        # against a reversal trigger's own NEW direction is still bad news
        # for that reversal.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}
        cvd = {"available": True, "cvd_score": 0.5, "whale_notional": 25000, "whale_direction": "SELL"}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_TRACKING_ENABLED", True), \
             patch.object(config, "WHALE_TRADE_MIN_NOTIONAL_USDT", 20000), \
             patch.object(config, "WHALE_AGAINST_REJECT_ENABLED", True):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                divergence_direction="BULLISH", divergence_level=88,
                cvd=cvd,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "WHALE_AGAINST")

    # config.CROSS_EXCHANGE_OI_TRACKING_ENABLED - informational only.
    # cross_exchange_oi.compute_agreement() itself is covered directly in
    # test_cross_exchange_oi.py; these only prove signal_engine wires the
    # real oi_snapshot/oi_snapshot_bybit/oi_snapshot_okx data through.

    def test_cross_exchange_oi_agree_true_when_both_venues_confirm(self):
        # Default oi_snapshot (OI_RISING) has oi_change_pct=5.0 (rising).
        bybit = {"available": True, "oi_change_pct": 2.0}
        okx = {"available": True, "oi_change_pct": 1.0}

        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True):
            result = self._run(oi_snapshot_bybit=bybit, oi_snapshot_okx=okx)

        self.assertEqual(result["oi_change_pct_bybit"], 2.0)
        self.assertEqual(result["oi_change_pct_okx"], 1.0)
        self.assertTrue(result["cross_exchange_oi_agree"])

    def test_cross_exchange_oi_agree_false_when_a_venue_disagrees(self):
        bybit = {"available": True, "oi_change_pct": -2.0}
        okx = {"available": True, "oi_change_pct": 1.0}

        # CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED patched off explicitly -
        # this test wants a real disagreement reading WITHOUT the reject
        # gate (added later, see the reject-path tests below) turning
        # this into a rejected signal instead.
        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", False):
            result = self._run(oi_snapshot_bybit=bybit, oi_snapshot_okx=okx)

        self.assertFalse(result["cross_exchange_oi_agree"])

    def test_cross_exchange_oi_unavailable_snapshot_reads_as_none(self):
        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True):
            result = self._run(
                oi_snapshot_bybit={"available": False}, oi_snapshot_okx={"available": False},
            )

        self.assertIsNone(result["oi_change_pct_bybit"])
        self.assertIsNone(result["oi_change_pct_okx"])
        self.assertIsNone(result["cross_exchange_oi_agree"])

    def test_cross_exchange_oi_disabled_by_config(self):
        bybit = {"available": True, "oi_change_pct": 2.0}
        okx = {"available": True, "oi_change_pct": 1.0}

        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", False):
            result = self._run(oi_snapshot_bybit=bybit, oi_snapshot_okx=okx)

        self.assertIsNone(result["oi_change_pct_bybit"])
        self.assertIsNone(result["oi_change_pct_okx"])
        self.assertIsNone(result["cross_exchange_oi_agree"])

    # config.CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED - informational
    # only, same shape as CROSS_EXCHANGE_OI_TRACKING_ENABLED above (reuses
    # cross_exchange_oi.compute_agreement directly - see signal_engine.py's
    # own comment for why that's not OI-specific despite the module name).
    # These only prove signal_engine wires liquidation_snapshot_bybit/_okx
    # through - real parsing is covered in test_cross_exchange_liquidation.py.

    def test_cross_exchange_liquidation_agree_true_when_both_venues_confirm(self):
        # Default signal direction is BULLISH - a positive net Binance
        # liquidation notional means longs are being liquidated in a
        # BULLISH break too, per liquidation_aligned's own definition, but
        # cross_exchange_liquidation_agree itself only compares sign
        # against Binance's own net_liquidation_notional, not direction.
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": 500.0}
        okx = {"available": True, "net_liquidation_notional": 200.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True):
            result = self._run(
                liquidation_snapshot=binance,
                liquidation_snapshot_bybit=bybit, liquidation_snapshot_okx=okx,
            )

        self.assertEqual(result["liquidation_notional_net_bybit"], 500.0)
        self.assertEqual(result["liquidation_notional_net_okx"], 200.0)
        self.assertTrue(result["cross_exchange_liquidation_agree"])

    def test_cross_exchange_liquidation_agree_false_when_a_venue_disagrees(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": -500.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED", False):
            result = self._run(liquidation_snapshot=binance, liquidation_snapshot_bybit=bybit)

        self.assertFalse(result["cross_exchange_liquidation_agree"])

    def test_cross_exchange_liquidation_unavailable_snapshot_reads_as_none(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True):
            result = self._run(
                liquidation_snapshot=binance,
                liquidation_snapshot_bybit={"available": False},
                liquidation_snapshot_okx={"available": False},
            )

        self.assertIsNone(result["liquidation_notional_net_bybit"])
        self.assertIsNone(result["liquidation_notional_net_okx"])
        self.assertIsNone(result["cross_exchange_liquidation_agree"])

    def test_cross_exchange_liquidation_disabled_by_config(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": 500.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", False):
            result = self._run(liquidation_snapshot=binance, liquidation_snapshot_bybit=bybit)

        self.assertIsNone(result["liquidation_notional_net_bybit"])
        self.assertIsNone(result["cross_exchange_liquidation_agree"])

    # config.VOLUME_PROFILE_TRACKING_ENABLED - descriptive only, no
    # "aligned" field (see config.py's own reasoning). volume_profile.py's
    # own bucketing/POC/value-area math is covered directly in
    # test_volume_profile.py; these only prove signal_engine wires a real
    # volume_profile_snapshot through.

    def test_volume_profile_fields_carried_through_when_available(self):
        snapshot = {
            "available": True, "poc_price": 100.0, "value_area_high": 105.0,
            "value_area_low": 95.0, "position": "INSIDE_VALUE_AREA",
        }

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["vp_poc_price"], 100.0)
        self.assertEqual(result["vp_value_area_high"], 105.0)
        self.assertEqual(result["vp_value_area_low"], 95.0)
        self.assertEqual(result["vp_position"], "INSIDE_VALUE_AREA")

    def test_volume_profile_unavailable_snapshot_reads_as_none(self):
        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True):
            result = self._run(volume_profile_snapshot={"available": False})

        self.assertIsNone(result["vp_poc_price"])
        self.assertIsNone(result["vp_position"])

    def test_volume_profile_disabled_by_config(self):
        snapshot = {
            "available": True, "poc_price": 100.0, "value_area_high": 105.0,
            "value_area_low": 95.0, "position": "INSIDE_VALUE_AREA",
        }

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", False):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertIsNone(result["vp_poc_price"])
        self.assertIsNone(result["vp_position"])


    # config.CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED - EXPLICIT LIVE TEST,
    # no resolved-trade evidence (see config.py's own comment). Real
    # cross_exchange_oi.compute_agreement() logic is covered directly in
    # test_cross_exchange_oi.py; these only prove the reject condition
    # itself.

    def test_cross_exchange_oi_disagree_rejects_when_gate_enabled(self):
        # Default oi_snapshot (OI_RISING) has oi_change_pct=5.0 (rising) -
        # Bybit disagrees (falling).
        bybit = {"available": True, "oi_change_pct": -2.0}

        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", True):
            result = self._run(oi_snapshot_bybit=bybit)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "CROSS_EXCHANGE_OI_DISAGREE")

    def test_cross_exchange_oi_agreement_does_not_reject(self):
        bybit = {"available": True, "oi_change_pct": 2.0}

        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", True):
            result = self._run(oi_snapshot_bybit=bybit)

        self.assertEqual(result["signal"], "BUY")

    def test_cross_exchange_oi_unavailable_does_not_reject(self):
        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", False), \
             patch.object(config, "CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", True):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")

    def test_cross_exchange_oi_reject_disabled_by_default_flag(self):
        bybit = {"available": True, "oi_change_pct": -2.0}

        with patch.object(config, "CROSS_EXCHANGE_OI_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", False):
            result = self._run(oi_snapshot_bybit=bybit)

        self.assertEqual(result["signal"], "BUY")

    # config.CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED - same zero-
    # evidence-yet, built-but-off convention as CROSS_EXCHANGE_OI_AGREE_
    # REJECT_ENABLED above.

    def test_cross_exchange_liquidation_disagree_rejects_when_gate_enabled(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": -500.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED", True):
            result = self._run(liquidation_snapshot=binance, liquidation_snapshot_bybit=bybit)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "CROSS_EXCHANGE_LIQUIDATION_DISAGREE")

    def test_cross_exchange_liquidation_agreement_does_not_reject(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": 500.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED", True):
            result = self._run(liquidation_snapshot=binance, liquidation_snapshot_bybit=bybit)

        self.assertEqual(result["signal"], "BUY")

    def test_cross_exchange_liquidation_unavailable_does_not_reject(self):
        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", False), \
             patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED", True):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")

    def test_cross_exchange_liquidation_reject_disabled_by_default_flag(self):
        binance = {"available": True, "net_liquidation_notional": 1000.0}
        bybit = {"available": True, "net_liquidation_notional": -500.0}

        with patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_TRACKING_ENABLED", True), \
             patch.object(config, "CROSS_EXCHANGE_LIQUIDATION_AGREE_REJECT_ENABLED", False):
            result = self._run(liquidation_snapshot=binance, liquidation_snapshot_bybit=bybit)

        self.assertEqual(result["signal"], "BUY")

    # config.VP_EXTENSION_REJECT_ENABLED - EXPLICIT LIVE TEST, no
    # resolved-trade evidence (see config.py's own comment).

    def test_vp_extension_rejects_buy_already_above_value_area(self):
        snapshot = {"available": True, "position": "ABOVE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "VP_ALREADY_EXTENDED")

    def test_vp_extension_rejects_sell_already_below_value_area(self):
        snapshot = {"available": True, "position": "BELOW_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", True):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.2},
                htf_structure=HTF_BEARISH, ltf_analysis=LTF_BEARISH_BREAK,
                sweep_direction="BEARISH", ema_value=115.0,
                volume_profile_snapshot=snapshot,
            )

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "VP_ALREADY_EXTENDED")

    def test_vp_extension_does_not_reject_the_opposite_sides_extension(self):
        # BUY candidate, price already BELOW the value area (not extended
        # in ITS own direction) - must not reject.
        # VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED disabled here to isolate
        # this test to VP_EXTENSION_REJECT_ENABLED alone - it defaults
        # True and would otherwise also reject this same fixture (a
        # favorable, not unfavorable, extension) for its own separate
        # reason - see VpInsideValueAreaRequiredTests below.
        snapshot = {"available": True, "position": "BELOW_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", True), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", False):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["signal"], "BUY")

    def test_vp_extension_does_not_reject_inside_value_area(self):
        snapshot = {"available": True, "position": "INSIDE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["signal"], "BUY")

    def test_vp_extension_does_not_reject_when_unavailable(self):
        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", False), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", True):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")

    def test_vp_extension_reject_disabled_by_default_flag(self):
        # VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED disabled here too - see the
        # comment on test_vp_extension_does_not_reject_the_opposite_sides_
        # extension above for why.
        snapshot = {"available": True, "position": "ABOVE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", False), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", False):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["signal"], "BUY")

    # config.VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED - 2026-08-31, real
    # evidence (see config.py's own comment). Independent of (stricter
    # than) VP_EXTENSION_REJECT_ENABLED above.

    def test_vp_inside_required_rejects_above_value_area_on_a_buy(self):
        snapshot = {"available": True, "position": "ABOVE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", False), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "VP_NOT_INSIDE_VALUE_AREA")

    def test_vp_inside_required_rejects_the_favorable_extension_too(self):
        # BELOW_VALUE_AREA on a BUY is the case VP_EXTENSION_REJECT_ENABLED
        # allows through (not "chasing" in the trade's own direction) -
        # this gate is stricter and requires actually-inside regardless.
        snapshot = {"available": True, "position": "BELOW_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", False), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "VP_NOT_INSIDE_VALUE_AREA")

    def test_vp_inside_required_does_not_reject_inside_value_area(self):
        snapshot = {"available": True, "position": "INSIDE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", True):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["signal"], "BUY")

    def test_vp_inside_required_does_not_reject_when_unavailable(self):
        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", False), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", True):
            result = self._run()

        self.assertEqual(result["signal"], "BUY")

    def test_vp_inside_required_disabled_by_flag(self):
        snapshot = {"available": True, "position": "ABOVE_VALUE_AREA"}

        with patch.object(config, "VOLUME_PROFILE_TRACKING_ENABLED", True), \
             patch.object(config, "VP_EXTENSION_REJECT_ENABLED", False), \
             patch.object(config, "VP_INSIDE_VALUE_AREA_REQUIRED_ENABLED", False):
            result = self._run(volume_profile_snapshot=snapshot)

        self.assertEqual(result["signal"], "BUY")

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

    # config.DCA_BREAKEVEN_CONFIRMATION_ENABLED's sibling investigation
    # (2026-08-19) - how many candles old the underlying setup actually
    # was at entry, distinct from trigger_candle_open_time above (that's
    # the RETEST candle, always fresh; this is the setup being retested).
    # signal_engine.evaluate() already computed this internally for each
    # trigger's own age-gate check but never returned it - see
    # signal_journal.py's setup_age_candles comment for why it's worth
    # journaling now. _ltf_candles() always produces a single candle at
    # index 0 (see test_choch_retest_ignored_when_event_too_old's own
    # comment on this), so setup_age_candles = 0 - index for these tests -
    # a negative index simulates an event/gap/block that formed before
    # the start of this LTF buffer, the same trick that test already uses.

    def test_setup_age_is_zero_for_structure_break(self):
        result = self._run()

        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")
        self.assertEqual(result["setup_age_candles"], 0)

    def test_setup_age_is_zero_for_ema_pullback(self):
        with patch.object(config, "EMA_PULLBACK_TRIGGER_ENABLED", True):
            result = self._run(
                sweep_direction=None,
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                ema_pullback_direction="BULLISH", ema_pullback_level=90,
            )

        self.assertEqual(result["signal_trigger"], "EMA_PULLBACK")
        self.assertEqual(result["setup_age_candles"], 0)

    def test_setup_age_is_none_for_liquidity_sweep(self):
        # No formation index available for a liquidity pool - genuinely
        # unknown, not "fresh".
        with patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", True):
            result = self._run(
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                sweep_direction="BULLISH", sweep_level=89,
            )

        self.assertEqual(result["signal_trigger"], "LIQUIDITY_SWEEP")
        self.assertIsNone(result["setup_age_candles"])

    def test_setup_age_reflects_how_old_the_retested_fvg_actually_is(self):
        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                sweep_direction=None,
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                fvg_retest_direction="BULLISH", fvg_retest_level=90, fvg_retest_index=-4,
            )

        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")
        self.assertEqual(result["setup_age_candles"], 4)  # tested_index(0) - gap_index(-4)

    def test_setup_age_reflects_how_old_the_retested_order_block_is(self):
        with patch.object(config, "ORDER_BLOCK_RETEST_TRIGGER_ENABLED", True):
            result = self._run(
                sweep_direction=None,
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                order_block_retest_direction="BULLISH", order_block_retest_level=90,
                order_block_retest_index=-7,
            )

        self.assertEqual(result["signal_trigger"], "ORDER_BLOCK_RETEST")
        self.assertEqual(result["setup_age_candles"], 7)

    def test_setup_age_reflects_how_old_the_choch_event_is(self):
        # CHOCH_TRIGGER_MIN_AGE_CANDLES pinned off - this test is about the
        # setup_age_candles VALUE at a specific age (6), not about the
        # min-age gate rejecting it (age 6 < the gate's own default of 9).
        analysis = self._choch_analysis("BULLISH", event_price=95, event_index=-6)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_TRIGGER_MIN_AGE_CANDLES", 0):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")
        self.assertEqual(result["setup_age_candles"], 6)

    def test_setup_age_reflects_how_old_the_cvd_divergence_swings_are(self):
        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "SIGNAL_MIN_CVD_SCORE", 0.15):
            result = self._run(
                sweep_direction=None,
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                divergence_direction="BULLISH", divergence_level=90, divergence_index=-2,
            )

        self.assertEqual(result["signal_trigger"], "CVD_DIVERGENCE")
        self.assertEqual(result["setup_age_candles"], 2)

    def test_setup_age_reflects_how_old_the_oi_divergence_swings_are(self):
        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True):
            result = self._run(
                sweep_direction=None,
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                oi_divergence_direction="BULLISH", oi_divergence_level=90, oi_divergence_index=-9,
            )

        self.assertEqual(result["signal_trigger"], "OI_DIVERGENCE")
        self.assertEqual(result["setup_age_candles"], 9)

    def test_setup_age_is_none_for_liquidation_sweep_confirmed(self):
        # LIQUIDITY_SWEEP outranks LIQUIDATION_SWEEP_CONFIRMED in the
        # fixed-priority selection - disabled here so it doesn't win
        # instead (same isolation existing tests already use).
        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDITY_SWEEP_TRIGGER_ENABLED", False):
            result = self._run(
                ltf_analysis=dict(LTF_BULLISH_BREAK, live_break={"broken": False}),
                sweep_direction="BULLISH", sweep_level=89,
                liquidation_confirmed_sweep_direction="BULLISH", liquidation_confirmed_sweep_level=89,
            )

        self.assertEqual(result["signal_trigger"], "LIQUIDATION_SWEEP_CONFIRMED")
        self.assertIsNone(result["setup_age_candles"])

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
                "BTCUSDT", [{"open_time": 0, "close": 100}], _ltf_candles(93.0),
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
                "BTCUSDT", [{"open_time": 0, "close": 100}], _ltf_candles(93.0),
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

    # config.OB_FVG_RETEST_MIN_DEPTH_IMBALANCE - real evidence (2026-08-25
    # signal-engine audit, 33 resolved OB_FVG_RETEST trades): depth_
    # imbalance clearly favorable (signed >=0.10) won 90.0% (n=10) vs only
    # 73.9% (n=23) when merely neutral. Trigger-scoped (only OB_FVG_
    # RETEST's own candidacy), reject-only, fail-open on missing data -
    # mirrors CHOCH_RETEST_MIN_DEPTH_IMBALANCE's own test shape.

    def test_ob_fvg_retest_rejected_when_depth_imbalance_too_weak_buy(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                depth={"available": True, "depth_imbalance": 0.05},  # below 0.10, but not opposing enough for DEPTH_OPPOSING
            )

        self.assertIn("OB_FVG_RETEST_DEPTH_WEAK", result["reason"])

    def test_ob_fvg_retest_rejected_when_depth_imbalance_too_weak_sell(self):
        analysis = dict(LTF_BEARISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.05},  # signed=0.05, below 0.10
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
                fvg_retest_direction="BEARISH", fvg_retest_level=110,
            )

        self.assertIn("OB_FVG_RETEST_DEPTH_WEAK", result["reason"])

    def test_ob_fvg_retest_accepted_when_depth_imbalance_meets_threshold(self):
        # Boundary is inclusive (">=" not ">") - signed depth exactly equal
        # to the threshold must still qualify.
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                depth={"available": True, "depth_imbalance": 0.10},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")

    def test_ob_fvg_retest_depth_requirement_disabled_lets_weak_depth_through(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                depth={"available": True, "depth_imbalance": 0.01},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")

    def test_ob_fvg_retest_depth_unavailable_does_not_block(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "OB_FVG_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                fvg_retest_direction="BULLISH", fvg_retest_level=90,
                depth={"available": False},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "OB_FVG_RETEST")

    def test_ob_fvg_retest_depth_requirement_does_not_affect_other_triggers(self):
        # Weak depth (0.05, below the 0.10 OB_FVG_RETEST-specific
        # threshold) must not block a DIFFERENT trigger (STRUCTURE_BREAK,
        # _run()'s own default) evaluated for the same direction/tick -
        # this check is trigger-scoped, not direction-scoped.
        with patch.object(config, "OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(depth={"available": True, "depth_imbalance": 0.05})

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")

    # config.CHOCH_RETEST_TRIGGER_ENABLED - an already-CONFIRMED reversal
    # (last_event type CHoCH), retested within CHOCH_TRIGGER_MAX_AGE_CANDLES.
    # Ranked last in priority: STRUCTURE_BREAK > OB_FVG_RETEST >
    # LIQUIDITY_SWEEP > CHOCH_RETEST.

    def _choch_analysis(self, direction, event_price, event_index=-9, event_type="CHoCH"):
        # event_index=-9 (age=9) by default - config.CHOCH_TRIGGER_MIN_AGE_
        # CANDLES defaults to 9, so every test below that isn't specifically
        # about age needs a fixture that already clears it, same as it
        # already needed to clear CHOCH_TRIGGER_MAX_AGE_CANDLES(10).
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

    def test_choch_retest_ignored_when_event_too_fresh(self):
        # config.CHOCH_TRIGGER_MIN_AGE_CANDLES - real evidence (2026-08-21):
        # a CHoCH still fresh (age well under the default 9) is
        # disproportionately a fakeout - see config.py's own comment for
        # the full numbers. age=3 here is well under that default.
        analysis = self._choch_analysis("BULLISH", event_price=95, event_index=-3)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_TRIGGER_MIN_AGE_CANDLES", 9):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertIsNone(result["signal"])
        self.assertEqual(result["reason"], "NO_LIVE_STRUCTURE_BREAK")

    def test_choch_retest_accepted_right_at_the_minimum_age(self):
        # Boundary is inclusive (">=" not ">") - age exactly equal to
        # CHOCH_TRIGGER_MIN_AGE_CANDLES must still qualify.
        analysis = self._choch_analysis("BULLISH", event_price=95, event_index=-9)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_TRIGGER_MIN_AGE_CANDLES", 9):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

    def test_choch_retest_min_age_of_zero_preserves_original_behavior(self):
        # 0 disables the new gate entirely (every age qualifies, same as
        # before this feature existed) - a fresh (age=0) event must still
        # be accepted when the gate is explicitly turned off.
        analysis = self._choch_analysis("BULLISH", event_price=95, event_index=0)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_TRIGGER_MIN_AGE_CANDLES", 0):
            result = self._run(ltf_analysis=analysis, sweep_direction=None)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

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

    # config.CHOCH_RETEST_MIN_DEPTH_IMBALANCE - real evidence (2026-08-24):
    # depth_imbalance clearly favorable (signed >=0.10) won 75.0% (n=12)
    # vs only 55.0% (n=20) when merely neutral. Trigger-scoped (only
    # CHOCH_RETEST's own candidacy), reject-only, fail-open on missing
    # data - see config.py's own comment for the full reasoning.

    def test_choch_retest_rejected_when_depth_imbalance_too_weak_buy(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                depth={"available": True, "depth_imbalance": 0.05},  # below 0.10, but not opposing enough for DEPTH_OPPOSING
            )

        self.assertIn("CHOCH_RETEST_DEPTH_WEAK", result["reason"])

    def test_choch_retest_rejected_when_depth_imbalance_too_weak_sell(self):
        analysis = self._choch_analysis("BEARISH", event_price=85)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_close=108.0, cvd={"available": True, "cvd_score": -0.5},
                depth={"available": True, "depth_imbalance": -0.05},  # signed=0.05, below 0.10
                htf_structure=HTF_BEARISH, ltf_analysis=analysis,
                sweep_direction=None, ema_value=115.0,
            )

        self.assertIn("CHOCH_RETEST_DEPTH_WEAK", result["reason"])

    def test_choch_retest_accepted_when_depth_imbalance_meets_threshold(self):
        # Boundary is inclusive (">=" not ">") - signed depth exactly equal
        # to the threshold must still qualify.
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                depth={"available": True, "depth_imbalance": 0.10},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

    def test_choch_retest_depth_requirement_disabled_lets_weak_depth_through(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                depth={"available": True, "depth_imbalance": 0.01},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

    def test_choch_retest_depth_unavailable_does_not_block(self):
        analysis = self._choch_analysis("BULLISH", event_price=95)

        with patch.object(config, "CHOCH_RETEST_TRIGGER_ENABLED", True), \
             patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(
                ltf_analysis=analysis, sweep_direction=None,
                depth={"available": False},
            )

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "CHOCH_RETEST")

    def test_choch_retest_depth_requirement_does_not_affect_other_triggers(self):
        # Weak depth (0.05, below the 0.10 CHOCH_RETEST-specific
        # threshold) must not block a DIFFERENT trigger (STRUCTURE_BREAK,
        # _run()'s own default) evaluated for the same direction/tick -
        # this check is trigger-scoped, not direction-scoped.
        with patch.object(config, "CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10):
            result = self._run(depth={"available": True, "depth_imbalance": 0.05})

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["signal_trigger"], "STRUCTURE_BREAK")

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

    # config.CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED - real swing-to-
    # swing CVD delta at every structural candidate, independent of
    # whether detect_divergence itself confirmed anything (2026-08-26
    # finding: CVD_DIVERGENCE has never produced a trade).
    # diagnostic_candidates() own logic is covered in test_cvd_divergence.
    # py; these only prove the wiring. find_swing_points is always []
    # under _run()'s own mocking (same as every other divergence
    # trigger's real detection) - cvd_divergence.diagnostic_candidates
    # itself is mocked directly to inject a controlled candidate list.

    def test_cvd_divergence_diagnostic_logs_a_real_structural_candidate(self):
        candidate = {
            "structural_direction": "BULLISH", "cvd_data_found": True, "delta_usdt": 42.0,
        }

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_called_once()
        message = mock_log.call_args[0][0]
        self.assertIn("CVD_DIVERGENCE_DIAGNOSTIC", message)
        self.assertIn("structural_direction=BULLISH", message)
        self.assertIn("cvd_data_found=True", message)
        self.assertIn("delta_usdt=42.0", message)
        self.assertIn("confirmed=False", message)  # divergence itself is None by default

    def test_cvd_divergence_diagnostic_reports_confirmed_true_when_it_matches(self):
        candidate = {
            "structural_direction": "BULLISH", "cvd_data_found": True, "delta_usdt": 42.0,
        }

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(divergence_direction="BULLISH", divergence_level=88)

        self.assertIn("confirmed=True", mock_log.call_args[0][0])

    def test_cvd_divergence_diagnostic_silent_when_disabled(self):
        candidate = {"structural_direction": "BULLISH", "cvd_data_found": True, "delta_usdt": 42.0}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", False), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_not_called()

    def test_cvd_divergence_diagnostic_silent_when_trigger_disabled(self):
        # divergence_swings never gets computed at all when the trigger
        # itself is off - diagnostic_candidates never even gets called.
        candidate = {"structural_direction": "BULLISH", "cvd_data_found": True, "delta_usdt": 42.0}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(
                 cvd_divergence, "diagnostic_candidates", return_value=[candidate]
             ) as mock_diag, \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_diag.assert_not_called()
        mock_log.assert_not_called()

    def test_cvd_divergence_diagnostic_silent_when_no_structural_candidate(self):
        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_not_called()

    def test_cvd_divergence_diagnostic_never_changes_the_returned_result(self):
        candidate = {"structural_direction": "BULLISH", "cvd_data_found": True, "delta_usdt": 42.0}

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[candidate]):
            result_on = self._run()

        with patch.object(config, "CVD_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", False), \
             patch.object(cvd_divergence, "diagnostic_candidates", return_value=[candidate]):
            result_off = self._run()

        self.assertEqual(result_on, result_off)

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

    # config.OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED - same shape/
    # motivation as CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED above.
    # oi_divergence.diagnostic_candidates itself is mocked directly, same
    # reason (find_swing_points is always [] under _run()'s own mocking).

    def test_oi_divergence_diagnostic_logs_a_real_structural_candidate(self):
        candidate = {
            "structural_direction": "BEARISH", "oi_data_found": True, "delta_pct": 12.5,
        }

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_called_once()
        message = mock_log.call_args[0][0]
        self.assertIn("OI_DIVERGENCE_DIAGNOSTIC", message)
        self.assertIn("structural_direction=BEARISH", message)
        self.assertIn("oi_data_found=True", message)
        self.assertIn("delta_pct=12.5", message)
        self.assertIn("confirmed=False", message)

    def test_oi_divergence_diagnostic_reports_confirmed_true_when_it_matches(self):
        candidate = {
            "structural_direction": "BULLISH", "oi_data_found": True, "delta_pct": 12.5,
        }

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(oi_divergence_direction="BULLISH", oi_divergence_level=88)

        self.assertIn("confirmed=True", mock_log.call_args[0][0])

    def test_oi_divergence_diagnostic_silent_when_disabled(self):
        candidate = {"structural_direction": "BULLISH", "oi_data_found": True, "delta_pct": 12.5}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", False), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[candidate]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_not_called()

    def test_oi_divergence_diagnostic_silent_when_trigger_disabled(self):
        candidate = {"structural_direction": "BULLISH", "oi_data_found": True, "delta_pct": 12.5}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", False), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(
                 oi_divergence, "diagnostic_candidates", return_value=[candidate]
             ) as mock_diag, \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_diag.assert_not_called()
        mock_log.assert_not_called()

    def test_oi_divergence_diagnostic_silent_when_no_structural_candidate(self):
        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", False), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[]), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run()

        mock_log.assert_not_called()

    def test_oi_divergence_diagnostic_never_changes_the_returned_result(self):
        candidate = {"structural_direction": "BULLISH", "oi_data_found": True, "delta_pct": 12.5}

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[candidate]):
            result_on = self._run()

        with patch.object(config, "OI_DIVERGENCE_TRIGGER_ENABLED", True), \
             patch.object(config, "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", False), \
             patch.object(oi_divergence, "diagnostic_candidates", return_value=[candidate]):
            result_off = self._run()

        self.assertEqual(result_on, result_off)

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

    # config.LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED - real-magnitude
    # observability for the 2026-08-25 finding that LIQUIDATION_SWEEP_
    # CONFIRMED has never produced a trade. Read-only: must never affect
    # `result` itself, only whether log_info gets called.

    def test_liquidation_sweep_diagnostic_logs_on_a_real_sweep_with_available_snapshot(self):
        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(sweep_direction="BULLISH")

        mock_log.assert_called_once()
        self.assertIn("LIQUIDATION_SWEEP_DIAGNOSTIC", mock_log.call_args[0][0])
        self.assertIn("total_notional=90000", mock_log.call_args[0][0])

    def test_liquidation_sweep_diagnostic_silent_when_disabled(self):
        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", False), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(sweep_direction="BULLISH")

        mock_log.assert_not_called()

    def test_liquidation_sweep_diagnostic_silent_when_no_sweep_occurred(self):
        analysis = dict(LTF_BULLISH_BREAK)
        analysis["live_break"] = {"broken": False}

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(ltf_analysis=analysis, sweep_direction=None)

        mock_log.assert_not_called()

    def test_liquidation_sweep_diagnostic_silent_when_snapshot_unavailable(self):
        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", True), \
             patch.object(signal_engine, "log_info") as mock_log:
            self._run(sweep_direction="BULLISH", liquidation_snapshot={"available": False})

        mock_log.assert_not_called()

    def test_liquidation_sweep_diagnostic_never_changes_the_returned_result(self):
        # Same candidate/result whether the diagnostic is on or off -
        # purely observational, never a side channel into the real signal.
        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", True):
            result_on = self._run(sweep_direction="BULLISH")

        with patch.object(config, "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", True), \
             patch.object(config, "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", False):
            result_off = self._run(sweep_direction="BULLISH")

        self.assertEqual(result_on, result_off)

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
                "BTCUSDT", [{"open_time": 0, "close": 100}], _ltf_candles(93.0),
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


class TakerFlowAgreesTests(unittest.TestCase):
    """signal_engine.taker_flow_agrees - config.CVD_DIVERGENCE_TAKER_FLOW_
    REJECT_ENABLED. Unlike long_short_favorable above (contrarian), this
    is a plain confirmation reading: >1 = more taker BUY volume than
    SELL volume."""

    def test_none_ratio_is_none(self):
        self.assertIsNone(signal_engine.taker_flow_agrees("BUY", None))

    def test_buy_agrees_when_ratio_above_1(self):
        self.assertTrue(signal_engine.taker_flow_agrees("BUY", 1.3))

    def test_buy_disagrees_when_ratio_below_1(self):
        self.assertFalse(signal_engine.taker_flow_agrees("BUY", 0.8))

    def test_sell_agrees_when_ratio_below_1(self):
        self.assertTrue(signal_engine.taker_flow_agrees("SELL", 0.8))

    def test_sell_disagrees_when_ratio_above_1(self):
        self.assertFalse(signal_engine.taker_flow_agrees("SELL", 1.3))


class DirectionStillConfirmedTests(unittest.TestCase):
    """config.DCA_BREAKEVEN_CONFIRMATION_ENABLED - the trend/order-flow-
    health subset of evaluate()'s own gates, reused against an ALREADY
    OPEN position's side. Deliberately fails safe (missing/unavailable
    data for a required check = NOT confirmed), the opposite convention
    from evaluate()'s own gates."""

    def _run(
        self, side="BUY", htf_trend="BULLISH", htf_trend_ema=95.0, current_price=100.0,
        cvd_score=0.5, efficiency_ratio=0.5, htf_available=True, ltf_available=True,
        cvd_available=True, htf_freshness_enabled=True, efficiency_gate_enabled=True,
        min_cvd=0.15, chop_threshold=0.3,
    ):
        htf_structure = {"available": htf_available, "trend": htf_trend}
        ltf_analysis = {"available": ltf_available, "efficiency_ratio": efficiency_ratio}
        cvd_snapshot = {"available": cvd_available, "cvd_score": cvd_score}

        with patch.object(market_structure, "structure_state", return_value=htf_structure), \
             patch.object(market_structure, "exponential_moving_average", return_value=htf_trend_ema), \
             patch.object(market_structure, "analyze", return_value=ltf_analysis), \
             patch.object(config, "HTF_TREND_FRESHNESS_ENABLED", htf_freshness_enabled), \
             patch.object(config, "EFFICIENCY_RATIO_GATE_ENABLED", efficiency_gate_enabled), \
             patch.object(config, "SIGNAL_MIN_CVD_SCORE", min_cvd), \
             patch.object(config, "EFFICIENCY_RATIO_CHOP_THRESHOLD", chop_threshold):
            return signal_engine.direction_still_confirmed(
                side, ["htf_candle"], ["ltf_candle"], cvd_snapshot, current_price
            )

    def test_all_checks_passing_confirms(self):
        confirmed, detail = self._run()

        self.assertTrue(confirmed)
        self.assertTrue(detail["htf_trend_agrees"])
        self.assertTrue(detail["htf_trend_stale_agrees"])
        self.assertTrue(detail["cvd_confirmed"])
        self.assertTrue(detail["market_not_choppy"])

    def test_htf_trend_disagreeing_fails(self):
        confirmed, detail = self._run(htf_trend="BEARISH")

        self.assertFalse(confirmed)
        self.assertFalse(detail["htf_trend_agrees"])

    def test_htf_structure_unavailable_fails_safe(self):
        confirmed, detail = self._run(htf_available=False)

        self.assertFalse(confirmed)
        self.assertIsNone(detail["htf_trend_agrees"])

    def test_htf_trend_stale_disagreeing_fails(self):
        # BUY, but current_price already back below the HTF trend EMA.
        confirmed, detail = self._run(current_price=90.0, htf_trend_ema=95.0)

        self.assertFalse(confirmed)
        self.assertFalse(detail["htf_trend_stale_agrees"])

    def test_htf_trend_stale_check_skipped_when_freshness_disabled(self):
        confirmed, detail = self._run(
            current_price=90.0, htf_trend_ema=95.0, htf_freshness_enabled=False,
        )

        self.assertTrue(confirmed)
        self.assertIsNone(detail["htf_trend_stale_agrees"])

    def test_missing_htf_trend_ema_fails_safe_when_freshness_enabled(self):
        confirmed, detail = self._run(htf_trend_ema=None)

        self.assertFalse(confirmed)
        self.assertFalse(detail["htf_trend_stale_agrees"])

    def test_cvd_below_threshold_fails(self):
        confirmed, detail = self._run(cvd_score=0.05, min_cvd=0.15)

        self.assertFalse(confirmed)
        self.assertFalse(detail["cvd_confirmed"])

    def test_cvd_unavailable_fails_safe(self):
        confirmed, detail = self._run(cvd_available=False)

        self.assertFalse(confirmed)
        self.assertFalse(detail["cvd_confirmed"])

    def test_choppy_market_fails(self):
        confirmed, detail = self._run(efficiency_ratio=0.1, chop_threshold=0.3)

        self.assertFalse(confirmed)
        self.assertFalse(detail["market_not_choppy"])

    def test_choppy_check_skipped_when_gate_disabled(self):
        confirmed, detail = self._run(
            efficiency_ratio=0.1, chop_threshold=0.3, efficiency_gate_enabled=False,
        )

        self.assertTrue(confirmed)
        self.assertIsNone(detail["market_not_choppy"])

    def test_sell_side_mirrors_buy_for_all_checks(self):
        confirmed, _ = self._run(
            side="SELL", htf_trend="BEARISH", current_price=90.0, htf_trend_ema=95.0,
            cvd_score=-0.5, min_cvd=0.15,
        )

        self.assertTrue(confirmed)

    def test_missing_htf_candles_fails_safe(self):
        confirmed, detail = signal_engine.direction_still_confirmed(
            "BUY", None, ["ltf_candle"], {"available": True, "cvd_score": 0.5}, 100.0
        )

        self.assertFalse(confirmed)

    def test_missing_ltf_candles_fails_safe(self):
        confirmed, detail = signal_engine.direction_still_confirmed(
            "BUY", ["htf_candle"], None, {"available": True, "cvd_score": 0.5}, 100.0
        )

        self.assertFalse(confirmed)


if __name__ == "__main__":
    unittest.main()
