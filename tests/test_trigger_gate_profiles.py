import unittest
from unittest.mock import patch

import config


ALL_TRIGGERS = (
    "STRUCTURE_BREAK", "OB_FVG_RETEST", "LIQUIDITY_SWEEP", "CHOCH_RETEST",
    "CVD_DIVERGENCE", "ORDER_BLOCK_RETEST", "OI_DIVERGENCE",
    "LIQUIDATION_SWEEP_CONFIRMED", "EMA_PULLBACK",
)
ALL_VARIABLE_GATES = frozenset({
    "AGAINST_HTF_BIAS", "HTF_TREND_STALE", "MARKET_CHOPPY",
    "NOT_IN_OTE", "NO_ORDER_BLOCK_OR_FVG", "CVD_NOT_CONFIRMED",
    "DEPTH_TREND_MIN_CONSISTENCY",
})
REVERSAL_EXEMPT_TRIGGERS = (
    "CVD_DIVERGENCE", "OI_DIVERGENCE", "LIQUIDATION_SWEEP_CONFIRMED", "CHOCH_RETEST",
)


class TriggerGateProfilesTests(unittest.TestCase):
    """config.trigger_gate_profiles() is the single source signal_engine
    reads from - these lock in the exact per-trigger table this project's
    architecture depends on, independent of any real signal evaluation
    (much cheaper/faster than exercising the full gate cascade for every
    combination, and pinpoints a wrong entry immediately instead of via a
    confusing downstream signal_engine assertion)."""

    def setUp(self):
        # Every profile-driving flag pinned to its documented default so
        # these tests don't depend on whatever the real .env happens to
        # have (same discipline test_signal_engine.py already uses for
        # its own trigger/gate flags).
        defaults = {
            "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED": True,
            "HTF_TREND_STALE_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED": True,
            "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED": True,
            "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED": True,
            "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED": True,
            "DEPTH_TREND_MIN_CONSISTENCY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED": True,
        }
        for name, value in defaults.items():
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_every_trigger_has_a_profile_entry(self):
        profiles = config.trigger_gate_profiles()
        self.assertEqual(set(profiles.keys()), set(ALL_TRIGGERS))

    def test_structure_break_gets_every_variable_gate(self):
        profiles = config.trigger_gate_profiles()
        self.assertEqual(profiles["STRUCTURE_BREAK"], ALL_VARIABLE_GATES)

    def test_cvd_divergence_is_exempt_from_everything_but_no_order_block_or_fvg(self):
        profiles = config.trigger_gate_profiles()
        self.assertEqual(profiles["CVD_DIVERGENCE"], {"NO_ORDER_BLOCK_OR_FVG"})

    def test_oi_divergence_and_liquidation_sweep_confirmed_keep_cvd_not_confirmed(self):
        # Narrower than CVD_DIVERGENCE's own exemption on purpose - see
        # config.py's CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED
        # comment for why these two don't share CVD_DIVERGENCE's specific
        # self-defeating structure.
        profiles = config.trigger_gate_profiles()
        self.assertEqual(profiles["OI_DIVERGENCE"], {"CVD_NOT_CONFIRMED", "NO_ORDER_BLOCK_OR_FVG"})
        self.assertEqual(
            profiles["LIQUIDATION_SWEEP_CONFIRMED"],
            {"CVD_NOT_CONFIRMED", "NO_ORDER_BLOCK_OR_FVG"},
        )

    def test_choch_retest_exempt_from_trend_agreement_but_not_chop(self):
        # CHOCH_RETEST joins the reversal group for AGAINST_HTF_BIAS/
        # HTF_TREND_STALE only, not MARKET_CHOPPY - see config.py's
        # _TREND_AGREEMENT_EXEMPT_TRIGGERS comment.
        profiles = config.trigger_gate_profiles()
        self.assertEqual(
            profiles["CHOCH_RETEST"],
            {"MARKET_CHOPPY", "CVD_NOT_CONFIRMED", "NO_ORDER_BLOCK_OR_FVG"},
        )

    def test_ob_fvg_retest_and_order_block_retest_skip_the_tautological_gate(self):
        profiles = config.trigger_gate_profiles()
        self.assertNotIn("NO_ORDER_BLOCK_OR_FVG", profiles["OB_FVG_RETEST"])
        self.assertNotIn("NO_ORDER_BLOCK_OR_FVG", profiles["ORDER_BLOCK_RETEST"])

    def test_only_structure_break_keeps_not_in_ote(self):
        profiles = config.trigger_gate_profiles()

        for trigger in ALL_TRIGGERS:
            if trigger == "STRUCTURE_BREAK":
                self.assertIn("NOT_IN_OTE", profiles[trigger])
            else:
                self.assertNotIn("NOT_IN_OTE", profiles[trigger])

    def test_all_flags_off_makes_every_gate_universal(self):
        with patch.object(config, "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False), \
             patch.object(config, "HTF_TREND_STALE_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False), \
             patch.object(config, "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False), \
             patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", False), \
             patch.object(config, "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED", False), \
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False):
            profiles = config.trigger_gate_profiles()

        for trigger in ALL_TRIGGERS:
            expected = (
                ALL_VARIABLE_GATES - {"NO_ORDER_BLOCK_OR_FVG"}
                if trigger in ("OB_FVG_RETEST", "ORDER_BLOCK_RETEST")
                else ALL_VARIABLE_GATES
            )
            self.assertEqual(profiles[trigger], expected)

    def test_depth_trend_min_consistency_exempt_only_for_reversal_group(self):
        # Same _TREND_AGREEMENT_EXEMPT_TRIGGERS group AGAINST_HTF_BIAS/
        # HTF_TREND_STALE already use (the 3 reversal triggers + CHOCH_
        # RETEST) - a reversal trigger's whole thesis is that book
        # pressure is CHANGING right now, so requiring it to have already
        # been stable before the change punishes exactly the freshness
        # that makes it a genuine reversal.
        profiles = config.trigger_gate_profiles()

        for trigger in ALL_TRIGGERS:
            if trigger in REVERSAL_EXEMPT_TRIGGERS:
                self.assertNotIn("DEPTH_TREND_MIN_CONSISTENCY", profiles[trigger])
            else:
                self.assertIn("DEPTH_TREND_MIN_CONSISTENCY", profiles[trigger])

    def test_recomputes_live_rather_than_caching_at_import_time(self):
        # Real bug caught 2026-08-17: an earlier version computed this
        # once into a module-level constant, so patch.object(config, ...)
        # overrides in tests (and any future live config reload) were
        # silently ignored. Flipping a flag and calling the function
        # again must observe the new value immediately.
        with patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", True):
            self.assertNotIn("NOT_IN_OTE", config.trigger_gate_profiles()["CHOCH_RETEST"])

        with patch.object(config, "OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", False):
            self.assertIn("NOT_IN_OTE", config.trigger_gate_profiles()["CHOCH_RETEST"])


if __name__ == "__main__":
    unittest.main()
