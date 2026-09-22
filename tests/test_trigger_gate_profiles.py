import unittest
from unittest.mock import patch

import config


ALL_TRIGGERS = (
    "STRUCTURE_BREAK", "OB_FVG_RETEST", "LIQUIDITY_SWEEP", "CHOCH_RETEST",
    "CVD_DIVERGENCE", "ORDER_BLOCK_RETEST", "OI_DIVERGENCE",
    "LIQUIDATION_SWEEP_CONFIRMED", "EMA_PULLBACK", "BREAK_OTE_RETEST",
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
            # Pinned empty for the same reason as the booleans above, and
            # for a concrete one: the live .env sets this to a real list,
            # which would otherwise break test_structure_break_gets_every_
            # variable_gate the moment it is deployed.
            "MARKET_CHOPPY_EXEMPT_TRIGGERS": [],
            # Same reason, same concrete risk: the enabling decision for
            # this list is STRUCTURE_BREAK, which is precisely what
            # test_structure_break_gets_every_variable_gate asserts about.
            "OTE_GATE_EXEMPT_TRIGGERS": [],
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
             patch.object(config, "DEPTH_TREND_MIN_CONSISTENCY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False), \
             patch.object(config, "MARKET_CHOPPY_EXEMPT_TRIGGERS", []):
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

    # config.MARKET_CHOPPY_EXEMPT_TRIGGERS (2026-09-22) - the measured
    # per-trigger scoping of MARKET_CHOPPY, on top of the structural
    # reversal-trigger skip. See that setting's config.py comment for the
    # full evidence table and why only three triggers are listed.
    def test_market_choppy_exempt_triggers_removes_the_gate_for_listed_only(self):
        listed = ["STRUCTURE_BREAK", "ORDER_BLOCK_RETEST", "OB_FVG_RETEST"]

        with patch.object(config, "MARKET_CHOPPY_EXEMPT_TRIGGERS", listed):
            profiles = config.trigger_gate_profiles()

        for trigger in listed:
            self.assertNotIn("MARKET_CHOPPY", profiles[trigger])

        # EMA_PULLBACK/LIQUIDITY_SWEEP/CHOCH_RETEST were measured and
        # deliberately left gated - a null, an n=87 SELL cell, and an n=93
        # cell respectively. They must not pick the exemption up by
        # association with the three above.
        for trigger in ("EMA_PULLBACK", "LIQUIDITY_SWEEP", "CHOCH_RETEST"):
            self.assertIn("MARKET_CHOPPY", profiles[trigger])

    def test_market_choppy_exempt_triggers_empty_by_default_still_gates(self):
        # env_str_list's own gotcha (CONFLUENCE_SHADOW_PROBE_EXCLUDE_
        # TRIGGERS hit this once, EMA_TREND_MIXED_EXEMPT_TRIGGERS guards it
        # too) - the default must be [], never a stale non-empty list, so an
        # unconfigured deploy keeps the gate universal.
        with patch.object(config, "MARKET_CHOPPY_EXEMPT_TRIGGERS", []):
            profiles = config.trigger_gate_profiles()

        for trigger in ALL_TRIGGERS:
            if trigger in ("CVD_DIVERGENCE", "OI_DIVERGENCE", "LIQUIDATION_SWEEP_CONFIRMED"):
                continue            # structural skip, tested separately
            self.assertIn("MARKET_CHOPPY", profiles[trigger])

    def test_market_choppy_exempt_triggers_disturbs_no_other_gate(self):
        # The discard must be surgical: exempting a trigger from
        # MARKET_CHOPPY must leave the rest of its profile byte-identical,
        # or this flag becomes a way to silently drop unrelated protection.
        before = config.trigger_gate_profiles()

        with patch.object(config, "MARKET_CHOPPY_EXEMPT_TRIGGERS", ["STRUCTURE_BREAK"]):
            after = config.trigger_gate_profiles()

        self.assertEqual(after["STRUCTURE_BREAK"], before["STRUCTURE_BREAK"] - {"MARKET_CHOPPY"})

        for trigger in ALL_TRIGGERS:
            if trigger != "STRUCTURE_BREAK":
                self.assertEqual(after[trigger], before[trigger])

    def test_market_choppy_exempt_triggers_is_additive_to_the_reversal_skip(self):
        # The two discards are deliberately separate - one structural, one
        # empirical. Turning the structural flag off must not resurrect
        # MARKET_CHOPPY for a trigger the empirical list also exempts.
        with patch.object(config, "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", False), \
             patch.object(config, "MARKET_CHOPPY_EXEMPT_TRIGGERS", ["CVD_DIVERGENCE"]):
            profiles = config.trigger_gate_profiles()

        self.assertNotIn("MARKET_CHOPPY", profiles["CVD_DIVERGENCE"])
        self.assertIn("MARKET_CHOPPY", profiles["OI_DIVERGENCE"])

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


class OteGateExemptTriggersTests(unittest.TestCase):
    """config.OTE_GATE_EXEMPT_TRIGGERS - 2026-09-22 Phase 1d. NOT_IN_OTE
    blocks 98.4% of STRUCTURE_BREAK detections because a break fires at a
    retracement depth of ~0.16 while the band demands 0.705-0.79. These lock
    in that the exemption is expressible at all - the pre-existing
    OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED lever points the wrong way, since
    turning it off applies the gate to EVERY trigger."""

    def setUp(self):
        for name, value in (
            ("OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", True),
            ("MARKET_CHOPPY_EXEMPT_TRIGGERS", []),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_default_is_empty_so_structure_break_keeps_the_gate(self):
        with patch.object(config, "OTE_GATE_EXEMPT_TRIGGERS", []):
            self.assertIn(
                "NOT_IN_OTE", config.trigger_gate_profiles()["STRUCTURE_BREAK"]
            )

    def test_listing_structure_break_removes_only_its_own_ote_gate(self):
        with patch.object(config, "OTE_GATE_EXEMPT_TRIGGERS", ["STRUCTURE_BREAK"]):
            profiles = config.trigger_gate_profiles()

        self.assertNotIn("NOT_IN_OTE", profiles["STRUCTURE_BREAK"])
        # every OTHER variable gate STRUCTURE_BREAK carries is untouched
        self.assertEqual(
            profiles["STRUCTURE_BREAK"], ALL_VARIABLE_GATES - {"NOT_IN_OTE"}
        )

    def test_the_source_default_is_an_empty_list(self):
        """Asserted against the source literal rather than the imported
        value, which the live .env can change - the same shape
        test_market_structure.py uses for its own default proofs."""
        import re
        from pathlib import Path

        source = Path(config.__file__).read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r'OTE_GATE_EXEMPT_TRIGGERS\s*=\s*env_str_list\(\s*"OTE_GATE_EXEMPT_TRIGGERS"\s*,\s*\[\]\s*\)',
            source,
        )
        self.assertIsNotNone(match, "OTE_GATE_EXEMPT_TRIGGERS must default to []")


if __name__ == "__main__":
    unittest.main()
