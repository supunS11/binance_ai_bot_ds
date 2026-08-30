import unittest
from unittest.mock import patch

import config
import risk_manager


class ComputeStopLossTests(unittest.TestCase):
    def test_buy_stop_is_below_structure_level_minus_atr_buffer(self):
        signal = {"structure_level": 100, "atr": 2}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertEqual(sl, 99.0)  # 100 - (2 * 0.5)

    def test_sell_stop_is_above_structure_level_plus_atr_buffer(self):
        signal = {"structure_level": 100, "atr": 2}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_stop_loss(signal, "SELL")

        self.assertEqual(sl, 101.0)

    def test_missing_structure_level_returns_none(self):
        sl = risk_manager.compute_stop_loss({"structure_level": None, "atr": 2}, "BUY")
        self.assertIsNone(sl)


class MinStopDistanceFloorTests(unittest.TestCase):
    """A structure level landing pathologically close to entry (fast/noisy
    market, tight fractal window) must not be allowed through as-is - it
    gets hit by ordinary noise, and risk-based sizing would compensate
    with an oversized position to match the tiny distance."""

    def test_pathologically_tight_buy_stop_gets_widened_to_the_floor(self):
        # Structure level just 0.02% below entry - realistic tiny-stop case.
        signal = {"structure_level": 99.98, "atr": 0.001, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0.15), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.3):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertAlmostEqual(sl, 99.7)  # 100 - 0.3%

    def test_pathologically_tight_sell_stop_gets_widened_to_the_floor(self):
        signal = {"structure_level": 100.02, "atr": 0.001, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0.15), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.3):
            sl = risk_manager.compute_stop_loss(signal, "SELL")

        self.assertAlmostEqual(sl, 100.3)  # 100 + 0.3%

    def test_a_stop_already_wider_than_the_floor_is_left_untouched(self):
        signal = {"structure_level": 95.0, "atr": 0, "entry_price": 100.0}

        with patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.3):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertEqual(sl, 95.0)

    def test_zero_min_pct_disables_the_floor(self):
        signal = {"structure_level": 99.999, "atr": 0, "entry_price": 100.0}

        with patch.object(config, "MIN_STOP_DISTANCE_PCT", 0):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertEqual(sl, 99.999)

    def test_missing_entry_price_skips_the_floor_without_crashing(self):
        signal = {"structure_level": 99.999, "atr": 0}

        with patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.3):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertEqual(sl, 99.999)


class MinStopDistanceAtrFloorTests(unittest.TestCase):
    """The floor is the WIDER of MIN_STOP_DISTANCE_PCT and
    MIN_STOP_DISTANCE_ATR_MULTIPLE*atr, not the percentage alone - real
    evidence (2026-08-13: a direct log-distance trace of 24 real trades
    plus three consecutive journal_analysis.py pulls) showed the flat 0.6%
    floor alone getting hit on ~40% of trades, every one resolving as a
    loss or scratch, because 0.6% is inside normal 1h noise for many of
    this watchlist's volatile small/mid-cap symbols."""

    def test_atr_floor_wins_when_wider_than_the_pct_floor_buy(self):
        # pct floor: 100 * 0.1% = 0.1. atr floor: 2 * 1.0 = 2.0 (wider).
        signal = {"structure_level": 99.95, "atr": 2, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.1), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 1.0):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertAlmostEqual(sl, 98.0)  # 100 - (2 * 1.0)

    def test_atr_floor_wins_when_wider_than_the_pct_floor_sell(self):
        signal = {"structure_level": 100.05, "atr": 2, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.1), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 1.0):
            sl = risk_manager.compute_stop_loss(signal, "SELL")

        self.assertAlmostEqual(sl, 102.0)  # 100 + (2 * 1.0)

    def test_pct_floor_still_wins_when_it_is_wider_than_the_atr_floor(self):
        # pct floor: 100 * 0.6% = 0.6. atr floor: 0.1 * 1.0 = 0.1 (narrower).
        signal = {"structure_level": 99.98, "atr": 0.1, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.6), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 1.0):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertAlmostEqual(sl, 99.4)  # 100 - 0.6%

    def test_zero_atr_multiple_disables_the_atr_floor(self):
        signal = {"structure_level": 99.95, "atr": 2, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.1), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 0):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertAlmostEqual(sl, 99.9)  # 100 - 0.1%, atr floor contributes nothing

    def test_a_stop_wider_than_both_floors_is_left_untouched(self):
        signal = {"structure_level": 90.0, "atr": 0.1, "entry_price": 100.0}

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "MIN_STOP_DISTANCE_PCT", 0.6), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 1.0):
            sl = risk_manager.compute_stop_loss(signal, "BUY")

        self.assertEqual(sl, 90.0)


class ComputeTargetsTests(unittest.TestCase):
    def test_buy_targets_are_r_multiples_above_entry(self):
        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, tp2 = risk_manager.compute_targets(entry_price=100, sl_price=98, side="BUY")

        self.assertEqual(tp1, 102)  # 100 + 2*1
        self.assertEqual(tp2, 104)  # 100 + 2*2

    def test_sell_targets_are_r_multiples_below_entry(self):
        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, tp2 = risk_manager.compute_targets(entry_price=100, sl_price=102, side="SELL")

        self.assertEqual(tp1, 98)
        self.assertEqual(tp2, 96)

    def test_zero_risk_distance_returns_none_none(self):
        tp1, tp2 = risk_manager.compute_targets(entry_price=100, sl_price=100, side="BUY")
        self.assertIsNone(tp1)
        self.assertIsNone(tp2)


class ComputeStaticTp1StructureTp2Tests(unittest.TestCase):
    """config.TP_STATIC_ROI_ENABLED - TP1 is a fixed ROI% price, TP2 is
    the SAME real-liquidity-first target compute_targets' own TP2 half
    would produce, just fed the static TP1 price as its own "at least 1R
    beyond TP1" floor input instead of a structure-resolved one."""

    def test_buy_tp1_is_static_tp2_falls_back_to_r_multiple(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0):
            tp1, tp2 = risk_manager.compute_static_tp1_structure_tp2(
                entry_price=100, sl_price=98, side="BUY", roi_pct=40,
            )

        self.assertAlmostEqual(tp1, 104.0)  # 40% ROI / 10x leverage = 4% price move
        # risk_distance=2, tp1 sits at 2.0R -> TP2 floor = 2.0+1.0 = 3.0R,
        # no pools -> plain fallback: 100 + 3*2.
        self.assertAlmostEqual(tp2, 106.0)

    def test_sell_mirrors_buy(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0):
            tp1, tp2 = risk_manager.compute_static_tp1_structure_tp2(
                entry_price=100, sl_price=102, side="SELL", roi_pct=40,
            )

        self.assertAlmostEqual(tp1, 96.0)
        self.assertAlmostEqual(tp2, 94.0)

    def test_tp2_targets_a_real_pool_beyond_the_static_tp1(self):
        pools = [{"type": "BUY_SIDE", "price": 108}]  # 4R - clears a 3R floor

        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0):
            tp1, tp2 = risk_manager.compute_static_tp1_structure_tp2(
                entry_price=100, sl_price=98, side="BUY", roi_pct=40, pools=pools,
            )

        self.assertAlmostEqual(tp1, 104.0)
        self.assertEqual(tp2, 108)  # real pool used instead of the fallback

    def test_tp1_unavailable_returns_none_none(self):
        # LEVERAGE=0 makes price_at_roi_pct fail.
        with patch.object(config, "LEVERAGE", 0):
            tp1, tp2 = risk_manager.compute_static_tp1_structure_tp2(
                entry_price=100, sl_price=98, side="BUY", roi_pct=40,
            )

        self.assertIsNone(tp1)
        self.assertIsNone(tp2)

    def test_zero_risk_distance_still_returns_the_static_tp1(self):
        # TP1 doesn't depend on risk_distance at all - only TP2 does.
        with patch.object(config, "LEVERAGE", 10):
            tp1, tp2 = risk_manager.compute_static_tp1_structure_tp2(
                entry_price=100, sl_price=100, side="BUY", roi_pct=40,
            )

        self.assertAlmostEqual(tp1, 104.0)
        self.assertIsNone(tp2)


class StructureBasedTargetTests(unittest.TestCase):
    """TP1/TP2 should target a real liquidity pool when one exists with
    enough room, matching v7's structure-based TP - the R-multiple is
    only the minimum-room floor / fallback, not the primary target."""

    def test_buy_targets_the_nearest_qualifying_buy_side_pool(self):
        pools = [
            {"type": "BUY_SIDE", "price": 103},  # 1.5R - clears the 1R floor
            {"type": "BUY_SIDE", "price": 110},  # farther, should not be picked for TP1
        ]

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 4.0):
            tp1, tp2 = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 103)  # nearest pool clearing 1R (>=102)
        self.assertEqual(tp2, 110)  # next pool out, clears the widened TP2 floor

    def test_sell_targets_the_nearest_qualifying_sell_side_pool(self):
        pools = [
            {"type": "SELL_SIDE", "price": 97},
            {"type": "SELL_SIDE", "price": 90},
        ]

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 4.0):
            tp1, tp2 = risk_manager.compute_targets(100, 102, "SELL", pools=pools)

        self.assertEqual(tp1, 97)
        self.assertEqual(tp2, 90)

    def test_pool_too_close_to_clear_the_floor_is_ignored(self):
        # Only 0.5R away - doesn't clear a 1R minimum, must fall back.
        pools = [{"type": "BUY_SIDE", "price": 101}]

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, _ = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 102)  # fallback: 100 + 1R

    def test_wrong_side_pool_type_is_ignored(self):
        # A SELL_SIDE pool must never be used as a BUY's target.
        pools = [{"type": "SELL_SIDE", "price": 105}]

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, _ = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 102)  # fallback, the pool was never eligible

    def test_pool_beyond_the_max_bound_is_rejected_not_used(self):
        # Real bug seen live: the nearest *qualifying* pool can still be
        # absurdly far away (~20R) if nothing closer exists - that's not
        # an achievable TP1, so it must be rejected in favor of the
        # bounded fallback instead of blindly taking "nearest that clears
        # the floor" with no ceiling.
        pools = [{"type": "BUY_SIDE", "price": 140}]  # 20R away

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP1_MAX_R_MULTIPLE", 6.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0):
            tp1, tp2 = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 102)  # fallback: 100 + 1R, the 20R pool rejected
        self.assertEqual(tp2, 104)  # fallback: 100 + 2R

    def test_pool_within_bounds_is_still_used_normally(self):
        pools = [{"type": "BUY_SIDE", "price": 106}]  # 3R - within [1, 6]

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP1_MAX_R_MULTIPLE", 6.0):
            tp1, _ = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 106)

    def test_tp2_still_clears_tp1_when_tp1_used_a_far_structure_pool(self):
        # TP1 lands on a pool at 4R (well beyond its 1R floor); TP2's
        # floor must adapt to sit beyond that, not just the configured 2R.
        pools = [{"type": "BUY_SIDE", "price": 108}]  # 4R away

        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, tp2 = risk_manager.compute_targets(100, 98, "BUY", pools=pools)

        self.assertEqual(tp1, 108)
        self.assertGreater(tp2, tp1)  # fallback must clear tp1, not just 2R

    def test_no_pools_falls_back_to_pure_r_multiples(self):
        with patch.object(config, "TP1_R_MULTIPLE", 1.0), patch.object(config, "TP2_R_MULTIPLE", 2.0):
            tp1, tp2 = risk_manager.compute_targets(100, 98, "BUY", pools=None)

        self.assertEqual(tp1, 102)
        self.assertEqual(tp2, 104)


class NearestFavorableStructureRTests(unittest.TestCase):
    """Informational-only field (see signal_journal.py's
    nearest_favorable_sr_r comment) - unlike compute_targets/
    _find_structure_target, this has NO minimum-room floor: it reports
    whatever real pool is actually nearest, even one too close to ever
    qualify as a TP itself."""

    def test_reports_a_pool_too_close_to_ever_qualify_as_a_tp(self):
        # Only 0.5R away - StructureBasedTargetTests's equivalent fixture
        # (test_pool_too_close_to_clear_the_floor_is_ignored) shows this
        # same pool gets ignored by compute_targets and falls back to a
        # pure R-multiple TP1 instead - this function must still report it.
        pools = [{"type": "BUY_SIDE", "price": 101}]

        result = risk_manager.nearest_favorable_structure_r(pools, 100, "BUY", risk_distance=2)

        self.assertAlmostEqual(result, 0.5)

    def test_picks_the_nearest_pool_not_the_farther_one(self):
        pools = [
            {"type": "BUY_SIDE", "price": 103},  # 1.5R
            {"type": "BUY_SIDE", "price": 110},  # 5R
        ]

        result = risk_manager.nearest_favorable_structure_r(pools, 100, "BUY", risk_distance=2)

        self.assertAlmostEqual(result, 1.5)

    def test_sell_side_mirrors_buy(self):
        pools = [{"type": "SELL_SIDE", "price": 99}]

        result = risk_manager.nearest_favorable_structure_r(pools, 100, "SELL", risk_distance=2)

        self.assertAlmostEqual(result, 0.5)

    def test_wrong_side_pool_type_is_ignored(self):
        pools = [{"type": "SELL_SIDE", "price": 101}]

        result = risk_manager.nearest_favorable_structure_r(pools, 100, "BUY", risk_distance=2)

        self.assertIsNone(result)

    def test_no_pools_returns_none(self):
        result = risk_manager.nearest_favorable_structure_r(None, 100, "BUY", risk_distance=2)

        self.assertIsNone(result)

    def test_zero_risk_distance_returns_none(self):
        pools = [{"type": "BUY_SIDE", "price": 101}]

        result = risk_manager.nearest_favorable_structure_r(pools, 100, "BUY", risk_distance=0)

        self.assertIsNone(result)


class ComputeBreakevenPriceTests(unittest.TestCase):
    def test_buy_breakeven_is_slightly_above_entry(self):
        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            price = risk_manager.compute_breakeven_price(100, "BUY")

        self.assertAlmostEqual(price, 100.02)

    def test_sell_breakeven_is_slightly_below_entry(self):
        with patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            price = risk_manager.compute_breakeven_price(100, "SELL")

        self.assertAlmostEqual(price, 99.98)


class ComputeEarlyBreakevenPriceTests(unittest.TestCase):
    def test_zero_lock_multiple_falls_back_to_flat_breakeven(self):
        with patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            price = risk_manager.compute_early_breakeven_price(100, "BUY", 2.0)

        self.assertAlmostEqual(price, 100.02)

    def test_buy_locks_profit_at_the_configured_r_multiple(self):
        with patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3):
            price = risk_manager.compute_early_breakeven_price(100, "BUY", 2.0)

        self.assertAlmostEqual(price, 100.6)  # 100 + 2*0.3

    def test_sell_locks_profit_at_the_configured_r_multiple(self):
        with patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3):
            price = risk_manager.compute_early_breakeven_price(100, "SELL", 2.0)

        self.assertAlmostEqual(price, 99.4)  # 100 - 2*0.3

    def test_negative_lock_multiple_is_clamped_to_zero(self):
        with patch.object(config, "EARLY_BREAKEVEN_LOCK_R_MULTIPLE", -1), \
             patch.object(config, "BREAKEVEN_BUFFER_PCT", 0.02):
            price = risk_manager.compute_early_breakeven_price(100, "BUY", 2.0)

        self.assertAlmostEqual(price, 100.02)


class ComputeProfitProtectionLockPriceTests(unittest.TestCase):
    """config.PROFIT_PROTECTION_ENABLED - locks in
    PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1% of TP1's own ROI (at
    LEVERAGE), used as both the activation trigger and the lock target
    (position_manager._profit_protection_lock_price). All fixtures here
    use a TP1 ROI of 100% (entry=100, tp1=110/90, leverage=10) - well
    above PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT's default (50), so
    HIGH_TP1_ROI_THRESHOLD_PCT is pinned high here to keep exercising the
    plain (non-tiered) activation path this class is actually about - see
    ComputeProfitProtectionLockPriceHighTp1RoiTests for the tiering itself."""

    def test_buy_locks_the_configured_pct_of_tp1s_roi(self):
        # entry=100, tp1=110 -> tp1 move=10 -> tp1 ROI = 10/100*10*100=100%
        # -> 60% of that = 60% ROI -> price move = 60/100/10*100 = 6
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 110)

        self.assertAlmostEqual(price, 106)

    def test_sell_locks_the_configured_pct_of_tp1s_roi(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            price = risk_manager.compute_profit_protection_lock_price(100, "SELL", 90)

        self.assertAlmostEqual(price, 94)

    def test_zero_activation_pct_locks_exactly_at_entry(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 0), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 110)

        self.assertAlmostEqual(price, 100)

    def test_negative_activation_pct_is_clamped_to_zero(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", -10), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 200):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 110)

        self.assertAlmostEqual(price, 100)

    def test_none_tp1_price_returns_none(self):
        price = risk_manager.compute_profit_protection_lock_price(100, "BUY", None)
        self.assertIsNone(price)

    def test_zero_entry_price_returns_none(self):
        price = risk_manager.compute_profit_protection_lock_price(0, "BUY", 110)
        self.assertIsNone(price)

    def test_tp1_equal_to_entry_returns_none(self):
        # Zero TP1 ROI - nothing meaningful to protect a fraction of.
        price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 100)
        self.assertIsNone(price)

    def test_zero_leverage_returns_none(self):
        with patch.object(config, "LEVERAGE", 0):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 110)

        self.assertIsNone(price)


class ComputeProfitProtectionLockPriceHighTp1RoiTests(unittest.TestCase):
    """config.PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT/
    PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI - real operator
    concern (2026-08-18): waiting for 80% of TP1's own ROI is fine when
    TP1 only pays out a modest amount, but not when TP1 itself is a big
    ROI move, where 80% of it is still a lot of unrealized profit sitting
    unprotected. Above the threshold, the HIGH_ROI activation fraction is
    used instead of the plain one."""

    def test_tp1_roi_above_threshold_uses_the_high_roi_activation_pct(self):
        # entry=100, tp1=110, leverage=10 -> tp1 ROI=100%, above the 50%
        # threshold -> uses ACTIVATION_PCT_OF_TP1_HIGH_ROI (50), not the
        # plain ACTIVATION_PCT_OF_TP1 (80) -> price move = 50/100/10*100 = 5
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 80), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 50), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI", 50):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 110)

        self.assertAlmostEqual(price, 105)

    def test_tp1_roi_at_or_below_threshold_uses_the_plain_activation_pct(self):
        # entry=100, tp1=104, leverage=10 -> tp1 ROI=40%, below the 50%
        # threshold -> uses the plain ACTIVATION_PCT_OF_TP1 (80), not
        # HIGH_ROI -> target ROI = 80% of 40% = 32% -> price move =
        # 32/100/10*100 = 3.2
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 80), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 50), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI", 50):
            price = risk_manager.compute_profit_protection_lock_price(100, "BUY", 104)

        self.assertAlmostEqual(price, 103.2)

    def test_tp1_roi_exactly_at_threshold_uses_the_plain_activation_pct(self):
        # entry=100, tp1=105, leverage=10 -> tp1 ROI=50%, exactly at the
        # threshold - strictly-greater-than semantics, so this still uses
        # the plain ACTIVATION_PCT_OF_TP1 (80): target ROI = 80% of 50% =
        # 40% -> price move = 40/100/10*100 = 4
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 80), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 50), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI", 50):
            price = risk_manager.compute_profit_protection_lock_price(100, "SELL", 95)

        self.assertAlmostEqual(price, 96)

    def test_sell_side_above_threshold_uses_the_high_roi_activation_pct(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 80), \
             patch.object(config, "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 50), \
             patch.object(config, "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI", 50):
            price = risk_manager.compute_profit_protection_lock_price(100, "SELL", 90)

        self.assertAlmostEqual(price, 95)


class ComputeProfitProtectionTrailingFloorTests(unittest.TestCase):
    """Where the SL actually gets set once profit protection has armed -
    the max (BUY) / min (SELL) of a fixed worst-case floor
    (PROFIT_PROTECTION_LOCK_PCT_OF_TP1) and a cushion behind the best
    price reached so far (PROFIT_PROTECTION_RETRACE_PCT of the entry ->
    peak gain). entry=100, tp1=110 -> tp1 ROI=10/100*10*100=100%."""

    def test_retrace_dominates_when_peak_has_run_far_beyond_the_floor(self):
        # LOCK_PCT_OF_TP1=10% of 100%=10% ROI -> distance=1 -> floor=101.
        # RETRACE_PCT=50% of (peak-entry=8) retained=4 -> retrace=104.
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, 108)

        self.assertAlmostEqual(price, 104)

    def test_lock_floor_dominates_right_at_arming(self):
        # peak_price == entry + a tiny move just past the trigger: the
        # retrace cushion off such a small peak is smaller than the fixed
        # worst-case floor, so the floor wins instead.
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 50), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 90):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, 100.5)

        self.assertAlmostEqual(price, 105)  # 50% of 100% ROI -> distance=5

    def test_sell_side_mirrors_buy(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 50):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "SELL", 90, 92)

        self.assertAlmostEqual(price, 96)  # mirror of the BUY retrace-dominates case

    def test_retrace_never_gives_back_more_than_the_full_peak_gain(self):
        # RETRACE_PCT=0 -> the entire gain since entry is retained, so the
        # floor sits exactly at the peak itself.
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 0):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, 106)

        self.assertAlmostEqual(price, 106)

    def test_retrace_pct_100_falls_back_to_the_lock_floor(self):
        # RETRACE_PCT=100 -> none of the gain is retained by the retrace
        # side of the formula (retrace_price collapses to entry_price),
        # so the fixed lock floor always wins.
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 100):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, 108)

        self.assertAlmostEqual(price, 101)

    def test_retrace_pct_out_of_range_is_clamped(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10), \
             patch.object(config, "PROFIT_PROTECTION_RETRACE_PCT", 150):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, 108)

        self.assertAlmostEqual(price, 101)  # clamped to 100 -> same as retrace_pct=100 above

    def test_none_peak_price_falls_back_to_the_lock_floor(self):
        with patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 10):
            price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", 110, None)

        self.assertAlmostEqual(price, 101)

    def test_none_tp1_price_returns_none(self):
        price = risk_manager.compute_profit_protection_trailing_floor(100, "BUY", None, 105)
        self.assertIsNone(price)

    def test_zero_entry_price_returns_none(self):
        price = risk_manager.compute_profit_protection_trailing_floor(0, "BUY", 110, 105)
        self.assertIsNone(price)


class BuildTradePlanTests(unittest.TestCase):
    def setUp(self):
        # MAX_ENTRY_EXTENSION_R defaults to 0.5 in config, but this
        # fixture's default entry_price/structure_level gap sits well
        # above that - off by default here so tests not about this
        # feature aren't coupled to it; ExtensionCapTests turns it back
        # on locally. Same treatment for MAX_SL_ROI_PCT - MaxSlRoiTests
        # turns it back on locally.
        self.extension_patcher = patch.object(config, "MAX_ENTRY_EXTENSION_R", 0)
        self.extension_patcher.start()
        self.sl_roi_patcher = patch.object(config, "MAX_SL_ROI_PCT", 0)
        self.sl_roi_patcher.start()
        # config.TP_STATIC_ROI_ENABLED - these tests predate the single-TP
        # feature and assert the ordinary tp1_price/tp2_price split; pinned
        # off here so a real .env flip to True (the operator's own live
        # setting) can't silently switch this whole class onto the
        # single-tp_price shape out from under them. The static-roi tests
        # further down turn it back on locally, same pattern already used
        # for MAX_ENTRY_EXTENSION_R/MAX_SL_ROI_PCT above.
        self.static_roi_patcher = patch.object(config, "TP_STATIC_ROI_ENABLED", False)
        self.static_roi_patcher.start()
        # config.TP2_ENABLED - live .env now has this False (2026-08-28,
        # operator's own choice) but every test in this class predating
        # that change assumes the ordinary TP1+TP2 split; pinned True
        # here so that live value can't silently switch this whole class
        # onto the single-tp shape, same pattern as static_roi_patcher
        # above. Tp2DisabledTests below turns it back off locally.
        self.tp2_enabled_patcher = patch.object(config, "TP2_ENABLED", True)
        self.tp2_enabled_patcher.start()
        # config.TP1_CLOSE_PCT - live .env now has this at 100 (operator's
        # own experiment, before TP2_ENABLED existed as the real fix) -
        # 100 zeroes out tp2_quantity and trips TP_SPLIT_INVALID for every
        # test in this class that doesn't explicitly override it. Pinned
        # to the module's own coded default (config.py's env_float(...,
        # 50)) so this whole class is insulated from that live value,
        # same pattern as every other flag pinned above.
        self.tp1_close_pct_patcher = patch.object(config, "TP1_CLOSE_PCT", 50)
        self.tp1_close_pct_patcher.start()

    def tearDown(self):
        self.extension_patcher.stop()
        self.sl_roi_patcher.stop()
        self.static_roi_patcher.stop()
        self.tp2_enabled_patcher.stop()
        self.tp1_close_pct_patcher.stop()

    def _signal(self, side="BUY", entry_price=100, structure_level=98, atr=1):
        return {
            "signal": side,
            "symbol": "BTCUSDT",
            "entry_price": entry_price,
            "structure_level": structure_level,
            "atr": atr,
        }

    def test_happy_path_produces_a_full_plan(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertEqual(plan["sl_price"], 98)
        self.assertEqual(plan["tp1_price"], 102)
        self.assertEqual(plan["tp2_price"], 104)
        self.assertEqual(plan["quantity"], 10.0)
        self.assertEqual(plan["tp1_quantity"], 5.0)
        self.assertEqual(plan["tp2_quantity"], 5.0)
        self.assertIsNone(plan["tp_price"])
        self.assertFalse(plan["single_tp"])

    def test_static_roi_mode_uses_static_tp1_and_structure_tp2(self):
        # 2026-08-21: revised from a single whole-position target to a
        # normal TP1(partial)+TP2(remainder) shape - TP1 is the fixed
        # ROI% price, TP2 is the SAME real-liquidity-first structure
        # target the non-static path would have produced (no pools here,
        # so it falls back to the plain R-multiple distance).
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "TP_TARGET_ROI_PCT", 40), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0), \
             patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertFalse(plan["single_tp"])
        self.assertIsNone(plan["tp_price"])
        self.assertAlmostEqual(plan["tp1_price"], 104.0)  # 40% ROI / 10x leverage = 4% price move
        # risk_distance=2, tp1 landed at 2.0R -> TP2 floor becomes 3.0R
        # (tp1's own 2.0R + 1.0R), no pool -> plain fallback: 100 + 3*2.
        self.assertAlmostEqual(plan["tp2_price"], 106.0)
        self.assertEqual(plan["quantity"], 10.0)
        self.assertAlmostEqual(plan["tp1_quantity"], 5.0)
        self.assertAlmostEqual(plan["tp2_quantity"], 5.0)
        # config.RETRACEMENT_ENTRY_ENABLED - position_manager._resolve_tp1_price
        # needs this to know TP1 must be recomputed if the real fill price
        # ever differs from this plan's own.
        self.assertEqual(plan["tp1_static_roi_pct"], 40)

    def test_non_static_mode_leaves_tp1_static_roi_pct_none(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertIsNone(plan["tp1_static_roi_pct"])

    def test_static_roi_mode_mirrors_for_sell(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "TP_TARGET_ROI_PCT", 40), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP2_MAX_R_MULTIPLE", 10.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(
                self._signal(side="SELL", entry_price=100, structure_level=102), balance=1000,
            )

        self.assertEqual(status, "OK")
        self.assertAlmostEqual(plan["tp1_price"], 96.0)
        self.assertAlmostEqual(plan["tp2_price"], 94.0)  # 100 - 3*2, mirrored

    def test_static_roi_mode_respects_tp1_close_pct_same_as_normal_mode(self):
        # 2026-08-21: no longer a special exemption - static-TP1 mode now
        # goes through the exact same TP1_CLOSE_PCT split as the ordinary
        # ladder, so a 100% close (leaving nothing for TP2) must still be
        # rejected the same way.
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "TP_TARGET_ROI_PCT", 40), \
             patch.object(config, "TP1_CLOSE_PCT", 100), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertIsNone(plan)
        self.assertEqual(status, "TP_SPLIT_INVALID")

    def test_static_roi_mode_unavailable_rejects_the_plan(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "TP_TARGET_ROI_PCT", 40), \
             patch.object(config, "LEVERAGE", 0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertIsNone(plan)
        self.assertEqual(status, "TARGETS_UNAVAILABLE")

    # config.TP2_ENABLED - live change (2026-08-28), the operator's own
    # explicit choice to drop TP2 and close the whole position at TP1.
    # Reuses the single-TP shape _execute_dca already produces post-DCA -
    # these tests confirm build_trade_plan produces that exact shape from
    # signal time too, and that the TP1_CLOSE_PCT=100 footgun this flag
    # replaces (test_static_roi_mode_respects_tp1_close_pct_same_as_
    # normal_mode above) doesn't apply to it.

    def test_tp2_disabled_produces_a_single_tp_plan(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP2_ENABLED", False), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertTrue(plan["single_tp"])
        self.assertEqual(plan["tp_price"], plan["tp1_price"])
        self.assertIsNone(plan["tp1_quantity"])
        self.assertIsNone(plan["tp2_quantity"])

    def test_tp2_disabled_does_not_trigger_tp_split_invalid(self):
        # The exact failure mode TP1_CLOSE_PCT=100 would cause (see the
        # static-roi test above) - TP1_CLOSE_PCT is irrelevant once
        # single_tp bypasses the split entirely, so even a value that
        # would normally zero out tp2_quantity has no effect here.
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP2_ENABLED", False), \
             patch.object(config, "TP1_CLOSE_PCT", 100), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertTrue(plan["single_tp"])

    def test_tp2_disabled_combined_with_static_roi_uses_the_roi_target(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP2_ENABLED", False), \
             patch.object(config, "TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "TP_TARGET_ROI_PCT", 40), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertTrue(plan["single_tp"])
        self.assertAlmostEqual(plan["tp_price"], 104.0)  # same 40% ROI / 10x target as tp1_price
        self.assertAlmostEqual(plan["tp1_price"], 104.0)
        self.assertEqual(plan["tp1_static_roi_pct"], 40)

    def test_tp2_enabled_default_preserves_the_existing_split(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertFalse(plan["single_tp"])
        self.assertIsNone(plan["tp_price"])
        self.assertEqual(plan["tp1_quantity"], 5.0)
        self.assertEqual(plan["tp2_quantity"], 5.0)

    def test_nearest_favorable_sr_r_is_threaded_through_from_the_signal_pools(self):
        signal = dict(
            self._signal(),
            liquidity_pools=[{"type": "BUY_SIDE", "price": 101}],  # 0.5R - too close for TP1 itself
        )

        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(signal, balance=1000)

        self.assertEqual(status, "OK")
        self.assertAlmostEqual(plan["nearest_favorable_sr_r"], 0.5)
        # TP1 itself still falls back to the pure R-multiple, unaffected -
        # this field is purely diagnostic, never feeds target selection.
        self.assertEqual(plan["tp1_price"], 102)

    def test_nearest_favorable_sr_r_is_none_without_any_pools(self):
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertIsNone(plan["nearest_favorable_sr_r"])

    def test_plan_is_unaffected_by_limit_entry_mode(self):
        # config.LIMIT_ENTRY_MODE_ENABLED only changes execution.py/
        # position_manager.py (how the entry gets placed and tracked) -
        # the risk plan itself (SL/TP/sizing) must be identical either
        # way, since a LIMIT entry uses the same signal["entry_price"]
        # build_trade_plan already computes off of today.
        with patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(config, "TP1_CLOSE_PCT", 50), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", False):
                plan_off, status_off = risk_manager.build_trade_plan(self._signal(), balance=1000)
            with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True):
                plan_on, status_on = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status_off, status_on)
        self.assertEqual(plan_off, plan_on)

    def test_sl_unavailable_when_no_structure_level(self):
        plan, status = risk_manager.build_trade_plan(
            self._signal(structure_level=None), balance=1000
        )
        self.assertIsNone(plan)
        self.assertEqual(status, "SL_UNAVAILABLE")

    def test_sl_on_wrong_side_rejected_for_buy(self):
        # SL above entry for a BUY is nonsensical
        plan, status = risk_manager.build_trade_plan(
            self._signal(entry_price=100, structure_level=105, atr=0), balance=1000
        )
        self.assertIsNone(plan)
        self.assertEqual(status, "SL_ON_WRONG_SIDE")

    def test_zero_position_size_is_rejected(self):
        with patch.object(risk_manager, "calculate_position_size", return_value=0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertIsNone(plan)
        self.assertEqual(status, "POSITION_SIZE_ZERO")

    def test_tp1_close_pct_of_100_makes_the_split_invalid(self):
        with patch.object(config, "TP1_CLOSE_PCT", 100), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertIsNone(plan)
        self.assertEqual(status, "TP_SPLIT_INVALID")

    def test_invalid_entry_price_is_rejected(self):
        plan, status = risk_manager.build_trade_plan(
            self._signal(entry_price=0), balance=1000
        )
        self.assertIsNone(plan)
        self.assertEqual(status, "INVALID_ENTRY_PRICE")

    def test_full_confluence_scales_the_risk_budget_up_to_the_max_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25), \
             patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(risk_manager, "get_position_risk_budget", return_value=10.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=5.0) as mock_size:
            plan, status = risk_manager.build_trade_plan(
                dict(self._signal(), confluence_ratio=1.0), balance=1000
            )

        self.assertEqual(status, "OK")
        self.assertEqual(plan["size_multiplier"], 1.25)
        _, kwargs = mock_size.call_args
        self.assertAlmostEqual(kwargs["risk_budget_override"], 12.5)

    def test_zero_confluence_scales_the_risk_budget_down_to_the_min_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25), \
             patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(risk_manager, "get_position_risk_budget", return_value=10.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=5.0) as mock_size:
            plan, status = risk_manager.build_trade_plan(
                dict(self._signal(), confluence_ratio=0.0), balance=1000
            )

        self.assertEqual(status, "OK")
        self.assertEqual(plan["size_multiplier"], 0.5)
        _, kwargs = mock_size.call_args
        self.assertAlmostEqual(kwargs["risk_budget_override"], 5.0)

    def test_missing_confluence_ratio_behaves_as_full_normal_size(self):
        # Every trade still trades - a signal with no confluence_ratio at
        # all (e.g. an older/unrelated caller) must size exactly as it did
        # before this feature existed, not get penalized.
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(risk_manager, "get_position_risk_budget", return_value=10.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=5.0) as mock_size:
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertEqual(plan["size_multiplier"], 1.0)
        _, kwargs = mock_size.call_args
        self.assertAlmostEqual(kwargs["risk_budget_override"], 10.0)

    def test_confluence_sizing_disabled_always_uses_normal_size(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", False), \
             patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(risk_manager, "get_position_risk_budget", return_value=10.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=5.0) as mock_size:
            plan, status = risk_manager.build_trade_plan(
                dict(self._signal(), confluence_ratio=0.0), balance=1000
            )

        self.assertEqual(status, "OK")
        self.assertEqual(plan["size_multiplier"], 1.0)
        _, kwargs = mock_size.call_args
        self.assertAlmostEqual(kwargs["risk_budget_override"], 10.0)

    def test_flat_sizing_mode_scales_margin_instead_of_risk_budget(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25), \
             patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", False), \
             patch.object(config, "MARGIN_PER_TRADE", 20), \
             patch.object(risk_manager, "calculate_position_size", return_value=5.0) as mock_size:
            plan, status = risk_manager.build_trade_plan(
                dict(self._signal(), confluence_ratio=1.0), balance=1000
            )

        self.assertEqual(status, "OK")
        _, kwargs = mock_size.call_args
        self.assertAlmostEqual(kwargs["margin_override"], 25.0)
        self.assertNotIn("risk_budget_override", kwargs)


class EntryExtensionCapTests(unittest.TestCase):
    """See config.MAX_ENTRY_EXTENSION_R - real motivation (2026-08-12,
    live): confirmation delays (REQUIRE_CLOSE_CONFIRMED_BREAK +
    SIGNAL_CONFIRM_TICKS) can let price run well past the break level
    before entry fires, and execution.py always market-orders at
    whatever price exists by then with no check on how far that already
    was from the level that made the setup valid in the first place."""

    def test_extension_ratio_for_a_buy(self):
        ratio = risk_manager._entry_extension_r(
            {"structure_level": 98}, entry_price=98.8, side="BUY", risk_distance=2.0
        )
        self.assertAlmostEqual(ratio, 0.4)  # 0.8/2.0

    def test_extension_ratio_for_a_sell_is_measured_in_the_opposite_direction(self):
        ratio = risk_manager._entry_extension_r(
            {"structure_level": 102}, entry_price=100, side="SELL", risk_distance=2.0
        )
        self.assertAlmostEqual(ratio, 1.0)  # (102-100)/2.0

    def test_extension_ratio_is_none_without_a_structure_level(self):
        # Never gate/route on data we don't actually have.
        ratio = risk_manager._entry_extension_r(
            {"structure_level": None}, entry_price=500, side="BUY", risk_distance=2.0
        )
        self.assertIsNone(ratio)

    def test_extension_ratio_is_none_with_zero_risk_distance(self):
        ratio = risk_manager._entry_extension_r(
            {"structure_level": 98}, entry_price=100, side="BUY", risk_distance=0
        )
        self.assertIsNone(ratio)

    def test_within_the_cap_is_not_extended(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 0.5):
            too_extended = risk_manager._entry_too_extended(0.4)

        self.assertFalse(too_extended)  # 0.4R <= 0.5R cap

    def test_beyond_the_cap_is_too_extended(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 0.5):
            too_extended = risk_manager._entry_too_extended(1.0)

        self.assertTrue(too_extended)  # 1.0R > 0.5R cap

    def test_zero_cap_disables_the_check(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 0):
            too_extended = risk_manager._entry_too_extended(100.0)

        self.assertFalse(too_extended)

    def test_none_extension_does_not_block(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 0.5):
            too_extended = risk_manager._entry_too_extended(None)

        self.assertFalse(too_extended)

    def test_build_trade_plan_rejects_an_extended_entry(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 0.5), \
             patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0):
            plan, status = risk_manager.build_trade_plan(
                {
                    "signal": "BUY", "symbol": "BTCUSDT", "entry_price": 100,
                    "structure_level": 98, "atr": 1,
                },
                balance=1000,
            )

        self.assertIsNone(plan)
        self.assertEqual(status, "ENTRY_TOO_EXTENDED")

    def test_build_trade_plan_allows_an_entry_within_the_cap(self):
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 2.0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0), \
             patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(
                {
                    "signal": "BUY", "symbol": "BTCUSDT", "entry_price": 100,
                    "structure_level": 98, "atr": 1,
                },
                balance=1000,
            )

        self.assertEqual(status, "OK")

    def test_build_trade_plan_includes_entry_extension_r(self):
        # config.ENTRY_ROUTING_EXTENSION_THRESHOLD_R (main.py) reads this
        # off the plan to decide market vs. limit routing per-signal.
        with patch.object(config, "MAX_ENTRY_EXTENSION_R", 2.0), \
             patch.object(config, "MAX_SL_ROI_PCT", 0), \
             patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(
                {
                    "signal": "BUY", "symbol": "BTCUSDT", "entry_price": 100,
                    "structure_level": 98, "atr": 1,
                },
                balance=1000,
            )

        self.assertEqual(status, "OK")
        self.assertAlmostEqual(plan["entry_extension_r"], 1.0)  # (100-98)/2.0


class MaxSlRoiTests(unittest.TestCase):
    """See config.MAX_SL_ROI_PCT - real motivation (2026-08-14, operator
    feedback): risk-based sizing already caps the ACCOUNT-level $ loss per
    trade regardless of stop width, but the POSITION-level ROI% (PnL
    against margin used, what the exchange UI shows) is a different
    number entirely - ROI_at_SL = stop_distance_% * LEVERAGE, independent
    of position size (quantity cancels out of the ratio), and wasn't
    capped anywhere before this. Rejects outright, per explicit operator
    choice, rather than shrinking the position to fit."""

    def test_within_the_cap_is_not_too_high(self):
        # 2% stop distance * 10x leverage = 20% ROI, under a 30% cap.
        with patch.object(config, "MAX_SL_ROI_PCT", 30), \
             patch.object(config, "LEVERAGE", 10):
            too_high = risk_manager._stop_roi_too_high(risk_distance=2, entry_price=100)

        self.assertFalse(too_high)

    def test_beyond_the_cap_is_too_high(self):
        # 4% stop distance * 10x leverage = 40% ROI, over a 30% cap.
        with patch.object(config, "MAX_SL_ROI_PCT", 30), \
             patch.object(config, "LEVERAGE", 10):
            too_high = risk_manager._stop_roi_too_high(risk_distance=4, entry_price=100)

        self.assertTrue(too_high)

    def test_higher_leverage_lowers_the_stop_distance_that_trips_the_cap(self):
        # Same 2% stop distance, but 20x leverage -> 40% ROI, over the cap.
        with patch.object(config, "MAX_SL_ROI_PCT", 30), \
             patch.object(config, "LEVERAGE", 20):
            too_high = risk_manager._stop_roi_too_high(risk_distance=2, entry_price=100)

        self.assertTrue(too_high)

    def test_zero_cap_disables_the_check(self):
        with patch.object(config, "MAX_SL_ROI_PCT", 0), \
             patch.object(config, "LEVERAGE", 10):
            too_high = risk_manager._stop_roi_too_high(risk_distance=100, entry_price=100)

        self.assertFalse(too_high)

    def test_zero_entry_price_does_not_crash(self):
        with patch.object(config, "MAX_SL_ROI_PCT", 30):
            too_high = risk_manager._stop_roi_too_high(risk_distance=2, entry_price=0)

        self.assertFalse(too_high)

    def test_build_trade_plan_rejects_a_stop_with_too_high_an_roi(self):
        with patch.object(config, "MAX_SL_ROI_PCT", 30), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0):
            plan, status = risk_manager.build_trade_plan(
                {
                    # 4% stop distance * 10x leverage = 40% ROI, over the cap.
                    "signal": "BUY", "symbol": "BTCUSDT", "entry_price": 100,
                    "structure_level": 96, "atr": 0,
                },
                balance=1000,
            )

        self.assertIsNone(plan)
        self.assertEqual(status, "SL_ROI_TOO_HIGH")

    def test_build_trade_plan_allows_a_stop_within_the_roi_cap(self):
        with patch.object(config, "MAX_SL_ROI_PCT", 30), \
             patch.object(config, "MAX_ENTRY_EXTENSION_R", 0), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(config, "STRUCTURE_STOP_ATR_BUFFER", 0), \
             patch.object(config, "TP1_R_MULTIPLE", 1.0), \
             patch.object(config, "TP2_R_MULTIPLE", 2.0), \
             patch.object(risk_manager, "calculate_position_size", return_value=10.0):
            plan, status = risk_manager.build_trade_plan(
                {
                    # 2% stop distance * 10x leverage = 20% ROI, under the cap.
                    "signal": "BUY", "symbol": "BTCUSDT", "entry_price": 100,
                    "structure_level": 98, "atr": 0,
                },
                balance=1000,
            )

        self.assertEqual(status, "OK")


class ConfluenceSizeMultiplierTests(unittest.TestCase):
    def test_disabled_config_always_returns_normal_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", False):
            multiplier = risk_manager._confluence_size_multiplier({"confluence_ratio": 0.0})

        self.assertEqual(multiplier, 1.0)

    def test_none_ratio_returns_normal_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True):
            multiplier = risk_manager._confluence_size_multiplier({"confluence_ratio": None})

        self.assertEqual(multiplier, 1.0)

    def test_full_ratio_returns_the_max_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25):
            multiplier = risk_manager._confluence_size_multiplier({"confluence_ratio": 1.0})

        self.assertAlmostEqual(multiplier, 1.25)

    def test_zero_ratio_returns_the_min_multiplier(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25):
            multiplier = risk_manager._confluence_size_multiplier({"confluence_ratio": 0.0})

        self.assertAlmostEqual(multiplier, 0.5)

    def test_half_ratio_returns_the_midpoint(self):
        with patch.object(config, "CONFLUENCE_SIZING_ENABLED", True), \
             patch.object(config, "CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5), \
             patch.object(config, "CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25):
            multiplier = risk_manager._confluence_size_multiplier({"confluence_ratio": 0.5})

        self.assertAlmostEqual(multiplier, 0.875)


class ComputeDcaPriceTests(unittest.TestCase):
    """Where the single DCA fires - the next real structure level in the
    ADVERSE direction from entry, the mirror image of compute_targets'
    favorable-side pool search."""

    def setUp(self):
        # Isolate the pool-search/fallback logic from the min-distance
        # floor here - MinStopDistanceFloorTests-style coverage below
        # tests that floor directly.
        self.pct_patcher = patch.object(config, "MIN_STOP_DISTANCE_PCT", 0)
        self.pct_patcher.start()
        self.atr_patcher = patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 0)
        self.atr_patcher.start()

    def tearDown(self):
        self.pct_patcher.stop()
        self.atr_patcher.stop()

    def test_buy_targets_the_nearest_sell_side_pool_below_entry(self):
        pools = [
            {"type": "SELL_SIDE", "price": 95},
            {"type": "SELL_SIDE", "price": 90},
        ]
        price = risk_manager.compute_dca_price(100, "BUY", pools, atr=1)
        self.assertEqual(price, 95)

    def test_sell_targets_the_nearest_buy_side_pool_above_entry(self):
        pools = [
            {"type": "BUY_SIDE", "price": 105},
            {"type": "BUY_SIDE", "price": 110},
        ]
        price = risk_manager.compute_dca_price(100, "SELL", pools, atr=1)
        self.assertEqual(price, 105)

    def test_favorable_side_pool_type_is_ignored_for_a_buy(self):
        # A BUY_SIDE pool (resistance, above entry) must never become a
        # BUY's DCA level - that's the TP direction, not the adverse one.
        pools = [{"type": "BUY_SIDE", "price": 105}]
        price = risk_manager.compute_dca_price(100, "BUY", pools, atr=2)
        self.assertEqual(price, 96)  # ATR fallback: entry - 2*atr

    def test_no_pools_falls_back_to_the_atr_multiple(self):
        price = risk_manager.compute_dca_price(100, "BUY", [], atr=2)
        self.assertEqual(price, 96)  # 100 - 2*2

    def test_none_pools_falls_back_to_the_atr_multiple(self):
        price = risk_manager.compute_dca_price(100, "SELL", None, atr=1.5)
        self.assertEqual(price, 103)  # 100 + 2*1.5

    def test_pathologically_close_pool_is_widened_by_the_min_distance_floor(self):
        pools = [{"type": "SELL_SIDE", "price": 99.9}]  # far too close

        with patch.object(config, "MIN_STOP_DISTANCE_PCT", 1.0), \
             patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 0):
            price = risk_manager.compute_dca_price(100, "BUY", pools, atr=1)

        self.assertEqual(price, 99.0)  # widened to the 1% floor, not 99.9


class ComputeDcaSlPriceTests(unittest.TestCase):
    """The first real SL a DCA-pending position ever gets, anchored to
    the next real structure level BEYOND the DCA fill itself."""

    def setUp(self):
        self.pct_patcher = patch.object(config, "MIN_STOP_DISTANCE_PCT", 0)
        self.pct_patcher.start()
        self.atr_patcher = patch.object(config, "MIN_STOP_DISTANCE_ATR_MULTIPLE", 0)
        self.atr_patcher.start()

    def tearDown(self):
        self.pct_patcher.stop()
        self.atr_patcher.stop()

    def test_buy_places_beyond_the_next_structure_level_with_atr_buffer(self):
        pools = [{"type": "SELL_SIDE", "price": 90}]

        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_dca_sl_price(95, "BUY", pools, atr=2)

        self.assertEqual(sl, 89.0)  # 90 - (2 * 0.5)

    def test_sell_places_beyond_the_next_structure_level_with_atr_buffer(self):
        pools = [{"type": "BUY_SIDE", "price": 110}]

        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_dca_sl_price(105, "SELL", pools, atr=2)

        self.assertEqual(sl, 111.0)  # 110 + (2 * 0.5)

    def test_no_further_pool_falls_back_to_the_atr_multiple(self):
        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 0):
            sl = risk_manager.compute_dca_sl_price(95, "BUY", [], atr=2)

        self.assertEqual(sl, 91.0)  # 95 - 2*2 (fallback), no buffer

    def test_buffer_atr_multiple_override_replaces_the_config_default(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - a not-confirmed DCA fire
        # passes a tighter buffer_atr_multiple explicitly; it must win
        # over config.DCA_STRUCTURE_STOP_ATR_BUFFER, not add to it.
        pools = [{"type": "SELL_SIDE", "price": 90}]

        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_dca_sl_price(95, "BUY", pools, atr=2, buffer_atr_multiple=0.25)

        self.assertEqual(sl, 89.5)  # 90 - (2 * 0.25), not the config's 0.5

    def test_buffer_atr_multiple_none_falls_back_to_config_default(self):
        pools = [{"type": "SELL_SIDE", "price": 90}]

        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 0.5):
            sl = risk_manager.compute_dca_sl_price(95, "BUY", pools, atr=2, buffer_atr_multiple=None)

        self.assertEqual(sl, 89.0)  # unchanged from the plain default-buffer test above


class ComputeDcaTargetTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_uses_dca_r_multiples_not_tp2s(self):
        with patch.object(config, "DCA_TP_R_MULTIPLE", 3.0), \
             patch.object(config, "DCA_TP_MAX_R_MULTIPLE", 6.0), \
             patch.object(config, "TP2_R_MULTIPLE", 999.0):
            target = risk_manager.compute_dca_target(100, 95, "BUY", pools=None)

        self.assertEqual(target, 115)  # 100 + 3 * (100-95)

    def test_zero_risk_distance_returns_none(self):
        target = risk_manager.compute_dca_target(100, 100, "BUY", pools=None)
        self.assertIsNone(target)

    def test_real_pool_within_bounds_is_used(self):
        pools = [{"type": "BUY_SIDE", "price": 112}]  # 2.4R on a 5-wide risk

        with patch.object(config, "DCA_TP_R_MULTIPLE", 1.0), \
             patch.object(config, "DCA_TP_MAX_R_MULTIPLE", 6.0):
            target = risk_manager.compute_dca_target(100, 95, "BUY", pools=pools)

        self.assertEqual(target, 112)

    def test_static_roi_mode_ignores_pools_and_sl_price_entirely(self):
        # A pool sits right at the entry price itself (would normally be
        # rejected by _find_structure_target's own min-room floor anyway)
        # - static ROI mode must not even look at it, or at sl_price.
        pools = [{"type": "BUY_SIDE", "price": 100.01}]

        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            target = risk_manager.compute_dca_target(100, sl_price=None, side="BUY", pools=pools)

        # 50% ROI at 10x leverage -> 5% price move -> 105.
        self.assertAlmostEqual(target, 105.0)

    def test_static_roi_mode_mirrors_for_sell(self):
        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            target = risk_manager.compute_dca_target(100, sl_price=None, side="SELL", pools=None)

        self.assertAlmostEqual(target, 95.0)


class ComputeRetracementPriceTests(unittest.TestCase):
    """config.RETRACEMENT_ENTRY_ENABLED - a small pullback toward the stop
    from the planned entry price, in RETRACEMENT_ENTRY_OFFSET_R units of
    the planned risk distance."""

    def test_buy_rests_below_entry_toward_the_stop(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1):
            price = risk_manager.compute_retracement_price(100, 90, "BUY")

        self.assertEqual(price, 99.0)  # 100 - 0.1 * (100 - 90)

    def test_sell_rests_above_entry_toward_the_stop(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1):
            price = risk_manager.compute_retracement_price(100, 110, "SELL")

        self.assertEqual(price, 101.0)  # 100 + 0.1 * (110 - 100)

    def test_zero_offset_rests_exactly_at_the_trigger_price(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0):
            price = risk_manager.compute_retracement_price(100, 90, "BUY")

        self.assertEqual(price, 100.0)

    def test_larger_offset_rests_further_toward_the_stop(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.5):
            price = risk_manager.compute_retracement_price(100, 90, "BUY")

        self.assertEqual(price, 95.0)  # halfway to the stop

    def test_negative_offset_is_clamped_to_zero(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", -0.5):
            price = risk_manager.compute_retracement_price(100, 90, "BUY")

        self.assertEqual(price, 100.0)

    def test_structure_target_disabled_ignores_fvgs_and_pools(self):
        # entry=100, sl=90 -> fallback = 100 - 0.1*10 = 99. A qualifying
        # real level at 97 exists but must be ignored while the flag is off.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", False):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 97, "bottom": 96}],
            )

        self.assertEqual(price, 99.0)

    def test_structure_target_prefers_a_real_level_within_the_cap(self):
        # 97 is 0.3R from entry (100-97=3, risk=10) - within the 0.35 cap.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 97, "bottom": 96}],
            )

        self.assertEqual(price, 97.0)

    def test_structure_target_sell_mirrors_the_buy_case(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 110, "SELL", fvgs=[{"top": 104, "bottom": 103}],
            )

        self.assertEqual(price, 103.0)

    def test_structure_target_falls_back_when_the_level_is_deeper_than_the_cap(self):
        # 80 is 2.0R from entry - far beyond a sane 0.35R cap, and even
        # past the stop itself; must fall back to the fixed-R calculation.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 80, "bottom": 79}],
            )

        self.assertEqual(price, 99.0)

    def test_structure_target_falls_back_when_no_level_is_in_the_risk_window(self):
        # 101 sits above entry, not between entry and the stop - not a
        # valid pullback target for a BUY at all.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 101, "bottom": 100.5}],
            )

        self.assertEqual(price, 99.0)

    def test_structure_target_falls_back_when_no_fvgs_or_pools_supplied(self):
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True):
            price = risk_manager.compute_retracement_price(100, 90, "BUY")

        self.assertEqual(price, 99.0)

    def test_structure_target_considers_liquidity_pools_too(self):
        # 97 is 0.3R from entry (100-97=3, risk=10) - within the 0.35 cap.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", pools=[{"type": "SELL_SIDE", "price": 97, "touches": 2}],
            )

        self.assertEqual(price, 97.0)

    def test_structure_target_picks_the_level_nearest_entry_among_several(self):
        # Both candidates qualify (0.3R and 0.15R, both under the 0.35 cap) -
        # the nearer one (98.5) must win over the farther one (97).
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY",
                fvgs=[{"top": 97, "bottom": 96}],
                pools=[{"type": "SELL_SIDE", "price": 98.5, "touches": 2}],
            )

        self.assertEqual(price, 98.5)

    def test_prefer_deeper_picks_the_farther_qualifying_level_for_buy(self):
        # Both candidates qualify (0.3R and 0.15R, both under the 0.35 cap) -
        # prefer_deeper must pick the FARTHER one (97), the opposite of the
        # default (nearest-entry) selection above.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY",
                fvgs=[{"top": 97, "bottom": 96}],
                pools=[{"type": "SELL_SIDE", "price": 98.5, "touches": 2}],
                prefer_deeper=True,
            )

        self.assertEqual(price, 97.0)

    def test_prefer_deeper_picks_the_farther_qualifying_level_for_sell(self):
        # Both candidates qualify (0.3R and 0.15R, both under the 0.35 cap,
        # risk=10) - prefer_deeper must pick the farther one (103).
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 110, "SELL",
                fvgs=[{"top": 103, "bottom": 102}],
                pools=[{"type": "BUY_SIDE", "price": 101.5, "touches": 2}],
                prefer_deeper=True,
            )

        self.assertEqual(price, 103.0)

    def test_prefer_deeper_still_falls_back_when_nothing_qualifies(self):
        # Same fixture as test_structure_target_falls_back_when_no_level_is_
        # in_the_risk_window - prefer_deeper must never invent a new
        # fallback distance, only change which real candidate wins.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 101, "bottom": 100.5}],
                prefer_deeper=True,
            )

        self.assertEqual(price, 99.0)

    def test_prefer_deeper_still_respects_the_max_r_cap(self):
        # 80 is 2.0R from entry - far beyond the 0.35R cap. Being "deeper"
        # doesn't exempt a candidate from the same real-structure cap every
        # other selection already respects.
        with patch.object(config, "RETRACEMENT_ENTRY_OFFSET_R", 0.1), \
             patch.object(config, "RETRACEMENT_STRUCTURE_TARGET_ENABLED", True), \
             patch.object(config, "RETRACEMENT_STRUCTURE_MAX_R", 0.35):
            price = risk_manager.compute_retracement_price(
                100, 90, "BUY", fvgs=[{"top": 80, "bottom": 79}],
                prefer_deeper=True,
            )

        self.assertEqual(price, 99.0)


class PriceAtRoiPctTests(unittest.TestCase):
    def test_buy_target_scales_with_roi_and_leverage(self):
        with patch.object(config, "LEVERAGE", 10):
            price = risk_manager.price_at_roi_pct(100, "BUY", 50)

        self.assertAlmostEqual(price, 105.0)  # 50% ROI / 10x leverage = 5% price move

    def test_sell_target_mirrors_buy(self):
        with patch.object(config, "LEVERAGE", 10):
            price = risk_manager.price_at_roi_pct(100, "SELL", 50)

        self.assertAlmostEqual(price, 95.0)

    def test_higher_leverage_needs_a_smaller_price_move_for_the_same_roi(self):
        with patch.object(config, "LEVERAGE", 20):
            price = risk_manager.price_at_roi_pct(100, "BUY", 50)

        self.assertAlmostEqual(price, 102.5)  # 50% / 20x = 2.5% price move

    def test_zero_roi_returns_entry_price_unchanged(self):
        with patch.object(config, "LEVERAGE", 10):
            price = risk_manager.price_at_roi_pct(100, "BUY", 0)

        self.assertAlmostEqual(price, 100.0)

    def test_negative_roi_is_clamped_to_zero(self):
        with patch.object(config, "LEVERAGE", 10):
            price = risk_manager.price_at_roi_pct(100, "BUY", -20)

        self.assertAlmostEqual(price, 100.0)

    def test_zero_entry_price_returns_none(self):
        price = risk_manager.price_at_roi_pct(0, "BUY", 50)
        self.assertIsNone(price)

    def test_zero_leverage_returns_none(self):
        with patch.object(config, "LEVERAGE", 0):
            price = risk_manager.price_at_roi_pct(100, "BUY", 50)

        self.assertIsNone(price)


class BuildDcaPlanTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("MIN_STOP_DISTANCE_PCT", 0), ("MIN_STOP_DISTANCE_ATR_MULTIPLE", 0),
            ("DCA_STRUCTURE_STOP_ATR_BUFFER", 0),
            ("DCA_TP_R_MULTIPLE", 1.0), ("DCA_TP_MAX_R_MULTIPLE", 10.0),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_blends_entry_price_by_quantity_weighted_average(self):
        # Original: 1.0 @ 100. DCA: 1.0 @ 90 (equal size) -> blended 95,
        # exactly halfway - the classic equal-size DCA outcome.
        plan = risk_manager.build_dca_plan(
            original_entry_price=100, original_quantity=1.0,
            dca_fill_price=90, dca_quantity=1.0, side="BUY", pools=[], atr=1,
        )
        self.assertAlmostEqual(plan["entry_price"], 95.0)
        self.assertAlmostEqual(plan["quantity"], 2.0)

    def test_unequal_dca_size_weights_the_blend_toward_the_dca_fill(self):
        # Original: 1.0 @ 100. DCA: 2.0 @ 90 -> weighted average, closer
        # to the DCA fill than the midpoint.
        plan = risk_manager.build_dca_plan(
            original_entry_price=100, original_quantity=1.0,
            dca_fill_price=90, dca_quantity=2.0, side="BUY", pools=[], atr=1,
        )
        self.assertAlmostEqual(plan["entry_price"], 100 * 1 / 3 + 90 * 2 / 3)
        self.assertAlmostEqual(plan["quantity"], 3.0)

    def test_returns_none_when_total_quantity_is_zero(self):
        plan = risk_manager.build_dca_plan(
            original_entry_price=100, original_quantity=0,
            dca_fill_price=90, dca_quantity=0, side="BUY", pools=[], atr=1,
        )
        self.assertIsNone(plan)

    def test_full_shape_includes_sl_tp_and_risk_distance(self):
        plan = risk_manager.build_dca_plan(
            original_entry_price=100, original_quantity=1.0,
            dca_fill_price=90, dca_quantity=1.0, side="BUY", pools=[], atr=1,
        )
        self.assertIn("sl_price", plan)
        self.assertIn("tp_price", plan)
        self.assertLess(plan["sl_price"], plan["entry_price"])
        self.assertGreater(plan["tp_price"], plan["entry_price"])
        self.assertAlmostEqual(
            plan["risk_distance"], abs(plan["entry_price"] - plan["sl_price"])
        )

    def test_static_roi_mode_still_places_a_real_structure_sl(self):
        # config.DCA_TP_STATIC_ROI_ENABLED only changes the TP - the first
        # real SL this position ever gets is still structure-anchored,
        # same as always.
        with patch.object(config, "DCA_TP_STATIC_ROI_ENABLED", True), \
             patch.object(config, "DCA_TP_TARGET_ROI_PCT", 50), \
             patch.object(config, "LEVERAGE", 10):
            plan = risk_manager.build_dca_plan(
                original_entry_price=100, original_quantity=1.0,
                dca_fill_price=90, dca_quantity=1.0, side="BUY", pools=[], atr=1,
            )

        self.assertAlmostEqual(plan["entry_price"], 95.0)
        self.assertLess(plan["sl_price"], plan["entry_price"])  # unaffected, still structure-based
        self.assertAlmostEqual(plan["tp_price"], 99.75)  # 95 * (1 + 0.5/10)

    def test_buffer_atr_multiple_override_passes_through_to_the_sl(self):
        # config.DCA_PRESSURE_CHECK_ENABLED - position_manager._execute_dca
        # passes a tighter buffer_atr_multiple for a not-confirmed fire;
        # build_dca_plan must forward it to compute_dca_sl_price rather
        # than always using config.DCA_STRUCTURE_STOP_ATR_BUFFER.
        with patch.object(config, "DCA_STRUCTURE_STOP_ATR_BUFFER", 2.0):
            wide_plan = risk_manager.build_dca_plan(
                original_entry_price=100, original_quantity=1.0,
                dca_fill_price=90, dca_quantity=1.0, side="BUY", pools=[], atr=1,
            )
            tight_plan = risk_manager.build_dca_plan(
                original_entry_price=100, original_quantity=1.0,
                dca_fill_price=90, dca_quantity=1.0, side="BUY", pools=[], atr=1,
                buffer_atr_multiple=0.25,
            )

        self.assertLess(wide_plan["sl_price"], tight_plan["sl_price"])
        self.assertGreater(wide_plan["risk_distance"], tight_plan["risk_distance"])


class BuildTradePlanDcaFieldsTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("MAX_ENTRY_EXTENSION_R", 0), ("MAX_SL_ROI_PCT", 0),
            ("STRUCTURE_STOP_ATR_BUFFER", 0), ("TP1_R_MULTIPLE", 1.0),
            ("TP2_R_MULTIPLE", 2.0), ("TP1_CLOSE_PCT", 50),
            ("MIN_STOP_DISTANCE_PCT", 0), ("MIN_STOP_DISTANCE_ATR_MULTIPLE", 0),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(risk_manager, "calculate_position_size", return_value=10.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _signal(self, side="BUY", entry_price=100, structure_level=98, atr=1):
        return {
            "signal": side, "symbol": "BTCUSDT", "entry_price": entry_price,
            "structure_level": structure_level, "atr": atr, "liquidity_pools": [],
        }

    def test_dca_price_is_computed_when_enabled(self):
        with patch.object(config, "DCA_ENABLED", True):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertIsNotNone(plan["dca_price"])
        self.assertLess(plan["dca_price"], plan["entry_price"])  # adverse (below) for a BUY

    def test_dca_price_is_none_when_disabled(self):
        with patch.object(config, "DCA_ENABLED", False):
            plan, status = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertEqual(status, "OK")
        self.assertIsNone(plan["dca_price"])

    def test_dca_quantity_uses_the_size_multiplier(self):
        with patch.object(config, "DCA_ENABLED", True), \
             patch.object(config, "DCA_SIZE_MULTIPLIER", 1.5):
            plan, _ = risk_manager.build_trade_plan(self._signal(), balance=1000)

        self.assertAlmostEqual(plan["dca_quantity"], 15.0)  # 10.0 * 1.5

    def test_atr_is_carried_through_to_the_plan(self):
        plan, _ = risk_manager.build_trade_plan(self._signal(atr=3.5), balance=1000)
        self.assertEqual(plan["atr"], 3.5)

    def test_fair_value_gaps_and_liquidity_pools_are_carried_through_to_the_plan(self):
        # config.RETRACEMENT_STRUCTURE_TARGET_ENABLED - execution.
        # enter_trade_retracement needs these on the plan to consider a
        # real structural level instead of only a synthetic R-fraction.
        signal = self._signal()
        signal["fair_value_gaps"] = [{"type": "BULLISH", "top": 99, "bottom": 97}]
        signal["liquidity_pools"] = [{"type": "SELL_SIDE", "price": 96, "touches": 2}]

        plan, _ = risk_manager.build_trade_plan(signal, balance=1000)

        self.assertEqual(plan["fair_value_gaps"], signal["fair_value_gaps"])
        self.assertEqual(plan["liquidity_pools"], signal["liquidity_pools"])


if __name__ == "__main__":
    unittest.main()
