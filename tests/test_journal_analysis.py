import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import signal_journal
import journal_analysis as ja


def _write_trade(path, trade_id, symbol, outcome=None, mae_r_multiple=None, mfe_r_multiple=None, early_breakeven_applied=None, break_confirmed_by_close=None, closed_at=None, **signal_fields):
    """Writes a signal row (and optionally its outcome row) directly via
    the real signal_journal writer, so these tests exercise the exact
    on-disk format the analysis code has to parse. `closed_at` (unix
    seconds) controls the outcome row's timestamp, for testing the
    since_timestamp filter."""
    fields = {
        "symbol": symbol,
        "signal": "BUY",
        "htf_trend": "BULLISH",
        "structure_level": 98.0,
        "atr": 1.0,
        "premium_discount_zone": "DISCOUNT",
        "order_block": None,
        "fvg": None,
        "cvd_score": 0.5,
        "depth_imbalance": 0.1,
        "sweep_confluence": False,
    }
    fields.update(signal_fields)
    plan = {
        "entry_price": 100.0, "sl_price": 98.0, "tp1_price": 102.0,
        "tp2_price": 104.0, "quantity": 1.0, "risk_distance": 2.0,
    }

    with patch.object(signal_journal, "JOURNAL_PATH", path), \
         patch.object(signal_journal, "_make_trade_id", return_value=trade_id):
        signal_journal.append_signal(fields, plan)

        if outcome:
            if closed_at is not None:
                with patch.object(signal_journal.time, "time", return_value=closed_at):
                    signal_journal.append_outcome(
                        symbol, outcome, trade_id,
                        mae_r_multiple=mae_r_multiple, mfe_r_multiple=mfe_r_multiple,
                        early_breakeven_applied=early_breakeven_applied,
                        break_confirmed_by_close=break_confirmed_by_close,
                    )
            else:
                signal_journal.append_outcome(
                    symbol, outcome, trade_id,
                    mae_r_multiple=mae_r_multiple, mfe_r_multiple=mfe_r_multiple,
                    early_breakeven_applied=early_breakeven_applied,
                    break_confirmed_by_close=break_confirmed_by_close,
                )


def _write_outcome_only(path, trade_id, symbol, outcome, mae_r_multiple=None, mfe_r_multiple=None):
    """Simulates a trade with no matching signal row at all - exactly
    what a startup-reconciliation-adopted position looks like on disk
    (real bug found live, 2026-08-09: these showed "unknown" for every
    field, including symbol, before load_trades() was fixed to merge
    outcome-row fields too)."""
    with patch.object(signal_journal, "JOURNAL_PATH", path):
        signal_journal.append_outcome(
            symbol, outcome, trade_id,
            mae_r_multiple=mae_r_multiple, mfe_r_multiple=mfe_r_multiple,
        )


def _write_csv(path, fieldnames, rows):
    """Raw CSV writer (bypasses signal_journal entirely) - used only by
    the cross-file-rotation tests below, which need files carrying
    DIFFERENT, deliberately-old-shaped headers (exactly what a real
    signal_journal.bak_*.csv rotation leaves behind), not the current
    live schema _write_trade above always writes."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ClassifyTests(unittest.TestCase):
    def test_sl_hit_variants_are_loss(self):
        self.assertEqual(ja.classify("SL_HIT"), "LOSS")
        self.assertEqual(ja.classify("SHADOW_SL_HIT"), "LOSS")

    def test_breakeven_variants_are_breakeven(self):
        self.assertEqual(ja.classify("BREAKEVEN_STOP_HIT"), "BREAKEVEN")
        self.assertEqual(ja.classify("BREAKEVEN_TRIGGER_MARKET_CLOSE"), "BREAKEVEN")

    def test_tp2_variants_are_win(self):
        self.assertEqual(ja.classify("TP2_HIT"), "WIN")
        self.assertEqual(ja.classify("SHADOW_TP2_HIT"), "WIN")

    def test_early_breakeven_profit_hit_is_win_not_breakeven(self):
        # A real locked-profit stop hit (EARLY_BREAKEVEN_LOCK_R_MULTIPLE >
        # 0) is a genuine small win, not a zero-sum scratch - distinct from
        # BREAKEVEN_STOP_HIT above.
        self.assertEqual(ja.classify("EARLY_BREAKEVEN_PROFIT_HIT"), "WIN")
        self.assertEqual(ja.classify("SHADOW_EARLY_BREAKEVEN_PROFIT_HIT"), "WIN")

    def test_trailing_stop_profit_hit_is_win(self):
        # config.STRUCTURE_STOP_MANAGEMENT_ENABLED - a real profit lock
        # from the post-TP1 structure-based trailing stop.
        self.assertEqual(ja.classify("TRAILING_STOP_PROFIT_HIT"), "WIN")
        self.assertEqual(ja.classify("SHADOW_TRAILING_STOP_PROFIT_HIT"), "WIN")

    def test_profit_protection_hit_is_win(self):
        # config.PROFIT_PROTECTION_ENABLED - a real profit lock from the
        # profit-protection trailing floor, same category as
        # TRAILING_STOP_PROFIT_HIT above (a different trailing mechanism
        # reaching the same outcome). Found missing 2026-08-17 via a real-
        # log cross-check: 5 PROFIT_PROTECTION_HIT closes in one 40h
        # window were silently falling to UNKNOWN instead of WIN.
        self.assertEqual(ja.classify("PROFIT_PROTECTION_HIT"), "WIN")
        self.assertEqual(ja.classify("SHADOW_PROFIT_PROTECTION_HIT"), "WIN")

    def test_dca_breakeven_trail_hit_is_win(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a native trailing
        # stop only ever activates at or past breakeven and only tightens
        # further, so a close via this order is at-or-better-than
        # breakeven by construction, same category as TRAILING_STOP_
        # PROFIT_HIT/PROFIT_PROTECTION_HIT above. Live-only (shadow never
        # sets dca_trail_order_id), so there is no SHADOW_ variant.
        self.assertEqual(ja.classify("DCA_BREAKEVEN_TRAIL_HIT"), "WIN")

    def test_unknown_outcome_is_unknown(self):
        self.assertEqual(ja.classify("SOMETHING_ELSE"), "UNKNOWN")

    def test_limit_fill_sl_placement_failure_is_loss(self):
        # config.LIMIT_ENTRY_MODE_ENABLED - a resting limit that filled,
        # then had to be emergency-closed when SL placement failed. Real
        # (if degraded) exposure existed, so it's counted conservatively
        # as a loss, same treatment as TP1_THEN_POSITION_ALREADY_CLOSED.
        self.assertEqual(ja.classify("LIMIT_FILL_SL_PLACEMENT_FAILED"), "LOSS")

    def test_unfilled_limit_outcomes_are_unknown_not_win_loss_or_breakeven(self):
        # A limit that never filled has zero real P&L - must not pollute
        # WIN/LOSS/BREAKEVEN stats, even though it's a resolved outcome.
        self.assertEqual(ja.classify("LIMIT_EXPIRED_UNFILLED"), "UNKNOWN")
        self.assertEqual(ja.classify("LIMIT_INVALIDATED_UNFILLED"), "UNKNOWN")


class BucketVolumeTests(unittest.TestCase):
    def test_below_the_liquidity_floor(self):
        self.assertEqual(ja._bucket_volume(2_999_999), "<3M (below the liquidity floor)")

    def test_boundaries_are_inclusive_on_the_low_end(self):
        self.assertEqual(ja._bucket_volume(3_000_000), "3M-10M")
        self.assertEqual(ja._bucket_volume(10_000_000), "10M-50M")
        self.assertEqual(ja._bucket_volume(50_000_000), ">=50M")

    def test_non_numeric_is_unknown(self):
        self.assertEqual(ja._bucket_volume(None), "unknown")
        self.assertEqual(ja._bucket_volume(""), "unknown")


class NewDataSourceBucketTests(unittest.TestCase):
    def test_efficiency_ratio_buckets(self):
        self.assertEqual(ja._bucket_efficiency(0.1), "choppy (<0.3)")
        self.assertEqual(ja._bucket_efficiency(0.4), "moderate (0.3-0.6)")
        self.assertEqual(ja._bucket_efficiency(0.9), "trending (>=0.6)")
        self.assertEqual(ja._bucket_efficiency(None), "unknown")

    def test_correlation_buckets_use_absolute_value(self):
        self.assertEqual(ja._bucket_correlation(-0.9), "strong (>=0.6)")
        self.assertEqual(ja._bucket_correlation(0.1), "weak (<0.3)")
        self.assertEqual(ja._bucket_correlation(None), "unknown")

    def test_funding_rate_buckets(self):
        self.assertEqual(ja._bucket_funding_rate(-0.001), "<-0.05% (crowded short)")
        self.assertEqual(ja._bucket_funding_rate(0.0), "-0.05% to 0.05% (neutral)")
        self.assertEqual(ja._bucket_funding_rate(0.001), ">0.05% (crowded long)")
        self.assertEqual(ja._bucket_funding_rate(None), "unknown")

    def test_long_short_ratio_buckets(self):
        self.assertEqual(ja._bucket_long_short_ratio(0.5), "<0.8 (short-heavy)")
        self.assertEqual(ja._bucket_long_short_ratio(1.0), "0.8-1.2 (balanced)")
        self.assertEqual(ja._bucket_long_short_ratio(1.5), ">1.2 (long-heavy)")
        self.assertEqual(ja._bucket_long_short_ratio(None), "unknown")

    def test_extension_r_buckets(self):
        self.assertEqual(ja._bucket_extension_r(-0.1), "<0R (entry at/before the level)")
        self.assertEqual(ja._bucket_extension_r(0.1), "0-0.2R (tight)")
        self.assertEqual(
            ja._bucket_extension_r(0.35),
            "0.2-0.5R (extended - limit-routed if LIMIT_ENTRY_MODE_ENABLED)",
        )
        self.assertEqual(
            ja._bucket_extension_r(0.7),
            ">=0.5R (should be rare - normally rejected outright)",
        )
        self.assertEqual(ja._bucket_extension_r(None), "unknown")

    def test_zone_retracement_buckets(self):
        self.assertEqual(ja._bucket_zone_retracement(0.65), "<0.705 (shallow - pre-tightening band)")
        self.assertEqual(ja._bucket_zone_retracement(0.72), "0.705-0.75")
        self.assertEqual(ja._bucket_zone_retracement(0.77), "0.75-0.79")
        self.assertEqual(ja._bucket_zone_retracement(0.85), ">=0.79 (deep)")
        self.assertEqual(ja._bucket_zone_retracement(None), "unknown")

    def test_setup_age_buckets(self):
        self.assertEqual(
            ja._bucket_setup_age(0), "0 (fresh - STRUCTURE_BREAK/EMA_PULLBACK)"
        )
        self.assertEqual(ja._bucket_setup_age(2), "1-3 candles old")
        self.assertEqual(ja._bucket_setup_age(6), "4-8 candles old")
        self.assertEqual(ja._bucket_setup_age(12), ">8 candles old")
        self.assertEqual(
            ja._bucket_setup_age(None),
            "unknown (no formation index available for this trigger)",
        )

    def test_nearest_sr_r_buckets(self):
        self.assertEqual(
            ja._bucket_nearest_sr_r(0.3), "<0.5R (very close - likely caps the move early)"
        )
        self.assertEqual(ja._bucket_nearest_sr_r(0.7), "0.5-1R (inside TP1's own floor)")
        self.assertEqual(ja._bucket_nearest_sr_r(1.5), "1-2R")
        self.assertEqual(ja._bucket_nearest_sr_r(2.5), ">=2R (clear past TP1)")
        self.assertEqual(
            ja._bucket_nearest_sr_r(None), "unknown (no real pool in that direction)"
        )


class LoadTradesAndSummarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.tmpdir.name) / "journal.csv"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_journal_returns_empty_dict(self):
        self.assertEqual(ja.load_trades(self.journal_path), {})

    def test_signal_and_outcome_rows_merge_by_trade_id(self):
        _write_trade(self.journal_path, "T1", "BTCUSDT", outcome="SL_HIT", cvd_score=0.8)

        trades = ja.load_trades(self.journal_path)

        self.assertIn("T1", trades)
        self.assertEqual(trades["T1"]["outcome"], "SL_HIT")
        self.assertEqual(trades["T1"]["symbol"], "BTCUSDT")
        self.assertEqual(trades["T1"]["cvd_score"], "0.8")

    def test_unresolved_trade_has_no_outcome_key_populated(self):
        _write_trade(self.journal_path, "T2", "ETHUSDT")
        trades = ja.load_trades(self.journal_path)
        self.assertNotIn("outcome", trades.get("T2", {}))

    def test_summarize_with_no_resolved_trades(self):
        _write_trade(self.journal_path, "T3", "ETHUSDT")
        report = ja.summarize(self.journal_path)
        self.assertIn("No resolved", report)

    def test_summarize_counts_wins_and_losses_correctly(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", cvd_score=0.9, sweep_confluence=True)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", cvd_score=0.1, sweep_confluence=False)
        _write_trade(self.journal_path, "C", "BTCUSDT", outcome="SL_HIT", cvd_score=0.15, sweep_confluence=False)

        report = ja.summarize(self.journal_path)

        self.assertIn("Resolved trades: 3", report)
        self.assertIn("WIN=1", report)
        self.assertIn("LOSS=2", report)
        # The weak-CVD bucket should show the 2/2 loss rate concentration.
        self.assertIn("weak (<0.3): n=2", report)

    def test_summarize_breaks_down_by_side(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", signal="BUY")
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", signal="BUY")
        _write_trade(self.journal_path, "C", "SOLUSDT", outcome="SL_HIT", signal="SELL")

        report = ja.summarize(self.journal_path)

        self.assertIn("By side (BUY/SELL):", report)
        self.assertIn("BUY: n=2", report)
        self.assertIn("SELL: n=1", report)

    def test_summarize_breaks_down_by_zone_retracement_depth(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", zone_retracement_pct=0.85)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", zone_retracement_pct=0.65)

        report = ja.summarize(self.journal_path)

        self.assertIn("By zone retracement depth:", report)
        self.assertIn(">=0.79 (deep): n=1", report)
        self.assertIn("<0.705 (shallow - pre-tightening band): n=1", report)

    def test_summarize_breaks_down_by_early_breakeven_applied(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SHADOW_BREAKEVEN_STOP_HIT", early_breakeven_applied=True)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SHADOW_TP2_HIT", early_breakeven_applied=True)
        _write_trade(self.journal_path, "C", "SOLUSDT", outcome="SHADOW_SL_HIT", early_breakeven_applied=False)

        report = ja.summarize(self.journal_path)

        self.assertIn("By early breakeven applied (new 1R profit-lock trigger):", report)
        self.assertIn("True: n=2 WIN=1 BREAKEVEN=1 LOSS=0 loss_rate=0%", report)
        self.assertIn("False: n=1 WIN=0 BREAKEVEN=0 LOSS=1 loss_rate=100%", report)
        self.assertIn("loss_rate=100%", report)

    def test_summarize_breaks_down_by_break_confirmed_by_close(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", break_confirmed_by_close=True)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", break_confirmed_by_close=False)
        _write_trade(self.journal_path, "C", "SOLUSDT", outcome="SL_HIT", break_confirmed_by_close=False)

        report = ja.summarize(self.journal_path)

        self.assertIn("By break confirmed by candle close (wick vs real break):", report)
        self.assertIn("False: n=2 WIN=0 BREAKEVEN=0 LOSS=2 loss_rate=100%", report)
        self.assertIn("True: n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)

    def test_summarize_breaks_down_by_24h_quote_volume(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", quote_volume_usdt=100_000_000)
        _write_trade(self.journal_path, "B", "SHITCOINUSDT", outcome="SL_HIT", quote_volume_usdt=500_000)

        report = ja.summarize(self.journal_path)

        self.assertIn("By 24h quote volume (liquidity floor):", report)
        self.assertIn(">=50M: n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)
        self.assertIn(
            "<3M (below the liquidity floor): n=1 WIN=0 BREAKEVEN=0 LOSS=1 loss_rate=100%",
            report,
        )

    def test_summarize_breaks_down_by_the_four_new_data_sources(self):
        _write_trade(
            self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT",
            efficiency_ratio=0.8, btc_correlation=0.7, btc_aligned=True,
            funding_rate=0.001, long_short_ratio=1.5,
        )
        _write_trade(
            self.journal_path, "B", "ETHUSDT", outcome="SL_HIT",
            efficiency_ratio=0.1, btc_correlation=0.1, btc_aligned=False,
            funding_rate=-0.001, long_short_ratio=0.5,
        )

        report = ja.summarize(self.journal_path)

        self.assertIn("By efficiency ratio (chop vs trend):", report)
        self.assertIn("trending (>=0.6): n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)
        self.assertIn("choppy (<0.3): n=1 WIN=0 BREAKEVEN=0 LOSS=1 loss_rate=100%", report)

        self.assertIn("By BTC correlation strength:", report)
        self.assertIn("By BTC aligned (informational):", report)
        self.assertIn("True: n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)

        self.assertIn("By funding rate:", report)
        self.assertIn(">0.05% (crowded long): n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)

        self.assertIn("By long/short account ratio:", report)
        self.assertIn(">1.2 (long-heavy): n=1 WIN=1 BREAKEVEN=0 LOSS=0 loss_rate=0%", report)

    def test_outcome_only_trade_still_gets_its_symbol_from_the_outcome_row(self):
        # Real bug found live: a startup-reconciliation-adopted position
        # (no matching signal row) used to show "unknown" for every
        # field, including symbol, even though the outcome row had it.
        _write_outcome_only(self.journal_path, "SYM_RECOVERED_1", "BTCUSDT", "SL_HIT")

        trades = ja.load_trades(self.journal_path)

        self.assertEqual(trades["SYM_RECOVERED_1"]["symbol"], "BTCUSDT")
        self.assertEqual(trades["SYM_RECOVERED_1"]["outcome"], "SL_HIT")

    def test_mae_mfe_are_merged_from_the_outcome_row(self):
        _write_trade(self.journal_path, "T4", "BTCUSDT", outcome="SL_HIT", mae_r_multiple=1.5, mfe_r_multiple=0.5)

        trades = ja.load_trades(self.journal_path)

        self.assertEqual(trades["T4"]["mae_r_multiple"], "1.5")
        self.assertEqual(trades["T4"]["mfe_r_multiple"], "0.5")

    def test_signal_time_fields_are_not_clobbered_by_the_blank_outcome_row(self):
        _write_trade(self.journal_path, "T5", "BTCUSDT", outcome="TP2_HIT", cvd_score=0.77)

        trades = ja.load_trades(self.journal_path)

        # The outcome row's cvd_score field is blank - must not overwrite
        # the real value the signal row already wrote.
        self.assertEqual(trades["T5"]["cvd_score"], "0.77")

    def test_summarize_includes_average_mae_mfe_by_outcome(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mae_r_multiple=1.5, mfe_r_multiple=0.5)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", mae_r_multiple=1.0, mfe_r_multiple=0.1)
        _write_trade(self.journal_path, "C", "BTCUSDT", outcome="TP2_HIT", mae_r_multiple=0.2, mfe_r_multiple=4.0)

        report = ja.summarize(self.journal_path)

        self.assertIn("Average MAE (adverse excursion) by outcome:", report)
        self.assertIn("LOSS: avg=1.250R n=2", report)  # (1.5+1.0)/2
        self.assertIn("Average MFE (favorable excursion) by outcome:", report)
        self.assertIn("WIN: avg=4.000R n=1", report)

    def test_implausible_outliers_are_excluded_not_silently_included(self):
        # Real bug found live: a since-fixed position_manager.py bug
        # could produce R-multiples in the billions, permanently sitting
        # in the journal - a plain average has no defense against that.
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mae_r_multiple=1.5, mfe_r_multiple=0.5)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="BREAKEVEN_STOP_HIT", mae_r_multiple=577181328244.11, mfe_r_multiple=6319090709162.06)

        report = ja.summarize(self.journal_path)

        self.assertIn("LOSS: avg=1.500R n=1", report)
        self.assertNotIn("BREAKEVEN: avg=577181328244", report)
        self.assertIn("1 value(s) excluded as implausible outliers", report)

    def test_outlier_exclusion_is_silent_when_nothing_is_excluded(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mae_r_multiple=1.5, mfe_r_multiple=0.5)

        report = ja.summarize(self.journal_path)

        self.assertNotIn("excluded", report)

    def test_since_timestamp_excludes_trades_that_closed_before_it(self):
        # Real need found live (2026-08-09): a fixed-size batch of stale
        # pre-fix rows doesn't get diluted out just because more trades
        # pile up in a different outcome bucket - most need to be
        # excluded outright by close time, not averaged against forever.
        _write_trade(self.journal_path, "OLD", "BTCUSDT", outcome="TP2_HIT", closed_at=1000.0)
        _write_trade(self.journal_path, "NEW", "ETHUSDT", outcome="SL_HIT", closed_at=2000.0)

        report = ja.summarize(self.journal_path, since_timestamp=1500.0)

        self.assertIn("Resolved trades: 1", report)
        self.assertIn("LOSS=1", report)
        self.assertIn("WIN=0", report)

    def test_since_timestamp_none_includes_everything(self):
        _write_trade(self.journal_path, "OLD", "BTCUSDT", outcome="TP2_HIT", closed_at=1000.0)
        _write_trade(self.journal_path, "NEW", "ETHUSDT", outcome="SL_HIT", closed_at=2000.0)

        report = ja.summarize(self.journal_path, since_timestamp=None)

        self.assertIn("Resolved trades: 2", report)

    def test_summarize_breaks_down_by_outcome_and_the_new_favorable_booleans(self):
        _write_trade(
            self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT",
            efficiency_favorable=True, funding_favorable=True, long_short_favorable=True,
        )
        _write_trade(
            self.journal_path, "B", "ETHUSDT", outcome="LIMIT_EXPIRED_UNFILLED",
            efficiency_favorable=False, funding_favorable=False, long_short_favorable=False,
        )

        report = ja.summarize(self.journal_path)

        # Both count as resolved (outcome is non-empty)...
        self.assertIn("Resolved trades: 2", report)
        # ...but only the real fill counts toward WIN/LOSS/BREAKEVEN.
        self.assertIn("WIN=1 BREAKEVEN=0 LOSS=0 UNKNOWN=1", report)
        self.assertIn("By outcome:", report)
        self.assertIn("TP2_HIT: n=1", report)
        self.assertIn("LIMIT_EXPIRED_UNFILLED: n=1", report)
        self.assertIn("By efficiency favorable (informational):", report)
        self.assertIn("By funding favorable (informational):", report)
        self.assertIn("By long/short favorable (informational):", report)

    def test_summarize_breaks_down_by_entry_trigger(self):
        # config.LIQUIDITY_SWEEP_TRIGGER_ENABLED - lets win rate be
        # compared by trigger type before the sweep path is trusted as
        # much as the existing structure-break path.
        _write_trade(
            self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", signal_trigger="STRUCTURE_BREAK",
        )
        _write_trade(
            self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", signal_trigger="LIQUIDITY_SWEEP",
        )
        _write_trade(
            self.journal_path, "C", "SOLUSDT", outcome="TP2_HIT", signal_trigger="LIQUIDITY_SWEEP",
        )

        report = ja.summarize(self.journal_path)

        self.assertIn("By entry trigger:", report)
        self.assertIn("STRUCTURE_BREAK: n=1", report)
        self.assertIn("LIQUIDITY_SWEEP: n=2", report)

    def test_no_trades_in_window_gives_a_clear_message(self):
        _write_trade(self.journal_path, "OLD", "BTCUSDT", outcome="TP2_HIT", closed_at=1000.0)

        report = ja.summarize(self.journal_path, since_timestamp=5000.0)

        self.assertIn("No resolved", report)
        self.assertIn("in this window", report)


class LossMfeDistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.tmpdir.name) / "journal.csv"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_buckets_loss_trades_by_mfe_range(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mfe_r_multiple=0.1)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", mfe_r_multiple=0.15)
        _write_trade(self.journal_path, "C", "SOLUSDT", outcome="SHADOW_SL_HIT", mfe_r_multiple=0.65)
        _write_trade(self.journal_path, "D", "BNBUSDT", outcome="SL_HIT", mfe_r_multiple=1.4)

        report = ja.summarize(self.journal_path)

        self.assertIn("0.0-0.2R (near-zero - wrong from the first tick): n=2 (50%)", report)
        self.assertIn("0.6-0.8R: n=1 (25%)", report)
        self.assertIn("1.0R+ (ran deep in profit before fully reversing): n=1 (25%)", report)

    def test_win_and_breakeven_trades_are_excluded_from_the_loss_distribution(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", mfe_r_multiple=4.0)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="BREAKEVEN_STOP_HIT", mfe_r_multiple=2.0)
        _write_trade(self.journal_path, "C", "SOLUSDT", outcome="SL_HIT", mfe_r_multiple=0.5)

        report = ja.summarize(self.journal_path)

        self.assertIn("0.4-0.6R: n=1 (100%)", report)

    def test_implausible_outlier_is_excluded_from_the_distribution_too(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mfe_r_multiple=0.5)
        _write_trade(self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", mfe_r_multiple=6319090709162.06)

        report = ja.summarize(self.journal_path)

        self.assertIn("0.4-0.6R: n=1 (100%)", report)
        self.assertNotIn("6319090709162", report)

    def test_no_loss_trades_gives_a_clear_message(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="TP2_HIT", mfe_r_multiple=4.0)

        report = ja.summarize(self.journal_path)

        self.assertIn("(no LOSS trades with mfe_r_multiple recorded yet)", report)


class NearZeroMfeLossBreakdownTests(unittest.TestCase):
    """The isolated-cohort breakdown (side/trigger/HTF trend/CVD/etc.)
    restricted to just the near-zero-MFE LOSS trades - the diagnostic for
    telling apart a genuinely wrong-direction/timing entry from the
    unrelated "ran deep in profit then reversed" LOSS population, which
    would otherwise dilute any breakdown run against all LOSS trades."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.tmpdir.name) / "journal.csv"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_breaks_down_the_near_zero_cohort_by_side_and_trigger(self):
        _write_trade(
            self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mfe_r_multiple=0.05,
            signal="BUY", signal_trigger="STRUCTURE_BREAK",
        )
        _write_trade(
            self.journal_path, "B", "ETHUSDT", outcome="SL_HIT", mfe_r_multiple=0.1,
            signal="BUY", signal_trigger="LIQUIDITY_SWEEP",
        )
        # Real movement in favor before reversing - NOT part of the
        # near-zero cohort, must not appear in the isolated breakdown.
        _write_trade(
            self.journal_path, "C", "SOLUSDT", outcome="SL_HIT", mfe_r_multiple=1.5,
            signal="SELL", signal_trigger="STRUCTURE_BREAK",
        )
        # A WIN, also must not appear.
        _write_trade(
            self.journal_path, "D", "BNBUSDT", outcome="TP2_HIT", mfe_r_multiple=0.05,
            signal="SELL", signal_trigger="STRUCTURE_BREAK",
        )

        report = ja.summarize(self.journal_path)

        # n=2 total, both BUY - already proves trade C (real movement,
        # SELL) and trade D (a WIN) didn't leak into this cohort, since
        # there's no room left for either once both slots are BUY.
        self.assertIn("Near-zero-MFE LOSS trades only (n=2)", report)
        self.assertIn("BUY: n=2", report)
        self.assertIn("STRUCTURE_BREAK: n=1", report)
        self.assertIn("LIQUIDITY_SWEEP: n=1", report)

    def test_no_near_zero_losses_omits_the_section_entirely(self):
        _write_trade(self.journal_path, "A", "BTCUSDT", outcome="SL_HIT", mfe_r_multiple=1.5)

        report = ja.summarize(self.journal_path)

        self.assertNotIn("Near-zero-MFE LOSS trades only", report)


class LoadTradesAcrossRotatedFilesTests(unittest.TestCase):
    """signal_journal.py's _ensure_header rotates the WHOLE file to a
    signal_journal.bak_<timestamp>.csv sibling any time FIELDNAMES changes
    shape while a journal already exists (see its docstring). Confirmed
    live 2026-08-17: ~30 such rotations since Aug 8 left ~95% of real
    trade history sitting in backup files load_trades() never used to
    read - every evidence-based read this project has ever done was drawn
    from whatever sliver of data happened to land after the MOST RECENT
    rotation. load_trades() now globs the target file's stem alongside
    every same-stem sibling in its directory."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.dir = Path(self._tmpdir.name)
        self.current = self.dir / "signal_journal.csv"

    def test_returns_empty_dict_when_no_matching_files_exist(self):
        self.assertEqual(ja.load_trades(self.current), {})

    def test_merges_a_rotated_backup_file_alongside_the_current_one(self):
        # Signal-time-only trade B, written before a rotation (schema-
        # mismatch backup), only ever lands in the .bak file - the current
        # file only knows about trade A. Both should still surface.
        _write_csv(self.current, signal_journal.FIELDNAMES, [
            {"trade_id": "A_1", "symbol": "BTCUSDT", "side": "BUY", "outcome": ""},
        ])
        _write_csv(self.dir / "signal_journal.bak_111.csv", ["trade_id", "symbol", "side", "outcome"], [
            {"trade_id": "B_1", "symbol": "ETHUSDT", "side": "SELL", "outcome": ""},
        ])

        trades = ja.load_trades(self.current)

        self.assertEqual(set(trades.keys()), {"A_1", "B_1"})
        self.assertEqual(trades["B_1"]["symbol"], "ETHUSDT")

    def test_a_trades_signal_row_in_the_backup_and_outcome_row_in_the_current_file_still_merge(self):
        # The exact real-world scenario found live 2026-08-17: a position
        # opened before a rotation, closed after it - its two halves end
        # up in different files and must still join into one trade.
        _write_csv(self.dir / "signal_journal.bak_111.csv", ["trade_id", "symbol", "side", "outcome"], [
            {"trade_id": "C_1", "symbol": "LTCUSDT", "side": "BUY", "outcome": ""},
        ])
        _write_csv(self.current, signal_journal.FIELDNAMES, [
            {"trade_id": "C_1", "symbol": "", "side": "", "outcome": "SL_HIT", "mae_r_multiple": "0.5"},
        ])

        trades = ja.load_trades(self.current)

        self.assertEqual(trades["C_1"]["symbol"], "LTCUSDT")
        self.assertEqual(trades["C_1"]["side"], "BUY")
        self.assertEqual(trades["C_1"]["outcome"], "SL_HIT")
        self.assertEqual(trades["C_1"]["mae_r_multiple"], "0.5")

    def test_blank_fields_never_overwrite_an_already_populated_one_across_files(self):
        _write_csv(self.dir / "signal_journal.bak_111.csv", signal_journal.FIELDNAMES, [
            {"trade_id": "D_1", "symbol": "BTCUSDT", "side": "BUY", "outcome": ""},
        ])
        _write_csv(self.current, signal_journal.FIELDNAMES, [
            {"trade_id": "D_1", "symbol": "", "side": "", "outcome": "TP2_HIT"},
        ])

        trades = ja.load_trades(self.current)

        self.assertEqual(trades["D_1"]["symbol"], "BTCUSDT")
        self.assertEqual(trades["D_1"]["outcome"], "TP2_HIT")

    def test_a_file_with_a_different_stem_is_not_picked_up(self):
        _write_csv(self.current, signal_journal.FIELDNAMES, [
            {"trade_id": "A_1", "symbol": "BTCUSDT", "side": "BUY", "outcome": ""},
        ])
        _write_csv(self.dir / "backtest_signal_journal_999.csv", signal_journal.FIELDNAMES, [
            {"trade_id": "Z_1", "symbol": "SOLUSDT", "side": "BUY", "outcome": ""},
        ])

        trades = ja.load_trades(self.current)

        self.assertEqual(set(trades.keys()), {"A_1"})


if __name__ == "__main__":
    unittest.main()
