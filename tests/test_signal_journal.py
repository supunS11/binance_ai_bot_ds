import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import signal_journal


def _signal(**overrides):
    base = {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "htf_trend": "BULLISH",
        "structure_level": 98.0,
        "atr": 1.0,
        "premium_discount_zone": "DISCOUNT",
        "order_block": {"index": 1},
        "fvg": None,
        "cvd_score": 0.6,
        "depth_imbalance": 0.2,
        "sweep_confluence": True,
    }
    base.update(overrides)
    return base


def _plan(**overrides):
    base = {
        "entry_price": 100.0,
        "sl_price": 98.0,
        "tp1_price": 102.0,
        "tp2_price": 104.0,
        "quantity": 1.0,
        "risk_distance": 2.0,
    }
    base.update(overrides)
    return base


class SignalJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.tmpdir.name) / "journal.csv"
        self.patcher = patch.object(signal_journal, "JOURNAL_PATH", self.journal_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmpdir.cleanup()

    def _read_rows(self):
        with open(self.journal_path, newline="") as handle:
            return list(csv.DictReader(handle))

    def test_append_signal_returns_a_trade_id(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        self.assertTrue(trade_id.startswith("BTCUSDT_"))

    # config.SHADOW_ONLY_TRIGGERS - execution_mode must reflect what
    # ACTUALLY happened for this specific trade (a per-trigger shadow
    # override can force shadow while EXECUTION_MODE stays LIVE), not just
    # echo the global config a second time - real gap found and fixed
    # 2026-08-29 alongside that feature.

    def test_execution_mode_reflects_a_real_shadow_result_even_when_global_mode_is_live(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"):
            signal_journal.append_signal(_signal(), _plan(), execution_result={"shadow": True})

        row = self._read_rows()[0]
        self.assertEqual(row["execution_mode"], "SHADOW")

    def test_execution_mode_reflects_a_real_live_result_even_when_global_mode_is_shadow(self):
        with patch.object(config, "EXECUTION_MODE", "SHADOW"):
            signal_journal.append_signal(_signal(), _plan(), execution_result={"shadow": False})

        row = self._read_rows()[0]
        self.assertEqual(row["execution_mode"], "LIVE")

    def test_execution_mode_falls_back_to_the_global_config_when_execution_result_is_omitted(self):
        with patch.object(config, "EXECUTION_MODE", "LIVE"):
            signal_journal.append_signal(_signal(), _plan())

        row = self._read_rows()[0]
        self.assertEqual(row["execution_mode"], "LIVE")

    def test_append_signal_writes_diagnostic_fields(self):
        signal_journal.append_signal(_signal(), _plan())
        rows = self._read_rows()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["side"], "BUY")
        self.assertEqual(row["risk_distance_pct"], "2.0")
        self.assertEqual(row["order_block_present"], "True")
        self.assertEqual(row["fvg_present"], "False")
        self.assertEqual(row["sweep_confluence"], "True")
        self.assertEqual(row["outcome"], "")

    def test_append_outcome_carries_the_same_trade_id(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "SL_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["trade_id"], trade_id)
        self.assertEqual(rows[1]["trade_id"], trade_id)
        self.assertEqual(rows[1]["outcome"], "SL_HIT")
        self.assertEqual(rows[0]["outcome"], "")

    # config.RETRACEMENT_ENTRY_ENABLED - a second, partial row for the
    # same trade_id once a retracement-pending signal settles into a
    # real position, carrying the REAL settled entry_price (correcting
    # the original row's stale planned/trigger one) plus the two new
    # fill-type/fill-lag diagnostic fields.

    def test_append_retracement_settle_carries_the_same_trade_id(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_retracement_settle("BTCUSDT", trade_id, 99.5, "LIMIT", 184.2)

        rows = self._read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["trade_id"], trade_id)

    def test_append_retracement_settle_writes_the_real_entry_price_and_fill_fields(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())  # planned entry_price=100.0
        signal_journal.append_retracement_settle("BTCUSDT", trade_id, 99.5, "LIMIT", 184.2)

        rows = self._read_rows()
        self.assertEqual(rows[0]["entry_price"], "100.0")  # original row untouched
        self.assertEqual(rows[1]["entry_price"], "99.5")  # real settled price, corrected
        self.assertEqual(rows[1]["retracement_fill_type"], "LIMIT")
        self.assertEqual(rows[1]["retracement_fill_lag_seconds"], "184.2")
        self.assertEqual(rows[1]["outcome"], "")  # not an outcome row

    def test_append_retracement_settle_market_fallback_fill_type(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_retracement_settle("BTCUSDT", trade_id, 100.8, "MARKET_FALLBACK", 300.4)

        rows = self._read_rows()
        self.assertEqual(rows[1]["retracement_fill_type"], "MARKET_FALLBACK")
        self.assertEqual(rows[1]["retracement_fill_lag_seconds"], "300.4")

    def test_append_retracement_settle_writes_used_deep_retracement(self):
        # config.RETRACEMENT_DEPTH_AWARE_ENABLED
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_retracement_settle(
            "BTCUSDT", trade_id, 99.5, "LIMIT", 184.2, used_deep_retracement=True,
        )

        rows = self._read_rows()
        self.assertEqual(rows[1]["used_deep_retracement"], "True")

    def test_append_retracement_settle_defaults_used_deep_retracement_to_false(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_retracement_settle("BTCUSDT", trade_id, 99.5, "LIMIT", 184.2)

        rows = self._read_rows()
        self.assertEqual(rows[1]["used_deep_retracement"], "False")

    def test_append_signal_writes_quote_volume_usdt(self):
        signal_journal.append_signal(_signal(quote_volume_usdt=12_500_000), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["quote_volume_usdt"], "12500000")

    def test_append_signal_writes_the_four_new_data_source_fields(self):
        signal_journal.append_signal(
            _signal(
                efficiency_ratio=0.75, btc_correlation=0.6, btc_aligned=True,
                funding_rate=0.0002, long_short_ratio=1.8,
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["efficiency_ratio"], "0.75")
        self.assertEqual(rows[0]["btc_correlation"], "0.6")
        self.assertEqual(rows[0]["btc_aligned"], "True")
        self.assertEqual(rows[0]["funding_rate"], "0.0002")
        self.assertEqual(rows[0]["long_short_ratio"], "1.8")

    def test_append_signal_writes_absorption_fields(self):
        # config.ABSORPTION_TRACKING_ENABLED - real absorption.py logic
        # covered in test_absorption.py/test_signal_engine.py; this only
        # proves the journal writes it through, same as btc_aligned above.
        signal_journal.append_signal(
            _signal(absorption_signal="BUY", absorption_aligned=True), _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["absorption_signal"], "BUY")
        self.assertEqual(rows[0]["absorption_aligned"], "True")

    def test_append_signal_writes_htf_trend_live_strength_fields(self):
        # config.HTF_TREND_LIVE_STRENGTH_REJECT_ENABLED - real
        # signal_engine.py distance/slope computation covered in
        # test_signal_engine.py; this only proves the journal writes it
        # through, same as absorption_signal above.
        signal_journal.append_signal(
            _signal(htf_trend_live_distance_pct=1.25, htf_trend_live_slope_pct=-0.4),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["htf_trend_live_distance_pct"], "1.25")
        self.assertEqual(rows[0]["htf_trend_live_slope_pct"], "-0.4")

    def test_append_signal_writes_ltf_trend_live(self):
        # config.LTF_TREND_FILTER_ENABLED - journaled unconditionally, even
        # while that gate ships off, so the prospective 1h-vs-4h horizon
        # comparison keeps accumulating real forward data.
        signal_journal.append_signal(_signal(ltf_trend_live="BEARISH"), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["ltf_trend_live"], "BEARISH")

    def test_append_signal_leaves_ltf_trend_live_blank_when_absent(self):
        # None with too little 1h history, or price exactly on the EMA.
        signal_journal.append_signal(_signal(), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["ltf_trend_live"], "")

    def test_append_signal_writes_the_ema_trend_bucket_fields(self):
        # config.EMA_TREND_MIXED_REJECT_ENABLED - all three journaled
        # unconditionally. The two raw regimes travel with the bucket so the
        # live 400-candle EMA200 can be checked against the 1000-candle
        # backtest the gate's evidence came from.
        signal_journal.append_signal(
            _signal(
                ltf_ema_regime="BULLISH",
                htf_ema_regime="BEARISH",
                ema_trend_bucket="MIXED",
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["ltf_ema_regime"], "BULLISH")
        self.assertEqual(rows[0]["htf_ema_regime"], "BEARISH")
        self.assertEqual(rows[0]["ema_trend_bucket"], "MIXED")

    def test_append_signal_leaves_ema_trend_fields_blank_when_absent(self):
        # None until the deeper trend buffer holds EMA_TREND_SLOW_PERIOD
        # candles - the gate is inert in that window and must not journal a
        # misleading bucket.
        signal_journal.append_signal(_signal(), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["ltf_ema_regime"], "")
        self.assertEqual(rows[0]["htf_ema_regime"], "")
        self.assertEqual(rows[0]["ema_trend_bucket"], "")

    def test_append_signal_writes_depth_trend_and_whale_fields(self):
        # config.DEPTH_TREND_TRACKING_ENABLED/WHALE_TRADE_TRACKING_ENABLED -
        # real orderbook.py/order_flow.py/signal_engine.py logic covered in
        # their own test files; this only proves the journal writes it
        # through, same as absorption_signal above.
        signal_journal.append_signal(
            _signal(
                depth_consistency_pct=0.8,
                depth_trend_aligned=True,
                whale_notional=25000,
                whale_direction="BUY",
                whale_aligned=True,
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["depth_consistency_pct"], "0.8")
        self.assertEqual(rows[0]["depth_trend_aligned"], "True")
        self.assertEqual(rows[0]["whale_notional"], "25000")
        self.assertEqual(rows[0]["whale_direction"], "BUY")
        self.assertEqual(rows[0]["whale_aligned"], "True")

    def test_append_signal_writes_cross_exchange_oi_fields(self):
        # config.CROSS_EXCHANGE_OI_TRACKING_ENABLED - real
        # cross_exchange_oi.py logic covered in test_cross_exchange_oi.py/
        # test_signal_engine.py; this only proves the journal writes it
        # through, same as absorption_signal above.
        signal_journal.append_signal(
            _signal(
                oi_change_pct_bybit=2.0, oi_change_pct_okx=1.0, cross_exchange_oi_agree=True,
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["oi_change_pct_bybit"], "2.0")
        self.assertEqual(rows[0]["oi_change_pct_okx"], "1.0")
        self.assertEqual(rows[0]["cross_exchange_oi_agree"], "True")

    def test_append_signal_writes_volume_profile_fields(self):
        signal_journal.append_signal(
            _signal(
                vp_poc_price=100.0, vp_value_area_high=105.0,
                vp_value_area_low=95.0, vp_position="INSIDE_VALUE_AREA",
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["vp_poc_price"], "100.0")
        self.assertEqual(rows[0]["vp_value_area_high"], "105.0")
        self.assertEqual(rows[0]["vp_value_area_low"], "95.0")
        self.assertEqual(rows[0]["vp_position"], "INSIDE_VALUE_AREA")

    def test_append_signal_writes_the_three_favorable_boolean_fields(self):
        signal_journal.append_signal(
            _signal(
                efficiency_favorable=True, funding_favorable=False, long_short_favorable=True,
            ),
            _plan(),
        )
        rows = self._read_rows()

        self.assertEqual(rows[0]["efficiency_favorable"], "True")
        self.assertEqual(rows[0]["funding_favorable"], "False")
        self.assertEqual(rows[0]["long_short_favorable"], "True")

    def test_append_signal_writes_ema_alignment_value(self):
        # config.EMA_ALIGNMENT_PERIOD - the faster EMA used only for
        # ema_aligned, deliberately separate from ema_value/
        # EMA_CONFIRMATION_PERIOD (still journaled independently).
        signal_journal.append_signal(_signal(ema_value=101.0, ema_alignment_value=99.5), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["ema_value"], "101.0")
        self.assertEqual(rows[0]["ema_alignment_value"], "99.5")

    def test_append_signal_writes_entry_extension_r_from_the_plan(self):
        signal_journal.append_signal(_signal(), _plan(entry_extension_r=0.35))
        rows = self._read_rows()

        self.assertEqual(rows[0]["entry_extension_r"], "0.35")

    def test_append_signal_writes_nearest_favorable_sr_r_from_the_plan(self):
        signal_journal.append_signal(_signal(), _plan(nearest_favorable_sr_r=0.5))
        rows = self._read_rows()

        self.assertEqual(rows[0]["nearest_favorable_sr_r"], "0.5")

    def test_append_signal_writes_setup_age_candles_from_the_signal(self):
        signal_journal.append_signal(_signal(setup_age_candles=6), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["setup_age_candles"], "6")

    def test_append_signal_writes_zero_setup_age_candles(self):
        # 0 is a real, meaningful value (STRUCTURE_BREAK/EMA_PULLBACK are
        # always fresh) - must not be treated as falsy/blank.
        signal_journal.append_signal(_signal(setup_age_candles=0), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["setup_age_candles"], "0")

    def test_append_signal_writes_zone_retracement_pct(self):
        signal_journal.append_signal(_signal(zone_retracement_pct=0.74), _plan())
        rows = self._read_rows()

        self.assertEqual(rows[0]["zone_retracement_pct"], "0.74")

    def test_zero_entry_price_does_not_crash_risk_distance_calc(self):
        signal_journal.append_signal(_signal(), _plan(entry_price=0))
        rows = self._read_rows()
        self.assertEqual(rows[0]["risk_distance_pct"], "")

    def test_append_outcome_writes_early_breakeven_applied(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "BREAKEVEN_STOP_HIT", trade_id, early_breakeven_applied=True)

        rows = self._read_rows()
        self.assertEqual(rows[1]["early_breakeven_applied"], "True")

    def test_append_outcome_leaves_early_breakeven_applied_blank_when_not_given(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "SL_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(rows[1]["early_breakeven_applied"], "")

    def test_append_outcome_writes_btc_max_adverse_move_pct(self):
        # config.BTC_ADVERSE_MOVE_TRACKING_ENABLED
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "SL_HIT", trade_id, btc_max_adverse_move_pct=0.42)

        rows = self._read_rows()
        self.assertEqual(rows[1]["btc_max_adverse_move_pct"], "0.42")

    def test_append_outcome_leaves_btc_max_adverse_move_pct_blank_when_not_given(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "SL_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(rows[1]["btc_max_adverse_move_pct"], "")

    def test_append_outcome_writes_dca_breakeven_direction_confirmed(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome(
            "BTCUSDT", "DCA_TP_HIT", trade_id, dca_breakeven_direction_confirmed=True,
        )

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_breakeven_direction_confirmed"], "True")

    def test_append_outcome_leaves_dca_breakeven_direction_confirmed_blank_when_not_given(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "DCA_SL_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_breakeven_direction_confirmed"], "")

    def test_append_outcome_writes_false_dca_breakeven_direction_confirmed(self):
        # False is a real, meaningful verdict (the check ran and failed) -
        # must not be treated the same as "never ran" (None/blank).
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome(
            "BTCUSDT", "DCA_SL_HIT", trade_id, dca_breakeven_direction_confirmed=False,
        )

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_breakeven_direction_confirmed"], "False")

    def test_append_outcome_writes_dca_pressure_confirmed(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome(
            "BTCUSDT", "DCA_TP_HIT", trade_id, dca_pressure_confirmed=True,
        )

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_pressure_confirmed"], "True")

    def test_append_outcome_leaves_dca_pressure_confirmed_blank_when_not_given(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "DCA_SL_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_pressure_confirmed"], "")

    def test_append_outcome_writes_false_dca_pressure_confirmed(self):
        # False is a real, meaningful verdict (order flow was NOT
        # confirmed, so this fire used the reduced size/tighter stop) -
        # must not be treated the same as "check never ran" (None/blank).
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome(
            "BTCUSDT", "DCA_SL_HIT", trade_id, dca_pressure_confirmed=False,
        )

        rows = self._read_rows()
        self.assertEqual(rows[1]["dca_pressure_confirmed"], "False")

    def test_append_outcome_writes_break_confirmed_by_close(self):
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "SL_HIT", trade_id, break_confirmed_by_close=False)

        rows = self._read_rows()
        self.assertEqual(rows[1]["break_confirmed_by_close"], "False")


class HeaderMigrationTests(unittest.TestCase):
    """Real bug found live: a schema change (new diagnostic fields) while
    a journal already existed left the OLD header sitting above NEW-shaped
    rows - csv.DictReader keys everything off that stale header, so
    trade_id and every newer field silently stopped resolving for every
    row written after the change (journal_analysis.py reported zero
    resolved trades despite real matching data being in the file)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.tmpdir.name) / "journal.csv"
        self.patcher = patch.object(signal_journal, "JOURNAL_PATH", self.journal_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmpdir.cleanup()

    def _read_rows(self):
        with open(self.journal_path, newline="") as handle:
            return list(csv.DictReader(handle))

    def _write_stale_file(self):
        stale_fields = ["timestamp", "symbol", "side", "entry_price", "sl_price",
                         "tp1_price", "tp2_price", "quantity", "htf_trend",
                         "cvd_score", "depth_imbalance", "sweep_confluence",
                         "execution_mode", "outcome"]
        with open(self.journal_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=stale_fields)
            writer.writeheader()
            writer.writerow({"timestamp": 1, "symbol": "OLDUSDT", "outcome": "SL_HIT"})

    def test_matching_header_is_left_alone(self):
        signal_journal.append_signal(_signal(), _plan())  # writes current-schema header
        before = self.journal_path.read_text()

        signal_journal.append_signal(_signal(), _plan())
        after = self.journal_path.read_text()

        self.assertTrue(after.startswith(before.splitlines()[0]))
        self.assertEqual(len(list(self.journal_path.parent.glob("*.bak_*.csv"))), 0)

    def test_stale_header_is_backed_up_and_replaced(self):
        self._write_stale_file()

        trade_id = signal_journal.append_signal(_signal(), _plan())

        backups = list(self.journal_path.parent.glob("signal_journal.bak_*.csv"))
        self.assertEqual(len(backups), 1)
        # The old data is preserved in the backup, not silently discarded.
        self.assertIn("OLDUSDT", backups[0].read_text())

        rows = self._read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trade_id"], trade_id)

    def test_appending_after_migration_uses_the_new_schema_correctly(self):
        self._write_stale_file()
        trade_id = signal_journal.append_signal(_signal(), _plan())
        signal_journal.append_outcome("BTCUSDT", "TP2_HIT", trade_id)

        rows = self._read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["trade_id"], trade_id)
        self.assertEqual(rows[1]["outcome"], "TP2_HIT")


if __name__ == "__main__":
    unittest.main()
