import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_append_signal_writes_entry_extension_r_from_the_plan(self):
        signal_journal.append_signal(_signal(), _plan(entry_extension_r=0.35))
        rows = self._read_rows()

        self.assertEqual(rows[0]["entry_extension_r"], "0.35")

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
