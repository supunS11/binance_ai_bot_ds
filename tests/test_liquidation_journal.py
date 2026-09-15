import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
import liquidation_journal


class AppendEventTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "liquidation_events.csv"
        self.patcher = patch.object(liquidation_journal, "EVENTS_PATH", self.path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmpdir.cleanup()

    def _rows(self):
        with open(self.path, newline="") as handle:
            return list(csv.DictReader(handle))

    def test_append_event_writes_one_row(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)
        row = self._rows()[0]

        self.assertEqual(row["exchange"], "BINANCE")
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["side"], "SELL")
        self.assertEqual(row["price"], "50000.0")
        self.assertEqual(row["notional"], "20000.0")
        self.assertEqual(row["timestamp"], "1000.0")

    def test_creates_the_header_on_first_use(self):
        self.assertFalse(self.path.exists())
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)

        with open(self.path, newline="") as handle:
            self.assertEqual(next(csv.reader(handle)), liquidation_journal.EVENT_FIELDNAMES)

    def test_append_event_is_a_noop_when_journal_disabled(self):
        with patch.object(config, "LIQUIDATION_EVENT_JOURNAL_ENABLED", False):
            liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)

        self.assertFalse(self.path.exists())

    def test_append_event_never_raises_on_oserror(self):
        with patch.object(liquidation_journal, "_ensure_event_header", side_effect=OSError("disk full")):
            liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)

    def test_mismatched_header_is_backed_up_and_rewritten(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=["timestamp", "symbol"]).writeheader()

        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)

        with open(self.path, newline="") as handle:
            self.assertEqual(next(csv.reader(handle)), liquidation_journal.EVENT_FIELDNAMES)
        self.assertTrue(any(
            p.name.startswith("liquidation_events.bak_") for p in self.path.parent.iterdir()
        ))

    def test_missing_price_is_written_as_empty_not_none(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, None)
        row = self._rows()[0]

        self.assertEqual(row["price"], "")


class LoadEventsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "liquidation_events.csv"
        self.patcher = patch.object(liquidation_journal, "EVENTS_PATH", self.path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmpdir.cleanup()

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(liquidation_journal.load_events(), [])

    def test_loads_written_events_back(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)
        liquidation_journal.append_event("BYBIT", "ETHUSDT", "BUY", 5000.0, 1500.0, 3000.0)

        events = liquidation_journal.load_events()

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["symbol"], "BTCUSDT")
        self.assertEqual(events[0]["price"], 50000.0)
        self.assertEqual(events[0]["notional"], 20000.0)

    def test_filters_by_symbol(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)
        liquidation_journal.append_event("BINANCE", "ETHUSDT", "SELL", 20000.0, 1000.0, 3000.0)

        events = liquidation_journal.load_events(symbol="ethusdt")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["symbol"], "ETHUSDT")

    def test_filters_by_since(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, 50000.0)
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 2000.0, 51000.0)

        events = liquidation_journal.load_events(since=1500.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp"], 2000.0)

    def test_rows_with_no_price_are_skipped(self):
        liquidation_journal.append_event("BINANCE", "BTCUSDT", "SELL", 20000.0, 1000.0, None)

        self.assertEqual(liquidation_journal.load_events(), [])


if __name__ == "__main__":
    unittest.main()
