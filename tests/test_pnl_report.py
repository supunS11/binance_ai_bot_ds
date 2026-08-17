import unittest
from unittest.mock import patch

import exchange
import pnl_report


def _record(symbol, income_type, income, time_ms):
    return {"symbol": symbol, "incomeType": income_type, "income": str(income), "time": time_ms}


class FetchAllTests(unittest.TestCase):
    def test_stops_after_a_short_batch(self):
        with patch.object(exchange, "get_income_history", return_value=[_record("BTCUSDT", "REALIZED_PNL", 1, 1000)]) as mock_call:
            results = pnl_report._fetch_all(start_time=0)

        self.assertEqual(len(results), 1)
        mock_call.assert_called_once()

    def test_pages_when_a_batch_hits_the_limit(self):
        full_batch = [_record("BTCUSDT", "REALIZED_PNL", 1, t) for t in range(1000)]
        short_batch = [_record("BTCUSDT", "REALIZED_PNL", 1, 2000)]

        with patch.object(exchange, "get_income_history", side_effect=[full_batch, short_batch]) as mock_call:
            results = pnl_report._fetch_all(start_time=0)

        self.assertEqual(len(results), 1001)
        self.assertEqual(mock_call.call_count, 2)
        # second call's cursor is the last record's time + 1
        _, second_kwargs = mock_call.call_args_list[1]
        self.assertEqual(second_kwargs["start_time"], 1000)

    def test_no_records_returns_empty_list(self):
        with patch.object(exchange, "get_income_history", return_value=[]):
            results = pnl_report._fetch_all(start_time=0)

        self.assertEqual(results, [])


class SummarizeTests(unittest.TestCase):
    def test_no_records_message(self):
        with patch.object(exchange, "get_income_history", return_value=[]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        self.assertIn("No income records", result)

    def test_totals_split_by_income_type(self):
        records = [
            _record("BTCUSDT", "REALIZED_PNL", 5.0, 1000),
            _record("BTCUSDT", "COMMISSION", -0.5, 1000),
            _record("BTCUSDT", "FUNDING_FEE", -0.1, 1000),
        ]

        with patch.object(exchange, "get_income_history", side_effect=[records, []]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        self.assertIn("Realized PNL: +5.0000 USDT", result)
        self.assertIn("Commission:   -0.5000 USDT", result)
        self.assertIn("Funding:      -0.1000 USDT", result)
        self.assertIn("Net:          +4.4000 USDT", result)

    def test_by_symbol_breakdown_sorted_worst_to_best(self):
        records = [
            _record("BTCUSDT", "REALIZED_PNL", 5.0, 1000),
            _record("ETHUSDT", "REALIZED_PNL", -3.0, 1000),
        ]

        with patch.object(exchange, "get_income_history", side_effect=[records, []]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        eth_index = result.index("ETHUSDT")
        btc_index = result.index("BTCUSDT: realized")
        self.assertLess(eth_index, btc_index)  # worst (ETH, -3) listed before best (BTC, +5)

    def test_unclassified_income_type_shown_as_other(self):
        records = [_record("BTCUSDT", "TRANSFER", 2.0, 1000)]

        with patch.object(exchange, "get_income_history", side_effect=[records, []]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        self.assertIn("Other:        +2.0000 USDT", result)

    def test_no_other_line_when_only_known_income_types(self):
        records = [_record("BTCUSDT", "REALIZED_PNL", 1.0, 1000)]

        with patch.object(exchange, "get_income_history", side_effect=[records, []]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        self.assertNotIn("Other:", result)

    def test_malformed_income_value_is_treated_as_zero_not_a_crash(self):
        records = [_record("BTCUSDT", "REALIZED_PNL", "not-a-number", 1000)]

        with patch.object(exchange, "get_income_history", side_effect=[records, []]):
            result = pnl_report.summarize(hours_back=24, now=1_000_000)

        self.assertIn("Realized PNL: +0.0000 USDT", result)


if __name__ == "__main__":
    unittest.main()
