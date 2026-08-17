import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import exchange


def _kline_row(open_time, close_time):
    return [open_time, "1.0", "2.0", "0.5", "1.5", "10.0", close_time, "15.0", 5, "5.0", "7.5", "0"]


def _agg_trade(ts_ms, price="1.0", qty="1.0", is_buyer_maker=False):
    return {"a": 1, "p": price, "q": qty, "f": 1, "l": 1, "T": ts_ms, "m": is_buyer_maker}


def _funding_row(funding_time, rate="0.0001"):
    return {"symbol": "BTCUSDT", "fundingTime": funding_time, "fundingRate": rate}


class _BacktestCacheIsolation(unittest.TestCase):
    """Every test here hits real disk via exchange._backtest_cache_path -
    isolate to a temp dir so no test run ever touches data/backtest_cache/."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cache_dir_patcher = patch.object(exchange, "_BACKTEST_CACHE_DIR", Path(self._tmp.name))
        self._cache_dir_patcher.start()

    def tearDown(self):
        self._cache_dir_patcher.stop()
        self._tmp.cleanup()


class GetHistoricalKlinesTests(_BacktestCacheIsolation):
    def test_single_page_returns_full_dataframe(self):
        rows = [_kline_row(0, 3_599_999), _kline_row(3_600_000, 7_199_999)]

        with patch.object(exchange.client, "futures_klines", return_value=rows) as mock_klines:
            df = exchange.get_historical_klines("BTCUSDT", "1h", 0, 7_200_000)

        mock_klines.assert_called_once()
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["open"]), [1.0, 1.0])

    def test_paginates_when_a_full_page_is_returned(self):
        interval_ms = exchange.KLINE_INTERVAL_MS["1h"]
        full_page = [_kline_row(i * interval_ms, (i + 1) * interval_ms - 1) for i in range(1500)]
        second_page = [_kline_row(1500 * interval_ms, 1501 * interval_ms - 1)]

        with patch.object(
            exchange.client, "futures_klines", side_effect=[full_page, second_page]
        ) as mock_klines:
            df = exchange.get_historical_klines("BTCUSDT", "1h", 0, 1502 * interval_ms)

        self.assertEqual(mock_klines.call_count, 2)
        self.assertEqual(len(df), 1501)
        second_call_kwargs = mock_klines.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs["startTime"], 1500 * interval_ms)

    def test_cache_hit_skips_the_real_call(self):
        rows = [_kline_row(0, 3_599_999)]

        with patch.object(exchange.client, "futures_klines", return_value=rows) as mock_klines:
            exchange.get_historical_klines("BTCUSDT", "1h", 0, 3_600_000)
            exchange.get_historical_klines("BTCUSDT", "1h", 0, 3_600_000)

        mock_klines.assert_called_once()

    def test_unknown_interval_returns_none_without_calling_client(self):
        with patch.object(exchange.client, "futures_klines") as mock_klines:
            df = exchange.get_historical_klines("BTCUSDT", "7h", 0, 3_600_000)

        mock_klines.assert_not_called()
        self.assertIsNone(df)

    def test_empty_response_returns_an_empty_dataframe_not_none(self):
        with patch.object(exchange.client, "futures_klines", return_value=[]):
            df = exchange.get_historical_klines("BTCUSDT", "1h", 0, 3_600_000)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 0)

    def test_exception_returns_none(self):
        with patch.object(exchange.client, "futures_klines", side_effect=RuntimeError("boom")):
            df = exchange.get_historical_klines("BTCUSDT", "1h", 0, 3_600_000)

        self.assertIsNone(df)


class GetHistoricalAggTradesTests(_BacktestCacheIsolation):
    def test_paginates_within_an_hour_window_when_over_1000_trades(self):
        first_page = [_agg_trade(i) for i in range(1000)]
        second_page = [_agg_trade(1000)]

        with patch.object(
            exchange.client, "futures_aggregate_trades", side_effect=[first_page, second_page]
        ) as mock_trades:
            result = exchange.get_historical_agg_trades("BTCUSDT", 0, 3_600_000)

        self.assertEqual(mock_trades.call_count, 2)
        self.assertEqual(len(result), 1001)

    def test_advances_to_the_next_hour_window(self):
        one_hour = 3_600_000

        with patch.object(
            exchange.client, "futures_aggregate_trades",
            side_effect=[[_agg_trade(100)], [_agg_trade(one_hour + 100)]],
        ) as mock_trades:
            result = exchange.get_historical_agg_trades("BTCUSDT", 0, 2 * one_hour)

        self.assertEqual(mock_trades.call_count, 2)
        self.assertEqual(len(result), 2)
        first_call_kwargs = mock_trades.call_args_list[0].kwargs
        second_call_kwargs = mock_trades.call_args_list[1].kwargs
        self.assertEqual(first_call_kwargs["startTime"], 0)
        self.assertEqual(second_call_kwargs["startTime"], one_hour)

    def test_cache_hit_skips_the_real_call(self):
        with patch.object(
            exchange.client, "futures_aggregate_trades", return_value=[_agg_trade(0)]
        ) as mock_trades:
            exchange.get_historical_agg_trades("BTCUSDT", 0, 3_600_000)
            exchange.get_historical_agg_trades("BTCUSDT", 0, 3_600_000)

        mock_trades.assert_called_once()

    def test_exception_returns_none(self):
        with patch.object(exchange.client, "futures_aggregate_trades", side_effect=RuntimeError("boom")):
            result = exchange.get_historical_agg_trades("BTCUSDT", 0, 3_600_000)

        self.assertIsNone(result)


class GetHistoricalFundingRatesTests(_BacktestCacheIsolation):
    def test_returns_parsed_float_rates(self):
        rows = [_funding_row(0, "0.0001"), _funding_row(28_800_000, "-0.0002")]

        with patch.object(exchange.client, "futures_funding_rate", return_value=rows):
            result = exchange.get_historical_funding_rates("BTCUSDT", 0, 57_600_000)

        self.assertEqual(result, [
            {"fundingTime": 0, "fundingRate": 0.0001},
            {"fundingTime": 28_800_000, "fundingRate": -0.0002},
        ])

    def test_paginates_past_1000_records(self):
        first_page = [_funding_row(i) for i in range(1000)]
        second_page = [_funding_row(1000)]

        with patch.object(
            exchange.client, "futures_funding_rate", side_effect=[first_page, second_page]
        ) as mock_funding:
            result = exchange.get_historical_funding_rates("BTCUSDT", 0, 1_000_000_000)

        self.assertEqual(mock_funding.call_count, 2)
        self.assertEqual(len(result), 1001)

    def test_cache_hit_skips_the_real_call(self):
        with patch.object(
            exchange.client, "futures_funding_rate", return_value=[_funding_row(0)]
        ) as mock_funding:
            exchange.get_historical_funding_rates("BTCUSDT", 0, 3_600_000)
            exchange.get_historical_funding_rates("BTCUSDT", 0, 3_600_000)

        mock_funding.assert_called_once()

    def test_exception_returns_none(self):
        with patch.object(exchange.client, "futures_funding_rate", side_effect=RuntimeError("boom")):
            result = exchange.get_historical_funding_rates("BTCUSDT", 0, 3_600_000)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
