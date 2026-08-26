import unittest
from unittest.mock import MagicMock, patch

import config
import cross_exchange_oi
from cross_exchange_oi import (
    compute_agreement,
    get_open_interest_bybit,
    get_open_interest_okx,
    _to_okx_symbol,
)


def _response(json_payload, status_ok=True):
    response = MagicMock()
    response.json.return_value = json_payload

    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = RuntimeError("HTTP error")

    return response


class ToOkxSymbolTests(unittest.TestCase):
    def test_usdt_symbol_gets_dash_and_swap_suffix(self):
        self.assertEqual(_to_okx_symbol("BTCUSDT"), "BTC-USDT-SWAP")

    def test_symbol_with_leading_digits_is_preserved(self):
        self.assertEqual(_to_okx_symbol("1000PEPEUSDT"), "1000PEPE-USDT-SWAP")

    def test_non_usdt_symbol_returns_none(self):
        self.assertIsNone(_to_okx_symbol("BTCBUSD"))

    def test_bare_usdt_returns_none(self):
        self.assertIsNone(_to_okx_symbol("USDT"))


class GetOpenInterestBybitTests(unittest.TestCase):
    def setUp(self):
        cross_exchange_oi.reset()
        self.addCleanup(cross_exchange_oi.reset)
        patcher = patch.object(config, "CROSS_EXCHANGE_OI_MIN_REQUEST_GAP_SECONDS", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parses_open_interest_from_result_list(self):
        payload = {"retCode": 0, "result": {"list": [{"openInterest": "12345.67"}]}}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)):
            value = get_open_interest_bybit("BTCUSDT")

        self.assertEqual(value, 12345.67)

    def test_non_zero_ret_code_returns_none(self):
        payload = {"retCode": 10001, "retMsg": "symbol invalid"}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)):
            value = get_open_interest_bybit("NOTASYMBOL")

        self.assertIsNone(value)

    def test_empty_list_returns_none(self):
        payload = {"retCode": 0, "result": {"list": []}}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)):
            value = get_open_interest_bybit("BTCUSDT")

        self.assertIsNone(value)

    def test_network_exception_returns_none(self):
        with patch.object(cross_exchange_oi.requests, "get", side_effect=RuntimeError("boom")):
            value = get_open_interest_bybit("BTCUSDT")

        self.assertIsNone(value)

    def test_http_error_returns_none(self):
        with patch.object(cross_exchange_oi.requests, "get", return_value=_response({}, status_ok=False)):
            value = get_open_interest_bybit("BTCUSDT")

        self.assertIsNone(value)

    def test_unavailable_symbol_is_skipped_on_next_call_without_a_new_request(self):
        payload = {"retCode": 0, "result": {"list": []}}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)) as mock_get:
            get_open_interest_bybit("BTCUSDT")
            get_open_interest_bybit("BTCUSDT")

        self.assertEqual(mock_get.call_count, 1)


class GetOpenInterestOkxTests(unittest.TestCase):
    def setUp(self):
        cross_exchange_oi.reset()
        self.addCleanup(cross_exchange_oi.reset)
        patcher = patch.object(config, "CROSS_EXCHANGE_OI_MIN_REQUEST_GAP_SECONDS", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parses_oi_from_data_list(self):
        payload = {"code": "0", "data": [{"oi": "999.5"}]}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)):
            value = get_open_interest_okx("BTCUSDT")

        self.assertEqual(value, 999.5)

    def test_non_usdt_symbol_returns_none_without_a_request(self):
        with patch.object(cross_exchange_oi.requests, "get") as mock_get:
            value = get_open_interest_okx("BTCBUSD")

        self.assertIsNone(value)
        mock_get.assert_not_called()

    def test_non_zero_code_returns_none(self):
        payload = {"code": "51001", "data": []}

        with patch.object(cross_exchange_oi.requests, "get", return_value=_response(payload)):
            value = get_open_interest_okx("NOTASYMBOLUSDT")

        self.assertIsNone(value)

    def test_network_exception_returns_none(self):
        with patch.object(cross_exchange_oi.requests, "get", side_effect=RuntimeError("boom")):
            value = get_open_interest_okx("BTCUSDT")

        self.assertIsNone(value)


class ComputeAgreementTests(unittest.TestCase):
    def test_binance_unavailable_returns_none(self):
        self.assertIsNone(compute_agreement(None, 1.0, 1.0))

    def test_no_cross_exchange_readings_returns_none(self):
        self.assertIsNone(compute_agreement(1.0, None, None))

    def test_both_venues_agree_rising(self):
        self.assertTrue(compute_agreement(2.0, 1.0, 0.5))

    def test_both_venues_agree_falling(self):
        self.assertTrue(compute_agreement(-2.0, -1.0, -0.5))

    def test_one_venue_disagrees_is_false(self):
        self.assertFalse(compute_agreement(2.0, 1.0, -0.5))

    def test_only_one_venue_available_and_it_agrees(self):
        self.assertTrue(compute_agreement(2.0, None, 1.0))

    def test_only_one_venue_available_and_it_disagrees(self):
        self.assertFalse(compute_agreement(2.0, None, -1.0))

    def test_zero_binance_change_counts_as_not_rising(self):
        # oi_rising = oi_change_pct > 0 (signal_engine.py's own definition)
        # - zero is NOT rising, so a positive cross-exchange reading
        # disagrees.
        self.assertFalse(compute_agreement(0.0, 1.0, None))


if __name__ == "__main__":
    unittest.main()
