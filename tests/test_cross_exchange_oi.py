import unittest
from unittest.mock import MagicMock, patch

import requests

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


def _http_error(status_code):
    """A real requests.exceptions.HTTPError with a mock .response carrying
    the given status code - the shape response.raise_for_status() actually
    raises, needed to exercise _handle_fetch_error's status-code branch."""
    exc = requests.exceptions.HTTPError(f"{status_code} Client Error")
    exc.response = MagicMock(status_code=status_code)
    return exc


def _response_raising(exc):
    response = MagicMock()
    response.raise_for_status.side_effect = exc
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

    def test_non_ascii_base_asset_returns_none(self):
        # Real event (2026-08-26): a handful of Binance-listed meme
        # perpetuals have non-ASCII (CJK) base asset names - these must
        # never reach a request at all, not just fail gracefully once
        # OKX rejects them.
        self.assertIsNone(_to_okx_symbol("我踏马来了USDT"))

    def test_lowercase_or_symbol_characters_in_base_asset_return_none(self):
        self.assertIsNone(_to_okx_symbol("btcUSDT"))
        self.assertIsNone(_to_okx_symbol("BTC-USDT"))


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

    def test_http_400_marks_symbol_unavailable_so_it_is_not_retried(self):
        # Real event (2026-08-26): a plain `except Exception` treated this
        # identically to a transient network hiccup, retrying (and
        # re-logging an ERROR) every poll cycle forever.
        response = _response_raising(_http_error(400))

        with patch.object(cross_exchange_oi.requests, "get", return_value=response) as mock_get:
            get_open_interest_bybit("BTCUSDT")
            get_open_interest_bybit("BTCUSDT")

        self.assertEqual(mock_get.call_count, 1)

    def test_http_429_backs_off_without_marking_the_symbol_unavailable(self):
        response = _response_raising(_http_error(429))

        with patch.object(cross_exchange_oi.requests, "get", return_value=response):
            get_open_interest_bybit("BTCUSDT")

        self.assertTrue(cross_exchange_oi._is_backing_off("bybit"))
        self.assertFalse(cross_exchange_oi._is_symbol_unavailable("bybit", "BTCUSDT"))

    def test_non_http_exception_does_not_mark_the_symbol_unavailable(self):
        # A generic network/timeout error is transient, not symbol-
        # specific - must not be treated the same as a real 4xx rejection.
        with patch.object(cross_exchange_oi.requests, "get", side_effect=RuntimeError("boom")):
            get_open_interest_bybit("BTCUSDT")

        self.assertFalse(cross_exchange_oi._is_symbol_unavailable("bybit", "BTCUSDT"))


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

    def test_http_400_marks_symbol_unavailable_so_it_is_not_retried(self):
        # Real event (2026-08-26): the exact incident this fix addresses -
        # a non-ASCII-named symbol reached this point (pre-fix, before
        # _to_okx_symbol's own charset guard existed) and OKX rejected it
        # with a flat 400, retried forever.
        response = _response_raising(_http_error(400))

        with patch.object(cross_exchange_oi.requests, "get", return_value=response) as mock_get:
            get_open_interest_okx("BTCUSDT")
            get_open_interest_okx("BTCUSDT")

        self.assertEqual(mock_get.call_count, 1)

    def test_http_429_backs_off_without_marking_the_symbol_unavailable(self):
        response = _response_raising(_http_error(429))

        with patch.object(cross_exchange_oi.requests, "get", return_value=response):
            get_open_interest_okx("BTCUSDT")

        self.assertTrue(cross_exchange_oi._is_backing_off("okx"))
        self.assertFalse(cross_exchange_oi._is_symbol_unavailable("okx", "BTCUSDT"))

    def test_non_http_exception_does_not_mark_the_symbol_unavailable(self):
        with patch.object(cross_exchange_oi.requests, "get", side_effect=RuntimeError("boom")):
            get_open_interest_okx("BTCUSDT")

        self.assertFalse(cross_exchange_oi._is_symbol_unavailable("okx", "BTCUSDT"))


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
