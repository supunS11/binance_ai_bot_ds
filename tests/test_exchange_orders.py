import time
import unittest
from unittest.mock import patch

import config
import exchange


class OrderParameterTests(unittest.TestCase):
    """Binance's algo-order endpoint (client.futures_create_algo_order)
    requires the trigger price under the key `triggerPrice` - NOT
    `stopPrice`, which is only valid on the plain /fapi/v1/order endpoint.
    Sending the wrong key fails with APIError(-1102) at the exchange, which
    a test that only mocks exchange.place_stop_loss itself (as
    test_execution.py does) can never catch - these test the actual
    parameters built by exchange.py."""

    def test_place_stop_loss_sends_trigger_price_not_stop_price(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_stop_loss("BTCUSDT", "BUY", 98.0)

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["triggerPrice"], 98.0)
        self.assertNotIn("stopPrice", kwargs)
        self.assertEqual(kwargs["closePosition"], "true")

    def test_place_take_profit_partial_sends_trigger_price_not_stop_price(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_take_profit_partial("BTCUSDT", "BUY", 0.5, 102.0)

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["triggerPrice"], 102.0)
        self.assertNotIn("stopPrice", kwargs)
        self.assertEqual(kwargs["reduceOnly"], "true")

    def test_place_take_profit_full_sends_trigger_price_not_stop_price(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_take_profit_full("BTCUSDT", "BUY", 104.0)

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["triggerPrice"], 104.0)
        self.assertNotIn("stopPrice", kwargs)
        self.assertEqual(kwargs["closePosition"], "true")

    def test_place_stop_loss_without_client_algo_id_omits_the_key(self):
        # Every ordinary (non-DCA) caller - clientAlgoId must not appear
        # at all, not even as an empty string, so _adopt_position's DCA
        # tag check never false-positives on an untagged order.
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_stop_loss("BTCUSDT", "BUY", 98.0)

        _, kwargs = mock_place.call_args
        self.assertNotIn("clientAlgoId", kwargs)

    def test_place_stop_loss_with_client_algo_id_passes_it_through(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_stop_loss("BTCUSDT", "BUY", 98.0, client_algo_id="dcaSL123")

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["clientAlgoId"], "dcaSL123")

    def test_place_take_profit_full_with_client_algo_id_passes_it_through(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_take_profit_full("BTCUSDT", "BUY", 104.0, client_algo_id="dcaTP123")

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["clientAlgoId"], "dcaTP123")

    def test_place_trailing_stop_loss_sends_activate_price_callback_rate_and_quantity(self):
        # config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a real TRAILING_
        # STOP_MARKET. Real bug found live (2026-08-26, MEUSDT): Binance
        # rejects closePosition=true for this order type with -4136 -
        # reduceOnly=true + an explicit quantity is the confirmed working
        # shape instead, same as place_take_profit_partial.
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_trailing_stop_loss("BTCUSDT", "BUY", 1.5, 98.0, 0.2)

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["type"], "TRAILING_STOP_MARKET")
        self.assertEqual(kwargs["activatePrice"], 98.0)
        self.assertEqual(kwargs["callbackRate"], 0.2)
        self.assertEqual(kwargs["quantity"], 1.5)
        self.assertEqual(kwargs["reduceOnly"], "true")
        self.assertNotIn("triggerPrice", kwargs)
        self.assertNotIn("closePosition", kwargs)

    def test_place_trailing_stop_loss_without_client_algo_id_omits_the_key(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_trailing_stop_loss("BTCUSDT", "BUY", 1.5, 98.0, 0.2)

        _, kwargs = mock_place.call_args
        self.assertNotIn("clientAlgoId", kwargs)

    def test_place_trailing_stop_loss_with_client_algo_id_passes_it_through(self):
        with patch.object(exchange, "place_algo_order") as mock_place, \
             patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v):
            exchange.place_trailing_stop_loss(
                "BTCUSDT", "BUY", 1.5, 98.0, 0.2, client_algo_id="dcaTrail123"
            )

        _, kwargs = mock_place.call_args
        self.assertEqual(kwargs["clientAlgoId"], "dcaTrail123")

    def test_place_trailing_stop_loss_rejects_invalid_activate_price(self):
        with patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: 0.0):
            with self.assertRaises(ValueError):
                exchange.place_trailing_stop_loss("BTCUSDT", "BUY", 1.5, 0.0, 0.2)

    def test_place_trailing_stop_loss_rejects_zero_quantity(self):
        with patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: 0.0), \
             patch.object(exchange, "normalize_trigger_price", side_effect=lambda s, side, t, p: p):
            with self.assertRaises(ValueError):
                exchange.place_trailing_stop_loss("BTCUSDT", "BUY", 0.0, 98.0, 0.2)


class NormalizeCallbackRateTests(unittest.TestCase):
    """config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - Binance requires
    0.1-10 for callbackRate; this project deliberately caps the top end
    at 5 (see exchange._normalize_callback_rate's own docstring)."""

    def test_clamps_below_binance_minimum_up_to_point_one(self):
        self.assertEqual(exchange._normalize_callback_rate(0.0), 0.1)

    def test_clamps_above_project_ceiling_down_to_five(self):
        self.assertEqual(exchange._normalize_callback_rate(10.0), 5.0)

    def test_passes_through_mid_range_value_rounded_to_one_decimal(self):
        self.assertEqual(exchange._normalize_callback_rate(0.234), 0.2)


class FormatPriceForApiTests(unittest.TestCase):
    """Real bug found live (NEIROUSDT, 2026-08-10, 100% reproducible):
    passing a raw Python float for a low-priced symbol's trigger/limit
    price serializes as scientific notation (e.g. 6.29e-05), which
    Binance's API rejects outright (-1102 malformed parameter). This
    formats a plain fixed-point decimal string instead."""

    def test_low_priced_symbol_never_produces_scientific_notation(self):
        with patch.object(
            exchange, "get_symbol_price_rules", return_value={"precision": 8}
        ):
            result = exchange._format_price_for_api("NEIROUSDT", 6.29e-05)

        self.assertNotIn("e", result.lower())
        self.assertEqual(result, "0.00006290")

    def test_uses_the_symbols_own_price_precision(self):
        with patch.object(
            exchange, "get_symbol_price_rules", return_value={"precision": 2}
        ):
            result = exchange._format_price_for_api("BTCUSDT", 100.5)

        self.assertEqual(result, "100.50")

    def test_negative_precision_is_clamped_to_zero(self):
        with patch.object(
            exchange, "get_symbol_price_rules", return_value={"precision": -1}
        ):
            result = exchange._format_price_for_api("BTCUSDT", 100.5)

        self.assertEqual(result, "100")

    def test_missing_precision_falls_back_to_eight_decimals(self):
        with patch.object(exchange, "get_symbol_price_rules", return_value={}):
            result = exchange._format_price_for_api("BTCUSDT", 100.5)

        self.assertEqual(result, "100.50000000")


class SetupLeverageCachingTests(unittest.TestCase):
    """Real bug found live (MTLUSDT/JOEUSDT, 2026-08-10): setup_leverage
    used to make a real futures_change_leverage call on every single
    entry attempt, burning private REST rate-limit budget for no reason
    and directly causing valid signals to die with "leverage Nx not
    available" whenever that redundant call landed during a backoff
    window."""

    def setUp(self):
        exchange._leverage_cache.clear()

    def tearDown(self):
        exchange._leverage_cache.clear()

    def test_second_call_for_the_same_symbol_skips_the_real_api_call(self):
        with patch.object(
            exchange.client, "futures_change_leverage",
            return_value={"leverage": config.LEVERAGE},
        ) as mock_call:
            exchange.setup_leverage("BTCUSDT")
            result = exchange.setup_leverage("BTCUSDT")

        mock_call.assert_called_once()
        self.assertTrue(result)

    def test_different_symbols_each_get_a_real_call(self):
        with patch.object(
            exchange.client, "futures_change_leverage",
            return_value={"leverage": config.LEVERAGE},
        ) as mock_call:
            exchange.setup_leverage("BTCUSDT")
            exchange.setup_leverage("ETHUSDT")

        self.assertEqual(mock_call.call_count, 2)

    def test_a_leverage_mismatch_is_not_cached_and_can_be_retried(self):
        with patch.object(
            exchange.client, "futures_change_leverage",
            return_value={"leverage": config.LEVERAGE + 1},
        ) as mock_call:
            first = exchange.setup_leverage("BTCUSDT")
            second = exchange.setup_leverage("BTCUSDT")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(mock_call.call_count, 2)

    def test_an_exception_is_not_cached_and_can_be_retried(self):
        with patch.object(
            exchange.client, "futures_change_leverage",
            side_effect=Exception("boom"),
        ) as mock_call:
            first = exchange.setup_leverage("BTCUSDT")
            second = exchange.setup_leverage("BTCUSDT")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(mock_call.call_count, 2)


def _exchange_info(symbol, lot_size_max, market_lot_size_max):
    return {
        "symbols": [{
            "symbol": symbol,
            "quantityPrecision": 3,
            "filters": [
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "1",
                    "minQty": "1",
                    "maxQty": str(lot_size_max),
                },
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "stepSize": "1",
                    "minQty": "1",
                    "maxQty": str(market_lot_size_max),
                },
            ],
        }]
    }


class QuantityRuleMaxFilterTests(unittest.TestCase):
    """Real bug found live (XAIUSDT, 2026-08-08): a TAKE_PROFIT_MARKET
    TP1 quantity that passed LOT_SIZE's max alone still got rejected by
    Binance with -4005 "Quantity greater than max quantity" - conditional
    orders that execute as MARKET once triggered are apparently validated
    against the tighter of the two filters, not LOT_SIZE alone."""

    def test_conditional_order_type_uses_the_tighter_of_both_filters(self):
        with patch.object(exchange, "get_exchange_info", return_value=_exchange_info("XAIUSDT", 900000, 300000)):
            rules = exchange.get_symbol_quantity_rules("XAIUSDT", order_type="CONDITIONAL")

        self.assertEqual(rules["max_qty"], "300000.0")

    def test_market_order_type_still_uses_market_lot_size(self):
        with patch.object(exchange, "get_exchange_info", return_value=_exchange_info("XAIUSDT", 900000, 300000)):
            rules = exchange.get_symbol_quantity_rules("XAIUSDT", order_type="MARKET")

        self.assertEqual(rules["max_qty"], "300000")

    def test_limit_order_type_uses_lot_size_only_not_the_tighter_of_both(self):
        # config.LIMIT_ENTRY_MODE_ENABLED - a plain resting LIMIT entry is
        # neither MARKET nor one of the algo/conditional types that
        # execute-as-MARKET-once-triggered, so it should validate against
        # LOT_SIZE alone, not the tighter-of-both clamp CONDITIONAL uses.
        with patch.object(exchange, "get_exchange_info", return_value=_exchange_info("XAIUSDT", 900000, 300000)):
            rules = exchange.get_symbol_quantity_rules("XAIUSDT", order_type="LIMIT")

        self.assertEqual(rules["max_qty"], "900000")

    def test_conditional_falls_back_to_lot_size_when_market_lot_size_missing(self):
        info = {
            "symbols": [{
                "symbol": "XAIUSDT",
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1", "maxQty": "900000"},
                ],
            }]
        }

        with patch.object(exchange, "get_exchange_info", return_value=info):
            rules = exchange.get_symbol_quantity_rules("XAIUSDT", order_type="CONDITIONAL")

        self.assertEqual(rules["max_qty"], "900000.0")


class RateLimitMinGapTests(unittest.TestCase):
    """Real bug found live (2026-08-11): the weight-budget check alone let
    an 80-symbol x 2-timeframe startup seed loop fire ~160 kline calls
    back-to-back - their summed weight fit comfortably under
    BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE, so that guard alone never
    blocked it, but Binance still hard-banned the IP (-1003). Its abuse
    detection is evidently sensitive to burst RATE, not just weight
    accounted for over a full minute. These lock in the added minimum-gap
    guard that's independent of weight - using small real sleeps (not
    mocked) so a bug that makes the retry loop spin forever would hang
    the test instead of silently passing."""

    def setUp(self):
        self._weights_backup = list(exchange._public_request_weights)
        self._last_at_backup = exchange._last_public_request_at
        exchange._public_request_weights.clear()
        exchange._last_public_request_at = 0.0

    def tearDown(self):
        exchange._public_request_weights.clear()
        exchange._public_request_weights.extend(self._weights_backup)
        exchange._last_public_request_at = self._last_at_backup

    def test_first_call_does_not_wait(self):
        with patch.object(config, "BINANCE_MIN_REQUEST_GAP_SECONDS", 0.05), \
             patch.object(config, "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800):
            start = time.perf_counter()
            exchange._rate_limit_public_request(weight=1)
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05)

    def test_rapid_second_call_is_paced_even_though_weight_budget_allows_it(self):
        with patch.object(config, "BINANCE_MIN_REQUEST_GAP_SECONDS", 0.05), \
             patch.object(config, "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800):
            exchange._rate_limit_public_request(weight=1)

            start = time.perf_counter()
            exchange._rate_limit_public_request(weight=1)  # weight 1+1=2, way under 1800
            elapsed = time.perf_counter() - start

        self.assertGreaterEqual(elapsed, 0.04)

    def test_zero_gap_disables_the_guard(self):
        with patch.object(config, "BINANCE_MIN_REQUEST_GAP_SECONDS", 0), \
             patch.object(config, "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800):
            exchange._rate_limit_public_request(weight=1)

            start = time.perf_counter()
            exchange._rate_limit_public_request(weight=1)
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05)


class PrivateRestWeightTests(unittest.TestCase):
    """Real bug found live (2026-08-11): every _private_rest_call site
    used to omit `weight`, so the pre-emptive throttle
    (_rate_limit_public_request) was never actually invoked for any
    account/order REST call - only the reactive backoff-after-error half
    worked, by which point Binance had already banned the IP (-1003 "Way
    too many requests"). These lock in that every private GET/query call
    site now passes a real, nonzero weight, and that order mutations
    (which Binance doesn't charge IP weight for) stay untouched."""

    def setUp(self):
        self.rate_limit_patcher = patch.object(exchange, "_rate_limit_public_request")
        self.rate_limit_mock = self.rate_limit_patcher.start()

    def tearDown(self):
        self.rate_limit_patcher.stop()

    def test_get_balance_is_weighted(self):
        with patch.object(
            exchange.client, "futures_account_balance",
            return_value=[{"asset": "USDT", "balance": "100"}],
        ):
            exchange.get_balance(force=True)

        self.rate_limit_mock.assert_called_once_with(5)

    def test_fetch_open_position_detail_is_weighted(self):
        with patch.object(exchange.client, "futures_position_information", return_value=[]):
            exchange._fetch_open_position_detail("BTCUSDT")

        self.rate_limit_mock.assert_called_once_with(5)

    def test_get_all_open_positions_is_weighted(self):
        with patch.object(exchange.client, "futures_position_information", return_value=[]):
            exchange.get_all_open_positions()

        self.rate_limit_mock.assert_called_once_with(5)

    def test_setup_leverage_is_weighted(self):
        # setup_leverage now caches per symbol (see SetupLeverageCachingTests)
        # - an empty cache here guarantees the real call actually happens,
        # regardless of what other tests in this run already cached.
        exchange._leverage_cache.clear()

        with patch.object(
            exchange.client, "futures_change_leverage",
            return_value={"leverage": config.LEVERAGE},
        ):
            exchange.setup_leverage("BTCUSDT")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_get_algo_order_status_is_weighted(self):
        with patch.object(
            exchange.client, "futures_get_algo_order", return_value={"algoStatus": "NEW"}
        ):
            exchange.get_algo_order_status("BTCUSDT", "algo1")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_get_open_algo_orders_is_weighted(self):
        with patch.object(exchange.client, "futures_get_open_algo_orders", return_value=[]):
            exchange.get_open_algo_orders("BTCUSDT")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_cancel_algo_order_is_weighted(self):
        with patch.object(exchange.client, "futures_cancel_algo_order", return_value={}):
            exchange.cancel_algo_order("BTCUSDT", "algo1")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_cancel_all_open_orders_is_weighted_for_both_calls(self):
        with patch.object(exchange.client, "futures_cancel_all_open_orders", return_value={}):
            exchange.cancel_all_open_orders("BTCUSDT")

        calls = [call.args for call in self.rate_limit_mock.call_args_list]
        self.assertEqual(calls, [(1,), (1,)])

    def test_income_history_is_weighted(self):
        with patch.object(exchange.client, "futures_income_history", return_value=[]):
            exchange.get_income_history()

        self.rate_limit_mock.assert_called_once_with(30)

    def test_place_algo_order_is_not_weighted(self):
        # Order mutations consume Binance's separate order-count limit,
        # not IP REQUEST_WEIGHT - must stay untouched, not newly throttled.
        with patch.object(exchange.client, "futures_create_algo_order", return_value={}):
            exchange.place_algo_order(symbol="BTCUSDT", side="SELL", type="STOP_MARKET")

        self.rate_limit_mock.assert_not_called()

    def test_place_limit_order_is_not_weighted(self):
        # config.LIMIT_ENTRY_MODE_ENABLED's entry order - same mutation
        # convention as place_market_order/place_algo_order above.
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=100.0), \
             patch.object(exchange.client, "futures_create_order", return_value={}):
            exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

        self.rate_limit_mock.assert_not_called()

    def test_get_order_status_is_weighted(self):
        with patch.object(exchange.client, "futures_get_order", return_value={"status": "NEW"}):
            exchange.get_order_status("BTCUSDT", "order1")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_cancel_order_is_weighted(self):
        with patch.object(exchange.client, "futures_cancel_order", return_value={}):
            exchange.cancel_order("BTCUSDT", "order1")

        self.rate_limit_mock.assert_called_once_with(1)

    def test_get_all_open_orders_is_weighted(self):
        with patch.object(exchange.client, "futures_get_open_orders", return_value=[]):
            exchange.get_all_open_orders()

        self.rate_limit_mock.assert_called_once_with(40)


class GetFundingRatesTests(unittest.TestCase):
    """Bulk (one call, every symbol) funding rate snapshot - reuses the
    same premiumIndex endpoint get_mark_price() already calls per-symbol,
    same pattern as get_24h_quote_volumes()."""

    def test_returns_a_symbol_to_rate_map(self):
        data = [
            {"symbol": "BTCUSDT", "markPrice": "100.0", "lastFundingRate": "0.0001"},
            {"symbol": "ETHUSDT", "markPrice": "50.0", "lastFundingRate": "-0.0002"},
        ]

        with patch.object(exchange.client, "futures_mark_price", return_value=data):
            result = exchange.get_funding_rates()

        self.assertEqual(result, {"BTCUSDT": 0.0001, "ETHUSDT": -0.0002})

    def test_error_returns_empty_dict_instead_of_raising(self):
        with patch.object(exchange.client, "futures_mark_price", side_effect=RuntimeError("boom")):
            result = exchange.get_funding_rates()

        self.assertEqual(result, {})


class GetLongShortRatioTests(unittest.TestCase):
    """On-demand (not polled across the watchlist - see
    config.LONG_SHORT_RATIO_ENABLED) global long/short account ratio."""

    def test_returns_the_latest_ratio(self):
        data = [
            {"symbol": "BTCUSDT", "longShortRatio": "1.5", "timestamp": 1000},
            {"symbol": "BTCUSDT", "longShortRatio": "1.8", "timestamp": 2000},
        ]

        with patch.object(exchange.client, "futures_global_longshort_ratio", return_value=data):
            result = exchange.get_long_short_ratio("BTCUSDT")

        self.assertEqual(result, 1.8)  # the last (most recent) entry, not the first

    def test_empty_response_returns_none(self):
        with patch.object(exchange.client, "futures_global_longshort_ratio", return_value=[]):
            result = exchange.get_long_short_ratio("BTCUSDT")

        self.assertIsNone(result)

    def test_error_returns_none_instead_of_raising(self):
        with patch.object(
            exchange.client, "futures_global_longshort_ratio", side_effect=RuntimeError("boom")
        ):
            result = exchange.get_long_short_ratio("BTCUSDT")

        self.assertIsNone(result)


class GetIncomeHistoryTests(unittest.TestCase):
    def test_returns_the_records_list(self):
        records = [{"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "1.5", "time": 1000}]

        with patch.object(exchange.client, "futures_income_history", return_value=records) as mock_call:
            result = exchange.get_income_history(start_time=500)

        self.assertEqual(result, records)
        _, kwargs = mock_call.call_args
        self.assertEqual(kwargs["startTime"], 500)
        self.assertNotIn("symbol", kwargs)
        self.assertNotIn("incomeType", kwargs)

    def test_passes_through_optional_filters(self):
        with patch.object(exchange.client, "futures_income_history", return_value=[]) as mock_call:
            exchange.get_income_history(
                symbol="BTCUSDT", income_type="REALIZED_PNL", start_time=1, end_time=2, limit=50
            )

        _, kwargs = mock_call.call_args
        self.assertEqual(kwargs["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["incomeType"], "REALIZED_PNL")
        self.assertEqual(kwargs["startTime"], 1)
        self.assertEqual(kwargs["endTime"], 2)
        self.assertEqual(kwargs["limit"], 50)

    def test_non_list_response_returns_empty_list(self):
        with patch.object(exchange.client, "futures_income_history", return_value={"unexpected": "shape"}):
            result = exchange.get_income_history()

        self.assertEqual(result, [])

    def test_error_returns_empty_list_instead_of_raising(self):
        with patch.object(exchange.client, "futures_income_history", side_effect=RuntimeError("boom")):
            result = exchange.get_income_history()

        self.assertEqual(result, [])


class GetSymbolMaxOrderQuantityTests(unittest.TestCase):
    def test_returns_the_tighter_max_across_market_and_conditional(self):
        with patch.object(exchange, "get_exchange_info", return_value=_exchange_info("XAIUSDT", 900000, 300000)):
            result = exchange.get_symbol_max_order_quantity("XAIUSDT")

        self.assertEqual(result, 300000.0)

    def test_returns_zero_when_symbol_unknown(self):
        with patch.object(exchange, "get_exchange_info", return_value={"symbols": []}):
            result = exchange.get_symbol_max_order_quantity("DOESNOTEXIST")

        self.assertEqual(result, 0.0)


class OpenInterestTests(unittest.TestCase):
    def setUp(self):
        exchange._oi_unavailable_symbols.clear()

    def tearDown(self):
        exchange._oi_unavailable_symbols.clear()

    def test_returns_open_interest_as_float(self):
        with patch.object(exchange.client, "futures_open_interest", return_value={"symbol": "BTCUSDT", "openInterest": "12345.67"}):
            result = exchange.get_open_interest("BTCUSDT")

        self.assertEqual(result, 12345.67)

    def test_returns_none_on_error_instead_of_raising(self):
        with patch.object(exchange.client, "futures_open_interest", side_effect=RuntimeError("boom")):
            result = exchange.get_open_interest("BTCUSDT")

        self.assertIsNone(result)


class OpenInterestUnavailableSymbolTests(unittest.TestCase):
    """Real bug found live (2026-08-08, WATCHING=519): a large watchlist
    includes symbols the OI endpoint permanently rejects (delisted/
    settling/pre-trading, or simply invalid) - unhandled, these got
    retried and logged on every single poll cycle forever."""

    def setUp(self):
        exchange._oi_unavailable_symbols.clear()

    def tearDown(self):
        exchange._oi_unavailable_symbols.clear()

    def test_delisted_symbol_error_marks_it_unavailable_and_skips_future_calls(self):
        error = Exception("APIError(code=-4108): Symbol is on delivering or delivered or settling or closed or pre-trading.")

        with patch.object(exchange.client, "futures_open_interest", side_effect=error) as mock_call:
            first = exchange.get_open_interest("ALLOUSDT")
            second = exchange.get_open_interest("ALLOUSDT")

        self.assertIsNone(first)
        self.assertIsNone(second)
        # Second call never hit the network - short-circuited by the cooldown.
        self.assertEqual(mock_call.call_count, 1)

    def test_invalid_symbol_error_also_triggers_the_cooldown(self):
        error = Exception("APIError(code=-1121): Invalid symbol.")

        with patch.object(exchange.client, "futures_open_interest", side_effect=error) as mock_call:
            exchange.get_open_interest("IRYSUSDT")
            exchange.get_open_interest("IRYSUSDT")

        self.assertEqual(mock_call.call_count, 1)

    def test_unavailable_symbol_is_retried_again_after_the_cooldown_expires(self):
        error = Exception("APIError(code=-4108): Symbol is on delivering or delivered or settling or closed or pre-trading.")

        # _rate_limit_public_request mocked here too: it stamps
        # _last_public_request_at from the SAME time.time() this test
        # fakes 61s into the future - left real, that poisons the shared
        # global rate-limit state for every other test that runs
        # afterward (real bug hit live while adding the min-gap guard:
        # the whole suite went from ~2s to 100s+ because of this).
        with patch.object(config, "OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS", 60), \
             patch.object(exchange.client, "futures_open_interest", side_effect=error) as mock_call, \
             patch.object(exchange, "_rate_limit_public_request"):
            exchange.get_open_interest("ALLOUSDT")

            with patch.object(exchange.time, "time", return_value=time.time() + 61):
                exchange.get_open_interest("ALLOUSDT")

        self.assertEqual(mock_call.call_count, 2)

    def test_unrelated_error_does_not_trigger_the_cooldown(self):
        with patch.object(exchange.client, "futures_open_interest", side_effect=RuntimeError("boom")) as mock_call:
            exchange.get_open_interest("BTCUSDT")
            exchange.get_open_interest("BTCUSDT")

        self.assertEqual(mock_call.call_count, 2)

    def test_a_symbol_that_recovers_clears_its_unavailable_marker(self):
        error = Exception("APIError(code=-4108): Symbol is on delivering or delivered or settling or closed or pre-trading.")

        with patch.object(config, "OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS", 60), \
             patch.object(exchange.client, "futures_open_interest", side_effect=error), \
             patch.object(exchange, "_rate_limit_public_request"):
            exchange.get_open_interest("ALLOUSDT")

        # Same fake-future-time.time() poisoning risk as the test above -
        # rate limiter mocked here too so it doesn't affect later tests.
        with patch.object(exchange.time, "time", return_value=time.time() + 61), \
             patch.object(exchange.client, "futures_open_interest", return_value={"symbol": "ALLOUSDT", "openInterest": "500"}), \
             patch.object(exchange, "_rate_limit_public_request"):
            result = exchange.get_open_interest("ALLOUSDT")

        self.assertEqual(result, 500.0)
        self.assertNotIn("ALLOUSDT", exchange._oi_unavailable_symbols)


class CancelAllOpenOrdersTests(unittest.TestCase):
    """Binance treats regular orders and algo/conditional orders (which is
    what our SL/TP1/TP2 actually are) as two separate cancel-all
    endpoints - calling only one silently leaves the other kind in place."""

    def test_calls_both_the_regular_and_conditional_cancel_all_endpoints(self):
        with patch.object(exchange.client, "futures_cancel_all_open_orders") as mock_cancel:
            exchange.cancel_all_open_orders("BTCUSDT")

        self.assertEqual(mock_cancel.call_count, 2)
        calls = mock_cancel.call_args_list
        self.assertEqual(calls[0].kwargs, {"symbol": "BTCUSDT"})
        self.assertEqual(calls[1].kwargs, {"symbol": "BTCUSDT", "conditional": True})

    def test_a_failure_on_the_regular_endpoint_does_not_skip_the_conditional_one(self):
        with patch.object(
            exchange.client,
            "futures_cancel_all_open_orders",
            side_effect=[Exception("regular failed"), {"ok": True}],
        ) as mock_cancel:
            exchange.cancel_all_open_orders("BTCUSDT")

        self.assertEqual(mock_cancel.call_count, 2)

    def test_never_raises_even_if_both_calls_fail(self):
        with patch.object(
            exchange.client,
            "futures_cancel_all_open_orders",
            side_effect=Exception("boom"),
        ):
            exchange.cancel_all_open_orders("BTCUSDT")  # must not raise


class ClosePositionMarketTests(unittest.TestCase):
    def test_closing_a_buy_position_sends_a_sell_reduce_only_order(self):
        with patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange.client, "futures_create_order") as mock_order:
            exchange.close_position_market("BTCUSDT", "BUY", 0.5)

        _, kwargs = mock_order.call_args
        self.assertEqual(kwargs["side"], "SELL")
        self.assertEqual(kwargs["type"], "MARKET")
        self.assertEqual(kwargs["quantity"], 0.5)
        self.assertEqual(kwargs["reduceOnly"], "true")

    def test_closing_a_sell_position_sends_a_buy_reduce_only_order(self):
        with patch.object(exchange, "normalize_order_quantity", side_effect=lambda s, q, order_type=None: q), \
             patch.object(exchange.client, "futures_create_order") as mock_order:
            exchange.close_position_market("BTCUSDT", "SELL", 0.5)

        _, kwargs = mock_order.call_args
        self.assertEqual(kwargs["side"], "BUY")

    def test_zero_normalized_quantity_raises(self):
        with patch.object(exchange, "normalize_order_quantity", return_value=0.0):
            with self.assertRaises(ValueError):
                exchange.close_position_market("BTCUSDT", "BUY", 0.5)


class LimitOrderTests(unittest.TestCase):
    """config.LIMIT_ENTRY_MODE_ENABLED's entry order - a plain GTC LIMIT
    on the regular order endpoint (python-binance does not auto-route
    "LIMIT" to the algo namespace the way it does STOP_MARKET/
    TAKE_PROFIT_MARKET)."""

    def test_sends_limit_type_and_gtc(self):
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=100.0), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v), \
             patch.object(exchange.client, "futures_create_order", return_value={}) as mock_order:
            exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

        _, kwargs = mock_order.call_args
        self.assertEqual(kwargs["type"], "LIMIT")
        self.assertEqual(kwargs["timeInForce"], "GTC")
        self.assertEqual(kwargs["price"], 100.0)
        self.assertEqual(kwargs["quantity"], 1.0)

    def test_price_is_normalized_via_normalize_order_price_not_trigger_price(self):
        # Deliberately not normalize_trigger_price - a resting limit has
        # no "must not fire immediately" constraint, unlike a stop/TP.
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=99.99) as normalize_price, \
             patch.object(exchange, "normalize_trigger_price") as normalize_trigger, \
             patch.object(exchange.client, "futures_create_order", return_value={}):
            exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

        normalize_price.assert_called_once_with("BTCUSDT", 100.0, rounding="nearest")
        normalize_trigger.assert_not_called()

    def test_zero_normalized_quantity_raises(self):
        with patch.object(exchange, "normalize_order_quantity", return_value=0.0), \
             patch.object(exchange, "normalize_order_price", return_value=100.0):
            with self.assertRaises(ValueError):
                exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

    def test_zero_normalized_price_raises(self):
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=0.0):
            with self.assertRaises(ValueError):
                exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

    # config.DCA_RESTING_ORDER_ENABLED - client_order_id tags the real
    # order so position_manager.reconcile_pending_entries_on_startup can
    # tell a legitimate DCA-add order apart from a genuinely orphaned
    # pending-entry LIMIT order on restart.

    def test_client_order_id_passed_through_when_given(self):
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=100.0), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v), \
             patch.object(exchange.client, "futures_create_order", return_value={}) as mock_order:
            exchange.place_limit_order(
                "BTCUSDT", "BUY", 1.0, 100.0, client_order_id="dcaAdd12345",
            )

        _, kwargs = mock_order.call_args
        self.assertEqual(kwargs["newClientOrderId"], "dcaAdd12345")

    def test_client_order_id_omitted_when_not_given(self):
        # Existing callers (LIMIT_ENTRY_MODE_ENABLED/RETRACEMENT_ENTRY_
        # ENABLED) never pass this - must stay byte-identical for them.
        with patch.object(exchange, "normalize_order_quantity", return_value=1.0), \
             patch.object(exchange, "normalize_order_price", return_value=100.0), \
             patch.object(exchange, "_format_price_for_api", side_effect=lambda s, v: v), \
             patch.object(exchange.client, "futures_create_order", return_value={}) as mock_order:
            exchange.place_limit_order("BTCUSDT", "BUY", 1.0, 100.0)

        _, kwargs = mock_order.call_args
        self.assertNotIn("newClientOrderId", kwargs)


class GetOrderStatusTests(unittest.TestCase):
    """Ground truth for a plain (non-algo) order - config.LIMIT_ENTRY_MODE_ENABLED's
    entry needs executed_qty/avg_price (not just status), since a LIMIT
    order can partially fill."""

    def test_maps_filled_status_and_fill_data(self):
        order = {"status": "FILLED", "executedQty": "1.0", "avgPrice": "100.5", "origQty": "1.0"}

        with patch.object(exchange.client, "futures_get_order", return_value=order):
            result = exchange.get_order_status("BTCUSDT", "order1")

        self.assertEqual(result, {
            "status": "FILLED", "executed_qty": 1.0, "avg_price": 100.5, "orig_qty": 1.0,
        })

    def test_maps_partially_filled_status(self):
        order = {"status": "PARTIALLY_FILLED", "executedQty": "0.4", "avgPrice": "100.0", "origQty": "1.0"}

        with patch.object(exchange.client, "futures_get_order", return_value=order):
            result = exchange.get_order_status("BTCUSDT", "order1")

        self.assertEqual(result["status"], "PARTIALLY_FILLED")
        self.assertEqual(result["executed_qty"], 0.4)

    def test_order_does_not_exist_returns_not_found_sentinel(self):
        with patch.object(
            exchange.client, "futures_get_order",
            side_effect=Exception("Order does not exist."),
        ):
            result = exchange.get_order_status("BTCUSDT", "order1")

        self.assertEqual(result["status"], "NOT_FOUND")

    def test_other_errors_return_unknown_and_do_not_raise(self):
        with patch.object(
            exchange.client, "futures_get_order", side_effect=Exception("timeout"),
        ):
            result = exchange.get_order_status("BTCUSDT", "order1")

        self.assertEqual(result["status"], "UNKNOWN")


class ResolveMarketFillPriceTests(unittest.TestCase):
    """The real average fill price for a just-placed MARKET order - see
    position_manager._resolve_real_entry, the caller this exists for."""

    def test_avg_price_already_in_the_create_order_response_is_used_directly(self):
        order = {"orderId": "o1", "avgPrice": "100.5"}

        with patch.object(exchange, "get_order_status") as status:
            result = exchange.resolve_market_fill_price("BTCUSDT", order, 100.0)

        self.assertEqual(result, 100.5)
        status.assert_not_called()

    def test_zero_avg_price_falls_back_to_a_fresh_status_lookup(self):
        order = {"orderId": "o1", "avgPrice": "0"}

        with patch.object(
            exchange, "get_order_status",
            return_value={"status": "FILLED", "executed_qty": 1.0, "avg_price": 100.7, "orig_qty": 1.0},
        ) as status:
            result = exchange.resolve_market_fill_price("BTCUSDT", order, 100.0)

        self.assertEqual(result, 100.7)
        status.assert_called_once_with("BTCUSDT", "o1")

    def test_missing_avg_price_key_falls_back_to_a_fresh_status_lookup(self):
        order = {"orderId": "o1"}

        with patch.object(
            exchange, "get_order_status",
            return_value={"status": "FILLED", "executed_qty": 1.0, "avg_price": 100.7, "orig_qty": 1.0},
        ):
            result = exchange.resolve_market_fill_price("BTCUSDT", order, 100.0)

        self.assertEqual(result, 100.7)

    def test_status_lookup_also_empty_falls_back_to_the_planned_price(self):
        order = {"orderId": "o1", "avgPrice": "0"}

        with patch.object(
            exchange, "get_order_status",
            return_value={"status": "UNKNOWN", "executed_qty": 0.0, "avg_price": 0.0, "orig_qty": 0.0},
        ):
            result = exchange.resolve_market_fill_price("BTCUSDT", order, 100.0)

        self.assertEqual(result, 100.0)

    def test_no_order_id_falls_back_to_the_planned_price_without_a_lookup(self):
        order = {"avgPrice": "0"}

        with patch.object(exchange, "get_order_status") as status:
            result = exchange.resolve_market_fill_price("BTCUSDT", order, 100.0)

        self.assertEqual(result, 100.0)
        status.assert_not_called()

    def test_none_order_falls_back_to_the_planned_price(self):
        result = exchange.resolve_market_fill_price("BTCUSDT", None, 100.0)
        self.assertEqual(result, 100.0)


class CancelOrderTests(unittest.TestCase):
    def test_calls_the_plain_cancel_endpoint(self):
        with patch.object(exchange.client, "futures_cancel_order", return_value={"status": "CANCELED"}) as mock_cancel:
            exchange.cancel_order("BTCUSDT", "order1")

        _, kwargs = mock_cancel.call_args
        self.assertEqual(kwargs["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["orderId"], "order1")

    def test_never_raises_on_failure(self):
        with patch.object(exchange.client, "futures_cancel_order", side_effect=Exception("boom")):
            result = exchange.cancel_order("BTCUSDT", "order1")  # must not raise

        self.assertIsNone(result)


class GetAllOpenOrdersTests(unittest.TestCase):
    """Non-algo analog of get_all_open_positions() -
    PositionManager.reconcile_pending_entries_on_startup's startup-only
    call to find any resting LIMIT entry left over from before a restart."""

    def test_returns_the_order_list(self):
        orders = [{"symbol": "BTCUSDT", "orderId": "1", "type": "LIMIT"}]

        with patch.object(exchange.client, "futures_get_open_orders", return_value=orders):
            result = exchange.get_all_open_orders()

        self.assertEqual(result, orders)

    def test_error_returns_an_empty_list_not_a_raise(self):
        with patch.object(exchange.client, "futures_get_open_orders", side_effect=Exception("boom")):
            result = exchange.get_all_open_orders()  # must not raise

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
