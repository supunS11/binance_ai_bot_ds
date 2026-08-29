import unittest

import cross_exchange_liquidation as cxl


class FromOkxSymbolTests(unittest.TestCase):
    def test_strips_the_usdt_swap_suffix(self):
        self.assertEqual(cxl._from_okx_symbol("BTC-USDT-SWAP"), "BTCUSDT")
        self.assertEqual(cxl._from_okx_symbol("XPL-USDT-SWAP"), "XPLUSDT")

    def test_non_usdt_swap_suffix_returns_none(self):
        self.assertIsNone(cxl._from_okx_symbol("BTC-USD-SWAP"))
        self.assertIsNone(cxl._from_okx_symbol("BTCUSDT"))

    def test_missing_or_non_string_input_returns_none(self):
        self.assertIsNone(cxl._from_okx_symbol(None))
        self.assertIsNone(cxl._from_okx_symbol(123))

    def test_empty_base_returns_none(self):
        self.assertIsNone(cxl._from_okx_symbol("-USDT-SWAP"))


class BybitStreamNamesAndSubscribeFrameTests(unittest.TestCase):
    def test_stream_names_are_uppercased_and_prefixed(self):
        names = cxl.bybit_liquidation_stream_names(["btcusdt", "ETHUSDT"])
        self.assertEqual(names, ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"])

    def test_subscribe_frame_shape(self):
        frame = cxl.bybit_subscribe_frame(["BTCUSDT"])
        self.assertEqual(frame, {"op": "subscribe", "args": ["allLiquidation.BTCUSDT"]})


class BybitParseRejectedSymbolTests(unittest.TestCase):
    """Real live finding (2026-08-29): Bybit rejects the ENTIRE subscribe
    batch if even one symbol has no liquidation handler - this is the
    parser for that rejection reply, feeding ws_client's strip-and-retry
    loop."""

    def test_extracts_the_rejected_symbol_from_a_real_reply_shape(self):
        reply = {
            "success": False,
            "ret_msg": "error:handler not found,topic:allLiquidation.1000000BOBUSDT",
            "op": "subscribe",
        }
        self.assertEqual(cxl.bybit_parse_rejected_symbol(reply), "1000000BOBUSDT")

    def test_successful_reply_returns_none(self):
        reply = {"success": True, "ret_msg": "", "op": "subscribe"}
        self.assertIsNone(cxl.bybit_parse_rejected_symbol(reply))

    def test_failure_with_unrecognized_message_shape_returns_none(self):
        reply = {"success": False, "ret_msg": "some other error"}
        self.assertIsNone(cxl.bybit_parse_rejected_symbol(reply))

    def test_non_dict_input_returns_none(self):
        self.assertIsNone(cxl.bybit_parse_rejected_symbol(None))
        self.assertIsNone(cxl.bybit_parse_rejected_symbol("not a dict"))


class ParseBybitLiquidationTests(unittest.TestCase):
    """Real payload shape confirmed live (2026-08-29):
    {"topic":"allLiquidation.SYMBOL","type":"snapshot","ts":...,
    "data":[{"T":ms,"s":"SYMBOL","S":"Buy"|"Sell","v":"size","p":"price"}]}"""

    def _message(self, side="Sell", price="0.04499", size="20000", symbol="ROSEUSDT", ts=1739502302929):
        return {
            "topic": f"allLiquidation.{symbol}",
            "type": "snapshot",
            "ts": ts,
            "data": [{"T": ts, "s": symbol, "S": side, "v": size, "p": price}],
        }

    def test_sell_side_maps_to_sell_long_liquidation(self):
        result = cxl.parse_bybit_liquidation(self._message(side="Sell"))
        symbol, side, notional, timestamp = result
        self.assertEqual(symbol, "ROSEUSDT")
        self.assertEqual(side, "SELL")
        self.assertAlmostEqual(notional, 0.04499 * 20000)
        self.assertAlmostEqual(timestamp, 1739502302.929)

    def test_buy_side_maps_to_buy_short_liquidation(self):
        result = cxl.parse_bybit_liquidation(self._message(side="Buy"))
        self.assertEqual(result[1], "BUY")

    def test_non_liquidation_topic_returns_none(self):
        self.assertIsNone(cxl.parse_bybit_liquidation({"op": "subscribe", "success": True}))

    def test_missing_data_returns_none(self):
        self.assertIsNone(cxl.parse_bybit_liquidation({"topic": "allLiquidation.BTCUSDT", "data": []}))

    def test_malformed_message_never_raises(self):
        self.assertIsNone(cxl.parse_bybit_liquidation(None))
        self.assertIsNone(cxl.parse_bybit_liquidation("garbage"))
        self.assertIsNone(cxl.parse_bybit_liquidation({"topic": "allLiquidation.X", "data": "not a list"}))
        self.assertIsNone(cxl.parse_bybit_liquidation({"topic": "allLiquidation.X", "data": [None]}))


class OkxSubscribeFrameTests(unittest.TestCase):
    def test_frame_covers_the_whole_swap_market_in_one_subscription(self):
        frame = cxl.okx_subscribe_frame()
        self.assertEqual(
            frame,
            {"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]},
        )


class ParseOkxLiquidationTests(unittest.TestCase):
    """Real payload shape confirmed live (2026-08-29):
    {"arg":{...},"data":[{"instId":"XPL-USDT-SWAP","details":[{"side":
    "sell","bkPx":"0.0825","sz":"467","ts":"1787990018113", ...}], ...}]}"""

    def _message(self, side="sell", bk_px="0.0825", sz="467", inst_id="XPL-USDT-SWAP", ts="1787990018113"):
        return {
            "arg": {"channel": "liquidation-orders", "instType": "SWAP"},
            "data": [{
                "instId": inst_id,
                "instType": "SWAP",
                "instFamily": inst_id.rsplit("-", 1)[0],
                "uly": inst_id.rsplit("-", 1)[0],
                "details": [{
                    "side": side, "bkPx": bk_px, "sz": sz, "ts": ts,
                    "bkLoss": "0", "ccy": "", "posSide": "long",
                }],
            }],
        }

    def test_parses_a_single_liquidation_and_reverse_maps_the_symbol(self):
        result = cxl.parse_okx_liquidation(self._message())
        self.assertEqual(len(result), 1)
        symbol, side, notional, timestamp = result[0]
        self.assertEqual(symbol, "XPLUSDT")
        self.assertEqual(side, "SELL")
        self.assertAlmostEqual(notional, 0.0825 * 467)
        self.assertAlmostEqual(timestamp, 1787990018.113)

    def test_buy_side_is_uppercased_to_buy(self):
        result = cxl.parse_okx_liquidation(self._message(side="buy"))
        self.assertEqual(result[0][1], "BUY")

    def test_multiple_instruments_and_multiple_details_all_parsed(self):
        message = {
            "arg": {"channel": "liquidation-orders", "instType": "SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "details": [
                        {"side": "sell", "bkPx": "50000", "sz": "1", "ts": "1000"},
                        {"side": "buy", "bkPx": "50100", "sz": "2", "ts": "1001"},
                    ],
                },
                {
                    "instId": "ETH-USDT-SWAP",
                    "details": [{"side": "sell", "bkPx": "3000", "sz": "5", "ts": "1002"}],
                },
            ],
        }
        result = cxl.parse_okx_liquidation(message)
        self.assertEqual(len(result), 3)
        self.assertEqual([r[0] for r in result], ["BTCUSDT", "BTCUSDT", "ETHUSDT"])

    def test_unmappable_inst_id_is_skipped_not_raised(self):
        result = cxl.parse_okx_liquidation(self._message(inst_id="BTC-USD-SWAP"))
        self.assertEqual(result, [])

    def test_malformed_message_never_raises_and_returns_empty_list(self):
        self.assertEqual(cxl.parse_okx_liquidation(None), [])
        self.assertEqual(cxl.parse_okx_liquidation("garbage"), [])
        self.assertEqual(cxl.parse_okx_liquidation({"data": "not a list"}), [])
        self.assertEqual(cxl.parse_okx_liquidation({"data": [None]}), [])
        self.assertEqual(cxl.parse_okx_liquidation({"data": [{"instId": "BTC-USDT-SWAP", "details": "not a list"}]}), [])
        self.assertEqual(cxl.parse_okx_liquidation({"data": [{"instId": "BTC-USDT-SWAP", "details": [None]}]}), [])


if __name__ == "__main__":
    unittest.main()
