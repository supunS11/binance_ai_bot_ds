import unittest
from unittest.mock import patch

import config
import risk_management


class CalculatePositionSizeTests(unittest.TestCase):
    def test_flat_sizing_when_risk_based_sizing_disabled(self):
        with patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", False), \
             patch.object(config, "MARGIN_PER_TRADE", 10), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_management, "get_symbol_precision", return_value=3), \
             patch.object(risk_management, "get_symbol_max_order_quantity", return_value=0):
            quantity = risk_management.calculate_position_size(1000, 100.0, 98.0, "BTCUSDT")

        # margin(10) * leverage(10) / entry(100) = 1.0
        self.assertAlmostEqual(quantity, 1.0)

    def test_risk_based_sizing_uses_stop_distance(self):
        with patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(config, "POSITION_RISK_PCT", 1.0), \
             patch.object(config, "POSITION_RISK_MAX_USDT", 0), \
             patch.object(config, "MARGIN_PER_TRADE", 1000), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_management, "get_symbol_precision", return_value=3), \
             patch.object(risk_management, "get_symbol_max_order_quantity", return_value=0):
            # balance=1000, risk 1% -> $10 risk budget, stop distance $2 -> qty=5
            quantity = risk_management.calculate_position_size(1000, 100.0, 98.0, "BTCUSDT")

        self.assertAlmostEqual(quantity, 5.0)

    def test_quantity_clamped_to_real_exchange_max_order_quantity(self):
        """Root cause of a real live bug (XAIUSDT, 2026-08-08): risk-based
        sizing on a low-priced symbol can compute a quantity that clears
        the margin-based sanity cap comfortably but still exceeds
        Binance's own max order quantity - leaving it unclamped here meant
        the entry silently got cut down to that same limit at placement
        time while tp1_quantity/tp2_quantity (exact fractions of this
        function's returned, still-oversized quantity) stayed too large
        and got rejected later with -4005."""
        with patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", True), \
             patch.object(config, "POSITION_RISK_PCT", 100.0), \
             patch.object(config, "POSITION_RISK_MAX_USDT", 0), \
             patch.object(config, "MARGIN_PER_TRADE", 1000), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_management, "get_symbol_precision", return_value=0), \
             patch.object(risk_management, "get_symbol_max_order_quantity", return_value=500):
            # balance=100000, risk 100% -> $100000 budget, stop distance
            # $0.01 -> naive qty = 10,000,000, far past the exchange max.
            quantity = risk_management.calculate_position_size(100000, 1.0, 0.99, "XAIUSDT")

        self.assertEqual(quantity, 500)

    def test_zero_exchange_max_quantity_means_no_extra_clamp(self):
        """get_symbol_max_order_quantity returns 0 when neither filter is
        available (e.g. a lookup failure) - that must not zero out sizing
        entirely, just skip this particular cap."""
        with patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", False), \
             patch.object(config, "MARGIN_PER_TRADE", 10), \
             patch.object(config, "LEVERAGE", 10), \
             patch.object(risk_management, "get_symbol_precision", return_value=3), \
             patch.object(risk_management, "get_symbol_max_order_quantity", return_value=0):
            quantity = risk_management.calculate_position_size(1000, 100.0, 98.0, "BTCUSDT")

        self.assertAlmostEqual(quantity, 1.0)

    def test_below_min_notional_returns_zero(self):
        with patch.object(config, "RISK_BASED_POSITION_SIZING_ENABLED", False), \
             patch.object(config, "MARGIN_PER_TRADE", 0.01), \
             patch.object(config, "LEVERAGE", 1), \
             patch.object(risk_management, "get_symbol_precision", return_value=3), \
             patch.object(risk_management, "get_symbol_max_order_quantity", return_value=0):
            quantity = risk_management.calculate_position_size(1000, 100.0, 98.0, "BTCUSDT")

        self.assertEqual(quantity, 0)


if __name__ == "__main__":
    unittest.main()
