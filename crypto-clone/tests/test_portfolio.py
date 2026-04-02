from __future__ import annotations

import unittest

from trading.portfolio import PortfolioConstraints, TargetAllocator


class TargetAllocatorTests(unittest.TestCase):
    def test_allocator_respects_cash_buffer_when_buying(self) -> None:
        allocator = TargetAllocator(
            PortfolioConstraints(
                max_symbol_weight=0.50,
                cash_buffer_pct=0.10,
                min_order_notional=25.0,
                allow_short=False,
            )
        )
        decision = allocator.allocate(
            symbol="BTC-USD",
            signal=1.0,
            price=100.0,
            account_equity=1_000.0,
            available_cash=200.0,
            current_quantity=0.0,
        )
        self.assertEqual(decision.side, "buy")
        self.assertAlmostEqual(decision.quantity, 1.0)

    def test_allocator_ignores_short_signal_when_shorts_disabled(self) -> None:
        allocator = TargetAllocator(
            PortfolioConstraints(
                max_symbol_weight=0.30,
                cash_buffer_pct=0.10,
                min_order_notional=25.0,
                allow_short=False,
            )
        )
        decision = allocator.allocate(
            symbol="ETH-USD",
            signal=-1.0,
            price=200.0,
            account_equity=2_000.0,
            available_cash=500.0,
            current_quantity=0.0,
        )
        self.assertIsNone(decision.side)
        self.assertEqual(decision.target_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
