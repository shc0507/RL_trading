from __future__ import annotations

import unittest

from trading.brokers import OrderRequest, PaperBroker
from trading.config import DEFAULT_CRYPTO_INSTRUMENT_MAP


class PaperBrokerTests(unittest.TestCase):
    def test_buy_then_sell_updates_balances(self) -> None:
        broker = PaperBroker(
            initial_cash=2_000.0,
            fee_rate_bp=10.0,
            instrument_map=DEFAULT_CRYPTO_INSTRUMENT_MAP,
        )

        buy = broker.place_market_order(OrderRequest(symbol="BTC-USD", side="buy", quantity=0.01), price_hint=50_000.0)
        self.assertEqual(buy.status, "filled")

        snapshot = broker.get_account_snapshot({"BTC-USD": 50_000.0})
        self.assertLess(snapshot.available_cash, 2_000.0)
        self.assertGreater(snapshot.equity, 0.0)

        sell = broker.place_market_order(OrderRequest(symbol="BTC-USD", side="sell", quantity=0.005), price_hint=60_000.0)
        self.assertEqual(sell.status, "filled")
        self.assertGreater(broker.balances["USD"], 0.0)


if __name__ == "__main__":
    unittest.main()
