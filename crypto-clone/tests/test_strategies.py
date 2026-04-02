from __future__ import annotations

import unittest

from trading.strategies import MACDStrategy, RSIMeanReversionStrategy, build_strategy


class StrategyTests(unittest.TestCase):
    def test_build_strategy_returns_named_strategy(self) -> None:
        strategy = build_strategy("macd")
        self.assertEqual(strategy.name, "macd")

    def test_macd_strategy_clamps_signal(self) -> None:
        strategy = MACDStrategy()
        signal = strategy.signal({"row": {"macd_signal": 1000.0}})
        self.assertLessEqual(signal, 1.0)
        self.assertGreaterEqual(signal, -1.0)

    def test_rsi_mean_reversion_goes_long_when_oversold(self) -> None:
        strategy = RSIMeanReversionStrategy()
        signal = strategy.signal({"row": {"rsi_30": 20.0, "norm_close": -1.0}})
        self.assertGreater(signal, 0.0)


if __name__ == "__main__":
    unittest.main()
