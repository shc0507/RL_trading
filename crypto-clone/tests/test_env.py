from __future__ import annotations

import unittest

import pandas as pd

from trading.env import EnvironmentConfig, TradingEnv
from trading.features import FeatureBuilder


def make_flat_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=400)
    price = pd.Series([100.0] * len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "asset_class": "equity_etf",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 1000,
            "source": "test",
            "currency": "USD",
        }
    )


class TradingEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        features = FeatureBuilder().transform(make_flat_bars())
        self.env = TradingEnv(
            feature_frame=features,
            config=EnvironmentConfig(action_mode="continuous", reward_mode="raw", cost_rate_bp=0.0),
            splits={"train": ("2020-01-01", "2022-12-31")},
        )

    def test_flat_series_yields_zero_reward_without_costs(self) -> None:
        observation = self.env.reset(symbol="AAA", split="train")
        observation, reward, done, info = self.env.step(1.0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["raw_return"], 0.0)
        self.assertFalse(done)

    def test_unchanged_position_has_zero_turnover(self) -> None:
        self.env.reset(symbol="AAA", split="train")
        self.env.step(1.0)
        _, _, _, info = self.env.step(1.0)
        self.assertEqual(info["turnover"], 0.0)

    def test_flip_position_doubles_turnover(self) -> None:
        self.env.reset(symbol="AAA", split="train")
        self.env.step(1.0)
        _, _, _, info = self.env.step(-1.0)
        self.assertEqual(info["turnover"], 2.0)


if __name__ == "__main__":
    unittest.main()
