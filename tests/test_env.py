from __future__ import annotations

import math
import unittest

import pandas as pd

from rl_trading.config import ANNUALIZATION_FACTOR
from rl_trading.env import EnvironmentConfig, TradingEnv
from rl_trading.features import FeatureBuilder


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

    def test_trade_return_matches_additive_formula(self) -> None:
        bars = make_flat_bars().copy()
        bars.loc[bars.index[-3:], "adj_close"] = [100.0, 101.5, 99.0]
        bars.loc[bars.index[-3:], "close"] = bars.loc[bars.index[-3:], "adj_close"]
        features = FeatureBuilder().transform(bars)
        env = TradingEnv(
            feature_frame=features,
            config=EnvironmentConfig(action_mode="continuous", reward_mode="zhang", cost_rate_bp=20.0),
            splits={"train": ("2020-01-01", "2022-12-31")},
        )
        env.reset(symbol="AAA", split="train")
        _, _, _, info = env.step(1.0)
        expected_pre_cost = info["scaled_position"] * (info["price_next"] - info["price_now"])
        expected_trade_cost = (20.0 / 10_000.0) * info["price_now"] * info["scaled_turnover"]
        expected_trade_return = expected_pre_cost - expected_trade_cost
        self.assertAlmostEqual(info["pre_cost_trade_return"], expected_pre_cost)
        self.assertAlmostEqual(info["trade_cost"], expected_trade_cost)
        self.assertAlmostEqual(info["trade_return"], expected_trade_return)

    def test_zhang_scaling_uses_daily_target_vol(self) -> None:
        bars = make_flat_bars().copy()
        prices = [100.0 + (0.04 * index) + (0.8 * math.sin(index / 9.0)) for index in range(len(bars))]
        bars["adj_close"] = prices
        bars["close"] = prices
        features = FeatureBuilder().transform(bars)
        env = TradingEnv(
            feature_frame=features,
            config=EnvironmentConfig(action_mode="continuous", reward_mode="zhang", cost_rate_bp=20.0, vol_target=0.15),
            splits={"train": ("2020-01-01", "2022-12-31")},
        )
        env.reset(symbol="AAA", split="train")
        _, _, _, info = env.step(1.0)

        current_vol = float(
            features.loc[
                (features["symbol"] == "AAA")
                & features["window_ready"]
                & (features["date"] == info["date"]),
                "ewm_vol_60",
            ].iloc[0]
        )
        expected_daily_target = 0.15 / math.sqrt(ANNUALIZATION_FACTOR)
        self.assertAlmostEqual(info["scaled_position"], expected_daily_target / current_vol)


if __name__ == "__main__":
    unittest.main()
