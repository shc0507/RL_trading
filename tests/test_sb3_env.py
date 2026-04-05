from __future__ import annotations

import unittest

import numpy as np

from rl_trading.features import FeatureBuilder
from rl_trading.sb3_env import SB3TradingEnv, encode_observation

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports direct module execution
    from tests.test_data_pipeline import make_bar_frame


class SB3EnvTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=900)
        self.features = FeatureBuilder().transform(bars)
        self.splits = {
            "train": ("2021-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-06-30"),
        }

    def test_encode_observation_has_fixed_size(self) -> None:
        env = SB3TradingEnv(
            feature_frame=self.features,
            split="train",
            symbols=["AAA"],
            action_mode="discrete",
            splits=self.splits,
            shuffle_symbols=False,
        )
        obs, info = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["symbol"], "AAA")

    def test_discrete_step_returns_vector_reward_and_done_flags(self) -> None:
        env = SB3TradingEnv(
            feature_frame=self.features,
            split="train",
            symbols=["AAA"],
            action_mode="discrete",
            splits=self.splits,
            shuffle_symbols=False,
        )
        obs, _ = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(2)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertFalse(truncated)
        self.assertIn("action", info)
        self.assertTrue(np.isfinite(reward))


if __name__ == "__main__":
    unittest.main()
