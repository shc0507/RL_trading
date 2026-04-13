from __future__ import annotations

import unittest

import numpy as np

from rl_trading.env import EnvironmentConfig
from rl_trading.features import FeatureBuilder
from rl_trading.rl.env_adapter import ContinuousTradingEnv, DiscreteTradingEnv

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports module-style unittest invocation
    from tests.test_data_pipeline import make_bar_frame


class DiscreteTradingEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=500)
        features = FeatureBuilder().transform(bars)
        self.env = DiscreteTradingEnv(
            feature_frame=features,
            config=EnvironmentConfig(action_mode="discrete", reward_mode="raw", cost_rate_bp=0.0),
            splits={"train": ("2021-06-01", "2021-12-31")},
        )

    def test_reset_emits_fixed_size_state(self) -> None:
        state, info = self.env.reset(symbol="AAA", split="train")
        self.assertEqual(state.shape, (self.env.state_size,))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(info["symbol"], "AAA")

    def test_action_ids_map_to_positions(self) -> None:
        self.env.reset(symbol="AAA", split="train")
        _, _, _, _, info = self.env.step(2)
        self.assertEqual(info["action_id"], 2)
        self.assertEqual(info["target_position"], 1.0)

    def test_terminal_step_returns_zero_state(self) -> None:
        self.env.reset(symbol="AAA", split="train")
        done = False
        state = None
        while not done:
            state, _, done, truncated, _ = self.env.step(1)
            self.assertFalse(truncated)
        self.assertIsNotNone(state)
        self.assertTrue(np.allclose(state, 0.0))


class ContinuousTradingEnvTests(unittest.TestCase):
    def test_continuous_adapter_passes_float_positions(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=500)
        features = FeatureBuilder().transform(bars)
        env = ContinuousTradingEnv(
            feature_frame=features,
            config=EnvironmentConfig(action_mode="continuous", reward_mode="raw", cost_rate_bp=0.0),
            splits={"train": ("2021-06-01", "2021-12-31")},
        )
        state, info = env.reset(symbol="AAA", split="train")
        self.assertEqual(state.shape, (env.state_size,))
        self.assertEqual(env.action_size, 1)
        self.assertEqual(info["symbol"], "AAA")
        _, _, _, _, step_info = env.step(np.array([0.25], dtype=np.float32))
        self.assertAlmostEqual(step_info["target_position"], 0.25)


if __name__ == "__main__":
    unittest.main()
