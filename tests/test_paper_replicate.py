from __future__ import annotations

import unittest

from rl_trading.features import FeatureBuilder
from rl_trading.paper_replicate import (
    ActorCriticNetwork,
    DQNNetwork,
    PaperRLConfig,
    PolicyGradientNetwork,
    infer_input_size,
    observation_to_tensor,
    run_paper_experiment,
)
from rl_trading.env import EnvironmentConfig, TradingEnv

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports direct module execution
    from tests.test_data_pipeline import make_bar_frame


class PaperReplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=900)
        self.features = FeatureBuilder().transform(bars)
        self.splits = {
            "train": ("2021-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-06-30"),
        }

    def test_networks_forward_shapes_match_action_spaces(self) -> None:
        env_config = EnvironmentConfig(action_mode="discrete", reward_mode="zhang")
        env = TradingEnv(feature_frame=self.features, config=env_config, splits=self.splits)
        observation = env.reset(symbol="AAA", split="train")
        tensor = observation_to_tensor(observation, device_name=__import__("torch").device("cpu"))
        input_size = infer_input_size(self.features, env_config, self.splits, "train")

        dqn = DQNNetwork(input_size)
        pg = PolicyGradientNetwork(input_size)
        a2c = ActorCriticNetwork(input_size)

        self.assertEqual(tuple(dqn(tensor).shape), (1, 3))
        self.assertEqual(tuple(pg(tensor).shape), (1, 3))
        mean, std, value = a2c(tensor)
        self.assertEqual(tuple(mean.shape), (1, 1))
        self.assertEqual(tuple(std.shape), (1, 1))
        self.assertEqual(tuple(value.shape), (1, 1))

    def test_run_paper_experiment_smoke_dqn(self) -> None:
        experiment = run_paper_experiment(
            feature_frame=self.features,
            config=PaperRLConfig(algo="dqn", episodes=1, batch_size=8, warmup_steps=8, replay_capacity=64),
            train_split="train",
            eval_split="test",
            symbols=["AAA"],
            splits=self.splits,
        )
        self.assertIn("paper_dqn", set(experiment["comparison"]["policy"]))
        self.assertEqual(len(experiment["episode_logs"]), 1)

    def test_run_paper_experiment_smoke_pg(self) -> None:
        experiment = run_paper_experiment(
            feature_frame=self.features,
            config=PaperRLConfig(algo="pg", episodes=1),
            train_split="train",
            eval_split="test",
            symbols=["AAA"],
            splits=self.splits,
        )
        self.assertIn("paper_pg", set(experiment["comparison"]["policy"]))
        self.assertEqual(len(experiment["episode_logs"]), 1)

    def test_run_paper_experiment_smoke_a2c(self) -> None:
        experiment = run_paper_experiment(
            feature_frame=self.features,
            config=PaperRLConfig(algo="a2c", episodes=1),
            train_split="train",
            eval_split="test",
            symbols=["AAA"],
            splits=self.splits,
        )
        self.assertIn("paper_a2c", set(experiment["comparison"]["policy"]))
        self.assertEqual(len(experiment["episode_logs"]), 1)


if __name__ == "__main__":
    unittest.main()
