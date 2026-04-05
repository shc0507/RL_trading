from __future__ import annotations

import unittest

from rl_trading.features import FeatureBuilder
from rl_trading.sb3_train import SB3TrainConfig, run_sb3_experiment

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports direct module execution
    from tests.test_data_pipeline import make_bar_frame


class SB3TrainTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=900)
        self.features = FeatureBuilder().transform(bars)
        self.splits = {
            "train": ("2021-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-06-30"),
        }

    def test_run_sb3_experiment_smoke_dqn(self) -> None:
        experiment = run_sb3_experiment(
            feature_frame=self.features,
            config=SB3TrainConfig(algo="dqn", total_timesteps=64),
            train_split="train",
            eval_split="test",
            symbols=["AAA"],
            splits=self.splits,
        )
        self.assertIn("sb3_dqn", set(experiment["comparison"]["policy"]))
        self.assertIn("annualized_return", experiment["comparison"].columns)


if __name__ == "__main__":
    unittest.main()
