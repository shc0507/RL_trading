from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from rl_trading.features import FeatureBuilder
from rl_trading.rl.continuous_trainer import ContinuousConfig, train_continuous
from rl_trading.rl.on_policy_trainer import OnPolicyConfig, train_on_policy

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports module-style unittest invocation
    from tests.test_data_pipeline import make_bar_frame


class AlgorithmTrainerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=900)
        cls.features = FeatureBuilder().transform(bars)
        cls.splits = {
            "train": ("2021-06-01", "2021-12-31"),
            "val": ("2022-01-01", "2022-06-30"),
            "test": ("2022-07-01", "2022-12-30"),
        }

    def test_a2c_and_ppo_trainers_write_eval_artifacts(self) -> None:
        for algorithm in ("a2c", "ppo"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp_dir:
                artifacts = train_on_policy(
                    output_dir=tmp_dir,
                    config=OnPolicyConfig(
                        algorithm=algorithm,
                        total_steps=32,
                        rollout_steps=8,
                        validation_interval=16,
                        hidden_sizes=(16,),
                        update_epochs=2,
                        minibatch_size=8,
                        seed=13,
                        device="cpu",
                    ),
                    feature_frame=self.features,
                    splits=self.splits,
                    symbols=["AAA"],
                )

                self.assertTrue(artifacts.best_checkpoint_path.exists())
                self.assertTrue(artifacts.final_checkpoint_path.exists())
                self.assertTrue((artifacts.zhang_eval_dir / "summary_scaled.csv").exists())
                summary = pd.read_csv(artifacts.summary_metrics_path)
                self.assertIn(algorithm, summary["policy"].tolist())
                self.assertTrue((artifacts.artifact_dir / "reports" / f"{algorithm}_test_trades.csv").exists())

    def test_td3_and_sac_trainers_write_eval_artifacts(self) -> None:
        for algorithm in ("td3", "sac"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp_dir:
                artifacts = train_continuous(
                    output_dir=tmp_dir,
                    config=ContinuousConfig(
                        algorithm=algorithm,
                        total_steps=32,
                        batch_size=8,
                        replay_capacity=128,
                        warmup_steps=8,
                        validation_interval=16,
                        hidden_sizes=(16,),
                        seed=17,
                        device="cpu",
                    ),
                    feature_frame=self.features,
                    splits=self.splits,
                    symbols=["AAA"],
                )

                self.assertTrue(artifacts.best_checkpoint_path.exists())
                self.assertTrue(artifacts.final_checkpoint_path.exists())
                self.assertTrue((artifacts.zhang_eval_dir / "summary_scaled.csv").exists())
                summary = pd.read_csv(artifacts.summary_metrics_path)
                self.assertIn(algorithm, summary["policy"].tolist())
                self.assertTrue((artifacts.artifact_dir / "reports" / f"{algorithm}_test_trades.csv").exists())


if __name__ == "__main__":
    unittest.main()
