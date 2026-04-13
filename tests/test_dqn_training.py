from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from rl_trading.features import FeatureBuilder
from rl_trading.rl.trainer import DQNConfig, train_dqn

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports module-style unittest invocation
    from tests.test_data_pipeline import make_bar_frame


class DQNTrainingSmokeTests(unittest.TestCase):
    def test_short_training_run_writes_artifacts_and_reports(self) -> None:
        bars = make_bar_frame(symbols=("AAA", "BBB"), periods=900)
        features = FeatureBuilder().transform(bars)
        config = DQNConfig(
            total_steps=64,
            batch_size=8,
            replay_capacity=256,
            warmup_steps=8,
            validation_interval=16,
            target_update_interval=8,
            hidden_sizes=(32, 32),
            seed=3,
            device="cpu",
        )
        splits = {
            "train": ("2021-06-01", "2021-12-31"),
            "val": ("2022-01-01", "2022-06-30"),
            "test": ("2022-07-01", "2022-12-30"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = train_dqn(
                output_dir=tmp_dir,
                config=config,
                feature_frame=features,
                splits=splits,
                symbols=["AAA", "BBB"],
            )

            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.training_curve_path.exists())
            self.assertTrue(artifacts.summary_metrics_path.exists())
            self.assertTrue(artifacts.best_checkpoint_path.exists())
            self.assertTrue(artifacts.final_checkpoint_path.exists())
            self.assertIsNotNone(artifacts.zhang_eval_dir)
            self.assertTrue((artifacts.zhang_eval_dir / "summary_scaled.csv").exists())
            self.assertTrue((artifacts.zhang_eval_dir / "cost_sweep.csv").exists())

            summary = pd.read_csv(artifacts.summary_metrics_path)
            self.assertIn("dqn", summary["policy"].tolist())
            self.assertTrue(((summary["policy"] == "dqn") & (summary["split"] == "test")).any())

            training_curve = pd.read_csv(artifacts.training_curve_path)
            self.assertFalse(training_curve.empty)
            self.assertTrue((artifacts.artifact_dir / "reports" / "dqn_test_daily_returns.csv").exists())

    def test_short_lstm_double_dueling_training_run_writes_variant_reports(self) -> None:
        bars = make_bar_frame(symbols=("AAA",), periods=900)
        features = FeatureBuilder().transform(bars)
        config = DQNConfig(
            policy_name="lstm_double_dueling_dqn",
            total_steps=32,
            batch_size=8,
            replay_capacity=128,
            warmup_steps=8,
            validation_interval=16,
            target_update_interval=8,
            hidden_sizes=(16,),
            double_dqn=True,
            dueling=True,
            network_type="lstm",
            recurrent_hidden_size=8,
            recurrent_layers=1,
            seed=5,
            device="cpu",
        )
        splits = {
            "train": ("2021-06-01", "2021-12-31"),
            "val": ("2022-01-01", "2022-06-30"),
            "test": ("2022-07-01", "2022-12-30"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = train_dqn(
                output_dir=tmp_dir,
                config=config,
                feature_frame=features,
                splits=splits,
                symbols=["AAA"],
            )

            summary = pd.read_csv(artifacts.summary_metrics_path)
            self.assertIn("lstm_double_dueling_dqn", summary["policy"].tolist())
            self.assertTrue(
                (artifacts.artifact_dir / "reports" / "lstm_double_dueling_dqn_test_trades.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
