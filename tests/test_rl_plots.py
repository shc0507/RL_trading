from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rl_trading.features import FeatureBuilder
from rl_trading.rl.plots import (
    compute_symbol_diagnostics,
    generate_paper_style_plots,
    load_daily_return_reports,
    load_zhang_contract_metrics,
    load_zhang_cost_sweep,
    load_zhang_portfolio_reports,
)
from rl_trading.rl.trainer import DQNConfig, train_dqn

try:
    from test_data_pipeline import make_bar_frame
    from test_zhang_eval import make_instruments, make_trade_logs
except ModuleNotFoundError:  # pragma: no cover - supports module-style unittest invocation
    from tests.test_data_pipeline import make_bar_frame
    from tests.test_zhang_eval import make_instruments, make_trade_logs


def write_trade_report_artifacts(root: Path) -> Path:
    artifact_dir = root / "rl"
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    trade_logs = make_trade_logs()
    for (policy, split), frame in trade_logs.groupby(["policy", "split"], sort=False):
        frame.drop(columns=["policy"]).to_csv(reports_dir / f"{policy}_{split}_trades.csv", index=False)
    make_instruments().loc[:, ["symbol", "asset_class"]].to_csv(
        processed_dir / "instruments.csv",
        index=False,
    )
    return artifact_dir


class RLPlotTests(unittest.TestCase):
    def test_plot_helpers_render_files_from_training_artifacts(self) -> None:
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
            seed=11,
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

            daily_returns = load_daily_return_reports(artifacts.artifact_dir, split="test")
            self.assertFalse(daily_returns.empty)
            diagnostics = compute_symbol_diagnostics(artifacts.artifact_dir, split="test")
            self.assertFalse(diagnostics.empty)

            plot_artifacts = generate_paper_style_plots(artifacts.artifact_dir, split="test")
            self.assertTrue(Path(plot_artifacts.cumulative_returns_path).exists())
            self.assertTrue(Path(plot_artifacts.symbol_diagnostics_path).exists())
            self.assertTrue(Path(plot_artifacts.cost_sweep_path).exists())
            self.assertEqual(plot_artifacts.style, "zhang")

            zhang_portfolios = load_zhang_portfolio_reports(artifacts.artifact_dir, split="test")
            self.assertFalse(zhang_portfolios.empty)

            legacy_artifacts = generate_paper_style_plots(
                artifacts.artifact_dir,
                split="test",
                style="legacy",
            )
            self.assertTrue(Path(legacy_artifacts.cumulative_returns_path).exists())
            self.assertTrue(Path(legacy_artifacts.symbol_diagnostics_path).exists())
            self.assertIsNone(legacy_artifacts.cost_sweep_path)

    def test_zhang_plot_helpers_render_multi_asset_panels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = write_trade_report_artifacts(Path(tmp_dir))

            plot_artifacts = generate_paper_style_plots(
                artifact_dir=artifact_dir,
                split="test",
                style="zhang",
            )
            self.assertTrue(Path(plot_artifacts.cumulative_returns_path).exists())
            self.assertTrue(Path(plot_artifacts.symbol_diagnostics_path).exists())
            self.assertTrue(Path(plot_artifacts.cost_sweep_path).exists())

            portfolios = load_zhang_portfolio_reports(artifact_dir, split="test")
            contract_metrics = load_zhang_contract_metrics(artifact_dir, split="test")
            cost_sweep = load_zhang_cost_sweep(artifact_dir, split="test")
            self.assertSetEqual(
                set(portfolios["asset_group"].unique().tolist()),
                {"all", "equity_index", "fx"},
            )
            self.assertSetEqual(
                set(contract_metrics["asset_group"].unique().tolist()),
                {"equity_index", "fx"},
            )
            self.assertEqual(
                sorted(cost_sweep["cost_rate_bp"].unique().tolist()),
                [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
            )


if __name__ == "__main__":
    unittest.main()
