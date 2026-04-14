from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rl_trading.features import FeatureBuilder
from rl_trading.rl.production_a2c import ProductionA2CConfig
from rl_trading.rl.production_pg import ProductionPGConfig
from rl_trading.rl.sequence_encoder import StackedLSTMStateEncoder
from rl_trading.rl.zhang_production import (
    PRODUCTION_POLICIES,
    ProductionDQNConfig,
    ProductionDatasetConfig,
    ProductionSuiteConfig,
    WalkForwardConfig,
    _build_train_cv_test_splits,
    _production_a2c_config,
    _production_dqn_config,
    run_production_suite,
    validate_production_dataset,
)


def make_fixture_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2023-12-29")
    specs = (
        {"symbol": "AAA", "asset_class": "commodity", "currency": "USD", "base": 100.0, "drift": 0.03},
        {"symbol": "BBB", "asset_class": "fx", "currency": "JPY", "base": 80.0, "drift": -0.01},
    )
    frames: list[pd.DataFrame] = []
    for index, spec in enumerate(specs, start=1):
        t = np.arange(len(dates), dtype=float)
        seasonal = 1.5 * np.sin(t / (16.0 + index))
        trend = spec["drift"] * t
        price = spec["base"] + trend + seasonal + (index * 0.5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": spec["symbol"],
                "asset_class": spec["asset_class"],
                "open": price - 0.25,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000 + (index * 10_000),
                "source": "fixture",
                "currency": spec["currency"],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_fixture_dataset(root: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    dataset_dir = root / "fixture_dataset"
    processed_dir = dataset_dir / "processed"
    manifests_dir = dataset_dir / "manifests"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    bars = make_fixture_bars()
    features = FeatureBuilder().transform(bars)
    instruments = (
        bars.loc[:, ["symbol", "asset_class", "currency", "source"]]
        .drop_duplicates()
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    universe = pd.DataFrame(
        [
            {"symbol": "AAA", "asset_class": "commodity", "currency": "USD", "description": "Fixture Commodity"},
            {"symbol": "BBB", "asset_class": "fx", "currency": "JPY", "description": "Fixture FX"},
        ]
    )
    bars.to_csv(processed_dir / "bars.csv", index=False)
    features.to_csv(processed_dir / "features.csv", index=False)
    instruments.to_csv(processed_dir / "instruments.csv", index=False)
    manifest_path = manifests_dir / "universe.csv"
    universe.to_csv(manifest_path, index=False)
    return dataset_dir, manifest_path, bars, universe


class ZhangProductionTests(unittest.TestCase):
    def test_walk_forward_windows_are_non_overlapping_and_anchored(self) -> None:
        config = WalkForwardConfig(
            train_start="2020-01-01",
            anchors=("2021-12-31", "2022-12-30"),
            dataset_end="2023-12-29",
            test_horizon_years=1,
        )

        windows = config.windows()

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].name, "wf_2021")
        self.assertEqual(windows[0].train_end, "2021-12-31")
        self.assertEqual(windows[0].test_start, "2022-01-03")
        self.assertEqual(windows[0].test_end, "2022-12-31")
        self.assertEqual(windows[1].name, "wf_2022")
        self.assertGreater(pd.Timestamp(windows[1].test_start), pd.Timestamp(windows[0].test_end))

    def test_validate_production_dataset_rejects_missing_symbols_mappings_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, bars, universe = write_fixture_dataset(Path(tmp_dir))
            instruments = universe.loc[:, ["symbol", "asset_class", "currency"]].copy()

            validate_production_dataset(
                bars=bars,
                instruments=instruments,
                universe=universe,
                start_date="2020-01-01",
                end_date="2023-12-29",
            )

            with self.assertRaises(ValueError):
                validate_production_dataset(
                    bars=bars,
                    instruments=instruments.loc[instruments["symbol"] != "BBB"].reset_index(drop=True),
                    universe=universe,
                    start_date="2020-01-01",
                    end_date="2023-12-29",
                )

            wrong_mapping = instruments.copy()
            wrong_mapping.loc[wrong_mapping["symbol"] == "BBB", "asset_class"] = "commodity"
            with self.assertRaises(ValueError):
                validate_production_dataset(
                    bars=bars,
                    instruments=wrong_mapping,
                    universe=universe,
                    start_date="2020-01-01",
                    end_date="2023-12-29",
                )

            truncated = bars.loc[
                ~((bars["symbol"] == "BBB") & (pd.to_datetime(bars["date"]) > pd.Timestamp("2023-11-30")))
            ].reset_index(drop=True)
            with self.assertRaises(ValueError):
                validate_production_dataset(
                    bars=truncated,
                    instruments=instruments,
                    universe=universe,
                    start_date="2020-01-01",
                    end_date="2023-12-29",
                )

    def test_production_configs_match_zhang_defaults(self) -> None:
        dqn = _production_dqn_config(
            ProductionDQNConfig(),
            selection_metric="sharpe",
            selection_split="cv",
            early_stopping_patience_epochs=20,
            epoch_steps=128,
        )
        self.assertEqual(dqn.network_type, "lstm")
        self.assertEqual(dqn.recurrent_layer_sizes, (64, 32))
        self.assertEqual(dqn.recurrent_dropout, 0.1)
        self.assertEqual(dqn.head_dropout, 0.1)
        self.assertEqual(dqn.batch_size, 64)
        self.assertEqual(dqn.replay_capacity, 5_000)
        self.assertEqual(dqn.target_update_interval, 1_000)
        self.assertEqual(dqn.gamma, 0.3)
        self.assertEqual(dqn.learning_rate, 1e-4)
        self.assertTrue(dqn.double_dqn)
        self.assertTrue(dqn.dueling)
        self.assertEqual(dqn.selection_mode, "validation")
        self.assertEqual(dqn.validation_split_name, "cv")
        self.assertEqual(dqn.early_stopping_patience_epochs, 20)

        a2c = _production_a2c_config(
            ProductionA2CConfig(),
            selection_metric="sharpe",
            selection_split="cv",
            early_stopping_patience_epochs=20,
            epoch_steps=128,
        )
        self.assertEqual(a2c.recurrent_layer_sizes, (64, 32))
        self.assertEqual(a2c.recurrent_dropout, 0.1)
        self.assertEqual(a2c.head_dropout, 0.1)
        self.assertEqual(a2c.batch_size, 128)
        self.assertEqual(a2c.gamma, 0.3)
        self.assertEqual(a2c.actor_lr, 1e-4)
        self.assertEqual(a2c.critic_lr, 1e-3)
        self.assertEqual(a2c.reward_mode, "zhang")
        self.assertEqual(a2c.validation_split_name, "cv")

        self.assertIn("dqn", PRODUCTION_POLICIES)
        self.assertIn("pg", PRODUCTION_POLICIES)
        self.assertIn("a2c", PRODUCTION_POLICIES)
        self.assertNotIn("ppo", PRODUCTION_POLICIES)
        self.assertNotIn("td3", PRODUCTION_POLICIES)
        self.assertNotIn("sac", PRODUCTION_POLICIES)

    def test_train_cv_test_split_construction_uses_last_ten_percent_for_cv(self) -> None:
        features = FeatureBuilder().transform(make_fixture_bars())
        window = WalkForwardConfig(
            train_start="2020-01-01",
            anchors=("2022-12-30",),
            dataset_end="2023-12-29",
            test_horizon_years=1,
        ).windows()[0]
        splits, metadata = _build_train_cv_test_splits(features, ["AAA"], window, 0.10)
        in_sample_dates = sorted(
            pd.to_datetime(
                features.loc[
                    (features["symbol"] == "AAA")
                    & (features["date"] >= pd.Timestamp(window.train_start))
                    & (features["date"] <= pd.Timestamp(window.train_end))
                    & features["window_ready"],
                    "date",
                ].unique()
            )
        )
        cv_count = max(1, int(np.ceil(len(in_sample_dates) * 0.10)))
        cv_count = min(cv_count, len(in_sample_dates) - 1)
        self.assertEqual(metadata["cv_start"], pd.Timestamp(in_sample_dates[-cv_count]).strftime("%Y-%m-%d"))
        self.assertLess(pd.Timestamp(splits["train"][1]), pd.Timestamp(splits["cv"][0]))
        self.assertLess(pd.Timestamp(splits["cv"][1]), pd.Timestamp(splits["test"][0]))

    def test_recurrent_encoder_supports_dropout(self) -> None:
        encoder = StackedLSTMStateEncoder(
            state_size=61,
            observation_window=12,
            feature_size=5,
            recurrent_layer_sizes=(64, 32),
            recurrent_dropout=0.2,
            output_dropout=0.1,
        )
        self.assertEqual(len(encoder.dropout_layers), 2)
        self.assertAlmostEqual(encoder.recurrent_dropout, 0.2)
        self.assertAlmostEqual(encoder.output_dropout, 0.1)

    def test_run_production_suite_writes_combined_out_of_sample_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir, manifest_path, _, _ = write_fixture_dataset(root)
            suite = ProductionSuiteConfig(
                dataset=ProductionDatasetConfig(
                    source_dataset_dir=dataset_dir,
                    start_date="2020-01-01",
                    end_date="2023-12-29",
                    universe_manifest_path=manifest_path,
                ),
                walk_forward=WalkForwardConfig(
                    train_start="2020-01-01",
                    anchors=("2021-12-31", "2022-12-30"),
                    dataset_end="2023-12-29",
                    test_horizon_years=1,
                ),
                dqn=ProductionDQNConfig(
                    optimizer_updates=4,
                    batch_size=4,
                    replay_capacity=32,
                    warmup_steps=4,
                    target_update_interval=4,
                    seed=3,
                    device="cpu",
                    search_grid=(
                        {"learning_rate": 1e-4, "gamma": 0.3, "recurrent_dropout": 0.1, "head_dropout": 0.1},
                    ),
                ),
                a2c=ProductionA2CConfig(
                    total_steps=16,
                    batch_size=8,
                    rollout_steps=4,
                    seed=5,
                    device="cpu",
                    search_grid=(
                        {"actor_lr": 1e-4, "critic_lr": 1e-3, "gamma": 0.3, "recurrent_dropout": 0.1, "head_dropout": 0.1, "entropy_coef": 0.01},
                    ),
                ),
                pg=ProductionPGConfig(
                    total_steps=16,
                    seed=7,
                    device="cpu",
                    search_grid=(
                        {"actor_lr": 1e-4, "gamma": 0.3, "recurrent_dropout": 0.1, "head_dropout": 0.1, "entropy_coef": 0.01},
                    ),
                ),
                early_stopping_patience_epochs=2,
                output_dir=root / "production_run",
            )

            artifacts = run_production_suite(suite)

            self.assertTrue(artifacts.individual_run_summaries_path.exists())
            self.assertTrue(artifacts.leaderboard_path.exists())
            self.assertTrue(artifacts.run_metadata_path.exists())
            self.assertTrue((artifacts.zhang_eval_dir / "summary_scaled.csv").exists())
            self.assertTrue((artifacts.zhang_eval_dir / "cost_sweep.csv").exists())
            self.assertTrue((artifacts.combined_rl_dir / "reports" / "dqn_test_trades.csv").exists())
            self.assertTrue((artifacts.combined_rl_dir / "reports" / "pg_test_trades.csv").exists())
            self.assertTrue((artifacts.combined_rl_dir / "reports" / "a2c_test_trades.csv").exists())
            self.assertTrue((artifacts.root_dir / "selected_hyperparameters.csv").exists())
            self.assertTrue((artifacts.root_dir / "split_manifest.csv").exists())
            for path in artifacts.plot_paths.values():
                self.assertTrue(Path(path).exists())

            combined_summary = pd.read_csv(artifacts.combined_rl_dir / "summary_metrics.csv")
            self.assertSetEqual(
                set(combined_summary["policy"].tolist()),
                {"dqn", "pg", "a2c", "long_only", "sign_12m", "macd"},
            )

            individual = pd.read_csv(artifacts.individual_run_summaries_path)
            self.assertSetEqual(set(individual["training_asset_class"].unique().tolist()), {"commodity", "fx"})
            self.assertSetEqual(set(individual["window_name"].unique().tolist()), {"wf_2021", "wf_2022"})

            selected = pd.read_csv(artifacts.root_dir / "selected_hyperparameters.csv")
            self.assertSetEqual(set(selected["algorithm"].unique().tolist()), {"dqn", "pg", "a2c"})
            self.assertIn("selected_candidate_index", selected.columns)

            split_manifest = pd.read_csv(artifacts.root_dir / "split_manifest.csv")
            self.assertTrue({"train_core_start", "train_core_end", "cv_start", "cv_end", "test_start", "test_end"}.issubset(split_manifest.columns))

            contract_daily = pd.read_csv(artifacts.zhang_eval_dir / "contract_daily_returns.csv")
            self.assertTrue(
                {"trade_return", "trade_cost", "scaled_turnover", "pre_cost_trade_return", "scaled_position"}.issubset(
                    contract_daily.columns
                )
            )

            leaderboard = pd.read_csv(artifacts.leaderboard_path)
            self.assertFalse(leaderboard.empty)


if __name__ == "__main__":
    unittest.main()
