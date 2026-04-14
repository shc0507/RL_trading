"""CLI entrypoint for production DQN training."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .rl.trainer import DQNConfig, train_dqn as train_dqn_legacy
from .rl.zhang_production import (
    ProductionDatasetConfig,
    ProductionDQNConfig,
    ProductionSuiteConfig,
    train_dqn_production,
)


def main() -> None:
    legacy_defaults = DQNConfig()
    production_defaults = ProductionDQNConfig()
    parser = argparse.ArgumentParser(description="Train the production Zhang DQN system.")
    parser.add_argument("--legacy", action="store_true", help="Run the legacy single-split trainer instead.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timesteps", type=int, default=legacy_defaults.total_steps)
    parser.add_argument("--validation-interval", type=int, default=legacy_defaults.validation_interval)
    parser.add_argument("--batch-size", type=int, default=production_defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=production_defaults.warmup_steps)
    parser.add_argument("--seed", type=int, default=production_defaults.seed)
    parser.add_argument("--device", default=production_defaults.device)
    parser.add_argument("--policy-name", default=legacy_defaults.policy_name)
    parser.add_argument("--double-dqn", action="store_true", help="Legacy mode only.")
    parser.add_argument("--dueling", action="store_true", help="Legacy mode only.")
    parser.add_argument("--network-type", choices=("mlp", "lstm"), default=legacy_defaults.network_type)
    parser.add_argument("--recurrent-hidden-size", type=int, default=legacy_defaults.recurrent_hidden_size)
    parser.add_argument("--recurrent-layers", type=int, default=legacy_defaults.recurrent_layers)
    parser.add_argument("--optimizer-updates", type=int, default=production_defaults.optimizer_updates)
    parser.add_argument("--data-root", default=None, help="Directory of institutional continuous-futures CSV files.")
    parser.add_argument("--source-dataset-dir", default=None, help="Existing processed production dataset directory.")
    parser.add_argument("--universe-manifest", default=None, help="Optional universe manifest CSV.")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    if args.legacy:
        config = DQNConfig(
            policy_name=args.policy_name,
            total_steps=args.timesteps,
            validation_interval=args.validation_interval,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=args.device,
            double_dqn=args.double_dqn,
            dueling=args.dueling,
            network_type=args.network_type,
            recurrent_hidden_size=args.recurrent_hidden_size,
            recurrent_layers=args.recurrent_layers,
        )
        artifacts = train_dqn_legacy(
            output_dir=args.output_dir,
            config=config,
            rebuild_data=args.rebuild_data,
        )
        summary = pd.read_csv(artifacts.summary_metrics_path)
        print(summary.to_string(index=False))
        print(f"\nBest checkpoint: {artifacts.best_checkpoint_path}")
        return

    dataset = ProductionDatasetConfig(
        data_root=args.data_root,
        source_dataset_dir=args.source_dataset_dir,
        universe_manifest_path=args.universe_manifest,
    )
    suite = ProductionSuiteConfig(
        dataset=dataset,
        dqn=ProductionDQNConfig(
            optimizer_updates=args.optimizer_updates,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=args.device,
        ),
        output_dir=args.output_dir,
    )
    artifacts = train_dqn_production(suite)
    leaderboard = pd.read_csv(artifacts.leaderboard_path)
    print(leaderboard.to_string(index=False))
    print(f"\nCombined RL dir: {artifacts.combined_rl_dir}")
    print(f"Run metadata: {artifacts.run_metadata_path}")


if __name__ == "__main__":
    main()
