"""CLI entrypoint for the full production Zhang walk-forward suite."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .rl.production_a2c import ProductionA2CConfig
from .rl.zhang_production import (
    ProductionDatasetConfig,
    ProductionDQNConfig,
    ProductionSuiteConfig,
    run_production_suite,
)


def main() -> None:
    dqn_defaults = ProductionDQNConfig()
    a2c_defaults = ProductionA2CConfig()
    parser = argparse.ArgumentParser(description="Run the full production Zhang walk-forward suite.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=None, help="Directory of institutional continuous-futures CSV files.")
    parser.add_argument("--source-dataset-dir", default=None, help="Existing processed production dataset directory.")
    parser.add_argument("--universe-manifest", default=None, help="Optional universe manifest CSV.")
    parser.add_argument("--device", default=dqn_defaults.device)
    parser.add_argument("--seed", type=int, default=dqn_defaults.seed)
    parser.add_argument("--dqn-optimizer-updates", type=int, default=dqn_defaults.optimizer_updates)
    parser.add_argument("--a2c-total-steps", type=int, default=a2c_defaults.total_steps)
    args = parser.parse_args()

    suite = ProductionSuiteConfig(
        dataset=ProductionDatasetConfig(
            data_root=args.data_root,
            source_dataset_dir=args.source_dataset_dir,
            universe_manifest_path=args.universe_manifest,
        ),
        dqn=ProductionDQNConfig(
            optimizer_updates=args.dqn_optimizer_updates,
            seed=args.seed,
            device=args.device,
        ),
        a2c=ProductionA2CConfig(
            total_steps=args.a2c_total_steps,
            seed=args.seed,
            device=args.device,
        ),
        output_dir=args.output_dir,
    )
    artifacts = run_production_suite(suite)
    leaderboard = pd.read_csv(artifacts.leaderboard_path)
    print(leaderboard.to_string(index=False))
    print(f"\nCombined RL dir: {artifacts.combined_rl_dir}")
    print(f"Run metadata: {artifacts.run_metadata_path}")


if __name__ == "__main__":
    main()
