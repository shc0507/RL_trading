#!/usr/bin/env python3
"""Run the Zhang production suite with Slurm-friendly CLI flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rl_trading.rl.production_a2c import ProductionA2CConfig
from rl_trading.rl.production_pg import ProductionPGConfig
from rl_trading.rl.zhang_production import (
    ProductionDatasetConfig,
    ProductionDQNConfig,
    ProductionSuiteConfig,
    run_production_suite,
)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_algorithms(value: str) -> tuple[str, ...]:
    algorithms = tuple(token.strip().lower() for token in value.split(",") if token.strip())
    valid = {"dqn", "pg", "a2c"}
    invalid = sorted(set(algorithms) - valid)
    if not algorithms:
        raise argparse.ArgumentTypeError("at least one algorithm is required")
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported algorithms: {', '.join(invalid)}")
    return algorithms


def main() -> None:
    project_root = _default_project_root()
    dqn_defaults = ProductionDQNConfig()
    pg_defaults = ProductionPGConfig()
    a2c_defaults = ProductionA2CConfig()

    parser = argparse.ArgumentParser(description="Run the Zhang production walk-forward suite.")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=None, help="Directory of institutional continuous-futures CSV inputs.")
    parser.add_argument("--source-dataset-dir", default=None, help="Existing processed production dataset directory.")
    parser.add_argument("--universe-manifest", default=None, help="Optional production universe manifest CSV.")
    parser.add_argument("--algorithms", type=_parse_algorithms, default=("dqn", "pg", "a2c"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=dqn_defaults.seed)
    parser.add_argument("--cv-fraction", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience-epochs", type=int, default=20)
    parser.add_argument("--selection-metric", default="sharpe")
    parser.add_argument("--selection-split", default="cv")
    parser.add_argument("--dqn-optimizer-updates", type=int, default=dqn_defaults.optimizer_updates)
    parser.add_argument("--dqn-batch-size", type=int, default=dqn_defaults.batch_size)
    parser.add_argument("--dqn-warmup-steps", type=int, default=dqn_defaults.warmup_steps)
    parser.add_argument("--pg-total-steps", type=int, default=pg_defaults.total_steps)
    parser.add_argument("--pg-actor-lr", type=float, default=pg_defaults.actor_lr)
    parser.add_argument("--a2c-total-steps", type=int, default=a2c_defaults.total_steps)
    parser.add_argument("--a2c-batch-size", type=int, default=a2c_defaults.batch_size)
    parser.add_argument("--a2c-actor-lr", type=float, default=a2c_defaults.actor_lr)
    parser.add_argument("--a2c-critic-lr", type=float, default=a2c_defaults.critic_lr)
    args = parser.parse_args()

    if not args.data_root and not args.source_dataset_dir:
        parser.error("one of --data-root or --source-dataset-dir is required")

    suite = ProductionSuiteConfig(
        dataset=ProductionDatasetConfig(
            data_root=args.data_root,
            source_dataset_dir=args.source_dataset_dir,
            universe_manifest_path=args.universe_manifest,
        ),
        dqn=ProductionDQNConfig(
            optimizer_updates=args.dqn_optimizer_updates,
            batch_size=args.dqn_batch_size,
            warmup_steps=args.dqn_warmup_steps,
            seed=args.seed,
            device=args.device,
        ),
        pg=ProductionPGConfig(
            total_steps=args.pg_total_steps,
            actor_lr=args.pg_actor_lr,
            seed=args.seed,
            device=args.device,
        ),
        a2c=ProductionA2CConfig(
            total_steps=args.a2c_total_steps,
            batch_size=args.a2c_batch_size,
            actor_lr=args.a2c_actor_lr,
            critic_lr=args.a2c_critic_lr,
            seed=args.seed,
            device=args.device,
        ),
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        policies_for_plots=(*args.algorithms, "long_only", "sign_12m", "macd"),
        cv_fraction=args.cv_fraction,
        early_stopping_patience_epochs=args.early_stopping_patience_epochs,
        selection_metric=args.selection_metric,
        selection_split=args.selection_split,
    )

    artifacts = run_production_suite(suite)
    leaderboard = pd.read_csv(artifacts.leaderboard_path)
    print(leaderboard.to_string(index=False))
    print(f"\nCombined RL dir: {artifacts.combined_rl_dir}")
    print(f"Run metadata: {artifacts.run_metadata_path}")


if __name__ == "__main__":
    main()
