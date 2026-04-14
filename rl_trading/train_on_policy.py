"""CLI entrypoint for production A2C training."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .rl.on_policy_trainer import OnPolicyConfig, train_on_policy as train_on_policy_legacy
from .rl.production_a2c import ProductionA2CConfig
from .rl.zhang_production import ProductionDatasetConfig, ProductionSuiteConfig, train_a2c_production


def main() -> None:
    legacy_defaults = OnPolicyConfig()
    production_defaults = ProductionA2CConfig()
    parser = argparse.ArgumentParser(description="Train the production Zhang A2C system.")
    parser.add_argument("--legacy", action="store_true", help="Run the legacy discrete A2C/PPO trainer instead.")
    parser.add_argument("--algorithm", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timesteps", type=int, default=production_defaults.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=legacy_defaults.rollout_steps)
    parser.add_argument("--validation-interval", type=int, default=legacy_defaults.validation_interval)
    parser.add_argument("--seed", type=int, default=production_defaults.seed)
    parser.add_argument("--device", default=production_defaults.device)
    parser.add_argument("--policy-name", default=legacy_defaults.policy_name)
    parser.add_argument("--batch-size", type=int, default=production_defaults.batch_size)
    parser.add_argument("--actor-lr", type=float, default=production_defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=production_defaults.critic_lr)
    parser.add_argument("--data-root", default=None, help="Directory of institutional continuous-futures CSV files.")
    parser.add_argument("--source-dataset-dir", default=None, help="Existing processed production dataset directory.")
    parser.add_argument("--universe-manifest", default=None, help="Optional universe manifest CSV.")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    if args.legacy or args.algorithm == "ppo":
        config = OnPolicyConfig(
            algorithm=args.algorithm,
            policy_name=args.policy_name,
            total_steps=args.timesteps,
            rollout_steps=args.rollout_steps,
            validation_interval=args.validation_interval,
            seed=args.seed,
            device=args.device,
        )
        artifacts = train_on_policy_legacy(
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
        a2c=ProductionA2CConfig(
            total_steps=args.timesteps,
            batch_size=args.batch_size,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            seed=args.seed,
            device=args.device,
        ),
        output_dir=args.output_dir,
    )
    artifacts = train_a2c_production(suite)
    leaderboard = pd.read_csv(artifacts.leaderboard_path)
    print(leaderboard.to_string(index=False))
    print(f"\nCombined RL dir: {artifacts.combined_rl_dir}")
    print(f"Run metadata: {artifacts.run_metadata_path}")


if __name__ == "__main__":
    main()
