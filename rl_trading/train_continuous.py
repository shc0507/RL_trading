"""CLI entrypoint for TD3/SAC training."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .rl.continuous_trainer import ContinuousConfig, train_continuous


def main() -> None:
    defaults = ContinuousConfig()
    parser = argparse.ArgumentParser(description="Train a continuous-action trading agent.")
    parser.add_argument("--algorithm", choices=("td3", "sac"), default=defaults.algorithm)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timesteps", type=int, default=defaults.total_steps)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--policy-name", default=defaults.policy_name)
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    config = ContinuousConfig(
        algorithm=args.algorithm,
        policy_name=args.policy_name,
        total_steps=args.timesteps,
        validation_interval=args.validation_interval,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
    )
    artifacts = train_continuous(
        output_dir=args.output_dir,
        config=config,
        rebuild_data=args.rebuild_data,
    )
    summary = pd.read_csv(artifacts.summary_metrics_path)
    print(summary.to_string(index=False))
    print(f"\nBest checkpoint: {artifacts.best_checkpoint_path}")


if __name__ == "__main__":
    main()
