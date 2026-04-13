"""CLI entrypoint for DQN training."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .rl.trainer import DQNConfig, train_dqn


def main() -> None:
    defaults = DQNConfig()
    parser = argparse.ArgumentParser(description="Train the DQN trading baseline.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timesteps", type=int, default=defaults.total_steps)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--policy-name", default=defaults.policy_name)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--network-type", choices=("mlp", "lstm"), default=defaults.network_type)
    parser.add_argument("--recurrent-hidden-size", type=int, default=defaults.recurrent_hidden_size)
    parser.add_argument("--recurrent-layers", type=int, default=defaults.recurrent_layers)
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

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
    artifacts = train_dqn(
        output_dir=args.output_dir,
        config=config,
        rebuild_data=args.rebuild_data,
    )
    summary = pd.read_csv(artifacts.summary_metrics_path)
    print(summary.to_string(index=False))
    print(f"\nBest checkpoint: {artifacts.best_checkpoint_path}")


if __name__ == "__main__":
    main()
