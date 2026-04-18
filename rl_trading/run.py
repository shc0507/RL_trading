"""Run the full Zhang et al. experiment."""

import argparse

from rl_trading.evaluate import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Zhang et al. (2019) DRL trading experiment")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols (default: full universe)")
    parser.add_argument("--agents", nargs="+", default=["dqn", "pg", "a2c"], help="RL agents to train")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per agent")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, mps (default: auto-detect)")
    parser.add_argument("--walk-forward", action="store_true", help="Use expanding-window walk-forward folds")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", default="rl-trading", help="wandb project name")
    parser.add_argument("--wandb-entity", default=None, help="wandb entity (team or user)")
    parser.add_argument("--wandb-name", default=None, help="wandb run name")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="wandb tags")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Inspect stable-universe data coverage and exit before training",
    )
    args = parser.parse_args()
    run_experiment(
        symbols=args.symbols,
        agents=args.agents,
        n_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        device=args.device,
        walk_forward=args.walk_forward,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        preflight_only=args.preflight,
    )


if __name__ == "__main__":
    main()
