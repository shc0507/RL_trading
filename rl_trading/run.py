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
    args = parser.parse_args()
    run_experiment(
        symbols=args.symbols,
        agents=args.agents,
        n_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
