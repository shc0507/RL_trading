"""CLI for RL experiment plots."""

from __future__ import annotations

import argparse

from .rl.plots import generate_paper_style_plots, plot_zhang_cumulative_trade_returns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RL plots from experiment artifacts."
    )
    parser.add_argument("--artifact-dir", required=True, help="Path to the run's rl artifact directory.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--style", choices=("legacy", "zhang"), default="zhang")
    parser.add_argument("--return-column", default="trade_return")
    parser.add_argument("--target-vol", type=float, default=None)
    parser.add_argument(
        "--robust-ylim",
        action="store_true",
        help="Also render a Zhang cumulative-return figure with robust per-panel y-limits.",
    )
    parser.add_argument(
        "--ylim-cap",
        type=float,
        default=None,
        help="Also render a Zhang cumulative-return figure with symmetric y-limits capped at this absolute value.",
    )
    parser.add_argument(
        "--no-cost-sweep",
        action="store_true",
        help="Skip the Zhang-style transaction-cost sweep figure.",
    )
    args = parser.parse_args()

    artifacts = generate_paper_style_plots(
        artifact_dir=args.artifact_dir,
        split=args.split,
        return_column=args.return_column,
        style=args.style,
        target_vol=args.target_vol,
        include_cost_sweep=not args.no_cost_sweep,
    )
    print(f"[{artifacts.style}] cumulative plot: {artifacts.cumulative_returns_path}")
    print(f"[{artifacts.style}] diagnostics plot: {artifacts.symbol_diagnostics_path}")
    if artifacts.cost_sweep_path is not None:
        print(f"[{artifacts.style}] cost-sweep plot: {artifacts.cost_sweep_path}")
    if args.style == "zhang" and args.robust_ylim:
        robust_path = plot_zhang_cumulative_trade_returns(
            artifact_dir=args.artifact_dir,
            split=args.split,
            target_vol=args.target_vol,
            return_column=args.return_column,
            robust_ylim=True,
        )
        print(f"[{artifacts.style}] robust cumulative plot: {robust_path}")
    if args.style == "zhang" and args.ylim_cap is not None:
        capped_path = plot_zhang_cumulative_trade_returns(
            artifact_dir=args.artifact_dir,
            split=args.split,
            target_vol=args.target_vol,
            return_column=args.return_column,
            ylim_cap=args.ylim_cap,
        )
        print(f"[{artifacts.style}] capped cumulative plot: {capped_path}")


if __name__ == "__main__":
    main()
