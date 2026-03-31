"""End-to-end smoke runner for the week-1 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import Backtester
from .config import DEFAULT_END_DATE, DEFAULT_OUTPUT_DIR, DEFAULT_SPLITS, DEFAULT_START_DATE, DEFAULT_SYMBOLS
from .data.pipeline import MarketDataPipeline
from .data.sources import PublicDailySource
from .env import EnvironmentConfig
from .features import FeatureBuilder
from .policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy


def run_smoke(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> Path:
    output_path = Path(output_dir)
    pipeline = MarketDataPipeline(output_dir=output_path)
    source = PublicDailySource()
    feature_builder = FeatureBuilder()
    artifacts = pipeline.build(
        source=source,
        feature_builder=feature_builder,
        symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
    )

    env_config = EnvironmentConfig(action_mode="continuous", reward_mode="zhang")
    backtester = Backtester(feature_frame=artifacts.features, env_config=env_config, splits=DEFAULT_SPLITS)
    policies = [LongOnlyPolicy(), Sign12MPolicy(), MACDPolicy()]

    summary_rows = []
    smoke_dir = output_path / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for split in DEFAULT_SPLITS:
        for policy in policies:
            report = backtester.run(policy=policy, split=split)
            summary_rows.append({"policy": policy.name, "split": split, **report.portfolio_metrics})
            report.daily_returns.to_csv(smoke_dir / f"{policy.name}_{split}_daily_returns.csv", index=False)
            report.trade_log.to_csv(smoke_dir / f"{policy.name}_{split}_trades.csv", index=False)
            report.symbol_metrics.to_csv(smoke_dir / f"{policy.name}_{split}_symbol_metrics.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "policy"]).reset_index(drop=True)
    summary_path = smoke_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the week-1 DRL trading smoke test.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary_path = run_smoke(output_dir=args.output_dir)
    summary = pd.read_csv(summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
