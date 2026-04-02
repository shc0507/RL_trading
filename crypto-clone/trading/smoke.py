"""End-to-end smoke runner for the strategy research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import Backtester
from .config import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLITS,
    DEFAULT_START_DATE,
    get_default_symbols,
    get_instrument_map,
    get_periods_per_year,
)
from .data.pipeline import MarketDataPipeline
from .data.sources import PublicDailySource
from .env import EnvironmentConfig
from .features import FeatureBuilder
from .strategies import LongOnlyStrategy, MACDStrategy, RSIMeanReversionStrategy, Sign12MStrategy


def run_smoke(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    symbols: tuple[str, ...] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    market: str = "equity",
) -> Path:
    output_path = Path(output_dir)
    symbol_list = tuple(symbols or get_default_symbols(market))
    instrument_map = get_instrument_map(market)
    pipeline = MarketDataPipeline(output_dir=output_path)
    source = PublicDailySource(instrument_map=instrument_map)
    feature_builder = FeatureBuilder()
    artifacts = pipeline.build(
        source=source,
        feature_builder=feature_builder,
        symbols=list(symbol_list),
        start_date=start_date,
        end_date=end_date,
    )

    env_config = EnvironmentConfig(action_mode="continuous", reward_mode="zhang")
    backtester = Backtester(
        feature_frame=artifacts.features,
        env_config=env_config,
        splits=DEFAULT_SPLITS,
        periods_per_year=get_periods_per_year(market),
    )
    strategies = [
        LongOnlyStrategy(),
        Sign12MStrategy(),
        MACDStrategy(),
        RSIMeanReversionStrategy(),
    ]

    summary_rows = []
    smoke_dir = output_path / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for split in DEFAULT_SPLITS:
        for strategy in strategies:
            report = backtester.run(policy=strategy, split=split)
            summary_rows.append({"strategy": strategy.name, "split": split, **report.portfolio_metrics})
            report.daily_returns.to_csv(smoke_dir / f"{strategy.name}_{split}_daily_returns.csv", index=False)
            report.trade_log.to_csv(smoke_dir / f"{strategy.name}_{split}_trades.csv", index=False)
            report.symbol_metrics.to_csv(smoke_dir / f"{strategy.name}_{split}_symbol_metrics.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "strategy"]).reset_index(drop=True)
    summary_path = smoke_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quant strategy smoke test.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--market", choices=("equity", "crypto"), default="equity")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()) or None
    summary_path = run_smoke(
        output_dir=args.output_dir,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        market=args.market,
    )
    summary = pd.read_csv(summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
