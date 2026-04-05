"""Equity-index-only approximation of Zhang et al.'s rolling methodology."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .backtest import EvalReport
from .config import DEFAULT_OUTPUT_DIR
from .data.pipeline import MarketDataPipeline
from .data.sources import PublicDailySource
from .features import FeatureBuilder
from .metrics import compute_performance_metrics
from .sb3_train import SB3TrainConfig, run_sb3_experiment, save_experiment_outputs


ZHANG_EQUITY_INSTRUMENT_MAP: dict[str, dict[str, str]] = {
    "CA": {"symbol": "CA", "yahoo_symbol": "^FCHI", "asset_class": "equity_index", "currency": "EUR"},
    "ES": {"symbol": "ES", "yahoo_symbol": "ES=F", "asset_class": "equity_index_future", "currency": "USD"},
    "LX": {"symbol": "LX", "yahoo_symbol": "^FTSE", "asset_class": "equity_index", "currency": "GBP"},
    "MD": {"symbol": "MD", "yahoo_symbol": "^MID", "asset_class": "equity_index", "currency": "USD"},
    "SC": {"symbol": "SC", "yahoo_symbol": "SP=F", "asset_class": "equity_index_future", "currency": "USD"},
    "SP": {"symbol": "SP", "yahoo_symbol": "^GSPC", "asset_class": "equity_index", "currency": "USD"},
    "XU": {"symbol": "XU", "yahoo_symbol": "^STOXX50E", "asset_class": "equity_index", "currency": "EUR"},
    "YM": {"symbol": "YM", "yahoo_symbol": "YM=F", "asset_class": "equity_index_future", "currency": "USD"},
}

ZHANG_EQUITY_SYMBOLS: tuple[str, ...] = tuple(ZHANG_EQUITY_INSTRUMENT_MAP.keys())
DEFAULT_FETCH_START = "2005-01-01"
DEFAULT_FETCH_END = "2019-12-31"
DEFAULT_EPOCH_MULTIPLIER = 20


def zhang_equity_splits() -> dict[str, tuple[str, str]]:
    return {
        "train_pre_2011": (DEFAULT_FETCH_START, "2010-12-31"),
        "test_2011_2015": ("2011-01-01", "2015-12-31"),
        "train_pre_2016": (DEFAULT_FETCH_START, "2015-12-31"),
        "test_2016_2019": ("2016-01-01", "2019-12-31"),
    }


def zhang_equity_windows() -> list[tuple[str, str, str]]:
    return [
        ("2011_2015", "train_pre_2011", "test_2011_2015"),
        ("2016_2019", "train_pre_2016", "test_2016_2019"),
    ]


def build_window_splits(
    feature_frame: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
    validation_fraction: float = 0.1,
) -> dict[str, tuple[str, str]]:
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    train_dates = sorted(pd.to_datetime(frame.loc[frame["date"] <= pd.Timestamp(train_end), "date"]).unique())
    if len(train_dates) < 20:
        raise ValueError("not enough dates to create Zhang-style train/validation split")
    validation_size = max(1, int(len(train_dates) * validation_fraction))
    fit_dates = train_dates[:-validation_size]
    validation_dates = train_dates[-validation_size:]
    return {
        "train_fit": (str(pd.Timestamp(fit_dates[0]).date()), str(pd.Timestamp(fit_dates[-1]).date())),
        "train_val": (str(pd.Timestamp(validation_dates[0]).date()), str(pd.Timestamp(validation_dates[-1]).date())),
        "test": (test_start, test_end),
    }


def count_training_steps(
    feature_frame: pd.DataFrame,
    symbols: list[str] | tuple[str, ...],
    split_range: tuple[str, str],
) -> int:
    start_date, end_date = split_range
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    total_steps = 0
    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"] == symbol]
        episode = symbol_frame.loc[
            (symbol_frame["date"] >= pd.Timestamp(start_date))
            & (symbol_frame["date"] <= pd.Timestamp(end_date))
            & symbol_frame["window_ready"]
        ]
        if len(episode) >= 2:
            total_steps += len(episode) - 1
    if total_steps <= 0:
        raise ValueError("no usable training steps found for the requested split")
    return int(total_steps)


def resolve_training_budget(
    base_training_steps: int,
    total_timesteps: int | None,
    epoch_multiplier: int = DEFAULT_EPOCH_MULTIPLIER,
) -> tuple[int, int]:
    if total_timesteps is not None:
        return int(total_timesteps), max(int(total_timesteps) // max(epoch_multiplier, 1), 1)
    return int(base_training_steps * epoch_multiplier), int(base_training_steps)


def build_equity_artifacts(
    output_dir: str | Path,
    start_date: str = DEFAULT_FETCH_START,
    end_date: str = DEFAULT_FETCH_END,
    symbols: list[str] | tuple[str, ...] = ZHANG_EQUITY_SYMBOLS,
):
    output_root = Path(output_dir)
    features_path = output_root / "processed" / "features.csv"
    bars_path = output_root / "processed" / "bars.csv"
    instruments_path = output_root / "processed" / "instruments.csv"
    split_manifest_path = output_root / "manifests" / "splits.json"
    leakage_report_path = output_root / "qa" / "feature_leakage.json"
    if features_path.exists() and bars_path.exists():
        return {
            "bars": pd.read_csv(bars_path),
            "features": pd.read_csv(features_path),
            "bars_path": bars_path,
            "features_path": features_path,
            "instruments_path": instruments_path,
            "split_manifest_path": split_manifest_path,
            "leakage_report_path": leakage_report_path,
        }

    pipeline = MarketDataPipeline(output_dir=output_dir)
    source = PublicDailySource(instrument_map=ZHANG_EQUITY_INSTRUMENT_MAP)
    feature_builder = FeatureBuilder()
    return pipeline.build(
        source=source,
        feature_builder=feature_builder,
        symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
        splits={"full_sample": (start_date, end_date)},
    )


def aggregate_eval_reports(policy_name: str, reports: list[EvalReport], combined_split: str) -> EvalReport:
    if not reports:
        raise ValueError(f"no reports provided for policy {policy_name}")

    daily_returns = pd.concat(
        [report.daily_returns.assign(window_index=index) for index, report in enumerate(reports)],
        ignore_index=True,
    ).sort_values(["date", "window_index"]).reset_index(drop=True)
    trade_log = pd.concat(
        [report.trade_log.assign(window_index=index) for index, report in enumerate(reports)],
        ignore_index=True,
    ).sort_values(["date", "symbol", "window_index"]).reset_index(drop=True)
    symbol_metrics = pd.concat(
        [report.symbol_metrics.assign(window_index=index) for index, report in enumerate(reports)],
        ignore_index=True,
    ).reset_index(drop=True)

    portfolio_metrics = compute_performance_metrics(
        daily_returns["portfolio_return"],
        turnover=daily_returns["avg_turnover"],
        total_cost=float(daily_returns["total_cost"].sum()),
    )

    return EvalReport(
        policy_name=policy_name,
        split=combined_split,
        portfolio_metrics=portfolio_metrics,
        symbol_metrics=symbol_metrics,
        daily_returns=daily_returns,
        trade_log=trade_log,
    )


def run_zhang_equity_experiment(
    algo: str,
    total_timesteps: int | None,
    epoch_multiplier: int = DEFAULT_EPOCH_MULTIPLIER,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "zhang_equity",
    symbols: list[str] | tuple[str, ...] = ZHANG_EQUITY_SYMBOLS,
) -> dict[str, object]:
    output_root = Path(output_dir)
    artifacts = build_equity_artifacts(output_root, symbols=symbols)
    if isinstance(artifacts, dict):
        features = artifacts["features"]
        features["date"] = pd.to_datetime(features["date"])
    else:
        features = artifacts.features
    splits = zhang_equity_splits()
    reports_by_policy: dict[str, list[EvalReport]] = defaultdict(list)
    window_comparisons: list[pd.DataFrame] = []
    window_base_step_map: dict[str, int] = {}
    window_timestep_map: dict[str, int] = {}

    for window_label, train_split, eval_split in zhang_equity_windows():
        train_end = splits[train_split][1]
        test_start, test_end = splits[eval_split]
        window_splits = build_window_splits(
            features,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        base_training_steps = count_training_steps(features, symbols, window_splits["train_fit"])
        resolved_timesteps, eval_frequency = resolve_training_budget(
            base_training_steps=base_training_steps,
            total_timesteps=total_timesteps,
            epoch_multiplier=epoch_multiplier,
        )
        window_base_step_map[window_label] = int(base_training_steps)
        window_timestep_map[window_label] = int(resolved_timesteps)
        config = SB3TrainConfig(
            algo=algo,
            total_timesteps=int(resolved_timesteps),
            train_reward_mode="zhang",
            eval_reward_mode="raw",
        )
        experiment = run_sb3_experiment(
            feature_frame=features,
            config=config,
            train_split="train_fit",
            eval_split="test",
            symbols=list(symbols),
            splits=window_splits,
            validation_split="train_val",
            early_stopping_patience_evals=20,
            eval_freq=eval_frequency,
        )
        comparison = experiment["comparison"].copy()
        comparison["window"] = window_label
        window_comparisons.append(comparison)

        for policy_name, report in experiment["reports"].items():
            reports_by_policy[policy_name].append(report)

        save_experiment_outputs(
            experiment,
            output_root / "qa" / f"window_{window_label}_{algo}.csv",
        )

    combined_reports = {
        policy_name: aggregate_eval_reports(policy_name, policy_reports, combined_split="test_2011_2019_rolling")
        for policy_name, policy_reports in reports_by_policy.items()
    }
    combined_rows = [
        {"policy": policy_name, "split": "test_2011_2019_rolling", **report.portfolio_metrics}
        for policy_name, report in combined_reports.items()
    ]
    combined_comparison = (
        pd.DataFrame(combined_rows)
        .sort_values(["annualized_return", "sharpe"], ascending=False)
        .reset_index(drop=True)
    )

    combined_path = output_root / "qa" / f"combined_{algo}.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_comparison.to_csv(combined_path, index=False)

    metadata_path = output_root / "manifests" / "zhang_equity_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "symbols": list(symbols),
                "instrument_map": ZHANG_EQUITY_INSTRUMENT_MAP,
                "splits": splits,
                "windows": zhang_equity_windows(),
                "train_reward_mode": config.train_reward_mode,
                "eval_reward_mode": config.eval_reward_mode,
                "validation_fraction": 0.1,
                "early_stopping_patience_evals": 20,
                "epoch_multiplier": epoch_multiplier,
                "window_base_training_steps": window_base_step_map,
                "window_timesteps": window_timestep_map,
                "note": (
                    "Approximation of Zhang's equity-index methodology using Yahoo-mappable futures/index proxies. "
                    "Uses expanding training windows ending 2010 and 2015, holds out the last 10% of each training "
                    "window for validation, applies early stopping with patience 20 eval rounds, sets the default "
                    "training budget to base_training_steps * epoch_multiplier, evaluates validation once per base "
                    "training pass, then produces out-of-sample tests for 2011-2015 and 2016-2019."
                ),
            },
            indent=2,
        )
    )

    return {
        "artifacts": artifacts,
        "window_comparisons": pd.concat(window_comparisons, ignore_index=True),
        "combined_comparison": combined_comparison,
        "combined_reports": combined_reports,
        "combined_path": combined_path,
        "metadata_path": metadata_path,
        "output_dir": output_root,
        "window_base_training_steps": window_base_step_map,
        "window_timesteps": window_timestep_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an equity-index-only approximation of Zhang et al.'s rolling methodology."
    )
    parser.add_argument("--algo", choices=("dqn", "a2c", "ppo"), default="dqn")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--epoch-multiplier", type=int, default=DEFAULT_EPOCH_MULTIPLIER)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "zhang_equity"))
    args = parser.parse_args()

    result = run_zhang_equity_experiment(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        epoch_multiplier=args.epoch_multiplier,
        output_dir=args.output_dir,
    )
    print(f"algo={args.algo}")
    print(f"requested_total_timesteps={args.total_timesteps}")
    print(f"epoch_multiplier={args.epoch_multiplier}")
    print(f"window_base_training_steps={result['window_base_training_steps']}")
    print(f"window_timesteps={result['window_timesteps']}")
    print("combined_comparison:")
    print(result["combined_comparison"].to_string(index=False))
    print(f"combined_path={result['combined_path']}")
    print(f"metadata_path={result['metadata_path']}")


if __name__ == "__main__":
    main()
