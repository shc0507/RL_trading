"""Rolling Zhang-style asset-class experiments using Yahoo-mappable instruments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtest import EvalReport
from .config import DEFAULT_OUTPUT_DIR
from .data.pipeline import MarketDataPipeline
from .data.sources import PublicDailySource
from .features import FeatureBuilder
from .metrics import compute_performance_metrics
from .sb3_train import SB3TrainConfig, run_sb3_experiment, save_experiment_outputs


DEFAULT_FETCH_START = "2005-01-01"
DEFAULT_FETCH_END = "2019-12-31"
DEFAULT_EPOCH_MULTIPLIER = 20
DEFAULT_COST_RATE_BP = 20.0


@dataclass(frozen=True, slots=True)
class AssetClassSpec:
    name: str
    output_subdir: str
    instrument_map: dict[str, dict[str, str]]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.instrument_map.keys())


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

ZHANG_FIXED_INCOME_INSTRUMENT_MAP: dict[str, dict[str, str]] = {
    "FB": {"symbol": "FB", "yahoo_symbol": "ZF=F", "asset_class": "fixed_income_future", "currency": "USD"},
    "TY": {"symbol": "TY", "yahoo_symbol": "ZN=F", "asset_class": "fixed_income_future", "currency": "USD"},
    "US": {"symbol": "US", "yahoo_symbol": "ZB=F", "asset_class": "fixed_income_future", "currency": "USD"},
    "UB": {"symbol": "UB", "yahoo_symbol": "UB=F", "asset_class": "fixed_income_future", "currency": "USD"},
}

ZHANG_COMMODITIES_INSTRUMENT_MAP: dict[str, dict[str, str]] = {
    "CC": {"symbol": "CC", "yahoo_symbol": "CC=F", "asset_class": "commodity_future", "currency": "USD"},
    "OJ": {"symbol": "OJ", "yahoo_symbol": "OJ=F", "asset_class": "commodity_future", "currency": "USD"},
    "KC": {"symbol": "KC", "yahoo_symbol": "KC=F", "asset_class": "commodity_future", "currency": "USD"},
    "KE": {"symbol": "KE", "yahoo_symbol": "KE=F", "asset_class": "commodity_future", "currency": "USD"},
    "LBS": {"symbol": "LBS", "yahoo_symbol": "LBS=F", "asset_class": "commodity_future", "currency": "USD"},
    "ZR": {"symbol": "ZR", "yahoo_symbol": "ZR=F", "asset_class": "commodity_future", "currency": "USD"},
    "SB": {"symbol": "SB", "yahoo_symbol": "SB=F", "asset_class": "commodity_future", "currency": "USD"},
    "PA": {"symbol": "PA", "yahoo_symbol": "PA=F", "asset_class": "commodity_future", "currency": "USD"},
    "ZC": {"symbol": "ZC", "yahoo_symbol": "ZC=F", "asset_class": "commodity_future", "currency": "USD"},
    "GF": {"symbol": "GF", "yahoo_symbol": "GF=F", "asset_class": "commodity_future", "currency": "USD"},
    "GC": {"symbol": "GC", "yahoo_symbol": "GC=F", "asset_class": "commodity_future", "currency": "USD"},
    "HO": {"symbol": "HO", "yahoo_symbol": "HO=F", "asset_class": "commodity_future", "currency": "USD"},
    "SI": {"symbol": "SI", "yahoo_symbol": "SI=F", "asset_class": "commodity_future", "currency": "USD"},
    "HG": {"symbol": "HG", "yahoo_symbol": "HG=F", "asset_class": "commodity_future", "currency": "USD"},
    "ZL": {"symbol": "ZL", "yahoo_symbol": "ZL=F", "asset_class": "commodity_future", "currency": "USD"},
    "NG": {"symbol": "NG", "yahoo_symbol": "NG=F", "asset_class": "commodity_future", "currency": "USD"},
    "ZO": {"symbol": "ZO", "yahoo_symbol": "ZO=F", "asset_class": "commodity_future", "currency": "USD"},
    "PL": {"symbol": "PL", "yahoo_symbol": "PL=F", "asset_class": "commodity_future", "currency": "USD"},
    "LE": {"symbol": "LE", "yahoo_symbol": "LE=F", "asset_class": "commodity_future", "currency": "USD"},
    "CL": {"symbol": "CL", "yahoo_symbol": "CL=F", "asset_class": "commodity_future", "currency": "USD"},
    "ZW": {"symbol": "ZW", "yahoo_symbol": "ZW=F", "asset_class": "commodity_future", "currency": "USD"},
    "HE": {"symbol": "HE", "yahoo_symbol": "HE=F", "asset_class": "commodity_future", "currency": "USD"},
}

ASSET_CLASS_SPECS: dict[str, AssetClassSpec] = {
    "equity": AssetClassSpec("equity", "zhang_equity", ZHANG_EQUITY_INSTRUMENT_MAP),
    "fixed_income": AssetClassSpec("fixed_income", "zhang_fixed_income", ZHANG_FIXED_INCOME_INSTRUMENT_MAP),
    "commodities": AssetClassSpec("commodities", "zhang_commodities", ZHANG_COMMODITIES_INSTRUMENT_MAP),
}


def zhang_splits() -> dict[str, tuple[str, str]]:
    return {
        "train_pre_2011": (DEFAULT_FETCH_START, "2010-12-31"),
        "test_2011_2015": ("2011-01-01", "2015-12-31"),
        "train_pre_2016": (DEFAULT_FETCH_START, "2015-12-31"),
        "test_2016_2019": ("2016-01-01", "2019-12-31"),
    }


def zhang_windows() -> list[tuple[str, str, str]]:
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


def format_cost_tag(cost_rate_bp: float) -> str:
    if float(cost_rate_bp).is_integer():
        return f"{int(cost_rate_bp)}bp"
    normalized = str(cost_rate_bp).replace(".", "p")
    return f"{normalized}bp"


def build_asset_class_artifacts(
    instrument_map: dict[str, dict[str, str]],
    output_dir: str | Path,
    start_date: str = DEFAULT_FETCH_START,
    end_date: str = DEFAULT_FETCH_END,
    symbols: Iterable[str] | None = None,
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
    source = PublicDailySource(instrument_map=instrument_map)
    feature_builder = FeatureBuilder()
    requested_symbols = list(symbols) if symbols is not None else list(instrument_map.keys())
    return pipeline.build(
        source=source,
        feature_builder=feature_builder,
        symbols=requested_symbols,
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


def symbol_has_minimum_rows(
    feature_frame: pd.DataFrame,
    symbol: str,
    split_range: tuple[str, str],
    minimum_rows: int = 2,
) -> bool:
    start_date, end_date = split_range
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    symbol_frame = frame.loc[frame["symbol"] == symbol]
    usable = symbol_frame.loc[
        (symbol_frame["date"] >= pd.Timestamp(start_date))
        & (symbol_frame["date"] <= pd.Timestamp(end_date))
        & symbol_frame["window_ready"]
    ]
    return len(usable) >= minimum_rows


def eligible_symbols_for_windows(
    feature_frame: pd.DataFrame,
    symbols: Iterable[str],
    windows: list[tuple[str, str, str]],
    splits: dict[str, tuple[str, str]],
) -> tuple[list[str], dict[str, list[str]]]:
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    eligible: list[str] = []
    dropped: dict[str, list[str]] = {}

    for symbol in symbols:
        missing: list[str] = []
        for _, train_split, eval_split in windows:
            train_end = splits[train_split][1]
            test_start, test_end = splits[eval_split]
            window_splits = build_window_splits(frame, train_end=train_end, test_start=test_start, test_end=test_end)
            for split_name, split_range in window_splits.items():
                if not symbol_has_minimum_rows(frame, symbol, split_range):
                    missing.append(f"{train_split}:{split_name}")
        if missing:
            dropped[symbol] = missing
        else:
            eligible.append(symbol)
    return eligible, dropped


def run_zhang_asset_class_experiment(
    asset_class: str,
    algo: str,
    total_timesteps: int | None,
    epoch_multiplier: int = DEFAULT_EPOCH_MULTIPLIER,
    cost_rate_bp: float = DEFAULT_COST_RATE_BP,
    output_dir: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    spec = ASSET_CLASS_SPECS[asset_class]
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR / f"{spec.output_subdir}_{format_cost_tag(cost_rate_bp)}"
    )
    requested_symbols = list(symbols) if symbols is not None else list(spec.symbols)
    artifacts = build_asset_class_artifacts(spec.instrument_map, output_root, symbols=requested_symbols)
    features = artifacts["features"] if isinstance(artifacts, dict) else artifacts.features
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])

    splits = zhang_splits()
    windows = zhang_windows()
    eligible_symbols, dropped_symbols = eligible_symbols_for_windows(features, requested_symbols, windows, splits)
    if not eligible_symbols:
        raise ValueError(f"no eligible symbols found for asset class {asset_class}")

    reports_by_policy: dict[str, list[EvalReport]] = defaultdict(list)
    window_comparisons: list[pd.DataFrame] = []
    window_base_step_map: dict[str, int] = {}
    window_timestep_map: dict[str, int] = {}

    for window_label, train_split, eval_split in windows:
        train_end = splits[train_split][1]
        test_start, test_end = splits[eval_split]
        window_splits = build_window_splits(
            features,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        base_training_steps = count_training_steps(features, eligible_symbols, window_splits["train_fit"])
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
            cost_rate_bp=float(cost_rate_bp),
            train_reward_mode="zhang",
            eval_reward_mode="raw",
        )
        experiment = run_sb3_experiment(
            feature_frame=features,
            config=config,
            train_split="train_fit",
            eval_split="test",
            symbols=eligible_symbols,
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

    metadata_path = output_root / "manifests" / f"zhang_{asset_class}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "asset_class": asset_class,
                "requested_symbols": requested_symbols,
                "eligible_symbols": eligible_symbols,
                "dropped_symbols": dropped_symbols,
                "instrument_map": spec.instrument_map,
                "splits": splits,
                "windows": windows,
                "train_reward_mode": config.train_reward_mode,
                "eval_reward_mode": config.eval_reward_mode,
                "cost_rate_bp": cost_rate_bp,
                "validation_fraction": 0.1,
                "early_stopping_patience_evals": 20,
                "epoch_multiplier": epoch_multiplier,
                "window_base_training_steps": window_base_step_map,
                "window_timesteps": window_timestep_map,
                "note": (
                    f"Approximation of Zhang's {asset_class.replace('_', ' ')} methodology using Yahoo-mappable "
                    "contracts or proxies. Uses expanding training windows ending 2010 and 2015, holds out the "
                    "last 10% of each training window for validation, applies early stopping with patience 20 eval "
                    "rounds, and sets the default training budget to base_training_steps * epoch_multiplier."
                ),
            },
            indent=2,
        )
    )

    return {
        "asset_class": asset_class,
        "artifacts": artifacts,
        "window_comparisons": pd.concat(window_comparisons, ignore_index=True),
        "combined_comparison": combined_comparison,
        "combined_reports": combined_reports,
        "combined_path": combined_path,
        "metadata_path": metadata_path,
        "output_dir": output_root,
        "requested_symbols": requested_symbols,
        "eligible_symbols": eligible_symbols,
        "dropped_symbols": dropped_symbols,
        "window_base_training_steps": window_base_step_map,
        "window_timesteps": window_timestep_map,
    }


def build_arg_parser(default_asset_class: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Zhang-style rolling asset-class experiment with SB3 models."
    )
    parser.add_argument("--asset-class", choices=tuple(ASSET_CLASS_SPECS.keys()), default=default_asset_class)
    parser.add_argument("--algo", choices=("dqn", "a2c", "ppo"), default="dqn")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--epoch-multiplier", type=int, default=DEFAULT_EPOCH_MULTIPLIER)
    parser.add_argument("--cost-rate-bp", type=float, default=DEFAULT_COST_RATE_BP)
    parser.add_argument("--output-dir", default=None)
    return parser


def run_cli(default_asset_class: str | None = None) -> None:
    parser = build_arg_parser(default_asset_class=default_asset_class)
    args = parser.parse_args()
    if args.asset_class is None:
        parser.error("--asset-class is required")

    result = run_zhang_asset_class_experiment(
        asset_class=args.asset_class,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        epoch_multiplier=args.epoch_multiplier,
        cost_rate_bp=args.cost_rate_bp,
        output_dir=args.output_dir,
    )
    print(f"asset_class={args.asset_class}")
    print(f"algo={args.algo}")
    print(f"requested_total_timesteps={args.total_timesteps}")
    print(f"epoch_multiplier={args.epoch_multiplier}")
    print(f"cost_rate_bp={args.cost_rate_bp}")
    print(f"requested_symbols={result['requested_symbols']}")
    print(f"eligible_symbols={result['eligible_symbols']}")
    print(f"dropped_symbols={result['dropped_symbols']}")
    print(f"window_base_training_steps={result['window_base_training_steps']}")
    print(f"window_timesteps={result['window_timesteps']}")
    print("combined_comparison:")
    print(result["combined_comparison"].to_string(index=False))
    print(f"combined_path={result['combined_path']}")
    print(f"metadata_path={result['metadata_path']}")


if __name__ == "__main__":
    run_cli()
