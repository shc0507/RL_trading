"""Production Zhang-style walk-forward orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from math import ceil
from pathlib import Path
import time

import pandas as pd

from ..config import DEFAULT_OUTPUT_DIR
from ..data.pipeline import MarketDataPipeline
from ..data.sources import InstitutionalCsvSource
from ..features import FeatureBuilder
from ..metrics import compute_performance_metrics
from ..zhang_universe import load_universe_frame, validate_universe_frame
from .plots import generate_zhang_style_plots
from .production_a2c import ProductionA2CConfig, ProductionA2CTrainer
from .production_pg import ProductionPGConfig, ProductionPGTrainer
from .trainer import DQNConfig, _coerce_loaded_feature_frame, train_dqn as train_dqn_legacy
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation


BASELINE_POLICIES = ("long_only", "sign_12m", "macd")
PRODUCTION_POLICIES = ("dqn", "pg", "a2c", *BASELINE_POLICIES)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(slots=True)
class WalkForwardConfig:
    train_start: str = "2005-01-01"
    anchors: tuple[str, ...] = ("2010-12-31", "2015-12-31")
    dataset_end: str = "2019-12-31"
    test_horizon_years: int = 5
    group_by: str = "asset_class"
    selection_mode: str = "cv_best_checkpoint"

    def windows(self) -> tuple[WalkForwardWindow, ...]:
        dataset_end = pd.Timestamp(self.dataset_end)
        windows: list[WalkForwardWindow] = []
        for anchor_value in self.anchors:
            anchor = pd.Timestamp(anchor_value)
            test_start = anchor + pd.offsets.BDay(1)
            scheduled_end = anchor + pd.DateOffset(years=self.test_horizon_years)
            test_end = min(dataset_end, scheduled_end)
            windows.append(
                WalkForwardWindow(
                    name=f"wf_{anchor.year}",
                    train_start=self.train_start,
                    train_end=anchor.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                )
            )
        for index, current in enumerate(windows):
            current_test_start = pd.Timestamp(current.test_start)
            current_test_end = pd.Timestamp(current.test_end)
            if current_test_end < current_test_start:
                raise ValueError(f"walk-forward window {current.name} has an empty test range")
            if index == 0:
                continue
            previous_test_end = pd.Timestamp(windows[index - 1].test_end)
            if current_test_start <= previous_test_end:
                raise ValueError("walk-forward test windows overlap")
        return tuple(windows)


@dataclass(slots=True)
class ProductionDatasetConfig:
    data_root: str | Path | None = None
    source_dataset_dir: str | Path | None = None
    output_dir: str | Path = Path(DEFAULT_OUTPUT_DIR) / "zhang_dataset"
    start_date: str = "2005-01-01"
    end_date: str = "2019-12-31"
    universe_manifest_path: str | Path | None = None
    source_name: str = "zhang_institutional_csv"


@dataclass(slots=True)
class ProductionDQNConfig:
    optimizer_updates: int = 250_000
    batch_size: int = 64
    replay_capacity: int = 5_000
    warmup_steps: int = 1_000
    train_every: int = 1
    target_update_interval: int = 1_000
    gamma: float = 0.3
    learning_rate: float = 1e-4
    head_hidden_sizes: tuple[int, ...] = (32,)
    activation: str = "leaky_relu"
    recurrent_layer_sizes: tuple[int, ...] = (64, 32)
    recurrent_dropout: float = 0.1
    head_dropout: float = 0.1
    cost_rate_bp: float = 20.0
    vol_target: float = 0.15
    seed: int = 101
    device: str = "auto"
    search_grid: tuple[dict[str, object], ...] | None = None


@dataclass(slots=True)
class ProductionSuiteConfig:
    dataset: ProductionDatasetConfig = field(default_factory=ProductionDatasetConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    dqn: ProductionDQNConfig = field(default_factory=ProductionDQNConfig)
    pg: ProductionPGConfig = field(default_factory=ProductionPGConfig)
    a2c: ProductionA2CConfig = field(default_factory=ProductionA2CConfig)
    output_dir: str | Path = Path(DEFAULT_OUTPUT_DIR) / "zhang_production"
    algorithms: tuple[str, ...] = ("dqn", "pg", "a2c")
    policies_for_plots: tuple[str, ...] = PRODUCTION_POLICIES
    cv_fraction: float = 0.10
    early_stopping_patience_epochs: int = 20
    selection_metric: str = "sharpe"
    selection_split: str = "cv"
    protocol_version: str = "newer_zhang"


@dataclass(slots=True)
class ProductionRunArtifacts:
    root_dir: Path
    source_dataset_dir: Path
    combined_rl_dir: Path
    individual_run_summaries_path: Path
    leaderboard_path: Path
    run_metadata_path: Path
    zhang_eval_dir: Path
    plot_paths: dict[str, Path]


def _write_frame_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _load_dataset_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if path.name == "features.csv":
        return _coerce_loaded_feature_frame(frame)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _first_business_day(date_value: str) -> pd.Timestamp:
    return pd.bdate_range(pd.Timestamp(date_value), periods=1)[0]


def validate_production_dataset(
    bars: pd.DataFrame,
    instruments: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> None:
    if instruments["symbol"].duplicated().any():
        duplicates = instruments.loc[instruments["symbol"].duplicated(), "symbol"].astype(str).tolist()
        raise ValueError(f"production instruments contain duplicate symbols: {duplicates[:5]}")

    expected_symbols = universe["symbol"].astype(str)
    observed_symbols = instruments["symbol"].astype(str)
    missing_symbols = sorted(set(expected_symbols) - set(observed_symbols))
    unexpected_symbols = sorted(set(observed_symbols) - set(expected_symbols))
    if missing_symbols or unexpected_symbols:
        raise ValueError(
            f"production instruments do not match the canonical universe: missing={missing_symbols[:5]}, unexpected={unexpected_symbols[:5]}"
        )

    merged = instruments.merge(
        universe.loc[:, ["symbol", "asset_class", "currency"]],
        on="symbol",
        suffixes=("_observed", "_expected"),
        how="outer",
    )
    if merged[["symbol"]].isna().any().any():
        raise ValueError("production instruments contain missing universe rows")
    if (merged["asset_class_observed"] != merged["asset_class_expected"]).any():
        raise ValueError("production instruments contain asset-class mismatches")
    if (merged["currency_observed"] != merged["currency_expected"]).any():
        raise ValueError("production instruments contain currency mismatches")

    required_start = _first_business_day(start_date)
    required_end = pd.Timestamp(end_date)
    coverage = (
        bars.groupby("symbol", as_index=False)
        .agg(min_date=("date", "min"), max_date=("date", "max"))
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    if (coverage["min_date"] > required_start).any():
        missing = coverage.loc[coverage["min_date"] > required_start, "symbol"].tolist()
        raise ValueError(f"production bars start too late for symbols: {missing[:5]}")
    if (coverage["max_date"] < required_end).any():
        missing = coverage.loc[coverage["max_date"] < required_end, "symbol"].tolist()
        raise ValueError(f"production bars end too early for symbols: {missing[:5]}")


def _canonicalize_instruments(instruments: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    observed = instruments.copy()
    observed["symbol"] = observed["symbol"].astype(str)
    canonical = universe.loc[:, ["symbol", "asset_class", "currency"]].copy()
    extra_columns = [column for column in observed.columns if column not in canonical.columns]
    if extra_columns:
        canonical = canonical.merge(observed.loc[:, ["symbol", *extra_columns]], on="symbol", how="left")
    return canonical.reset_index(drop=True)


def prepare_production_dataset(
    config: ProductionDatasetConfig,
    walk_forward: WalkForwardConfig | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = load_universe_frame(config.universe_manifest_path)
    validate_universe_frame(universe)

    if config.source_dataset_dir is not None:
        dataset_dir = Path(config.source_dataset_dir)
        bars = _load_dataset_frame(dataset_dir / "processed" / "bars.csv")
        features = _load_dataset_frame(dataset_dir / "processed" / "features.csv")
        instruments = _canonicalize_instruments(_load_dataset_frame(dataset_dir / "processed" / "instruments.csv"), universe)
        validate_production_dataset(
            bars=bars,
            instruments=instruments,
            universe=universe,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        return dataset_dir, bars, features, instruments, universe

    if config.data_root is None:
        raise ValueError("either data_root or source_dataset_dir is required for the production dataset")

    dataset_dir = Path(config.output_dir)
    source = InstitutionalCsvSource(root=config.data_root, source_name=config.source_name, strict=True)
    split_config = (
        {f"test_{window.name}": (window.test_start, window.test_end) for window in (walk_forward.windows() if walk_forward else ())}
        or {"train": (config.start_date, config.end_date)}
    )
    pipeline = MarketDataPipeline(
        output_dir=dataset_dir,
        max_symbol_end_lag_business_days=0,
        min_symbol_calendar_coverage=0.9,
    )
    artifacts = pipeline.build(
        source=source,
        feature_builder=FeatureBuilder(),
        symbols=universe["symbol"].astype(str).tolist(),
        start_date=config.start_date,
        end_date=config.end_date,
        splits=split_config,
    )
    validate_production_dataset(
        bars=artifacts.bars,
        instruments=pd.read_csv(artifacts.instruments_path),
        universe=universe,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    universe_path = dataset_dir / "manifests" / "universe.csv"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame_csv_atomic(universe_path, universe)
    instruments = _canonicalize_instruments(pd.read_csv(artifacts.instruments_path), universe)
    _write_frame_csv_atomic(artifacts.instruments_path, instruments)
    return dataset_dir, artifacts.bars.copy(), artifacts.features.copy(), instruments, universe


def _symbols_by_asset_class(instruments: pd.DataFrame) -> dict[str, list[str]]:
    return {
        str(asset_class): frame["symbol"].astype(str).tolist()
        for asset_class, frame in instruments.groupby("asset_class", sort=False)
    }


def _default_dqn_search_grid(config: ProductionDQNConfig) -> tuple[dict[str, object], ...]:
    return config.search_grid or (
        {
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "recurrent_dropout": config.recurrent_dropout,
            "head_dropout": config.head_dropout,
        },
        {
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "recurrent_dropout": min(config.recurrent_dropout + 0.1, 0.5),
            "head_dropout": min(config.head_dropout + 0.1, 0.5),
        },
    )


def _default_pg_search_grid(config: ProductionPGConfig) -> tuple[dict[str, object], ...]:
    return config.search_grid or (
        {
            "actor_lr": config.actor_lr,
            "gamma": config.gamma,
            "recurrent_dropout": config.recurrent_dropout,
            "head_dropout": config.head_dropout,
            "entropy_coef": config.entropy_coef,
        },
        {
            "actor_lr": config.actor_lr * 0.5,
            "gamma": config.gamma,
            "recurrent_dropout": min(config.recurrent_dropout + 0.1, 0.5),
            "head_dropout": min(config.head_dropout + 0.1, 0.5),
            "entropy_coef": config.entropy_coef,
        },
    )


def _default_a2c_search_grid(config: ProductionA2CConfig) -> tuple[dict[str, object], ...]:
    return config.search_grid or (
        {
            "actor_lr": config.actor_lr,
            "critic_lr": config.critic_lr,
            "gamma": config.gamma,
            "recurrent_dropout": config.recurrent_dropout,
            "head_dropout": config.head_dropout,
            "entropy_coef": config.entropy_coef,
        },
        {
            "actor_lr": config.actor_lr * 0.5,
            "critic_lr": config.critic_lr,
            "gamma": config.gamma,
            "recurrent_dropout": min(config.recurrent_dropout + 0.1, 0.5),
            "head_dropout": min(config.head_dropout + 0.1, 0.5),
            "entropy_coef": config.entropy_coef,
        },
    )


def _build_train_cv_test_splits(
    feature_frame: pd.DataFrame,
    symbols: list[str],
    window: WalkForwardWindow,
    cv_fraction: float,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    train_frame = feature_frame.loc[
        feature_frame["symbol"].isin(symbols)
        & (feature_frame["date"] >= pd.Timestamp(window.train_start))
        & (feature_frame["date"] <= pd.Timestamp(window.train_end))
        & feature_frame["window_ready"]
    ].copy()
    unique_dates = pd.DatetimeIndex(sorted(train_frame["date"].dropna().unique()))
    if len(unique_dates) < 2:
        raise ValueError(f"not enough in-sample dates to build cv split for window {window.name}")
    cv_count = max(1, int(ceil(len(unique_dates) * float(cv_fraction))))
    cv_count = min(cv_count, len(unique_dates) - 1)
    train_core_end = unique_dates[-cv_count - 1]
    cv_start = unique_dates[-cv_count]
    splits = {
        "train": (window.train_start, train_core_end.strftime("%Y-%m-%d")),
        "cv": (cv_start.strftime("%Y-%m-%d"), window.train_end),
        "test": (window.test_start, window.test_end),
    }
    metadata = {
        "train_core_start": window.train_start,
        "train_core_end": train_core_end.strftime("%Y-%m-%d"),
        "cv_start": cv_start.strftime("%Y-%m-%d"),
        "cv_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
    }
    return splits, metadata


def _epoch_steps_for_features(feature_frame: pd.DataFrame, symbols: list[str], train_range: tuple[str, str]) -> int:
    start_date, end_date = train_range
    frame = feature_frame.loc[
        feature_frame["symbol"].isin(symbols)
        & (feature_frame["date"] >= pd.Timestamp(start_date))
        & (feature_frame["date"] <= pd.Timestamp(end_date))
        & feature_frame["window_ready"]
    ]
    return max(1, int(len(frame)))


def _production_dqn_config(
    config: ProductionDQNConfig,
    *,
    selection_metric: str,
    selection_split: str,
    early_stopping_patience_epochs: int,
    epoch_steps: int,
    overrides: dict[str, object] | None = None,
) -> DQNConfig:
    merged = {**asdict(config), **(overrides or {})}
    return DQNConfig(
        policy_name="dqn",
        total_steps=int(merged["optimizer_updates"]) + int(merged["warmup_steps"]),
        optimizer_updates=int(merged["optimizer_updates"]),
        batch_size=int(merged["batch_size"]),
        replay_capacity=int(merged["replay_capacity"]),
        warmup_steps=int(merged["warmup_steps"]),
        train_every=int(merged["train_every"]),
        target_update_interval=int(merged["target_update_interval"]),
        gamma=float(merged["gamma"]),
        learning_rate=float(merged["learning_rate"]),
        hidden_sizes=tuple(config.head_hidden_sizes),
        activation=str(merged["activation"]),
        double_dqn=True,
        dueling=True,
        network_type="lstm",
        recurrent_layer_sizes=tuple(config.recurrent_layer_sizes),
        recurrent_dropout=float(merged["recurrent_dropout"]),
        head_dropout=float(merged["head_dropout"]),
        validation_interval=0,
        selection_mode="validation",
        validation_split_name=selection_split,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        epoch_steps=epoch_steps,
        reward_mode="zhang",
        cost_rate_bp=float(merged["cost_rate_bp"]),
        vol_target=float(merged["vol_target"]),
        seed=int(merged["seed"]),
        device=str(merged["device"]),
        checkpoint_metric=selection_metric,
    )


def _production_pg_config(
    config: ProductionPGConfig,
    *,
    selection_metric: str,
    selection_split: str,
    early_stopping_patience_epochs: int,
    epoch_steps: int,
    overrides: dict[str, object] | None = None,
) -> ProductionPGConfig:
    merged = {**asdict(config), **(overrides or {})}
    return replace(
        config,
        actor_lr=float(merged["actor_lr"]),
        gamma=float(merged["gamma"]),
        recurrent_dropout=float(merged["recurrent_dropout"]),
        head_dropout=float(merged["head_dropout"]),
        entropy_coef=float(merged["entropy_coef"]),
        validation_split_name=selection_split,
        selection_metric=selection_metric,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        epoch_steps=epoch_steps,
    )


def _production_a2c_config(
    config: ProductionA2CConfig,
    *,
    selection_metric: str,
    selection_split: str,
    early_stopping_patience_epochs: int,
    epoch_steps: int,
    overrides: dict[str, object] | None = None,
) -> ProductionA2CConfig:
    merged = {**asdict(config), **(overrides or {})}
    return replace(
        config,
        actor_lr=float(merged["actor_lr"]),
        critic_lr=float(merged["critic_lr"]),
        gamma=float(merged["gamma"]),
        recurrent_dropout=float(merged["recurrent_dropout"]),
        head_dropout=float(merged["head_dropout"]),
        entropy_coef=float(merged["entropy_coef"]),
        validation_split_name=selection_split,
        selection_metric=selection_metric,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        epoch_steps=epoch_steps,
    )


def _extract_policy(path: Path, split: str) -> str:
    suffix = f"_{split}_trades.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected report filename {path.name}")
    return path.name[: -len(suffix)]


def _consolidate_policy_reports(policy: str, trade_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    trade_log = trade_log.sort_values(["date", "symbol"]).reset_index(drop=True)
    daily_returns = (
        trade_log.groupby("date", as_index=False)
        .agg(
            portfolio_return=("trade_return", "mean"),
            avg_turnover=("scaled_turnover", "mean"),
            total_cost=("trade_cost", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_returns["portfolio_cumulative_trade_return"] = daily_returns["portfolio_return"].fillna(0.0).cumsum()
    daily_returns["cumulative_trade_return"] = daily_returns["portfolio_cumulative_trade_return"]
    symbol_metrics_rows: list[dict[str, object]] = []
    for symbol, frame in trade_log.groupby("symbol", sort=False):
        metrics = compute_performance_metrics(
            frame["trade_return"],
            turnover=frame["scaled_turnover"],
            total_cost=float(frame["trade_cost"].sum()),
        )
        metrics["symbol"] = symbol
        symbol_metrics_rows.append(metrics)
    symbol_metrics = pd.DataFrame(symbol_metrics_rows).sort_values("symbol").reset_index(drop=True)
    portfolio_metrics = compute_performance_metrics(
        daily_returns["portfolio_return"],
        turnover=daily_returns["avg_turnover"],
        total_cost=float(daily_returns["total_cost"].sum()),
    )
    return daily_returns, symbol_metrics, portfolio_metrics


def _metric_from_summary(summary_metrics_path: Path, policy_name: str, split: str, metric: str) -> float:
    summary = pd.read_csv(summary_metrics_path)
    row = summary.loc[(summary["policy"] == policy_name) & (summary["split"] == split)]
    if row.empty:
        raise ValueError(f"missing {policy_name}/{split} row in {summary_metrics_path}")
    return float(row.iloc[0][metric])


def _train_candidate(
    *,
    algorithm: str,
    suite: ProductionSuiteConfig,
    feature_frame: pd.DataFrame,
    symbols: list[str],
    splits: dict[str, tuple[str, str]],
    epoch_steps: int,
    run_dir: Path,
    overrides: dict[str, object],
) -> tuple[Path, Path, float]:
    if algorithm == "dqn":
        artifacts = train_dqn_legacy(
            output_dir=run_dir,
            config=_production_dqn_config(
                suite.dqn,
                selection_metric=suite.selection_metric,
                selection_split=suite.selection_split,
                early_stopping_patience_epochs=suite.early_stopping_patience_epochs,
                epoch_steps=epoch_steps,
                overrides=overrides,
            ),
            feature_frame=feature_frame,
            splits=splits,
            symbols=symbols,
        )
        policy_name = "dqn"
    elif algorithm == "pg":
        trainer = ProductionPGTrainer(
            feature_frame=feature_frame,
            output_dir=run_dir,
            config=_production_pg_config(
                suite.pg,
                selection_metric=suite.selection_metric,
                selection_split=suite.selection_split,
                early_stopping_patience_epochs=suite.early_stopping_patience_epochs,
                epoch_steps=epoch_steps,
                overrides=overrides,
            ),
            splits=splits,
            symbols=symbols,
        )
        artifacts = trainer.train()
        policy_name = "pg"
    elif algorithm == "a2c":
        trainer = ProductionA2CTrainer(
            feature_frame=feature_frame,
            output_dir=run_dir,
            config=_production_a2c_config(
                suite.a2c,
                selection_metric=suite.selection_metric,
                selection_split=suite.selection_split,
                early_stopping_patience_epochs=suite.early_stopping_patience_epochs,
                epoch_steps=epoch_steps,
                overrides=overrides,
            ),
            splits=splits,
            symbols=symbols,
        )
        artifacts = trainer.train()
        policy_name = "a2c"
    else:
        raise ValueError(f"unsupported production algorithm {algorithm}")

    metric = _metric_from_summary(Path(artifacts.summary_metrics_path), policy_name, suite.selection_split, suite.selection_metric)
    return Path(artifacts.artifact_dir), Path(artifacts.summary_metrics_path), metric


def _candidate_grid(suite: ProductionSuiteConfig, algorithm: str) -> tuple[dict[str, object], ...]:
    if algorithm == "dqn":
        return _default_dqn_search_grid(suite.dqn)
    if algorithm == "pg":
        return _default_pg_search_grid(suite.pg)
    if algorithm == "a2c":
        return _default_a2c_search_grid(suite.a2c)
    raise ValueError(f"unsupported algorithm {algorithm}")


def run_production_suite(config: ProductionSuiteConfig | None = None) -> ProductionRunArtifacts:
    suite = config or ProductionSuiteConfig()
    start_time = time.time()
    root_dir = Path(suite.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir, _, features, instruments, universe = prepare_production_dataset(config=suite.dataset, walk_forward=suite.walk_forward)
    asset_class_symbols = _symbols_by_asset_class(instruments)
    windows = suite.walk_forward.windows()

    run_records: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for window in windows:
        for asset_class, symbols in asset_class_symbols.items():
            subset_features = features.loc[features["symbol"].isin(symbols)].copy()
            splits, split_metadata = _build_train_cv_test_splits(subset_features, symbols, window, suite.cv_fraction)
            split_rows.append({"window_name": window.name, "asset_class": asset_class, **split_metadata})
            epoch_steps = _epoch_steps_for_features(subset_features, symbols, splits["train"])

            for algorithm in suite.algorithms:
                algorithm_dir = root_dir / window.name / asset_class / algorithm
                algorithm_dir.mkdir(parents=True, exist_ok=True)
                candidate_rows: list[dict[str, object]] = []
                best_candidate: dict[str, object] | None = None
                best_metric = float("-inf")
                for index, overrides in enumerate(_candidate_grid(suite, algorithm), start=1):
                    candidate_dir = algorithm_dir / "hpo" / f"candidate_{index:02d}"
                    artifact_dir, summary_metrics_path, metric = _train_candidate(
                        algorithm=algorithm,
                        suite=suite,
                        feature_frame=subset_features,
                        symbols=symbols,
                        splits=splits,
                        epoch_steps=epoch_steps,
                        run_dir=candidate_dir,
                        overrides=overrides,
                    )
                    row = {
                        "window_name": window.name,
                        "asset_class": asset_class,
                        "algorithm": algorithm,
                        "candidate_index": index,
                        "metric": metric,
                        "selection_metric": suite.selection_metric,
                        "summary_metrics_path": str(summary_metrics_path),
                        "artifact_dir": str(artifact_dir),
                        **overrides,
                    }
                    candidate_rows.append(row)
                    if metric > best_metric:
                        best_metric = metric
                        best_candidate = row
                if best_candidate is None:
                    raise ValueError(f"no candidate completed for {algorithm} {window.name} {asset_class}")
                hpo_results_path = algorithm_dir / "hpo_results.csv"
                _write_frame_csv_atomic(hpo_results_path, pd.DataFrame(candidate_rows))
                final_dir = algorithm_dir / "final"
                final_artifact_dir, final_summary_path, final_cv_metric = _train_candidate(
                    algorithm=algorithm,
                    suite=suite,
                    feature_frame=subset_features,
                    symbols=symbols,
                    splits=splits,
                    epoch_steps=epoch_steps,
                    run_dir=final_dir,
                    overrides={key: value for key, value in best_candidate.items() if key not in {"window_name", "asset_class", "algorithm", "candidate_index", "metric", "selection_metric", "summary_metrics_path", "artifact_dir"}},
                )
                selected_rows.append(
                    {
                        "window_name": window.name,
                        "asset_class": asset_class,
                        "algorithm": algorithm,
                        "selected_candidate_index": int(best_candidate["candidate_index"]),
                        "selection_metric": suite.selection_metric,
                        "selection_split": suite.selection_split,
                        "cv_metric": best_metric,
                        "final_cv_metric": final_cv_metric,
                        "hpo_results_path": str(hpo_results_path),
                        "final_artifact_dir": str(final_artifact_dir),
                        **{key: value for key, value in best_candidate.items() if key not in {"window_name", "asset_class", "algorithm", "candidate_index", "metric", "selection_metric", "summary_metrics_path", "artifact_dir"}},
                    }
                )
                run_records.append(
                    {
                        "window_name": window.name,
                        "asset_class": asset_class,
                        "algorithm": algorithm,
                        "artifact_dir": final_artifact_dir,
                        "summary_metrics_path": final_summary_path,
                    }
                )

    selected_hyperparameters_path = root_dir / "selected_hyperparameters.csv"
    _write_frame_csv_atomic(selected_hyperparameters_path, pd.DataFrame(selected_rows))
    split_manifest_path = root_dir / "split_manifest.csv"
    _write_frame_csv_atomic(split_manifest_path, pd.DataFrame(split_rows))

    individual_frames: list[pd.DataFrame] = []
    combined_policy_frames: dict[str, list[pd.DataFrame]] = {}
    for record in run_records:
        summary = pd.read_csv(record["summary_metrics_path"])
        summary["window_name"] = record["window_name"]
        summary["training_asset_class"] = record["asset_class"]
        summary["algorithm_family"] = record["algorithm"]
        individual_frames.append(summary)

        report_dir = Path(record["artifact_dir"]) / "reports"
        for trade_path in sorted(report_dir.glob("*_test_trades.csv")):
            policy = _extract_policy(trade_path, split="test")
            if policy not in PRODUCTION_POLICIES:
                continue
            trade_frame = pd.read_csv(trade_path)
            trade_frame["date"] = pd.to_datetime(trade_frame["date"])
            trade_frame["next_date"] = pd.to_datetime(trade_frame["next_date"])
            trade_frame["window_name"] = record["window_name"]
            trade_frame["training_asset_class"] = record["asset_class"]
            combined_policy_frames.setdefault(policy, []).append(trade_frame)

    individual_run_summaries = pd.concat(individual_frames, ignore_index=True).sort_values(
        ["window_name", "training_asset_class", "split", "policy"]
    ).reset_index(drop=True)
    individual_run_summaries_path = root_dir / "individual_run_summaries.csv"
    _write_frame_csv_atomic(individual_run_summaries_path, individual_run_summaries)

    combined_root = root_dir / "combined"
    combined_rl_dir = combined_root / "rl"
    combined_reports_dir = combined_rl_dir / "reports"
    combined_processed_dir = combined_root / "processed"
    combined_reports_dir.mkdir(parents=True, exist_ok=True)
    combined_processed_dir.mkdir(parents=True, exist_ok=True)
    _write_frame_csv_atomic(combined_processed_dir / "instruments.csv", universe.loc[:, ["symbol", "asset_class", "currency"]])

    combined_summary_rows: list[dict[str, object]] = []
    for policy, frames in combined_policy_frames.items():
        trade_log = pd.concat(frames, ignore_index=True)
        trade_log = trade_log.drop_duplicates(subset=["date", "next_date", "symbol", "split"]).reset_index(drop=True)
        daily_returns, symbol_metrics, portfolio_metrics = _consolidate_policy_reports(policy=policy, trade_log=trade_log)
        _write_frame_csv_atomic(combined_reports_dir / f"{policy}_test_trades.csv", trade_log)
        _write_frame_csv_atomic(combined_reports_dir / f"{policy}_test_daily_returns.csv", daily_returns)
        _write_frame_csv_atomic(combined_reports_dir / f"{policy}_test_symbol_metrics.csv", symbol_metrics)
        combined_summary_rows.append({"policy": policy, "split": "test", **portfolio_metrics})

    combined_summary = pd.DataFrame(combined_summary_rows).sort_values(["split", "policy"]).reset_index(drop=True)
    _write_frame_csv_atomic(combined_rl_dir / "summary_metrics.csv", combined_summary)
    _write_frame_csv_atomic(combined_rl_dir / "training_curve.csv", pd.DataFrame())

    zhang_eval_artifacts = run_zhang_evaluation(
        artifact_dir=combined_rl_dir,
        config=ZhangEvalConfig(target_vol=suite.dqn.vol_target, split=None, return_column="trade_return"),
    )
    plot_artifacts = generate_zhang_style_plots(
        artifact_dir=combined_rl_dir,
        split="test",
        policies=list(suite.policies_for_plots),
        target_vol=suite.dqn.vol_target,
        return_column="trade_return",
    )

    summary_scaled = pd.read_csv(zhang_eval_artifacts.summary_scaled_path)
    leaderboard = (
        summary_scaled.loc[(summary_scaled["split"] == "test") & (summary_scaled["asset_group"] == "all")]
        .sort_values(["sharpe", "annualized_return"], ascending=[False, False])
        .reset_index(drop=True)
    )
    leaderboard_path = root_dir / "test_all_scaled_leaderboard.csv"
    _write_frame_csv_atomic(leaderboard_path, leaderboard)

    plot_paths = {
        "cumulative": plot_artifacts.cumulative_returns_path,
        "diagnostics": plot_artifacts.symbol_diagnostics_path,
    }
    if plot_artifacts.cost_sweep_path is not None:
        plot_paths["cost_sweep"] = plot_artifacts.cost_sweep_path

    run_metadata_path = root_dir / "run_metadata.json"
    _write_json_atomic(
        run_metadata_path,
        {
            "dataset": "zhang_institutional_csv",
            "source_dataset_dir": str(dataset_dir),
            "root": str(root_dir),
            "symbols": universe["symbol"].tolist(),
            "walk_forward": asdict(suite.walk_forward),
            "algorithms": list(suite.algorithms),
            "cv_fraction": suite.cv_fraction,
            "early_stopping_patience_epochs": suite.early_stopping_patience_epochs,
            "selection_metric": suite.selection_metric,
            "selection_split": suite.selection_split,
            "protocol_version": suite.protocol_version,
            "dqn_search_grid": list(_default_dqn_search_grid(suite.dqn)),
            "pg_search_grid": list(_default_pg_search_grid(suite.pg)),
            "a2c_search_grid": list(_default_a2c_search_grid(suite.a2c)),
            "selected_hyperparameters_path": str(selected_hyperparameters_path),
            "split_manifest_path": str(split_manifest_path),
            "elapsed_seconds": time.time() - start_time,
            "combined_rl_dir": str(combined_rl_dir),
            "leaderboard_path": str(leaderboard_path),
            "zhang_eval_dir": str(zhang_eval_artifacts.output_dir),
            "plots": {key: str(value) for key, value in plot_paths.items()},
        },
    )
    return ProductionRunArtifacts(
        root_dir=root_dir,
        source_dataset_dir=dataset_dir,
        combined_rl_dir=combined_rl_dir,
        individual_run_summaries_path=individual_run_summaries_path,
        leaderboard_path=leaderboard_path,
        run_metadata_path=run_metadata_path,
        zhang_eval_dir=zhang_eval_artifacts.output_dir,
        plot_paths=plot_paths,
    )


def train_dqn_production(config: ProductionSuiteConfig | None = None) -> ProductionRunArtifacts:
    suite = replace(config, algorithms=("dqn",), policies_for_plots=("dqn", *BASELINE_POLICIES)) if config else ProductionSuiteConfig(
        algorithms=("dqn",),
        policies_for_plots=("dqn", *BASELINE_POLICIES),
    )
    return run_production_suite(suite)


def train_pg_production(config: ProductionSuiteConfig | None = None) -> ProductionRunArtifacts:
    suite = replace(config, algorithms=("pg",), policies_for_plots=("pg", *BASELINE_POLICIES)) if config else ProductionSuiteConfig(
        algorithms=("pg",),
        policies_for_plots=("pg", *BASELINE_POLICIES),
    )
    return run_production_suite(suite)


def train_a2c_production(config: ProductionSuiteConfig | None = None) -> ProductionRunArtifacts:
    suite = replace(config, algorithms=("a2c",), policies_for_plots=("a2c", *BASELINE_POLICIES)) if config else ProductionSuiteConfig(
        algorithms=("a2c",),
        policies_for_plots=("a2c", *BASELINE_POLICIES),
    )
    return run_production_suite(suite)
