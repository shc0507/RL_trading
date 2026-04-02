"""Pairs trading research utilities with cointegration testing and CV tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...config import get_instrument_map
from ...data.sources import BarDataSource, PublicDailySource
from ...metrics import compute_performance_metrics


def _parse_csv_floats(raw_value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())


def _ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    # Fit y = intercept + beta * x so we can treat the residual as the spread.
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coefficients[0])
    beta = float(coefficients[1])
    residuals = y - (intercept + (beta * x))
    return intercept, beta, residuals


@dataclass(slots=True)
class PairsConfig:
    lookback: int = 20
    cost_rate_bp: float = 10.0
    entry_thresholds: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    exit_thresholds: tuple[float, ...] = (0.0, 0.5, 1.0)
    cv_folds: int = 3
    min_train_size: int = 120
    adf_t_stat_threshold: float = -2.86
    periods_per_year: int = 252


@dataclass(slots=True)
class CointegrationResult:
    symbol_x: str
    symbol_y: str
    intercept: float
    hedge_ratio: float
    adf_t_stat: float
    mean_reversion_coefficient: float
    half_life: float
    spread_std: float
    n_obs: int
    is_cointegrated: bool


@dataclass(slots=True)
class PairsBacktestReport:
    symbol_x: str
    symbol_y: str
    entry_threshold: float
    exit_threshold: float
    cointegration: CointegrationResult
    metrics: dict[str, float]
    daily_returns: pd.DataFrame
    trade_log: pd.DataFrame


@dataclass(slots=True)
class ThresholdSelectionResult:
    objective_name: str
    best_entry_threshold: float | None
    best_exit_threshold: float | None
    best_objective_value: float | None
    cv_results: pd.DataFrame


@dataclass(slots=True)
class PairsResearchResult:
    symbol_x: str
    symbol_y: str
    train_cointegration: CointegrationResult
    threshold_selection: ThresholdSelectionResult
    test_report: PairsBacktestReport | None
    is_tradeable: bool
    failure_reason: str | None = None


class PairsTradingResearcher:
    """Research workflow for a single pair.

    The workflow is intentionally simple and leakage-aware:
    - fit the hedge ratio only on the training window
    - cross-validate the entry/exit thresholds on expanding folds of the training window
    - rerun the backtest on a held-out test window with the selected thresholds
    """

    def __init__(self, bar_frame: pd.DataFrame, config: PairsConfig | None = None) -> None:
        self.bar_frame = bar_frame.copy()
        self.bar_frame["date"] = pd.to_datetime(self.bar_frame["date"])
        self.config = config or PairsConfig()

    @classmethod
    def from_source(
        cls,
        source: BarDataSource,
        symbol_x: str,
        symbol_y: str,
        start_date: str,
        end_date: str,
        config: PairsConfig | None = None,
    ) -> "PairsTradingResearcher":
        bars = source.fetch(symbols=[symbol_x, symbol_y], start_date=start_date, end_date=end_date)
        return cls(bar_frame=bars, config=config)

    def run(
        self,
        symbol_x: str,
        symbol_y: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> PairsResearchResult:
        pair_frame = self._build_pair_frame(symbol_x=symbol_x, symbol_y=symbol_y)
        train_frame = self._slice_frame(pair_frame, train_start, train_end)
        test_frame = self._slice_frame(pair_frame, test_start, test_end)

        # First answer the structural question: did this pair even behave like a
        # mean-reverting spread in the training window?
        train_cointegration = self.test_cointegration(train_frame, symbol_x=symbol_x, symbol_y=symbol_y)
        cv_selection = self.cross_validate_thresholds(train_frame, symbol_x=symbol_x, symbol_y=symbol_y)

        if not train_cointegration.is_cointegrated:
            return PairsResearchResult(
                symbol_x=symbol_x,
                symbol_y=symbol_y,
                train_cointegration=train_cointegration,
                threshold_selection=cv_selection,
                test_report=None,
                is_tradeable=False,
                failure_reason="training window is not cointegrated",
            )

        if cv_selection.best_entry_threshold is None or cv_selection.best_exit_threshold is None:
            return PairsResearchResult(
                symbol_x=symbol_x,
                symbol_y=symbol_y,
                train_cointegration=train_cointegration,
                threshold_selection=cv_selection,
                test_report=None,
                is_tradeable=False,
                failure_reason="no threshold combination survived cross-validation",
            )

        test_report = self.backtest(
            train_frame=train_frame,
            evaluation_frame=test_frame,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            entry_threshold=cv_selection.best_entry_threshold,
            exit_threshold=cv_selection.best_exit_threshold,
        )
        return PairsResearchResult(
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            train_cointegration=train_cointegration,
            threshold_selection=cv_selection,
            test_report=test_report,
            is_tradeable=True,
        )

    def test_cointegration(
        self,
        frame: pd.DataFrame,
        symbol_x: str,
        symbol_y: str,
    ) -> CointegrationResult:
        prepared = frame.copy()
        if len(prepared) < self.config.min_train_size:
            raise ValueError(f"need at least {self.config.min_train_size} observations to test cointegration")

        log_x = np.log(prepared["price_x"].to_numpy(dtype=float))
        log_y = np.log(prepared["price_y"].to_numpy(dtype=float))

        # Engle-Granger step 1: estimate the long-run relationship and extract
        # the spread as the regression residual.
        intercept, hedge_ratio, spread = _ols_fit(y=log_x, x=log_y)
        lagged_spread = spread[:-1]
        delta_spread = np.diff(spread)
        design = np.column_stack([np.ones(len(lagged_spread)), lagged_spread])
        coefficients, _, _, _ = np.linalg.lstsq(design, delta_spread, rcond=None)
        residuals = delta_spread - (design @ coefficients)
        dof = max(len(delta_spread) - design.shape[1], 1)
        sigma2 = float((residuals @ residuals) / dof)
        covariance = sigma2 * np.linalg.inv(design.T @ design)
        mean_reversion_coefficient = float(coefficients[1])
        standard_error = float(np.sqrt(max(covariance[1, 1], 1e-12)))

        # Engle-Granger step 2: check whether the spread itself looks stationary.
        # We use the t-stat on the lagged spread coefficient as a lightweight
        # ADF-style filter rather than pulling in a heavier stats package.
        adf_t_stat = mean_reversion_coefficient / standard_error
        ar_root = 1.0 + mean_reversion_coefficient
        if 0.0 < ar_root < 1.0:
            half_life = float(-np.log(2.0) / np.log(ar_root))
        else:
            half_life = float("inf")
        spread_std = float(np.std(spread, ddof=0))
        is_cointegrated = bool(
            np.isfinite(adf_t_stat)
            and adf_t_stat <= self.config.adf_t_stat_threshold
            and np.isfinite(half_life)
            and spread_std > 1e-8
        )
        return CointegrationResult(
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            adf_t_stat=float(adf_t_stat),
            mean_reversion_coefficient=mean_reversion_coefficient,
            half_life=half_life,
            spread_std=spread_std,
            n_obs=len(prepared),
            is_cointegrated=is_cointegrated,
        )

    def cross_validate_thresholds(
        self,
        train_frame: pd.DataFrame,
        symbol_x: str,
        symbol_y: str,
        objective_name: str = "sharpe",
    ) -> ThresholdSelectionResult:
        folds = self._build_expanding_folds(train_frame)
        rows: list[dict[str, float | int | bool]] = []

        for entry_threshold in self.config.entry_thresholds:
            for exit_threshold in self.config.exit_thresholds:
                if entry_threshold <= exit_threshold:
                    continue
                for fold_index, (fit_end, validation_start, validation_end) in enumerate(folds, start=1):
                    fit_frame = train_frame.iloc[:fit_end].copy()
                    validation_frame = train_frame.iloc[validation_start:validation_end].copy()

                    # Refit the spread on each fold using only the data available
                    # at that point so threshold selection stays leakage-aware.
                    cointegration = self.test_cointegration(fit_frame, symbol_x=symbol_x, symbol_y=symbol_y)
                    if not cointegration.is_cointegrated:
                        rows.append(
                            {
                                "entry_threshold": entry_threshold,
                                "exit_threshold": exit_threshold,
                                "fold": fold_index,
                                "objective": float("-inf"),
                                "annualized_return": float("-inf"),
                                "cointegrated": False,
                            }
                        )
                        continue

                    report = self.backtest(
                        train_frame=fit_frame,
                        evaluation_frame=validation_frame,
                        symbol_x=symbol_x,
                        symbol_y=symbol_y,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                    )
                    rows.append(
                        {
                            "entry_threshold": entry_threshold,
                            "exit_threshold": exit_threshold,
                            "fold": fold_index,
                            "objective": float(report.metrics.get(objective_name, 0.0)),
                            "annualized_return": float(report.metrics.get("annualized_return", 0.0)),
                            "cointegrated": True,
                        }
                    )

        cv_results = pd.DataFrame(rows)
        if cv_results.empty:
            return ThresholdSelectionResult(
                objective_name=objective_name,
                best_entry_threshold=None,
                best_exit_threshold=None,
                best_objective_value=None,
                cv_results=cv_results,
            )

        grouped = (
            cv_results.groupby(["entry_threshold", "exit_threshold"], as_index=False)
            .agg(
                objective=("objective", "mean"),
                annualized_return=("annualized_return", "mean"),
                valid_folds=("cointegrated", "sum"),
            )
            .sort_values(["objective", "annualized_return", "valid_folds"], ascending=False)
            .reset_index(drop=True)
        )
        best = grouped.iloc[0]
        if not np.isfinite(float(best["objective"])) or int(best["valid_folds"]) == 0:
            return ThresholdSelectionResult(
                objective_name=objective_name,
                best_entry_threshold=None,
                best_exit_threshold=None,
                best_objective_value=None,
                cv_results=cv_results,
            )
        return ThresholdSelectionResult(
            objective_name=objective_name,
            best_entry_threshold=float(best["entry_threshold"]),
            best_exit_threshold=float(best["exit_threshold"]),
            best_objective_value=float(best["objective"]),
            cv_results=cv_results,
        )

    def backtest(
        self,
        train_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        symbol_x: str,
        symbol_y: str,
        entry_threshold: float,
        exit_threshold: float,
    ) -> PairsBacktestReport:
        if evaluation_frame.empty:
            raise ValueError("evaluation frame is empty")

        # Always estimate the hedge ratio on the fit window only, then freeze it
        # while we trade the evaluation window.
        cointegration = self.test_cointegration(train_frame, symbol_x=symbol_x, symbol_y=symbol_y)
        if not cointegration.is_cointegrated:
            empty_returns = pd.DataFrame(columns=["date", "pair_return", "spread", "z_score", "position"])
            empty_trades = pd.DataFrame(columns=["date", "position", "reason"])
            metrics = compute_performance_metrics(
                pd.Series(dtype=float),
                periods_per_year=self.config.periods_per_year,
            )
            return PairsBacktestReport(
                symbol_x=symbol_x,
                symbol_y=symbol_y,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                cointegration=cointegration,
                metrics=metrics,
                daily_returns=empty_returns,
                trade_log=empty_trades,
            )

        context_frame = pd.concat(
            [train_frame.tail(self.config.lookback).copy(), evaluation_frame.copy()],
            ignore_index=True,
        )

        # Keep a short train tail so the first evaluation rows can form a rolling
        # z-score without peeking into future test data.
        spread_frame = self._compute_spread_frame(context_frame, cointegration)

        unit_weight_x = 1.0 / (1.0 + abs(cointegration.hedge_ratio))
        unit_weight_y = -cointegration.hedge_ratio / (1.0 + abs(cointegration.hedge_ratio))
        cost_rate = self.config.cost_rate_bp / 10_000.0
        current_position = 0.0
        trade_rows: list[dict[str, object]] = []
        return_rows: list[dict[str, object]] = []
        evaluation_start = pd.Timestamp(evaluation_frame["date"].iloc[0])

        for index in range(len(spread_frame) - 1):
            row = spread_frame.iloc[index]
            next_row = spread_frame.iloc[index + 1]
            if not np.isfinite(float(row["z_score"])):
                continue

            target_position = current_position
            reason = "hold"
            z_score = float(row["z_score"])

            # Long spread means buy X / sell Y when the spread is unusually low.
            if current_position == 0.0 and z_score <= -entry_threshold:
                target_position = 1.0
                reason = "enter_long_spread"
            elif current_position == 0.0 and z_score >= entry_threshold:
                target_position = -1.0
                reason = "enter_short_spread"
            elif current_position != 0.0 and abs(z_score) <= exit_threshold:
                target_position = 0.0
                reason = "exit_mean_reversion"

            turnover = abs(target_position - current_position)
            spread_return = (
                (target_position * unit_weight_x * float(row["return_x"]))
                + (target_position * unit_weight_y * float(row["return_y"]))
            )
            net_return = spread_return - (cost_rate * turnover)
            current_position = target_position

            if pd.Timestamp(row["date"]) < evaluation_start:
                continue

            return_rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "next_date": pd.Timestamp(next_row["date"]),
                    "pair_return": net_return,
                    "gross_return": spread_return,
                    "turnover": turnover,
                    "position": current_position,
                    "spread": float(row["spread"]),
                    "z_score": z_score,
                }
            )
            if turnover > 0.0:
                trade_rows.append(
                    {
                        "date": pd.Timestamp(row["date"]),
                        "next_date": pd.Timestamp(next_row["date"]),
                        "position": current_position,
                        "reason": reason,
                        "z_score": z_score,
                        "turnover": turnover,
                    }
                )

        daily_returns = pd.DataFrame(return_rows)
        trade_log = pd.DataFrame(trade_rows)
        if daily_returns.empty:
            metric_series = pd.Series(dtype=float)
            turnover_series = pd.Series(dtype=float)
            total_cost = 0.0
        else:
            metric_series = daily_returns["pair_return"]
            turnover_series = daily_returns["turnover"]
            total_cost = float(cost_rate * daily_returns["turnover"].sum())
        metrics = compute_performance_metrics(
            metric_series,
            turnover=turnover_series,
            total_cost=total_cost,
            periods_per_year=self.config.periods_per_year,
        )
        return PairsBacktestReport(
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            cointegration=cointegration,
            metrics=metrics,
            daily_returns=daily_returns,
            trade_log=trade_log,
        )

    def _build_pair_frame(self, symbol_x: str, symbol_y: str) -> pd.DataFrame:
        frame = self.bar_frame.loc[self.bar_frame["symbol"].isin([symbol_x, symbol_y])].copy()
        if frame["symbol"].nunique() != 2:
            raise ValueError(f"expected exactly two symbols in bar_frame, found {frame['symbol'].nunique()}")

        # Pivot into one row per date so both legs are aligned before we model
        # spreads or simulate trades.
        pivot = (
            frame.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
            .sort_index()
            .dropna()
        )
        if symbol_x not in pivot.columns or symbol_y not in pivot.columns:
            raise ValueError(f"missing price history for pair {symbol_x}/{symbol_y}")
        pair_frame = pd.DataFrame(
            {
                "date": pivot.index,
                "price_x": pivot[symbol_x].astype(float).values,
                "price_y": pivot[symbol_y].astype(float).values,
            }
        )
        pair_frame["return_x"] = pair_frame["price_x"].pct_change().shift(-1)
        pair_frame["return_y"] = pair_frame["price_y"].pct_change().shift(-1)
        return pair_frame.dropna().reset_index(drop=True)

    def _slice_frame(self, frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))
        sliced = frame.loc[mask].reset_index(drop=True)
        if sliced.empty:
            raise ValueError(f"no rows found between {start_date} and {end_date}")
        return sliced

    def _compute_spread_frame(self, frame: pd.DataFrame, cointegration: CointegrationResult) -> pd.DataFrame:
        prepared = frame.copy()
        log_x = np.log(prepared["price_x"].astype(float))
        log_y = np.log(prepared["price_y"].astype(float))
        prepared["spread"] = log_x - (
            cointegration.intercept + (cointegration.hedge_ratio * log_y)
        )

        # Rolling normalization lets the same fixed thresholds work across
        # quieter and noisier market regimes.
        rolling_mean = prepared["spread"].rolling(self.config.lookback, min_periods=self.config.lookback).mean()
        rolling_std = prepared["spread"].rolling(self.config.lookback, min_periods=self.config.lookback).std(ddof=0)
        prepared["z_score"] = (prepared["spread"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
        return prepared

    def _build_expanding_folds(self, train_frame: pd.DataFrame) -> list[tuple[int, int, int]]:
        n_obs = len(train_frame)
        if n_obs < (self.config.min_train_size + self.config.cv_folds):
            raise ValueError("training sample is too small for the requested cross-validation settings")

        validation_total = n_obs - self.config.min_train_size
        fold_size = max(validation_total // self.config.cv_folds, 1)
        folds: list[tuple[int, int, int]] = []
        train_end = self.config.min_train_size
        for fold_index in range(self.config.cv_folds):
            validation_start = train_end
            if fold_index == self.config.cv_folds - 1:
                validation_end = n_obs
            else:
                validation_end = min(validation_start + fold_size, n_obs)
            # Expanding folds mimic how you'd actually retune in production:
            # train on the past, validate on the next unseen block.
            folds.append((train_end, validation_start, validation_end))
            train_end = validation_end
        return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairs trading research with CV threshold tuning.")
    parser.add_argument("--symbol-x", required=True)
    parser.add_argument("--symbol-y", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--train-end-date", required=True)
    parser.add_argument("--test-end-date", required=True)
    parser.add_argument("--market", choices=("equity", "crypto"), default="equity")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--entry-thresholds", default="1.0,1.5,2.0,2.5")
    parser.add_argument("--exit-thresholds", default="0.0,0.5,1.0")
    args = parser.parse_args()

    config = PairsConfig(
        lookback=args.lookback,
        cv_folds=args.cv_folds,
        entry_thresholds=_parse_csv_floats(args.entry_thresholds),
        exit_thresholds=_parse_csv_floats(args.exit_thresholds),
        periods_per_year=365 if args.market == "crypto" else 252,
    )
    source = PublicDailySource(instrument_map=get_instrument_map(args.market))
    researcher = PairsTradingResearcher.from_source(
        source=source,
        symbol_x=args.symbol_x,
        symbol_y=args.symbol_y,
        start_date=args.start_date,
        end_date=args.test_end_date,
        config=config,
    )
    result = researcher.run(
        symbol_x=args.symbol_x,
        symbol_y=args.symbol_y,
        train_start=args.start_date,
        train_end=args.train_end_date,
        test_start=(pd.Timestamp(args.train_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        test_end=args.test_end_date,
    )

    print(f"pair={result.symbol_x}/{result.symbol_y}")
    print(
        "cointegration:",
        {
            "is_cointegrated": result.train_cointegration.is_cointegrated,
            "adf_t_stat": round(result.train_cointegration.adf_t_stat, 4),
            "hedge_ratio": round(result.train_cointegration.hedge_ratio, 4),
            "half_life": round(result.train_cointegration.half_life, 4)
            if np.isfinite(result.train_cointegration.half_life)
            else "inf",
        },
    )
    if not result.is_tradeable:
        print(f"tradeable=False reason={result.failure_reason}")
        return

    print(
        "best_thresholds:",
        {
            "entry": result.threshold_selection.best_entry_threshold,
            "exit": result.threshold_selection.best_exit_threshold,
            "objective": result.threshold_selection.best_objective_value,
        },
    )
    if result.test_report is not None:
        print("test_metrics:", result.test_report.metrics)


if __name__ == "__main__":
    main()
