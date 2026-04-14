"""Zhang-aligned evaluation utilities over saved trade reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ..asset_groups import canonical_asset_group
from ..config import ANNUALIZATION_FACTOR, DEFAULT_COST_RATE_BP, DEFAULT_VOL_TARGET, VOLATILITY_SPAN
from ..metrics import compute_performance_metrics


@dataclass(slots=True)
class ZhangEvalConfig:
    target_vol: float = DEFAULT_VOL_TARGET
    portfolio_vol_scaling: bool = True
    cost_grid_bp: tuple[float, ...] = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
    split: str | None = None
    group_by_asset_class: bool = True
    return_column: str = "trade_return"
    vol_span: int = VOLATILITY_SPAN


@dataclass(slots=True)
class ZhangEvalArtifacts:
    output_dir: Path
    config_path: Path
    contract_daily_returns_path: Path
    contract_metrics_path: Path
    daily_portfolios_unscaled_path: Path
    daily_portfolios_scaled_path: Path
    summary_unscaled_path: Path
    summary_scaled_path: Path
    cost_sweep_path: Path
    instrument_groups_path: Path


def _annual_target_to_daily(target_vol: float) -> float:
    return float(target_vol) / np.sqrt(ANNUALIZATION_FACTOR)


def _safe_vol_scale(series: pd.Series, target_vol: float, span: int) -> pd.DataFrame:
    returns = series.fillna(0.0).astype(float)
    lagged_vol = (
        returns.ewm(span=span, adjust=False, min_periods=span).std(bias=False).shift(1)
    )
    daily_target = _annual_target_to_daily(target_vol)
    scale_factor = daily_target / lagged_vol.replace(0.0, np.nan)
    scale_factor = scale_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scaled_returns = returns * scale_factor
    return pd.DataFrame(
        {
            "portfolio_return": scaled_returns,
            "portfolio_scale_factor": scale_factor,
            "portfolio_lagged_vol": lagged_vol,
        }
    )


def _extract_policy_and_split(path: Path) -> tuple[str, str]:
    stem = path.stem
    if not stem.endswith("_trades"):
        raise ValueError(f"unexpected report filename {path.name}")
    prefix = stem[: -len("_trades")]
    policy, split = prefix.rsplit("_", 1)
    return policy, split


def _load_trade_reports(report_dir: Path, split: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(report_dir.glob("*_trades.csv")):
        policy, report_split = _extract_policy_and_split(path)
        if split is not None and report_split != split:
            continue
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["next_date"] = pd.to_datetime(frame["next_date"])
        frame["policy"] = policy
        frame["split"] = report_split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no trade reports found under {report_dir}")
    return pd.concat(frames, ignore_index=True)


def _load_instrument_groups(artifact_dir: Path, trade_logs: pd.DataFrame) -> pd.DataFrame:
    run_dir = artifact_dir.parent
    instruments_path = run_dir / "processed" / "instruments.csv"
    if instruments_path.exists():
        instruments = pd.read_csv(instruments_path)
        frame = instruments.loc[:, ["symbol", "asset_class"]].drop_duplicates().copy()
    else:
        frame = pd.DataFrame({"symbol": sorted(trade_logs["symbol"].astype(str).unique()), "asset_class": "unknown"})
    frame["asset_group"] = frame["asset_class"].map(canonical_asset_group)
    return frame.sort_values("symbol").reset_index(drop=True)


def _effective_columns(return_column: str) -> tuple[str, str, str]:
    if return_column in {"trade_return", "zhang_reward"}:
        return "scaled_turnover", "trade_cost", "pre_cost_trade_return"
    if return_column == "zhang_return":
        return "scaled_turnover", "zhang_cost", "zhang_cost_return"
    if return_column == "raw_return":
        return "turnover", "raw_cost", "raw_cost_return"
    if return_column == "raw_pnl":
        return "turnover", "raw_cost", "raw_pre_cost_pnl"
    raise ValueError(f"unsupported return_column {return_column}")


def _uses_additive_trade_returns(return_column: str) -> bool:
    return return_column in {"trade_return", "zhang_reward", "raw_pnl"}


class ZhangEvaluator:
    def __init__(
        self,
        trade_logs: pd.DataFrame,
        instruments: pd.DataFrame,
        config: ZhangEvalConfig | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.trade_logs = trade_logs.copy()
        self.trade_logs["date"] = pd.to_datetime(self.trade_logs["date"])
        self.trade_logs["next_date"] = pd.to_datetime(self.trade_logs["next_date"])
        self.instruments = instruments.copy()
        self.config = config or ZhangEvalConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

        turnover_column, cost_column, pre_cost_column = _effective_columns(self.config.return_column)
        self.turnover_column = turnover_column
        self.cost_column = cost_column
        self.pre_cost_column = pre_cost_column

        instrument_frame = self.instruments.loc[:, ["symbol", "asset_class", "asset_group"]].drop_duplicates()
        self.trade_logs = self.trade_logs.merge(instrument_frame, how="left", on="symbol")
        self.trade_logs["asset_class"] = self.trade_logs["asset_class"].fillna("unknown")
        self.trade_logs["asset_group"] = self.trade_logs["asset_group"].fillna(
            self.trade_logs["asset_class"].map(canonical_asset_group)
        )
        self.trade_logs["contract_return"] = self.trade_logs[self.config.return_column].fillna(0.0).astype(float)
        self.trade_logs["effective_turnover"] = self.trade_logs[self.turnover_column].fillna(0.0).astype(float)
        self.trade_logs["cost_amount"] = self.trade_logs[self.cost_column].fillna(0.0).astype(float)
        if self.config.return_column in {"zhang_return", "raw_return"}:
            self.trade_logs["cost_return"] = self.trade_logs[self.pre_cost_column].fillna(0.0).astype(float)
            self.trade_logs["pre_cost_return"] = self.trade_logs["contract_return"] + self.trade_logs["cost_return"]
        else:
            self.trade_logs["pre_cost_return"] = self.trade_logs[self.pre_cost_column].fillna(0.0).astype(float)
            self.trade_logs["cost_return"] = self.trade_logs["pre_cost_return"] - self.trade_logs["contract_return"]

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        config: ZhangEvalConfig | None = None,
    ) -> "ZhangEvaluator":
        artifact_path = Path(artifact_dir)
        trade_logs = _load_trade_reports(artifact_path / "reports", split=(config.split if config else None))
        instruments = _load_instrument_groups(artifact_path, trade_logs)
        return cls(trade_logs=trade_logs, instruments=instruments, config=config, artifact_dir=artifact_path)

    def _group_frame(self) -> pd.DataFrame:
        base = self.trade_logs.copy()
        if not self.config.group_by_asset_class:
            base["asset_group"] = "all"
            return base

        all_group = base.copy()
        all_group["asset_group"] = "all"
        return pd.concat([base, all_group], ignore_index=True)

    def build_contract_daily_returns(self) -> pd.DataFrame:
        columns = [
            "policy",
            "split",
            "asset_group",
            "asset_class",
            "symbol",
            "date",
            "contract_return",
            "effective_turnover",
            "cost_return",
            "cost_amount",
            "pre_cost_return",
        ]
        optional_columns = [
            column
            for column in ("scaled_position", "scaled_turnover", "pre_cost_trade_return", "trade_return", "trade_cost")
            if column in self.trade_logs.columns
        ]
        frame = self.trade_logs.loc[:, columns + optional_columns].copy()
        if "scaled_position" not in frame.columns:
            frame["scaled_position"] = np.nan
        if "scaled_turnover" not in frame.columns:
            frame["scaled_turnover"] = frame["effective_turnover"]
        if "pre_cost_trade_return" not in frame.columns:
            frame["pre_cost_trade_return"] = frame["pre_cost_return"]
        frame["trade_return"] = frame["contract_return"]
        frame["trade_cost"] = frame["cost_amount"]
        return frame.sort_values(["policy", "split", "asset_group", "symbol", "date"]).reset_index(drop=True)

    def build_contract_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (policy, split, asset_group, symbol), frame in self.trade_logs.groupby(
            ["policy", "split", "asset_group", "symbol"], sort=False
        ):
            metrics = compute_performance_metrics(
                frame["contract_return"],
                turnover=frame["effective_turnover"],
                total_cost=float(frame["cost_amount"].sum()),
            )
            avg_loss = abs(float(metrics["avg_loss"])) if float(metrics["avg_loss"]) != 0.0 else np.nan
            rows.append(
                {
                    "policy": policy,
                    "split": split,
                    "asset_group": asset_group,
                    "symbol": symbol,
                    **metrics,
                    "pct_positive_returns": float(metrics["hit_rate"]),
                    "avg_positive_negative_return_ratio": float(metrics["avg_win"] / avg_loss) if pd.notna(avg_loss) else 0.0,
                    "avg_return_per_turnover": (
                        float(frame["contract_return"].sum() / frame["effective_turnover"].sum())
                        if float(frame["effective_turnover"].sum()) > 0.0
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values(["policy", "split", "asset_group", "symbol"]).reset_index(drop=True)

    def _build_daily_portfolios(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        base = daily_returns.copy()
        for (policy, split, asset_group), frame in base.groupby(["policy", "split", "asset_group"], sort=False):
            contract_count = int(frame["symbol"].nunique())
            daily = (
                frame.groupby("date", as_index=False)
                .agg(
                    portfolio_return=("contract_return", "mean"),
                    total_cost=("cost_amount", "sum"),
                    avg_turnover=("effective_turnover", "mean"),
                )
                .sort_values("date")
                .reset_index(drop=True)
            )
            daily["policy"] = policy
            daily["split"] = split
            daily["asset_group"] = asset_group
            daily["contract_count"] = contract_count
            daily["portfolio_cumulative_trade_return"] = daily["portfolio_return"].fillna(0.0).cumsum()
            daily["cumulative_trade_return"] = daily["portfolio_cumulative_trade_return"]
            rows.append(daily)
        return pd.concat(rows, ignore_index=True).sort_values(["policy", "split", "asset_group", "date"]).reset_index(drop=True)

    def _apply_portfolio_scaling(self, daily_portfolios: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for (policy, split, asset_group), frame in daily_portfolios.groupby(["policy", "split", "asset_group"], sort=False):
            scaled_frame = frame.copy()
            scaled = _safe_vol_scale(
                series=scaled_frame["portfolio_return"],
                target_vol=self.config.target_vol,
                span=self.config.vol_span,
            )
            scaled_frame["portfolio_return"] = scaled["portfolio_return"]
            scaled_frame["portfolio_scale_factor"] = scaled["portfolio_scale_factor"]
            scaled_frame["portfolio_lagged_vol"] = scaled["portfolio_lagged_vol"]
            scaled_frame["portfolio_cumulative_trade_return"] = scaled_frame["portfolio_return"].fillna(0.0).cumsum()
            scaled_frame["cumulative_trade_return"] = scaled_frame["portfolio_cumulative_trade_return"]
            rows.append(scaled_frame)
        return pd.concat(rows, ignore_index=True).sort_values(["policy", "split", "asset_group", "date"]).reset_index(drop=True)

    def _build_summary(self, daily_portfolios: pd.DataFrame, scaled: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (policy, split, asset_group), frame in daily_portfolios.groupby(["policy", "split", "asset_group"], sort=False):
            metrics = compute_performance_metrics(
                frame["portfolio_return"],
                turnover=frame["avg_turnover"],
                total_cost=float(frame["total_cost"].sum()),
            )
            avg_loss = abs(float(metrics["avg_loss"])) if float(metrics["avg_loss"]) != 0.0 else np.nan
            rows.append(
                {
                    "policy": policy,
                    "split": split,
                    "asset_group": asset_group,
                    "portfolio_vol_scaled": bool(scaled),
                    **metrics,
                    "pct_positive_returns": float(metrics["hit_rate"]),
                    "avg_positive_negative_return_ratio": float(metrics["avg_win"] / avg_loss) if pd.notna(avg_loss) else 0.0,
                    "cumulative_trade_return_final": float(frame["portfolio_cumulative_trade_return"].iloc[-1]) if not frame.empty else 0.0,
                    "contract_count": int(frame["contract_count"].iloc[0]) if not frame.empty else 0,
                }
            )
        return pd.DataFrame(rows).sort_values(["split", "asset_group", "policy"]).reset_index(drop=True)

    def _cost_sweep_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        grouped = self._group_frame()
        additive_returns = _uses_additive_trade_returns(self.config.return_column)
        for cost_bp in self.config.cost_grid_bp:
            for (policy, split, asset_group), frame in grouped.groupby(["policy", "split", "asset_group"], sort=False):
                adjusted = frame.copy()
                cost_rate = float(cost_bp) / 10_000.0
                adjusted["adjusted_cost_amount"] = cost_rate * adjusted["price_now"] * adjusted["effective_turnover"]
                if additive_returns:
                    adjusted["adjusted_contract_return"] = adjusted["pre_cost_return"] - adjusted["adjusted_cost_amount"]
                else:
                    adjusted["adjusted_cost_return"] = cost_rate * adjusted["effective_turnover"]
                    adjusted["adjusted_contract_return"] = adjusted["pre_cost_return"] - adjusted["adjusted_cost_return"]

                daily = (
                    adjusted.groupby("date", as_index=False)
                    .agg(
                        portfolio_return=("adjusted_contract_return", "mean"),
                        total_cost=("adjusted_cost_amount", "sum"),
                        avg_turnover=("effective_turnover", "mean"),
                    )
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                if self.config.portfolio_vol_scaling:
                    scaled = _safe_vol_scale(
                        series=daily["portfolio_return"],
                        target_vol=self.config.target_vol,
                        span=self.config.vol_span,
                    )
                    daily["portfolio_return"] = scaled["portfolio_return"]
                metrics = compute_performance_metrics(
                    daily["portfolio_return"],
                    turnover=daily["avg_turnover"],
                    total_cost=float(daily["total_cost"].sum()),
                )
                contract_count = int(adjusted["symbol"].nunique())
                rows.append(
                    {
                        "policy": policy,
                        "split": split,
                        "asset_group": asset_group,
                        "cost_rate_bp": float(cost_bp),
                        "portfolio_vol_scaled": bool(self.config.portfolio_vol_scaling),
                        "annualized_return": float(metrics["annualized_return"]),
                        "annualized_volatility": float(metrics["annualized_volatility"]),
                        "sharpe": float(metrics["sharpe"]),
                        "sortino": float(metrics["sortino"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "avg_daily_turnover": float(metrics.get("avg_daily_turnover", 0.0)),
                        "total_transaction_cost": float(daily["total_cost"].sum()),
                        "avg_cost_per_contract": float(daily["total_cost"].sum() / contract_count) if contract_count else 0.0,
                        "contract_count": contract_count,
                    }
                )
        return rows

    def evaluate(self) -> dict[str, pd.DataFrame]:
        contract_daily_returns = self.build_contract_daily_returns()
        contract_metrics = self.build_contract_metrics()
        grouped = self._group_frame()
        daily_unscaled = self._build_daily_portfolios(
            grouped.loc[:, ["policy", "split", "asset_group", "symbol", "date", "contract_return", "effective_turnover", "cost_amount"]]
        )
        daily_scaled = self._apply_portfolio_scaling(daily_unscaled) if self.config.portfolio_vol_scaling else daily_unscaled.copy()
        summary_unscaled = self._build_summary(daily_unscaled, scaled=False)
        summary_scaled = self._build_summary(daily_scaled, scaled=bool(self.config.portfolio_vol_scaling))
        cost_sweep = pd.DataFrame(self._cost_sweep_rows()).sort_values(
            ["split", "asset_group", "cost_rate_bp", "policy"]
        ).reset_index(drop=True)
        return {
            "contract_daily_returns": contract_daily_returns,
            "contract_metrics": contract_metrics,
            "daily_portfolios_unscaled": daily_unscaled,
            "daily_portfolios_scaled": daily_scaled,
            "summary_unscaled": summary_unscaled,
            "summary_scaled": summary_scaled,
            "cost_sweep": cost_sweep,
            "instrument_groups": self.instruments.sort_values("symbol").reset_index(drop=True),
        }

    def _write_frame_csv_atomic(self, path: Path, frame: pd.DataFrame) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def write_artifacts(self, output_dir: str | Path | None = None) -> ZhangEvalArtifacts:
        destination = Path(output_dir) if output_dir is not None else (self.artifact_dir / "zhang_eval" if self.artifact_dir else None)
        if destination is None:
            raise ValueError("output_dir is required when the evaluator has no artifact_dir")
        destination.mkdir(parents=True, exist_ok=True)

        results = self.evaluate()
        config_path = destination / "config.json"
        contract_daily_returns_path = destination / "contract_daily_returns.csv"
        contract_metrics_path = destination / "contract_metrics.csv"
        daily_portfolios_unscaled_path = destination / "daily_portfolios_unscaled.csv"
        daily_portfolios_scaled_path = destination / "daily_portfolios_scaled.csv"
        summary_unscaled_path = destination / "summary_unscaled.csv"
        summary_scaled_path = destination / "summary_scaled.csv"
        cost_sweep_path = destination / "cost_sweep.csv"
        instrument_groups_path = destination / "instrument_groups.csv"

        self._write_json_atomic(
            config_path,
            {
                "target_vol": self.config.target_vol,
                "portfolio_vol_scaling": self.config.portfolio_vol_scaling,
                "cost_grid_bp": list(self.config.cost_grid_bp),
                "split": self.config.split,
                "group_by_asset_class": self.config.group_by_asset_class,
                "return_column": self.config.return_column,
                "vol_span": self.config.vol_span,
            },
        )
        self._write_frame_csv_atomic(contract_daily_returns_path, results["contract_daily_returns"])
        self._write_frame_csv_atomic(contract_metrics_path, results["contract_metrics"])
        self._write_frame_csv_atomic(daily_portfolios_unscaled_path, results["daily_portfolios_unscaled"])
        self._write_frame_csv_atomic(daily_portfolios_scaled_path, results["daily_portfolios_scaled"])
        self._write_frame_csv_atomic(summary_unscaled_path, results["summary_unscaled"])
        self._write_frame_csv_atomic(summary_scaled_path, results["summary_scaled"])
        self._write_frame_csv_atomic(cost_sweep_path, results["cost_sweep"])
        self._write_frame_csv_atomic(instrument_groups_path, results["instrument_groups"])
        return ZhangEvalArtifacts(
            output_dir=destination,
            config_path=config_path,
            contract_daily_returns_path=contract_daily_returns_path,
            contract_metrics_path=contract_metrics_path,
            daily_portfolios_unscaled_path=daily_portfolios_unscaled_path,
            daily_portfolios_scaled_path=daily_portfolios_scaled_path,
            summary_unscaled_path=summary_unscaled_path,
            summary_scaled_path=summary_scaled_path,
            cost_sweep_path=cost_sweep_path,
            instrument_groups_path=instrument_groups_path,
        )


def run_zhang_evaluation(
    artifact_dir: str | Path,
    config: ZhangEvalConfig | None = None,
) -> ZhangEvalArtifacts:
    evaluator = ZhangEvaluator.from_artifact_dir(artifact_dir=artifact_dir, config=config)
    return evaluator.write_artifacts()
