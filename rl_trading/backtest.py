"""Backtesting orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .env import EnvironmentConfig, TradingEnv
from .metrics import compute_performance_metrics


@dataclass(slots=True)
class EvalReport:
    policy_name: str
    split: str
    portfolio_metrics: dict[str, float]
    symbol_metrics: pd.DataFrame
    daily_returns: pd.DataFrame
    trade_log: pd.DataFrame


class Backtester:
    """Runs a policy through single-symbol environments and aggregates the portfolio."""

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        env_config: EnvironmentConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.env = TradingEnv(feature_frame=self.feature_frame, config=env_config, splits=splits)
        self.env_config = self.env.config
        self.splits = self.env.splits

    def run(self, policy, split: str, symbols: list[str] | None = None) -> EvalReport:
        symbol_list = symbols or sorted(self.feature_frame["symbol"].unique().tolist())
        logs: list[dict[str, object]] = []
        symbol_metric_rows = []

        return_column = "raw_return" if self.env_config.reward_mode == "raw" else "zhang_return"
        cost_column = "raw_cost" if self.env_config.reward_mode == "raw" else "zhang_cost"

        for symbol in symbol_list:
            observation = self.env.reset(symbol=symbol, split=split)
            done = False
            while not done:
                action = policy.act(observation)
                observation, _, done, info = self.env.step(action)
                logs.append(info)

            symbol_log = pd.DataFrame([row for row in logs if row["symbol"] == symbol and row["split"] == split])
            metrics = compute_performance_metrics(
                symbol_log[return_column],
                turnover=symbol_log["turnover"],
                total_cost=float(symbol_log[cost_column].sum()),
            )
            metrics["symbol"] = symbol
            symbol_metric_rows.append(metrics)

        trade_log = pd.DataFrame(logs)
        split_trade_log = trade_log.loc[trade_log["split"] == split].copy()
        daily_returns = (
            split_trade_log.groupby("date", as_index=False)
            .agg(
                portfolio_return=(return_column, "mean"),
                avg_turnover=("turnover", "mean"),
                total_cost=(cost_column, "sum"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        portfolio_scale_factor: float | None = None
        pre_target_annual_vol: float | None = None
        if self.env_config.reward_mode == "zhang" and not daily_returns.empty:
            pre_target_annual_vol = float(
                daily_returns["portfolio_return"].std(ddof=0) * math.sqrt(252.0)
            )
            if pre_target_annual_vol > 0:
                portfolio_scale_factor = float(self.env_config.vol_target / pre_target_annual_vol)
                daily_returns["portfolio_return_unscaled"] = daily_returns["portfolio_return"]
                daily_returns["avg_turnover_unscaled"] = daily_returns["avg_turnover"]
                daily_returns["total_cost_unscaled"] = daily_returns["total_cost"]
                daily_returns["portfolio_return"] = daily_returns["portfolio_return"] * portfolio_scale_factor
                daily_returns["avg_turnover"] = daily_returns["avg_turnover"] * abs(portfolio_scale_factor)
                daily_returns["total_cost"] = daily_returns["total_cost"] * abs(portfolio_scale_factor)
                daily_returns["portfolio_scale_factor"] = portfolio_scale_factor

        portfolio_metrics = compute_performance_metrics(
            daily_returns["portfolio_return"],
            turnover=daily_returns["avg_turnover"],
            total_cost=float(daily_returns["total_cost"].sum()),
        )
        if portfolio_scale_factor is not None and pre_target_annual_vol is not None:
            portfolio_metrics["portfolio_scale_factor"] = float(portfolio_scale_factor)
            portfolio_metrics["pre_target_annualized_volatility"] = float(pre_target_annual_vol)
            portfolio_metrics["portfolio_vol_target"] = float(self.env_config.vol_target)
        symbol_metrics = pd.DataFrame(symbol_metric_rows).sort_values("symbol").reset_index(drop=True)

        return EvalReport(
            policy_name=policy.name,
            split=split,
            portfolio_metrics=portfolio_metrics,
            symbol_metrics=symbol_metrics,
            daily_returns=daily_returns,
            trade_log=split_trade_log.reset_index(drop=True),
        )
