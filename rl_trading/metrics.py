"""Performance metric helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import ANNUALIZATION_FACTOR


def compute_performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    total_cost: float | None = None,
    curve_mode: str = "additive",
) -> dict[str, float]:
    clean_returns = returns.fillna(0.0)
    annual_return = float(clean_returns.mean() * ANNUALIZATION_FACTOR)
    annual_vol = float(clean_returns.std(ddof=0) * math.sqrt(ANNUALIZATION_FACTOR))
    downside = clean_returns[clean_returns < 0.0]
    downside_deviation = float(downside.std(ddof=0) * math.sqrt(ANNUALIZATION_FACTOR)) if not downside.empty else 0.0
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    sortino = annual_return / downside_deviation if downside_deviation > 0 else 0.0

    if curve_mode == "multiplicative":
        equity_curve = (1.0 + clean_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 0.0
    else:
        cumulative_trade_return = clean_returns.cumsum()
        running_max = cumulative_trade_return.cummax()
        drawdown = cumulative_trade_return - running_max
        max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 0.0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    positive = clean_returns[clean_returns > 0.0]
    negative = clean_returns[clean_returns < 0.0]

    metrics = {
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "hit_rate": float((clean_returns > 0.0).mean()),
        "avg_win": float(positive.mean()) if not positive.empty else 0.0,
        "avg_loss": float(negative.mean()) if not negative.empty else 0.0,
        "num_days": float(len(clean_returns)),
    }
    if turnover is not None:
        metrics["avg_daily_turnover"] = float(turnover.fillna(0.0).mean())
        metrics["total_turnover"] = float(turnover.fillna(0.0).sum())
    if total_cost is not None:
        metrics["total_transaction_cost"] = float(total_cost)
    return metrics
