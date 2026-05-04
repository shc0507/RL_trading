"""Performance metrics matching Zhang et al. (2019) Exhibit 2."""

from __future__ import annotations

import math

import numpy as np


def compute_metrics(daily_rewards: np.ndarray) -> dict[str, float]:
    """Zhang Exhibit 2 metrics from a daily trade-return series."""
    r = np.asarray(daily_rewards, dtype=np.float64)
    n = len(r)
    if n < 2:
        return {k: 0.0 for k in [
            "E(R)", "Std(R)", "DD", "Sharpe", "Sortino",
            "MDD", "Calmar", "%+Ret", "AvgP/AvgL",
        ]}

    mean_daily = r.mean()
    std_daily = r.std(ddof=1)
    er = mean_daily * 252
    std_r = std_daily * math.sqrt(252)

    neg = r[r < 0]
    dd = neg.std(ddof=1) * math.sqrt(252) if len(neg) > 1 else 1e-10

    sharpe = er / std_r if std_r > 1e-10 else 0.0
    sortino = er / dd if dd > 1e-10 else 0.0

    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    mdd = (running_max - cum).max() if len(cum) > 0 else 0.0
    calmar = er / mdd if mdd > 1e-10 else 0.0

    pct_pos = (r > 0).sum() / n
    pos = r[r > 0]
    neg_abs = r[r < 0]
    avg_p = pos.mean() if len(pos) > 0 else 0.0
    avg_l = abs(neg_abs.mean()) if len(neg_abs) > 0 else 1e-10
    avg_p_avg_l = avg_p / avg_l if avg_l > 1e-10 else 0.0

    return {
        "E(R)": er,
        "Std(R)": std_r,
        "DD": dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Calmar": calmar,
        "%+Ret": pct_pos,
        "AvgP/AvgL": avg_p_avg_l,
    }
