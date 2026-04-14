"""Performance metrics matching Zhang et al. (2019) Exhibit 2."""

from __future__ import annotations

import math

import numpy as np


def compute_metrics(daily_rewards: np.ndarray) -> dict[str, float]:
    """Compute all Zhang-style performance metrics from daily trade returns.

    Parameters
    ----------
    daily_rewards : 1-D array of daily trade returns (already vol-scaled, net of costs).

    Returns
    -------
    Dict with keys: E(R), Std(R), DD, Sharpe, Sortino, MDD, Calmar, %+Ret, AvgP/AvgL
    """
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

    # Downside deviation: annualized std of negative returns only
    neg = r[r < 0]
    if len(neg) > 1:
        dd = neg.std(ddof=1) * math.sqrt(252)
    else:
        dd = 1e-10

    sharpe = er / std_r if std_r > 1e-10 else 0.0
    sortino = er / dd if dd > 1e-10 else 0.0

    # Maximum drawdown from cumulative returns
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    mdd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    calmar = er / mdd if mdd > 1e-10 else 0.0

    pct_pos = (r > 0).sum() / n * 100

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
