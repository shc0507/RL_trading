"""Baseline strategies from Zhang et al. (2019)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rl_trading.config import (
    DEFAULT_COST_RATE_BP,
    DEFAULT_VOL_TARGET,
    TRAIN_START, TRAIN_END,
    VAL_START, VAL_END,
    TEST_START, TEST_END,
)

_SPLITS: dict[str, tuple[str, str]] = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}


def _get_symbol_split(
    feature_frame: pd.DataFrame, symbol: str, split: str = "test",
    *, start: str | None = None, end: str | None = None,
) -> pd.DataFrame:
    """Filter feature_frame to one symbol's window_ready rows in the split."""
    if start is not None and end is not None:
        date_start, date_end = start, end
    else:
        date_start, date_end = _SPLITS[split]
    mask = (
        (feature_frame["symbol"] == symbol)
        & (feature_frame["window_ready"])
        & (feature_frame["date"] >= date_start)
        & (feature_frame["date"] <= date_end)
    )
    return feature_frame.loc[mask].sort_values("date").reset_index(drop=True)


def long_only(
    feature_frame: pd.DataFrame, symbol: str, split: str = "test",
    *, start: str | None = None, end: str | None = None,
) -> np.ndarray:
    df = _get_symbol_split(feature_frame, symbol, split, start=start, end=end)
    return np.ones(len(df))


def sign_r(
    feature_frame: pd.DataFrame, symbol: str, split: str = "test",
    *, start: str | None = None, end: str | None = None,
) -> np.ndarray:
    """A_t = sign(r_{t-252,t}) per Moskowitz/Lim."""
    df = _get_symbol_split(feature_frame, symbol, split, start=start, end=end)
    return np.sign(df["ret_252"].to_numpy())


def macd_signal(
    feature_frame: pd.DataFrame, symbol: str, split: str = "test",
    *, start: str | None = None, end: str | None = None,
) -> np.ndarray:
    """A_t = combined φ-smoothed MACD signal (Zhang Eq. 12)."""
    df = _get_symbol_split(feature_frame, symbol, split, start=start, end=end)
    return df["macd_signal"].to_numpy()


def compute_baseline_rewards(
    positions: np.ndarray,
    feature_frame: pd.DataFrame,
    symbol: str,
    split: str = "test",
    *,
    start: str | None = None,
    end: str | None = None,
    vol_target: float = DEFAULT_VOL_TARGET,
    cost_rate_bp: float = DEFAULT_COST_RATE_BP,
) -> pd.Series:
    """Daily trade returns for a baseline position series.

    Mirrors TradingEnv.step exactly: same σ-floor, same μ=1/p_ref
    normalization, same Eq. 4 cost term. Returns one fewer row than
    the position array (need the next price to compute the return).
    """
    df = _get_symbol_split(feature_frame, symbol, split, start=start, end=end)
    prices = df["close"].to_numpy(dtype=np.float64)
    ewm_vol = df["ewm_vol"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"].to_numpy())
    bp = cost_rate_bp / 10_000

    assert len(positions) >= len(prices) - 1, (
        f"Position array too short: {len(positions)} < {len(prices) - 1}"
    )
    n = len(prices) - 1
    rewards = np.empty(n, dtype=np.float64)
    prev_pos = 0.0
    prev_vol_scale = 0.0
    ref_price = float(prices[0]) if prices[0] > 0 else 1.0

    for t in range(n):
        daily_ret = prices[t + 1] - prices[t]
        ann_vol = ewm_vol[t] * math.sqrt(252)
        if np.isnan(ann_vol) or ann_vol <= 0.0:
            vol_scale = 0.0
        else:
            vol_scale = vol_target / max(ann_vol, 0.01)

        pos = positions[t]
        position_return = vol_scale * pos * daily_ret
        tc = bp * prices[t] * abs(vol_scale * pos - prev_vol_scale * prev_pos)
        rewards[t] = (position_return - tc) / ref_price
        prev_pos = pos
        prev_vol_scale = vol_scale

    return pd.Series(rewards, index=dates[:n], name=symbol)
