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
    feature_frame: pd.DataFrame, symbol: str, split: str,
) -> pd.DataFrame:
    """Filter feature_frame to a single symbol and split (window_ready rows only)."""
    start, end = _SPLITS[split]
    mask = (
        (feature_frame["symbol"] == symbol)
        & (feature_frame["window_ready"])
        & (feature_frame["date"] >= start)
        & (feature_frame["date"] <= end)
    )
    return feature_frame.loc[mask].sort_values("date").reset_index(drop=True)


def long_only(feature_frame: pd.DataFrame, symbol: str, split: str) -> np.ndarray:
    """Returns array of positions (all 1.0)."""
    df = _get_symbol_split(feature_frame, symbol, split)
    return np.ones(len(df))


def sign_r(feature_frame: pd.DataFrame, symbol: str, split: str) -> np.ndarray:
    """Returns array of positions based on sign of 252-day return."""
    df = _get_symbol_split(feature_frame, symbol, split)
    return np.sign(df["ret_252_vol"].to_numpy())


def macd_signal(feature_frame: pd.DataFrame, symbol: str, split: str) -> np.ndarray:
    """Returns array of positions from the combined MACD signal."""
    df = _get_symbol_split(feature_frame, symbol, split)
    return df["macd_signal"].to_numpy()


def compute_baseline_rewards(
    positions: np.ndarray,
    feature_frame: pd.DataFrame,
    symbol: str,
    split: str,
    vol_target: float = DEFAULT_VOL_TARGET,
    cost_rate_bp: float = DEFAULT_COST_RATE_BP,
) -> pd.Series:
    """Compute daily trade returns for a given position series.

    Uses the same reward formula as TradingEnv:
        reward_t = vol_scale * position_t * daily_ret_t - tc_t

    Returns a Series indexed by date (length = len(positions) - 1,
    since we need the next price to compute daily return).
    """
    df = _get_symbol_split(feature_frame, symbol, split)
    prices = df["adj_close"].to_numpy(dtype=np.float64)
    ewm_vol = df["ewm_vol"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"].to_numpy())
    bp = cost_rate_bp / 10_000

    n = min(len(positions), len(prices) - 1)
    rewards = np.empty(n, dtype=np.float64)
    prev_pos = 0.0

    for t in range(n):
        daily_ret = prices[t + 1] / prices[t] - 1.0
        ann_vol = ewm_vol[t] * math.sqrt(252)
        if ann_vol < 1e-8 or np.isnan(ann_vol):
            ann_vol = 1e-8
        vol_scale = vol_target / ann_vol

        pos = positions[t]
        position_return = vol_scale * pos * daily_ret
        tc = bp * abs(vol_scale * pos - vol_scale * prev_pos)
        rewards[t] = position_return - tc
        prev_pos = pos

    return pd.Series(rewards, index=dates[:n], name=symbol)
