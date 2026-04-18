"""Feature engineering per Zhang et al. (2019)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rl_trading.config import (
    ANNUALIZATION_FACTOR,
    MACD_NORMALIZATION_WINDOW,
    MACD_PRICE_STD_WINDOW,
    MACD_WINDOWS,
    RETURN_HORIZONS,
    RSI_WINDOW,
    VOLATILITY_SPAN,
)

# ── Feature column names (the state vector) ─────────────────────────
FEATURE_COLS: list[str] = [
    "norm_close",
    "ret_21_vol",
    "ret_42_vol",
    "ret_63_vol",
    "ret_252_vol",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96",
    "macd_signal",
    "rsi_30",
]


def _phi(x: pd.Series) -> pd.Series:
    """Zhang's non-linear mapping: x * exp(-x^2/4) / 0.89"""
    return x * np.exp(-(x**2) / 4) / 0.89


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Standard RSI scaled to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return rs / (1 + rs)  # equivalent to RSI/100


def _compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all feature columns to a single-symbol DataFrame (sorted by date)."""
    df = df.copy()
    close = df["adj_close"]
    daily_ret = close.pct_change()

    # EWM volatility (annualized daily std)
    ewm_vol = daily_ret.ewm(span=VOLATILITY_SPAN, min_periods=VOLATILITY_SPAN, adjust=False).std()
    df["ewm_vol"] = ewm_vol
    ann_vol = ewm_vol * math.sqrt(ANNUALIZATION_FACTOR)

    # Normalized close: r_{t-252} / (sigma_t * sqrt(252))
    ret_252 = close / close.shift(252) - 1
    df["norm_close"] = ret_252 / ann_vol

    # Vol-adjusted returns for each horizon
    for h in RETURN_HORIZONS:
        cum_ret = close / close.shift(h) - 1
        df[f"ret_{h}"] = cum_ret
        df[f"ret_{h}_vol"] = cum_ret / ann_vol

    # MACD for each scale pair
    price_std = close.rolling(MACD_PRICE_STD_WINDOW, min_periods=MACD_PRICE_STD_WINDOW).std()
    macd_values: list[pd.Series] = []
    for short, long in MACD_WINDOWS:
        ema_s = close.ewm(span=short, min_periods=short).mean()
        ema_l = close.ewm(span=long, min_periods=long).mean()
        q = (ema_s - ema_l) / price_std
        q_std = q.rolling(MACD_NORMALIZATION_WINDOW, min_periods=MACD_NORMALIZATION_WINDOW).std()
        macd = q / q_std
        col = f"macd_{short}_{long}"
        df[col] = macd
        macd_values.append(macd)

    # Combined MACD signal: mean of phi(MACD) across scales
    phi_stack = pd.concat([_phi(m) for m in macd_values], axis=1)
    df["macd_signal"] = phi_stack.mean(axis=1)

    # RSI(30) scaled to [0, 1]
    df["rsi_30"] = _rsi(close, RSI_WINDOW)

    # Window-ready flag: enough history for all features
    max_ema_span = max(long for _, long in MACD_WINDOWS)
    min_needed = max(
        252,  # norm_close / ret_252
        max(max_ema_span, MACD_PRICE_STD_WINDOW) + MACD_NORMALIZATION_WINDOW - 2,  # MACD
    )
    df["window_ready"] = False
    df.iloc[min_needed:, df.columns.get_loc("window_ready")] = True

    return df


class FeatureBuilder:
    """Compute Zhang et al. features for a multi-symbol bar DataFrame."""

    def transform(self, bar_frame: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns per symbol.

        Parameters
        ----------
        bar_frame : DataFrame with at least columns: date, symbol, close

        Returns
        -------
        DataFrame with original columns plus feature columns and window_ready flag.
        """
        groups = []
        for _sym, grp in bar_frame.groupby("symbol"):
            grp = grp.sort_values("date").reset_index(drop=True)
            groups.append(_compute_symbol_features(grp))
        return pd.concat(groups, ignore_index=True)
