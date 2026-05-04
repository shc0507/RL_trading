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
    """Zhang's non-linear smoother x · exp(-x²/4) / 0.89."""
    return x * np.exp(-(x**2) / 4) / 0.89


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder (1978) RSI on 0..100 scale.

    Wilder's smoother is α = 1/n (``ewm(alpha=1/n, adjust=False)``), NOT
    ``span=n`` which gives α = 2/(n+1) ≈ 2× faster.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan))
    return rsi.where(avg_loss > 0, 100.0)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that yields NaN (not inf) for near-zero denominators.

    Resulting NaN rows are excluded from ``window_ready`` downstream.
    """
    safe_den = den.where(den.abs() > 1e-12)
    return num / safe_den


def _compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    daily_ret = close.pct_change()

    ewm_vol = daily_ret.ewm(span=VOLATILITY_SPAN, min_periods=VOLATILITY_SPAN, adjust=False).std()
    df["ewm_vol"] = ewm_vol
    ann_vol = ewm_vol * math.sqrt(ANNUALIZATION_FACTOR)

    ret_252 = close / close.shift(252) - 1
    df["norm_close"] = _safe_div(ret_252, ann_vol)

    for h in RETURN_HORIZONS:
        cum_ret = close / close.shift(h) - 1
        df[f"ret_{h}"] = cum_ret
        df[f"ret_{h}_vol"] = _safe_div(cum_ret, ann_vol)

    # adjust=False = recursive EMA (Baz/Zhang); pandas default differs
    # during warm-up.
    price_std = close.rolling(MACD_PRICE_STD_WINDOW, min_periods=MACD_PRICE_STD_WINDOW).std()
    macd_values: list[pd.Series] = []
    for short, long in MACD_WINDOWS:
        ema_s = close.ewm(span=short, min_periods=short, adjust=False).mean()
        ema_l = close.ewm(span=long, min_periods=long, adjust=False).mean()
        q = _safe_div(ema_s - ema_l, price_std)
        q_std = q.rolling(MACD_NORMALIZATION_WINDOW, min_periods=MACD_NORMALIZATION_WINDOW).std()
        macd = _safe_div(q, q_std)
        df[f"macd_{short}_{long}"] = macd
        macd_values.append(macd)

    phi_stack = pd.concat([_phi(m) for m in macd_values], axis=1)
    df["macd_signal"] = phi_stack.mean(axis=1)

    df["rsi_30"] = _rsi(close, RSI_WINDOW)

    # window_ready also requires every feature finite on the row, not
    # just enough history: flat-price stretches collapse denominators
    # to zero and would otherwise leak NaN into the LSTM.
    max_ema_span = max(long for _, long in MACD_WINDOWS)
    min_needed = max(
        252,
        max(max_ema_span, MACD_PRICE_STD_WINDOW) + MACD_NORMALIZATION_WINDOW - 2,
    )
    history_ok = np.zeros(len(df), dtype=bool)
    history_ok[min_needed:] = True
    finite_ok = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    df["window_ready"] = history_ok & finite_ok

    return df


class FeatureBuilder:
    """Compute Zhang features for a multi-symbol bar DataFrame."""

    def transform(self, bar_frame: pd.DataFrame) -> pd.DataFrame:
        groups = []
        for _sym, grp in bar_frame.groupby("symbol"):
            grp = grp.sort_values("date").reset_index(drop=True)
            groups.append(_compute_symbol_features(grp))
        return pd.concat(groups, ignore_index=True)
