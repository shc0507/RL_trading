"""Feature engineering following Zhang-style daily signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_FEATURE_COLUMNS,
    MACD_NORMALIZATION_WINDOW,
    MACD_PRICE_STD_WINDOW,
    MACD_WINDOWS,
    OBSERVATION_WINDOW,
    RETURN_HORIZONS,
    RSI_WINDOW,
    VOLATILITY_SPAN,
)


def _safe_series_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)


def _compute_rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return rsi.clip(lower=0.0, upper=100.0)


@dataclass(slots=True)
class FeatureBuilder:
    observation_window: int = OBSERVATION_WINDOW
    volatility_span: int = VOLATILITY_SPAN
    return_horizons: tuple[int, ...] = RETURN_HORIZONS
    macd_windows: tuple[tuple[int, int], ...] = MACD_WINDOWS
    rsi_window: int = RSI_WINDOW
    macd_price_std_window: int = MACD_PRICE_STD_WINDOW
    macd_normalization_window: int = MACD_NORMALIZATION_WINDOW

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return DEFAULT_FEATURE_COLUMNS

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            "date",
            "symbol",
            "adj_close",
            "return_1d",
            "ewm_vol_60",
            "window_ready",
            *self.feature_columns,
        )

    def transform(self, bar_frame: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for symbol, symbol_frame in bar_frame.groupby("symbol", sort=False):
            frames.append(self._transform_symbol(symbol, symbol_frame.copy()))
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _transform_symbol(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("date").reset_index(drop=True)
        row_number = np.arange(len(frame))
        price = frame["adj_close"].astype(float)
        daily_return = price.pct_change()
        ewm_vol = daily_return.ewm(
            span=self.volatility_span,
            adjust=False,
            min_periods=self.volatility_span,
        ).std(bias=False)
        vol_ready = pd.Series(row_number >= self.volatility_span, index=frame.index)
        frame["ewm_vol_60"] = ewm_vol.where(vol_ready, np.nan)
        frame.loc[vol_ready, "ewm_vol_60"] = frame.loc[vol_ready, "ewm_vol_60"].fillna(0.0)

        rolling_mean = price.rolling(self.observation_window, min_periods=self.observation_window).mean()
        rolling_std = price.rolling(self.observation_window, min_periods=self.observation_window).std(ddof=0)
        norm_ready = pd.Series(row_number >= self.observation_window - 1, index=frame.index)
        norm_close = ((price - rolling_mean) / rolling_std.replace(0.0, np.nan)).where(norm_ready, np.nan)
        frame["norm_close"] = norm_close
        frame.loc[norm_ready, "norm_close"] = frame.loc[norm_ready, "norm_close"].fillna(0.0)

        frame["return_1d"] = daily_return
        for horizon in self.return_horizons:
            raw_column = f"ret_{horizon}_raw"
            vol_column = f"ret_{horizon}_vol"
            frame[raw_column] = price.pct_change(horizon)
            return_ready = pd.Series(row_number >= max(horizon, self.volatility_span), index=frame.index)
            normalized = frame[raw_column] / (frame["ewm_vol_60"].replace(0.0, np.nan) * np.sqrt(horizon))
            normalized = normalized.where(return_ready, np.nan)
            frame[vol_column] = normalized
            frame.loc[return_ready, vol_column] = frame.loc[return_ready, vol_column].replace(
                [np.inf, -np.inf], np.nan
            )
            frame.loc[return_ready, vol_column] = frame.loc[return_ready, vol_column].fillna(0.0)

        macd_columns = []
        price_std = _safe_series_std(price, self.macd_price_std_window)
        for short_window, long_window in self.macd_windows:
            short_ema = price.ewm(span=short_window, adjust=False, min_periods=short_window).mean()
            long_ema = price.ewm(span=long_window, adjust=False, min_periods=long_window).mean()
            q = (short_ema - long_ema) / price_std
            macd_name = f"macd_{short_window}_{long_window}"
            macd_ready = pd.Series(
                row_number >= (max(long_window, self.macd_price_std_window) + self.macd_normalization_window - 2),
                index=frame.index,
            )
            macd_value = (q / _safe_series_std(q, self.macd_normalization_window)).where(macd_ready, np.nan)
            frame[macd_name] = macd_value
            frame.loc[macd_ready, macd_name] = frame.loc[macd_ready, macd_name].replace([np.inf, -np.inf], np.nan)
            frame.loc[macd_ready, macd_name] = frame.loc[macd_ready, macd_name].fillna(0.0)
            macd_columns.append(macd_name)

        frame["macd_signal"] = frame[macd_columns].mean(axis=1)
        frame[f"rsi_{self.rsi_window}"] = _compute_rsi(price, self.rsi_window)

        window_start = frame["date"].shift(self.observation_window - 1)
        frame["window_start_date"] = pd.to_datetime(window_start)
        frame["window_end_date"] = pd.to_datetime(frame["date"])
        frame["window_size"] = np.where(frame["window_start_date"].notna(), self.observation_window, np.nan)
        frame["window_ready"] = (
            frame["window_start_date"].notna()
            & frame[list(self.feature_columns)].notna().all(axis=1)
            & frame["ewm_vol_60"].notna()
        )
        frame["symbol"] = symbol
        return frame

    def check_leakage(self, bar_frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
        report: dict[str, list[dict[str, object]]] = {}
        full_features = self.transform(bar_frame)
        compare_columns = list(self.feature_columns) + ["ewm_vol_60", "return_1d"]

        for symbol, symbol_bars in bar_frame.groupby("symbol", sort=False):
            symbol_full = full_features.loc[full_features["symbol"] == symbol].reset_index(drop=True)
            ready_indices = symbol_full.index[symbol_full["window_ready"]].tolist()
            if not ready_indices:
                report[symbol] = []
                continue

            sample_indices = sorted({ready_indices[0], ready_indices[len(ready_indices) // 2], ready_indices[-1]})
            symbol_report = []
            for index in sample_indices:
                truncated = symbol_bars.iloc[: index + 1].copy()
                truncated_features = self.transform(truncated)
                full_row = symbol_full.loc[index, compare_columns]
                truncated_row = truncated_features.loc[index, compare_columns]
                max_abs_diff = (
                    (full_row.astype(float) - truncated_row.astype(float)).abs().replace(np.nan, 0.0).max()
                )
                symbol_report.append(
                    {
                        "row_index": int(index),
                        "date": str(symbol_full.loc[index, "date"].date()),
                        "max_abs_diff": float(max_abs_diff),
                        "passed": bool(max_abs_diff < 1e-12),
                    }
                )
            report[symbol] = symbol_report
        return report


def validate_feature_frame(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature frame missing required columns: {missing}")

    if frame[["symbol", "date"]].duplicated().any():
        raise ValueError("duplicate symbol/date rows found in feature frame")

    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        if not symbol_frame["date"].is_monotonic_increasing:
            raise ValueError(f"feature dates are not increasing for symbol {symbol}")

    ready_frame = frame.loc[frame["window_ready"]].copy()
    if ready_frame.empty:
        raise ValueError("no rows remain after feature warmup")

    required_numeric = [column for column in required_columns if column not in {"date", "symbol", "window_ready"}]
    if ready_frame[required_numeric].isna().any().any():
        raise ValueError("feature frame contains NaNs after warmup")

    rsi_columns = [column for column in frame.columns if column.startswith("rsi_")]
    for column in rsi_columns:
        valid = frame[column].dropna()
        if ((valid < 0.0) | (valid > 100.0)).any():
            raise ValueError(f"{column} falls outside 0-100")
