"""Canonical data-frame schema helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

CANONICAL_BAR_COLUMNS = (
    "date",
    "symbol",
    "asset_class",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "currency",
)

REQUIRED_BAR_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "source",
)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def coerce_canonical_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(frame, REQUIRED_BAR_COLUMNS, "bar frame")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=False).dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "asset_class" not in normalized.columns:
        normalized["asset_class"] = "unknown"
    if "currency" not in normalized.columns:
        normalized["currency"] = "USD"
    normalized = normalized.loc[:, CANONICAL_BAR_COLUMNS]
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def validate_canonical_bars(frame: pd.DataFrame) -> None:
    ensure_columns(frame, CANONICAL_BAR_COLUMNS, "bar frame")
    if frame[["symbol", "date"]].duplicated().any():
        duplicates = frame.loc[frame[["symbol", "date"]].duplicated(), ["symbol", "date"]]
        raise ValueError(f"duplicate symbol/date rows found: {duplicates.head().to_dict('records')}")

    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        if not symbol_frame["date"].is_monotonic_increasing:
            raise ValueError(f"dates are not strictly increasing for symbol {symbol}")

    numeric_columns = ["open", "high", "low", "close", "adj_close"]
    if frame[numeric_columns].isna().any().any():
        raise ValueError("price columns contain NaNs after normalization")
