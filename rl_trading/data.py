"""Data fetching with local parquet caching."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_path(symbols: list[str], start: str, end: str) -> Path:
    key = hashlib.md5(
        f"{sorted(symbols)}|{start}|{end}".encode()
    ).hexdigest()[:12]
    return _CACHE_DIR / f"bars_{key}.parquet"


def fetch_bars(
    symbols: list[str],
    start: str,
    end: str,
    *,
    cache: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV bars via yfinance.

    Returns a DataFrame with columns:
        date, symbol, open, high, low, close, adj_close, volume
    sorted by (symbol, date).
    """
    if cache:
        path = _cache_path(symbols, start, end)
        if path.exists():
            return pd.read_parquet(path)

    # yfinance treats `end` as exclusive; add one day to include it.
    end_inclusive = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        symbols,
        start=start,
        end=end_inclusive,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = raw[sym].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue
        # Flatten any remaining MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # Rename 'adj_close' if yfinance returns 'adj close'
        if "adj_close" not in df.columns and "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        frames.append(
            df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
        )

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    if cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)

    return result
