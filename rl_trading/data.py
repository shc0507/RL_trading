"""Data fetching with local parquet caching."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from rl_trading.config import DATA_SOURCE

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_CLC_DIR = Path(__file__).resolve().parent.parent / "CLCDATA"


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
    """Daily OHLCV via yfinance, returned as a (symbol, date)-sorted frame."""
    if cache:
        path = _cache_path(symbols, start, end)
        if path.exists():
            return pd.read_parquet(path)

    # yfinance treats `end` as exclusive; bump one day so it's included.
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
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


def load_clc_bars(
    symbols: list[str],
    start: str,
    end: str,
    *,
    adjustment: str = "RAD",
) -> pd.DataFrame:
    """Load Pinnacle CLC daily bars from local CSVs.

    Files: ``CLCDATA/<SYMBOL>_<ADJ>.CSV``, no header,
    ``date,open,high,low,close,volume,open_interest`` with date in MM/DD/YYYY.
    ``adjustment`` is RAD (Zhang's choice), REV, or NON.
    """
    adj = adjustment.upper()
    if adj not in {"RAD", "REV", "NON"}:
        raise ValueError(f"adjustment must be RAD/REV/NON; got {adjustment!r}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        path = _CLC_DIR / f"{sym}_{adj}.CSV"
        if not path.exists():
            raise FileNotFoundError(f"CLC file not found: {path}")
        df = pd.read_csv(
            path,
            header=None,
            names=["date", "open", "high", "low", "close", "volume", "open_interest"],
        )
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        df = df.dropna(subset=["close"])
        if df.empty:
            continue
        df["symbol"] = sym
        # RAD is already adjusted; keep adj_close column for schema parity.
        df["adj_close"] = df["close"]
        frames.append(
            df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
        )

    if not frames:
        return pd.DataFrame(
            columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def load_bars(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if DATA_SOURCE == "clc":
        return load_clc_bars(symbols, start, end)
    return fetch_bars(symbols, start=start, end=end)
