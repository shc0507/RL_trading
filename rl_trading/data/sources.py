"""Market data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from pathlib import Path
from time import sleep
from typing import Iterable

import pandas as pd
import yfinance as yf

from ..config import DEFAULT_INSTRUMENT_MAP
from ..schema import CANONICAL_BAR_COLUMNS, coerce_canonical_bar_frame


class DataFetchError(RuntimeError):
    """Raised when a source cannot satisfy the requested fetch."""

    def __init__(self, message: str, failures: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


class BarDataSource(ABC):
    """Abstract bar data source."""

    @abstractmethod
    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_last_fetch_metadata(self) -> dict[str, object]:
        metadata = getattr(self, "_last_fetch_metadata", {})
        return copy.deepcopy(metadata) if isinstance(metadata, dict) else {}

    def get_last_raw_frames(self) -> dict[str, pd.DataFrame]:
        raw_frames = getattr(self, "_last_raw_frames", {})
        if not isinstance(raw_frames, dict):
            return {}
        return {name: frame.copy() for name, frame in raw_frames.items()}

    def _reset_fetch_state(self) -> None:
        self._last_fetch_metadata = {}
        self._last_raw_frames = {}


def _as_date_string(value: pd.Timestamp | str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            "__".join(str(part) for part in column if part not in ("", None))
            for column in flattened.columns.to_flat_index()
        ]
    return flattened


class PublicDailySource(BarDataSource):
    """Immediate public data source backed by Yahoo Finance daily bars."""

    def __init__(
        self,
        instrument_map: dict[str, dict[str, str]] | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        strict: bool = True,
    ) -> None:
        self.instrument_map = instrument_map or DEFAULT_INSTRUMENT_MAP
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.strict = strict

    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._reset_fetch_state()
        requested_symbols = list(symbols)
        frames: list[pd.DataFrame] = []
        raw_frames: dict[str, pd.DataFrame] = {}
        results: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        inclusive_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        for symbol in requested_symbols:
            metadata = self.instrument_map.get(symbol)
            if metadata is None:
                result = {
                    "symbol": symbol,
                    "ticker": symbol,
                    "status": "failed",
                    "attempts": 0,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "error": f"no metadata configured for symbol {symbol}",
                    "snapshot_key": None,
                }
                results.append(result)
                failures.append(result)
                continue

            ticker = metadata.get("yahoo_symbol", symbol)
            frame: pd.DataFrame | None = None
            last_error: str | None = None
            attempts = 0
            for attempts in range(1, self.max_retries + 1):
                try:
                    downloaded = yf.download(
                        tickers=ticker,
                        start=start_date,
                        end=inclusive_end,
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                    )
                except Exception as exc:  # pragma: no cover - exercised through mocking
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if downloaded.empty:
                        last_error = f"no rows returned for symbol {symbol}"
                    else:
                        frame = downloaded
                        break

                if attempts < self.max_retries and self.retry_delay_seconds > 0:
                    sleep(self.retry_delay_seconds * attempts)

            if frame is None:
                result = {
                    "symbol": symbol,
                    "ticker": ticker,
                    "status": "failed",
                    "attempts": attempts,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "error": last_error,
                    "snapshot_key": None,
                }
                results.append(result)
                failures.append(result)
                continue

            snapshot_key = f"public_daily_{symbol.lower()}"
            raw_frames[snapshot_key] = _flatten_columns(frame).reset_index()

            normalized_frame = frame.copy()
            if isinstance(normalized_frame.columns, pd.MultiIndex):
                normalized_frame.columns = normalized_frame.columns.get_level_values(0)
            normalized_frame = normalized_frame.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            normalized_frame["symbol"] = symbol
            normalized_frame["asset_class"] = metadata["asset_class"]
            normalized_frame["currency"] = metadata["currency"]
            if "adj_close" not in normalized_frame.columns:
                normalized_frame["adj_close"] = normalized_frame["close"]
            normalized_frame["source"] = "public_daily:yfinance"
            normalized_frame = normalized_frame.loc[:, CANONICAL_BAR_COLUMNS]
            normalized_frame = coerce_canonical_bar_frame(normalized_frame)
            mask = (normalized_frame["date"] >= pd.Timestamp(start_date)) & (
                normalized_frame["date"] <= pd.Timestamp(end_date)
            )
            normalized_frame = normalized_frame.loc[mask].reset_index(drop=True)
            if normalized_frame.empty:
                result = {
                    "symbol": symbol,
                    "ticker": ticker,
                    "status": "failed",
                    "attempts": attempts,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "error": f"no rows remained for symbol {symbol} after date filtering",
                    "snapshot_key": snapshot_key,
                }
                results.append(result)
                failures.append(result)
                continue
            frames.append(normalized_frame)
            results.append(
                {
                    "symbol": symbol,
                    "ticker": ticker,
                    "status": "success",
                    "attempts": attempts,
                    "rows": int(len(normalized_frame)),
                    "min_date": _as_date_string(normalized_frame["date"].min()),
                    "max_date": _as_date_string(normalized_frame["date"].max()),
                    "error": None,
                    "snapshot_key": snapshot_key,
                }
            )

        self._last_raw_frames = raw_frames
        self._last_fetch_metadata = {
            "source": "public_daily:yfinance",
            "requested_symbols": requested_symbols,
            "start_date": start_date,
            "end_date": end_date,
            "strict": self.strict,
            "max_retries": self.max_retries,
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "results": results,
            "success_count": len(frames),
            "failure_count": len(failures),
        }

        if failures and self.strict:
            raise DataFetchError(
                "public Yahoo Finance fetch failed for one or more symbols",
                failures=failures,
            )
        if not frames:
            raise DataFetchError("public Yahoo Finance fetch produced no successful symbols", failures=failures)

        return pd.concat(frames, ignore_index=True)


class InstitutionalCsvSource(BarDataSource):
    """CSV adapter for Bloomberg, Refinitiv, WRDS, or manual exports."""

    column_aliases = {
        "date": ("date", "Date", "DATE"),
        "symbol": ("symbol", "Symbol", "ticker", "Ticker", "TICKER"),
        "open": ("open", "Open", "PX_OPEN"),
        "high": ("high", "High", "PX_HIGH"),
        "low": ("low", "Low", "PX_LOW"),
        "close": ("close", "Close", "PX_LAST"),
        "adj_close": ("adj_close", "Adj Close", "adjusted_close", "TOT_RETURN_INDEX_GROSS_DVDS"),
        "volume": ("volume", "Volume", "PX_VOLUME"),
        "currency": ("currency", "Currency", "CRNCY"),
        "asset_class": ("asset_class", "AssetClass"),
    }

    def __init__(
        self,
        root: str | Path,
        source_name: str = "institutional_csv",
        default_asset_class: str = "unknown",
        default_currency: str = "USD",
        strict: bool = True,
    ) -> None:
        self.root = Path(root)
        self.source_name = source_name
        self.default_asset_class = default_asset_class
        self.default_currency = default_currency
        self.strict = strict

    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._reset_fetch_state()
        requested_symbols = list(symbols)
        files = [self.root] if self.root.is_file() else sorted(self.root.glob("*.csv"))
        if not files:
            self._last_fetch_metadata = {
                "source": self.source_name,
                "requested_symbols": requested_symbols,
                "start_date": start_date,
                "end_date": end_date,
                "strict": self.strict,
                "generated_at": pd.Timestamp.now("UTC").isoformat(),
                "results": [],
                "success_count": 0,
                "failure_count": 1,
            }
            raise FileNotFoundError(f"no csv files found under {self.root}")

        wanted_symbols = set(requested_symbols)
        frames: list[pd.DataFrame] = []
        raw_frames: dict[str, pd.DataFrame] = {}
        results: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        for path in files:
            snapshot_key = f"institutional_csv_{path.stem.lower()}"
            try:
                frame = pd.read_csv(path)
                raw_frames[snapshot_key] = frame.copy()
                normalized = self._normalize_columns(frame, path)
                normalized["source"] = self.source_name
                normalized = coerce_canonical_bar_frame(normalized)
                if wanted_symbols:
                    normalized = normalized.loc[normalized["symbol"].isin(wanted_symbols)].copy()
                normalized = normalized.loc[(normalized["date"] >= start) & (normalized["date"] <= end)].reset_index(
                    drop=True
                )
            except Exception as exc:
                result = {
                    "path": str(path),
                    "status": "failed",
                    "rows": 0,
                    "symbols": [],
                    "min_date": None,
                    "max_date": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "snapshot_key": snapshot_key if snapshot_key in raw_frames else None,
                }
                results.append(result)
                failures.append(result)
                continue

            if normalized.empty:
                results.append(
                    {
                        "path": str(path),
                        "status": "skipped",
                        "rows": 0,
                        "symbols": [],
                        "min_date": None,
                        "max_date": None,
                        "error": None,
                        "snapshot_key": snapshot_key,
                    }
                )
                continue

            frames.append(normalized)
            results.append(
                {
                    "path": str(path),
                    "status": "success",
                    "rows": int(len(normalized)),
                    "symbols": sorted(normalized["symbol"].astype(str).unique().tolist()),
                    "min_date": _as_date_string(normalized["date"].min()),
                    "max_date": _as_date_string(normalized["date"].max()),
                    "error": None,
                    "snapshot_key": snapshot_key,
                }
            )

        self._last_raw_frames = raw_frames
        self._last_fetch_metadata = {
            "source": self.source_name,
            "requested_symbols": requested_symbols,
            "start_date": start_date,
            "end_date": end_date,
            "strict": self.strict,
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "results": results,
            "success_count": len(frames),
            "failure_count": len(failures),
        }

        if failures and self.strict:
            raise DataFetchError(
                f"{self.source_name} fetch failed for one or more csv inputs",
                failures=failures,
            )
        if not frames:
            raise DataFetchError(f"{self.source_name} fetch produced no successful rows", failures=failures)

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _normalize_columns(self, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        normalized = {}
        for canonical_name, aliases in self.column_aliases.items():
            for alias in aliases:
                if alias in frame.columns:
                    normalized[canonical_name] = frame[alias]
                    break

        if "symbol" not in normalized:
            normalized["symbol"] = path.stem.upper()
        if "asset_class" not in normalized:
            normalized["asset_class"] = self.default_asset_class
        if "currency" not in normalized:
            normalized["currency"] = self.default_currency
        if "adj_close" not in normalized and "close" in normalized:
            normalized["adj_close"] = normalized["close"]
        if "volume" not in normalized:
            normalized["volume"] = 0.0

        return pd.DataFrame(normalized, copy=False)
