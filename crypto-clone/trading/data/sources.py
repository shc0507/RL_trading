"""Market data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from ..config import DEFAULT_INSTRUMENT_MAP
from ..schema import CANONICAL_BAR_COLUMNS, coerce_canonical_bar_frame


class BarDataSource(ABC):
    """Abstract bar data source."""

    @abstractmethod
    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError


class PublicDailySource(BarDataSource):
    """Immediate public data source backed by Yahoo Finance daily bars."""

    def __init__(self, instrument_map: dict[str, dict[str, str]] | None = None) -> None:
        self.instrument_map = instrument_map or DEFAULT_INSTRUMENT_MAP

    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        # Yahoo's `end` parameter is exclusive, so add one day to keep the requested end date.
        inclusive_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for symbol in symbols:
            metadata = self.instrument_map.get(
                symbol,
                {
                    # Allow ad-hoc tickers for one-off research modules like pairs trading.
                    "yahoo_symbol": symbol,
                    "asset_class": "unknown",
                    "currency": "USD",
                },
            )
            ticker = metadata.get("yahoo_symbol", symbol)
            frame = yf.download(
                tickers=ticker,
                start=start_date,
                end=inclusive_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if frame.empty:
                raise ValueError(f"no rows returned for symbol {symbol}")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index().rename(
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
            frame["symbol"] = symbol
            frame["asset_class"] = metadata["asset_class"]
            frame["currency"] = metadata["currency"]
            if "adj_close" not in frame.columns:
                frame["adj_close"] = frame["close"]
            frame["source"] = "public_daily:yfinance"
            frame = frame.loc[:, CANONICAL_BAR_COLUMNS]
            frames.append(frame)

        result = pd.concat(frames, ignore_index=True)
        result = coerce_canonical_bar_frame(result)
        mask = (result["date"] >= pd.Timestamp(start_date)) & (result["date"] <= pd.Timestamp(end_date))
        return result.loc[mask].reset_index(drop=True)


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
    ) -> None:
        self.root = Path(root)
        self.source_name = source_name
        self.default_asset_class = default_asset_class
        self.default_currency = default_currency

    def fetch(self, symbols: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        files = [self.root] if self.root.is_file() else sorted(self.root.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no csv files found under {self.root}")

        wanted_symbols = set(symbols)
        frames: list[pd.DataFrame] = []
        for path in files:
            frame = pd.read_csv(path)
            normalized = self._normalize_columns(frame, path)
            frames.append(normalized)

        result = pd.concat(frames, ignore_index=True)
        if wanted_symbols:
            result = result.loc[result["symbol"].isin(wanted_symbols)].copy()
        result["source"] = self.source_name
        result = coerce_canonical_bar_frame(result)
        mask = (result["date"] >= pd.Timestamp(start_date)) & (result["date"] <= pd.Timestamp(end_date))
        return result.loc[mask].reset_index(drop=True)

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
