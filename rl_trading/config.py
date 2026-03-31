"""Project-level defaults."""

from __future__ import annotations

from pathlib import Path

DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_SPLITS = {
    "train": ("2005-01-01", "2015-12-31"),
    "val": ("2016-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2025-12-31"),
}

DEFAULT_INSTRUMENTS = (
    {"symbol": "SPY", "stooq_symbol": "spy.us", "asset_class": "equity_etf", "currency": "USD"},
    {"symbol": "QQQ", "stooq_symbol": "qqq.us", "asset_class": "equity_etf", "currency": "USD"},
    {"symbol": "IWM", "stooq_symbol": "iwm.us", "asset_class": "equity_etf", "currency": "USD"},
    {"symbol": "DIA", "stooq_symbol": "dia.us", "asset_class": "equity_etf", "currency": "USD"},
    {"symbol": "EFA", "stooq_symbol": "efa.us", "asset_class": "equity_etf", "currency": "USD"},
    {"symbol": "EEM", "stooq_symbol": "eem.us", "asset_class": "equity_etf", "currency": "USD"},
)

DEFAULT_SYMBOLS = tuple(instrument["symbol"] for instrument in DEFAULT_INSTRUMENTS)
DEFAULT_INSTRUMENT_MAP = {instrument["symbol"]: instrument for instrument in DEFAULT_INSTRUMENTS}

OBSERVATION_WINDOW = 60
VOLATILITY_SPAN = 60
RSI_WINDOW = 30
MACD_WINDOWS = ((8, 24), (16, 48), (32, 96))
RETURN_HORIZONS = (21, 42, 63, 252)
MACD_PRICE_STD_WINDOW = 63
MACD_NORMALIZATION_WINDOW = 252
ANNUALIZATION_FACTOR = 252
DEFAULT_VOL_TARGET = 0.15
DEFAULT_COST_RATE_BP = 20.0

DEFAULT_FEATURE_COLUMNS = (
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
)

DEFAULT_OUTPUT_DIR = Path("artifacts")
