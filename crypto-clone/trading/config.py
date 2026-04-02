"""Project-level defaults."""

from __future__ import annotations

from pathlib import Path

DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_CRYPTO_START_DATE = "2017-01-01"
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

DEFAULT_CRYPTO_INSTRUMENTS = (
    {
        "symbol": "BTC-USD",
        "yahoo_symbol": "BTC-USD",
        "broker_symbol": "BTC/USD",
        "base_asset": "BTC",
        "quote_asset": "USD",
        "asset_class": "crypto_spot",
        "currency": "USD",
    },
    {
        "symbol": "ETH-USD",
        "yahoo_symbol": "ETH-USD",
        "broker_symbol": "ETH/USD",
        "base_asset": "ETH",
        "quote_asset": "USD",
        "asset_class": "crypto_spot",
        "currency": "USD",
    },
    {
        "symbol": "SOL-USD",
        "yahoo_symbol": "SOL-USD",
        "broker_symbol": "SOL/USD",
        "base_asset": "SOL",
        "quote_asset": "USD",
        "asset_class": "crypto_spot",
        "currency": "USD",
    },
)

DEFAULT_CRYPTO_SYMBOLS = tuple(instrument["symbol"] for instrument in DEFAULT_CRYPTO_INSTRUMENTS)
DEFAULT_CRYPTO_INSTRUMENT_MAP = {instrument["symbol"]: instrument for instrument in DEFAULT_CRYPTO_INSTRUMENTS}

OBSERVATION_WINDOW = 60
VOLATILITY_SPAN = 60
RSI_WINDOW = 30
MACD_WINDOWS = ((8, 24), (16, 48), (32, 96))
RETURN_HORIZONS = (21, 42, 63, 252)
MACD_PRICE_STD_WINDOW = 63
MACD_NORMALIZATION_WINDOW = 252
ANNUALIZATION_FACTOR = 252
CRYPTO_ANNUALIZATION_FACTOR = 365
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


def get_default_symbols(market: str = "equity") -> tuple[str, ...]:
    normalized = market.lower().strip()
    if normalized in {"crypto", "crypto_daily"}:
        return DEFAULT_CRYPTO_SYMBOLS
    return DEFAULT_SYMBOLS


def get_instrument_map(market: str = "equity") -> dict[str, dict[str, str]]:
    normalized = market.lower().strip()
    if normalized in {"crypto", "crypto_daily"}:
        return DEFAULT_CRYPTO_INSTRUMENT_MAP
    return DEFAULT_INSTRUMENT_MAP


def get_periods_per_year(market: str = "equity") -> int:
    normalized = market.lower().strip()
    if normalized in {"crypto", "crypto_daily"}:
        return CRYPTO_ANNUALIZATION_FACTOR
    return ANNUALIZATION_FACTOR
