"""Project-wide defaults following Zhang et al. (2019)."""

from __future__ import annotations

# ── Feature windows ──────────────────────────────────────────────────
OBSERVATION_WINDOW: int = 60
VOLATILITY_SPAN: int = 60  # EWM vol lookback
RSI_WINDOW: int = 30
MACD_WINDOWS: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
MACD_PRICE_STD_WINDOW: int = 63
MACD_NORMALIZATION_WINDOW: int = 252
RETURN_HORIZONS: tuple[int, ...] = (21, 42, 63, 252)
ANNUALIZATION_FACTOR: int = 252

# ── Trading defaults ─────────────────────────────────────────────────
DEFAULT_VOL_TARGET: float = 0.15
DEFAULT_COST_RATE_BP: float = 2.0  # basis points

# ── Train / val / test splits ────────────────────────────────────────
TRAIN_START = "2005-01-01"
TRAIN_END = "2015-12-31"
VAL_START = "2016-01-01"
VAL_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2025-12-31"

# ── Asset universe (~50 tickers) ─────────────────────────────────────
UNIVERSE: list[dict[str, str]] = [
    # Commodities (~25)
    {"symbol": "GLD", "asset_class": "commodity"},
    {"symbol": "SLV", "asset_class": "commodity"},
    {"symbol": "USO", "asset_class": "commodity"},
    {"symbol": "UNG", "asset_class": "commodity"},
    {"symbol": "CORN", "asset_class": "commodity"},
    {"symbol": "SOYB", "asset_class": "commodity"},
    {"symbol": "WEAT", "asset_class": "commodity"},
    {"symbol": "DBA", "asset_class": "commodity"},
    {"symbol": "DBC", "asset_class": "commodity"},
    {"symbol": "CPER", "asset_class": "commodity"},
    {"symbol": "PALL", "asset_class": "commodity"},
    {"symbol": "PPLT", "asset_class": "commodity"},
    {"symbol": "JO", "asset_class": "commodity"},
    {"symbol": "NIB", "asset_class": "commodity"},
    {"symbol": "SGG", "asset_class": "commodity"},
    {"symbol": "COW", "asset_class": "commodity"},
    {"symbol": "UGA", "asset_class": "commodity"},
    {"symbol": "BNO", "asset_class": "commodity"},
    {"symbol": "PDBC", "asset_class": "commodity"},
    {"symbol": "GSG", "asset_class": "commodity"},
    # Equity Indexes (~11)
    {"symbol": "SPY", "asset_class": "equity_index"},
    {"symbol": "QQQ", "asset_class": "equity_index"},
    {"symbol": "IWM", "asset_class": "equity_index"},
    {"symbol": "DIA", "asset_class": "equity_index"},
    {"symbol": "EFA", "asset_class": "equity_index"},
    {"symbol": "EEM", "asset_class": "equity_index"},
    {"symbol": "VGK", "asset_class": "equity_index"},
    {"symbol": "EWJ", "asset_class": "equity_index"},
    {"symbol": "FXI", "asset_class": "equity_index"},
    {"symbol": "ACWI", "asset_class": "equity_index"},
    {"symbol": "MDY", "asset_class": "equity_index"},
    # Fixed Income (~5)
    {"symbol": "TLT", "asset_class": "fixed_income"},
    {"symbol": "IEF", "asset_class": "fixed_income"},
    {"symbol": "SHY", "asset_class": "fixed_income"},
    {"symbol": "LQD", "asset_class": "fixed_income"},
    {"symbol": "AGG", "asset_class": "fixed_income"},
    # Foreign Exchange (~9)
    {"symbol": "FXE", "asset_class": "fx"},
    {"symbol": "FXB", "asset_class": "fx"},
    {"symbol": "FXC", "asset_class": "fx"},
    {"symbol": "FXA", "asset_class": "fx"},
    {"symbol": "FXY", "asset_class": "fx"},
    {"symbol": "FXF", "asset_class": "fx"},
    {"symbol": "UUP", "asset_class": "fx"},
    {"symbol": "CYB", "asset_class": "fx"},
    {"symbol": "CEW", "asset_class": "fx"},
]
