"""Project-wide defaults following Zhang et al. (2019)."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

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

# ── Asset universe (strict subset of original ETF proxies) ───────────
UNIVERSE: list[dict[str, str]] = [
    # Commodities
    {"symbol": "GLD", "asset_class": "commodity"},
    {"symbol": "SLV", "asset_class": "commodity"},
    {"symbol": "USO", "asset_class": "commodity"},
    {"symbol": "UNG", "asset_class": "commodity"},
    {"symbol": "DBA", "asset_class": "commodity"},
    {"symbol": "DBC", "asset_class": "commodity"},
    {"symbol": "UGA", "asset_class": "commodity"},
    {"symbol": "GSG", "asset_class": "commodity"},
    # Equity Indexes
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
]


def walk_forward_splits(
    data_start: str = TRAIN_START,
    data_end: str = TEST_END,
    min_train_years: int = 5,
    val_years: int = 3,
    test_years: int = 3,
    val_frac: float = 0.10,
) -> list[dict[str, tuple[str, str]]]:
    """Generate expanding-window walk-forward folds (Zhang et al. 2019 style).

    For each fold, the "train+val window" spans data_start to the calendar
    year before test_start_year (same fold cadence as before). The last
    ``val_frac`` portion (by calendar months) of that window becomes the
    validation block; the earlier 1 - ``val_frac`` portion is training.
    This keeps val adjacent to (and upstream of) test without a separate
    multi-year block, matching the paper's "10% of training data as a
    separate cross-validation set" prescription.

    Parameters
    ----------
    val_years:
        Deprecated. Retained for backwards compatibility with existing
        callers; a non-default value is ignored with a warning. Fold
        cadence is still driven by the old ``val_years`` default so the
        5-fold layout is unchanged.
    val_frac:
        Fraction of the combined train+val window (in months) assigned
        to validation. Default 0.10 per the paper.
    """
    if val_years != 3:
        warnings.warn(
            "walk_forward_splits: `val_years` is deprecated and ignored; "
            "validation is now the last `val_frac` of the train+val window.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")

    start_year = int(data_start[:4])
    end_year = int(data_end[:4])

    # Keep prior fold cadence: test_start_year advances every test_years,
    # first test year = data_start + min_train_years + 3 (old val_years=3).
    first_test_year = start_year + min_train_years + 3

    folds: list[dict[str, tuple[str, str]]] = []
    test_start_year = first_test_year

    while test_start_year <= end_year:
        test_end_year = min(test_start_year + test_years - 1, end_year)
        # Combined train+val window ends the year before test starts.
        combined_end_year = test_start_year - 1
        combined_start = date.fromisoformat(data_start)
        combined_end = date(combined_end_year, 12, 31)

        # Split by calendar months: last ceil(val_frac * total_months)
        # months are validation, with at least one month of val.
        total_months = (
            (combined_end.year - combined_start.year) * 12
            + (combined_end.month - combined_start.month)
            + 1
        )
        val_months = max(1, int(round(val_frac * total_months)))
        train_months = total_months - val_months

        # train_end = last day of the month `train_months` months after start.
        train_end_month_index = combined_start.month - 1 + train_months - 1
        train_end_year = combined_start.year + train_end_month_index // 12
        train_end_month = train_end_month_index % 12 + 1
        # Last day of that month.
        if train_end_month == 12:
            train_end_day = date(train_end_year, 12, 31)
        else:
            train_end_day = date(train_end_year, train_end_month + 1, 1) - timedelta(days=1)

        # val_start = first day of the following month; ensures >=1-day gap
        # from train_end (next calendar day, and always a fresh month).
        val_start_month_index = train_end_month_index + 1
        val_start_year = combined_start.year + val_start_month_index // 12
        val_start_month = val_start_month_index % 12 + 1
        val_start_day = date(val_start_year, val_start_month, 1)
        val_end_day = combined_end

        folds.append({
            "train": (data_start, train_end_day.isoformat()),
            "val": (val_start_day.isoformat(), val_end_day.isoformat()),
            "test": (f"{test_start_year}-01-01", f"{test_end_year}-12-31"),
        })
        test_start_year += test_years

    return folds
