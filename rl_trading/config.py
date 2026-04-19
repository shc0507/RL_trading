"""Project-wide defaults following Zhang et al. (2019)."""

from __future__ import annotations

import os
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
# Env-level vol target used inside the reward (Eq. 4): scales per-bar
# rewards to ~15% annualized vol per contract. Kept at 0.15 because
# PG/A2C diverge when training on dollar-unit rewards amplified by
# σ_tgt ≫ daily_sigma (observed: NaN in softmax / Normal(loc=NaN) with
# σ_tgt=1.0 on commodity contracts).
DEFAULT_VOL_TARGET: float = 0.15
# Portfolio-level vol target applied at REPORTING time only (Zhang
# Exhibit 2 appears to target ≈1.0 so Std(R) ≈ 0.97 across methods).
# Scale-invariant metrics (Sharpe, Sortino, Calmar) are unaffected.
PORTFOLIO_VOL_TARGET: float = 1.0
# Paper Exhibit 1: bp = 0.0020 (= 20 basis points; paper defines 1 bp = 0.0001).
DEFAULT_COST_RATE_BP: float = 20.0  # basis points

# ── Train / val / test splits ────────────────────────────────────────
# Paper test horizon ends 2019-12-31; anything beyond is post-paper extension.
TRAIN_START = "2005-01-01"
TRAIN_END = "2015-12-31"
VAL_START = "2016-01-01"
VAL_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2019-12-31"

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

# ── CLC / Pinnacle futures universe (Zhang 2019 Appendix A) ──────────
# 49 distinct continuous futures; paper header says 50 but table lists 49.
# Bucketing follows Zhang exactly (NK stays under FX even though it's an
# equity index contract — that's how Appendix A groups it).
CLC_UNIVERSE: list[dict[str, str]] = [
    # Commodities (25)
    {"symbol": "CC", "asset_class": "commodity"},
    {"symbol": "DA", "asset_class": "commodity"},
    {"symbol": "GI", "asset_class": "commodity"},
    {"symbol": "JO", "asset_class": "commodity"},
    {"symbol": "KC", "asset_class": "commodity"},
    {"symbol": "KW", "asset_class": "commodity"},
    {"symbol": "LB", "asset_class": "commodity"},
    {"symbol": "NR", "asset_class": "commodity"},
    {"symbol": "SB", "asset_class": "commodity"},
    {"symbol": "ZA", "asset_class": "commodity"},
    {"symbol": "ZC", "asset_class": "commodity"},
    {"symbol": "ZF", "asset_class": "commodity"},
    {"symbol": "ZG", "asset_class": "commodity"},
    {"symbol": "ZH", "asset_class": "commodity"},
    {"symbol": "ZI", "asset_class": "commodity"},
    {"symbol": "ZK", "asset_class": "commodity"},
    {"symbol": "ZL", "asset_class": "commodity"},
    {"symbol": "ZN", "asset_class": "commodity"},
    {"symbol": "ZO", "asset_class": "commodity"},
    {"symbol": "ZP", "asset_class": "commodity"},
    {"symbol": "ZR", "asset_class": "commodity"},
    {"symbol": "ZT", "asset_class": "commodity"},
    {"symbol": "ZU", "asset_class": "commodity"},
    {"symbol": "ZW", "asset_class": "commodity"},
    {"symbol": "ZZ", "asset_class": "commodity"},
    # Equity Indexes (10)
    {"symbol": "CA", "asset_class": "equity_index"},
    {"symbol": "ER", "asset_class": "equity_index"},
    {"symbol": "ES", "asset_class": "equity_index"},
    {"symbol": "LX", "asset_class": "equity_index"},
    {"symbol": "MD", "asset_class": "equity_index"},
    {"symbol": "SC", "asset_class": "equity_index"},
    {"symbol": "SP", "asset_class": "equity_index"},
    {"symbol": "XU", "asset_class": "equity_index"},
    {"symbol": "XX", "asset_class": "equity_index"},
    {"symbol": "YM", "asset_class": "equity_index"},
    # Fixed Income (5)
    {"symbol": "DT", "asset_class": "fixed_income"},
    {"symbol": "FB", "asset_class": "fixed_income"},
    {"symbol": "TY", "asset_class": "fixed_income"},
    {"symbol": "UB", "asset_class": "fixed_income"},
    {"symbol": "US", "asset_class": "fixed_income"},
    # FX (9)
    {"symbol": "AN", "asset_class": "fx"},
    {"symbol": "BN", "asset_class": "fx"},
    {"symbol": "CN", "asset_class": "fx"},
    {"symbol": "DX", "asset_class": "fx"},
    {"symbol": "FN", "asset_class": "fx"},
    {"symbol": "JN", "asset_class": "fx"},
    {"symbol": "MP", "asset_class": "fx"},
    {"symbol": "NK", "asset_class": "fx"},
    {"symbol": "SN", "asset_class": "fx"},
]

# ── Data-source selector ─────────────────────────────────────────────
# Set RL_DATA_SOURCE=etf to use the legacy yfinance ETF proxies; default
# is 'clc' (Pinnacle RAD futures, matches the paper).
DATA_SOURCE: str = os.environ.get("RL_DATA_SOURCE", "clc").lower()
if DATA_SOURCE not in ("etf", "clc"):
    raise ValueError(
        f"RL_DATA_SOURCE must be 'etf' or 'clc'; got {DATA_SOURCE!r}"
    )

# Pipeline callers should import ACTIVE_UNIVERSE; it tracks DATA_SOURCE.
ACTIVE_UNIVERSE: list[dict[str, str]] = (
    CLC_UNIVERSE if DATA_SOURCE == "clc" else UNIVERSE
)


def walk_forward_splits(
    data_start: str = TRAIN_START,
    data_end: str = TEST_END,
    min_train_years: int = 6,
    val_years: int = 3,
    test_years: int = 5,
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

    # Paper p.6: "We retrain our model at every five years, using all data
    # available up to that point ... testing period is from 2011 to 2019."
    # With data_start=2005 and min_train_years=6 we get first test year
    # 2011, and test_years=5 cadence gives exactly two folds:
    #   train 2005-01→2010-12 (10% val at tail), test 2011-01→2015-12
    #   train 2005-01→2015-12 (10% val at tail), test 2016-01→2019-12 (clamped)
    first_test_year = start_year + min_train_years

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
