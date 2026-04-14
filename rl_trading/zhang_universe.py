"""Canonical Zhang et al. production universe metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ZhangInstrument:
    symbol: str
    asset_class: str
    currency: str
    description: str


ZHANG_UNIVERSE: tuple[ZhangInstrument, ...] = (
    ZhangInstrument("CC", "commodity", "USD", "COCOA"),
    ZhangInstrument("DA", "commodity", "USD", ".MILK III, Comp"),
    ZhangInstrument("GI", "commodity", "USD", "GOLDMAN SAKS C. I."),
    ZhangInstrument("JO", "commodity", "USD", "ORANGE JUICE"),
    ZhangInstrument("KC", "commodity", "USD", "COFFEE"),
    ZhangInstrument("KW", "commodity", "USD", "WHEAT, KC"),
    ZhangInstrument("LB", "commodity", "USD", "LUMBER"),
    ZhangInstrument("NR", "commodity", "USD", "ROUGH RICE"),
    ZhangInstrument("SB", "commodity", "USD", "SUGAR #11"),
    ZhangInstrument("ZA", "commodity", "USD", "PALLADIUM, Electronic"),
    ZhangInstrument("ZC", "commodity", "USD", "CORN, Electronic"),
    ZhangInstrument("ZF", "commodity", "USD", "FEEDER CATTLE, Electronic"),
    ZhangInstrument("ZG", "commodity", "USD", "GOLD, Electronic"),
    ZhangInstrument("ZH", "commodity", "USD", "HEATING OIL, Electronic"),
    ZhangInstrument("ZI", "commodity", "USD", "SILVER, Electronic"),
    ZhangInstrument("ZK", "commodity", "USD", "COPPER, Electronic"),
    ZhangInstrument("ZL", "commodity", "USD", "SOYBEAN OIL, Electronic"),
    ZhangInstrument("ZN", "commodity", "USD", "NATURAL GAS, Electronic"),
    ZhangInstrument("ZO", "commodity", "USD", "OATS, Electronic"),
    ZhangInstrument("ZP", "commodity", "USD", "PLATINUM, electronic"),
    ZhangInstrument("ZR", "commodity", "USD", "ROUGH RICE, Electronic"),
    ZhangInstrument("ZT", "commodity", "USD", "LIVE CATTLE, Electronic"),
    ZhangInstrument("ZU", "commodity", "USD", "CRUDE OIL, Electronic"),
    ZhangInstrument("ZW", "commodity", "USD", "WHEAT, Electronic"),
    ZhangInstrument("ZZ", "commodity", "USD", "LEAN HOGS, Electronic"),
    ZhangInstrument("CA", "equity_index", "EUR", "CAC40 INDEX"),
    ZhangInstrument("EN", "equity_index", "USD", "NASDAQ, MINI"),
    ZhangInstrument("ER", "equity_index", "USD", "RUSSELL 2000, MINI"),
    ZhangInstrument("ES", "equity_index", "USD", "S & P 500, MINI"),
    ZhangInstrument("LX", "equity_index", "GBP", "FTSE 100 INDEX"),
    ZhangInstrument("MD", "equity_index", "USD", "S&P 400 (Mini Electronic)"),
    ZhangInstrument("SC", "equity_index", "USD", "S & P 500, Composite"),
    ZhangInstrument("SP", "equity_index", "USD", "S & P 500, Day Session"),
    ZhangInstrument("XU", "equity_index", "EUR", "DOW JONES EUROSTOXX50"),
    ZhangInstrument("XX", "equity_index", "EUR", "DOW JONES STOXX 50"),
    ZhangInstrument("YM", "equity_index", "USD", "Mini Dow Jones ($5.00)"),
    ZhangInstrument("DT", "fixed_income", "EUR", "EURO BOND (BUND)"),
    ZhangInstrument("FB", "fixed_income", "USD", "T-NOTE, 5-year Composite"),
    ZhangInstrument("TY", "fixed_income", "USD", "T-NOTE, 10-year Composite"),
    ZhangInstrument("UB", "fixed_income", "EUR", "EURO BOBL"),
    ZhangInstrument("US", "fixed_income", "USD", "T-BONDS, Composite"),
    ZhangInstrument("AN", "fx", "AUD", "AUSTRALIAN, Day Session"),
    ZhangInstrument("BN", "fx", "GBP", "BRITISH POUND, Composite"),
    ZhangInstrument("CN", "fx", "CAD", "CANADIAN, Composite"),
    ZhangInstrument("DX", "fx", "USD", "US DOLLAR INDEX"),
    ZhangInstrument("FN", "fx", "EUR", "EURO, Composite"),
    ZhangInstrument("JN", "fx", "JPY", "JAPANESE YEN, Composite"),
    ZhangInstrument("MP", "fx", "MXN", "MEXICAN PESO"),
    ZhangInstrument("NK", "fx", "JPY", "NIKKEI INDEX"),
    ZhangInstrument("SN", "fx", "CHF", "SWISS FRANC, Composite"),
)

ZHANG_SYMBOLS: tuple[str, ...] = tuple(instrument.symbol for instrument in ZHANG_UNIVERSE)


def zhang_universe_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(instrument) for instrument in ZHANG_UNIVERSE])


def load_universe_frame(universe_manifest_path: str | Path | None = None) -> pd.DataFrame:
    if universe_manifest_path is None:
        return zhang_universe_frame()

    path = Path(universe_manifest_path)
    frame = pd.read_csv(path)
    required = {"symbol", "asset_class", "currency"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"universe manifest is missing required columns: {missing}")
    if "description" not in frame.columns:
        frame["description"] = frame["symbol"]
    frame["symbol"] = frame["symbol"].astype(str)
    frame["asset_class"] = frame["asset_class"].astype(str)
    frame["currency"] = frame["currency"].astype(str)
    frame["description"] = frame["description"].astype(str)
    return frame.loc[:, ["symbol", "asset_class", "currency", "description"]].copy()


def validate_universe_frame(universe: pd.DataFrame) -> None:
    if universe.empty:
        raise ValueError("universe manifest is empty")
    if universe["symbol"].duplicated().any():
        duplicates = universe.loc[universe["symbol"].duplicated(), "symbol"].tolist()
        raise ValueError(f"universe manifest contains duplicate symbols: {duplicates[:5]}")

    expected_asset_classes = {"commodity", "equity_index", "fixed_income", "fx"}
    invalid = sorted(set(universe["asset_class"].astype(str)) - expected_asset_classes)
    if invalid:
        raise ValueError(f"universe manifest contains unsupported asset classes: {invalid}")
