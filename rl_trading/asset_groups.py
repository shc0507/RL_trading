"""Helpers for canonical Zhang-style asset-class group names."""

from __future__ import annotations


def canonical_asset_group(asset_class: str | None) -> str:
    value = (asset_class or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in value for token in ("equity_index", "equity_etf", "index", "equity")):
        return "equity_index"
    if any(token in value for token in ("fixed_income", "bond", "rates", "treasury")):
        return "fixed_income"
    if any(token in value for token in ("fx", "forex", "currency")):
        return "fx"
    if any(token in value for token in ("commodity", "energy", "metal", "agri", "grain", "livestock")):
        return "commodity"
    return value or "unknown"


def display_asset_group(asset_group: str) -> str:
    labels = {
        "all": "All",
        "commodity": "Commodity",
        "equity_index": "Equity Index",
        "fixed_income": "Fixed Income",
        "fx": "FX",
        "unknown": "Unknown",
    }
    return labels.get(asset_group, asset_group.replace("_", " ").title())
