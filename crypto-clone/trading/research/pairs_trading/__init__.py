"""Pairs trading research package."""

from .pairs import (
    CointegrationResult,
    PairsBacktestReport,
    PairsConfig,
    PairsResearchResult,
    PairsTradingResearcher,
    ThresholdSelectionResult,
    main,
)

__all__ = [
    "CointegrationResult",
    "PairsBacktestReport",
    "PairsConfig",
    "PairsResearchResult",
    "PairsTradingResearcher",
    "ThresholdSelectionResult",
    "main",
]
