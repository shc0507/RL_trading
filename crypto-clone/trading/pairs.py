"""Backward-compatible export for the pairs trading research module."""

from .research.pairs_trading.pairs import (
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


if __name__ == "__main__":
    main()
