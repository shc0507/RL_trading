"""Strategy-first quant trading research and execution package."""

from .backtest import Backtester, EvalReport
from .config import DEFAULT_CRYPTO_SYMBOLS, DEFAULT_SPLITS, DEFAULT_SYMBOLS
from .env import EnvironmentConfig, TradingEnv
from .features import FeatureBuilder
from .live import LiveTrader, LiveTradingConfig
from .pairs import (
    CointegrationResult,
    PairsBacktestReport,
    PairsConfig,
    PairsResearchResult,
    PairsTradingResearcher,
    ThresholdSelectionResult,
)
from .portfolio import AllocationDecision, PortfolioConstraints, TargetAllocator
from .strategies import BaseStrategy, LongOnlyStrategy, MACDStrategy, RSIMeanReversionStrategy, Sign12MStrategy

__all__ = [
    "AllocationDecision",
    "Backtester",
    "BaseStrategy",
    "CointegrationResult",
    "DEFAULT_CRYPTO_SYMBOLS",
    "DEFAULT_SPLITS",
    "DEFAULT_SYMBOLS",
    "EnvironmentConfig",
    "EvalReport",
    "FeatureBuilder",
    "LiveTrader",
    "LiveTradingConfig",
    "LongOnlyStrategy",
    "MACDStrategy",
    "PairsBacktestReport",
    "PairsConfig",
    "PairsResearchResult",
    "PairsTradingResearcher",
    "PortfolioConstraints",
    "RSIMeanReversionStrategy",
    "Sign12MStrategy",
    "ThresholdSelectionResult",
    "TargetAllocator",
    "TradingEnv",
]
