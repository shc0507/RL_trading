"""Backward-compatible aliases for the older `policies` module name."""

from .strategies import BaseStrategy as BasePolicy
from .strategies import LongOnlyStrategy as LongOnlyPolicy
from .strategies import MACDStrategy as MACDPolicy
from .strategies import RSIMeanReversionStrategy as RSIMeanReversionPolicy
from .strategies import Sign12MStrategy as Sign12MPolicy

__all__ = [
    "BasePolicy",
    "LongOnlyPolicy",
    "MACDPolicy",
    "RSIMeanReversionPolicy",
    "Sign12MPolicy",
]
