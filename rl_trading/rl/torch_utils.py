"""Local PyTorch helpers for RL modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def activation_from_name(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name}")


def build_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: Sequence[int],
    activation: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_size, int(hidden_size)))
        layers.append(activation_from_name(activation))
        current_size = int(hidden_size)
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)


def as_float_tensor(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def as_long_tensor(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.int64, device=device)
