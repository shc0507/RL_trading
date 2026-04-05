"""Fixed-income Zhang-style rolling experiment."""

from __future__ import annotations

from .zhang_asset_class_experiment import run_cli


if __name__ == "__main__":
    run_cli(default_asset_class="fixed_income")
