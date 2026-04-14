"""Full experiment runner for Zhang et al. (2019) recreation."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from rl_trading.agents import A2CAgent, DQNAgent, PGAgent
from rl_trading.baselines import (
    compute_baseline_rewards,
    long_only,
    macd_signal,
    sign_r,
)
from rl_trading.config import DEFAULT_VOL_TARGET, UNIVERSE
from rl_trading.data import fetch_bars
from rl_trading.env import EnvConfig, TradingEnv
from rl_trading.features import FEATURE_COLS, FeatureBuilder
from rl_trading.metrics import compute_metrics
from rl_trading.trainer import Trainer, TrainerConfig


# ── Helpers ─────────────────────────────────────────────────────────


def _detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_agent_factories(device: str) -> dict:
    return {
        "dqn": lambda: (
            DQNAgent(n_features=len(FEATURE_COLS), device=device),
            EnvConfig(action_mode="discrete", seq_len=60),
        ),
        "pg": lambda: (
            PGAgent(n_features=len(FEATURE_COLS), device=device),
            EnvConfig(action_mode="discrete", seq_len=60),
        ),
        "a2c": lambda: (
            A2CAgent(n_features=len(FEATURE_COLS), device=device),
            EnvConfig(action_mode="continuous", seq_len=60),
        ),
    }

_BASELINE_FNS = {
    "Long": long_only,
    "Sign(R)": sign_r,
    "MACD": macd_signal,
}


def _collect_rl_rewards(
    agent: DQNAgent | PGAgent | A2CAgent,
    feature_frame: pd.DataFrame,
    symbol: str,
    split: str,
    action_mode: str,
    seq_len: int,
) -> pd.Series:
    """Run a trained agent on the given split and return date-indexed reward Series."""
    cfg = EnvConfig(action_mode=action_mode, seq_len=seq_len)
    env = TradingEnv(feature_frame, cfg)
    state = env.reset(symbol, split)
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        if next_state is not None:
            state = next_state
    dates = pd.to_datetime(env.history["date"])
    rewards = np.array(env.history["reward"], dtype=np.float64)
    return pd.Series(rewards, index=dates, name=symbol)


def _portfolio_vol_scale(
    port_returns: np.ndarray,
    vol_target: float = DEFAULT_VOL_TARGET,
) -> np.ndarray:
    """Apply portfolio-level vol targeting to an already equal-weighted return series."""
    if len(port_returns) < 60:
        return port_returns
    # Realized vol using expanding window (min 60 days), annualized.
    # Shift by 1 day so today's return is not used to size today's exposure.
    cum_std = pd.Series(port_returns).expanding(min_periods=60).std() * math.sqrt(252)
    cum_std = cum_std.shift(1).to_numpy().copy()
    cum_std[np.isnan(cum_std) | (cum_std < 1e-8)] = 1e-8
    scale = vol_target / cum_std
    # Don't scale first 60 days (not enough data after shift)
    scale[:60] = 1.0
    return port_returns * scale


# ── Main experiment ─────────────────────────────────────────────────

def run_experiment(
    symbols: list[str] | None = None,
    agents: list[str] | None = None,
    n_epochs: int = 200,
    patience: int = 20,
    output_dir: str = "artifacts",
    device: str | None = None,
):
    """Run the full Zhang et al. experiment.

    Parameters
    ----------
    symbols : list of ticker symbols (defaults to full UNIVERSE).
    agents : list of agent names, subset of ["dqn", "pg", "a2c"].
    n_epochs : training epochs per agent per symbol.
    patience : early stopping patience.
    output_dir : directory for output CSV and PNG.
    """
    if agents is None:
        agents = ["dqn", "pg", "a2c"]

    if device is None:
        device = _detect_device()
    print(f"Using device: {device}")

    agent_factories = _make_agent_factories(device)

    # Resolve symbols and asset classes
    universe = UNIVERSE
    if symbols is not None:
        sym_set = set(symbols)
        universe = [u for u in UNIVERSE if u["symbol"] in sym_set]
        # Add symbols not in UNIVERSE with unknown class
        known = {u["symbol"] for u in universe}
        for s in symbols:
            if s not in known:
                universe.append({"symbol": s, "asset_class": "other"})

    all_symbols = [u["symbol"] for u in universe]
    sym_to_class = {u["symbol"]: u["asset_class"] for u in universe}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data and build features
    print("Fetching data ...")
    bars = fetch_bars(all_symbols, start="2004-01-01", end="2025-12-31")
    print("Building features ...")
    fb = FeatureBuilder()
    feat = fb.transform(bars)

    # 2. Methods = baselines + RL agents
    method_names = list(_BASELINE_FNS.keys()) + [a.upper() for a in agents]

    # symbol -> method -> daily_rewards as date-indexed Series
    results: dict[str, dict[str, pd.Series]] = {}

    for i, sym in enumerate(all_symbols, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(all_symbols)}] {sym} ({sym_to_class[sym]})")
        print(f"{'='*60}")

        results[sym] = {}

        # Baselines
        for name, fn in _BASELINE_FNS.items():
            try:
                positions = fn(feat, sym, "test")
                rewards = compute_baseline_rewards(positions, feat, sym, "test")
                results[sym][name] = rewards
                print(f"  {name:10s}: {len(rewards)} days, mean={rewards.mean():.6f}")
            except Exception as e:
                print(f"  {name:10s}: FAILED ({e})")
                results[sym][name] = pd.Series(dtype=np.float64)

        # RL agents
        for agent_name in agents:
            label = agent_name.upper()
            try:
                agent, env_cfg = agent_factories[agent_name]()
                env = TradingEnv(feat, env_cfg)
                tcfg = TrainerConfig(
                    n_epochs=n_epochs,
                    patience=patience,
                    checkpoint_dir=str(out_dir / "checkpoints" / sym),
                )
                trainer = Trainer(agent, env, sym, tcfg)
                train_result = trainer.train()
                print(f"  {label} train: {train_result}")

                # Load best checkpoint and evaluate on test
                ckpt = Path(tcfg.checkpoint_dir) / f"{sym}_best.pt"
                if ckpt.exists():
                    agent.load(ckpt)

                rewards = _collect_rl_rewards(
                    agent, feat, sym, "test",
                    action_mode=env_cfg.action_mode,
                    seq_len=env_cfg.seq_len,
                )
                results[sym][label] = rewards
                print(f"  {label:10s}: {len(rewards)} days, mean={rewards.mean():.6f}")
            except Exception as e:
                print(f"  {label:10s}: FAILED ({e})")
                results[sym][label] = pd.Series(dtype=np.float64)

    # 3. Aggregate portfolios
    print(f"\n{'='*60}")
    print("Aggregating portfolios ...")
    print(f"{'='*60}")

    # Group symbols by asset class
    class_symbols: dict[str, list[str]] = defaultdict(list)
    for sym in all_symbols:
        class_symbols[sym_to_class[sym]].append(sym)

    # Build groupings: per asset class + "All"
    groupings: dict[str, list[str]] = dict(class_symbols)
    groupings["All"] = all_symbols

    # grouping -> method -> portfolio daily rewards
    portfolio_rewards: dict[str, dict[str, np.ndarray]] = {}

    for grp_name, grp_syms in groupings.items():
        portfolio_rewards[grp_name] = {}
        for method in method_names:
            # Collect date-indexed reward Series and join on date
            series = []
            for sym in grp_syms:
                r = results.get(sym, {}).get(method, pd.Series(dtype=np.float64))
                if len(r) > 0:
                    series.append(r.rename(sym))
            if not series:
                portfolio_rewards[grp_name][method] = np.array([])
                continue
            # Inner join: only keep dates present in all symbols
            combined = pd.concat(series, axis=1, join="inner").sort_index()
            # Equal-weight average across symbols
            port_ret = combined.mean(axis=1).to_numpy()
            # Portfolio-level vol targeting
            port_ret = _portfolio_vol_scale(port_ret)
            portfolio_rewards[grp_name][method] = port_ret

    # 4. Compute metrics table
    rows = []
    for grp_name in groupings:
        for method in method_names:
            r = portfolio_rewards[grp_name].get(method, np.array([]))
            m = compute_metrics(r)
            m["Group"] = grp_name
            m["Method"] = method
            rows.append(m)

    metrics_df = pd.DataFrame(rows)
    # Reorder columns
    col_order = ["Group", "Method", "E(R)", "Std(R)", "DD", "Sharpe",
                 "Sortino", "MDD", "Calmar", "%+Ret", "AvgP/AvgL"]
    metrics_df = metrics_df[col_order]

    # 5. Print results table
    print(f"\n{'='*60}")
    print("Results (Zhang et al. Exhibit 2 style)")
    print(f"{'='*60}")
    _print_table(metrics_df)

    # Save CSV
    csv_path = out_dir / "results.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to {csv_path}")

    # 6. Plot cumulative trade returns
    _plot_cumulative(portfolio_rewards, groupings, method_names, out_dir)

    return metrics_df


def _print_table(df: pd.DataFrame) -> None:
    """Print aligned metrics table grouped by asset class."""
    float_cols = [c for c in df.columns if c not in ("Group", "Method")]
    for grp, grp_df in df.groupby("Group", sort=False):
        print(f"\n  {grp}")
        print(f"  {'Method':<10s}", end="")
        for c in float_cols:
            print(f" {c:>10s}", end="")
        print()
        print("  " + "-" * (10 + 11 * len(float_cols)))
        for _, row in grp_df.iterrows():
            print(f"  {row['Method']:<10s}", end="")
            for c in float_cols:
                print(f" {row[c]:10.4f}", end="")
            print()


def _plot_cumulative(
    portfolio_rewards: dict[str, dict[str, np.ndarray]],
    groupings: dict[str, list[str]],
    method_names: list[str],
    out_dir: Path,
) -> None:
    """Plot cumulative trade returns (Zhang Exhibit 3 style)."""
    grp_names = list(groupings.keys())
    n = len(grp_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, grp_name in zip(axes, grp_names):
        for method in method_names:
            r = portfolio_rewards[grp_name].get(method, np.array([]))
            if len(r) == 0:
                continue
            cum = np.cumsum(r)
            ax.plot(cum, label=method, linewidth=1.0)
        ax.set_title(grp_name)
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Trading Day")
    fig.tight_layout()
    png_path = out_dir / "cumulative_returns.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {png_path}")
