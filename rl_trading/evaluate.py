"""Full experiment runner for Zhang et al. (2019) recreation."""

from __future__ import annotations

import math
import traceback
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
from rl_trading.config import (
    DEFAULT_VOL_TARGET,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    UNIVERSE,
    VAL_END,
    VAL_START,
    walk_forward_splits,
)
from rl_trading.data import fetch_bars
from rl_trading.env import EnvConfig, STATE_DIM, TradingEnv
from rl_trading.features import FeatureBuilder
from rl_trading.metrics import compute_metrics
from rl_trading.trainer import Trainer, TrainerConfig
from rl_trading.wandb_logger import WandbLogger


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
            DQNAgent(n_features=STATE_DIM, device=device),
            EnvConfig(action_mode="discrete", seq_len=60),
        ),
        "pg": lambda: (
            PGAgent(n_features=STATE_DIM, device=device),
            EnvConfig(action_mode="discrete", seq_len=60),
        ),
        "a2c": lambda: (
            A2CAgent(n_features=STATE_DIM, device=device),
            EnvConfig(action_mode="continuous", seq_len=60),
        ),
    }

_BASELINE_FNS = {
    "Long": long_only,
    "Sign(R)": sign_r,
    "MACD": macd_signal,
}


def _extract_hparams(agent, env_cfg) -> dict:
    """Introspect an agent instance for its hyperparameters."""
    h: dict = {}
    for key in (
        "gamma", "batch_size", "target_update_freq",
        "eps_start", "eps_end", "eps_decay_steps",
    ):
        if hasattr(agent, key):
            h[key] = getattr(agent, key)
    if hasattr(agent, "memory") and hasattr(agent.memory, "maxlen"):
        h["memory_size"] = agent.memory.maxlen
    if hasattr(agent, "optimizer"):
        lrs = [pg.get("lr") for pg in agent.optimizer.param_groups]
        h["lr"] = lrs[0] if len(set(lrs)) == 1 else lrs
    h["env"] = {
        "action_mode": env_cfg.action_mode,
        "cost_rate_bp": env_cfg.cost_rate_bp,
        "vol_target": env_cfg.vol_target,
        "seq_len": env_cfg.seq_len,
    }
    return h


def _collect_rl_rewards(
    agent: DQNAgent | PGAgent | A2CAgent,
    feature_frame: pd.DataFrame,
    symbol: str,
    split: str = "test",
    *,
    action_mode: str,
    seq_len: int,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Run a trained agent on the given split and return date-indexed reward Series."""
    cfg = EnvConfig(action_mode=action_mode, seq_len=seq_len)
    env = TradingEnv(feature_frame, cfg)
    if start is not None and end is not None:
        state = env.reset(symbol, start=start, end=end)
    else:
        state = env.reset(symbol, split)
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, _, done, _ = env.step(action)
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
    cum_std = pd.Series(port_returns).expanding(min_periods=60).std() * math.sqrt(252)
    cum_std = cum_std.shift(1).to_numpy().copy()
    cum_std[np.isnan(cum_std) | (cum_std < 1e-8)] = 1e-8
    scale = vol_target / cum_std
    scale[:60] = 1.0
    return port_returns * scale


def _append_rewards(
    results: dict[str, dict[str, pd.Series]],
    sym: str, method: str, rewards: pd.Series,
) -> None:
    """Append rewards to results, concatenating across folds."""
    if method in results[sym] and len(results[sym][method]) > 0:
        results[sym][method] = pd.concat([results[sym][method], rewards])
    else:
        results[sym][method] = rewards


# ── Main experiment ─────────────────────────────────────────────────

def run_experiment(
    symbols: list[str] | None = None,
    agents: list[str] | None = None,
    n_epochs: int = 200,
    patience: int = 20,
    output_dir: str = "artifacts",
    device: str | None = None,
    walk_forward: bool = False,
    wandb_enabled: bool = False,
    wandb_project: str = "rl-trading",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
):
    """Run the full Zhang et al. experiment.

    Parameters
    ----------
    symbols : list of ticker symbols (defaults to full UNIVERSE).
    agents : list of agent names, subset of ["dqn", "pg", "a2c"].
    n_epochs : training epochs per agent per asset class.
    patience : early stopping patience.
    output_dir : directory for output CSV and PNG.
    walk_forward : if True, use expanding-window walk-forward folds
        instead of a single train/val/test split.
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
        known_syms = {u["symbol"] for u in UNIVERSE}
        unknown = [s for s in symbols if s not in known_syms]
        if unknown:
            raise ValueError(f"Unknown symbols not in UNIVERSE: {unknown}")
        sym_set = set(symbols)
        universe = [u for u in UNIVERSE if u["symbol"] in sym_set]

    all_symbols = [u["symbol"] for u in universe]
    sym_to_class = {u["symbol"]: u["asset_class"] for u in universe}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group symbols by asset class
    class_symbols: dict[str, list[str]] = defaultdict(list)
    for sym in all_symbols:
        class_symbols[sym_to_class[sym]].append(sym)

    # 1. Fetch data and build features
    print("Fetching data ...")
    fetch_start = str(int(TRAIN_START[:4]) - 2) + TRAIN_START[4:]
    bars = fetch_bars(all_symbols, start=fetch_start, end=TEST_END)
    print("Building features ...")
    fb = FeatureBuilder()
    feat = fb.transform(bars)

    # 2. Determine folds
    if walk_forward:
        folds = walk_forward_splits()
    else:
        folds = [{
            "train": (TRAIN_START, TRAIN_END),
            "val": (VAL_START, VAL_END),
            "test": (TEST_START, TEST_END),
        }]

    # Initialize wandb (no-op if disabled)
    agent_hparams: dict[str, dict] = {}
    for agent_name in agents:
        a, env_cfg = agent_factories[agent_name]()
        agent_hparams[agent_name] = _extract_hparams(a, env_cfg)
    wandb_config = {
        "agents": agents,
        "n_epochs": n_epochs,
        "patience": patience,
        "device": device,
        "walk_forward": walk_forward,
        "symbols": all_symbols,
        "asset_classes": dict(class_symbols),
        "folds": [
            {k: list(v) for k, v in fold.items()} for fold in folds
        ],
        "default_vol_target": DEFAULT_VOL_TARGET,
        "hparams": agent_hparams,
    }
    logger = WandbLogger.init(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        tags=wandb_tags,
        config=wandb_config,
    )

    method_names = list(_BASELINE_FNS.keys()) + [a.upper() for a in agents]
    results: dict[str, dict[str, pd.Series]] = {sym: {} for sym in all_symbols}
    failures: list[tuple[str, str, str]] = []

    # 3. Run each fold
    for fold_idx, fold in enumerate(folds, 1):
        train_dates = fold["train"]
        val_dates = fold["val"]
        test_dates = fold["test"]

        if len(folds) > 1:
            print(f"\n{'#'*60}")
            print(f"  Fold {fold_idx}/{len(folds)}: "
                  f"train {train_dates[0]}–{train_dates[1]}, "
                  f"test {test_dates[0]}–{test_dates[1]}")
            print(f"{'#'*60}")

        # ── Baselines (per-symbol, non-learned) ──
        print(f"\n  Baselines ...")
        for sym in all_symbols:
            for name, fn in _BASELINE_FNS.items():
                try:
                    positions = fn(feat, sym, start=test_dates[0], end=test_dates[1])
                    rewards = compute_baseline_rewards(
                        positions, feat, sym,
                        start=test_dates[0], end=test_dates[1],
                    )
                    _append_rewards(results, sym, name, rewards)
                except Exception as e:
                    failures.append((sym, name, traceback.format_exc()))
        baseline_syms = sum(1 for s in all_symbols
                           if "Long" in results[s] and len(results[s]["Long"]) > 0)
        print(f"  Baselines: {baseline_syms}/{len(all_symbols)} symbols OK")

        # ── RL agents (one model per asset class) ──
        for agent_name in agents:
            label = agent_name.upper()
            for cls, class_syms in class_symbols.items():
                print(f"\n  {label} / {cls} ({len(class_syms)} symbols)")
                try:
                    agent, env_cfg = agent_factories[agent_name]()
                    env = TradingEnv(feat, env_cfg)
                    fold_label = f"{cls}_fold{fold_idx}" if len(folds) > 1 else cls
                    tcfg = TrainerConfig(
                        n_epochs=n_epochs,
                        patience=patience,
                        checkpoint_dir=str(out_dir / "checkpoints" / fold_label),
                        train_dates=train_dates,
                        val_dates=val_dates,
                    )
                    ctx = f"{agent_name}/{cls}/fold{fold_idx}"
                    trainer = Trainer(
                        agent, env, class_syms, tcfg, label=cls,
                        logger=logger, context=ctx,
                    )
                    train_result = trainer.train()
                    print(f"  {cls}/{label}: {train_result}")

                    # Load best checkpoint and evaluate on test
                    ckpt = Path(tcfg.checkpoint_dir) / f"{cls}_best.pt"
                    if ckpt.exists():
                        agent.load(ckpt)

                    for sym in class_syms:
                        try:
                            rewards = _collect_rl_rewards(
                                agent, feat, sym,
                                action_mode=env_cfg.action_mode,
                                seq_len=env_cfg.seq_len,
                                start=test_dates[0], end=test_dates[1],
                            )
                            _append_rewards(results, sym, label, rewards)
                            print(f"    {sym}/{label}: {len(rewards)} days, "
                                  f"mean={rewards.mean():.6f}")
                        except Exception as e:
                            failures.append((sym, label, traceback.format_exc()))
                except Exception as e:
                    print(f"  {cls}/{label}: FAILED ({e})")
                    failures.append((cls, label, traceback.format_exc()))
                    for sym in class_syms:
                        if label not in results[sym]:
                            results[sym][label] = pd.Series(dtype=np.float64)

    # 4. Report failures
    if failures:
        print(f"\n{'!'*60}")
        print(f"  FAILURES: {len(failures)} method/symbol runs failed")
        print(f"{'!'*60}")
        for sym, method, tb in failures:
            print(f"\n  [{sym} / {method}]")
            print(tb)
        print(f"{'!'*60}\n")

    # 5. Aggregate portfolios
    print(f"\n{'='*60}")
    print("Aggregating portfolios ...")
    print(f"{'='*60}")

    groupings: dict[str, list[str]] = dict(class_symbols)
    groupings["All"] = all_symbols

    portfolio_rewards: dict[str, dict[str, np.ndarray]] = {}

    for grp_name, grp_syms in groupings.items():
        portfolio_rewards[grp_name] = {}
        for method in method_names:
            series = []
            for sym in grp_syms:
                r = results.get(sym, {}).get(method, pd.Series(dtype=np.float64))
                if len(r) > 0:
                    series.append(r.rename(sym))
            if not series:
                portfolio_rewards[grp_name][method] = np.array([])
                continue
            combined = pd.concat(series, axis=1, join="outer").sort_index()
            port_ret = combined.mean(axis=1).dropna().to_numpy()
            port_ret = _portfolio_vol_scale(port_ret)
            portfolio_rewards[grp_name][method] = port_ret

    # 6. Compute metrics table
    rows = []
    for grp_name in groupings:
        for method in method_names:
            r = portfolio_rewards[grp_name].get(method, np.array([]))
            m = compute_metrics(r)
            m["Group"] = grp_name
            m["Method"] = method
            rows.append(m)

    metrics_df = pd.DataFrame(rows)
    col_order = ["Group", "Method", "E(R)", "Std(R)", "DD", "Sharpe",
                 "Sortino", "MDD", "Calmar", "%+Ret", "AvgP/AvgL"]
    metrics_df = metrics_df[col_order]

    # 7. Print results table
    print(f"\n{'='*60}")
    print("Results (Zhang et al. Exhibit 2 style)")
    print(f"{'='*60}")
    _print_table(metrics_df)

    csv_path = out_dir / "results.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to {csv_path}")

    # 8. Plot cumulative trade returns
    fig = _plot_cumulative(portfolio_rewards, groupings, method_names, out_dir)

    # 9. Final wandb artifacts
    if logger is not None:
        try:
            logger.log_table("final/metrics_table", metrics_df)
            logger.log_image("final/cumulative_returns", fig)
        finally:
            plt.close(fig)
            logger.finish()
    else:
        plt.close(fig)

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
):
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
    print(f"Plot saved to {png_path}")
    return fig
