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
    ACTIVE_UNIVERSE,
    DEFAULT_VOL_TARGET,
    PORTFOLIO_VOL_TARGET,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    UNIVERSE,
    VAL_END,
    VAL_START,
    walk_forward_splits,
)
from rl_trading.data import load_bars
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


def _collect_rl_test(
    agent: DQNAgent | PGAgent | A2CAgent,
    feature_frame: pd.DataFrame,
    symbol: str,
    split: str = "test",
    *,
    action_mode: str,
    seq_len: int,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Run a trained agent and return DataFrame[reward, position, price, daily_return] by date."""
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
    return pd.DataFrame(
        {
            "reward": np.asarray(env.history["reward"], dtype=np.float64),
            "position": np.asarray(env.history["position"], dtype=np.float64),
            "price": np.asarray(env.history["price"], dtype=np.float64),
            "daily_return": np.asarray(env.history["daily_return"], dtype=np.float64),
        },
        index=pd.to_datetime(env.history["date"]),
    )


def _baseline_frac_daily(
    feature_frame: pd.DataFrame, symbol: str, fn, start: str, end: str,
) -> pd.Series:
    """Unscaled fractional daily portfolio return w_t · (p_{t+1}/p_t - 1) for a baseline."""
    mask = (
        (feature_frame["symbol"] == symbol)
        & (feature_frame["window_ready"])
        & (feature_frame["date"] >= start)
        & (feature_frame["date"] <= end)
    )
    sub = feature_frame.loc[mask].sort_values("date").reset_index(drop=True)
    if len(sub) < 2:
        return pd.Series(dtype=np.float64)
    positions = fn(feature_frame, symbol, start=start, end=end).astype(np.float64)
    prices = sub["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(sub["date"].to_numpy())
    n = len(prices)
    simple_r = prices[1:] / prices[:-1] - 1.0
    return pd.Series(positions[: n - 1] * simple_r, index=dates[: n - 1])


def _portfolio_vol_scale(
    port_returns: np.ndarray,
    vol_target: float = PORTFOLIO_VOL_TARGET,
) -> np.ndarray:
    """Portfolio-level vol normalization, Zhang 2019 Exhibit 2.

    Paper says the extra scaling "brings volatility of different methods
    to the same target" — a reporting normalization applied after the
    full test stream is in hand (Exhibit 2's Std(R) is near-identical
    across methods, which only happens with a single constant rescale).
    So we scale the whole series by one factor = σ_tgt / σ_realized_full,
    uncapped; returns raw if variance is zero.
    """
    arr = np.asarray(port_returns, dtype=np.float64)
    if len(arr) < 2:
        return arr
    realized = arr.std(ddof=1) * math.sqrt(252)
    if not np.isfinite(realized) or realized <= 0.0:
        return arr
    return arr * (vol_target / realized)


def _dump_series_csv(
    nested: dict[str, dict[str, pd.Series]],
    path: Path,
    *,
    key_col: str,
    value_col: str,
) -> None:
    """Flatten {outer: {method: Series[date]}} to tidy CSV [date, key_col, method, value_col]."""
    rows = []
    for outer_key, inner in nested.items():
        for method, s in inner.items():
            if len(s) == 0:
                continue
            df = pd.DataFrame({
                "date": s.index,
                key_col: outer_key,
                "method": method,
                value_col: s.to_numpy(),
            })
            rows.append(df)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def _append_rewards(
    results: dict[str, dict[str, pd.Series]],
    sym: str, method: str, rewards: pd.Series,
) -> None:
    """Append rewards to results, concatenating across folds."""
    if method in results[sym] and len(results[sym][method]) > 0:
        results[sym][method] = pd.concat([results[sym][method], rewards])
    else:
        results[sym][method] = rewards


def _count_window_ready_rows(
    feature_frame: pd.DataFrame,
    symbol: str,
    start: str,
    end: str,
) -> int:
    mask = (
        (feature_frame["symbol"] == symbol)
        & (feature_frame["window_ready"])
        & (feature_frame["date"] >= start)
        & (feature_frame["date"] <= end)
    )
    return int(mask.sum())


def _filter_full_history_universe(
    universe: list[dict[str, str]],
    feature_frame: pd.DataFrame,
    folds: list[dict[str, tuple[str, str]]],
    min_rows: int,
) -> tuple[list[dict[str, str]], list[dict[str, str | int]]]:
    """Keep only symbols that can support every fold and split."""
    eligible: list[dict[str, str]] = []
    exclusions: list[dict[str, str | int]] = []

    for entry in universe:
        symbol = entry["symbol"]
        exclusion: dict[str, str | int] | None = None
        for fold_idx, fold in enumerate(folds, 1):
            for split_name in ("train", "val", "test"):
                start, end = fold[split_name]
                rows = _count_window_ready_rows(feature_frame, symbol, start, end)
                if rows < min_rows:
                    exclusion = {
                        "symbol": symbol,
                        "asset_class": entry["asset_class"],
                        "fold_idx": fold_idx,
                        "split": split_name,
                        "rows": rows,
                        "min_rows": min_rows,
                        "start": start,
                        "end": end,
                    }
                    break
            if exclusion is not None:
                break
        if exclusion is None:
            eligible.append(entry)
        else:
            exclusions.append(exclusion)

    return eligible, exclusions


def _print_universe_summary(
    requested_symbols: list[str],
    eligible_universe: list[dict[str, str]],
    exclusions: list[dict[str, str | int]],
    min_rows: int,
) -> None:
    print(
        f"Eligible stable universe: {len(eligible_universe)}/{len(requested_symbols)} "
        f"symbols (requires >= {min_rows} window-ready rows per fold split)"
    )
    if not exclusions:
        return
    print("Excluded symbols:")
    for item in exclusions:
        print(
            f"  {item['symbol']} ({item['asset_class']}): fold {item['fold_idx']} "
            f"{item['split']} has {item['rows']} rows in "
            f"{item['start']}–{item['end']}; need >= {item['min_rows']}"
        )


def aggregate_and_report(
    results: dict[str, dict[str, pd.Series]],
    raw_daily: dict[str, dict[str, pd.Series]],
    *,
    class_symbols: dict[str, list[str]],
    all_symbols: list[str],
    method_names: list[str],
    out_dir: Path,
    logger: "WandbLogger | None" = None,
) -> pd.DataFrame:
    """Portfolio aggregation + metrics + CSV + plots.

    Extracted from run_experiment so both the monolithic pipeline and the
    offline aggregator (rl_trading.aggregate) can share the reporting logic.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groupings: dict[str, list[str]] = dict(class_symbols)
    groupings["All"] = all_symbols

    portfolio_rewards: dict[str, dict[str, pd.Series]] = {}
    portfolio_rewards_scaled: dict[str, dict[str, pd.Series]] = {}
    portfolio_raw: dict[str, dict[str, pd.Series]] = {}

    for grp_name, grp_syms in groupings.items():
        portfolio_rewards[grp_name] = {}
        portfolio_rewards_scaled[grp_name] = {}
        portfolio_raw[grp_name] = {}
        for method in method_names:
            series = []
            raw_series = []
            for sym in grp_syms:
                r = results.get(sym, {}).get(method, pd.Series(dtype=np.float64))
                if len(r) > 0:
                    series.append(r.rename(sym))
                f = raw_daily.get(sym, {}).get(method, pd.Series(dtype=np.float64))
                if len(f) > 0:
                    raw_series.append(f.rename(sym))
            if series:
                combined = pd.concat(series, axis=1, join="outer").sort_index()
                port = combined.mean(axis=1).dropna()
                portfolio_rewards[grp_name][method] = port
                scaled = _portfolio_vol_scale(port.to_numpy())
                portfolio_rewards_scaled[grp_name][method] = pd.Series(
                    scaled, index=port.index,
                )
            else:
                portfolio_rewards[grp_name][method] = pd.Series(dtype=np.float64)
                portfolio_rewards_scaled[grp_name][method] = pd.Series(dtype=np.float64)
            if raw_series:
                combined_raw = pd.concat(raw_series, axis=1, join="outer").sort_index()
                portfolio_raw[grp_name][method] = combined_raw.mean(axis=1).dropna()
            else:
                portfolio_raw[grp_name][method] = pd.Series(dtype=np.float64)

    col_order = ["Group", "Method", "E(R)", "Std(R)", "DD", "Sharpe",
                 "Sortino", "MDD", "Calmar", "%+Ret", "AvgP/AvgL"]

    def _metrics_from(nested: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
        rows = []
        for grp_name in groupings:
            for method in method_names:
                r = nested[grp_name].get(method, pd.Series(dtype=np.float64))
                m = compute_metrics(r.to_numpy())
                m["Group"] = grp_name
                m["Method"] = method
                rows.append(m)
        return pd.DataFrame(rows)[col_order]

    metrics_df = _metrics_from(portfolio_rewards_scaled)
    metrics_df_raw = _metrics_from(portfolio_rewards)

    print(f"\n{'='*60}")
    print("Results — Table 2 (portfolio-level vol scaling applied)")
    print(f"{'='*60}")
    _print_table(metrics_df)
    print(f"\n{'='*60}")
    print("Results — Table 3 (no portfolio-level vol scaling)")
    print(f"{'='*60}")
    _print_table(metrics_df_raw)

    metrics_df.to_csv(out_dir / "results.csv", index=False, float_format="%.4f")
    metrics_df_raw.to_csv(out_dir / "results_unscaled.csv", index=False, float_format="%.4f")
    print(f"\nResults saved to {out_dir / 'results.csv'} and {out_dir / 'results_unscaled.csv'}")

    _dump_series_csv(
        portfolio_rewards_scaled, out_dir / "portfolio_zhang.csv",
        key_col="group", value_col="reward_scaled",
    )
    _dump_series_csv(
        portfolio_rewards, out_dir / "portfolio_zhang_unscaled.csv",
        key_col="group", value_col="reward",
    )
    _dump_series_csv(
        portfolio_raw, out_dir / "portfolio_raw.csv",
        key_col="group", value_col="frac_return",
    )
    _dump_series_csv(
        results, out_dir / "per_symbol_zhang.csv",
        key_col="symbol", value_col="reward_scaled",
    )
    _dump_series_csv(
        raw_daily, out_dir / "per_symbol_raw.csv",
        key_col="symbol", value_col="frac_return",
    )

    fig = _plot_cumulative(portfolio_rewards_scaled, groupings, method_names, out_dir)
    fig_raw = _plot_cumulative_raw(portfolio_raw, groupings, method_names, out_dir)

    try:
        if logger is not None:
            logger.log_table("final/metrics_table", metrics_df)
            logger.log_image("final/cumulative_returns", fig)
            logger.log_image("final/cumulative_return_raw", fig_raw)
    finally:
        plt.close(fig)
        plt.close(fig_raw)

    return metrics_df


def build_feature_frame(symbols: list[str]) -> pd.DataFrame:
    """Fetch bars and build the feature frame used by training and evaluation.

    Fetch starts two years before TRAIN_START so rolling-window features are
    warmed up by the time the training split begins.
    """
    fetch_start = str(int(TRAIN_START[:4]) - 2) + TRAIN_START[4:]
    bars = load_bars(symbols, start=fetch_start, end=TEST_END)
    return FeatureBuilder().transform(bars)


def load_or_build_feature_cache(
    symbols: list[str],
    cache_path: str | Path,
) -> pd.DataFrame:
    """Read a prebuilt feature parquet if it exists, else build and cache it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    feat = build_feature_frame(symbols)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(cache_path, index=False)
    return feat


def _print_failures(
    failures: list[tuple[str, str, str]],
    *,
    label: str,
) -> None:
    print(f"\n{'!'*60}")
    print(f"  {label}: {len(failures)} runs failed")
    print(f"{'!'*60}")
    for scope, method, tb in failures:
        print(f"\n  [{scope} / {method}]")
        print(tb)
    print(f"{'!'*60}\n")


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
    preflight_only: bool = False,
    feature_cache_path: str | Path | None = None,
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
    feature_cache_path : optional path to a prebuilt feature parquet. If it
        exists, features are loaded from disk instead of refetching.
    """
    if agents is None:
        agents = ["dqn", "pg", "a2c"]

    if device is None:
        device = _detect_device()
    print(f"Using device: {device}")

    agent_factories = _make_agent_factories(device)

    # Resolve symbols and asset classes
    universe = ACTIVE_UNIVERSE
    if symbols is not None:
        known_syms = {u["symbol"] for u in ACTIVE_UNIVERSE}
        unknown = [s for s in symbols if s not in known_syms]
        if unknown:
            raise ValueError(f"Unknown symbols not in UNIVERSE: {unknown}")
        sym_set = set(symbols)
        universe = [u for u in ACTIVE_UNIVERSE if u["symbol"] in sym_set]

    requested_symbols = [u["symbol"] for u in universe]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data and build features (or reuse prebuilt cache)
    if feature_cache_path is not None:
        print(f"Loading feature cache: {feature_cache_path}")
        feat = load_or_build_feature_cache(requested_symbols, feature_cache_path)
        feat = feat.loc[feat["symbol"].isin(requested_symbols)].reset_index(drop=True)
    else:
        print("Fetching data and building features ...")
        feat = build_feature_frame(requested_symbols)

    # 2. Determine folds
    if walk_forward:
        folds = walk_forward_splits()
    else:
        folds = [{
            "train": (TRAIN_START, TRAIN_END),
            "val": (VAL_START, VAL_END),
            "test": (TEST_START, TEST_END),
        }]

    # Resolve agent metadata up front so eligibility filtering can respect
    # the longest state sequence required by any configured agent.
    agent_hparams: dict[str, dict] = {}
    min_history_rows = 2
    for agent_name in agents:
        a, env_cfg = agent_factories[agent_name]()
        agent_hparams[agent_name] = _extract_hparams(a, env_cfg)
        min_history_rows = max(min_history_rows, env_cfg.seq_len + 1)

    universe, exclusions = _filter_full_history_universe(
        universe,
        feat,
        folds,
        min_rows=min_history_rows,
    )
    if not universe:
        raise ValueError(
            "No eligible symbols remain after full-history filtering. "
            "Reduce the requested universe or shorten the walk-forward horizon."
        )

    _print_universe_summary(requested_symbols, universe, exclusions, min_history_rows)

    all_symbols = [u["symbol"] for u in universe]
    if len(all_symbols) != len(requested_symbols):
        feat = feat.loc[feat["symbol"].isin(all_symbols)].reset_index(drop=True)

    sym_to_class = {u["symbol"]: u["asset_class"] for u in universe}

    # Group symbols by asset class after filtering so every fold sees one stable universe.
    class_symbols: dict[str, list[str]] = defaultdict(list)
    for sym in all_symbols:
        class_symbols[sym_to_class[sym]].append(sym)

    if preflight_only:
        print("Preflight only: stopping after stable-universe inspection.")
        return None

    # Initialize wandb (no-op if disabled)
    wandb_config = {
        "agents": agents,
        "n_epochs": n_epochs,
        "patience": patience,
        "device": device,
        "walk_forward": walk_forward,
        "requested_symbols": requested_symbols,
        "symbols": all_symbols,
        "excluded_symbols": [str(item["symbol"]) for item in exclusions],
        "eligibility_min_rows": min_history_rows,
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
    raw_daily: dict[str, dict[str, pd.Series]] = {sym: {} for sym in all_symbols}
    fatal_failures: list[tuple[str, str, str]] = []
    fig = None
    fig_raw = None

    try:
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
                        frac = _baseline_frac_daily(
                            feat, sym, fn, test_dates[0], test_dates[1],
                        )
                        _append_rewards(raw_daily, sym, name, frac)
                    except Exception:
                        fatal_failures.append((sym, name, traceback.format_exc()))
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

                        ckpt = Path(tcfg.checkpoint_dir) / f"{cls}_best.pt"
                        if not ckpt.exists():
                            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
                        agent.load(ckpt)

                        for sym in class_syms:
                            try:
                                trade = _collect_rl_test(
                                    agent, feat, sym,
                                    action_mode=env_cfg.action_mode,
                                    seq_len=env_cfg.seq_len,
                                    start=test_dates[0], end=test_dates[1],
                                )
                                rewards = pd.Series(
                                    trade["reward"].to_numpy(), index=trade.index,
                                )
                                _append_rewards(results, sym, label, rewards)
                                simple_r = trade["daily_return"].to_numpy() / trade["price"].to_numpy()
                                frac = pd.Series(
                                    trade["position"].to_numpy() * simple_r,
                                    index=trade.index,
                                )
                                _append_rewards(raw_daily, sym, label, frac)
                                print(f"    {sym}/{label}: {len(rewards)} days, "
                                      f"mean={rewards.mean():.6f}")
                            except Exception:
                                fatal_failures.append((sym, label, traceback.format_exc()))
                    except Exception as exc:
                        print(f"  {cls}/{label}: FAILED ({exc})")
                        fatal_failures.append((cls, label, traceback.format_exc()))

        if fatal_failures:
            _print_failures(fatal_failures, label="POST-FILTER FAILURES")
            raise RuntimeError(
                f"Aborting result generation after {len(fatal_failures)} post-filter failures."
            )

        # 5-8. Aggregate portfolios, compute metrics, dump CSVs, plot.
        metrics_df = aggregate_and_report(
            results, raw_daily,
            class_symbols=dict(class_symbols),
            all_symbols=all_symbols,
            method_names=method_names,
            out_dir=out_dir,
            logger=logger,
        )
        return metrics_df
    finally:
        if logger is not None:
            logger.finish()


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


_ZHANG_STYLE: dict[str, dict] = {
    "Long":    {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.2},
    "Sign(R)": {"color": "#ff7f0e", "linestyle": "-",  "linewidth": 1.0,
                "marker": "o", "markersize": 4, "markevery": 60},
    "MACD":    {"color": "#2ca02c", "linestyle": "-",  "linewidth": 1.0,
                "marker": "*", "markersize": 6, "markevery": 60},
    "DQN":     {"color": "#d62728", "linestyle": "-",  "linewidth": 1.5},
    "PG":      {"color": "#9467bd", "linestyle": "-",  "linewidth": 1.0,
                "marker": "h", "markersize": 5, "markevery": 60},
    "A2C":     {"color": "#8c564b", "linestyle": ":",  "linewidth": 1.0,
                "marker": "+", "markersize": 6, "markevery": 60},
}

_ZHANG_ORDER = ["commodity", "equity_index", "fixed_income", "fx", "All"]

_ZHANG_TITLES = {
    "commodity": "Commodity",
    "equity_index": "Equity Index",
    "fixed_income": "Fixed Income",
    "fx": "FX",
    "All": "All",
}


def _plot_cumulative(
    portfolio_rewards: dict[str, dict[str, pd.Series]],
    groupings: dict[str, list[str]],
    method_names: list[str],
    out_dir: Path,
):
    """Plot cumulative trade returns in Zhang (2020) Exhibit 3 style:
    2x3 grid (commodity, equity_index, fixed_income / fx, All, hidden),
    calendar-year x-axis, no titles/labels/grids, single bottom legend.
    """
    import matplotlib.dates as mdates

    grps_ordered = [g for g in _ZHANG_ORDER if g in groupings]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ax_flat = axes.flatten()

    method_handles: dict[str, object] = {}
    for i, grp in enumerate(grps_ordered):
        ax = ax_flat[i]
        for method in method_names:
            s = portfolio_rewards[grp].get(method, pd.Series(dtype=np.float64))
            if len(s) == 0:
                continue
            cum = s.cumsum()
            style = _ZHANG_STYLE.get(method, {"linewidth": 1.0})
            line, = ax.plot(cum.index, cum.values, label=method, **style)
            method_handles.setdefault(method, line)
        ax.set_title(_ZHANG_TITLES.get(grp, grp), fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Hide unused slots
    for j in range(len(grps_ordered), len(ax_flat)):
        ax_flat[j].axis("off")

    # Single horizontal legend below all subplots
    handles = [method_handles[m] for m in method_names if m in method_handles]
    labels = [m for m in method_names if m in method_handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    png_path = out_dir / "cumulative_returns.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {png_path}")
    return fig


def _plot_cumulative_raw(
    portfolio_raw: dict[str, dict[str, pd.Series]],
    groupings: dict[str, list[str]],
    method_names: list[str],
    out_dir: Path,
):
    """Plot compounded $1 cumulative return in % — wealth_t = prod(1 + w_s · r_s) - 1,
    equal-weighted per-group portfolio, Zhang Exhibit 3 2×3 layout."""
    import matplotlib.dates as mdates

    grps_ordered = [g for g in _ZHANG_ORDER if g in groupings]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ax_flat = axes.flatten()

    method_handles: dict[str, object] = {}
    for i, grp in enumerate(grps_ordered):
        ax = ax_flat[i]
        for method in method_names:
            s = portfolio_raw[grp].get(method, pd.Series(dtype=np.float64))
            if len(s) == 0:
                continue
            gross = np.clip(1.0 + s.to_numpy(), 1e-8, None)
            cum_pct = (np.cumprod(gross) - 1.0) * 100.0
            style = _ZHANG_STYLE.get(method, {"linewidth": 1.0})
            line, = ax.plot(s.index, cum_pct, label=method, **style)
            method_handles.setdefault(method, line)
        ax.set_title(_ZHANG_TITLES.get(grp, grp), fontsize=10)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for j in range(len(grps_ordered), len(ax_flat)):
        ax_flat[j].axis("off")

    handles = [method_handles[m] for m in method_names if m in method_handles]
    labels = [m for m in method_names if m in method_handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Raw Cumulative Return (%) — compounded $1, equal-weighted",
                 fontsize=11, y=0.995)

    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    png_path = out_dir / "cumulative_return_raw.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {png_path}")
    return fig
