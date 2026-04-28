"""Single-tuple trainer: one (agent, fold, asset_class) per process.

Used as a slurm-array worker for parallel walk-forward experiments. Each
invocation loads the cached feature frame, trains one agent model, runs the
trained model across the fold's test window for every symbol in the class,
and writes per-symbol reward parquets plus a train-result json. The offline
aggregator (`rl_trading.aggregate`) stitches these back into the full
Zhang-style results table.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rl_trading.config import ACTIVE_UNIVERSE, walk_forward_splits
from rl_trading.env import TradingEnv
from rl_trading.evaluate import (
    _collect_rl_test,
    _detect_device,
    _extract_hparams,
    _filter_full_history_universe,
    _make_agent_factories,
    load_or_build_feature_cache,
)
from rl_trading.trainer import Trainer, TrainerConfig
from rl_trading.wandb_logger import WandbLogger


def _agent_label(agent_name: str) -> str:
    return agent_name.upper()


def _resolve_fold(fold_idx: int) -> dict[str, tuple[str, str]]:
    folds = walk_forward_splits()
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"fold_idx {fold_idx} out of range 0..{len(folds) - 1}")
    return folds[fold_idx]


def _eligible_universe(feat: pd.DataFrame, seq_len: int) -> list[dict[str, str]]:
    """Apply the same full-history filter used by the monolithic pipeline."""
    folds = walk_forward_splits()
    eligible, _ = _filter_full_history_universe(
        ACTIVE_UNIVERSE, feat, folds, min_rows=seq_len + 1,
    )
    return eligible


def run_one(
    agent_name: str,
    fold_idx: int,
    asset_class: str,
    *,
    seed: int = 0,
    feature_cache: str | Path,
    output_dir: str | Path,
    n_epochs: int,
    patience: int,
    early_stop_policy: str = "sharpe_only",
    device: str | None = None,
    wandb_enabled: bool = False,
    wandb_project: str = "rl-trading",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_group: str | None = None,
) -> dict:
    """Train one (agent, fold, class, seed) and dump results to output_dir."""
    # Set every random source before any model init or env step.
    import random as _random
    import torch as _torch
    _random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    if device is None:
        device = _detect_device()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Features
    requested = [u["symbol"] for u in ACTIVE_UNIVERSE]
    feat = load_or_build_feature_cache(requested, feature_cache)

    # Build agent + env config up-front so we know seq_len for the filter
    factories = _make_agent_factories(device)
    if agent_name not in factories:
        raise ValueError(f"Unknown agent: {agent_name}")
    agent, env_cfg = factories[agent_name]()

    universe = _eligible_universe(feat, env_cfg.seq_len)
    class_symbols = [u["symbol"] for u in universe if u["asset_class"] == asset_class]
    if not class_symbols:
        raise RuntimeError(
            f"No eligible symbols in class '{asset_class}' after full-history filter"
        )
    feat = feat.loc[feat["symbol"].isin([u["symbol"] for u in universe])].reset_index(drop=True)

    fold = _resolve_fold(fold_idx)
    fold_label = f"{asset_class}_fold{fold_idx + 1}"

    # wandb (optional)
    wandb_config = {
        "mode": "train_one",
        "agent": agent_name,
        "asset_class": asset_class,
        "fold_idx": fold_idx,
        "seed": seed,
        "fold": {k: list(v) for k, v in fold.items()},
        "class_symbols": class_symbols,
        "n_epochs": n_epochs,
        "patience": patience,
        "early_stop_policy": early_stop_policy,
        "device": device,
        "hparams": _extract_hparams(agent, env_cfg),
    }
    per_tuple_tags = list(wandb_tags or []) + [
        f"class:{asset_class}", f"fold:{fold_idx + 1}", f"seed:{seed}",
    ]
    logger = WandbLogger.init(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{agent_name}-{fold_label}-s{seed}",
        tags=per_tuple_tags,
        config=wandb_config,
        group=wandb_group,
        job_type=agent_name,
    )

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_result: dict = {
        "agent": agent_name,
        "fold_idx": fold_idx,
        "asset_class": asset_class,
        "seed": seed,
        "train": list(fold["train"]),
        "val": list(fold["val"]),
        "test": list(fold["test"]),
        "class_symbols": class_symbols,
    }

    try:
        env = TradingEnv(feat, env_cfg)
        tcfg = TrainerConfig(
            n_epochs=n_epochs,
            patience=patience,
            checkpoint_dir=str(ckpt_dir),
            train_dates=fold["train"],
            val_dates=fold["val"],
            early_stop_policy=early_stop_policy,
        )
        ctx = f"{agent_name}/{asset_class}/fold{fold_idx + 1}"
        trainer = Trainer(
            agent, env, class_symbols, tcfg, label=asset_class,
            logger=logger, context=ctx,
        )
        summary = trainer.train()
        train_result.update({"status": "ok", **summary})

        ckpt = ckpt_dir / f"{asset_class}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        agent.load(ckpt)

        # Collect test-window rewards per symbol.
        rows_zhang: list[pd.DataFrame] = []
        rows_raw: list[pd.DataFrame] = []
        per_symbol_failures: dict[str, str] = {}
        for sym in class_symbols:
            try:
                trade = _collect_rl_test(
                    agent, feat, sym,
                    action_mode=env_cfg.action_mode,
                    seq_len=env_cfg.seq_len,
                    start=fold["test"][0], end=fold["test"][1],
                )
                rows_zhang.append(pd.DataFrame({
                    "date": trade.index,
                    "symbol": sym,
                    "reward_scaled": trade["reward"].to_numpy(dtype=np.float64),
                }))
                simple_r = trade["daily_return"].to_numpy() / trade["price"].to_numpy()
                rows_raw.append(pd.DataFrame({
                    "date": trade.index,
                    "symbol": sym,
                    "frac_return": trade["position"].to_numpy() * simple_r,
                }))
            except Exception:
                per_symbol_failures[sym] = traceback.format_exc()

        if rows_zhang:
            pd.concat(rows_zhang, ignore_index=True).to_parquet(
                out_dir / "rewards_zhang.parquet", index=False,
            )
        if rows_raw:
            pd.concat(rows_raw, ignore_index=True).to_parquet(
                out_dir / "rewards_raw.parquet", index=False,
            )
        if per_symbol_failures:
            train_result["per_symbol_failures"] = {
                s: tb.splitlines()[-1] for s, tb in per_symbol_failures.items()
            }
    except Exception:
        train_result["status"] = "failed"
        train_result["error"] = traceback.format_exc()
        raise
    finally:
        with (out_dir / "train_result.json").open("w") as f:
            json.dump(train_result, f, indent=2, default=str)
        if logger is not None:
            logger.finish()

    return train_result


# ── CLI ─────────────────────────────────────────────────────────────

_ASSET_CLASSES = ["commodity", "equity_index", "fixed_income", "fx"]
_AGENTS = ["dqn", "pg", "a2c"]
_N_SEEDS = 3  # multi-seed per (agent, fold, class); aggregator picks best by val Sharpe


def _tuple_from_array_index(idx: int) -> tuple[str, int, str, int]:
    """Decode a slurm array index 0..N-1 into (agent, fold_idx, asset_class, seed)."""
    folds = walk_forward_splits()
    n_folds = len(folds)
    n_classes = len(_ASSET_CLASSES)
    per_seed = n_folds * n_classes
    per_agent = per_seed * _N_SEEDS
    total = per_agent * len(_AGENTS)
    if idx < 0 or idx >= total:
        raise ValueError(
            f"array_index {idx} out of range 0..{total - 1} "
            f"(agents={len(_AGENTS)} x seeds={_N_SEEDS} x folds={n_folds} x classes={n_classes})"
        )
    agent_i, rem = divmod(idx, per_agent)
    seed_i, rem = divmod(rem, per_seed)
    fold_i, class_i = divmod(rem, n_classes)
    return _AGENTS[agent_i], fold_i, _ASSET_CLASSES[class_i], seed_i


def _canonical_output_dir(
    run_dir: Path, agent: str, fold_idx: int, asset_class: str, seed: int,
) -> Path:
    return Path(run_dir) / agent / f"{asset_class}_fold{fold_idx + 1}_seed{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=_AGENTS, default=None)
    parser.add_argument("--fold-idx", type=int, default=None,
                        help="Zero-based fold index into walk_forward_splits()")
    parser.add_argument("--asset-class", choices=_ASSET_CLASSES, default=None)
    parser.add_argument("--array-index", type=int, default=None,
                        help="Alternative to the three flags above; decodes "
                             "SLURM_ARRAY_TASK_ID into (agent, fold, class)")
    parser.add_argument("--feature-cache", required=True,
                        help="Path to prebuilt feature parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir",
                       help="Write artifacts here directly (for standalone runs)")
    group.add_argument("--run-dir",
                       help="Parent directory; output goes to "
                            "<run-dir>/<agent>/<class>_fold<n>")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=None,
                        help="Override per-agent default patience")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for torch/numpy/random (overrides --array-index)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rl-trading")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-group", default=None,
                        help="wandb run group; defaults to basename of --run-dir so "
                             "all tasks in one parallel sweep share a group")
    args = parser.parse_args()

    if args.array_index is not None:
        agent, fold_idx, asset_class, seed = _tuple_from_array_index(args.array_index)
    else:
        if args.agent is None or args.fold_idx is None or args.asset_class is None:
            parser.error(
                "Must specify either --array-index or all three of "
                "--agent / --fold-idx / --asset-class"
            )
        agent, fold_idx, asset_class = args.agent, args.fold_idx, args.asset_class
        seed = args.seed if args.seed is not None else 0
    if args.seed is not None:
        seed = args.seed

    # Per-agent default patience and early-stop policy.
    # DQN/PG are slow learners under our settings → OR patience.
    # A2C is the fastest learner → paper-literal Sharpe-only.
    _DEFAULTS = {
        "dqn": {"patience": 50, "policy": "or_multi"},
        "pg":  {"patience": 50, "policy": "or_multi"},
        "a2c": {"patience": 30, "policy": "sharpe_only"},
    }
    patience = args.patience if args.patience is not None else _DEFAULTS[agent]["patience"]
    early_stop_policy = _DEFAULTS[agent]["policy"]

    if args.run_dir is not None:
        output_dir = _canonical_output_dir(Path(args.run_dir), agent, fold_idx, asset_class, seed)
        wandb_group = args.wandb_group or Path(args.run_dir).name
    else:
        output_dir = Path(args.output_dir)
        wandb_group = args.wandb_group

    print(
        f"Training tuple: agent={agent} fold={fold_idx + 1} class={asset_class} "
        f"seed={seed} patience={patience} policy={early_stop_policy}"
    )
    print(f"Output dir: {output_dir}")
    run_one(
        agent_name=agent,
        fold_idx=fold_idx,
        asset_class=asset_class,
        seed=seed,
        feature_cache=args.feature_cache,
        output_dir=output_dir,
        n_epochs=args.epochs,
        patience=patience,
        early_stop_policy=early_stop_policy,
        device=args.device,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_group=wandb_group,
    )


if __name__ == "__main__":
    main()
