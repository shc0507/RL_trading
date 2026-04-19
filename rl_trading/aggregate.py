"""Offline aggregator for parallel walk-forward runs.

Scans a run directory produced by `rl_trading.train_one` slurm-array tasks,
stitches per-agent test-window rewards across folds, computes baselines on the
same test windows from the cached feature frame, and runs the shared portfolio
aggregation / metrics / plotting pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rl_trading.baselines import (
    compute_baseline_rewards,
    long_only,
    macd_signal,
    sign_r,
)
from rl_trading.config import ACTIVE_UNIVERSE, walk_forward_splits
from rl_trading.env import EnvConfig
from rl_trading.evaluate import (
    _baseline_frac_daily,
    _filter_full_history_universe,
    _print_universe_summary,
    aggregate_and_report,
    load_or_build_feature_cache,
)

_BASELINE_FNS = {
    "Long": long_only,
    "Sign(R)": sign_r,
    "MACD": macd_signal,
}
_AGENT_LABELS = ["DQN", "PG", "A2C"]
_ASSET_CLASSES = ["commodity", "equity_index", "fixed_income", "fx"]


def _append_series(
    nested: dict[str, dict[str, pd.Series]],
    sym: str,
    method: str,
    s: pd.Series,
) -> None:
    if s is None or len(s) == 0:
        return
    if method in nested[sym] and len(nested[sym][method]) > 0:
        nested[sym][method] = pd.concat([nested[sym][method], s]).sort_index()
    else:
        nested[sym][method] = s.sort_index()


def aggregate(run_dir: Path, output_dir: Path, feature_cache: Path) -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    feature_cache = Path(feature_cache)

    # 1. Features + eligible universe (matches train_one's filter logic).
    requested_syms = [u["symbol"] for u in ACTIVE_UNIVERSE]
    feat = load_or_build_feature_cache(requested_syms, feature_cache)
    folds = walk_forward_splits()
    # We don't know agent seq_len here but all three agents use seq_len=60
    # per _make_agent_factories. Use the largest defensible value.
    min_rows = EnvConfig().seq_len + 1
    universe, exclusions = _filter_full_history_universe(
        ACTIVE_UNIVERSE, feat, folds, min_rows=min_rows,
    )
    _print_universe_summary(requested_syms, universe, exclusions, min_rows)
    all_symbols = [u["symbol"] for u in universe]
    sym_to_class = {u["symbol"]: u["asset_class"] for u in universe}
    class_symbols: dict[str, list[str]] = defaultdict(list)
    for s in all_symbols:
        class_symbols[sym_to_class[s]].append(s)
    feat = feat.loc[feat["symbol"].isin(all_symbols)].reset_index(drop=True)

    results: dict[str, dict[str, pd.Series]] = {sym: {} for sym in all_symbols}
    raw_daily: dict[str, dict[str, pd.Series]] = {sym: {} for sym in all_symbols}

    # 2. Baselines per fold.
    print(f"\nComputing baselines across {len(folds)} folds ...")
    for fold_idx, fold in enumerate(folds, 1):
        test_start, test_end = fold["test"]
        for sym in all_symbols:
            for name, fn in _BASELINE_FNS.items():
                try:
                    positions = fn(feat, sym, start=test_start, end=test_end)
                    rewards = compute_baseline_rewards(
                        positions, feat, sym, start=test_start, end=test_end,
                    )
                    _append_series(results, sym, name, rewards)
                    frac = _baseline_frac_daily(feat, sym, fn, test_start, test_end)
                    _append_series(raw_daily, sym, name, frac)
                except Exception as exc:
                    print(f"  baseline failure fold{fold_idx} {sym}/{name}: {exc}")

    # 3. Stitch RL agent rewards from the run directory.
    found: dict[tuple[str, int, str], bool] = {}
    for agent_label in _AGENT_LABELS:
        agent_dir = run_dir / agent_label.lower()
        if not agent_dir.is_dir():
            print(f"  (no directory for {agent_label})")
            continue
        for fold_idx, _ in enumerate(folds):
            for cls in _ASSET_CLASSES:
                tuple_dir = agent_dir / f"{cls}_fold{fold_idx + 1}"
                zhang_p = tuple_dir / "rewards_zhang.parquet"
                raw_p = tuple_dir / "rewards_raw.parquet"
                if not zhang_p.exists():
                    continue
                found[(agent_label, fold_idx, cls)] = True
                z = pd.read_parquet(zhang_p)
                z["date"] = pd.to_datetime(z["date"])
                for sym, group in z.groupby("symbol"):
                    if sym not in results:
                        continue
                    s = pd.Series(
                        group["reward_scaled"].to_numpy(dtype=np.float64),
                        index=group["date"].to_numpy(),
                    )
                    _append_series(results, sym, agent_label, s)
                if raw_p.exists():
                    r = pd.read_parquet(raw_p)
                    r["date"] = pd.to_datetime(r["date"])
                    for sym, group in r.groupby("symbol"):
                        if sym not in raw_daily:
                            continue
                        s = pd.Series(
                            group["frac_return"].to_numpy(dtype=np.float64),
                            index=group["date"].to_numpy(),
                        )
                        _append_series(raw_daily, sym, agent_label, s)

    # Report completeness
    total_tuples = len(_AGENT_LABELS) * len(folds) * len(_ASSET_CLASSES)
    print(
        f"\nLoaded {len(found)}/{total_tuples} agent tuples from {run_dir}"
    )
    if len(found) < total_tuples:
        missing = [
            (a, f + 1, c)
            for a in _AGENT_LABELS
            for f in range(len(folds))
            for c in _ASSET_CLASSES
            if (a, f, c) not in found
        ]
        print(f"Missing tuples ({len(missing)}):")
        for m in missing[:20]:
            print(f"  {m[0]}/fold{m[1]}/{m[2]}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    # 4. Aggregation + reporting (shared helper).
    method_names = list(_BASELINE_FNS.keys()) + [
        label for label in _AGENT_LABELS
        if any(label in results[s] for s in all_symbols)
    ]
    aggregate_and_report(
        results, raw_daily,
        class_symbols=dict(class_symbols),
        all_symbols=all_symbols,
        method_names=method_names,
        out_dir=output_dir,
        logger=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Directory containing <agent>/<class>_fold<n>/ subdirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache", required=True)
    args = parser.parse_args()
    aggregate(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        feature_cache=Path(args.feature_cache),
    )


if __name__ == "__main__":
    main()
