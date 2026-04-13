"""Plotting helpers for RL experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..asset_groups import display_asset_group
from ..metrics import compute_performance_metrics
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation

POLICY_ORDER = (
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "lstm_double_dueling_dqn",
    "a2c",
    "ppo",
    "td3",
    "sac",
    "long_only",
    "sign_12m",
    "macd",
)
POLICY_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "lstm_double_dueling_dqn": "LSTM Double Dueling DQN",
    "a2c": "A2C",
    "ppo": "PPO",
    "td3": "TD3",
    "sac": "SAC",
    "long_only": "Long Only",
    "sign_12m": "Sign(R)",
    "macd": "MACD",
}
POLICY_COLORS = {
    "dqn": "#1f77b4",
    "double_dqn": "#ff7f0e",
    "dueling_dqn": "#17becf",
    "lstm_double_dueling_dqn": "#9467bd",
    "a2c": "#8c564b",
    "ppo": "#e377c2",
    "td3": "#7f7f7f",
    "sac": "#bcbd22",
    "long_only": "#2ca02c",
    "sign_12m": "#d62728",
    "macd": "#111111",
}
POLICY_LINESTYLES = {
    "dqn": "-",
    "double_dqn": "-",
    "dueling_dqn": "-",
    "lstm_double_dueling_dqn": "-",
    "a2c": "-",
    "ppo": "-",
    "td3": "-",
    "sac": "-",
    "long_only": "--",
    "sign_12m": "--",
    "macd": ":",
}
ASSET_GROUP_ORDER = ("commodity", "equity_index", "fixed_income", "fx", "unknown", "all")


@dataclass(slots=True)
class PlotArtifacts:
    cumulative_returns_path: Path
    symbol_diagnostics_path: Path
    cost_sweep_path: Path | None = None
    style: str = "zhang"


def _reports_dir(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "reports"


def _plot_dir(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _zhang_eval_dir(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "zhang_eval"


def _policy_sort_key(policy: str) -> tuple[int, str]:
    try:
        return (POLICY_ORDER.index(policy), policy)
    except ValueError:
        return (len(POLICY_ORDER), policy)


def _asset_group_sort_key(asset_group: str) -> tuple[int, str]:
    try:
        return (ASSET_GROUP_ORDER.index(asset_group), asset_group)
    except ValueError:
        return (len(ASSET_GROUP_ORDER), asset_group)


def _policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def _policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#4c566a")


def _policy_linestyle(policy: str) -> str:
    return POLICY_LINESTYLES.get(policy, "-")


def _policy_linewidth(policy: str) -> float:
    return 2.2 if policy in {"dqn", "long_only"} else 1.8


def _ordered_policies(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["policy"].unique().tolist(), key=_policy_sort_key)


def _policy_legend_handles(policies: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=_policy_color(policy),
            linestyle=_policy_linestyle(policy),
            linewidth=_policy_linewidth(policy),
            label=_policy_label(policy),
        )
        for policy in policies
    ]


def _add_side_policy_legend(figure: plt.Figure, policies: list[str]) -> None:
    legend = figure.legend(
        handles=_policy_legend_handles(policies),
        loc="center left",
        bbox_to_anchor=(0.805, 0.5),
        title="Policy",
        frameon=True,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.8,
    )
    legend.get_frame().set_edgecolor("#d0d7de")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)


def _apply_robust_symmetric_ylim(axis: plt.Axes, values: pd.Series, quantile: float = 0.98) -> None:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    limit = float(finite.abs().quantile(float(quantile)))
    if not np.isfinite(limit) or limit <= 0.0:
        return
    limit *= 1.08
    axis.set_ylim(-limit, limit)
    axis.text(
        0.01,
        0.98,
        f"y clipped at +/-{limit:.2g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#57606a",
        bbox={"facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.82, "pad": 2.0},
    )


def _apply_symmetric_ylim_cap(axis: plt.Axes, limit: float) -> None:
    if not np.isfinite(limit) or limit <= 0.0:
        return
    axis.set_ylim(-float(limit), float(limit))
    axis.text(
        0.01,
        0.98,
        f"y capped at +/-{limit:g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#57606a",
        bbox={"facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.82, "pad": 2.0},
    )


def _format_limit_for_filename(limit: float) -> str:
    return f"{float(limit):g}".replace("-", "neg").replace(".", "p")


def _filter_requested_policies(frame: pd.DataFrame, policies: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if policies is None:
        return frame
    return frame.loc[frame["policy"].isin(set(policies))].copy()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_zhang_eval_artifacts(
    artifact_dir: str | Path,
    *,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> Path:
    artifact_path = Path(artifact_dir)
    zhang_dir = _zhang_eval_dir(artifact_path)
    required_paths = (
        zhang_dir / "summary_scaled.csv",
        zhang_dir / "summary_unscaled.csv",
        zhang_dir / "daily_portfolios_scaled.csv",
        zhang_dir / "contract_metrics.csv",
        zhang_dir / "cost_sweep.csv",
        zhang_dir / "config.json",
    )
    should_refresh = target_vol is not None or any(not path.exists() for path in required_paths)
    if not should_refresh:
        config_payload = _load_json(zhang_dir / "config.json")
        should_refresh = config_payload.get("return_column") != return_column

    if should_refresh:
        run_zhang_evaluation(
            artifact_dir=artifact_path,
            config=ZhangEvalConfig(
                target_vol=target_vol if target_vol is not None else float(_load_json(zhang_dir / "config.json").get("target_vol")) if (zhang_dir / "config.json").exists() else ZhangEvalConfig().target_vol,
                return_column=return_column,
            ),
        )
    return zhang_dir


def load_daily_return_reports(
    artifact_dir: str | Path,
    split: str,
    policies: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    reports_dir = _reports_dir(artifact_dir)
    frames: list[pd.DataFrame] = []
    requested = set(policies) if policies else None
    for path in sorted(reports_dir.glob(f"*_{split}_daily_returns.csv")):
        policy = path.name[: -len(f"_{split}_daily_returns.csv")]
        if requested is not None and policy not in requested:
            continue
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["policy"] = policy
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no daily return reports found for split={split}")
    combined = pd.concat(frames, ignore_index=True)
    combined["policy_label"] = combined["policy"].map(_policy_label)
    return combined.sort_values(["policy", "date"]).reset_index(drop=True)


def load_zhang_portfolio_reports(
    artifact_dir: str | Path,
    split: str,
    *,
    scaled: bool = True,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> pd.DataFrame:
    zhang_dir = ensure_zhang_eval_artifacts(
        artifact_dir=artifact_dir,
        target_vol=target_vol,
        return_column=return_column,
    )
    filename = "daily_portfolios_scaled.csv" if scaled else "daily_portfolios_unscaled.csv"
    frame = pd.read_csv(zhang_dir / filename)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["split"] == split].copy()
    frame = _filter_requested_policies(frame, policies)
    if frame.empty:
        raise FileNotFoundError(f"no Zhang portfolio reports found for split={split}")
    frame["policy_label"] = frame["policy"].map(_policy_label)
    return frame.sort_values(["asset_group", "policy", "date"]).reset_index(drop=True)


def load_zhang_contract_metrics(
    artifact_dir: str | Path,
    split: str,
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> pd.DataFrame:
    zhang_dir = ensure_zhang_eval_artifacts(
        artifact_dir=artifact_dir,
        target_vol=target_vol,
        return_column=return_column,
    )
    frame = pd.read_csv(zhang_dir / "contract_metrics.csv")
    frame = frame.loc[frame["split"] == split].copy()
    frame = _filter_requested_policies(frame, policies)
    if frame.empty:
        raise FileNotFoundError(f"no Zhang contract metrics found for split={split}")
    frame["policy_label"] = frame["policy"].map(_policy_label)
    return frame.sort_values(["asset_group", "policy", "symbol"]).reset_index(drop=True)


def load_zhang_cost_sweep(
    artifact_dir: str | Path,
    split: str,
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> pd.DataFrame:
    zhang_dir = ensure_zhang_eval_artifacts(
        artifact_dir=artifact_dir,
        target_vol=target_vol,
        return_column=return_column,
    )
    frame = pd.read_csv(zhang_dir / "cost_sweep.csv")
    frame = frame.loc[frame["split"] == split].copy()
    frame = _filter_requested_policies(frame, policies)
    if frame.empty:
        raise FileNotFoundError(f"no Zhang cost-sweep reports found for split={split}")
    frame["policy_label"] = frame["policy"].map(_policy_label)
    return frame.sort_values(["asset_group", "cost_rate_bp", "policy"]).reset_index(drop=True)


def compute_symbol_diagnostics(
    artifact_dir: str | Path,
    split: str,
    policies: list[str] | tuple[str, ...] | None = None,
    return_column: str = "zhang_return",
) -> pd.DataFrame:
    reports_dir = _reports_dir(artifact_dir)
    rows: list[dict[str, object]] = []
    requested = set(policies) if policies else None
    for path in sorted(reports_dir.glob(f"*_{split}_trades.csv")):
        policy = path.name[: -len(f"_{split}_trades.csv")]
        if requested is not None and policy not in requested:
            continue
        trades = pd.read_csv(path)
        if return_column not in trades.columns:
            raise ValueError(f"{path.name} does not contain return column {return_column}")
        for symbol, symbol_frame in trades.groupby("symbol", sort=False):
            turnover = symbol_frame["turnover"].fillna(0.0)
            returns = symbol_frame[return_column].fillna(0.0)
            metrics = compute_performance_metrics(returns, turnover=turnover)
            total_turnover = float(turnover.sum())
            avg_return_per_turnover = float(returns.sum() / total_turnover) if total_turnover > 0 else 0.0
            rows.append(
                {
                    "policy": policy,
                    "policy_label": _policy_label(policy),
                    "symbol": symbol,
                    "sharpe": float(metrics["sharpe"]),
                    "annualized_return": float(metrics["annualized_return"]),
                    "avg_return_per_turnover": avg_return_per_turnover,
                    "total_turnover": total_turnover,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no trade reports found for split={split}")
    diagnostics = pd.DataFrame(rows)
    return diagnostics.sort_values(["policy", "symbol"]).reset_index(drop=True)


def _subplot_grid(panel_count: int, *, ncols: int, panel_width: float, panel_height: float):
    ncols = max(1, min(ncols, panel_count))
    nrows = math.ceil(panel_count / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
    )
    flat_axes = axes.flatten()
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    return figure, flat_axes[:panel_count]


def plot_cumulative_returns(
    artifact_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    policies: list[str] | tuple[str, ...] | None = None,
) -> Path:
    frame = load_daily_return_reports(artifact_dir=artifact_dir, split=split, policies=policies)
    plot_path = Path(output_path) if output_path is not None else _plot_dir(artifact_dir) / f"cumulative_returns_{split}.png"

    figure, axis = plt.subplots(figsize=(11, 6))
    ordered_policies = _ordered_policies(frame)
    for policy in ordered_policies:
        policy_frame = frame.loc[frame["policy"] == policy].copy()
        policy_frame["cumulative_return"] = (1.0 + policy_frame["portfolio_return"].fillna(0.0)).cumprod()
        axis.plot(
            policy_frame["date"],
            policy_frame["cumulative_return"],
            label=_policy_label(policy),
            color=_policy_color(policy),
            linestyle=_policy_linestyle(policy),
            linewidth=_policy_linewidth(policy),
        )

    axis.set_title(f"Cumulative Portfolio Returns ({split})")
    axis.set_xlabel("Date")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.25)
    _add_side_policy_legend(figure, ordered_policies)
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def plot_zhang_cumulative_trade_returns(
    artifact_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    scaled: bool = True,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
    robust_ylim: bool = False,
    robust_quantile: float = 0.98,
    ylim_cap: float | None = None,
) -> Path:
    frame = load_zhang_portfolio_reports(
        artifact_dir=artifact_dir,
        split=split,
        scaled=scaled,
        policies=policies,
        target_vol=target_vol,
        return_column=return_column,
    )
    suffix = f"_cap_{_format_limit_for_filename(ylim_cap)}" if ylim_cap is not None else ("_robust" if robust_ylim else "")
    plot_path = (
        Path(output_path)
        if output_path is not None
        else _plot_dir(artifact_dir) / f"zhang_cumulative_trade_returns_{split}{suffix}.png"
    )
    asset_groups = sorted(frame["asset_group"].unique().tolist(), key=_asset_group_sort_key)
    figure, axes = _subplot_grid(
        len(asset_groups),
        ncols=1 if len(asset_groups) <= 2 else 2,
        panel_width=11.0,
        panel_height=3.6,
    )
    figure.set_size_inches(14.5, figure.get_figheight())
    ordered_policies = _ordered_policies(frame)

    for axis, asset_group in zip(axes, asset_groups):
        group_frame = frame.loc[frame["asset_group"] == asset_group]
        for policy in ordered_policies:
            policy_frame = group_frame.loc[group_frame["policy"] == policy]
            if policy_frame.empty:
                continue
            axis.plot(
                policy_frame["date"],
                policy_frame["cumulative_trade_return"],
                label=_policy_label(policy),
                color=_policy_color(policy),
                linestyle=_policy_linestyle(policy),
                linewidth=_policy_linewidth(policy),
            )
        axis.axhline(0.0, color="#adb5bd", linewidth=0.8)
        axis.set_title(f"{display_asset_group(asset_group)} ({'Scaled' if scaled else 'Unscaled'})")
        axis.set_xlabel("Date")
        axis.set_ylabel("Cumulative Trade Return")
        axis.grid(alpha=0.25)
        if ylim_cap is not None:
            _apply_symmetric_ylim_cap(axis, ylim_cap)
        elif robust_ylim:
            _apply_robust_symmetric_ylim(
                axis,
                group_frame["cumulative_trade_return"],
                quantile=robust_quantile,
            )

    _add_side_policy_legend(figure, ordered_policies)
    if ylim_cap is not None:
        title_suffix = f" - Y-Limits Capped at +/-{ylim_cap:g}"
    else:
        title_suffix = " - Robust Y-Limits" if robust_ylim else ""
    figure.suptitle(f"Zhang-Aligned Cumulative Trade Returns ({split}){title_suffix}", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 0.96))
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def plot_symbol_diagnostics(
    artifact_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    policies: list[str] | tuple[str, ...] | None = None,
    return_column: str = "zhang_return",
) -> Path:
    diagnostics = compute_symbol_diagnostics(
        artifact_dir=artifact_dir,
        split=split,
        policies=policies,
        return_column=return_column,
    )
    plot_path = Path(output_path) if output_path is not None else _plot_dir(artifact_dir) / f"symbol_diagnostics_{split}.png"

    ordered_policies = _ordered_policies(diagnostics)
    positions = list(range(1, len(ordered_policies) + 1))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panels = [
        ("sharpe", "Per-Symbol Sharpe"),
        ("avg_return_per_turnover", "Per-Symbol Avg Return per Turnover"),
    ]
    for axis, (metric, title) in zip(axes, panels):
        series = [
            diagnostics.loc[diagnostics["policy"] == policy, metric].tolist()
            for policy in ordered_policies
        ]
        boxplot = axis.boxplot(series, positions=positions, patch_artist=True, widths=0.6)
        for patch, policy in zip(boxplot["boxes"], ordered_policies):
            patch.set_facecolor(_policy_color(policy))
            patch.set_alpha(0.55)
        axis.set_title(f"{title} ({split})")
        axis.set_xticks(positions)
        axis.set_xticklabels([_policy_label(policy) for policy in ordered_policies], rotation=15)
        axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return plot_path


def plot_zhang_contract_diagnostics(
    artifact_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> Path:
    diagnostics = load_zhang_contract_metrics(
        artifact_dir=artifact_dir,
        split=split,
        policies=policies,
        target_vol=target_vol,
        return_column=return_column,
    )
    plot_path = Path(output_path) if output_path is not None else _plot_dir(artifact_dir) / f"zhang_contract_diagnostics_{split}.png"
    asset_groups = sorted(diagnostics["asset_group"].unique().tolist(), key=_asset_group_sort_key)
    ordered_policies = _ordered_policies(diagnostics)
    positions = list(range(1, len(ordered_policies) + 1))
    figure, axes = plt.subplots(
        len(asset_groups),
        2,
        figsize=(12.5, max(4.5, 4.0 * len(asset_groups))),
        squeeze=False,
    )
    metrics = (
        ("sharpe", "Annualized Sharpe"),
        ("avg_return_per_turnover", "Avg Trade Return per Turnover"),
    )
    for row_index, asset_group in enumerate(asset_groups):
        group_frame = diagnostics.loc[diagnostics["asset_group"] == asset_group]
        for axis, (metric, title) in zip(axes[row_index], metrics):
            series = [
                group_frame.loc[group_frame["policy"] == policy, metric].tolist()
                for policy in ordered_policies
            ]
            boxplot = axis.boxplot(series, positions=positions, patch_artist=True, widths=0.6)
            for patch, policy in zip(boxplot["boxes"], ordered_policies):
                patch.set_facecolor(_policy_color(policy))
                patch.set_alpha(0.55)
            axis.set_title(f"{display_asset_group(asset_group)}: {title}")
            axis.set_xticks(positions)
            axis.set_xticklabels([_policy_label(policy) for policy in ordered_policies], rotation=15)
            axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def plot_zhang_cost_sweep(
    artifact_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
) -> Path:
    frame = load_zhang_cost_sweep(
        artifact_dir=artifact_dir,
        split=split,
        policies=policies,
        target_vol=target_vol,
        return_column=return_column,
    )
    plot_path = Path(output_path) if output_path is not None else _plot_dir(artifact_dir) / f"zhang_cost_sweep_{split}.png"
    asset_groups = sorted(frame["asset_group"].unique().tolist(), key=_asset_group_sort_key)
    figure, axes = plt.subplots(
        len(asset_groups),
        2,
        figsize=(12.5, max(4.5, 4.0 * len(asset_groups))),
        squeeze=False,
    )
    figure.set_size_inches(15.0, figure.get_figheight())
    ordered_policies = _ordered_policies(frame)
    metrics = (
        ("sharpe", "Sharpe"),
        ("avg_cost_per_contract", "Average Cost per Contract"),
    )
    for row_index, asset_group in enumerate(asset_groups):
        group_frame = frame.loc[frame["asset_group"] == asset_group]
        for axis, (metric, title) in zip(axes[row_index], metrics):
            for policy in ordered_policies:
                policy_frame = group_frame.loc[group_frame["policy"] == policy]
                if policy_frame.empty:
                    continue
                axis.plot(
                    policy_frame["cost_rate_bp"],
                    policy_frame[metric],
                    label=_policy_label(policy),
                    color=_policy_color(policy),
                    linestyle=_policy_linestyle(policy),
                    linewidth=_policy_linewidth(policy),
                )
            axis.set_title(f"{display_asset_group(asset_group)}: {title}")
            axis.set_xlabel("Transaction Cost (bp)")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)

    _add_side_policy_legend(figure, ordered_policies)
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def generate_legacy_plots(
    artifact_dir: str | Path,
    split: str = "test",
    policies: list[str] | tuple[str, ...] | None = None,
    return_column: str = "zhang_return",
) -> PlotArtifacts:
    return PlotArtifacts(
        cumulative_returns_path=plot_cumulative_returns(
            artifact_dir=artifact_dir,
            split=split,
            policies=policies,
        ),
        symbol_diagnostics_path=plot_symbol_diagnostics(
            artifact_dir=artifact_dir,
            split=split,
            policies=policies,
            return_column=return_column,
        ),
        style="legacy",
    )


def generate_zhang_style_plots(
    artifact_dir: str | Path,
    split: str = "test",
    *,
    policies: list[str] | tuple[str, ...] | None = None,
    target_vol: float | None = None,
    return_column: str = "zhang_return",
    include_cost_sweep: bool = True,
) -> PlotArtifacts:
    cumulative_path = plot_zhang_cumulative_trade_returns(
        artifact_dir=artifact_dir,
        split=split,
        policies=policies,
        scaled=True,
        target_vol=target_vol,
        return_column=return_column,
    )
    diagnostics_path = plot_zhang_contract_diagnostics(
        artifact_dir=artifact_dir,
        split=split,
        policies=policies,
        target_vol=target_vol,
        return_column=return_column,
    )
    cost_sweep_path = (
        plot_zhang_cost_sweep(
            artifact_dir=artifact_dir,
            split=split,
            policies=policies,
            target_vol=target_vol,
            return_column=return_column,
        )
        if include_cost_sweep
        else None
    )
    return PlotArtifacts(
        cumulative_returns_path=cumulative_path,
        symbol_diagnostics_path=diagnostics_path,
        cost_sweep_path=cost_sweep_path,
        style="zhang",
    )


def generate_paper_style_plots(
    artifact_dir: str | Path,
    split: str = "test",
    policies: list[str] | tuple[str, ...] | None = None,
    return_column: str = "zhang_return",
    *,
    style: str = "zhang",
    target_vol: float | None = None,
    include_cost_sweep: bool = True,
) -> PlotArtifacts:
    if style == "legacy":
        return generate_legacy_plots(
            artifact_dir=artifact_dir,
            split=split,
            policies=policies,
            return_column=return_column,
        )
    if style == "zhang":
        return generate_zhang_style_plots(
            artifact_dir=artifact_dir,
            split=split,
            policies=policies,
            target_vol=target_vol,
            return_column=return_column,
            include_cost_sweep=include_cost_sweep,
        )
    raise ValueError(f"unsupported plot style {style}")


generate_rl_plots = generate_paper_style_plots
