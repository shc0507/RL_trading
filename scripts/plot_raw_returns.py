"""Plot true cumulative return (%) for each strategy on GLD test window.

For each strategy with position series w_t and simple daily return r_t:
    wealth_t = prod_{s<t} (1 + w_s * r_s)
    cumulative_return_t = wealth_t - 1  (plotted as %)

This is the real "how much richer did $1 get" number — compounded, dollar-scale,
no vol-targeting or Zhang-reward transform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_trading.agents import A2CAgent, DQNAgent, PGAgent
from rl_trading.baselines import long_only, macd_signal, sign_r
from rl_trading.config import TEST_END, TEST_START
from rl_trading.data import fetch_bars
from rl_trading.env import STATE_DIM, EnvConfig, TradingEnv
from rl_trading.features import FeatureBuilder
from rl_trading.trainer import Trainer, TrainerConfig


SYMBOL = "GLD"
EPOCHS = 5
OUT = Path("artifacts/cumulative_return_pct.png")


def train_and_eval(agent_name, agent, env_cfg, feat):
    env = TradingEnv(feat, env_cfg)
    tcfg = TrainerConfig(
        n_epochs=EPOCHS, patience=EPOCHS,
        checkpoint_dir=f"artifacts/checkpoints/raw_{agent_name}",
    )
    Trainer(agent, env, [SYMBOL], tcfg, label=SYMBOL).train()

    state = env.reset(SYMBOL, "test")
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, _, done, _ = env.step(action)
        if next_state is not None:
            state = next_state
    return pd.DataFrame(env.history)


def baseline_positions_and_returns(feat, fn, start, end):
    """Return (position, simple_return) arrays aligned day-by-day."""
    positions = fn(feat, SYMBOL, start=start, end=end)
    sub = feat[(feat["symbol"] == SYMBOL)
               & (feat["date"] >= start) & (feat["date"] <= end)].copy()
    sub = sub.set_index("date")
    # r_t = (p_{t+1} - p_t) / p_t, earned by position held on day t
    simple_ret = sub["close"].pct_change().shift(-1).fillna(0.0)
    pos = pd.Series(positions, index=sub.index).reindex(sub.index).fillna(0.0)
    return pos.to_numpy(), simple_ret.to_numpy()


def compounded_cum_return(position, simple_return):
    """wealth_t = prod(1 + w_s * r_s); returns cumulative_return_t in %."""
    gross = 1.0 + position * simple_return
    # guard against wipe-outs
    gross = np.clip(gross, 1e-8, None)
    wealth = np.cumprod(gross)
    return (wealth - 1.0) * 100.0


def main():
    print("Fetching data ...")
    bars = fetch_bars([SYMBOL], start="2003-01-01", end=TEST_END)
    feat = FeatureBuilder().transform(bars)

    results = {}  # method -> cumulative_return_pct array

    # Baselines
    for name, fn in [("Long", long_only), ("Sign(R)", sign_r), ("MACD", macd_signal)]:
        print(f"Baseline {name} ...")
        pos, ret = baseline_positions_and_returns(feat, fn, TEST_START, TEST_END)
        results[name] = compounded_cum_return(pos, ret)

    # RL agents
    device = "mps"
    specs = [
        ("DQN", DQNAgent(n_features=STATE_DIM, device=device),
         EnvConfig(action_mode="discrete", seq_len=60)),
        ("PG",  PGAgent(n_features=STATE_DIM, device=device),
         EnvConfig(action_mode="discrete", seq_len=60)),
        ("A2C", A2CAgent(n_features=STATE_DIM, device=device),
         EnvConfig(action_mode="continuous", seq_len=60)),
    ]

    for name, agent, env_cfg in specs:
        print(f"Training + eval {name} ...")
        hist = train_and_eval(name.lower(), agent, env_cfg, feat)
        prices = hist["price"].to_numpy()
        # Simple return earned by position on day t: r_t = p_{t+1}/p_t - 1
        simple_ret = np.zeros_like(prices, dtype=float)
        simple_ret[:-1] = prices[1:] / prices[:-1] - 1.0
        pos = hist["position"].to_numpy()
        results[name] = compounded_cum_return(pos, simple_ret)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name, cr in results.items():
        ax.plot(cr, label=f"{name} (final={cr[-1]:+.0f}%)", linewidth=1.2)

    ax.set_title(f"Cumulative Return (%) — {SYMBOL}, test window {TEST_START}–{TEST_END}")
    ax.set_ylabel("Cumulative Return (%)  —  wealth_t / wealth_0 - 1")
    ax.set_xlabel("Trading Day")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
