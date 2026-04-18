"""Quick smoke test: download 3 symbols, compute features, plot."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from rl_trading.data import fetch_bars
from rl_trading.env import EnvConfig, STATE_DIM, TradingEnv
from rl_trading.features import FEATURE_COLS, FeatureBuilder


def main() -> None:
    symbols = ["SPY", "GLD", "TLT"]
    bars = fetch_bars(symbols, start="2004-01-01", end="2025-12-31")
    print(f"Bars shape: {bars.shape}")
    print(f"Bars columns: {list(bars.columns)}")
    print(bars.head())

    fb = FeatureBuilder()
    feat = fb.transform(bars)
    print(f"\nFeature frame shape: {feat.shape}")
    print(f"Feature columns: {list(feat.columns)}")

    ready = feat[feat["window_ready"]]
    print(f"\nRows with window_ready=True: {len(ready)} / {len(feat)}")
    print(ready[["date", "symbol"] + FEATURE_COLS].head(10))

    # Plot features for SPY
    spy = ready[ready["symbol"] == "SPY"].set_index("date")
    fig, axes = plt.subplots(len(FEATURE_COLS), 1, figsize=(12, 2.5 * len(FEATURE_COLS)), sharex=True)
    for ax, col in zip(axes, FEATURE_COLS):
        ax.plot(spy.index, spy[col], linewidth=0.7)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("SPY – Zhang features")
    fig.tight_layout()
    fig.savefig("smoke_features.png", dpi=100)
    print("\nPlot saved to smoke_features.png")

    # ── Env smoke test ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TradingEnv smoke test (SPY, train split)")
    print("=" * 60)

    cfg = EnvConfig(action_mode="discrete", seq_len=1)
    env = TradingEnv(feat, cfg)

    # Random-action rollout
    rng = np.random.default_rng(42)
    state = env.reset("SPY", "train")
    total_reward = 0.0
    trades = 0
    prev_action = 1  # flat
    for _ in range(100):
        action = rng.integers(0, 3)
        if action != prev_action:
            trades += 1
        prev_action = action
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    steps = len(env.history["reward"])
    print(f"\n[Random policy]  steps={steps}  total_reward={total_reward:.6f}  "
          f"trades={trades}  avg_reward={total_reward / max(steps, 1):.6f}")

    # Long-only rollout
    state = env.reset("SPY", "train")
    total_reward = 0.0
    for i in range(100):
        state, reward, done, info = env.step(2)  # long
        total_reward += reward
        if done:
            break
    steps = len(env.history["reward"])
    print(f"[Long-only]      steps={steps}  total_reward={total_reward:.6f}  "
          f"trades=1  avg_reward={total_reward / max(steps, 1):.6f}")


def smoke_rl() -> None:
    """Quick RL smoke test: 2 epochs per agent on SPY."""
    from rl_trading.agents import A2CAgent, DQNAgent, PGAgent
    from rl_trading.trainer import Trainer, TrainerConfig

    symbols = ["SPY"]
    bars = fetch_bars(symbols, start="2004-01-01", end="2025-12-31")
    fb = FeatureBuilder()
    feat = fb.transform(bars)

    tcfg = TrainerConfig(n_epochs=2, patience=5, checkpoint_dir="/tmp/rl_smoke_ckpt")

    # DQN (discrete, seq_len=60 for LSTM)
    print("\n" + "=" * 60)
    print("DQN smoke test")
    print("=" * 60)
    env_d = TradingEnv(feat, EnvConfig(action_mode="discrete", seq_len=60))
    dqn = DQNAgent(n_features=STATE_DIM)
    trainer = Trainer(dqn, env_d, "SPY", tcfg)
    result = trainer.train()
    print(f"DQN result: {result}")

    # PG (discrete, seq_len=60)
    print("\n" + "=" * 60)
    print("PG (REINFORCE) smoke test")
    print("=" * 60)
    env_p = TradingEnv(feat, EnvConfig(action_mode="discrete", seq_len=60))
    pg = PGAgent(n_features=STATE_DIM)
    trainer = Trainer(pg, env_p, "SPY", tcfg)
    result = trainer.train()
    print(f"PG result: {result}")

    # A2C (continuous, seq_len=60)
    print("\n" + "=" * 60)
    print("A2C smoke test")
    print("=" * 60)
    env_a = TradingEnv(feat, EnvConfig(action_mode="continuous", seq_len=60))
    a2c = A2CAgent(n_features=STATE_DIM)
    trainer = Trainer(a2c, env_a, "SPY", tcfg)
    result = trainer.train()
    print(f"A2C result: {result}")


if __name__ == "__main__":
    main()
    smoke_rl()
