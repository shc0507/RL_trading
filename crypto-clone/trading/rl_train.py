"""Minimal direct RL training on top of TradingEnv."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .backtest import Backtester, EvalReport
from .config import DEFAULT_OUTPUT_DIR
from .env import EnvironmentConfig, TradingEnv
from .policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy


ACTION_VALUES = (-1.0, 0.0, 1.0)


@dataclass(slots=True)
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    episodes: int = 500


@dataclass(slots=True)
class QTablePolicy:
    q_tables: dict[str, dict[tuple[tuple[int, ...], int], float]]
    name: str = "q_learning"

    def act(self, observation: dict[str, object]) -> float:
        symbol = str(observation["symbol"])
        state = discretize_observation(observation)
        q_table = self.q_tables.get(symbol, {})
        action_index = choose_greedy_action(q_table, state)
        return ACTION_VALUES[action_index]


def bucketize(value: float, threshold: float = 0.0) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def discretize_observation(observation: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    """Map the rich observation into a small tabular state.

    We still keep the state discrete for tabular Q-learning, but we use
    more of the existing engineered features:
    - short-horizon vol-normalized return regime
    - long-horizon raw return regime
    - normalized price regime
    - macd regime
    - rsi regime
    - current position bucket
    """

    row = observation["row"]
    ret_21_vol = float(row.get("ret_21_vol", 0.0) or 0.0)
    ret_252_raw = float(row.get("ret_252_raw", 0.0) or 0.0)
    norm_close = float(row.get("norm_close", 0.0) or 0.0)
    macd_signal = float(row.get("macd_signal", 0.0) or 0.0)
    rsi_30 = float(row.get("rsi_30", 50.0) or 50.0)
    position = float(observation["position"])

    short_return_bucket = bucketize(ret_21_vol, threshold=0.25)
    long_return_bucket = bucketize(ret_252_raw, threshold=0.0)
    norm_close_bucket = bucketize(norm_close, threshold=0.5)
    macd_bucket = bucketize(macd_signal, threshold=0.1)
    rsi_bucket = 1 if rsi_30 > 55.0 else (-1 if rsi_30 < 45.0 else 0)
    position_bucket = int(position)
    return (
        short_return_bucket,
        long_return_bucket,
        norm_close_bucket,
        macd_bucket,
        rsi_bucket,
        position_bucket,
    )


def choose_action(
    q_table: dict[tuple[tuple[int, ...], int], float],
    state: tuple[int, ...],
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randrange(len(ACTION_VALUES))

    q_values = [q_table.get((state, action_index), 0.0) for action_index in range(len(ACTION_VALUES))]
    return int(np.argmax(q_values))


def choose_greedy_action(
    q_table: dict[tuple[tuple[int, ...], int], float],
    state: tuple[int, ...],
) -> int:
    q_values = [q_table.get((state, action_index), 0.0) for action_index in range(len(ACTION_VALUES))]
    return int(np.argmax(q_values))


def train_q_learning(
    feature_frame: pd.DataFrame,
    split: str = "train",
    symbols: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    config: QLearningConfig | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[tuple[tuple[int, ...], int], float]], list[dict[str, object]]]:
    training_config = config or QLearningConfig()
    trading_env = TradingEnv(
        feature_frame=feature_frame,
        config=env_config or EnvironmentConfig(action_mode="discrete", reward_mode="raw"),
        splits=splits,
    )
    symbol_list = symbols or sorted(feature_frame["symbol"].unique().tolist())
    q_tables: dict[str, dict[tuple[tuple[int, ...], int], float]] = {
        symbol: {} for symbol in symbol_list
    }
    episode_logs: list[dict[str, object]] = []

    for symbol in symbol_list:
        q_table = q_tables[symbol]
        for episode in range(training_config.episodes):
            observation = trading_env.reset(symbol=symbol, split=split)
            state = discretize_observation(observation)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action_index = choose_action(q_table, state, training_config.epsilon)
                next_observation, reward, done, info = trading_env.step(ACTION_VALUES[action_index])
                total_reward += reward
                steps += 1

                next_state = state if done or next_observation is None else discretize_observation(next_observation)
                best_next = max(q_table.get((next_state, idx), 0.0) for idx in range(len(ACTION_VALUES)))
                old_value = q_table.get((state, action_index), 0.0)
                updated_value = old_value + training_config.alpha * (
                    reward + training_config.gamma * best_next - old_value
                )
                q_table[(state, action_index)] = updated_value

                state = next_state

            episode_logs.append(
                {
                    "episode": episode,
                    "symbol": symbol,
                    "split": split,
                    "total_reward": float(total_reward),
                    "steps": steps,
                }
            )

    return q_tables, episode_logs


def summarize_training(episode_logs: list[dict[str, object]]) -> dict[str, float]:
    rewards = pd.Series([float(log["total_reward"]) for log in episode_logs], dtype=float)
    steps = pd.Series([float(log["steps"]) for log in episode_logs], dtype=float)
    return {
        "episodes": float(len(episode_logs)),
        "avg_episode_reward": float(rewards.mean()) if not rewards.empty else 0.0,
        "best_episode_reward": float(rewards.max()) if not rewards.empty else 0.0,
        "worst_episode_reward": float(rewards.min()) if not rewards.empty else 0.0,
        "avg_episode_steps": float(steps.mean()) if not steps.empty else 0.0,
    }


def compare_policies(
    feature_frame: pd.DataFrame,
    learned_policy: QTablePolicy,
    split: str,
    symbols: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, EvalReport]]:
    backtester = Backtester(
        feature_frame=feature_frame,
        env_config=env_config or EnvironmentConfig(action_mode="discrete", reward_mode="raw"),
        splits=splits,
    )
    policies = [learned_policy, LongOnlyPolicy(), Sign12MPolicy(), MACDPolicy()]
    comparison_rows: list[dict[str, float | str]] = []
    reports: dict[str, EvalReport] = {}

    for policy in policies:
        report = backtester.run(policy=policy, split=split, symbols=symbols)
        reports[policy.name] = report
        comparison_rows.append({"policy": policy.name, "split": split, **report.portfolio_metrics})

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(["annualized_return", "sharpe"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison, reports


def run_q_learning_experiment(
    feature_frame: pd.DataFrame,
    train_split: str = "train",
    eval_split: str = "test",
    symbols: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    config: QLearningConfig | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    experiment_env_config = env_config or EnvironmentConfig(action_mode="discrete", reward_mode="raw")
    q_tables, episode_logs = train_q_learning(
        feature_frame=feature_frame,
        split=train_split,
        symbols=symbols,
        env_config=experiment_env_config,
        config=config,
        splits=splits,
    )
    learned_policy = QTablePolicy(q_tables=q_tables)
    comparison, reports = compare_policies(
        feature_frame=feature_frame,
        learned_policy=learned_policy,
        split=eval_split,
        symbols=symbols,
        env_config=experiment_env_config,
        splits=splits,
    )
    return {
        "q_table": q_tables,
        "episode_logs": episode_logs,
        "training_summary": summarize_training(episode_logs),
        "comparison": comparison,
        "reports": reports,
        "policy": learned_policy,
    }


def save_experiment_outputs(
    experiment: dict[str, object],
    comparison_path: str | Path,
    episode_log_path: str | Path | None = None,
) -> tuple[Path, Path]:
    comparison_output = Path(comparison_path)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison = experiment["comparison"]
    comparison.to_csv(comparison_output, index=False)

    logs_output = Path(episode_log_path) if episode_log_path is not None else comparison_output.with_name(
        comparison_output.stem + "_episodes.csv"
    )
    logs_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiment["episode_logs"]).to_csv(logs_output, index=False)
    return comparison_output, logs_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny tabular Q-learning agent and compare it against baseline policies."
    )
    parser.add_argument("--features-path", default=str(DEFAULT_OUTPUT_DIR / "processed" / "features.csv"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--comparison-path", default=None)
    parser.add_argument("--episode-log-path", default=None)
    args = parser.parse_args()

    feature_frame = pd.read_csv(args.features_path)
    experiment = run_q_learning_experiment(
        feature_frame=feature_frame,
        train_split=args.train_split,
        eval_split=args.eval_split,
        config=QLearningConfig(episodes=args.episodes),
    )
    q_tables = experiment["q_table"]
    episode_logs = experiment["episode_logs"]
    training_summary = experiment["training_summary"]
    comparison = experiment["comparison"]

    comparison_path = args.comparison_path or str(DEFAULT_OUTPUT_DIR / "qa" / f"q_learning_{args.eval_split}.csv")
    saved_comparison_path, saved_episode_path = save_experiment_outputs(
        experiment,
        comparison_path=comparison_path,
        episode_log_path=args.episode_log_path,
    )

    total_states = sum(len({state for state, _ in table}) for table in q_tables.values())
    total_q_entries = sum(len(table) for table in q_tables.values())
    print(f"trained_symbols={len(q_tables)}")
    print(f"episodes_per_symbol={args.episodes}")
    print(f"trained_states={total_states}")
    print(f"q_entries={total_q_entries}")
    print("training_summary:")
    print(training_summary)
    print("last_5_episodes:")
    for row in episode_logs[-5:]:
        print(row)
    print("comparison:")
    print(comparison.to_string(index=False))
    print(f"saved_comparison={saved_comparison_path}")
    print(f"saved_episode_logs={saved_episode_path}")


if __name__ == "__main__":
    main()
