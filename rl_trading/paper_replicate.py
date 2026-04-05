"""Paper-style DRL methodology replication on top of TradingEnv.

This module aims to reproduce the methodology from
"Deep Reinforcement Learning for Trading" at a repo-friendly scale:

- full feature windows as state input
- Zhang-style volatility-scaled reward
- LSTM-based sequence encoder
- three algorithms:
  - DQN (discrete actions)
  - PG / REINFORCE (discrete actions)
  - A2C (continuous actions)

It does not try to fully reproduce the paper's futures universe or rolling
retraining protocol yet. The goal is to align the environment, reward, state,
action space, and model families first.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .backtest import Backtester, EvalReport
from .config import DEFAULT_COST_RATE_BP, DEFAULT_OUTPUT_DIR, DEFAULT_VOL_TARGET, OBSERVATION_WINDOW
from .env import EnvironmentConfig, TradingEnv
from .policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy


DISCRETE_ACTIONS = (-1.0, 0.0, 1.0)


@dataclass(slots=True)
class PaperRLConfig:
    algo: Literal["dqn", "pg", "a2c"] = "dqn"
    episodes: int = 40
    gamma: float = 0.3
    batch_size: int = 64
    replay_capacity: int = 5_000
    warmup_steps: int = 128
    target_update_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    entropy_weight: float = 1e-3
    value_weight: float = 0.5
    grad_clip: float = 1.0
    seed: int = 7
    hidden_one: int = 64
    hidden_two: int = 32

    def __post_init__(self) -> None:
        if self.algo == "a2c":
            self.batch_size = max(self.batch_size, 128)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("cpu")


def observation_to_tensor(observation: dict[str, object], device_name: torch.device) -> torch.Tensor:
    window = np.asarray(observation["window"], dtype=np.float32)
    if window.ndim != 2:
        raise ValueError(f"expected 2D observation window, got shape {window.shape}")
    if window.shape[0] < OBSERVATION_WINDOW:
        pad = np.zeros((OBSERVATION_WINDOW - window.shape[0], window.shape[1]), dtype=np.float32)
        window = np.vstack([pad, window])
    elif window.shape[0] > OBSERVATION_WINDOW:
        window = window[-OBSERVATION_WINDOW:]
    return torch.from_numpy(window).unsqueeze(0).to(device_name)


class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_one: int = 64, hidden_two: int = 32) -> None:
        super().__init__()
        self.lstm_one = nn.LSTM(input_size=input_size, hidden_size=hidden_one, batch_first=True)
        self.lstm_two = nn.LSTM(input_size=hidden_one, hidden_size=hidden_two, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm_one(x)
        out, _ = self.lstm_two(out)
        return out[:, -1, :]


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_one: int = 64, hidden_two: int = 32) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(input_size=input_size, hidden_one=hidden_one, hidden_two=hidden_two)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.value_head = nn.Linear(hidden_two, 1)
        self.advantage_head = nn.Linear(hidden_two, len(DISCRETE_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.encoder(x))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PolicyGradientNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_one: int = 64, hidden_two: int = 32) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(input_size=input_size, hidden_one=hidden_one, hidden_two=hidden_two)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.policy_head = nn.Linear(hidden_two, len(DISCRETE_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.encoder(x))
        return self.policy_head(features)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_one: int = 64, hidden_two: int = 32) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(input_size=input_size, hidden_one=hidden_one, hidden_two=hidden_two)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.actor_mean = nn.Linear(hidden_two, 1)
        self.critic = nn.Linear(hidden_two, 1)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.activation(self.encoder(x))
        mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value


class DQNPolicy:
    name = "paper_dqn"

    def __init__(self, network: DQNNetwork, device_name: torch.device) -> None:
        self.network = network
        self.device = device_name
        self.network.eval()

    def act(self, observation: dict[str, object]) -> float:
        with torch.no_grad():
            tensor = observation_to_tensor(observation, self.device)
            q_values = self.network(tensor)
            action_index = int(torch.argmax(q_values, dim=1).item())
        return DISCRETE_ACTIONS[action_index]


class PGPolicy:
    name = "paper_pg"

    def __init__(self, network: PolicyGradientNetwork, device_name: torch.device) -> None:
        self.network = network
        self.device = device_name
        self.network.eval()

    def act(self, observation: dict[str, object]) -> float:
        with torch.no_grad():
            tensor = observation_to_tensor(observation, self.device)
            logits = self.network(tensor)
            action_index = int(torch.argmax(logits, dim=1).item())
        return DISCRETE_ACTIONS[action_index]


class A2CPolicy:
    name = "paper_a2c"

    def __init__(self, network: ActorCriticNetwork, device_name: torch.device) -> None:
        self.network = network
        self.device = device_name
        self.network.eval()

    def act(self, observation: dict[str, object]) -> float:
        with torch.no_grad():
            tensor = observation_to_tensor(observation, self.device)
            mean, _, _ = self.network(tensor)
        return float(mean.squeeze().clamp(-1.0, 1.0).item())


def discrete_env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        action_mode="discrete",
        reward_mode="zhang",
        cost_rate_bp=DEFAULT_COST_RATE_BP,
        vol_target=DEFAULT_VOL_TARGET,
    )


def continuous_env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        action_mode="continuous",
        reward_mode="zhang",
        cost_rate_bp=DEFAULT_COST_RATE_BP,
        vol_target=DEFAULT_VOL_TARGET,
    )


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor | None, bool]] = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        action_index: int,
        reward: float,
        next_state: torch.Tensor | None,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                state.detach().cpu().squeeze(0),
                action_index,
                reward,
                None if next_state is None else next_state.detach().cpu().squeeze(0),
                done,
            )
        )

    def sample(
        self, batch_size: int, device_name: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([item[0] for item in batch], dim=0).to(device_name)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device_name)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device_name)
        nonterminal_mask = torch.tensor([not item[4] for item in batch], dtype=torch.bool, device=device_name)
        next_states_list = [item[3] for item in batch]
        next_states = torch.zeros_like(states)
        for idx, next_state in enumerate(next_states_list):
            if next_state is not None:
                next_states[idx] = next_state.to(device_name)
        return states, actions, rewards, next_states, nonterminal_mask

    def __len__(self) -> int:
        return len(self.buffer)


def linear_epsilon(step: int, config: PaperRLConfig) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    fraction = step / max(config.epsilon_decay_steps, 1)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def infer_input_size(feature_frame: pd.DataFrame, env_config: EnvironmentConfig, splits: dict[str, tuple[str, str]], split: str) -> int:
    env = TradingEnv(feature_frame=feature_frame, config=env_config, splits=splits)
    symbol = sorted(feature_frame["symbol"].unique().tolist())[0]
    observation = env.reset(symbol=symbol, split=split)
    return int(np.asarray(observation["window"]).shape[1])


def train_dqn(
    feature_frame: pd.DataFrame,
    split: str,
    symbols: list[str] | None,
    config: PaperRLConfig,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[DQNPolicy, list[dict[str, object]]]:
    env_config = discrete_env_config()
    actual_splits = splits or TradingEnv(feature_frame, env_config).splits
    input_size = infer_input_size(feature_frame, env_config, actual_splits, split)
    dev = device()
    online = DQNNetwork(input_size=input_size, hidden_one=config.hidden_one, hidden_two=config.hidden_two).to(dev)
    target = DQNNetwork(input_size=input_size, hidden_one=config.hidden_one, hidden_two=config.hidden_two).to(dev)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)
    env = TradingEnv(feature_frame=feature_frame, config=env_config, splits=actual_splits)
    symbol_list = symbols or sorted(feature_frame["symbol"].unique().tolist())
    step_count = 0
    episode_logs: list[dict[str, object]] = []

    online.train()
    target.eval()
    for episode in range(config.episodes):
        symbol = symbol_list[episode % len(symbol_list)]
        observation = env.reset(symbol=symbol, split=split)
        state = observation_to_tensor(observation, dev)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            epsilon = linear_epsilon(step_count, config)
            if random.random() < epsilon:
                action_index = random.randrange(len(DISCRETE_ACTIONS))
            else:
                with torch.no_grad():
                    q_values = online(state)
                    action_index = int(torch.argmax(q_values, dim=1).item())
            next_observation, reward, done, _ = env.step(DISCRETE_ACTIONS[action_index])
            next_state = None if done or next_observation is None else observation_to_tensor(next_observation, dev)
            replay.add(state, action_index, reward, next_state, done)
            total_reward += float(reward)
            steps += 1
            step_count += 1

            if len(replay) >= max(config.warmup_steps, config.batch_size):
                states, actions, rewards, next_states, nonterminal_mask = replay.sample(config.batch_size, dev)
                q_values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online_actions = online(next_states).argmax(dim=1, keepdim=True)
                    next_target_q = target(next_states).gather(1, next_online_actions).squeeze(1)
                    targets = rewards + config.gamma * next_target_q * nonterminal_mask.float()
                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), config.grad_clip)
                optimizer.step()

                if step_count % config.target_update_steps == 0:
                    target.load_state_dict(online.state_dict())

            state = state if next_state is None else next_state

        episode_logs.append(
            {
                "episode": episode,
                "symbol": symbol,
                "split": split,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

    return DQNPolicy(online, dev), episode_logs


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    tensor = torch.tensor(returns, dtype=torch.float32)
    if len(tensor) > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-8)
    return tensor


def train_pg(
    feature_frame: pd.DataFrame,
    split: str,
    symbols: list[str] | None,
    config: PaperRLConfig,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[PGPolicy, list[dict[str, object]]]:
    env_config = discrete_env_config()
    actual_splits = splits or TradingEnv(feature_frame, env_config).splits
    input_size = infer_input_size(feature_frame, env_config, actual_splits, split)
    dev = device()
    network = PolicyGradientNetwork(input_size=input_size, hidden_one=config.hidden_one, hidden_two=config.hidden_two).to(dev)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    env = TradingEnv(feature_frame=feature_frame, config=env_config, splits=actual_splits)
    symbol_list = symbols or sorted(feature_frame["symbol"].unique().tolist())
    episode_logs: list[dict[str, object]] = []

    network.train()
    for episode in range(config.episodes):
        symbol = symbol_list[episode % len(symbol_list)]
        observation = env.reset(symbol=symbol, split=split)
        done = False
        total_reward = 0.0
        steps = 0
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        while not done:
            state = observation_to_tensor(observation, dev)
            logits = network(state)
            dist = Categorical(logits=logits)
            action_index = int(dist.sample().item())
            log_probs.append(dist.log_prob(torch.tensor(action_index, device=dev)))
            entropies.append(dist.entropy())
            next_observation, reward, done, _ = env.step(DISCRETE_ACTIONS[action_index])
            rewards.append(float(reward))
            total_reward += float(reward)
            steps += 1
            observation = next_observation if next_observation is not None else observation

        returns = discounted_returns(rewards, config.gamma).to(dev)
        policy_loss = torch.stack([-log_prob * ret for log_prob, ret in zip(log_probs, returns)]).sum()
        entropy_bonus = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=dev)
        loss = policy_loss - config.entropy_weight * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
        optimizer.step()

        episode_logs.append(
            {
                "episode": episode,
                "symbol": symbol,
                "split": split,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

    return PGPolicy(network, dev), episode_logs


def train_a2c(
    feature_frame: pd.DataFrame,
    split: str,
    symbols: list[str] | None,
    config: PaperRLConfig,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[A2CPolicy, list[dict[str, object]]]:
    env_config = continuous_env_config()
    actual_splits = splits or TradingEnv(feature_frame, env_config).splits
    input_size = infer_input_size(feature_frame, env_config, actual_splits, split)
    dev = device()
    network = ActorCriticNetwork(input_size=input_size, hidden_one=config.hidden_one, hidden_two=config.hidden_two).to(dev)
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(network.encoder.parameters()) + list(network.actor_mean.parameters()) + [network.log_std],
                "lr": config.actor_learning_rate,
            },
            {
                "params": network.critic.parameters(),
                "lr": config.critic_learning_rate,
            },
        ]
    )
    env = TradingEnv(feature_frame=feature_frame, config=env_config, splits=actual_splits)
    symbol_list = symbols or sorted(feature_frame["symbol"].unique().tolist())
    episode_logs: list[dict[str, object]] = []

    network.train()
    for episode in range(config.episodes):
        symbol = symbol_list[episode % len(symbol_list)]
        observation = env.reset(symbol=symbol, split=split)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            state = observation_to_tensor(observation, dev)
            mean, std, value = network(state)
            dist = Normal(mean, std)
            sample = torch.tanh(dist.rsample())
            action_value = float(sample.squeeze().clamp(-1.0, 1.0).item())
            next_observation, reward, done, _ = env.step(action_value)
            reward_tensor = torch.tensor([[float(reward)]], dtype=torch.float32, device=dev)

            with torch.no_grad():
                if done or next_observation is None:
                    next_value = torch.zeros_like(value)
                else:
                    next_state = observation_to_tensor(next_observation, dev)
                    _, _, next_value = network(next_state)
            target_value = reward_tensor + config.gamma * next_value
            advantage = target_value - value

            log_prob = dist.log_prob(sample).sum(dim=1, keepdim=True)
            entropy = dist.entropy().sum(dim=1, keepdim=True)
            actor_loss = -(log_prob * advantage.detach()).mean() - config.entropy_weight * entropy.mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + config.value_weight * critic_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
            optimizer.step()

            total_reward += float(reward)
            steps += 1
            observation = next_observation if next_observation is not None else observation

        episode_logs.append(
            {
                "episode": episode,
                "symbol": symbol,
                "split": split,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

    return A2CPolicy(network, dev), episode_logs


def train_paper_model(
    feature_frame: pd.DataFrame,
    config: PaperRLConfig,
    train_split: str = "train",
    symbols: list[str] | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
):
    set_seed(config.seed)
    if config.algo == "dqn":
        return train_dqn(feature_frame, train_split, symbols, config, splits=splits)
    if config.algo == "pg":
        return train_pg(feature_frame, train_split, symbols, config, splits=splits)
    if config.algo == "a2c":
        return train_a2c(feature_frame, train_split, symbols, config, splits=splits)
    raise ValueError(f"unsupported algo {config.algo}")


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
    learned_policy,
    split: str,
    algo: str,
    symbols: list[str] | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, EvalReport]]:
    env_config = continuous_env_config() if algo == "a2c" else discrete_env_config()
    backtester = Backtester(feature_frame=feature_frame, env_config=env_config, splits=splits)
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


def run_paper_experiment(
    feature_frame: pd.DataFrame,
    config: PaperRLConfig,
    train_split: str = "train",
    eval_split: str = "test",
    symbols: list[str] | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    policy, episode_logs = train_paper_model(
        feature_frame=feature_frame,
        config=config,
        train_split=train_split,
        symbols=symbols,
        splits=splits,
    )
    comparison, reports = compare_policies(
        feature_frame=feature_frame,
        learned_policy=policy,
        split=eval_split,
        algo=config.algo,
        symbols=symbols,
        splits=splits,
    )
    return {
        "policy": policy,
        "episode_logs": episode_logs,
        "training_summary": summarize_training(episode_logs),
        "comparison": comparison,
        "reports": reports,
    }


def save_experiment_outputs(
    experiment: dict[str, object],
    comparison_path: str | Path,
    episode_log_path: str | Path | None = None,
) -> tuple[Path, Path]:
    comparison_output = Path(comparison_path)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    experiment["comparison"].to_csv(comparison_output, index=False)

    logs_output = Path(episode_log_path) if episode_log_path is not None else comparison_output.with_name(
        comparison_output.stem + "_episodes.csv"
    )
    logs_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiment["episode_logs"]).to_csv(logs_output, index=False)
    return comparison_output, logs_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style DRL methodology replication on TradingEnv.")
    parser.add_argument("--algo", choices=("dqn", "pg", "a2c"), default="dqn")
    parser.add_argument("--features-path", default=str(DEFAULT_OUTPUT_DIR / "processed" / "features.csv"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--comparison-path", default=None)
    parser.add_argument("--episode-log-path", default=None)
    args = parser.parse_args()

    feature_frame = pd.read_csv(args.features_path)
    config = PaperRLConfig(algo=args.algo, episodes=args.episodes)
    experiment = run_paper_experiment(
        feature_frame=feature_frame,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    comparison_path = args.comparison_path or str(DEFAULT_OUTPUT_DIR / "qa" / f"paper_{args.algo}_{args.eval_split}.csv")
    saved_comparison, saved_episode = save_experiment_outputs(
        experiment,
        comparison_path=comparison_path,
        episode_log_path=args.episode_log_path,
    )
    print(f"algo={args.algo}")
    print("training_summary:")
    print(experiment["training_summary"])
    print("comparison:")
    print(experiment["comparison"].to_string(index=False))
    print(f"saved_comparison={saved_comparison}")
    print(f"saved_episode_logs={saved_episode}")


if __name__ == "__main__":
    main()
