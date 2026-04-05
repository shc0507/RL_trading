"""Package-based RL training using Stable-Baselines3 on top of TradingEnv."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Literal

import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import A2C, DQN, PPO

from .config import DEFAULT_OUTPUT_DIR
from .sb3_env import SB3TradingEnv, compare_sb3_policy


@dataclass(slots=True)
class SB3TrainConfig:
    algo: Literal["dqn", "a2c", "ppo"] = "dqn"
    total_timesteps: int = 20_000
    learning_rate: float = 1e-4
    gamma: float = 0.3
    seed: int = 7
    train_reward_mode: Literal["raw", "zhang"] = "zhang"
    eval_reward_mode: Literal["raw", "zhang"] = "raw"


def build_model(algo: str, env: SB3TradingEnv, config: SB3TrainConfig):
    if algo == "dqn":
        return DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=64,
            buffer_size=5_000,
            learning_starts=512,
            target_update_interval=1_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.2,
            seed=config.seed,
            verbose=1,
        )
    if algo == "a2c":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            n_steps=128,
            seed=config.seed,
            verbose=1,
        )
    if algo == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            n_steps=128,
            batch_size=128,
            seed=config.seed,
            verbose=1,
        )
    raise ValueError(f"unsupported algo {algo}")


def algo_action_mode(algo: str) -> Literal["continuous", "discrete"]:
    if algo == "a2c":
        return "continuous"
    return "discrete"


def run_sb3_experiment(
    feature_frame: pd.DataFrame,
    config: SB3TrainConfig,
    train_split: str = "train",
    eval_split: str = "test",
    symbols: list[str] | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
    validation_split: str | None = None,
    early_stopping_patience_evals: int | None = None,
    eval_freq: int | None = None,
) -> dict[str, object]:
    action_mode = algo_action_mode(config.algo)
    env = SB3TradingEnv(
        feature_frame=feature_frame,
        split=train_split,
        symbols=symbols,
        action_mode=action_mode,
        reward_mode=config.train_reward_mode,
        splits=splits,
        shuffle_symbols=True,
        seed=config.seed,
    )
    model = build_model(config.algo, env, config)
    callback = None
    best_model_path: Path | None = None
    if validation_split is not None:
        validation_env = SB3TradingEnv(
            feature_frame=feature_frame,
            split=validation_split,
            symbols=symbols,
            action_mode=action_mode,
            reward_mode=config.train_reward_mode,
            splits=splits,
            shuffle_symbols=True,
            seed=config.seed,
        )
        stop_callback = None
        if early_stopping_patience_evals is not None:
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=early_stopping_patience_evals,
                min_evals=1,
                verbose=0,
            )
        temp_dir = Path(tempfile.mkdtemp(prefix="sb3_best_model_"))
        callback = EvalCallback(
            validation_env,
            best_model_save_path=str(temp_dir),
            log_path=str(temp_dir),
            eval_freq=eval_freq or max(config.total_timesteps // 20, 1),
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback,
            verbose=0,
        )
        best_model_path = temp_dir / "best_model.zip"

    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    if best_model_path is not None and best_model_path.exists():
        model = type(model).load(str(best_model_path), env=env)

    comparison, reports = compare_sb3_policy(
        feature_frame=feature_frame,
        model=model,
        name=f"sb3_{config.algo}",
        split=eval_split,
        action_mode=action_mode,
        reward_mode=config.eval_reward_mode,
        symbols=symbols,
        splits=splits,
    )
    return {
        "model": model,
        "comparison": comparison,
        "reports": reports,
        "algo": config.algo,
        "total_timesteps": config.total_timesteps,
        "train_reward_mode": config.train_reward_mode,
        "eval_reward_mode": config.eval_reward_mode,
        "validation_split": validation_split,
    }


def save_experiment_outputs(
    experiment: dict[str, object],
    comparison_path: str | Path,
) -> Path:
    comparison_output = Path(comparison_path)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    experiment["comparison"].to_csv(comparison_output, index=False)
    return comparison_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train package-based RL models with Stable-Baselines3 on the trading environment."
    )
    parser.add_argument("--algo", choices=("dqn", "a2c", "ppo"), default="dqn")
    parser.add_argument("--features-path", default=str(DEFAULT_OUTPUT_DIR / "processed" / "features.csv"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--comparison-path", default=None)
    args = parser.parse_args()

    feature_frame = pd.read_csv(args.features_path)
    experiment = run_sb3_experiment(
        feature_frame=feature_frame,
        config=SB3TrainConfig(algo=args.algo, total_timesteps=args.total_timesteps),
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    comparison_path = args.comparison_path or str(DEFAULT_OUTPUT_DIR / "qa" / f"sb3_{args.algo}_{args.eval_split}.csv")
    saved_comparison = save_experiment_outputs(experiment, comparison_path)
    print(f"algo={args.algo}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"train_reward_mode={experiment['train_reward_mode']}")
    print(f"eval_reward_mode={experiment['eval_reward_mode']}")
    print("comparison:")
    print(experiment["comparison"].to_string(index=False))
    print(f"saved_comparison={saved_comparison}")


if __name__ == "__main__":
    main()
