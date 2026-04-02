from __future__ import annotations

import unittest

from trading.features import FeatureBuilder
from trading.rl_train import (
    QLearningConfig,
    QTablePolicy,
    discretize_observation,
    run_q_learning_experiment,
    train_q_learning,
)

from test_data_pipeline import make_bar_frame


class RLTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = make_bar_frame(symbols=("AAA", "BBB"), periods=500)
        self.features = FeatureBuilder().transform(bars)

    def test_discretize_observation_returns_discrete_state_tuple(self) -> None:
        row = self.features.loc[self.features["window_ready"]].iloc[0].to_dict()
        observation = {
            "symbol": row["symbol"],
            "row": row,
            "position": 0.0,
        }
        state = discretize_observation(observation)
        self.assertEqual(len(state), 6)
        self.assertTrue(all(value in {-1, 0, 1} for value in state))

    def test_train_q_learning_populates_q_table(self) -> None:
        q_tables, episode_logs = train_q_learning(
            feature_frame=self.features,
            split="train",
            symbols=["AAA", "BBB"],
            config=QLearningConfig(episodes=6, epsilon=0.1),
            splits={"train": ("2020-01-01", "2022-12-31")},
        )
        self.assertTrue(q_tables)
        self.assertEqual(set(q_tables), {"AAA", "BBB"})
        self.assertEqual(len(episode_logs), 12)
        self.assertTrue(all(log["steps"] > 0 for log in episode_logs))

    def test_q_table_policy_acts_using_greedy_lookup(self) -> None:
        row = self.features.loc[self.features["window_ready"]].iloc[0].to_dict()
        observation = {
            "symbol": row["symbol"],
            "row": row,
            "position": 0.0,
        }
        state = discretize_observation(observation)
        policy = QTablePolicy(q_tables={row["symbol"]: {(state, 0): -1.0, (state, 1): 0.5, (state, 2): 2.0}})
        self.assertEqual(policy.act(observation), 1.0)

    def test_run_q_learning_experiment_returns_baseline_comparison(self) -> None:
        experiment = run_q_learning_experiment(
            feature_frame=self.features,
            train_split="train",
            eval_split="test",
            symbols=["AAA", "BBB"],
            config=QLearningConfig(episodes=4, epsilon=0.1),
            splits={
                "train": ("2020-01-01", "2021-06-30"),
                "test": ("2021-07-01", "2021-12-31"),
            },
        )
        comparison = experiment["comparison"]
        self.assertEqual(len(experiment["episode_logs"]), 8)
        self.assertTrue({"q_learning", "long_only", "sign_12m", "macd"}.issubset(set(comparison["policy"])))
        self.assertIn("annualized_return", comparison.columns)
        self.assertTrue(experiment["q_table"])


if __name__ == "__main__":
    unittest.main()
