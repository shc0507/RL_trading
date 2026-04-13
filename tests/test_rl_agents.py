from __future__ import annotations

import unittest

import numpy as np
import torch

from rl_trading.rl.continuous_control import ContinuousReplayBuffer, SACAgent, TD3Agent
from rl_trading.rl.policy_gradient import (
    A2CAgent,
    OnPolicyBatch,
    PPOAgent,
    compute_gae,
    discounted_return,
    discounted_reward_to_go,
)


class PolicyGradientAgentTests(unittest.TestCase):
    def test_discount_helpers_and_gae_are_shape_stable(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(discounted_return(rewards, gamma=0.5), np.array([2.75, 2.75, 2.75], dtype=np.float32))
        np.testing.assert_allclose(discounted_reward_to_go(rewards, gamma=0.5), np.array([2.75, 3.5, 3.0], dtype=np.float32))

        advantages, returns = compute_gae(
            rewards=torch.tensor([1.0, 1.0], dtype=torch.float32),
            dones=torch.tensor([0.0, 1.0], dtype=torch.float32),
            values=torch.tensor([0.5, 0.25], dtype=torch.float32),
            next_value=torch.tensor(0.0, dtype=torch.float32),
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertEqual(advantages.shape, (2,))
        self.assertEqual(returns.shape, (2,))

    def test_a2c_and_ppo_update_on_discrete_batches(self) -> None:
        states = torch.randn((16, 5), dtype=torch.float32)
        actions = torch.randint(0, 3, (16,), dtype=torch.int64)
        returns = torch.randn(16, dtype=torch.float32)
        advantages = torch.randn(16, dtype=torch.float32)

        a2c = A2CAgent(
            state_size=5,
            action_size=3,
            hidden_sizes=(16,),
            actor_lr=1e-3,
            critic_lr=1e-3,
            device="cpu",
        )
        info = a2c.update(
            OnPolicyBatch(
                states=states,
                actions=actions,
                returns=returns,
                advantages=advantages,
            )
        )
        self.assertIn("actor_loss", info)
        self.assertIn(a2c.get_greedy_action(np.zeros(5, dtype=np.float32)), [0, 1, 2])

        with torch.no_grad():
            old_log_probs = a2c.actor(states).log_prob(actions)
        ppo = PPOAgent(
            state_size=5,
            action_size=3,
            hidden_sizes=(16,),
            actor_lr=1e-3,
            critic_lr=1e-3,
            update_epochs=2,
            minibatch_size=8,
            device="cpu",
        )
        ppo_info = ppo.update(
            OnPolicyBatch(
                states=states,
                actions=actions,
                returns=returns,
                advantages=advantages,
                old_log_probs=old_log_probs,
            )
        )
        self.assertIn("actor_loss", ppo_info)
        self.assertIn(ppo.get_greedy_action(np.zeros(5, dtype=np.float32)), [0, 1, 2])


class ContinuousControlAgentTests(unittest.TestCase):
    def test_continuous_replay_buffer_and_agents_update(self) -> None:
        buffer = ContinuousReplayBuffer(capacity=32, state_size=5, action_size=1, device="cpu")
        for index in range(20):
            buffer.add(
                (
                    np.full(5, index / 10.0, dtype=np.float32),
                    np.array([(-1.0) ** index * 0.25], dtype=np.float32),
                    float(index) / 100.0,
                    np.full(5, (index + 1) / 10.0, dtype=np.float32),
                    index % 7 == 0,
                )
            )
        batch = buffer.sample(8)
        self.assertEqual(batch.actions.shape, (8, 1))

        td3 = TD3Agent(
            state_size=5,
            action_size=1,
            hidden_sizes=(16,),
            actor_lr=1e-3,
            critic_lr=1e-3,
            device="cpu",
        )
        td3_info = td3.update(batch, step=2)
        self.assertIn("critic_loss", td3_info)
        self.assertIn("actor_loss", td3_info)
        self.assertGreaterEqual(td3.get_action(np.zeros(5, dtype=np.float32)), -1.0)
        self.assertLessEqual(td3.get_action(np.zeros(5, dtype=np.float32)), 1.0)

        sac = SACAgent(
            state_size=5,
            action_size=1,
            hidden_sizes=(16,),
            actor_lr=1e-3,
            critic_lr=1e-3,
            device="cpu",
        )
        sac_info = sac.update(batch)
        self.assertIn("critic_loss", sac_info)
        self.assertIn("actor_loss", sac_info)
        self.assertGreaterEqual(sac.get_action(np.zeros(5, dtype=np.float32), deterministic=True), -1.0)
        self.assertLessEqual(sac.get_action(np.zeros(5, dtype=np.float32), deterministic=True), 1.0)


if __name__ == "__main__":
    unittest.main()
