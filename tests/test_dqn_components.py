from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl_trading.rl.dqn_agent import DQNAgent
from rl_trading.rl.dqn_model import DuelingQNetwork
from rl_trading.rl.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_wraps_and_samples_tensor_batch(self) -> None:
        buffer = ReplayBuffer(capacity=3, state_size=4, device="cpu")
        for idx in range(5):
            buffer.add(
                (
                    np.full(4, idx, dtype=np.float32),
                    idx % 3,
                    float(idx),
                    np.full(4, idx + 1, dtype=np.float32),
                    idx % 2 == 0,
                )
            )

        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(2)
        self.assertEqual(batch.states.shape, (2, 4))
        self.assertEqual(batch.next_states.shape, (2, 4))
        self.assertEqual(batch.actions.dtype, torch.int64)
        self.assertEqual(batch.rewards.dtype, torch.float32)
        self.assertEqual(batch.dones.dtype, torch.float32)


class DQNAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DQNAgent(
            state_size=4,
            action_size=3,
            hidden_sizes=(8,),
            lr=1e-3,
            gamma=0.5,
            target_update_interval=10,
            device="cpu",
        )

    def test_bellman_target_respects_terminal_flag(self) -> None:
        with torch.no_grad():
            for parameter in self.agent.target_net.parameters():
                parameter.zero_()
            self.agent.target_net.q_head[-1].bias.copy_(torch.tensor([0.0, 2.0, 1.0]))

        rewards = torch.tensor([1.0, 1.0], dtype=torch.float32)
        dones = torch.tensor([1.0, 0.0], dtype=torch.float32)
        next_states = torch.zeros((2, 4), dtype=torch.float32)

        targets = self.agent.get_q_target(rewards=rewards, dones=dones, next_states=next_states)
        self.assertAlmostEqual(float(targets[0]), 1.0)
        self.assertAlmostEqual(float(targets[1]), 2.0)

    def test_double_dqn_target_uses_online_argmax_and_target_value(self) -> None:
        agent = DQNAgent(
            state_size=4,
            action_size=3,
            hidden_sizes=(),
            lr=1e-3,
            gamma=0.5,
            double_dqn=True,
            device="cpu",
        )
        with torch.no_grad():
            for parameter in agent.q_net.parameters():
                parameter.zero_()
            for parameter in agent.target_net.parameters():
                parameter.zero_()
            agent.q_net.q_head[-1].bias.copy_(torch.tensor([2.0, 1.0, 0.0]))
            agent.target_net.q_head[-1].bias.copy_(torch.tensor([0.0, 5.0, 10.0]))

        rewards = torch.tensor([1.0], dtype=torch.float32)
        dones = torch.tensor([0.0], dtype=torch.float32)
        next_states = torch.zeros((1, 4), dtype=torch.float32)

        target = agent.get_q_target(rewards=rewards, dones=dones, next_states=next_states)
        self.assertAlmostEqual(float(target[0]), 1.0)

    def test_greedy_action_and_checkpoint_round_trip(self) -> None:
        with torch.no_grad():
            for parameter in self.agent.q_net.parameters():
                parameter.zero_()
            self.agent.q_net.q_head[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
            self.agent.target_net.load_state_dict(self.agent.q_net.state_dict())

        state = np.zeros(4, dtype=np.float32)
        self.assertEqual(self.agent.get_action(state), 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "agent.pt"
            self.agent.save(checkpoint_path)
            with torch.no_grad():
                self.agent.q_net.q_head[-1].bias.zero_()
            self.agent.load(checkpoint_path)
            self.assertEqual(self.agent.get_action(state), 1)

    def test_dueling_q_network_combines_value_and_centered_advantage(self) -> None:
        network = DuelingQNetwork(
            state_size=4,
            action_size=3,
            hidden_sizes=(),
        )
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.value_head[-1].bias.fill_(1.0)
            network.advantage_head[-1].bias.copy_(torch.tensor([0.0, 2.0, 4.0]))

        q_values = network(torch.zeros((2, 4), dtype=torch.float32))
        expected = torch.tensor([[-1.0, 1.0, 3.0], [-1.0, 1.0, 3.0]])
        self.assertTrue(torch.allclose(q_values, expected))

    def test_lstm_dqn_outputs_action_values_and_round_trips_checkpoint(self) -> None:
        agent = DQNAgent(
            state_size=7,
            action_size=3,
            hidden_sizes=(8,),
            lr=1e-3,
            network_type="lstm",
            observation_window=3,
            feature_size=2,
            recurrent_hidden_size=5,
            recurrent_layers=1,
            device="cpu",
        )
        state = np.zeros(7, dtype=np.float32)
        with torch.no_grad():
            q_values = agent.q_net(torch.zeros((4, 7), dtype=torch.float32))
        self.assertEqual(q_values.shape, (4, 3))
        self.assertIn(agent.get_action(state), [0, 1, 2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "lstm_agent.pt"
            agent.save(checkpoint_path)
            agent.load(checkpoint_path)
            self.assertIn(agent.get_action(state), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
