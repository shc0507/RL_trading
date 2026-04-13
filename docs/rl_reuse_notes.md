# RL Reuse Notes

The DQN baseline in this repo reuses the structure of the homework code, but the implementation is copied into repo-local modules and cleaned up for the trading project.

It should be treated as a Zhang-inspired training baseline, not as a full reproduction of Zhang et al.'s DQN architecture. The repo's Zhang alignment currently applies to evaluation and plotting conventions on the ETF dataset.

## Reused for v1 DQN

- `rl_trading/rl/dqn_model.py` adapts the Q-network layout from `/Users/haochen/Downloads/hw25100/model.py`.
- `rl_trading/rl/dqn_agent.py` adapts the greedy action path, Bellman target computation, target-network updates, and save/load flow from `/Users/haochen/Downloads/hw25100/agent.py`.
- `rl_trading/rl/replay_buffer.py` adapts the plain replay-buffer storage and sampling pattern from `/Users/haochen/Downloads/hw25100/buffer.py`.

## Intentionally Not Reused Yet

- The video, plotting, and Gym wrapper logic in `/Users/haochen/Downloads/hw25100/core.py` is not imported into this repo. The trading trainer uses repo-specific artifact files and validation through `Backtester` instead.
- Prioritized replay and n-step replay are still deferred.
- Double DQN, dueling networks, and LSTM DQN are now implemented as options in the repo-local DQN agent/trainer.

## Correctness Fixes Applied During Adaptation

- Discrete actions are stored as `int64`, not float.
- Terminal flags are stored as numeric tensors with stable shape for Bellman targets.
- The DQN code no longer depends on `hydra.utils.instantiate` or course-local helper modules.
- Q-value gathering avoids shape-dropping behavior that can break batch size `1`.
- Checkpoints are written under `artifacts/rl/checkpoints/` instead of a hardcoded `models/` folder.

## Future Reuse For Other RL Models

- `/Users/haochen/Downloads/5100_hw3 2/src/policies.py`
  - Reuse the actor/distribution scaffolding for REINFORCE or PPO.
  - For continuous trading actions, add output squashing or clipping to keep positions inside `[-1, 1]`.
- `/Users/haochen/Downloads/5100_hw3 2/src/critics.py`
  - Reuse the value-network shape and training pattern.
  - Change scalar value outputs to use `.squeeze(-1)` instead of a bare `.squeeze()` so batch size `1` stays stable.
- `/Users/haochen/Downloads/5100_hw3 2/src/pg_agent.py`
  - Reuse the discounted-return helpers and the actor/critic update orchestration.
  - Do not reuse the current advantage path as a GAE implementation, because `gae_lambda` is present but unused in that homework version.

## Implemented After v1

- `rl_trading/rl/dqn_agent.py` now supports plain DQN and Double DQN target computation.
- `rl_trading/rl/dqn_model.py` now supports MLP, dueling MLP, LSTM, and dueling LSTM Q-networks.
- `rl_trading/rl/policy_gradient.py` ports the familiar policy-gradient structure into repo-local A2C/PPO modules with shape-stable value outputs and GAE support.
- `rl_trading/rl/on_policy_trainer.py` wires A2C/PPO into the repo artifact and Zhang-evaluation flow.
- `rl_trading/rl/continuous_control.py` adds repo-local TD3/SAC agent modules with action squashing to keep continuous target positions inside `[-1, 1]`.
- `rl_trading/rl/continuous_trainer.py` wires TD3/SAC into the repo artifact and Zhang-evaluation flow.
