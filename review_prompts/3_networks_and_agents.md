You are doing a code review of two files in a PyTorch project that recreates Zhang, Zohren, Roberts (2020) "Deep Reinforcement Learning for Trading" (JFDS). Do NOT edit files — review only.

**Files to review (read in full):**
- /Users/haochen/projects/RL_trading/rl_trading/networks.py
- /Users/haochen/projects/RL_trading/rl_trading/agents.py

**Supporting files (read only if needed to understand shapes/interfaces):**
- /Users/haochen/projects/RL_trading/rl_trading/env.py (state shape contract: (seq_len, n_features+1), and what the `+1` is)

**Read the paper directly to avoid relying on paraphrases.** The PDF is at:
/Users/haochen/projects/RL_trading/docs/Deep-Reinforcement-Learning-for-Trading (1).pdf

Focus on pages 5–7. Pages 5–6 describe the three RL algorithms (DQN with fixed Q-targets + double DQN + dueling; PG / REINFORCE; A2C). Page 6 mentions the 2-layer LSTM (64, 32 units) + Leaky-ReLU. Page 7 is Exhibit 1 hyperparameters (LRs, γ=0.3, bp=0.002, batch sizes, DQN replay memory 5000, τ=1000). Use `pages: "5-7"`.

**What I want you to check (think independently — flag anything else that looks wrong):**
1. LSTMEncoder:
   - Paper says 2-layer LSTM 64 → 32 + Leaky-ReLU. Code stacks two separate `nn.LSTM` modules. Is the Leaky-ReLU + dropout placement between them reasonable, or does it break the LSTM's hidden-state logic?
   - The encoder takes the last hidden state `out[:, -1, :]`. Is that correct for this setup?
2. DQNNetwork: dueling split V + (A − A.mean(dim=1)). Standard form. Anything off? `n_features=11` default — but env's STATE_DIM is 10 features + 1 position = 11. Verify consistency.
3. PGNetwork: softmax head. Standard.
4. A2CNetwork: actor uses `tanh(actor(h))` to output a deterministic mean in [−1, 1]. The agent adds Gaussian noise with std=0.2 for exploration. Paper specifies continuous action space [−1, 1] for A2C. Is learning a fixed-std Gaussian (not state-dependent) defensible? Flag.
5. DQNAgent:
   - Hyperparams: lr=1e-4, γ=0.3, batch=64, memory=5000, target_update=1000 — compare to Exhibit 1.
   - ε-decay: linear over 5000 steps from 1.0 → 0.01. Paper doesn't specify; is 5000 enough given replay buffer size? Also, training step count is used to decay ε even though ε should arguably decay on env steps (select_action calls). Note: `step_count` only increments *after* training begins (once buffer has ≥batch_size samples). Is ε still 1.0 for the first 64 steps? That's fine but worth verifying.
   - Double DQN implementation: online net selects next action, target net evaluates — looks right. Verify tensor shapes (gather on dim=1).
   - Terminal handling: `targets = r + γ * q_next * (1 - d)` — correct.
6. PGAgent (REINFORCE):
   - γ=0.3 → very aggressive discount. Paper Exhibit 1 uses γ=0.3 for all. Consistent.
   - Baseline: mean-centered returns (not a learned baseline). Paper says PG uses MC updates. OK.
   - `torch.log(probs.gather(...) + 1e-8)` — safer than log_softmax? Note the forward pass already does softmax. Flag if this is numerically suboptimal.
   - Buffers cleared after each episode; `select_action` uses `torch.multinomial` during training, argmax for eval.
7. A2CAgent:
   - Optimizer has two param groups: actor(+encoder) @ 1e-4 and critic @ 1e-3. Encoder is shared — is putting encoder params under the actor LR only correct, or does the critic path also need to update the encoder? Check: when critic_loss.backward() is called together with actor_loss, gradients flow to the encoder from both. With `optim.Adam` param groups, the encoder params are in the actor group (lr=1e-4), so the critic's gradients through the encoder will be applied with lr=1e-4, not 1e-3. Is that the intent?
   - Fixed std=0.2 for exploration. Paper doesn't specify. Flag.
   - `td_target = r + γ * next_value * (1 - d)` with `next_value` computed under `no_grad`. Advantage detached. Correct TD(0) actor-critic.
   - Buffer flushed once size ≥ batch_size, then cleared. Note: trainer.py has an end-of-episode flush that sets batch_size=1. Any risk of single-step updates dominating?
   - `select_action` uses Gaussian sampling at train time, `mean` at eval — reasonable.
8. Save/load: `weights_only=True` on load — good.
9. Device handling: each agent takes a `device` string; do they move all operations to device? Any tensor still created on CPU and implicitly moved?

**Report format (under ~400 words):**
- One short paragraph with your overall take.
- A bulleted list of findings. For each: `file.py:line` reference, one-line problem statement, severity (bug / spec-deviation / nit).
- If you're unsure about something, say "UNSURE:" and explain.

Report back to me — the main Claude instance will consolidate findings across categories.
