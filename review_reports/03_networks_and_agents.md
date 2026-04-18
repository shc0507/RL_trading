# Subagent 3 — Networks & Agents Review

**Scope.** `networks.py`, `agents.py`
**Prompt.** `../review_prompts/3_networks_and_agents.md`

> Annotations by the main reviewer are prefixed with **[MAIN]:**.

---

## Subagent overall assessment

The implementation reasonably maps the three RL algorithms (DQN with double DQN + dueling, PG/REINFORCE, A2C) onto the paper's architecture. The 2-layer LSTM (64→32) with Leaky-ReLU is structurally sound, and most hyperparameters match Exhibit 1. However, there are several design choices and potential issues worth flagging: the placement of activation/dropout between LSTM layers breaks sequential composition in an unusual way, fixed-std Gaussian exploration is not specified in the paper, and the PGAgent uses numerically suboptimal log-computation.

**[MAIN]: Agree overall — this was the most accurate of the four subagent reports.**

---

## Findings

### networks.py:19–25 (LSTMEncoder) — Activation/dropout between LSTM layers

> Subagent: The code applies `LeakyReLU()` and `Dropout()` to the *output* of LSTM1 before feeding it to LSTM2. This is non-standard: LSTMs typically expect raw sequences as input, and interposing nonlinearity/regularization may suppress temporal patterns learned by LSTM1. While not strictly wrong, it's architecturally unusual.
>
> Severity: nit.

**[MAIN]: Agree.** Two LSTMs stacked via `nn.LSTM(..., num_layers=2)` (with `dropout` kwarg for inter-layer dropout) would be the idiomatic way. Paper just says "2-layer LSTM"; either implementation is defensible. Flag for reconsideration if training is unstable.

---

### networks.py:25 — Last hidden state extraction

> Subagent: `out[:, -1, :]` — standard, correct.

**[MAIN]: Agree.**

---

### networks.py:31 — DQNNetwork `n_features=11` default

> Subagent: Matches `env.STATE_DIM = 10 + 1 = 11`.

**[MAIN]: Agree.** Verified — `FEATURE_COLS` has 10 entries in `features.py:21–32`, plus the position channel → 11.

---

### networks.py:41 — Dueling split

> Subagent: `v + (a - a.mean(dim=1, keepdim=True))`. Standard form.

**[MAIN]: Agree.**

---

### networks.py:54 — PGNetwork softmax

> Subagent: Standard.

**[MAIN]: Agree.** See agents.py:168 annotation below — the softmax here combined with `log(probs+eps)` downstream is suboptimal vs `log_softmax` on logits.

---

### networks.py:68 — A2CNetwork tanh actor

> Subagent: `tanh(actor(h))` outputs continuous action in [−1, 1]. Correct per paper.

**[MAIN]: Agree.** Only the *mean* is learned; noise is added in the agent (fixed std). See D3 in overall review.

---

### agents.py:55 — ε-decay schedule

> Subagent: ε decays linearly from 1.0 to 0.01 over `eps_decay_steps=5000`, controlled by `step_count`. Paper doesn't specify. `step_count` only increments *after* training begins. During the first ~64 steps (before first training call), ε stays at 1.0.
>
> Severity: nit.

**[MAIN]: Agree** — with one small caveat. `epsilon` is also called during eval in `_collect_rl_rewards` via `select_action(state, training=False)`, which skips the ε check entirely. So the eval path is greedy regardless — fine. Just worth confirming the first-64-steps-all-random behavior is intended (it is, effectively, since it just fills the replay buffer with exploration).

---

### agents.py:82–85 — Double DQN + gather shapes

> Subagent: Online net selects `best_actions`, target net evaluates them. Tensor shapes correct.

**[MAIN]: Agree.** Verified — `gather(1, a.unsqueeze(1)).squeeze(1)` produces `(batch,)`.

---

### agents.py:86 — Terminal handling

> Subagent: `targets = r + γ * q_next * (1 - d)` masks with `(1 - d)`.

**[MAIN]: Agree.** Correct.

---

### agents.py:26, 119 — γ=0.3

> Subagent: Matches Exhibit 1 for all algorithms.

**[MAIN]: Agree.** Aggressive discount but matches paper.

---

### agents.py:168 — `torch.log(probs + 1e-8)`

> Subagent: Taking log directly with a small epsilon is numerically safe but less standard than `F.log_softmax()` applied to logits.
>
> Severity: nit.

**[MAIN]: Agree.** Cheap improvement — expose logits from the PGNetwork (or have the agent call `log_softmax` on stored logits) instead of softmax+log. Not urgent unless you see NaN losses.

---

### agents.py:138 — PGAgent action sampling

> Subagent: `torch.multinomial(probs, 1)` during training, argmax for eval.

**[MAIN]: Agree.**

---

### agents.py:210–215 — A2C optimizer param groups

> Subagent: Two param groups: encoder + actor @ lr_actor=1e-4, critic @ lr_critic=1e-3. Gradients flow from critic_loss.backward() to the shared encoder, but the optimizer applies the actor's LR (1e-4) to those gradients. If the intent is for the critic to also update the encoder at 1e-3, this is a bug.
>
> UNSURE — Severity: spec-deviation candidate.

**[MAIN]: Agree it's worth flagging, but not a bug — it's a design choice.** In PyTorch, each parameter belongs to exactly one group; Adam applies the group's LR regardless of which loss produced the gradient. So the encoder updates at `lr_actor=1e-4`, effectively treating actor-path shaping as the dominant signal. That may actually suit actor–critic stability (the critic's noisy early targets would otherwise swing the encoder around). If you want the critic to have equal pull, put the encoder in its own group with an LR of your choosing. Add a comment explaining the convention either way. See D4 in overall review.

---

### agents.py:201 — Fixed exploration std

> Subagent: Fixed `std=0.2` for exploration. Paper doesn't specify.
>
> Severity: spec-deviation.

**[MAIN]: Agree.** Consider a learned `log_std` head or a decay schedule. See D3 in overall review.

---

### agents.py:255 — TD target + advantage

> Subagent: `r + γ * next_value * (1 - d)` under `no_grad`; advantage detached. Correct TD(0) actor-critic.

**[MAIN]: Agree.**

---

### agents.py:268 — Combined backward pass

> Subagent: Single backward for `critic_loss + actor_loss` ensures shared encoder receives consistent gradient updates. Good design.

**[MAIN]: Agree.** Better than two separate backwards that could conflict via retain_graph.

---

### agents.py:102–103, 185 — `weights_only=True` on load

> Subagent: Good practice.

**[MAIN]: Agree.** Safe against pickle RCE in untrusted checkpoints.

---

### agents.py:35, 122, 204 — Device handling

> Subagent: Each agent moves networks to device via `.to(self.device)`. Tensors created with `device=self.device`. No implicit CPU-to-device transfers detected.

**[MAIN]: Agree.** Verified.

---

## Main reviewer's additional items (not in subagent report)

- **`agents.py:133–141` (PGAgent.select_action)** — `probs = self.net(t).squeeze(0)` relies on the softmax happening in forward. Coupled with `torch.log(probs + 1e-8)` later, this is the same numerical-stability issue mentioned under agents.py:168. Fixing it requires changing `PGNetwork.forward` to return logits; then `select_action` uses `softmax` on the fly and `train_episode` uses `log_softmax`.
- **`agents.py:59` (DQNAgent.select_action)** — `random.randint(0, 2)` hardcodes the 3-action space. If `n_actions` is ever configurable, this needs to be parameterized. Low priority.
- **`agents.py:230` (A2CAgent.select_action)** — `np.random.normal(...)` uses numpy's *global* RNG, not a seeded one. Reproducibility gotcha if you ever need deterministic rollouts.

---

## Summary severity distribution after annotations

- Bugs: 0.
- Spec deviations (legitimate): 2 (fixed A2C std, encoder LR asymmetry).
- Nits: 3 (LSTM layer stacking, log+softmax vs log_softmax, hardcoded action count / numpy global RNG).
