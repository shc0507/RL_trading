# Overall Code Review — Zhang et al. (2019) Recreation

**Scope.** Review of the `rl_trading/` package against Zhang, Zohren, Roberts (2020) "Deep Reinforcement Learning for Trading" (JFDS). Category-level reviews were done by four Explore subagents (see `01_`–`04_` in this folder); this document is the consolidated pass, with subagent findings verified against the code and false positives called out.

**Files covered.**
- Data & features: `config.py`, `data.py`, `features.py`
- Environment: `env.py`
- Networks & agents: `networks.py`, `agents.py`
- Training & evaluation: `trainer.py`, `baselines.py`, `metrics.py`, `evaluate.py`
- Entry points: `run.py`, `smoke.py` (self-review)
- Cross-file wiring (self-review)

---

## Overall take

The pipeline is structurally sound and most formulas match the paper. Three real bugs would cause wrong numbers on a live run; one major spec deviation (multiplicative vs additive returns) contradicts an explicit paper choice; and several smaller items are worth addressing. Subagent false positives are filtered below.

---

## Bugs (fix these)

### B1. Position-history off-by-one — `env.py:168` + `_get_state` lines 186–188

The state at time `t+1` never contains the action just taken.

**Trace:**
- In `step()` at code-index `t`, after computing reward, line 168 writes `self._pos_history[self._t] = position` (writes at index `t`).
- Line 171 increments `self._t → t+1`.
- `_get_state()` then returns `pos_col = self._pos_history[start : self._t + 1]` = `pos_history[t−58 : t+2]` (for `seq_len=60`).
- The last slot of `pos_col` is `pos_history[t+1]`, which has **never been written** (still the zero from `reset`).
- The second-to-last slot is `pos_history[t] = a_t` (the action just taken).

**Effect.** The "current position" channel in the agent's state is always one step behind, and the last slot is always zero. The RL agent cannot cleanly observe its own most-recent action.

**Fix (one of):**
- Slice `pos_col = self._pos_history[start − 1 : self._t]` so the last element aligns with the most recent action.
- Or write `self._pos_history[self._t + 1] = position` before incrementing, adopting the convention that `pos_history[i]` = position held going into period `i`.

**Severity: bug.** High priority — changes what the policy can learn.

---

### B2. Silent exception swallowing — `evaluate.py:181–183, 212–214`

Both the baseline loop and the RL-agent loop wrap per-symbol work in `try: ... except Exception as e: print(f"... FAILED ({e})"); results[sym][label] = pd.Series(...)`.

**Effect.** Any failure (data gap, training crash, checkpoint-load error, feature NaN, device OOM) becomes a one-line print and an empty `pd.Series`. The empty series is then fed into the outer-joined portfolio, where `.mean(axis=1).dropna()` quietly drops the all-NaN rows. The downstream metrics table and plots contain whatever survived, with no flag.

**Fix.**
- Let exceptions propagate, or
- Maintain a `failures: list[tuple[str, str, str]]` (symbol, method, message-with-traceback) and print a prominent summary at the end before writing the CSV.

**Severity: bug.** Silent data loss in a research experiment is dangerous.

---

### B3. `%+Ret` unit mismatch — `metrics.py:53`

Code: `pct_pos = (r > 0).sum() / n * 100` returns 0–100.
Paper Exhibit 2: `% of + Ret` is a fraction 0–1 (e.g. 0.473).

**Effect.** The CSV mixes a single 0–100 column with nine 0–1ish columns under `%.4f`, so the table silently disagrees with the paper. Also hurts cross-column legibility.

**Fix.** Drop the `* 100` and rename to match paper if desired.

**Severity: bug (units).** Trivial fix; directly affects paper comparability.

---

## Spec deviations (document, or reconsider)

### D1. ~~Reward adaptation~~ Additive vs multiplicative returns — `env.py:138, 148–153`

**CORRECTION (post-review).** My initial review defended the code's use of percentage returns as a "correct adaptation." On closer reading of the paper, that defense was wrong. The paper explicitly addresses this choice and rejects multiplicative returns:

> "We define r_t = p_t − p_{t−1}, and this expression represents additive profits. ... If we want to trade a fraction of our accumulated wealth at each time, multiplicative profits should be used, and r_t = p_t/p_{t−1} − 1. ... We stick to additive profits in our work because logarithmic transformation needs to be done for multiplicative profits to have the cumulative rewards required by the RL setup, but logarithmic transformation penalizes large wealth growth."
> — Zhang et al. (2020), page 5

The paper's Eq. 4 was designed around additive returns. Code uses multiplicative:

```python
# env.py:138
r_t = self._prices[t + 1] / price_t - 1.0   # multiplicative (percentage)
```

**Why this matters:**
1. **Reward scale.** Additive returns for a $1000 contract are ~100× larger than percentage returns (~1.0 vs ~0.01). This shifts the effective learning signal relative to the hyperparameters (γ=0.3, batch size, etc.) that were tuned for additive returns in the paper.
2. **Transaction cost term.** Paper's `bp · p_{t−1} · |Δ(vol_scale · position)|` converts position changes into dollar costs, matching additive reward units. Code drops `p_{t−1}`, making costs unit-less fractions. With additive returns this would be wrong; with multiplicative returns the code's form is internally consistent — but the paper didn't intend multiplicative returns.
3. **Cumulative rewards.** The paper warns that multiplicative returns need a log transform for RL's cumulative-reward framework. The code does not apply any log transform — it sums raw percentage returns, which is an approximation that degrades for large moves.
4. **`μ` parameter.** Paper sets μ=1 as a fixed-dollar multiplier. With percentage returns, μ loses its original interpretation.

Subagent 2 flagged the `p_{t−1}` drop and cost formula as a "significant deviation." My initial review dismissed this, arguing the adaptation was internally consistent. **I now partially agree with the subagent:** while the code is self-consistent within percentage-return units, it contradicts the paper's explicit choice of additive profits and the reward function was not adapted end-to-end (no log transform, no μ reinterpretation).

**Recommendation.** Either:
- (a) Switch to additive returns (`r_t = p[t+1] − p[t]`) and restore `p_{t−1}` in the cost to match the paper faithfully, or
- (b) Keep multiplicative returns but document the deviation, acknowledge the missing log transform, and consider whether hyperparameters (especially γ and bp) need re-tuning for the different reward scale.

**Severity: spec-deviation (significant).** Elevated from "not a deviation" to the most impactful spec deviation in the codebase.

---

### D2. Vol timing: σ_t vs σ_{t−1} — `env.py:141`

Code uses `ewm_vol[t]` (info through close of day `t`) to scale the return `p[t+1]/p[t] − 1`. Paper uses `σ_{t−1}`.

**Both are causal — no lookahead.**
- `features.py:58`: `ewm_vol = daily_ret.ewm(span=60, min_periods=60).std()`, where `daily_ret = close.pct_change()`.
- At env-index `t`, `ewm_vol[t]` is the EWM std of `daily_ret[1..t]`, all of which is realized by close of day `t`.
- The return `p[t+1]/p[t] − 1` is earned after day `t`'s close. Scaling it by end-of-day-`t` vol is legitimate.

Code is one index more *timely* than paper. Defensible.

Subagent 2 called this "lookahead" — I disagree.

**Recommendation.** Add a comment stating the shift so the timing is not mistaken for a bug.

---

### D3. A2C fixed exploration std — `agents.py:201, 230`

Gaussian exploration uses a constant `std=0.2`. Paper doesn't specify. Not state-dependent, not decayed.

**Recommendation.** Either add a `log_std` head (parameterize per-state), or decay `std` over training. Low priority if the current results look reasonable.

**Severity: spec-deviation.**

---

### D4. A2C encoder under actor learning rate — `agents.py:210–215`

```python
self.optimizer = optim.Adam([
    {"params": encoder+actor params, "lr": lr_actor=1e-4},
    {"params": critic params,         "lr": lr_critic=1e-3},
])
```

Gradients from `critic_loss.backward()` flow to the encoder, but Adam applies them with the actor's LR (1e-4), not the critic's (1e-3), because a parameter belongs to exactly one group.

**Not a bug** — just a design choice that's easy to miss when reading. Making the encoder update at 1e-4 may suit actor–critic stability; making it update at 1e-3 would give the critic more say in shaping the encoder.

**Recommendation.** Either accept it (and add a comment) or split the encoder into its own param group with an explicit LR.

---

### D5. Acknowledged deviations (TODO.md)

- Per-symbol training vs paper's per-asset-class models.
- ETFs vs 50 back-adjusted CLC futures.
- Sharpe-based early stopping (paper doesn't specify).
- Single split vs rolling 5-year retraining.

No new comment — already tracked.

---

## Nits / design concerns

- **`env.py:183–184`** — `seq_len<=1` branch returns a 1D state incompatible with the LSTM agents. Exercised only by `smoke.py`'s random/long-only rollout, never by an agent. Either delete the branch or reject `seq_len<=1` when building an RL env.
- **`env.py:143–144`** — When `ann_vol_t` hits the `1e-8` floor, `vol_scale ≈ 1.5e7`, producing massive position/cost spikes. Cap `vol_scale` directly (e.g. 10×) rather than clipping the raw vol.
- **`env.py:73 vs 103`** — Positions tracked in `self.history["position"]` AND `self._pos_history`. Slight redundancy; the history dict is the canonical post-episode record.
- **`features.py:58`** — `ewm(span=60).std()` uses `adjust=True` (pandas default). Paper isn't explicit; `adjust=False` gives the pure recursive formula. Subtle effect on vol estimate.
- **`features.py:94–99`** — `window_ready` is conservative by ~1 row (MACD first valid at row 346, flag flips at row 347). Negligible. (Subagent 1 argued it was off by ~64 rows; their warmup chain was wrong — EMA and the 63-day price std run in parallel on the same prices, not sequentially. See `01_data_and_features.md`.)
- **`features.py:69–70`** — Raw `ret_{h}` columns stored but only `ret_{h}_vol` is in `FEATURE_COLS`. `ret_252` is consumed by `baselines.sign_r`, so not dead. A comment clarifying that would help.
- **`networks.py:19–25`** — `LeakyReLU + Dropout` between LSTM1 and LSTM2 is non-standard. LSTMs usually stack directly (or via `num_layers` on a single `nn.LSTM`). Works, but unusual — flag for re-examination if training is unstable.
- **`agents.py:168`** — `torch.log(probs + 1e-8)` is less numerically stable than `F.log_softmax(logits)`. Cheap switch if NaNs ever appear.
- **`agents.py:55`** — `epsilon` decays on `step_count` (training steps), not env steps. First ~64 env steps run at ε=1.0 (buffer filling). Acceptable, worth a comment.
- **`trainer.py:118–121`** — End-of-episode A2C flush temporarily sets `batch_size=1`. One noisy update at the end of each episode. Consider dropping the leftover buffer instead.
- **`evaluate.py:140–147`** — Passing `--symbols FOO` for a symbol not in `UNIVERSE` silently creates an `"other"` asset class. Validate or warn.
- **`evaluate.py:156`** — Hardcoded `start="2004-01-01", end="2025-12-31"` duplicated from `config.py`. Prefer pulling from config.
- **`baselines.py:54`** — `np.clip(macd_signal, -1.0, 1.0)` is redundant — `φ(x) = x · exp(−x²/4)/0.89` is bounded roughly in [−0.5, 0.5]. Harmless.
- **`baselines.py:79`** — `n = min(len(positions), len(prices)-1)` silently truncates. If a baseline ever emits more positions than prices support, you'd never know. Not urgent.

---

## Wiring pass (cross-file consistency)

- `STATE_DIM = len(FEATURE_COLS) + 1 = 11` in `env.py`; `n_features=11` defaults in `networks.py`; `evaluate.py` factories pass `n_features=STATE_DIM`. Consistent.
- `action_mode` routed correctly: DQN/PG → discrete (3 actions), A2C → continuous; `_decode_action` handles both.
- `device` propagated from `_detect_device()` → agent constructors → `.to(device)` on networks. No implicit CPU tensors.
- Split filtering (`window_ready` + date range) is identical in `env.reset()` and `baselines._get_symbol_split()` — RL and baselines see the same universe.
- Baseline reward math in `compute_baseline_rewards` matches `env.step` exactly (modulo the position-history bug, which is env-only — baselines don't feed positions through a neural net, so they're unaffected).
- Portfolio aggregation in `evaluate.py`: `combined.mean(axis=1)` is NaN-safe by default; `.dropna()` drops only all-NaN rows. Correct for staggered symbol starts.

---

## Priority

1. **D1** (additive vs multiplicative returns) — most impactful deviation; the reward function contradicts the paper's explicit design choice. Decide whether to switch to additive or document the deviation.
2. **B1** (position-history alignment) — highest-priority bug; changes what the policy can learn.
3. **B2** (silent failures in `evaluate.py`) — easy, important for trust in the results table.
4. **B3** (`%+Ret` units) — trivial fix, affects paper comparability.
5. D4 (A2C encoder LR), `env.py:143` vol floor, `networks.py` activation placement — worth a second pass when convenient.
6. Remaining nits as time allows.

---

## Index

- Subagent 1 (data & features) → `01_data_and_features.md`
- Subagent 2 (environment) → `02_environment.md`
- Subagent 3 (networks & agents) → `03_networks_and_agents.md`
- Subagent 4 (training & evaluation) → `04_training_and_evaluation.md`
- Subagent prompts: `../review_prompts/`
