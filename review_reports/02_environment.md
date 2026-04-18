# Subagent 2 — Trading Environment Review

**Scope.** `env.py`
**Prompt.** `../review_prompts/2_environment.md`

> Annotations by the main reviewer are prefixed with **[MAIN]:**. I pushed back on several of this subagent's findings — see annotations.

---

## Subagent overall assessment

The implementation has several significant deviations from the paper's Equation 4 reward function, mostly centering on timing of position and volatility scaling. The most critical issue is **lookahead in the volatility scaling factor**: using `self._ewm_vol[t]` (which includes information up to time t) to scale returns from `p_t → p_{t+1}` violates the causal constraint that scaling should use only historical volatility. Position history indexing also appears problematic. The reward formula uses percentage returns (defensible) but drops `p_{t-1}` from the transaction cost term. Some code structures suggest untested edge cases.

**[MAIN]: Disagree with the lookahead characterization (still wrong). However, the transaction-cost and multiplicative-return flags deserve more credit than I originally gave — see revised annotations below.** The subagent also correctly surfaced a real alignment problem around `_pos_history` timing (though with slightly off-center reasoning).

---

## Findings

### env.py:146 — "Lookahead in vol scaling"

> Subagent: Paper Eq. 4 uses σ_{t-1} (volatility estimate ending the previous day) to scale returns earned during [t-1, t]. Code uses `vol_scale = self.cfg.vol_target / self._ewm_vol[t]` where `ewm_vol[t]` is computed by `features.py:58` as `daily_ret.ewm(span=60, min_periods=60).std()`. This EWM at index t includes information from return at t, making it future-looking relative to the return `r_t = p_{t+1}/p_t - 1` realized during [t, t+1]. ... a form of lookahead bias.
>
> Severity: BUG.

**[MAIN]: Disagree — this is NOT lookahead.**

- `daily_ret[t] = p[t]/p[t−1] − 1` is realized by close of day `t` (uses `p[t]`, known at day-`t` close).
- `ewm_vol[t]` therefore uses only information available at close of day `t`.
- The return the agent earns by the action taken at time `t` is `p[t+1]/p[t] − 1`, realized after day `t` closes.
- Using end-of-day-`t` vol to size a position earning the next-period return is standard practice and causal.

Paper writes `σ_{t−1}`; code's equivalent would be `σ_t` in paper notation — a one-index convention shift, not lookahead. The code is more *timely*, not future-peeking. See `00_overall_review.md` D2.

**Recommendation.** Add an inline comment so the convention is obvious.

---

### env.py:151 — "Position timing in return calculation"

> Subagent: Paper Eq. 4: `R_t = μ · [(σ_tgt/σ_{t-1}) · A_{t-1} · r_t − ...]`, where A_{t-1} is the position held during [t-1, t], yielding return r_t. Code: `position_return = vol_scale * position * r_t` where `position` is the action just taken at time t. This means code multiplies the *newly chosen* position by the return from [t, t+1], which is correct for the forward-looking return. However, combined with the vol_scale lookahead (issue above), the full product is: (σ_tgt / σ_t) * A_t * r_t, which differs from the paper.
>
> Severity: SPEC-DEVIATION.

**[MAIN]: The indexing is semantically equivalent under a one-index shift — still agree on that.**

Paper indexes "position held *during* period t" as `A_{t−1}` (chosen at `t−1`). Code indexes "position chosen at step `t`" as `position` (which then earns return on period `[t, t+1]`). These are the same action under different time labels.

The apparent mismatch between subagent's claim `(σ_t) · A_t · r_t` and paper's `(σ_{t−1}) · A_{t−1} · r_t` disappears once you realize code's `r_t = p[t+1]/p[t]−1` is paper's `r_{t+1} = p_{t+1} − p_t`, and code's `(t)` is paper's `(t)` *after* reindexing actions forward by one.

**No position-timing bug.** However, the subagent is right that code's `r_t = p[t+1]/p[t]−1` (multiplicative) differs from the paper's `r_t = p_t − p_{t-1}` (additive). This is the broader D1 issue — see `00_overall_review.md`.

---

### env.py:152 — "Transaction cost formula deviation"

> Subagent: Paper Eq. 4: `− bp · p_{t-1} · |σ_tgt/σ_{t-1} · A_{t-1} − σ_tgt/σ_{t-2} · A_{t-2}|` includes p_{t-1} (price level) in the cost. Code: `tc = self._bp * abs(vol_scale * position - self._prev_vol_scale * self._position)` omits price entirely. Paper's cost scales with price level (doubling your capital doubles transaction costs), while code treats cost as price-neutral. This is a significant deviation; unclear if intentional normalization or oversight.
>
> Severity: SPEC-DEVIATION.

**[MAIN]: REVISED — partially agree with the subagent.**

My original annotation dismissed this as a "correct adaptation." On closer reading of the paper (page 5), the paper explicitly chose additive returns and rejected multiplicative returns. The subagent was right to flag both the missing `p_{t-1}` and the return-type switch as significant.

The nuance:
- The code IS internally consistent within its own multiplicative-return convention: with percentage returns, positions are capital fractions, reward is in fraction units, and the cost `bp · |Δ(vol_scale · position)|` has matching units. So the code is not dimensionally broken.
- But the paper **explicitly chose additive returns** and built Eq. 4 around them. The `p_{t-1}` in the cost was not optional — it was part of the paper's reward design. Dropping it, together with the switch to multiplicative returns, contradicts the paper's stated methodology.r

**Severity: SPEC-DEVIATION (significant).** See `00_overall_review.md` D1 for full analysis and recommended fixes.

---

### env.py:168 — Position-history index-to-action mapping

> Subagent: Line 168: `self._pos_history[self._t] = position` occurs *after* computing reward at time t and *before* incrementing `self._t`. So `_pos_history[t]` stores the position chosen at index t. Later in `_get_state()` (line 188), `pos_col = self._pos_history[start : self._t + 1]` appends this position history to features for the state at t+1. Semantically: at timestep t+1, your state includes the position you *just took* at t. This is defensible (action becomes observation), but differs from the paper's interpretation.
>
> Severity: SPEC-DEVIATION.

**[MAIN]: Disagree with the characterization — this is actually a BUG, promoted to B1 in the overall review.**

The subagent's claim "at t+1 your state includes the position you just took at t" is the behavior the subagent believed the code has — but it's not what the code actually does.

Trace:
- After step at `old_t = t`: `pos_history[t] = a_t` (line 168), `_t = t+1`.
- `_get_state()` at new `_t = t+1`: `pos_col = pos_history[(t+1 − 59) : (t+2)]` = `pos_history[t−58 : t+2]` (length 60).
- Last element: `pos_history[t+1]` — **never written, still 0**.
- Second-to-last: `pos_history[t] = a_t` — the most recent action.

So the state at `t+1` does NOT include the action just taken in the "current position" slot. The last slot is zero; the action is one step back. The agent never cleanly observes its current position.

**Severity: BUG** — promoted to B1 in the overall review. Fix: slice `pos_history[start − 1 : self._t]`, or write `pos_history[self._t + 1] = position` before incrementing.

---

### env.py:100 — Initial `seq_len-1` skip

> Subagent: `self._t = max(self.cfg.seq_len - 1, 0)` starts at index 59 when `seq_len=60`. This ensures the first call to `_get_state()` can fetch 60 prior values. However, at t=59, `_pos_history[:60]` is all zeros (line 103), meaning the first state includes a "fake" history of zero positions. Is this intended? If agents expect position history to reflect prior actions, the zero padding may confuse them.
>
> Severity: MINOR.

**[MAIN]: Agree — and reasonable as initial condition.** Before any action is taken, "position = 0" is literally correct. More of a note than a concern.

---

### env.py:143–144 — Vol guard floor too small

> Subagent: `if ann_vol_t < 1e-8 or np.isnan(ann_vol_t): ann_vol_t = 1e-8` sets a floor of 1e-8. Then `vol_scale = 0.15 / 1e-8 = 1.5e7`, which inflates position sizes wildly during low-vol regimes or data anomalies. Defensible as an edge-case guard, but document the rationale. Consider clipping `vol_scale` instead of `ann_vol_t` for safety.
>
> Severity: SPEC-DEVIATION.

**[MAIN]: Agree.** Should clip `vol_scale` to a sane range (e.g. 10×) rather than the raw vol. Covered in overall review nits.

---

### env.py:134–135 — Terminal state returns None

> Subagent: `if t >= len(self._prices) - 1: return None, 0.0, True, {}` returns None for next_state. Trainer.py handles this: `ns = next_state if next_state is not None else state`. Zeros out bootstrap for the terminal step. Correct if episode-end is hard boundary.
>
> Severity: OK but verify.

**[MAIN]: Agree.** Terminal bootstrap is zeroed correctly via the `(1 - d)` mask in DQN/A2C target computation. `done` is set to True on the transition that yielded the terminal reward. Acceptable.

One subtle thing the subagent didn't raise: when this early-exit fires at the top of `step` (before any reward computation), it returns `reward=0.0` and an empty `info={}` without appending to `self.history`. That's fine for the legitimate "no actionable step" case but could mask a bug if it ever fires on step 1.

---

### env.py:183–184 — `seq_len=1` returns 1D state

> Subagent: Smoke test (smoke.py:46) does use `seq_len=1`, so this branch is exercised. Agents' networks must handle both 1D and 2D states; confirm the network auto-broadcasts or states are reshape/squeezed elsewhere.
>
> Severity: OK if tested.

**[MAIN]: Disagree — the branch is broken for RL agents.** With `seq_len=1`, state has shape `(n_features+1,)`. Agent `select_action` does `torch.tensor(state).unsqueeze(0)` → `(1, n_features+1)`. The LSTM interprets this as `(batch=1, seq_len=n_features+1, features=??)`, which is wrong and would either error or silently misbehave. It works only in `smoke.py`'s random/long-only rollout because no agent is used. Either delete the branch or raise when an RL agent is paired with `seq_len<=1`.

---

### env.py:32, 195 — Action space + continuous clipping

> Subagent: Standard convention, no issue.

**[MAIN]: Agree.**

---

### env.py:73, 105–112 — `history` dict vs `_pos_history`

> Subagent: Both track positions; slight redundancy.
>
> Severity: NITPICK.

**[MAIN]: Agree.**

---

## UNSURE items (subagent)

> Subagent:
> - env.py:149 — Is the vol-scale *intentionally* using current vol (σ_t) rather than lagged vol (σ_{t-1}), and is this documented as an improvement over Zhang?
> - env.py:152 — Is the omission of `p_{t-1}` from transaction cost intentional or an incomplete adaptation?

**[MAIN]:**
- Vol timing: resolved — causal and one-index-shifted, not lookahead.
- `p_{t-1}` drop: **REVISED.** Originally dismissed as a correct adaptation. The paper explicitly chose additive returns; the code's multiplicative convention and the resulting `p_{t-1}` omission are a significant spec deviation. See updated annotation on env.py:152 above and `00_overall_review.md` D1.

---

## Main reviewer's additional items (not in subagent report)

- **`env.py:73–74`** — The `history` dict is initialized in `reset` but never cleared between resets if `reset` is called multiple times on the same env (which does re-init here, but worth noting). OK as is.
- **`env.py:127, 169`** — The `position` local variable shadows `self._position`. After the reward and history writes, `self._position = position` promotes it to the new state. Readable but easy to confuse during debugging.

---

## Summary severity distribution after annotations

- Bugs: 1 (position-history alignment — promoted to B1).
- Spec deviations: 2 accepted (transaction cost / multiplicative returns — promoted to D1; vol floor). 2 filtered out as incorrect (lookahead claim; position-timing claim — both are indexing-convention differences, not bugs).
- Nits: 2 retained (history/pos_history redundancy, seq_len=1 dead branch).
