# Metrics Reference

What PhoenX can log, what it measures, and how to read it. Coverage is not yet
exhaustive — see [Not yet implemented](#not-yet-implemented) for the gaps.

Metrics are collected by `MetricsManager` and handed to whatever callbacks are
configured. `WandbCallback` forwards them to Weights & Biases verbatim, so the
key names below are exactly what appears in your W&B run.

## How to read this reference

**Two logging paths.** Per-step metrics arrive every environment step through
`on_train_step_end`. Per-episode metrics arrive when an episode finishes,
through `on_train_epoch_end`. Both are logged against the trainer's global
environment step, so everything shares one x-axis.

**Absent, never zero.** A metric with no value in the current window is omitted
entirely rather than logged as `0.0` or `null`. A gap in a W&B chart means "not
recorded", not "recorded as zero". This matters when reading a divergence: a
loss that goes non-finite is dropped and `health/non_finite` flips to `1.0`,
rather than the loss charting as `NaN`.

**Two conditions for a metric to appear.** It must be *enabled* — the agent,
buffer, and environment must have the capability it needs — and something must
actually *record* a value. The "Applies to" column below is the first
condition. The second depends on the code path: a metric produced inside
`learn()` only appears on steps where a learn happened.

**Legacy names.** Metrics that predate the namespaced scheme are emitted twice:
once under the new key and once under the old flat name, so existing charts and
alerts keep working. See [Legacy aliases](#legacy-aliases). Set
`dual_emit: false` in the `metrics:` config block to emit only the new keys.

**Agent names** below use this codebase's classes: `REINFORCE`, `ActorCritic`,
`PPO` (on-policy) and `DDPG`, `TD3`, `SAC` (off-policy).

---

## Episode metrics

Logged once per finished episode. With multiple parallel environments, several
can arrive on the same step.

| Key | What it measures |
|---|---|
| `episode` | Cumulative count of episodes finished across all envs. |
| `env` | Which parallel env index finished. Useful for spotting one env behaving differently from the rest. |
| `episode_steps` | Length of the episode that just finished, in env steps. |
| `episode_reward` | Undiscounted sum of extrinsic reward for that episode. The headline "how good was that run" number. |
| `avg_reward` | Mean `episode_reward` over the last 100 finished episodes. This is the metric the bundled sweeps optimize, and the one the best-model checkpoint tracks. |
| `success_rate` | Fraction of the last 100 episodes that met the configured `success_criterion`. Absent when no criterion is configured. |
| `goal_distance` | Final L2 distance between achieved and desired goal. Goal-conditioned environments only. |
| `best` | Present and true only on the step where `avg_reward` set a new run best. |
| `saved` | Present and true only on the step where a checkpoint was actually written to disk. Gated by `schedule.save_every`, so a new `best` does not always produce a `saved`. |

`avg_reward` is smoother than `episode_reward` and is what you should watch for
learning progress. `episode_reward` alone is noisy enough on most environments
to hide a real trend.

---

## Rollout — windowed episode statistics

Aggregates over finished episodes, computed by the trainer. Available for every
agent.

| Key | What it measures |
|---|---|
| `rollout/return_mean` | Mean episode return over episodes finishing in the current window. |
| `rollout/return_rolling_100` | Mean return over the last 100 finished episodes. Survives across windows, so it is defined even on steps where no episode ended. |
| `rollout/return_std` | Population standard deviation of the last 100 returns. Rising std with flat mean means the policy became inconsistent, which often precedes a collapse. |
| `rollout/ep_length_mean` | Mean episode length in env steps. |
| `rollout/ep_length_std` | Population standard deviation of the last 100 episode lengths. |
| `rollout/termination_rate` | Fraction of finished episodes that ended in a true environment terminal. |
| `rollout/truncation_rate` | Fraction that ended in a time-limit cutoff. |

**Why termination and truncation are separate.** They bootstrap differently: a
true terminal has no future value, a time-limit cutoff does. Watching the split
tells you what your agent is actually experiencing. On a task the agent is
supposed to solve, a rising termination rate with rising return means it is
reaching the goal; a termination rate near 1.0 with *low* return usually means
it is failing out early. The two rates sum to 1.0 over any window containing at
least one finished episode — an episode flagged both ways counts as a
termination.

---

## Throughput

| Key | What it measures |
|---|---|
| `perf/sps` | Environment steps per second of wall clock, measured over the interval between emissions. Absent on the first emission, since a rate needs two samples. |

A gradual `perf/sps` decline over a long run is worth investigating — it usually
means something is growing without bound.

---

## Buffer

Occupancy and readiness of whatever buffer is configured. These apply to
**on-policy and off-policy agents alike**; capacity is a property of the
buffer, not of the algorithm.

| Key | Applies to | What it measures |
|---|---|---|
| `buffer/fill` | any buffer | Fraction of allocated timestep capacity holding valid data. |
| `buffer/env_steps` | any buffer | Cumulative env steps recorded into the buffer since it was constructed or restored. Not reset by a rollout flush. |
| `buffer/is_ready` | any buffer | `1.0` when the buffer can supply a full batch, `0.0` otherwise. |
| `buffer/pending_trajectories` | `TrajectoryBuffer` (typically REINFORCE) | Number of entries in the completed-trajectory list awaiting a learn. |

**`buffer/fill` has one definition and three behaviors. Do not assume
monotonicity.**

- `ReplayBuffer` / `PrioritizedReplayBuffer` — climbs from 0 toward 1.0 and
  pins there once the buffer wraps. A low value early means you are still
  filling; a low value late means something is wrong.
- `RolloutBuffer` — a sawtooth. Climbs across a rollout, snaps to 0 at each
  `sample()`. The **peak** is the number to read: if it peaks well below 1.0
  you have allocated more buffer than the rollout uses, and if it approaches
  1.0 you are close to the point where `learn_every / num_envs` exceeds
  `buffer_size` and indexing will raise.
- `TrajectoryBuffer` — tracks the longest still-in-progress episode. It does
  *not* reset at `sample()`; it drops only when the env holding that episode
  finishes it.

An alert like "fill below 0.1 means still warming up" is correct for replay and
wrong for the other two.

**`buffer/is_ready` is not the warmup gate.** It reports the buffer's own
readiness. `schedule.warmup_steps` is a separate gate that blocks learning
before the buffer is even consulted, so `buffer/is_ready` can read `1.0` while
no learning is happening. For on-policy buffers it is always `1.0`.

**`buffer/pending_trajectories` counts list entries, not episodes.** With
hindsight replay configured, relabeled copies land in the same list, so the
count exceeds the number of episodes actually played.

---

## Losses

| Key | Applies to | What it measures |
|---|---|---|
| `loss/policy` | all | The policy/actor objective actually backpropagated. |
| `loss/value` | agents with a value head | Value-function regression loss. |
| `loss/critic` | agents with a critic | Critic (Q-function) loss. For twin-critic agents this is the combined term. |

Absolute loss values mean little across algorithms — the useful signal is the
trend and any sudden discontinuity. A policy loss near zero is not necessarily
good; for PPO it often just means the clipping is saturated.

---

## Critic and value estimation

| Key | Applies to | What it measures |
|---|---|---|
| `critic/td_error_mean` | all | Mean temporal-difference error on the training batch. |
| `critic/value_pred_mean` | agents with a critic | Mean predicted value/Q over the batch. |
| `value/pred_mean` | on-policy agents with a value head | Mean predicted state value V(s) over the batch. |
| `critic/return_mean_batch` | all | Mean target return over the same batch. |

**`critic/value_pred_mean` versus `critic/return_mean_batch` (off-policy
only).** `critic/value_pred_mean` drifting persistently above
`critic/return_mean_batch` is the classic signature of Q-value
overestimation, which is what twin critics exist to suppress. A widening gap on
an off-policy run is worth acting on well before the return curve degrades.
This reading applies only to the off-policy critic key: `REINFORCE`,
`ActorCritic`, and `PPO` never log `critic/value_pred_mean` at all, since none
of them owns a `critic` branch.

`critic/td_error_mean` is a *signed* mean, so cancellation can hide large
errors; it trending away from zero indicates systematic bias rather than noise.

**`value/pred_mean` is a scale check, not a calibration pair with
`critic/return_mean_batch`.** On-policy returns are built as
`returns = advantages + values`, so over the same batch
`mean(returns) - mean(value_pred)` is exactly `mean(advantages)` —
already logged as `policy/advantage_mean` — for both GAE-based agents
(`ActorCritic`, `PPO`) and REINFORCE's Monte Carlo returns. Comparing
`value/pred_mean` against `critic/return_mean_batch` on an on-policy run tells
you nothing that `policy/advantage_mean` does not already tell you more
directly. What `value/pred_mean` genuinely gives you is the **absolute scale
and drift of the value function**: a value head exploding toward large
magnitudes or collapsing toward zero shows up here and nowhere else —
`policy/advantage_mean` stays small either way, and `loss/value` is
scale-confounded, since a large loss can mean a large error or simply a large
target. It is also the natural sanity check against reward scale and discount:
a bounded-reward task with a given `gamma` has a knowable range for V(s), and a
value head far outside that range is misconfigured. Because V(s) and Q(s, a)
are different quantities, on-policy runs chart `value/pred_mean` and
off-policy runs chart `critic/value_pred_mean` — treat them as two distinct
metrics that happen to serve a similar diagnostic purpose, not one metric
under two names.

---

## Policy

| Key | Applies to | What it measures |
|---|---|---|
| `policy/entropy` | REINFORCE, ActorCritic, PPO, SAC | Mean action-distribution entropy. |
| `policy/advantage_mean` | on-policy | Mean advantage over the batch. |
| `policy/kl` | on-policy | Mean KL divergence between the old and updated policy. |
| `policy/temperature` | discrete action spaces | Current categorical sampling temperature, including any schedule factor. |

**Entropy is your exploration gauge.** It should decline gradually. A sharp
early collapse means the policy committed before it explored and the run is
usually already lost; entropy that never declines means the policy is not
converging. Which direction is a problem depends on the task, but a
discontinuity in either is worth investigating.

`policy/kl` measures how far each update moved the policy. Large spikes mean
updates that are too aggressive, which for on-policy methods is a common cause
of sudden collapse.

---

## PPO-specific

| Key | What it measures |
|---|---|
| `ppo/ratio_mean` | Mean probability ratio between the new and old policy. |
| `ppo/policy_clip` | Current policy clipping coefficient, including any schedule factor. |
| `ppo/value_clip` | Current value clipping coefficient, including any schedule factor. |
| `onpolicy/entropy_coef` | Current entropy bonus coefficient. |
| `onpolicy/kl_coef` | Current KL penalty coefficient. |

`ppo/ratio_mean` should sit near 1.0. Persistent drift away from it means the
policy is moving further per update than the clipping intends.

Log the coefficients even when you think they are constant — a schedule that
is silently misconfigured is one of the easier ways to lose a run without any
error appearing.

---

## SAC-specific

| Key | What it measures |
|---|---|
| `sac/alpha` | Current entropy temperature. Under automatic tuning this is `log_alpha.exp()` and moves every update. |

Alpha collapsing toward zero means the policy has stopped being rewarded for
exploring; alpha growing without bound means it cannot satisfy the target
entropy. Both show up here before they show up in the return curve.

---

## Exploration noise — DDPG and TD3

| Key | Applies to | What it measures |
|---|---|---|
| `exploration/noise_anneal` | DDPG, TD3 | Current exploration-noise schedule factor. |
| `td3/target_noise_anneal` | TD3 | Current target-policy-smoothing noise schedule factor. |

These are schedule factors, not the noise standard deviation itself. They exist
because a broken schedule is silent otherwise.

---

## Optimizer

| Key | Applies to | What it measures |
|---|---|---|
| `optim/lr/policy` | all | Current policy learning rate, read from the live optimizer. |
| `optim/lr/value` | agents with a value head | Current value learning rate. |
| `optim/lr/critic` | agents with a critic | Current critic learning rate. |
| `optim/lr/critic_b` | twin-critic agents (TD3, SAC) | Current second-critic learning rate. |
| `optim/updates_per_env_step` | all | Optimizer updates performed per environment step during the current run segment. |

Learning rates are read from `optimizer.param_groups`, not recomputed from the
schedule, so what you see is what the optimizer actually used.

**`optim/updates_per_env_step` is not the replay ratio of the literature.** It
counts coordinated optimizer steps, and different algorithms structure those
differently: off-policy agents take two per learn call (critic modules, then
policy), TD3 skips its policy step on delayed iterations, and PPO takes one per
minibatch. It is a consistent measure of how hard you are training per unit of
collected data, and it is comparable across runs of the *same* algorithm; do
not compare its absolute value across algorithm families. The temperature and
intrinsic-motivation optimizers are not counted, since neither goes through the
shared model.

A *run segment* is the stretch since the current `train()` call began — the
whole run normally, or everything since the resume after `Trainer.load`. Both
the numerator and denominator are scoped that way because the environment step
counter is restored from the checkpoint while the optimizer counter is not.
Expect this series to restart from its own baseline after a resume rather than
continuing the previous curve.

---

## Prioritized experience replay

Only when the buffer is a `PrioritizedReplayBuffer`.

| Key | What it measures |
|---|---|
| `per/beta` | Current importance-sampling exponent, annealed over training. |
| `per/is_weight_mean` | Mean importance-sampling weight applied to the batch. |
| `per/is_weight_max` | Largest weight in the batch. |
| `per/is_weight_std` | Spread of the weights. |
| `per/sampled_priority_mean` | Mean priority of the transitions actually sampled. |
| `per/sampled_priority_max` | Largest sampled priority. |
| `per/buffer_priority_mean` | Mean priority across the whole buffer. |
| `per/buffer_priority_max` | Largest priority anywhere in the buffer. |
| `per/prob_mean` | Mean sampling probability of the batch. |
| `per/prob_max` | Largest sampling probability in the batch. |
| `per/priority_entropy` | Shannon entropy of the buffer's sampling distribution, normalized to `[0, 1]`. |

**`per/priority_entropy` is the one to watch.** `1.0` means every transition is
equally likely to be sampled — prioritization is doing nothing. Values falling
toward `0` mean sampling has collapsed onto a handful of transitions, which
starves the rest of the buffer and usually shows up as the critic overfitting a
narrow slice of experience.

Compare `per/sampled_priority_mean` against `per/buffer_priority_mean` to see
how aggressively sampling is skewed: the further apart, the more selective.

---

## Intrinsic motivation

Only when an intrinsic-motivation module is configured.

| Key | What it measures |
|---|---|
| `step_intrinsic_reward` | Mean intrinsic reward generated this env step. `0.0` when no module is configured. |
| `intrinsic_reward` | Mean intrinsic reward over the rollout used for the update. |
| `learn_intrinsic_reward` | Mean intrinsic reward recomputed at learn time. |
| `intrinsic_loss` | Training loss of the intrinsic-motivation module itself. |
| `reward_weight` | Current intrinsic reward weight, including any schedule factor. |

Intrinsic reward should decay as states stop being novel. A curiosity signal
that stays flat is usually not measuring novelty at all.

---

## N-step and retrace diagnostics

Present only when the relevant machinery is configured.

| Key | What it measures |
|---|---|
| `nstep/avg_trajectory_length` | Mean length of the n-step windows actually collected. Shorter than configured means episodes are ending inside the window. |
| `nstep/pct_short_windows_after_term` | Fraction of windows truncated by a terminal. |
| `nstep/avg_final_cum_c_on_done_windows` | Mean final cumulative retrace coefficient on windows containing a terminal. |
| `nstep/max_leakage_in_mask_after_done` | Largest amount of value that leaked past a terminal boundary. Should be at or very near zero. |

`nstep/max_leakage_in_mask_after_done` is a correctness check, not a tuning
knob. A non-trivial value means credit is flowing across an episode boundary.

---

## Raw step and update values

| Key | Applies to | What it measures |
|---|---|---|
| `step_reward` | all | Mean extrinsic reward across envs this step. |
| `extrinsic_rewards` | off-policy | Mean extrinsic reward over the sampled batch. |
| `policy_predictions` | off-policy | Mean action produced by the policy on the batch. |
| `target_policy_predictions` | DDPG, TD3 | Mean action from the target policy. |
| `target_critic_predictions` | off-policy | Mean target-network Q value. |
| `env_step` | all | Trainer global environment step at emission. |
| `grad_step` | all | Number of `learn()` calls so far this run. |

`env_step` is the primary x-axis. `grad_step` counts learn *calls*, which is a
coarser unit than optimizer steps — use `optim/updates_per_env_step` if you
care about actual update volume.

---

## Health

| Key | What it measures |
|---|---|
| `health/non_finite` | `1.0` if any loss recorded in the window was `NaN` or infinite, `0.0` if losses were recorded and all were finite. Absent when no loss was recorded. |

Because a non-finite value is dropped rather than charted, this flag is how you
find out a divergence happened. It is worth an alert.

---

## Logged automatically by W&B

`WandbCallback` calls `wandb.watch(..., log="all", log_freq=100)` on the agent's
models, so W&B additionally records gradient and parameter **histograms** under
its own `gradients/` and `parameters/` prefixes. Those are useful for spotting a
dead layer, but they are histograms rather than scalars — you cannot alert on
them or compare them across runs the way you can the keys above.

---

## Not yet implemented

Deliberately absent, so you know these gaps are known rather than broken:
per-module gradient norms, pre-clip gradient norms, parameter norms,
update-to-weight ratios, recurrent-specific diagnostics (hidden-state norms,
gate saturation), and a held-out evaluation harness for off-policy explained
variance.

---

## Legacy aliases

These older flat names are emitted alongside the namespaced keys so existing
charts keep working. Prefer the new names for anything you build now.

| Legacy name | New key |
|---|---|
| `policy_loss` | `loss/policy` |
| `value_loss` | `loss/value` |
| `critic_loss` | `loss/critic` |
| `temporal_difference`, `td_error` | `critic/td_error_mean` |
| `critic_predictions` | `critic/value_pred_mean` |
| `value_predictions` | `value/pred_mean` |
| `returns` | `critic/return_mean_batch` |
| `advantages` | `policy/advantage_mean` |
| `entropy` | `policy/entropy` |
| `kl` | `policy/kl` |
| `prob_ratio` | `ppo/ratio_mean` |
| `policy_clip` | `ppo/policy_clip` |
| `value_clip` | `ppo/value_clip` |
| `entropy_coefficient` | `onpolicy/entropy_coef` (on-policy), `sac/alpha` (SAC) |
| `kl_coefficient` | `onpolicy/kl_coef` |
| `policy_learning_rate` | `optim/lr/policy` |
| `value_learning_rate` | `optim/lr/value` |
| `critic_learning_rate` | `optim/lr/critic` |
| `critic_b_learning_rate` | `optim/lr/critic_b` |
| `noise_anneal` | `exploration/noise_anneal` |
| `target_noise_anneal` | `td3/target_noise_anneal` |
| `temperature` | `policy/temperature` |
| `PER/beta` | `per/beta` |
| `PER/weight_mean` | `per/is_weight_mean` |
| `PER/weight_std` | `per/is_weight_std` |
| `PER/sampled_priority_mean` | `per/sampled_priority_mean` |
| `PER/sampled_priority_max` | `per/sampled_priority_max` |
| `PER/buffer_priority_mean` | `per/buffer_priority_mean` |
| `PER/buffer_priority_max` | `per/buffer_priority_max` |
| `PER/prob_mean` | `per/prob_mean` |
| `PER/prob_max` | `per/prob_max` |

---

## Controlling what gets logged

The optional top-level `metrics:` block configures collection. See
[Configuration Files](configurations.md) for the full schema.

```yaml
metrics:
  emit_every: 1        # env steps between emissions; raise to cut W&B traffic
  dual_emit: true      # also emit the legacy names above
  exclude: []          # exact keys to suppress, aliases included
  include: null        # exact allow-list; null or [] means "no allow-list"
```

`include` is the one key here that is not a filter you can leave empty and
forget. Setting it to a **non-empty** list restricts logging to exactly those
keys; `null` and `[]` both mean "no allow-list, gate on `tiers` instead".

If a run ever logs only flat legacy names and no namespaced ones, the cause is
almost always that gating enabled nothing — an `include` naming no registered
key, an empty `tiers`, or an `exclude` covering everything. With no metric
enabled, values still reach the logger through the unmapped passthrough under
their own flat names, so the failure looks like "the old metrics came back"
rather than like silence. `MetricsManager` logs a warning when this happens.

Raising `emit_every` is the first thing to reach for if W&B logging becomes a
bottleneck — window state is retained across suppressed emissions, so
`MEAN`-reduced metrics still average over everything that happened.
