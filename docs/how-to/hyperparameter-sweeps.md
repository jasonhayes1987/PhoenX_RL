# Hyperparameter Sweeps

PhoenX drives Ray Tune sweeps from a **standalone YAML** that never edits
the training config on disk. Each trial deep-copies the base config in
memory, applies overrides and sampled values, then trains. Run sweeps with
`phoenx-tune`.

API surface: [phoenx.ray_tune][]. Trial reporting uses
[RayTuneCallback][phoenx.rl_callbacks.RayTuneCallback] (injected per trial;
documented under [RL callbacks][phoenx.rl_callbacks]).

## Quick start

Validate first — sample and fully resolve trial configs, print the dotted
path diff against the base, and exit without starting Ray or training:

```bash
phoenx-tune --config lunarlander_ppo.yml --validate-only
```

Then run the bundled LunarLander PPO multi-phase example (small
`num_samples` / `stop_units` for a CPU smoke test):

```bash
phoenx-tune --config lunarlander_ppo.yml
```

Bundled sweeps live under `phoenx/examples/sweeps/` (package data, separate
from training configs under `phoenx/examples/configs/`). List them with
[available_example_sweeps][phoenx.ray_tune.available_example_sweeps]:

| Bundled name | Role |
|--------------|------|
| `lunarlander_ppo.yml` | Three-phase PPO sweep on LunarLanderContinuous-v3 |
| `isaac_franka_cube_lift.yml` | Three-phase PPO sweep on Franka cube-lift (Isaac) |

An on-disk path always wins; otherwise a relative `--config` is looked up
among those bundled names (same resolution pattern as
[load_config][phoenx.builder.load_config]).

When the sweep finishes, the root
`<storage-path>/<name>/best_config.yml` is directly runnable:

```bash
phoenx-train --config ray_results/ppo_sweep/best_config.yml
```

## Anatomy of a sweep

Top-level keys:

| Key | Role |
|-----|------|
| `base_config` | Training YAML (bundled name or path). Required. |
| `overrides` | Dotted path → constant, applied to every trial of every phase |
| `defaults` | Inherited by every phase; a phase's own key wins |
| `blocks` | Reusable layer-template library for architecture search |
| `phases` | Ordered list of phases run in series |
| `ray_init` | Optional kwargs forwarded to `ray.init()` (e.g. `runtime_env.env_vars`) |

A sweep with no `phases:` but a top-level `search_space` / `tune` /
`architecture` / `optimizers` is treated as one implicit phase named
`phase_0`, so a one-phase sweep stays short.

Minimal single-phase shape (illustrative — prefer the bundled files for a
full runnable example):

```yaml
base_config: LunarLanderContinuous-v3/ppo.yml

overrides:
  schedule.stop_units: 300000
  log_level: ERROR

defaults:
  metric: avg_reward
  mode: max
  resources: {cpu: 1, gpu: 0}
  max_concurrent_trials: 2
  report: {every: 20000, unit: timestep}
  auto_learn_every: true

search_space:
  agent.config.discount: {dist: uniform, low: 0.95, high: 0.999}

tune:
  num_samples: 8
  search_alg: {type: random}
```

### Search specs use `dist:`, not `type:`

`type:` already discriminates envs, agents, buffers, layers, heads, and
optimizers in a training config. Reusing it inside a search spec would be
ambiguous, so every sampler is declared with `dist:`.

Supported values map 1:1 onto Ray samplers:

| `dist` | Required keys | Notes |
|--------|---------------|-------|
| `uniform` | `low`, `high` | |
| `loguniform` | `low`, `high` | |
| `quniform` | `low`, `high`, `q` | |
| `randint` | `lower`, `upper` | **Upper bound is exclusive** (Ray semantics) |
| `qrandint` | `lower`, `upper`, `q` | Upper exclusive |
| `lograndint` | `lower`, `upper` | Upper exclusive; optional `base` |
| `choice` | `values` | |
| `grid_search` | `values` | |
| `randn` | — | Optional `mean`, `sd` |
| `fixed` | `value` | Constant (not a sampler) |

`randint: {lower: 1, upper: 4}` therefore samples `{1, 2, 3}`, not `4`.
Unknown `dist` values, unknown keys inside a spec, and missing required
keys are all rejected with `ValueError` — a silently ignored typo would
produce a plausible but meaningless sweep.

### Dotted paths

Paths address anything in the training config
(`agent.config.discount`, `schedule.learning_epochs`,
`env.config.num_envs`). List indices work in both `a.b.0.c` and `a.b[0].c`
forms.

A path whose intermediate segment does not exist **raises** rather than
being created. A typo would otherwise leave the trial running at the base
value and look like a legitimate result. The only create-on-demand writer
is the `optimizers:` section (it must invent missing per-module
`optimizer_params` blocks).

**YAML-anchor gotcha.** Anchors in the base config are resolved at parse
time. An `overrides` entry for an anchored key (for example a top-level
`device: &device`) does not retroactively change places that already
dereferenced `*device`.

## Multi-phase sweeps and promotion

Each phase is one `tune.Tuner` under
`<storage_path>/<sweep name>/<phase name>/`, executed in series. Every
phase writes `best_config.yml` and `phase_summary.json`.

```yaml
phases:
  - name: architecture
    # ... search space ...
    promote: {mode: best, seed_next: 2}

  - name: optimizers
    # ...
    promote: {mode: best, seed_next: 3}

  - name: refine
    # ...
    promote: {mode: best}
```

`promote: {mode: best}` makes the winning trial's **fully resolved**
config the next phase's base. Anything a phase searched and the next
phase does not stays frozen at the winner's value; anything searched again
is re-searched.

`promote` is schema-validated before any compute: an unsupported `mode`,
an unknown key, or `seed_next` without `mode: best` is rejected. The only
supported mode today is `best`.

The winner is chosen by each trial's **last** reported metric value (Ray's
default). That is intentional here: the reported `avg_reward` is already a
100-episode rolling mean, so ranking on peak values would reward noise
spikes.

### Seeding with `seed_next`

`seed_next: k` additionally passes the top-k sampled configs to the next
phase's searcher as `points_to_evaluate`. Filtering rules:

1. Seeds keep only keys the next phase actually searches.
2. `arch.*` keys carry over only when the next phase searches the
   **identical** `architecture:` section; otherwise they are skipped with
   a log line.
3. A partial point is passed only to `search_alg: random`
   (`BasicVariantGenerator`). Optuna and HyperOpt require a point to cover
   the entire search space and raise otherwise — partial points are
   dropped for those searchers.
4. When nothing survives the filter, seeding is skipped with a log line.

**Practical consequence:** when consecutive phases search disjoint key
sets — as both bundled examples do — `seed_next` has nothing to carry and
is a no-op. Promotion of the winner's resolved config is what actually
carries information forward.

Resume a later phase after earlier ones finished:

```bash
phoenx-tune --config lunarlander_ppo.yml --from-phase optimizers
```

`--from-phase` loads the preceding phase's promoted `best_config.yml`.
`--resume` restores the starting phase's own interrupted `Tuner` run when
one already exists on disk.

## Architecture search (block grammar)

Architecture search does **not** freely choose among every registered
layer type — that mostly generates degenerate stacks. Instead you declare
small fixed **blocks** (templates) whose parameters are searched, then
compose modules from candidate blocks.

Per module (`roots.<name>`, `trunk`, `branches.<role>`):

- `depth` — how many blocks to stack (searchable)
- `blocks` — candidate block names from the top-level `blocks:` library
- optional `prefix` / `suffix` — fixed layers around the stack
- optional `share_across_slots: true` on a param — collapse it to one
  dimension per module
- optional `max_count` on a block — cap how often it may appear (used for
  at most one temporal block)

Encoding is a flat max-depth Ray space (`arch.<module>.depth`,
`arch.<module>.slot{i}.block`, …) rather than `tune.sample_from`, because
only `BasicVariantGenerator` supports `sample_from` — Optuna and HyperOpt
would reject it.

Two facts that bite first-time authors:

1. Temporal layers (`lstm`, `gru`, causal `transformer_encoder`) are legal
   **only in the trunk**.
2. `flatten` is **not** auto-inserted inside a stack. A conv → dense stack
   needs an explicit `flatten` (both bundled examples show this; the
   Isaac camera root uses a `suffix` with `flatten`).

Excerpt from `lunarlander_ppo.yml`:

```yaml
blocks:
  dense_block:
    layers:
      - type: dense
        params:
          units: {dist: choice, values: [64, 128, 256]}
          kernel: {dist: choice, values: [orthogonal, default]}
      - type: {dist: choice, values: [relu, tanh]}

phases:
  - name: architecture
    architecture:
      branches.policy:
        depth: {dist: randint, lower: 1, upper: 4}
        blocks: [dense_block]
      branches.value:
        depth: {dist: randint, lower: 1, upper: 4}
        blocks: [dense_block]
```

**Dimensionality warning.** Five modules at depth four with three params
each approaches ~200 dimensions, which degrades Bayesian search badly. Use
random search plus ASHA for architecture, settle architecture in an early
phase, then run a Bayesian phase on training hyperparameters.

**Mixing temporal and feedforward blocks needs a compatible base config.**
Offering a recurrent block (`gru_block`, `lstm_block`) alongside a dense one
in the same trunk means some trials sample a recurrent trunk and some do
not. `mini_batch_size` is interpreted in *timesteps* for a feedforward model
but in *env units* for a recurrent one, so a base config whose
`mini_batch_size` was tuned for a feedforward model makes every
recurrent-sampling trial fail constraint validation. Either sweep
`schedule.mini_batch_size` alongside the architecture with a range valid for
both, or keep temporal and feedforward architectures in separate sweeps.

## Per-module optimizers

Every module already gets its own optimizer
(`roots.<name>`, `trunk`, `branches.<role>`). A training config that only
declares model-wide `optimizer_params` makes all of them share one LR —
there is no per-module key to write into. The sweep's `optimizers:`
section creates those blocks:

```yaml
optimizers:
  branches.policy: {lr: {dist: loguniform, low: 1.0e-5, high: 1.0e-3}}
  branches.value:  {lr: {dist: loguniform, low: 1.0e-5, high: 1.0e-3}}
```

`lr` and any other field land under `params`; `type` sits at the top level
of `optimizer_params`. An unswept `type` inherits module-own → model-wide
→ `Adam`.

## Searchers, schedulers, and stoppers

Searchers and schedulers delegate to Ray's `create_searcher` /
`create_scheduler`, so all ten searchers and every scheduler alias are
available (Ray raises if a searcher-specific extra is missing). Use
`asha` as the friendly alias for AsyncHyperBand with
`time_attr: timestep`.

**PBT is not yet supported.** `scheduler.type: pbt` (or
`population_based_training`) raises `ValueError` with a clear message —
checkpoint plumbing lands in a later delivery pass.

`stop:` accepts:

- a plain metric-threshold dict, e.g. `{avg_reward: 250}`
- a single `{type: TrialPlateauStopper, ...}` (also
  `ExperimentPlateauStopper`, `MaximumIterationStopper`,
  `TimeoutStopper`, `CombinedStopper`)
- a list of either, combined via `CombinedStopper` (OR semantics)

## Constraint validation

Constraints run at trial resolution, before anything is built, so a bad
combination fails in milliseconds with an actionable message.

| Rule | Why |
|------|-----|
| `auto_learn_every: true` (default) sets `learn_every = buffer_size * num_envs` for rollout-buffer runs | `learn_every` counts global steps while the buffer allocates `(buffer_size, num_envs, ...)` and `add` has no wrap and no bounds check. Sweeping `num_envs` while leaving `learn_every` fixed either raises `IndexError` mid-run or silently trains on partial rollouts. |
| Reject `mini_batch_size` larger than the feedforward rollout batch | Feedforward PPO computes `num_valid // mini_batch_size`; zero batches means zero learning while a sweep misreads it as a bad hyperparameter. |
| Reject non-positive `mini_batch_size` | Same division would raise or yield zero batches. |
| For recurrent models, `mini_batch_size` is in env units and must divide `num_envs` | The silent fallback to all envs makes the swept value a lie. |
| SAC N triple must agree: `agent.config.N`, `buffer.config.N`, and `VectorNStepReward` wrapper `n` | Otherwise the return computation, replay window, and reward wrapper disagree about the horizon. |

## CLI flags

| Flag | Effect |
|------|--------|
| `--config` | Sweep YAML path or bundled name (required) |
| `--base-config` | Override `base_config` |
| `--num-samples` | Override every phase's `tune.num_samples` |
| `--max-concurrent` | Override every phase's `max_concurrent_trials` |
| `--name` | Sweep id; artifacts under `<storage-path>/<name>/<phase>/` |
| `--storage-path` | Ray experiment root (default `./ray_results`) |
| `--from-phase` | Resume from this phase using the preceding `best_config.yml` |
| `--resume` | Restore the starting phase's interrupted `Tuner` if present |
| `--validate-only` | Sample/resolve N configs per phase; print diffs; no training |
| `--log-level` | Logging level (default `INFO`) |

Recommend `--validate-only` as the first command on any new sweep.

## Operational notes

**Windows paths.** Trial directories are deliberately short (`trial_00001`).
Ray's default embeds every sampled param in the path and would exceed
`MAX_PATH` mid-sweep.

**Weights & Biases.** Add `wandb: {project: my-project}` under `defaults`
or a phase. Each trial gets a full-fidelity `WandbCallback` run with an
explicit name and a per-phase group; swept params appear under a
`sweep/*` prefix so parallel-coordinates and parameter-importance panels
work. Ray's own `WandbLoggerCallback` is **not** used — it could only log
what is inside `tune.report`, whereas PhoenX's callback logs every step
and episode metric, per-module LRs, and best-model artifacts. The base
config's `schedule.save_every` cooldown (default `50_000` timesteps)
applies per trial automatically, and since uploads follow checkpoints a
sweep picks up the throttle with no schema change — which matters because
a sweep multiplies checkpoint and upload cost by its trial count. An
`artifact_every` set on the base config's `WandbCallback` does **not**
carry over: the trial's callback is rebuilt by the sweep engine, so only
`schedule.save_every` throttles a sweep today. Leave
`wandb` out (as `lunarlander_ppo.yml` does) when you do not want
credentials required to run the example.

**Isaac Sim concurrency.** `max_concurrent_trials` and fractional
`resources: {gpu: 0.25}` are untested knobs here — no PhoenX benchmark has
been run. Isaac Lab's own Ray integration allocates one trial per GPU, and
its maintainer reports two Isaac Sim instances on one GPU interfering and
effectively running in series
([IsaacLab#2989](https://github.com/isaac-sim/IsaacLab/issues/2989)).
Ray's fractional GPU request is bookkeeping only: it partitions neither
VRAM nor SM time. The bundled Isaac sweep keeps `max_concurrent_trials: 1`
and `resources: {gpu: 1}` for that reason. Pass Isaac paths to workers via
`ray_init`:

```yaml
ray_init:
  runtime_env:
    env_vars:
      ISAACLAB_PATH: "/absolute/path/to/IsaacLab"
      PYTHONPATH: "/absolute/path/to/IsaacLab/source"
```

## Related

- Training YAML schema: [Configuration Files](configurations.md)
- Isaac Lab setup: [Isaac Sim Environments](isaac-sim.md)
- Module API: [phoenx.ray_tune][]
