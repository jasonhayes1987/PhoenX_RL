# Configuration Files

Every training and evaluation run is defined by a YAML file. You keep your own
configs wherever you like on disk and pass a path to `--config`. PhoenX also
ships four portable example configs as package data under
`phoenx/examples/configs/`; a relative name that is not an on-disk path is
resolved against that set.

## Bundled example configs

| Bundled name | Role |
|--------------|------|
| `LunarLander-v3/reinforce.yml` | REINFORCE on LunarLander-v3 |
| `LunarLanderContinuous-v3/ppo.yml` | PPO on LunarLanderContinuous-v3 |
| `LunarLanderContinuous-v3/sac.yml` | SAC on LunarLanderContinuous-v3 |
| `IsaacSim/franka/cube_lift/dense/ppo_camera.yml` | PPO + camera on Franka cube-lift (Isaac) |

All four use relative `save_dir` values (for example `./Trained_Models/...`)
and are portable as-is. List the names at runtime with
[available_example_configs][phoenx.builder.available_example_configs].

Each of those examples lists `WandbCallback` under `callbacks`. Authenticate
before a run (precedence order): `WANDB_API_KEY`, a `wandb_api_key` file next
to the installed `phoenx` package, or a prior `wandb login`. Prefer the env
var or `wandb login` — the key file holds a live credential. Details:
[Getting Started](getting-started.md#train-an-agent).

## How `--config` is resolved

[load_config][phoenx.builder.load_config] resolves the value in this order:

1. An existing on-disk path always wins.
2. Otherwise, if the value is not absolute, it is looked up among the bundled
   examples under `phoenx/examples/configs/` (including nested paths).
3. Otherwise `FileNotFoundError`, listing the bundled names.

```bash
# Bundled name — works from a bare pip install
phoenx-train --config LunarLanderContinuous-v3/sac.yml

# Any path on your machine
phoenx-train --config path/to/my_experiment.yml
```

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
phoenx-train --config LunarLanderContinuous-v3/sac.yml
```

## Editing a config in the GUI

Open the window from the environment where PhoenX is installed. The
installer for the window does not include Python or PhoenX. See
[The PhoenX window](window.md).

```bash
phoenx-config gui                      # start blank, or New > From example
phoenx-config gui path/to/my_run.yml   # open an existing config
```

The editor is generated from the same schema as `phoenx-config validate`, so
every setting PhoenX understands is there: dropdowns for agent, head, layer,
normalizer and buffer types (the fields below change with the choice), number
and text fields that start at the value PhoenX would use anyway, add / remove /
reorder buttons for layer lists and model branches, and on/off switches for
optional sections such as normalizers and schedules. Hover a field name for
the description from the code's docstring.

- **YAML preview** (right): exactly what Save writes. Every key the opened
  file had is written back, even one equal to its default (sweeps can only
  change values a config sets explicitly). A field you never set is written
  only if you change it from the default; clearing a field removes it. Tick
  **Write default values** to write everything.
- **Problems** (bottom): updates as you type: input that doesn't parse, unknown
  keys, schema errors, n-step mismatches. Double-click one to jump to its
  field. Save warns before writing a config with problems.
- **Show only set fields** hides everything the config does not set, which is
  a quick way to see what it actually changes.
- **Right-click a field name** to send it to Sweep Builder (see
  [Hyperparameter sweeps](hyperparameter-sweeps.md#building-a-sweep-in-the-gui)).

Comments and YAML anchors are not preserved when an existing file is saved;
anchored values are written out in full. Keys the form has no field for
(diagnostic `_` knobs, or typos) are kept in each section's *other keys* field
rather than dropped.

## Running from the GUI

The same window has a **Launch** page that starts the PhoenX commands for you:

| Mode | Runs |
|------|------|
| Train | `phoenx-train --config FILE` (**Use Config Builder file** saves and picks the file open in Config Builder) |
| Resume | `phoenx-train --agent_dir DIR --stop_units N` |
| Test | `phoenx-test --agent_dir DIR ...` |
| Sweep | `phoenx-tune --config SWEEP ...` |

Each run is a separate process started with the same Python as the GUI,
exactly as the terminal would run it; the full command is shown (with a
**Copy** button) before you press **Start**. Relative paths in the config,
such as `save_dir: ./Trained_Models/...`, resolve against the **Working
folder**.

- The log pane shows the command's output. Progress drives the progress bar,
  and saved checkpoints appear as one updating green line.
- Choosing an agent folder (a run's `save_dir`) shows where its best
  checkpoint is, its best reward, and its training limit. For Resume, *Train
  until* must be past the checkpoint; it is prefilled with the original
  limit, or the checkpoint plus that limit once the run has already reached it.
- **Stop** interrupts the run the way Ctrl+C does: training saves a best
  checkpoint that is still waiting out `save_every`, then exits. If the
  process has not exited after 15 seconds (Isaac Sim's Kit can ignore the
  interrupt), Stop becomes **Kill**, and the whole process tree, including
  Ray workers, is killed.
- After a training run finishes, its folder is filled in for Resume and Test.

When a run's output is not a terminal (the Launch page, `nohup`, or
`phoenx-train ... > train.log`), the Trainer prints plain lines instead of
the live dashboard: `[progress] timestep 14,596/40,000 episodes=499
avg_reward=43.10 elapsed=0:00:05` every 5 seconds, `[checkpoint] ...` after
saves, and `[test] episode 10/10 avg_reward=...` when testing.

## Checking a config

PhoenX generates a JSON Schema for configs straight from its own source: the
constructor signatures give every key, type and default, and their docstrings
give the descriptions (see [phoenx.config_schema][]). It never imports torch,
so it runs instantly.

```bash
# Check files without building an env or a model
phoenx-config validate path/to/my_experiment.yml

# Autocomplete, hover docs and inline errors in VS Code
phoenx-config vscode
```

`validate` reports misspelled keys with a suggestion
(`agent.config: unknown key 'polcy_clip' (did you mean 'policy_clip'?)`),
wrong types, unknown layer / head / normalizer types, missing required layer
params, and n-step horizon mismatches. `vscode` writes
`phoenx-config.schema.json` into the project folder and points the
[YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
at it for `**/configs/**/*.yml` and `**/fixtures/*.yml` (change with
`--glob`). To attach the schema to a single file instead, put this on its
first line:

```yaml
# yaml-language-server: $schema=./phoenx-config.schema.json
```

Regenerate the schema (`phoenx-config schema`) after upgrading PhoenX.
Building a trainer applies the same unknown-key check at runtime, so a typo
fails at startup instead of training on a default.

## Isaac Lab environment classes

Isaac example configs can point at a custom env class via the
`env.config.cfg` field using a real importable module path. Two definition
modules ship under `phoenx.examples.isaac` (they import `isaaclab` at module
scope, so reference them by dotted path rather than importing the package
root):

- `phoenx.examples.isaac.custom_franka_cube_lift_cfg` — cube-lift variants,
  including camera and “blind” (no object pose in proprio) configs such as
  `FrankaCubeLiftCameraBlindEnvCfg`
- `phoenx.examples.isaac.custom_franka_reach_cfg` — reach variants such as
  `FrankaReachEnvCfg_Custom`

[create_env][phoenx.builder.create_env] forwards `env.config` as kwargs to the
wrapper named by `env.type`. For `type: isaacsim`, that is
[IsaacSimWrapper][phoenx.env_wrapper.IsaacSimWrapper]. Its `cfg` string is
split on `:` into `module_path` and `class_name`, then
`importlib.import_module(module_path)` plus `getattr(..., class_name)` builds
the Isaac Lab config class. Gymnasium and EnvPool wrappers treat `cfg` as a
plain environment / task id (for example `LunarLanderContinuous-v3`), not a
`module:Class` path.

Example (from the bundled camera config):

```yaml
env:
  type: isaacsim
  config:
    cfg: "phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg"
    enable_cameras: true   # required when the cfg exposes camera sensors
```

## Anatomy of a config

[build_trainer_from_config][phoenx.builder.build_trainer_from_config] reads a
fixed set of top-level sections (`env`, `agent`, `buffer`, `schedule`, plus
optional `callbacks`, `renderer`, `success_criterion`, `log_level`,
`save_dir`). Every algorithm builder under `phoenx.builders` uses the same
network schema: `agent.config.model` (roots → trunk → branches). Normalizers
and related schedules live under `agent.config` (not at the YAML root).

[apply_model_config][phoenx.builder.apply_model_config] pops `model` and
materializes roots, trunk, and branch heads. If `agent.config.model` is
absent or empty, it raises `ValueError` naming the algorithm and pointing
here. Top-level `models:` / `normalizers:` and flat `policy:` / `critic:`
dicts under `agent.config` are no longer read.

Top-level keys that only exist as YAML anchors (commonly `device`,
`learning_rate`) are not read by the trainer factory; they are merged into
nested fields via `*anchor` references.

### Non-multi-modal: LunarLander Continuous SAC

Bundled path: `LunarLanderContinuous-v3/sac.yml`. Flat `Box` observation, so
`roots` and `trunk` are `null` and each head keeps its own body and
optimizer. Annotated structure (same keys as the file; some layer lists
shortened with `# ...`):

```yaml
# Anchor-only: not read by build_trainer_from_config; referenced via *device
device: &device cuda

# Trainer checkpoint / renderer root (default if omitted: "models/")
save_dir: &save_dir ./Trained_Models/LunarLanderContinuous-v3/SAC/

# Trainer log level (default if omitted: "INFO"; CLI --log_level overrides)
log_level: INFO

# Required. Unpacked into phoenx.trainer.TrainingSchedule
schedule:
  stop_unit: timestep          # "timestep" | "episode"
  stop_units: &stop_units 1000000
  learn_every_unit: timestep
  learn_every: 64
  updates_per_learn: 1
  batch_size: 2048
  warmup_steps: 10000
  seed: &seed 42               # also seeds set_seed() before build

# Required. type selects phoenx.builders.sac; config kwargs feed SAC(...)
agent:
  type: SAC
  config:
    model:                         # required for every algorithm
      device: *device
      roots: null                  # flat Box → no per-modality encoder
      trunk: null
      branches:
        policy:
          type: StochasticContinuousHead   # HEAD_REGISTRY name; required
          # layer_config: dense(400) → relu → dense(300) → relu
          # output_config: dense out
          optimizer_params: {type: Adam, params: {lr: &policy_lr 0.00073}}
          distribution: normal
          lr_scheduler:            # per-branch; decays 7.3e-4 → 0.0
            schedule_type: linear
            steps: *stop_units
            start_value: *policy_lr
            end_value: 0.0
        critic:
          type: ContinuousQHead
          # layer_config / merged_config / output_config in the file
          optimizer_params: {type: Adam, params: {lr: &critic_lr 0.00073}}
          lr_scheduler: {schedule_type: linear, steps: *stop_units,
                         start_value: *critic_lr, end_value: 0.0}
        critic_b:
          type: ContinuousQHead    # twin Q; same shape as critic
          optimizer_params: {type: Adam, params: {lr: &critic_b_lr 0.00073}}
          lr_scheduler: {schedule_type: linear, steps: *stop_units,
                         start_value: *critic_b_lr, end_value: 0.0}
    discount: 0.99
    tau: 0.01
    entropy_coefficient: 1.0
    auto_entropy_tuning: true
    entropy_lr: 0.0003
    policy_grad_clip: 1.0
    critic_grad_clip: 1.0
    N: 1                       # n-step horizon: set it only here (see below)
    log_level: DEBUG
    _diag_freq: 1000
    state_normalizer:          # under agent.config, not top-level normalizers:
      type: RunningNorm
      config: {clip_value: 10.0, device: *device, log_level: DEBUG, _diag_freq: 10000}

# Required. type: envpool | gymnasium | isaacsim
env:
  type: envpool
  config:
    cfg: LunarLanderContinuous-v3   # EnvPool task id (not module:Class)
    num_envs: 64
    obs_key: null
    goal_key: null
    ach_goal_key: null
    wrappers: []                    # VectorNStepReward is added for ReplayBuffer / PER
    render_mode: null
    seed: *seed

# Required. type → Buffer.create_instance; config kwargs (+ injected env)
buffer:
  type: PrioritizedReplayBuffer
  config:
    buffer_size: 100000
    alpha: 0.6
    beta_start: 0.4
    beta_iter: 100000
    beta_update_freq: 10
    priority: proportional
    epsilon: 1.0e-6
    sort_freq: 1000
    device: *device                 # N comes from agent.config.N

# Optional. Forwarded to Renderer; Trainer overwrites renderer.save_dir
renderer:
  render_freq: 500
  save_dir: *save_dir

# Optional. Each entry: {type, config} → phoenx.rl_callbacks.load
callbacks:
  - type: "WandbCallback"
    config: {project_name: "LunarLanderContinuous-v3"}
```

### Multi-modal: Isaac Franka cube-lift PPO (camera)

Bundled path: `IsaacSim/franka/cube_lift/dense/ppo_camera.yml`. This is the
canonical roots → trunk → branches layout consumed by
[apply_model_config][phoenx.builder.apply_model_config]. Observation groups
with `obs_key: null` are a dict (`policy` proprio + `rgb` camera).

```yaml
device: &device cuda                    # YAML anchor only
save_dir: &save_dir ./Trained_Models/IsaacSim/Franka/CubeLift/PPO_CAM_1_BLIND/
log_level: INFO

learning_rate: &learning_rate 1.0e-4   # YAML anchor only

schedule:
  stop_unit: timestep
  stop_units: &stop_units 200000000    # committed bundled value
  learn_every_unit: timestep
  learn_every: 24576                   # 24 rollout steps * 1024 envs
  mini_batch_size: 1536
  learning_epochs: 5
  seed: &seed 42

agent:
  type: PPO
  config:
    name: PPO
    model:                             # modular schema → ModularModel
      device: *device
      optimizer_params: {type: Adam, params: {lr: *learning_rate}}
      shared_update: combined          # on-policy: one combined backward
      roots:
        camera:
          input_keys: [rgb]            # must match env observation group
          # layer_config: conv×3 → flatten → dense(256) → relu
        state:
          input_keys: [policy]
          # layer_config: dense(256) → relu → dense(128) → relu
      trunk:
        # layer_config: dense(256) → relu → dense(128) → relu
      branches:
        policy:
          type: StochasticContinuousHead
          distribution: normal
          # layer_config: dense(64) → relu; output dense with gain 0.01
        value:
          type: ValueHead
          # layer_config: dense(64) → relu; output dense with gain 1.0
    discount: 0.98
    gae_coefficient: 0.95
    state_normalizer:                  # under agent.config for PPO
      type: DictNormalizer
      config:
        per_key:
          # Blind cfg drops object_position → 33 features (not 36)
          policy: {type: RunningNorm, config: {num_features: 33, clip_value: 5.0, min_std: 1.0e-2}}
          # rgb omitted → passes through; uint8 frames scaled at model input
        device: *device
    advantage_normalizer:
      type: BatchNorm
      config: {clip_value: 10.0, device: *device}
    entropy_coefficient: 0.006
    auto_entropy_tuning: false
    kl_coefficient: 0.0
    kl_adapter:                        # → phoenx.adaptive_kl.AdaptiveKL
      initial_beta: 0.0
      target_kl: 0.01
      scale_up: 2.0
      scale_down: 0.5
      kl_tolerance_high: 1.5
      kl_tolerance_low: 0.5
    policy_clip: 0.2
    policy_grad_clip: 1.0
    value_clip: 0.2
    value_grad_clip: 1.0
    value_coef: 1.0
    reward_clip: .inf
    bootstrap_truncations: true

env:
  type: isaacsim
  config:
    cfg: "phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg"
    num_envs: 1024
    obs_key: null                      # expose all groups as a dict
    goal_key: null
    ach_goal_key: null
    wrappers: []
    render_mode: headless
    seed: *seed
    enable_cameras: true               # required for tiled RGB sensors

buffer:
  type: RolloutBuffer
  config:
    buffer_size: 24                    # rollout horizon (not total capacity)
    device: *device

callbacks:
  - type: "WandbCallback"
    config: {project_name: IsaacSimFrankaCubeLiftCameraBlind}
```

## Key reference

Source of truth:
[build_trainer_from_config][phoenx.builder.build_trainer_from_config] and the
factories it calls. Defaults below are the values those call sites use when a
key is missing (constructor defaults for passthrough sections).

### Top-level keys read by the trainer factory

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `env` | mapping | required | [create_env][phoenx.builder.create_env]: `type` + `config` |
| `agent` | mapping | required | [build_agent][phoenx.builder.build_agent]: `type` + per-algo builder |
| `buffer` | mapping | required | [build_buffer][phoenx.builder.build_buffer]: `type` + `config` |
| `schedule` | mapping | required | [build_schedule][phoenx.builder.build_schedule] → [TrainingSchedule][phoenx.trainer.TrainingSchedule] |
| `callbacks` | list | omitted → `None` | [build_callbacks][phoenx.builder.build_callbacks] → [load][phoenx.rl_callbacks.load] |
| `renderer` | mapping | omitted → `None` | [build_renderer][phoenx.builder.build_renderer] → `Renderer(**kwargs)` |
| `success_criterion` | mapping | omitted → `None` | [build_success_criterion][phoenx.builder.build_success_criterion] → [SuccessCriterion][phoenx.trainer.SuccessCriterion] |
| `log_level` | str | `"INFO"` | Trainer logger; CLI `--log_level` overrides when passed |
| `save_dir` | str | `"models/"` | Trainer checkpoint / renderer root |
| `metrics` | mapping | omitted → default manager | [build_metrics][phoenx.builder.build_metrics] → [MetricsManager][phoenx.metrics.MetricsManager] |

Keys that appear in some YAML but are **not** read by
`build_trainer_from_config` include top-level `device` and `learning_rate`
(YAML anchors only). Top-level `models`, `normalizers`, `*_lr_schedule`,
`entropy_schedule`, and `temperature_schedule` are also ignored — put the
equivalents under `agent.config` / `agent.config.model` (see Agent section
and the migration table below).

### `schedule` → [TrainingSchedule][phoenx.trainer.TrainingSchedule]

Passthrough kwargs. Dataclass defaults:

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `stop_unit` | `"timestep"` \| `"episode"` | `"timestep"` | Unit for run length |
| `stop_units` | int | `1_000_000` | Stop when progress reaches this many units |
| `learn_every_unit` | `"timestep"` \| `"episode"` | `"timestep"` | Unit for learn cadence |
| `learn_every` | int | `1` | Progress between `agent.learn` calls |
| `updates_per_learn` | int | `1` | Gradient updates per learn call |
| `batch_size` | int | `1` | Sample size when `updates_per_learn > 1` |
| `mini_batch_size` | int | `1` | Forwarded to `agent.learn` |
| `learning_epochs` | int | `1` | Forwarded to `agent.learn` |
| `warmup_steps` | int | `0` | Env steps before the first learn |
| `seed` | int \| null | `None` | If set, seeds the run via `phoenx.torch_utils.set_seed` before build |
| `save_every` | int | `50_000` | Min env timesteps between automatic best-model checkpoints (`0` = every new best) |

`save_every` exists because a new rolling-average best used to trigger a full
checkpoint tree (and, with W&B, a full artifact upload) nearly every step
early in training — `avg_reward` is a mean over the last 100 episodes and
climbs almost monotonically, and many parallel envs finish episodes on
nearly every step. The cooldown is counted in **timesteps** regardless of
`stop_unit` / `learn_every_unit`. The first new best of a run always saves;
later peaks inside the window are pending until it expires (disk weights may
lag the exact peak by design). A pending peak is written at the first episode
boundary after the window expires, so real gaps run somewhat longer than
`save_every`, and one still pending when training stops is flushed at the end
of the run — so a short run that never reaches `save_every` again still ends
with its best on disk. Set `save_every: 0` to restore save-on-every-new-best.

### `env`

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `type` | str | required | `"isaacsim"` \| `"gymnasium"` \| `"envpool"` |
| `config` | mapping | required | Kwargs for the matching wrapper constructor |

Common `config` fields (wrapper defaults):

| Key | Gymnasium / EnvPool default | IsaacSim default | Notes |
|-----|-----------------------------|------------------|-------|
| `cfg` | required str | required `module:Class` str | Env id vs Isaac config class |
| `num_envs` | `1` | `1` | Parallel envs; Isaac also writes `cfg.scene.num_envs` |
| `obs_key` | `None` | `"policy"` | Dict key for agent state; `null` keeps multi-modal dict |
| `goal_key` | `None` | `None` | Desired-goal key for HER / goal spaces |
| `ach_goal_key` | `None` | `None` | Achieved-goal key |
| `wrappers` | `None` | `None` | List of `{type, params}` from `WRAPPER_REGISTRY` |
| `render_mode` | `None` | `"headless"` | Isaac: non-`headless` launches headed Kit |
| `seed` | `None` (random 31-bit) | same | |
| `num_threads` | EnvPool only (`None` → `num_envs`) | — | |
| `enable_cameras` | — | `False` | Required for camera / tiled sensors |
| `distance_threshold` | — | `None` | Sparse goal radius for HER on Isaac |

Registered wrapper `type` names include `VectorNStepReward`,
`OneHotObservationWrapper`, `VectorOneHotObservation`, `AtariPreprocessing`,
`TimeLimit`, `TimeAwareObservation`, `FrameStackObservation`, and
`ResizeObservation`. Off-policy replay buffers require `VectorNStepReward` on
the env; PhoenX appends it for you (as the outermost wrapper) when it is not
listed. Full wrapper API: [Environment wrappers][phoenx.env_wrapper].

#### The n-step horizon `N`

Set the n-step horizon once, as `agent.config.N`. Before anything is built,
[resolve_n_step][phoenx.config_utils.resolve_n_step] copies it to
`buffer.config.N`, to the `VectorNStepReward` wrapper's `n` (adding the
wrapper when a replay buffer is used and none is listed), and to an `n_step`
HER relabeler's `N`. The wrapper's `obs_key` / `goal_key` / `ach_goal_key`
default to the values in `env.config`. Older configs that repeat `N` in those
places still load, as long as every copy agrees; a mismatch raises
`ValueError`.

### `agent`

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `type` | str | required | `"ActorCritic"` \| `"Reinforce"` \| `"PPO"` \| `"DDPG"` \| `"TD3"` \| `"SAC"` |
| `config` | mapping | required | Algorithm kwargs after the builder materializes models / normalizers |

**Network schema (all six algorithms):** `agent.config.model` with
`roots`, optional `trunk`, `branches`, plus optional
`optimizer_params`, `lr_scheduler`, and `shared_update`. Processed by
[apply_model_config][phoenx.builder.apply_model_config]. Every branch needs
an explicit `type:` naming a [HEAD_REGISTRY][phoenx.models.HEAD_REGISTRY]
entry (`StochasticContinuousHead`, `StochasticDiscreteHead`,
`ContinuousQHead`, `ValueHead`, …). A `device` key under `model:` is not
read; the composite model is placed on the agent's own `device`.

[Head.from_config][phoenx.models.Head.from_config] silently drops keys the
head constructor does not accept. A misplaced field (for example an LR
schedule parked next to `discount` instead of on the branch) fails quietly
rather than loudly — check the live branch keys against the head class.

**Branches-only vs shared roots.** A flat `Box` observation needs no
per-modality encoder: set `roots: null` and `trunk: null` so the observation
is preprocessed and routed straight to each head. The three LunarLander
bundled configs use that layout; each head keeps its own body and optimizer,
and `shared_update` is inert when nothing is shared (omit it and the agent
default applies). Multi-modal dict observations need named roots with
`input_keys` (and usually a trunk): see
`IsaacSim/franka/cube_lift/dense/ppo_camera.yml`, which encodes `rgb` and
`policy` separately before a shared trunk.

**Missing `model`:** if `agent.config.model` is absent or empty,
[apply_model_config][phoenx.builder.apply_model_config] raises `ValueError`
(`"{algo}: 'agent.config.model' is required … See docs/how-to/configurations.md."`).
Add a `model:` block as in the examples above; do not fall back to top-level
`models:` or flat `policy:` / `critic:` keys.

**Migrating an older YAML**

| Old location | New location |
|--------------|--------------|
| top-level `models.<role>` | `agent.config.model.branches.<role>` (add explicit `type:`) |
| flat `agent.config.<role>` head dict | `agent.config.model.branches.<role>` (add explicit `type:`) |
| top-level `normalizers.state` / `.goal` / `.advantage` / `.reward` | `agent.config.state_normalizer` / `goal_normalizer` / `advantage_normalizer` / `reward_normalizer` |
| top-level `entropy_schedule` | `agent.config.entropy_schedule` |
| top-level `<role>_lr_schedule` | `agent.config.model.branches.<role>.lr_scheduler` |
| top-level `temperature_schedule` | `agent.config.model.branches.policy.temperature_schedule` |

Legacy `Model` / policy / critic classes, `head_from_legacy_model_config`,
`LEGACY_MODEL_TO_HEAD_TYPE`, `map_legacy_state_dict`, and the legacy branch
of `Agent.from_config` / `_load_legacy_checkpoint` remain for **checkpoint
and saved-config reload only**. They do not make legacy YAML loadable.

**Normalizers / extras under `agent.config`:**
`state_normalizer`, `goal_normalizer`, `reward_normalizer`,
`advantage_normalizer` (on-policy), `intrinsic_motivation`,
`entropy_schedule`, `policy_clip_schedule` / `value_clip_schedule` (PPO),
`kl_adapter` (PPO → [AdaptiveKL][phoenx.adaptive_kl.AdaptiveKL]), `noise` /
`noise_schedule` (DDPG/TD3). Each normalizer is a `{type, config}` mapping;
see [Normalizers][phoenx.normalizer].

**Intrinsic motivation.** `intrinsic_motivation` is a `{type, config}`
mapping. `type` is `ICM`, `RND`, `EpisodicNovelty`, or `Composite`, which
combines several of them. Each network under `model_configs` is a
`layer_config` list of hidden layers plus a one-item `output_layer` list
holding a dense layer. The network names depend on the type:

| `type` | `model_configs` networks |
|---|---|
| `ICM` | `encoder` (optional), `inverse_model`, `forward_model` |
| `RND` | `target`, `predictor` |
| `EpisodicNovelty` | `encoder`, `inverse_model` |

A `Composite` lists its modules under `components`, each written the same way
as a single module. `combination_rule` merges their rewards: `additive`
(optional `combination_kwargs.weights`, one per component), `multiplicative`,
`max`, or `ngu`. `ngu` takes exactly two components, episodic first, and
gives `episodic * clip(lifelong, 1, L)` with `combination_kwargs.L`
(default 5.0). Reward normalizers go on the components; the composite has
no normalizer of its own.

```yaml
intrinsic_motivation:
  type: Composite
  config:
    combination_rule: ngu
    combination_kwargs: {L: 5.0}
    components:
      - type: EpisodicNovelty
        config:
          model_configs:
            encoder:
              layer_config: [{type: dense, params: {units: 64}}, {type: relu}]
              output_layer: [{type: dense, params: {units: 32}}]
            inverse_model:
              layer_config: [{type: dense, params: {units: 64}}, {type: relu}]
              output_layer: [{type: dense}]   # width defaults to the action size
          optimizer_params: {type: Adam, params: {lr: 1.0e-4}}
      - type: RND
        config:
          model_configs:
            target:
              layer_config: [{type: dense, params: {units: 128}}, {type: relu}]
              output_layer: [{type: dense, params: {units: 64}}]
            predictor:
              layer_config: [{type: dense, params: {units: 128}}, {type: relu}]
              output_layer: [{type: dense, params: {units: 64}}]
          optimizer_params: {type: Adam, params: {lr: 1.0e-4}}
          reward_normalizer: {type: RewardNorm, config: {}}
```

Remaining algorithm hyperparameters (`discount`, clip coefficients, `tau`,
entropy settings, …) are constructor kwargs for the agent class. See
[Agents][phoenx.rl_agents] rather than duplicating every field here.

### `buffer`

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `type` | str | required | `"ReplayBuffer"` \| `"PrioritizedReplayBuffer"` \| `"RolloutBuffer"` \| `"TrajectoryBuffer"` |
| `config` | mapping | `{}` | Kwargs to `Buffer.create_instance` (`env` injected) |

Shared / common `config` fields (constructor defaults):

| Key | Default | Notes |
|-----|---------|-------|
| `buffer_size` | `100000` (replay / PER); required for rollout / trajectory | Replay: max windows; Rollout: horizon per env |
| `N` | `1` | Replay / PER window length; filled in from `agent.config.N` |
| `device` | `None` → `get_device()` | |
| `hindsight` | `None` | Mapping → [HindsightRelabeler][phoenx.her.HindsightRelabeler] (`env` injected) |

PER-only defaults: `alpha=0.6`, `beta_start=0.4`, `beta_iter=100000`,
`beta_update_freq=1`, `priority="proportional"`, `sort_freq=1000`,
`epsilon=1e-6`. Full buffer API: [Buffer][phoenx.buffer].

### `callbacks`

List of `{type, config}` mappings. `type` must be a registered callback class
name (bundled examples use `"WandbCallback"`). `config` is constructor kwargs.
For `WandbCallback`: required `project_name`; optional `run_name`, `group`,
`tags`, `run_id`, `resume`, `sweep_params`, `artifact_every`, and
`extra_config`. See [RL callbacks][phoenx.rl_callbacks].

`artifact_every` (int or omit / `null`; default `null`) is an optional extra
minimum number of trainer timesteps between W&B model-artifact uploads,
applied **on top of** `schedule.save_every`. Omit it, or set `0` / `null`, to
upload on every checkpoint written during the run — the end-of-run flush of a
still-pending best writes to disk without uploading. Uploads gate on the
episode-log `saved` flag (checkpoint written this step), not `best` (new
rolling-average peak) — so a custom callback that still keys off `best` can
publish a stale on-disk model when the save cooldown skipped the write.

```yaml
callbacks:
  - type: "WandbCallback"
    config:
      project_name: "LunarLanderContinuous-v3"
      artifact_every: 200000   # rarer than local saves; omit to match save_every
```

`extra_config` (mapping or omit / `null`; default `null`) is an
already-flat mapping merged into `wandb.init(config=...)` **last** — after
the trainer config and any flattened `sweep_params` — so it wins any key
collision. It is copied at construction time, so mutating the dict you
passed in afterwards does not change what `get_config()`/`save()` later
write. It exists for run provenance that is not itself a swept
hyperparameter — `phoenx-tune` sets it automatically on each sweep trial's
callback to record the underlying Ray trial id (see
[Hyperparameter Sweeps](hyperparameter-sweeps.md)) — so a hand-written
training YAML has no reason to set it.

### `renderer`

Optional mapping forwarded as `Renderer(**kwargs)`. Fields used by the
dataclass: `render_freq` (default `0`), `save_dir` (default `"models/"`;
overwritten by the trainer’s `save_dir`), `fps` (default `30`), `codec`
(default `"libx264"`).

### `success_criterion`

Optional passthrough to [SuccessCriterion][phoenx.trainer.SuccessCriterion]:
`metric` (`"info_flag"` \| `"goal_distance"` \| `"episode_reward"`),
`threshold` (required for the distance / reward metrics), `info_key`
(default `"is_success"`).

### `metrics`

Optional mapping, passed through
[build_metrics][phoenx.builder.build_metrics] to
[MetricsManager][phoenx.metrics.MetricsManager]. Its keys are exactly the
constructor kwargs — there is no nesting:

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `tiers` | list[int] | `[0, 1, 2]` | Cost tiers enabled by default (every registered metric is currently tier 0) |
| `emit_every` | int | `1` | `collect()` calls between emitting calls; raise to cut logging traffic |
| `include` | list[str] \| null | `null` | Exact-key allow-list applied **after** capability gating; when non-empty it *replaces* the tier gate rather than widening it. `null` **or `[]`** means "no allow-list" |
| `exclude` | list[str] | `[]` | Exact keys to suppress, along with their legacy aliases |
| `dual_emit` | bool | `true` | Also emit each aliased metric under its legacy flat name |

Omitting the block entirely builds a default `MetricsManager()` — every
registered metric whose agent/env/buffer capability is present,
`emit_every: 1` (today's per-step cadence), and both new and legacy key
names emitted side by side.

```yaml
metrics:
  emit_every: 10        # cut W&B traffic to one collected window in 10
  exclude: [policy/kl]  # drop this key AND its legacy alias `kl`
```

**What this replaces.** Metric collection used to be scattered across the
trainer's rollout/PER bookkeeping and each agent's `learn()` return dict,
all funneling directly into `callback.on_train_step_end`. A `MetricsManager`
now owns namespacing, windowed reduction, cadence, and capability gating,
and returns a flat `dict[str, float]` that the trainer merges into
`step_log` before callbacks run — `WandbCallback` and `RayTuneCallback` were
not changed.

- **Capability gating.** A metric is only enabled when the bound
  agent/env/buffer actually has the relevant capability — `sac/alpha` never
  appears on a PPO run rather than logging as null, and `include` cannot
  force it to.
- **Dual-emit.** Every aliased metric (e.g. `loss/policy`) is also emitted
  under its pre-existing flat name (`policy_loss`), so existing W&B history
  stays continuous. Set `dual_emit: false` to emit only the namespaced keys.
- **Absent, not zero.** A metric with no recorded value this window is
  omitted from the merged log entirely — never `0.0`, never `null`.
- **Behavior changes worth knowing about:** with `schedule.updates_per_learn
  > 1`, mean-reduced losses now average over every update in that `learn()`
  call instead of reporting only the last one; `RayTuneCallback` gains a
  few reported keys, because values that used to arrive as live one-element
  tensors (silently skipped by its accumulator) now arrive as plain floats;
  and per-step W&B key cardinality rises from roughly 15 to roughly 45.
- **Resume caveat.** Window state (rolling windows and rate anchors) is not
  checkpointed. After `Trainer.load`, `rollout/return_rolling_100` starts
  from an empty window even though the trainer's own 100-episode score
  history is restored from `trainer_state.pt`, so the two disagree for the
  first window after a resume.
- **Buffer occupancy and optimizer activity.** `buffer/fill`,
  `buffer/env_steps`, and `buffer/is_ready` are enabled for every buffer
  type (`buffer/pending_trajectories` for `TrajectoryBuffer` only), gated on
  the buffer's own capability rather than on the agent being off-policy, so
  PPO, REINFORCE, and `ActorCritic` runs report buffer state alongside
  DDPG/TD3/SAC. `buffer/fill` shares one definition but not one dynamic:
  it saturates at `1.0` for the replay buffers, resets to `0.0` on every
  `sample()` for `RolloutBuffer`, and for `TrajectoryBuffer` tracks the
  longest still-in-progress episode — it stays put across a `sample()` and
  only drops when that episode's env finishes. Do not treat a low
  `buffer/fill` as "still warming up" for every buffer type; that reading
  only holds for the replay buffers. `optim/updates_per_env_step` counts
  `ModularModel.step()` calls per environment step for the current run
  segment — a useful, well-defined ratio, but **not** the literature's
  replay ratio, since off-policy agents call `step()` twice per `learn()`
  (critic modules, then policy) while on-policy agents call it once per
  minibatch; the temperature (`entropy_optimizer`) and intrinsic-motivation
  optimizers are not counted. A *run segment* is the stretch since the
  current `train()` call began — on a fresh run that is the whole run, and
  after `Trainer.load` it is everything since the resume. Both the numerator
  and the denominator are scoped that way on purpose: the trainer's step
  counter is restored from `trainer_state.pt` while the optimizer counter is
  not, so dividing one by the other would report a ratio collapsing toward
  zero the moment you resume a long run. Expect the series to restart from
  its own baseline after a resume rather than continuing the prior curve.

**Scope.** Covers rollout/perf metrics, buffer occupancy and readiness, a
single optimizer-activity counter (`optim/updates_per_env_step`),
prioritized-replay diagnostics, and metrics ingested from each agent's
existing `learn()` return dict. Per-module gradient norms, pre-clip norms,
param norms, and update/weight ratios remain unimplemented, as do metrics
that would require new computation inside `learn()`, recurrent-specific
metrics, and a held-out evaluation harness for off-policy explained
variance.

## Hyperparameter sweeps

To search over a training config without editing it on disk, write a
standalone sweep YAML and run `phoenx-tune`. Schema, multi-phase
promotion, architecture search, and the bundled examples:
[Hyperparameter Sweeps](hyperparameter-sweeps.md).

## Creating a config for a new environment

1. **Copy the nearest bundled file.** Flat Gymnasium / EnvPool continuous →
   `LunarLanderContinuous-v3/sac.yml` or `ppo.yml` (branches-only
   `agent.config.model`). Discrete on-policy → `LunarLander-v3/reinforce.yml`.
   Multi-modal Isaac → `IsaacSim/franka/cube_lift/dense/ppo_camera.yml`
   (named roots + trunk).
2. **Change env identity.** Set `env.type` and `env.config.cfg` (Gymnasium /
   EnvPool id, or Isaac `module.path:ClassName`). Adjust `num_envs`,
   `obs_key` / `goal_key` / `ach_goal_key`, `wrappers`, and Isaac-only
   `enable_cameras` / `distance_threshold` as needed.
3. **Align the model to observation and action spaces.** Root `input_keys`
   must match observation-group names when `obs_key` is `null`. Head
   `distribution` must match the action space (`categorical` vs continuous).
   For `RunningNorm`, either omit `num_features` (inferred via
   [infer_dim][phoenx.builder.infer_dim] for non-dict normalizers) or set it
   to the true feature width — a blind Isaac cfg that drops object pose needs
   `33`, not `36`.
4. **Match buffer and schedule to the algorithm.** Off-policy replay uses
   `agent.config.N` for the n-step window (the buffer and wrapper follow it). On-policy PPO should set
   `learn_every` ≈ `buffer.config.buffer_size * num_envs` when counting
   timesteps. Scale `mini_batch_size` / `learning_epochs` with rollout size.
5. **Update run metadata.** `save_dir`, callback `project_name`, and optional
   `success_criterion` for logging.

### Common pitfalls

- **Obs / action mismatch** — Dense `num_features`, root `input_keys`, or
  head type that does not match the live spaces fails at build or first step.
- **`obs_key` selection** — Isaac defaults to `"policy"` (single tensor). Set
  `obs_key: null` for multi-modal dicts; keep `"policy"` (or another key) for
  state-only.
- **`enable_cameras: false` with camera sensors** — Kit will not produce RGB
  groups; set `enable_cameras: true` whenever the cfg defines cameras.
- **`learn_every` vs `num_envs`** — Timestep cadence is global steps. Raising
  `num_envs` without scaling `learn_every` learns more often per env-step of
  experience.
- **Misspelled keys** — An unknown key under `agent.config`, a normalizer
  `config`, a head, a `roots` / `trunk` subnetwork, `agent.config.model`, or a
  layer's `params` raises `TypeError` with a "did you mean" suggestion. Keys
  that start with `_` (such as `_diag_freq`) are diagnostic knobs and always
  allowed. Run `phoenx-config validate my_config.yml` to check a file
  without building anything.
- **HER keys** — `goal_key` (and usually `ach_goal_key`) must be set;
  `buffer.config.hindsight` must use `output_format` compatible with the
  buffer (`n_step` for replay, `flat` for trajectory).
- **Schema placement** — Networks belong under `agent.config.model`;
  normalizers and `entropy_schedule` under `agent.config`. Top-level
  `models:` / `normalizers:` / `*_lr_schedule` are ignored for every
  algorithm. Missing `agent.config.model` raises `ValueError` from
  [apply_model_config][phoenx.builder.apply_model_config].
