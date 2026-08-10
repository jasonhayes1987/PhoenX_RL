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
    N: &n 1                    # must match buffer N and VectorNStepReward n
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
    obs_key: &obs_key null
    goal_key: &goal_key null
    ach_goal_key: &ach_goal_key null
    wrappers:
      - type: VectorNStepReward     # required for ReplayBuffer / PER
        params: {n: *n, obs_key: *obs_key, goal_key: *goal_key, ach_goal_key: *ach_goal_key}
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
    N: *n
    device: *device

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
the env. Full wrapper API: [Environment wrappers][phoenx.env_wrapper].

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
| `N` | `1` | Replay / PER window length; must match `VectorNStepReward` |
| `device` | `None` → `get_device()` | |
| `hindsight` | `None` | Mapping → [HindsightRelabeler][phoenx.her.HindsightRelabeler] (`env` injected) |

PER-only defaults: `alpha=0.6`, `beta_start=0.4`, `beta_iter=100000`,
`beta_update_freq=1`, `priority="proportional"`, `sort_freq=1000`,
`epsilon=1e-6`. Full buffer API: [Buffer][phoenx.buffer].

### `callbacks`

List of `{type, config}` mappings. `type` must be a registered callback class
name (bundled examples use `"WandbCallback"`). `config` is constructor kwargs
(for W&B: `project_name`, optional `run_name`). See
[RL callbacks][phoenx.rl_callbacks].

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
4. **Match buffer and schedule to the algorithm.** Off-policy replay needs
   `VectorNStepReward` and matching `N`. On-policy PPO should set
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
- **PER / replay without `VectorNStepReward`** — `ReplayBuffer` and
  `PrioritizedReplayBuffer` raise if the wrapper is missing.
- **HER keys** — `goal_key` (and usually `ach_goal_key`) must be set;
  `buffer.config.hindsight` must use `output_format` compatible with the
  buffer (`n_step` for replay, `flat` for trajectory).
- **Schema placement** — Networks belong under `agent.config.model`;
  normalizers and `entropy_schedule` under `agent.config`. Top-level
  `models:` / `normalizers:` / `*_lr_schedule` are ignored for every
  algorithm. Missing `agent.config.model` raises `ValueError` from
  [apply_model_config][phoenx.builder.apply_model_config].
