# Isaac Sim Environments

PhoenX integrates with [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) for
GPU-accelerated robotics environments.

## Environment setup

Follow NVIDIA's installation guide to create the Isaac Lab environment
(conda, Python version matching your Isaac Sim release), then install PhoenX
into the same environment. See [Getting Started](getting-started.md).

## Verify your installation

After installing PhoenX into your Isaac Lab environment, confirm that Isaac
Sim, CUDA, and GPU environments work:

```bash
pytest tests/test_isaac_setup.py -v
```

These tests are part of the main suite (`pytest tests`) but are gated by the
`isaac` marker and auto-skip when `isaaclab` or CUDA is unavailable. To exclude
them on machines without Isaac:

```bash
pytest -m "not isaac"
```

You can also run the module standalone inside the Isaac container:
`python tests/test_isaac_setup.py`.

## Training on Isaac Lab tasks

The bundled multi-modal worked example is
`IsaacSim/franka/cube_lift/dense/ppo_camera.yml` (package path
`phoenx/examples/configs/IsaacSim/franka/cube_lift/dense/ppo_camera.yml`).
It is resolvable by the bundled-example shortcut, so from a bare install:

```bash
phoenx-train --config IsaacSim/franka/cube_lift/dense/ppo_camera.yml
```

That config trains PPO on a Franka cube-lift task with proprioception plus an
RGB camera (`obs_key: null`, `enable_cameras: true`). The Isaac `cfg` field is
a `module:Class` import path — unlike Gymnasium / EnvPool, which take an
environment id string. Resolution details live under
[Isaac Lab environment classes](configurations.md#isaac-lab-environment-classes).

[IsaacSimWrapper][phoenx.env_wrapper.IsaacSimWrapper] defaults matter for this
path: `render_mode='headless'`, `enable_cameras=False`, `obs_key='policy'`,
and `distance_threshold=None`. `num_envs` is written into
`cfg.scene.num_envs`. Kit launches via
`AppLauncher(headless=(render_mode == 'headless'), device="cuda:0", enable_cameras=...)`,
so `render_mode: headless` runs without a display and any other value opens a
headed Kit app; the device is hard-coded to `cuda:0`. Set
`enable_cameras: true` whenever the cfg exposes camera / tiled sensors.
`distance_threshold` is the sparse-goal radius used by HER; `None` means
“not configured” (HER consumers raise an actionable error rather than a
`TypeError`).

On an RTX 4090 / Windows 11 machine with warm caches, that config took about
40 seconds from the first Kit log line to
`[INFO]: Completed setting up the environment...`, and about 50 seconds to
the first training step. Cold first runs were not measured here — Isaac Sim’s
first launch on a machine pulls and compiles extensions and shader caches and
is substantially slower. A 120,000-step run at `num_envs: 128` finished in
roughly 1 minute 22 seconds of wall clock at about 4.7 episodes/sec.

VRAM scales with cameras. The shipped `num_envs: 1024` with cameras enabled
saturated a 24 GB card: CUDA spilled into host memory and throughput
collapsed until the run had to be killed. Reducing to `num_envs: 128` made it
workable. Camera observations multiply per-env VRAM, so scale `num_envs` down
when `enable_cameras` is true and watch for a throughput collapse rather than
an out-of-memory error.

Startup console noise, the Rich live dashboard, checkpoint layout, and W&B
behavior are covered under
[What training prints](getting-started.md#what-training-prints) — this page
adds only the Isaac-specific facts above.
`Renderer.render_episode` raises `ValueError` for Isaac Sim environments, so
Isaac runs produce no mp4 files regardless of a `renderer:` section. Watching
live via a non-headless `render_mode` is the only option.

## Custom environment configurations

PhoenX ships two custom Isaac Lab environment definitions in the
`phoenx.examples.isaac` package: `custom_franka_cube_lift_cfg` and
`custom_franka_reach_cfg`. Select one from a YAML config by module path, e.g.

```yaml
env:
  config:
    cfg: "phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg"
```

That same `module:Class` string is what
[IsaacSimWrapper][phoenx.env_wrapper.IsaacSimWrapper] imports at load time
(see [Isaac Lab environment classes](configurations.md#isaac-lab-environment-classes)).
The bundled camera example and a saved agent both use
`FrankaCubeLiftCameraBlindEnvCfg` — the “blind” variant, not the full-state
camera cfg.

### Cube lift (`custom_franka_cube_lift_cfg`)

The stock base is `FrankaCubeLiftEnvCfg` from
`isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg`;
the observation-group subclasses extend
`isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg.ObservationsCfg`
(imported as `LiftObservationsCfg`). The `_PLAY` classes below derive from their
own custom parents, not from Isaac Lab's stock `_PLAY` variants.

| Class | Role |
|-------|------|
| `FrankaCubeLiftEnvCfg_Custom` | Subclasses stock `FrankaCubeLiftEnvCfg`; exposes `action_scale` (default `0.5`) and rebuilds `actions.arm_action` as `JointPositionActionCfg` with that scale. |
| `FrankaCubeLiftEnvCfg_Custom_PLAY` | Play variant of Custom: `num_envs=50`, `env_spacing=2.5`, policy corruption off. |
| `FrankaCubeLiftEnvCfg_Custom_Limits` | Subclasses stock lift cfg; uses `JointPositionToLimitsActionCfg` so policy outputs in ~[-1, 1] span joint limits (`rescale_to_limits=True`). |
| `FrankaCubeLiftEnvCfg_Custom_Limits_PLAY` | Play variant of Limits (same overrides as Custom_PLAY). |
| `GoalObservationsCfg` | Extends `LiftObservationsCfg` with `achieved_goal` / `desired_goal` groups for HER. |
| `FrankaCubeLiftEnvCfg_Custom_Goal` | Limits action space plus `GoalObservationsCfg`. |
| `FrankaCubeLiftEnvCfg_Custom_Goal_PLAY` | Play variant of Goal. |
| `CameraObservationsCfg` | Extends `LiftObservationsCfg` with a separate `rgb` group (table-view TiledCamera, uint8). |
| `FrankaCubeLiftCameraEnvCfg` | Limits base + camera sensor + `CameraObservationsCfg`; keeps `object_position` in the policy group (36 features) so vision is optional. |
| `FrankaCubeLiftCameraBlindEnvCfg` | Same camera scene, but sets `observations.policy.object_position = None` (36 → 33 features) so the cube’s location is only in the image. |
| `FrankaCubeLiftCameraEnvCfg_PLAY` / `…Blind…_PLAY` | Play variants: `num_envs=16`, `env_spacing=2.5`, policy corruption off. |

### Reach (`custom_franka_reach_cfg`)

The stock base is `FrankaReachEnvCfg` from
`isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg`.

| Class | Role |
|-------|------|
| `FrankaReachEnvCfg_Custom` | Subclasses stock `FrankaReachEnvCfg`; `action_scale` default `2.0` on `JointPositionActionCfg`. The attribute's own docstring records the pre-override default as `0.5`. |
| `FrankaReachEnvCfg_Custom_PLAY` | Play variant: `num_envs=50`, `env_spacing=2.5`, policy corruption off. |
| `FrankaReachEnvCfg_Custom_Limits` | `JointPositionToLimitsActionCfg` with `scale=1.0` and `rescale_to_limits=True`. No `_PLAY` sibling ships for this class. |

Across both modules, `_PLAY` means fewer parallel envs, wider spacing, and
`observations.policy.enable_corruption = False` — not a different task
definition. Camera play variants use 16 envs; the non-camera play variants
use 50.

### Writing a new Isaac cfg for PhoenX

On the PhoenX side the contract is small: any importable `module:Class` works
as `env.config.cfg`, because the string is resolved by import at load time.
Put the module somewhere on `PYTHONPATH` (for example under
`phoenx.examples.isaac`, or your own package installed into the Isaac Lab
env). If a saved agent’s `config.json` still points at a stale path,
`phoenx-test --env` overrides that field — see
[Evaluate a trained agent](getting-started.md#evaluate-a-trained-agent).

For the Isaac Lab side — scene, managers, rewards, registration — follow
Isaac Lab’s own guides rather than this page:
[Create new project or task](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)
and the manager-based / direct workflow tutorials linked from
[Task Design Workflows](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html).
