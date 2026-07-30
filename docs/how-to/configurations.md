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
`phoenx.builder.available_example_configs()`.

## How `--config` is resolved

`phoenx.builder.load_config` resolves the value in this order:

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
`env.config.cfg` field using a real importable module path. Two definitions
ship in `phoenx.examples.isaac`:

- `phoenx.examples.isaac.custom_franka_cube_lift_cfg`
- `phoenx.examples.isaac.custom_franka_reach_cfg`

Example (from the bundled camera config):

```yaml
env:
  config:
    cfg: "phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg"
```

These modules import `isaaclab` at module scope and are never imported at
package-import time; they only load when a config references them.

## Anatomy of a config

<!-- TODO: paste 2 real configs. A non-multi-modal config
     (e.g. LunarLander sac.yml) and a multi-modal (e.g )
     and annotate every top-level key: agent/algorithm selection, environment
     spec, network architecture, training hyperparameters, buffer/HER options,
     logging/W&B, save paths. This section is the single most useful page in
     the docs — be thorough. -->

## Key reference

<!-- TODO: table of all recognized config keys per section, with
     type, default, and effect. Source of truth: phoenx.builder (load_config /
     build_trainer_from_config) — document what the code actually reads, not
     what seems plausible. -->

## Creating a config for a new environment

<!-- TODO: minimal steps — copy nearest existing config, keys that
     must change per environment, common pitfalls (obs/action space mismatches). -->
