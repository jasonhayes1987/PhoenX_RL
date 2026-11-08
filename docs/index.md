# PhoenX RL

An open-source reinforcement learning framework with Isaac Lab integration.
Train agents on standard Gymnasium environments in minutes, or scale the same
agents to GPU-accelerated robotics simulation in Isaac Sim.

## Quick start

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
phoenx-train --config LunarLanderContinuous-v3/sac.yml
```

The `--config` value is resolved from bundled package examples when no on-disk
file matches, so this works with no clone. Full install sequences (Gymnasium
and Isaac Lab): [Installation](how-to/installation.md).

## Where to go next

- **[Installation](how-to/installation.md)** — Gymnasium mode, Isaac Lab mode, `setup.ps1`, troubleshooting
- **[Getting Started](how-to/getting-started.md)** — `phoenx-train` / `phoenx-test` CLI usage
- **[Configuration Files](how-to/configurations.md)** — YAML schema, bundled examples, config resolution
- **[Hyperparameter Sweeps](how-to/hyperparameter-sweeps.md)** — `phoenx-tune` multi-phase Ray Tune sweeps, search specs, architecture search
- **[Isaac Sim Environments](how-to/isaac-sim.md)** — training in Isaac Lab, including custom environment configs
- **[API Reference](api/rl_agents.md)** — every public module, generated from the source docstrings
- **[Changelog](changelog.md)** — what changed and when
