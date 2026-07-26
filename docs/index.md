# PhoenX RL

An open-source reinforcement learning framework with Isaac Lab integration.
Train agents on standard Gymnasium environments in minutes, or scale the same
agents to GPU-accelerated robotics simulation in Isaac Sim.

## Quick start

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
phoenx-train --config configs/LunarLander-v3/sac.yml
```

## Where to go next

- **[Getting Started](how-to/getting-started.md)** — install (Gymnasium or Isaac Lab mode) and run your first training job
- **[Configuration Files](how-to/configurations.md)** — the YAML schema that defines agents, environments, and training runs
- **[Isaac Sim Environments](how-to/isaac-sim.md)** — training in Isaac Lab, including custom environment configs
- **[API Reference](api/rl_agents.md)** — every public module, generated from the source docstrings
- **[Changelog](changelog.md)** — what changed and when