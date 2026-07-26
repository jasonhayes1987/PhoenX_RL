# Getting Started

PhoenX runs in two modes. **Gymnasium mode** needs nothing but pip and works on
any machine. **Isaac Lab mode** adds GPU-accelerated robotics simulation and
requires NVIDIA's Isaac Sim environment.

## Install — Gymnasium mode

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
```

For development (editable install with test tooling):

```bash
git clone https://github.com/jasonhayes1987/PhoenX_RL.git
cd PhoenX_RL
pip install -e ".[dev]"
```

## Install — Isaac Lab mode

Create the Isaac Lab environment first, following
[NVIDIA's installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
(conda environment with the Python version matching your Isaac Sim release),
then install PhoenX into that same environment with the pip command above.
Verify the setup with `pytest tests/test_isaac_setup.py -v` (see
[Isaac Sim Environments](isaac-sim.md)).

## Train an agent

Training is driven entirely by a YAML config file:

```bash
phoenx-train --config configs/LunarLander-v3/sac.yml
```

<!-- TODO(docs-writer): document what appears during training — console output,
     where checkpoints/logs are written, W&B integration if enabled. -->

## Evaluate a trained agent

```bash
phoenx-test --config <path-to-config>
```

<!-- TODO(docs-writer): document required config fields for evaluation,
     where results/videos land, and one full worked example. -->

## Next steps

- Understand and customize the config schema: [Configuration Files](configuration.md)
- Train in simulation: [Isaac Sim Environments](isaac-sim.md)
