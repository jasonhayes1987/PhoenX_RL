# Getting Started

PhoenX runs in two modes. **Gymnasium mode** uses conda + pip and works on any
machine with an NVIDIA GPU recommended. **Isaac Lab mode** adds
GPU-accelerated robotics simulation and requires NVIDIA's Isaac Sim
environment. On Windows, both modes need SWIG from conda-forge before the
final PhoenX `pip install` (or use `setup.ps1`, which installs it for you).

Full install sequences, extras, `setup.ps1`, and troubleshooting:
[Installation](installation.md).

## Install — Gymnasium mode

From a clone (editable, with test tooling):

```bash
conda create -n phoenx python=3.11
conda activate phoenx
pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge swig
pip install -e ".[dev,docs]"
```

No-clone (still run the SWIG step on Windows first):

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
```

## Install — Isaac Lab mode

Same sequence with one extra command before PyTorch (PhoenX still last):

```bash
conda create -n phoenx python=3.11
conda activate phoenx
pip install --upgrade pip
pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge swig
pip install -e ".[dev,docs]"
```

Or install Isaac Lab first (see
[NVIDIA's pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)),
then install PhoenX into that same environment (SWIG first on Windows). Verify
with `pytest tests/test_isaac_setup.py -v` (see
[Isaac Sim Environments](isaac-sim.md)).

## Train an agent

At least one of `--config` or `--agent_dir` is required; if both are passed,
`--config` wins. Optional: `--log_level`.

```bash
# Bundled example (works with no clone)
phoenx-train --config LunarLanderContinuous-v3/sac.yml

# Any config file on disk
phoenx-train --config path/to/my_experiment.yml

# Resume from a previously saved agent directory
phoenx-train --agent_dir path/to/saved/agent_dir
```

`phoenx.builder.load_config` resolves `--config` in this order: an existing
on-disk path always wins; otherwise a relative value is looked up among the
bundled examples under `phoenx/examples/configs/`; otherwise
`FileNotFoundError` lists what is available. List them with
`phoenx.builder.available_example_configs()`.

Console entry points: `phoenx-train` and `phoenx-test`.
`python -m phoenx.cli.train` also works.

<!-- TODO: document what appears during training — console output,
     where checkpoints/logs are written, W&B integration if enabled. -->

## Evaluate a trained agent

`phoenx-test` loads a saved agent directory (requires `--agent_dir`):

```bash
phoenx-test --agent_dir path/to/saved/agent_dir
phoenx-test --agent_dir path/to/saved/agent_dir --num_episodes 10 --render_mode human
```

Optional flags: `--env`, `--num_episodes`, `--num_envs`, `--render_mode`,
`--seed`, `--log_level`.

<!-- TODO: document required config fields for evaluation,
     where results/videos land, and one full worked example. -->

## Next steps

- Understand and customize the config schema: [Configuration Files](configurations.md)
- Train in simulation: [Isaac Sim Environments](isaac-sim.md)
