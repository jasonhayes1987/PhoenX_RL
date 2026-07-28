# PhoenX RL
![](src/assets/Python-3.8+-blue.svg)
![](src/assets/PyTorch-2.0+-orange.svg)
![](src/assets/License-MIT-yellow.svg)
<table>
  <tr>
    <td><img src="assets/cheetah.gif" alt="Cheetah" width="200"></td>
    <td><img src="assets/lunar.gif" alt="Lunar Lander" width="200"></td>
    <td><img src="assets/pong.gif" alt="Atari Pong" width="200"></td>
    <td><img src="assets/double_pendulum.gif" alt="Inverted Pendulum" width="200"></td>
  </tr>
  <tr>
    <td><img src="assets/reach.gif" alt="Fetch Reach" width="200"></td>
    <td><img src="assets/push.gif" alt="Fetch Push" width="200"></td>
    <td><img src="assets/slide.gif" alt="Fetch Slide" width="200"></td>
    <td><img src="assets/place.gif" alt="Fetch Pick and Place" width="200"></td>
  </tr>
</table>

## Overview
PhoenX RL is a flexible, modular reinforcement learning (RL) framework designed for rapid experimentation and development of RL agents. Built on PyTorch, it supports a variety of on-policy and off-policy algorithms, integrates with Gymnasium and IsaacSim environments, and includes components like intrinsic curiosity, N-step returns, replay buffers, schedulers, normalizers, and optional experiment logging via Weights & Biases (WandB).

The framework emphasizes extensibility, allowing users to customize models, noise processes, buffers, and callbacks. It is suitable for both research and practical applications in robotics, games, and control tasks, with support for goal-oriented learning and scaling.

## Key Features
- **Supported Algorithms:**
   - **On-Policy**: Reinforce, Actor-Critic, Proximal Policy Optimization (PPO) with adaptive KL divergence.
   - **Off-Policy**: Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), Soft Actor-Critic (SAC).
   - **Goal-Oriented**: DDPG, TD3, SAC, PPO, Hindsight Experience Replay (HER) with DDPG/TD3/SAC backends.
- **Modular roots → trunk → branches networks:** every agent is built on a
  composite `ModularModel` — user-named per-modality encoder **roots**, an
  optional shared **trunk** (the single place temporal layers may live), and
  per-role **branches** (policy / value / critic heads). Per-module optimizers
  with research-backed gradient ownership: on-policy agents combine losses into
  ONE backward + coordinated step (SB3 / RSL-RL standard); off-policy agents
  let the critic loss own the shared body while the policy trains on detached
  features (SAC-AE / DrQ-v2 standard). See
  `` and
  `configs/multi_modal_cfg.yml`.
- **Multi-modal observations:** Dict observation spaces (e.g. camera +
  proprioception) flow end to end — env wrappers, every buffer, per-key
  `DictNormalizer`/`ImageScale` normalizers, HER, and intrinsic motivation —
  with uint8 image storage and per-root `input_keys` routing.
- **Temporal policies:** LSTM/GRU trunks (recurrent PPO/ActorCritic with
  sequence minibatching and masked mid-sequence resets; recurrent
  SAC/TD3/DDPG with R2D2-style stored initial hidden + optional burn-in) and
  causal-transformer trunks with rolling context-window inference.
- **Modular Components:**
   - **Models**: Head classes (stochastic discrete/continuous policies, deterministic actors, value and Q heads) with a data-driven layer registry (dense, conv1d/2d/3d, pooling, norms, activations, embeddings, attention, transformer encoders, LSTM/GRU) and per-layer weight initializers.
   - **Noise Processes**: Ornstein-Uhlenbeck (OU), Normal, Uniform noise for exploration, with optional scheduling.
   - **Normalizers**: Running statistics for observations/actions/goals, per-key Dict normalizers, image scaling.
   - **Schedulers**: Learning rate and parameter schedulers (linear, step, cosine annealing, exponential), attachable per module.
   - **Buffers**: Standard ReplayBuffer, PrioritizedReplayBuffer (proportional or rank-based), with N-step return support, trajectory tracking, and R2D2 stored-state support.
   - **Intrinsic Curiosity Module (ICM)**: For exploration in sparse-reward environments.
   - **Callbacks**: WandB integration for logging, metrics, and artifact saving; extensible for custom hooks, with distributed variants.
- **Experiment Logging**: Optional Weights & Biases (WandB) callback support for metrics/logging and artifact saving.
- **Environment Support**: Gymnasium and IsaacSim

## Installation
PhoenX RL is designed to be installed locally via Conda + Poetry. The recommended path is to run the provided PowerShell setup script after cloning.

See `installation.md` for the full, up-to-date instructions.

### Quick start (Windows / PowerShell)
```bash
git clone <your-repo-url>
cd PhoenX_RL
.\setup.ps1
```

## Usage
PhoenX RL uses a simple two-step workflow:

1. **Create an agent directory from a YAML config** (writes `config.json`, `train_config.json`, and `test_config.json` into `save_dir`)
2. **Train using `train.py` by pointing at that saved directory**

### 1) Edit a YAML config
Configs live in `configs/`. The ones currently wired up to the YAML build scripts are:
- `configs/reinforce.yml`
- `configs/actor_critic.yml`
- `configs/ddpg.yml`

Update at least:
- `save_dir`: where the agent + train/test configs will be written (recommended: change this to a path on your machine)
- `env`: environment type/config
- `device`: `"cuda"` or `"cpu"`

### 2) Build/save the agent from the YAML config
```bash
python src/phoenx/builders/reinforce.py --config_file configs/reinforce.yml
python src/phoenx/builders/actor_critic.py --config_file configs/actor_critic.yml
python src/phoenx/builders/ddpg.py --config_file configs/ddpg.yml
```

### 3) Train from the saved agent directory
```bash
python src/phoenx/cli/train.py --agent_dir "path/to/your/saved/agent_dir"
```

You can override some training options from the CLI (for example `--render_freq`, `--num_episodes`, etc.); otherwise `train.py` reads them from `train_config.json` in the agent directory.

# Roadmap
PhoenX RL is actively evolving. Future plans include:
- **MARL**: Adding Multi-Agent variants for DDPG, TD3, PPO, and SAC
- **HRL**: Adding Hierarchical Reinforcement Learning support (Most likely through Ray RLib)

# License
This project is licensed under the MIT License - see the LICENSE file for details.
