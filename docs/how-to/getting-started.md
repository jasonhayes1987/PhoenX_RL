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

The bundled examples enable `WandbCallback`. Authenticate before training —
checked in this order: set the `WANDB_API_KEY` environment variable; place a
`wandb_api_key` file next to the installed `phoenx` package (in a clone that
is `src/phoenx/wandb_api_key`); or run `wandb login` so credentials are cached
in `~/.netrc` / `~/_netrc`. Prefer the env var or `wandb login`: the key file
holds a live credential, and while `src/phoenx/wandb_api_key` is gitignored, an
installed (non-clone) copy sits outside any repository that could ignore it.

### What training prints

PhoenX itself does not attach a console log handler
(`configure_logging` in `phoenx.logging_config`). Everything on your
terminal comes from the simulator (Isaac Sim / Omniverse Kit when using
Isaac Lab), Weights & Biases, and the Rich live dashboard below.

An Isaac Lab run spends the first stretch printing Kit startup noise: the
user config path, a Kit log file path, optional carb crash-reporter warnings
(noise, not a failure), a GPU / OS table, and extension deprecation warnings.
Then Isaac Lab prints manager summary tables (Command, Event, Recorder,
Action, Observation, Termination, Reward, Curriculum). Those tables are the
most useful startup output — they show the exact observation and action shapes
the agent will see. Excerpt from a real
`IsaacSim/franka/cube_lift/dense/ppo_camera.yml`-based run (elided Kit
warnings above):

```text
[INFO]: Time taken for simulation start : 4.325698 seconds
[INFO] Action Manager:  <ActionManager> contains 2 active terms.
+------------------------------------+
|   Active Action Terms (shape: 8)   |
+-------+----------------+-----------+
| Index | Name           | Dimension |
+-------+----------------+-----------+
|   0   | arm_action     |         7 |
|   1   | gripper_action |         1 |
+-------+----------------+-----------+

[INFO] Observation Manager: <ObservationManager> contains 2 groups.
+-----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (33,)) |
+----------+-------------------------------------+----------+
|  Index   | Name                                |  Shape   |
+----------+-------------------------------------+----------+
|    0     | joint_pos                           |   (9,)   |
|    1     | joint_vel                           |   (9,)   |
|    2     | target_object_position              |   (7,)   |
|    3     | actions                             |   (8,)   |
+----------+-------------------------------------+----------+

[INFO]: Completed setting up the environment...
Setting seed: 42
```

On an RTX 4090 / Windows 11 machine with warm caches, that Isaac path took
about 40 seconds from the first Kit line to `Completed setting up the
environment...`, and about 50 seconds to the first training step. Cold first
runs were not measured here.

After the env is up, W&B prints its sync block (see below), then the live
dashboard takes over.

### Live training dashboard

[Trainer.train][phoenx.trainer.Trainer.train] wraps the loop in a Rich
`Live` display (`transient=True`) and rebuilds a table titled
**Live Training Dashboard** every iteration with five columns: Steps,
Episodes, Avg Reward, Episodes/sec, and Elapsed. It redraws in place rather
than scrolling. Because it is transient, the table is erased when the run
exits — piping stdout to a file captures Isaac / W&B lines but not the
dashboard.

`Avg Reward` is the mean over the trainer's score history. `Episodes` counts
finished episodes summed across all parallel envs, so with many envs it often
jumps by `num_envs` when they time out together, and it sits at 0 for the
whole first episode length.

Representative frame from a 120,000-step Isaac PPO camera run (column widths
follow the terminal; values change every refresh):

```text
                            Live Training Dashboard
┌──────────────┬───────────────┬──────────────────┬─────────────────────┬─────────────┐
│        Steps │      Episodes │       Avg Reward │        Episodes/sec │     Elapsed │
├──────────────┼───────────────┼──────────────────┼─────────────────────┼─────────────┤
│      119,936 │           388 │             1.08 │                4.73 │     0:01:22 │
└──────────────┴───────────────┴──────────────────┴─────────────────────┴─────────────┘
```

### Where checkpoints and logs land

The top-level YAML `save_dir` is used verbatim. A relative value resolves
against the **current working directory** — not the config file's location and
not the package. The same config with
`save_dir: ./Trained_Models/IsaacSim/Franka/CubeLift/...` wrote under the repo
root when training was started from the repo root, and under `src/` when
started from `src/`. Pick one cwd and stick to it, or you will lose a run or
overwrite another.

Checkpoints are not periodic.
[Trainer.step][phoenx.trainer.Trainer.step] saves only when a just-finished
episode pushes the running average reward above the best seen so far in this
run (`best: True` on that episode log, then `self.save()`). A 120,000-step
Isaac PPO camera run saved five times.

On-disk layout is documented on [Trainer.save][phoenx.trainer.Trainer.save].
After that same run:

```text
Trained_Models/IsaacSim/Franka/CubeLift/PPO_CAM_DOCS_EVIDENCE/
  config.json                                  9,723
  phoenx.log                                       0
  rng.pt                                      14,533
  trainer_state.pt                             4,759
  agent/agent_state.pt                         1,285
  agent/model.pt                          51,018,195
  agent/normalizers/advantage_normalizer.pt    2,043
  agent/normalizers/state_normalizer.pt        2,915
```

There is no `buffer.pt`: it is written only when `save_buffer=True`, which a
normal training run does not pass. That matters for resume —
`phoenx-train --agent_dir ...` loads with `load_buffer=True`, so an off-policy
run resumed from a typical checkpoint starts with an empty replay buffer.

Logging: `configure_logging` attaches only a `RotatingFileHandler` to
`<save_dir>/phoenx.log`, opened with `mode="w"` (truncated at the start of
every run). There is no console handler. At the default `INFO` level that
file was 0 bytes across several measured runs; the bundled configs'
interesting diagnostics are emitted at `DEBUG`.

### Weights & Biases during training

Authentication is covered above. At run start,
[WandbCallback.on_train_begin][phoenx.rl_callbacks.WandbCallback.on_train_begin]
authenticates, then `wandb.init` with the run named `train-<N>` (next run
number in the project), `group="group-<N>"`, `job_type="train"`, and the
entire trainer config tree as the W&B config; then it registers the agent's
modules with `wandb.watch(log='all', log_freq=100, log_graph=True)`.

```text
wandb: Currently logged in as: <user> to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.27.0
wandb: Run data is saved locally in .../PhoenX_RL/wandb/run-...
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run train-7
wandb: View project at https://wandb.ai/<entity>/<project>
wandb: View run at https://wandb.ai/<entity>/<project>/runs/<id>
wandb: logging graph, to disable use `wandb.watch(log_graph=False)`
```

The local W&B run directory `wandb/` is created under the current working
directory, not under `save_dir`. Every best-model checkpoint also uploads a
model artifact (you will see an `Adding directory to artifact (...)` line).
Use `wandb offline` if you want the run recorded without syncing.

## Evaluate a trained agent

`phoenx-test` loads a saved agent directory (requires `--agent_dir`):

```bash
phoenx-test --agent_dir path/to/saved/agent_dir
phoenx-test --agent_dir path/to/saved/agent_dir --num_episodes 10 --render_mode human
```
Optional flags: `--env`, `--num_episodes`, `--num_envs`, `--render_mode`,
`--seed`, `--log_level`.

### What the agent directory must contain

`config.json` is mandatory — without it `phoenx-test` raises
`FileNotFoundError: No config.json in <dir>`. The `agent/` subtree supplies
weights, optimizers, and normalizers; `trainer_state.pt` and `rng.pt` are
loaded when present. `buffer.pt` is not needed —
[Trainer.load][phoenx.trainer.Trainer.load] is called with
`load_buffer=False`.

The environment is rebuilt from the saved `config["env"]`, with `cfg`,
`num_envs`, `render_mode`, and `seed` overridden by the matching CLI flags.
Training-only wrappers (`VectorNStepReward`) are stripped because they need a
per-step `set_action` call. Saved observation / goal normalizers are restored
and set to eval mode, so the policy sees the inputs it was trained on.

If the env class has moved since the agent was saved, the `env.config.cfg`
string inside `config.json` is still a `module:Class` path resolved by import
at load time. An older Isaac agent whose cfg pointed at a removed
`Configs.IsaacSim...` module failed with
`No module named 'Configs'`. Pass `--env` with the current `module:Class`
path to override that field.

The default `--render_mode` is `human`. For Isaac agents any value other than
`headless` opens the Isaac Sim GUI; for Gymnasium agents `human` opens a
window (see the module docstring on `phoenx.cli.test`).

### Where results and videos land

Evaluation writes nothing into the agent directory except `phoenx.log`
(`configure_logging` is called with `log_dir=agent_dir`, and
[Trainer.test][phoenx.trainer.Trainer.test] never calls `save()`). Results
surface in two places: the transient **Live Testing Dashboard** on the
console, and W&B if the loaded config has a `WandbCallback` (run named
`test-<N>`, `job_type="test"`).

Representative dashboard frame from a real Isaac evaluation
(`--num_episodes 2`, `--num_envs 4`):

```text
                            Live Testing Dashboard
┌───────────┬────────────────┬───────────────────┬──────────────────────┬─────────────┐
│     Steps │       Episodes │        Avg Reward │         Episodes/sec │     Elapsed │
├───────────┼────────────────┼───────────────────┼──────────────────────┼─────────────┤
│     1,000 │              4 │            150.79 │                 0.36 │     0:00:11 │
└───────────┴────────────────┴───────────────────┴──────────────────────┴─────────────┘
```

That run requested 2 episodes but finished 4: all four parallel envs hit the
250-step time-out on the same step, and the budget is only checked once per
step. With `--num_envs > 1`, `--num_episodes` is a floor, not an exact count.

W&B end-of-run summary from the same evaluation (elided sync noise):

```text
wandb: Run summary:
wandb:            avg_reward 150.79
wandb:                   env 3
wandb:               episode 4
wandb:        episode_reward 151.55
wandb:         episode_steps 250
wandb: step_intrinsic_reward 0
wandb:           step_reward 0.70274
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
```

Videos are produced only when the config has a `renderer` section.
`Renderer` writes `<renderer.save_dir>/renders/<train|test>/episode_<N>.mp4`
via imageio (default `fps: 30`, `codec: libx264`) and logs a `wandb.Video`
when a `WandbCallback` is present. `render_freq` is in episodes;
`render_freq <= 0` disables rendering. `Renderer.render_episode` raises
`ValueError` for Isaac Sim environments, so Isaac agents produce no mp4s —
watch live instead with a non-headless `--render_mode`. That is why the
evaluation above reported `0 media file(s)`.

### Worked example (Isaac PPO camera)

From the repo root, evaluate a saved Franka cube-lift agent headless with
four parallel envs (override `--env` because this agent's saved cfg path was
stale):

```bash
python src/phoenx/cli/test.py \
  --agent_dir "src/Trained_Models/IsaacSim/Franka/CubeLift/PPO_CAM_1_BLIND" \
  --env "phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg_PLAY" \
  --num_envs 4 --num_episodes 2 --render_mode headless
```

`phoenx-test --agent_dir ...` with the same flags is equivalent when the
console entry point is on your `PATH`.

## Next steps

- Understand and customize the config schema: [Configuration Files](configurations.md)
- Train in simulation: [Isaac Sim Environments](isaac-sim.md)
