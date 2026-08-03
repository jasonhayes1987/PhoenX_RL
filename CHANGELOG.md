# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Machine-local `.phoenx-env` record (gitignored) plus `scripts/use-env.ps1`,
  `scripts/activate.ps1`, and `scripts/activate.sh`. An environment *name* is
  ambiguous when several conda installations on one machine each have an env
  called e.g. `rl_env`; the record stores the absolute prefix (and the conda
  root that owns it) so activation always hits the intended install. `use-env.ps1`
  writes the record for an existing checkout; the activate scripts load conda
  from the *recorded* root (not whichever `conda` is first on `PATH`), activate
  by prefix, and set `PYTHONNOUSERSITE=1`.
- `ruff` to the `dev` extra. `[tool.ruff.lint]` selects pydocstyle (`D`) with the
  google convention.
- Three portable example configs bundled as package data under `phoenx/examples/configs/`, so `phoenx-train --config LunarLanderContinuous-v3/sac.yml` works from a bare `pip install` with no clone and no `configs/` directory.
- A fourth bundled example config, `IsaacSim/franka/cube_lift/dense/ppo_camera.yml`, covering a multi-modal Isaac run (proprioception plus an RGB camera).
- `phoenx.examples.isaac`, shipping the `custom_franka_cube_lift_cfg` and `custom_franka_reach_cfg` Isaac Lab environment definitions so configs can reference them by real module path.
- `phoenx.builder.available_example_configs()`, and a bundled-example fallback in `load_config` (an existing on-disk path still always wins).
- `tests/test_example_configs.py`, 11 tests pinning the resolution order, separator handling, and error message. The suite went from 414 to 425 collected tests (`411 passed, 14 deselected`).
- MkDocs documentation site with auto-generated API reference.

### Changed
- `setup.ps1` gained `-CondaRoot` (default `%USERPROFILE%\miniforge3`; the
  Miniforge path was previously hardcoded) so an install can target an existing
  conda installation, and now writes `.phoenx-env` after a successful run so
  new terminals can activate via the scripts above.
- `installation.md` rewritten around the two canonical install sequences (Gymnasium mode and Isaac Lab mode); README install and usage sections corrected to match the real `phoenx-train` / `phoenx-test` CLI and the bundled example configs.
- Documented the SWIG prerequisite in the manual install sequences (and related README / Getting Started pages). Without `conda install -y -c conda-forge swig` before the final `pip install`, those sequences could not succeed on a clean Windows machine. `setup.ps1` users are unaffected — the script installs SWIG. Recorded the one validated configuration from the first end-to-end `setup.ps1 -Isaac` run (Windows 11 / RTX 4090 / driver 581.08 / Python 3.11.15; Isaac Lab 2.3.2.post1 + Isaac Sim 5.1.0.0 + torch 2.7.0+cu128; 410 passed / 14 deselected Gymnasium and Isaac; 13 passed / 1 skipped GPU `isaac` integration tests). Linux has not been tested.
- Isaac Sim/Lab installation and GPU verification is now part of the pytest suite as `tests/test_isaac_setup.py` (marker `isaac`, auto-skips without `isaaclab`/CUDA), replacing the uncollected `tests/isaac_test.py` script.
- The off-policy buffers' missing-`VectorNStepReward` error now names the buffer class and explains the config remedy (add the wrapper with `n >= 1` to the env's `wrappers` list).
- `available_example_configs()` now walks the packaged configs tree recursively instead of one level of subdirectories. The previous version silently omitted bundled configs at other depths, which also meant `load_config`'s `FileNotFoundError` advertised an incomplete list of names that were in fact loadable.
- The newly bundled Isaac camera config had its `save_dir` rewritten from an absolute path on the author's machine to a relative `./Trained_Models/...` path, so every shipped example is now portable.
- The bundled Isaac camera example's `env.config.cfg` now points at `phoenx.examples.isaac.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraBlindEnvCfg`. It previously used a `Configs.IsaacSim...` prefix that raised `ModuleNotFoundError: No module named 'Configs'`.
- Tests no longer depend on the untracked `configs/` tree. `TestCanonicalConfig`'s Isaac camera test reads the packaged copy via `importlib.resources`, while the multi-modal config and `test_roundtrip.py`'s legacy-schema SAC config moved to `tests/fixtures/`.

### Removed
- The top-level `configs/` tree (32 files) is no longer tracked; `/configs/` is gitignored, anchored with a leading slash so it cannot also match the shipped `src/phoenx/examples/configs/`. It remains a personal working directory on the author's machine and is simply absent from a fresh clone.
- `environment.yml`, superseded by `pyproject.toml`.
- `RayWandbCallback` (Ray-distributed W&B logging callback in `src/phoenx/rl_callbacks.py`), the unused `convert_to_distributed_callbacks` helper in `src/phoenx/agent_utils.py`, and the co-occurrence sweep helpers `calculate_co_occurrence_matrix` and `plot_co_occurrence_heatmap` in `src/phoenx/wandb_support.py` (the latter was the plotly-based one). `phoenx` no longer imports `plotly` or `scipy.stats` anywhere.

### Fixed
- `setup.ps1` prepended the base Miniforge installation to `PATH`, and because `conda run -p <prefix> python` resolves `python` from `PATH`, every command the script ran used the BASE interpreter instead of the environment's. On this machine that meant Python 3.13 rather than the environment's 3.11, so pip filtered out every cp311 wheel and the Isaac install failed with the misleading "Could not find a version that satisfies the requirement isaaclab==2.3.2.post1 (from versions: 3.0.0b2, 3.0.0b2.post1)" — the requested version existed all along. The prepend is removed, and Step 1 now asserts that `conda run` resolves `sys.prefix` to the target environment, failing loudly rather than silently installing into the wrong one.
- `setup.ps1`'s `Invoke-EnvPython` helper declared a `[Parameter()]` attribute, making it an advanced function and pulling in PowerShell's common parameters. `pip install -e .` then failed with "Parameter cannot be processed because the parameter name 'e' is ambiguous. Possible matches include: -ErrorAction -ErrorVariable." It is now a simple function that forwards the automatic `$args` untouched.
- `setup.ps1` did not install SWIG, so `pip install -e .` failed while building `box2d-py` (pinned by `gymnasium[box2d]` and lacking a cp311 Windows wheel). A build-prerequisites step now installs `swig` from conda-forge, chosen over the PyPI package because the latter is a console-script shim that cannot import itself under pip's build isolation.
- `setup.ps1` installed in the wrong order: PyTorch, then PhoenX, then Isaac Lab. Because `isaacsim` declares its own torch dependency and the last install wins, and because PhoenX's floor constraints are only satisfied cheaply once Isaac is already in place, the order is now Isaac Lab, then the cu128 PyTorch build, then PhoenX — matching NVIDIA's Isaac Lab pip-installation docs.
- `setup.ps1` treated `envpool` as Linux-only, attempting it, swallowing the failure, and printing that vectorized envs "will fall back to Gymnasium's own vector API". All of that was wrong: envpool 1.2.5 ships Windows wheels, `envpool>=1.2.5` is a declared dependency, and `src/phoenx/env_wrapper.py` imports it unconditionally at module scope, so the advertised fallback never existed. The step is removed.
- `setup.ps1` installed `gymnasium-robotics` from git pinned at v1.4.0, downgrading the 1.4.2 that pip already resolves from the declared `gymnasium-robotics>=1.4.0`. The redundant step is removed.
- `setup.ps1` pinned `isaaclab[isaacsim]==2.3.0`; it now installs `isaaclab[isaacsim,all]==2.3.2.post1`.
- `setup.ps1`'s closing suggestion pointed at `configs/LunarLander-v3/sac.yml`, whose `save_dir` is an absolute path on the original author's machine. It now suggests the bundled name `LunarLanderContinuous-v3/sac.yml`, which resolves from a bare install and needs no `configs/` directory.
- Wheel contents: `[tool.setuptools.packages.find]` relied on setuptools' default `namespaces = true`, so package discovery treated any directory with a valid Python identifier as a package. Under `src/` that found 9,897 packages, including top-level `Trained_Models` (a large tree of training artifacts) and `wandb` (run output). Setting `namespaces = false` restricts discovery to real packages, yielding exactly `phoenx`, `phoenx.builders`, `phoenx.cli`, and `phoenx.examples`. This required adding `src/phoenx/builders/__init__.py`, which had been resolving only as an implicit namespace package.
- `tests/test_dict_datapath.py` storage-dtype tests failed at `ReplayBuffer` construction because their stub env lacked the required `VectorNStepReward` wrapper; they now exercise the real `GymnasiumWrapper` + `VectorNStepReward(n=1)` stack against a registered lying-space env (`PhoenXLyingSpace-v0`).
- Importing any `phoenx` submodule that reached `src/phoenx/wandb_support.py` raised `ModuleNotFoundError: No module named 'plotly'`, because that module did a hard top-level `import plotly.graph_objs` on an undeclared dependency. This blocked pytest collection for seven test modules, taking the suite from 281 to 407 collected tests once the import was removed.
- `rich` is now declared as a runtime dependency. `src/phoenx/trainer.py` imports `rich.live`, `rich.table`, and `rich.console` at module scope for its live training and testing dashboards, but the package was never listed in `pyproject.toml`; the resulting `ModuleNotFoundError` had been masked by the `plotly` failure firing earlier in the import chain. Declaring and installing it takes the suite to 414 collected tests with no collection errors.
- `pyproject.toml` was invalid TOML: the `envpool>=1.2.5` dependency entry was missing its closing quote, so the file could not be parsed.
- `tests/test_learning_smoke.py::test_multimodal_ppo_learns_on_dict_obs` failed its `mean_action > 0.3` assertion (measuring 0.285) even though the policy had learned, because it sampled the deterministic action only at reset states. `MultiModalTestEnv` encodes the timestep in its observation, so the learned action ramps from 0.28 at `t=0` to 0.999 at `t=19`, making reset the weakest state in the episode. The test now rolls out one full deterministic episode and asserts the return exceeds 20.0 of a possible 40.0 (measured 34.3, random ~0).
