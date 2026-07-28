# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- MkDocs documentation site with auto-generated API reference.

### Changed
- Isaac Sim/Lab installation and GPU verification is now part of the pytest suite as `tests/test_isaac_setup.py` (marker `isaac`, auto-skips without `isaaclab`/CUDA), replacing the uncollected `tests/isaac_test.py` script.
- The off-policy buffers' missing-`VectorNStepReward` error now names the buffer class and explains the config remedy (add the wrapper with `n >= 1` to the env's `wrappers` list).

### Removed
- `RayWandbCallback` (Ray-distributed W&B logging callback in `src/phoenx/rl_callbacks.py`), the unused `convert_to_distributed_callbacks` helper in `src/phoenx/agent_utils.py`, and the co-occurrence sweep helpers `calculate_co_occurrence_matrix` and `plot_co_occurrence_heatmap` in `src/phoenx/wandb_support.py` (the latter was the plotly-based one). `phoenx` no longer imports `plotly` or `scipy.stats` anywhere.

### Fixed
- `tests/test_dict_datapath.py` storage-dtype tests failed at `ReplayBuffer` construction because their stub env lacked the required `VectorNStepReward` wrapper; they now exercise the real `GymnasiumWrapper` + `VectorNStepReward(n=1)` stack against a registered lying-space env (`PhoenXLyingSpace-v0`).
- Importing any `phoenx` submodule that reached `src/phoenx/wandb_support.py` raised `ModuleNotFoundError: No module named 'plotly'`, because that module did a hard top-level `import plotly.graph_objs` on an undeclared dependency. This blocked pytest collection for seven test modules, taking the suite from 281 to 407 collected tests once the import was removed.
- `rich` is now declared as a runtime dependency. `src/phoenx/trainer.py` imports `rich.live`, `rich.table`, and `rich.console` at module scope for its live training and testing dashboards, but the package was never listed in `pyproject.toml`; the resulting `ModuleNotFoundError` had been masked by the `plotly` failure firing earlier in the import chain. Declaring and installing it takes the suite to 414 collected tests with no collection errors.
- `pyproject.toml` was invalid TOML: the `envpool>=1.2.5` dependency entry was missing its closing quote, so the file could not be parsed.
- `tests/test_learning_smoke.py::test_multimodal_ppo_learns_on_dict_obs` failed its `mean_action > 0.3` assertion (measuring 0.285) even though the policy had learned, because it sampled the deterministic action only at reset states. `MultiModalTestEnv` encodes the timestep in its observation, so the learned action ramps from 0.28 at `t=0` to 0.999 at `t=19`, making reset the weakest state in the episode. The test now rolls out one full deterministic episode and asserts the return exceeds 20.0 of a possible 40.0 (measured 34.3, random ~0).
