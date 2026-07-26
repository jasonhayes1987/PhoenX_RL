# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- MkDocs documentation site with auto-generated API reference.

### Changed
- Isaac Sim/Lab installation and GPU verification is now part of the pytest suite as `tests/test_isaac_setup.py` (marker `isaac`, auto-skips without `isaaclab`/CUDA), replacing the uncollected `tests/isaac_test.py` script.
- The off-policy buffers' missing-`VectorNStepReward` error now names the buffer class and explains the config remedy (add the wrapper with `n >= 1` to the env's `wrappers` list).

### Fixed
- `tests/test_dict_datapath.py` storage-dtype tests failed at `ReplayBuffer` construction because their stub env lacked the required `VectorNStepReward` wrapper; they now exercise the real `GymnasiumWrapper` + `VectorNStepReward(n=1)` stack against a registered lying-space env (`PhoenXLyingSpace-v0`).