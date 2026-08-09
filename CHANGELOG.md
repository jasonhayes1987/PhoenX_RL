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
- API reference pages for `phoenx.adaptive_kl`, `phoenx.agent_utils`, `phoenx.distributions`, and `phoenx.rl_callbacks` (mkdocstrings stubs under `docs/api/` plus `mkdocs.yml` nav entries). All four are part of the public surface but had no page, so their docstrings were unreachable from the site; the API section now covers 15 modules.

### Changed
- Four dead commented lines in `Trainer.set_normalizers` that would have
  propagated normalizer mode into the agent's intrinsic-motivation module are
  deleted. They were inert for two independent reasons: `RewardNorm.normalize`
  and `RewardNorm.add` never read `self.training`, so the mode flag does
  nothing for the only normalizer an `IntrinsicMotivation` owns; and the
  trainer computes intrinsic rewards only when `training` is true, so the
  intrinsic path never runs during evaluation. `IntrinsicMotivation.set_normalizers_mode`
  is kept, since it becomes meaningful if `RewardNorm` ever honours the flag.
- `phoenx.intrinsic_motivation.CompositeIntrinsicMotivation` type annotations
  corrected to match the docstrings written during the docstring pass:
  `_split_components` now returns
  `tuple[list[tuple[int, IntrinsicMotivation]], list[tuple[int, IntrinsicMotivation]]]`
  (it returns `(index, component)` pairs, not bare components), `_weights_for`
  now returns `list[float] | None` (it returns `None` when no `weights` are
  configured), and `set_normalizers_mode` now takes `Literal['train', 'eval']`
  to match the base class it forwards to instead of the nonexistent `'test'`.
  `is_online` on the composite is now `True` when any child component is
  online, rather than always `False`; nothing in the package currently reads
  the composite's own `is_online` (only its children's, in
  `_split_components`), so this changes no behavior today.
- Google-style docstrings completed for `phoenx.schedulers`, `phoenx.noise`,
  `phoenx.distributions`, `phoenx.builder`, `phoenx.her`, `phoenx.trainer`,
  `phoenx.rl_callbacks`, `phoenx.normalizer`, `phoenx.buffer`,
  `phoenx.adaptive_kl`, `phoenx.agent_utils`, `phoenx.intrinsic_motivation`,
  `phoenx.env_wrapper`, and `phoenx.models` (docstring-and-docs-completion
  plan). `her.py`'s module docstring also lost the `===`
  banner rows that mkdocstrings rendered as literal `=` characters on the HER
  page, and two of its usage examples were simply wrong: buffers take
  `hindsight=`, not `relabeler=`, and `strategy='future'` samples `[t, T_ep)`
  or `(t, T_ep)` depending on `future_lo`. The `intrinsic_motivation` module
  docstring was rewritten: it previously opened with its own filename as the
  summary line and used box-drawing characters to sketch the class hierarchy.
- Docstring cross-references in those modules now use Markdown and
  mkdocs-autorefs (`[load_config][phoenx.builder.load_config]`) rather than
  Sphinx roles. mkdocstrings renders docstring bodies as Markdown, so
  `:func:` and `:meth:` reached the built pages as literal text; twelve of the
  thirteen converted references are now real links, the exception being one in
  `obs_utils.py`, which has no API reference page to link to.
- The draft commented-out implementation inside `ScheduleWrapper.from_config`
  is removed; the method has always simply returned `cls(**config)`.
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
- `Buffer` is now a real ABC (`class Buffer(ABC)`). It previously imported `abstractmethod` but never `ABC`, so the base was directly instantiable and its declared `add`/`sample` signatures matched no concrete subclass. Direct instantiation now raises `TypeError`. In-repo callers are unaffected (`Buffer.create_instance` and all concrete subclasses are unchanged), but any out-of-repo code that subclassed without implementing `add`/`sample`, or that instantiated `Buffer` directly, will break.

### Removed
- The dead single-env `NStepReward` wrapper (`src/phoenx/env_wrapper.py`), superseded roughly five months ago by `VectorNStepReward` and never reachable since: `EnvWrapper._find_nstep_wrapper` only matches `VectorNStepReward`, and `NStepReward` never gained the `set_action` / `set_intrinsic_motivation` methods the trainer and renderer call to feed it per-step data, so it could never receive an action. Nothing in the repo — no test, no bundled or committed config, no doc — ever referenced it by name. The only visible effect is on `WRAPPER_REGISTRY`: a hand-written YAML or an old saved run's `config.json` listing `{"type": "NStepReward"}` will now fail env construction with `ValueError: Unknown wrapper type 'NStepReward'` instead of silently building an inert wrapper. Use `{"type": "VectorNStepReward", "params": {"n": ...}}` instead.
- `wrap_env` (`src/phoenx/env_wrapper.py`), which nothing in the repository called. It rebuilt a `SyncVectorEnv` from scratch — one fresh `gym.make(vec_env.spec.id)` per sub-env with the registry wrappers applied — duplicating what each adapter's own `_initialize_env` already does, and it silently skipped wrapper types missing from `WRAPPER_REGISTRY` where `_initialize_env` raises `ValueError`. It was never imported or referenced, so no configuration or call site changes.
- Twelve unused import statements from `src/phoenx/agent_utils.py`: `json`, `math.e`, `pathlib.Path`, `typing.{Dict, Any, List, Optional}`, `numpy`, `torch.nn`, and the relative imports of `.models`, `.env_wrapper`, `.buffer`, `.noise`, `.normalizer`, and `.schedulers`. Nothing the module defines referenced any of them. This is API-visible rather than purely internal: the six relative imports re-exported roughly two dozen names, so code doing `from phoenx.agent_utils import ScheduleWrapper` (or reaching `agent_utils.np`) was leaning on a re-export that no longer exists and should import from the owning module instead.
- The top-level `configs/` tree (32 files) is no longer tracked; `/configs/` is gitignored, anchored with a leading slash so it cannot also match the shipped `src/phoenx/examples/configs/`. It remains a personal working directory on the author's machine and is simply absent from a fresh clone.
- `environment.yml`, superseded by `pyproject.toml`.
- `RayWandbCallback` (Ray-distributed W&B logging callback in `src/phoenx/rl_callbacks.py`), the unused `convert_to_distributed_callbacks` helper in `src/phoenx/agent_utils.py`, and the co-occurrence sweep helpers `calculate_co_occurrence_matrix` and `plot_co_occurrence_heatmap` in `src/phoenx/wandb_support.py` (the latter was the plotly-based one). `phoenx` no longer imports `plotly` or `scipy.stats` anywhere.
- `src/phoenx/gym_helper.py` (180 lines), unreferenced anywhere in the repo. It predates `her.py`: a hardcoded table mapping Gymnasium spec IDs to per-env `desired_goal` / `achieved_goal` / reward callables, which `HindsightRelabeler` replaced by resolving `compute_reward` off the env stack and reading the goal keys out of the dict observation. The table had also rotted past the pinned dependencies — under `gymnasium==1.2.0` and `gymnasium-robotics` 1.4.0, five of its six keys (`CarRacing-v2`, `FetchReach-v2`, `FetchPickAndPlace-v2`, `FetchPush-v2`, `FetchSlide-v2`) are no longer registered, so `get_her_goal_functions` would `KeyError` on everything but `Reacher-v4`. The neighboring `get_goal_envs` built an empty list and never returned it, so it could only ever have returned `None`.
- `src/phoenx/ray_tune.py` (684 lines), unreferenced anywhere in the repo and
  unable to run in the first place. Its whole import block is flat and
  pre-package (`from schedulers import ScheduleWrapper`, `from models import
  ActorModel, ...`, `from buffer import Buffer`, `from rl_callbacks import
  WandbCallback`), so `import phoenx.ray_tune` raised `ModuleNotFoundError`
  before reaching any sweep code, and one of those imports — `from icm import
  ICM` — names a module that no longer exists at all, `icm.py` having become
  `intrinsic_motivation.py`. Even with the imports repaired it would have
  raised `TypeError`: it called `ScheduleWrapper({"type": ..., "params": ...})`,
  passing a single positional dict to a constructor that takes
  `schedule_type, steps, start_value, end_value`. Ray Tune sweep support will
  be rebuilt from scratch when it is next needed. The `ray[tune,default]`
  dependency is still declared in `pyproject.toml`, though nothing in the
  package imports `ray` any more.
- `src/phoenx/builders/her.py` (298 lines), a scratch script rather than a
  builder. Its siblings in `phoenx.builders` export `build_*` functions; this
  one executed work at import time — a `sys.path` insertion, `from agent_utils
  import *` and `from icm import ICM` (neither resolvable), a hardcoded
  `device='cuda'`, and a `gym.make('FetchPush-v4')` call at module level — so
  importing it could only ever have failed. Nothing referenced it. It is
  unrelated to `phoenx.her`, the live HER implementation that `buffer.py`,
  `trainer.py`, and `builder.py` all import and which is untouched.

### Fixed
- Intrinsic-motivation `reward_scheduler` progress was never restored on
  reload. All four registered classes (`ICM`, `RND`, `EpisodicNovelty`,
  `CompositeIntrinsicMotivation`) persisted the schedule through `get_config()`
  into `config.json` alone, so a reloaded module rebuilt at step 0 and
  silently re-applied the full undecayed intrinsic reward weight on a resumed
  run. This completes the fix begun in commit `2f91896`, which restored the
  schedule's existence after reload but not its progress. `IntrinsicMotivation`
  now writes and reads `intrinsic_motivation/schedule_state.pt` (mirroring
  `agent_state.pt`) via private helpers called from both save paths and once
  from the load dispatcher after `_load_impl`. A missing `schedule_state.pt` is
  a deliberate silent no-op, so checkpoints written before this change keep
  loading with progress simply reset. New `tests/test_schedulers.py` pins
  `ScheduleWrapper.get_state`/`set_state` across linear, cosine, and
  exponential (plus the negative case that `get_config()` alone loses
  progress), and three new tests in `tests/test_intrinsic_motivation.py` cover
  stepped-schedule resume across all four classes, a composite child's
  scheduler, and the missing-file back-compat no-op; the fast subset went from
  `456 passed, 23 deselected` to `470 passed, 23 deselected`.
- `CompositeIntrinsicMotivation` silently dropped its `reward_scheduler` on
  reload. Its `_load_impl` read `config['im_reward_scheduler']`, a key nothing
  in the package has ever written — `get_config` serializes the schedule under
  `reward_scheduler`, which is what `ICM`, `RND`, and `EpisodicNovelty` all read
  back. A reloaded composite therefore came back with `reward_scheduler=None`,
  so `_scaled_reward_weight()` stopped applying the decay and
  `Trainer._step_schedulers` had nothing to advance: a resumed run used an
  undecayed intrinsic reward weight and raised nothing. Child schedulers were
  unaffected, since each child loads through its own `_load_impl`. Note that
  `ScheduleWrapper.get_config()` does not persist `last_epoch`, so a restored
  schedule restarts from step 0 — true for every subclass, and unchanged here.
  A new parametrized regression test covers all four classes, since the
  existing round-trip tests never constructed a module with a scheduler at all.
- Rank-mode prioritized sampling could not reach recently added windows.
  `PrioritizedReplayBuffer._maybe_resort` rebuilt the `sorted_indices` cache
  only when it was `None`, `_sample_rank` pinned the rank support to whatever
  that cache covered, and the only invalidation lived in `update_priorities` —
  so up to `sort_freq` (default 1000) of the newest windows were unsampleable,
  and an agent that sampled without ever reporting TD errors never refreshed
  the ordering at all; importance-sampling weights were also computed against
  the pinned support rather than the true buffer size. `_maybe_resort` now also
  triggers on a coverage mismatch (`sorted_indices.numel() != size`) and on
  staleness (`_samples_since_sort >= sort_freq`), guaranteeing
  `sorted_indices.numel() == size` on exit. Rank mode remains opt-in
  (`priority` defaults to `"proportional"`).
- `PrioritizedReplayBuffer.reset` left prioritized state stale. It inherited
  `ReplayBuffer.reset` unchanged, so the sum tree kept old priorities and
  `_sample_proportional` drew leaf indices across the whole capacity, which
  `sample` then clamped into the small valid region — piling an entire batch
  onto the last valid index. `SumTree.max_priority` and `max_priority_rank`
  also ratcheted upward permanently. New `SumTree.reset()` (zeros the tree,
  restores `max_priority` to 1.0) and new `PrioritizedReplayBuffer.reset()`
  clear the sum tree, or `priorities` / `sorted_indices` / `max_priority_rank`
  / the segment cache in rank mode, and zero `_samples_since_sort`. They
  deliberately do not reset `beta`, `_sample_calls`, or `_beta_updates`,
  because the β-annealing schedule tracks gradient steps rather than buffer
  contents. This bug was latent — nothing in `src/` calls `reset()` on an
  off-policy buffer today.
- Covering all three buffer fixes above, eleven new tests in
  `tests/test_buffer.py` pin rank coverage and wraparound, cache refresh
  without `update_priorities`, rank probabilities against the true buffer
  size, `Buffer` abstractness, reset clearing priority state, and post-reset
  sampling spread; the fast subset went from `441 passed, 23 deselected` to
  `452 passed, 23 deselected`.
- The bundled Isaac camera example set `learn_every: 12288`, sized for the 512
  envs an earlier revision used, while the file itself requests
  `num_envs: 1024` — so it learned twice per rollout instead of once. It is now
  `24576` (24 rollout steps x 1024 envs). The header comment, which still
  described a 256-env run with a 6144 cadence and a 260 MB image buffer, now
  matches the config it documents (~1.04 GB at 1024 envs).
- Best-checkpoint W&B artifacts were never tagged `best`. `WandbCallback`
  branches on `logs.get("best", False)` to re-upload the saved model with
  `model_is_best=True`, which is what appends the `"best"` alias in
  `wandb_support.save_model_artifact` — but nothing in the package ever set
  that key, so the branch was unreachable and every artifact carried only
  `latest` plus its version alias. `Trainer` now sets the key on the last
  episode log of a step whose running average beat the previous best, next to
  the `save()` it already performed. Seven new tests
  pin it: four in `tests/test_trainer.py` covering the marking, its absence
  when the average does not improve, and the `training=False` path, and three
  in `tests/test_rl_callbacks.py` covering the artifact upload.
- `WandbCallback` aborted training with `WANDB_API_KEY not found` even when a
  prior `wandb login` had cached credentials in `~/.netrc` / `~/_netrc`, because
  `_ensure_wandb_login` only accepted the env var or a sibling `wandb_api_key`
  file and never asked wandb to resolve its own store. The fallback now calls
  `wandb.login(relogin=False)` with a 30s prompt timeout, and the error text
  lists all three remedies (env var, `wandb login`, or the key file at its
  real path next to `rl_callbacks.py`) instead of the obsolete "app directory"
  wording left over from the `src/app` → `src/phoenx` move. Ten new tests in
  `tests/test_rl_callbacks.py` cover the short-circuit, both explicit sources,
  an empty key file falling through to the fallback, the netrc success path, and
  the raise / exception-chaining failure paths; the fast subset went from 424 to
  434 collected-and-selected tests.
- `.gitignore` covered only `src/app/wandb_api_key`, the pre-restructure location
  of the W&B key file. The path `rl_callbacks.py` actually reads,
  `src/phoenx/wandb_api_key`, is now ignored as well.
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
- The Isaac sparse-goal `distance_threshold` was effectively unusable by default: `IsaacSimWrapper` defaults it to `None` and forwards that `None` explicitly to `IsaacLabAdapter`, which never fell back to its own `0.05` default because an explicit keyword argument always wins. `HindsightRelabeler` construction then crashed inside `float(None)` while resolving the threshold, before its `compute_reward` was ever reached. `IsaacLabAdapter` now also defaults to `None`, `compute_reward` raises an actionable `ValueError` naming the `distance_threshold` argument (`env.config.distance_threshold` in a YAML config) when called with no threshold configured, and `HindsightRelabeler._resolve_distance_threshold` returns `None` instead of raising, so envs whose `distance_threshold` attribute is present but `None` now reach HER's existing reward-sign achievement check instead of crashing (envs with no such attribute at all always did). Note what this does *not* buy on the Isaac path: HER recomputes every relabeled reward through `compute_reward`, so an Isaac env with no threshold still cannot relabel — it now fails on the first relabeled episode with the actionable message instead of at relabeler construction with a `TypeError` from `float(None)`.
