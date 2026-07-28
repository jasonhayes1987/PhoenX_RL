"""Tests for ``src/phoenx/env_wrapper.py``.

This is the single pytest module that validates every public class in
``env_wrapper.py``. It is built to grow over time, so it must stay importable
and runnable on a CPU-only machine (no Isaac Sim): all Isaac work is lazy and
the GPU/Isaac integration tests auto-skip when ``isaaclab``/CUDA are missing.

Currently covered
-----------------
    * ``NextStepManagerBasedRLEnv`` - same-step -> NextStep autoreset conversion
    * ``IsaacLabAdapter``           - VectorEnv adapter + sparse HER reward
    * ``IsaacSimWrapper``           - Observation plumbing, action-space bounding,
                                      (de)serialization (env init mocked)
    * Isaac integration             - real Franka reach boot/reset/step, plus the
                                      real truncation -> phantom NextStep cycle
                                      (``@pytest.mark.isaac`` / ``slow``; opt-in)

Planned (slot new ``Test*`` classes into the marked sections below)
-------------------------------------------------------------------
    * ``GymnasiumWrapper`` / ``EnvPoolWrapper`` + ``EnvPoolAdapter``
    * ``EnvWrapper`` base: ``extract_states_goals``, ``config``/``to_json``/
      ``from_json``, ``clone``, ``get_base_env``, ``_find_nstep_wrapper``
    * ``VectorNStepReward`` and the wrapper registry
    * ``Observation`` / ``Action`` dataclasses

Why a fake base for the NextStep test
-------------------------------------
``NextStepManagerBasedRLEnv`` only exists after Isaac's Kit app is launched - it
is built by the real factory ``_get_next_step_env_cls`` on top of
``isaaclab.envs.ManagerBasedRLEnv``. To verify the *conversion logic*
deterministically without a GPU we inject a controllable fake base that
reproduces Isaac's verified same-step reset contract, then let the real factory
build the real subclass on top of it (see ``next_step_cls`` fixture).
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types

import numpy as np
import pytest
import torch as T

import gymnasium as gym

import phoenx.env_wrapper as ew
from phoenx.env_wrapper import IsaacLabAdapter, IsaacSimWrapper, Observation

DEVICE = "cpu"

T.manual_seed(0)
np.random.seed(0)


# =============================================================================
# Shared helpers
# =============================================================================
def _mask(n: int, idxs: list[int]) -> T.Tensor:
    m = T.zeros(n, dtype=T.bool, device=DEVICE)
    if idxs:
        m[idxs] = True
    return m


@contextlib.contextmanager
def _isaac_boot_safe_argv():
    """Hide pytest's CLI flags from Isaac while the Kit app boots.

    ``AppLauncher``/Omniverse Kit parse ``sys.argv`` on launch and *hard-crash*
    (access violation) on flags they don't recognise, e.g. ``-m``/``-q`` - an
    uncatchable C-level fault that would kill the whole pytest run. Reduce argv to
    just the program name during boot, then restore it.
    """
    saved = sys.argv
    sys.argv = saved[:1]
    try:
        yield
    finally:
        sys.argv = saved


# =============================================================================
# Fakes
# =============================================================================
class _FakeObsManager:
    """Minimal stand-in for Isaac's ``observation_manager``."""

    def __init__(self, env: "FakeManagerBasedRLEnv"):
        self._env = env

    def compute(self, update_history: bool = True) -> dict[str, T.Tensor]:
        # Returns the *current* obs: the terminal obs before ``_reset_idx`` runs,
        # the reset obs afterwards (exactly how the subclass relies on it).
        return {"policy": self._env._obs()}


class FakeManagerBasedRLEnv:
    """Controllable stand-in for ``isaaclab.envs.ManagerBasedRLEnv``.

    Reproduces Isaac's *same-step* autoreset contract:
        * ``step`` advances all envs, then resets done envs **in step** via
          ``_reset_idx`` and computes observations **after** the reset, so a done
          env's returned obs is the new-episode reset obs (terminal obs lost),
        * ``observation_manager.compute`` returns the current obs.

    Observations are ``[epoch, age]`` per env so a test can decode terminal
    (``age > 0``) vs reset (``age == 0``) states; reward equals the pre-reset
    ``age`` so the terminal reward is traceable; ``reset_count`` records how many
    times each env has been reset.
    """

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = T.device(device)
        self.observation_manager = _FakeObsManager(self)
        self._epoch = T.zeros(num_envs, dtype=T.long, device=self.device)
        self._age = T.zeros(num_envs, dtype=T.long, device=self.device)
        self.reset_count = T.zeros(num_envs, dtype=T.long, device=self.device)
        self._sched_term = T.zeros(num_envs, dtype=T.bool, device=self.device)
        self._sched_trunc = T.zeros(num_envs, dtype=T.bool, device=self.device)

    def _obs(self) -> T.Tensor:
        return T.stack([self._epoch.float(), self._age.float()], dim=-1)

    def schedule(self, terminations=None, truncations=None) -> None:
        """Set which envs terminate/truncate on the *next* ``step`` (consumed once)."""
        if terminations is not None:
            self._sched_term = T.as_tensor(terminations, dtype=T.bool, device=self.device)
        if truncations is not None:
            self._sched_trunc = T.as_tensor(truncations, dtype=T.bool, device=self.device)

    def _reset_idx(self, env_ids) -> None:
        if len(env_ids) > 0:
            self._epoch[env_ids] += 1
            self._age[env_ids] = 0
            self.reset_count[env_ids] += 1

    def reset(self, seed=None, options=None):
        self._epoch.zero_()
        self._age.zero_()
        self.reset_count.zero_()
        self._reset_idx(T.arange(self.num_envs, device=self.device))  # mimic full-reset path
        return self.observation_manager.compute(), {}

    def step(self, action):
        self._age += 1  # physics advances all envs
        rewards = self._age.float().clone()  # reward == terminal age (traceable)
        terminations = self._sched_term.clone()
        truncations = self._sched_trunc.clone()
        self._sched_term = T.zeros(self.num_envs, dtype=T.bool, device=self.device)
        self._sched_trunc = T.zeros(self.num_envs, dtype=T.bool, device=self.device)
        done_ids = (terminations | truncations).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)  # same-step in-step reset
        obs = self.observation_manager.compute()  # computed AFTER reset
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        pass


class _FakeInnerEnv:
    """Inner env for the ``IsaacLabAdapter`` (records forwarded calls)."""

    def __init__(self, num_envs: int = 3):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (num_envs, 2), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (num_envs, 2), np.float32)
        self.spec = "fake-spec"
        self.reset_seed = None
        self.step_action = None
        self.closed = False

    def reset(self, seed=None):
        self.reset_seed = seed
        return {"policy": T.zeros((self.num_envs, 2))}, {"reset": True}

    def step(self, action):
        self.step_action = action
        return (
            {"policy": T.ones((self.num_envs, 2))},
            T.ones(self.num_envs),
            T.zeros(self.num_envs, dtype=T.bool),
            T.zeros(self.num_envs, dtype=T.bool),
            {"step": True},
        )

    def render(self, **kwargs):
        return "frame"

    def close(self):
        self.closed = True


class _FakeApp:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeIsaacVecEnv:
    """Vectorized env returned by a mocked ``IsaacSimWrapper._initialize_env``."""

    def __init__(self, *, num_envs=2, obs_dim=3, act_dim=4, obs_key="policy",
                 goal_key=None, ach_goal_key=None, act_low=-1.0, act_high=1.0):
        self.num_envs = num_envs
        self._obs_key = obs_key
        self._goal_key = goal_key
        self._ach_goal_key = ach_goal_key
        self._obs_dim = obs_dim
        self.single_action_space = gym.spaces.Box(act_low, act_high, (act_dim,), np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.last_action = None

    def _make_obs(self, val: float) -> dict[str, T.Tensor]:
        obs = {self._obs_key: T.full((self.num_envs, self._obs_dim), val)}
        if self._goal_key:
            obs[self._goal_key] = T.full((self.num_envs, 2), val)
        if self._ach_goal_key:
            obs[self._ach_goal_key] = T.full((self.num_envs, 2), val)
        return obs

    def reset(self, seed=None):
        return self._make_obs(0.0), {}

    def step(self, action):
        self.last_action = action
        return (
            self._make_obs(1.0),
            T.ones(self.num_envs),
            T.zeros(self.num_envs, dtype=T.bool),
            T.zeros(self.num_envs, dtype=T.bool),
            {},
        )

    def close(self):
        pass


# =============================================================================
# NextStepManagerBasedRLEnv  (unit; real factory + real subclass, fake base)
# =============================================================================
@pytest.fixture
def next_step_cls():
    """Build the real ``NextStepManagerBasedRLEnv`` on top of a fake base.

    Injects a fake ``isaaclab.envs`` module so the real ``_get_next_step_env_cls``
    resolves ``FakeManagerBasedRLEnv`` as the base, then restores the module
    table and the factory cache on teardown.
    """
    keys = ("isaaclab", "isaaclab.envs")
    saved_modules = {k: sys.modules.get(k) for k in keys}
    saved_cache = ew._NEXT_STEP_ENV_CLS

    pkg = types.ModuleType("isaaclab")
    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.ManagerBasedRLEnv = FakeManagerBasedRLEnv
    pkg.envs = envs_mod
    sys.modules["isaaclab"] = pkg
    sys.modules["isaaclab.envs"] = envs_mod
    ew._NEXT_STEP_ENV_CLS = None
    try:
        yield ew._get_next_step_env_cls()
    finally:
        ew._NEXT_STEP_ENV_CLS = saved_cache
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _make_next_step_env(cls, num_envs: int = 4):
    env = cls(num_envs=num_envs, device=DEVICE)
    env.reset()
    return env


class TestNextStepManagerBasedRLEnv:
    def test_factory_builds_real_subclass_on_injected_base(self, next_step_cls):
        assert next_step_cls.__name__ == "NextStepManagerBasedRLEnv"
        assert next_step_cls.__bases__ == (FakeManagerBasedRLEnv,)

    def test_factory_caches_class(self, next_step_cls):
        assert ew._get_next_step_env_cls() is next_step_cls

    def test_reset_initializes_state(self, next_step_cls):
        env = _make_next_step_env(next_step_cls)
        assert env._phantom_mask.dtype == T.bool
        assert not env._phantom_mask.any()
        assert env._capture_terminal is True
        assert env._terminal_obs is None

    def test_normal_step_passthrough(self, next_step_cls):
        env = _make_next_step_env(next_step_cls)
        obs, _, term, trunc, _ = env.step(None)
        assert not term.any() and not trunc.any()
        assert (obs["policy"][:, 1] == 1).all()  # age 1 after one step
        assert not env._phantom_mask.any()
        assert (env.reset_count == env.reset_count[0]).all()  # no spurious resets

    @pytest.mark.parametrize("flag", ["terminations", "truncations"])
    def test_nextstep_lifecycle(self, next_step_cls, flag):
        """Full done -> phantom -> new-episode cycle for env 0."""
        env = _make_next_step_env(next_step_cls, num_envs=4)
        env.step(None)
        env.step(None)  # ages -> 2
        rc0 = env.reset_count.clone()
        env.schedule(**{flag: _mask(4, [0])})

        # --- done step: terminal obs returned, env reset in-step, flagged phantom
        obs, rewards, term, trunc, _ = env.step(None)  # age -> 3, env0 done
        done = term if flag == "terminations" else trunc
        assert done[0].item() and not done[1:].any()
        assert obs["policy"][0, 1].item() == 3.0  # TERMINAL age (not reset 0)
        assert rewards[0].item() == 3.0  # terminal reward preserved
        assert (env.reset_count[0] - rc0[0]).item() == 1
        assert env._phantom_mask[0].item()
        # other envs untouched on the done step
        assert (obs["policy"][1:, 1] == 3.0).all()
        assert not term[1:].any() and not trunc[1:].any()

        # --- phantom step: reset obs, not done, re-reset, mask cleared.
        # NOTE: production no longer zeroes the phantom step's reward inside the
        # env (the old `rewards[ids] = 0.0` is disabled); phantom rewards are
        # masked downstream instead (trainer `valid_steps = ~prev_done`, buffer
        # `first_steps`, and VectorNStepReward's `active = ~prev_done`). With
        # this fake env (reward == age) the passed-through value is 1.0.
        obs, rewards, term, trunc, _ = env.step(None)
        assert obs["policy"][0, 1].item() == 0.0  # RESET age
        assert rewards[0].item() == 1.0  # passes through; masked downstream
        assert not term[0].item() and not trunc[0].item()
        assert (env.reset_count[0] - rc0[0]).item() == 2  # done-reset + phantom re-reset
        assert not env._phantom_mask.any()
        assert (obs["policy"][1:, 1] == 4.0).all()  # other envs keep stepping

        # --- first real step of the new episode is preserved (age 1, not lost)
        obs, *_ = env.step(None)
        assert obs["policy"][0, 1].item() == 1.0
        # non-done envs were never reset across the whole cycle
        assert (env.reset_count[1:] == rc0[1:]).all()

    def test_simultaneous_phantom_and_new_done(self, next_step_cls):
        """A phantom env and a newly-done env in the same step stay independent."""
        env = _make_next_step_env(next_step_cls, num_envs=4)
        env.step(None)
        env.step(None)
        env.schedule(terminations=_mask(4, [0]))
        env.step(None)  # env0 done -> phantom next step
        assert env._phantom_mask[0].item()

        env.schedule(terminations=_mask(4, [1]))  # env1 newly terminates while env0 phantoms
        obs, _, term, _, _ = env.step(None)
        # env0 serviced as phantom: reset obs, not done
        assert obs["policy"][0, 1].item() == 0.0 and not term[0].item()
        # env1 keeps its terminal obs + done flag (capture not clobbered by env0)
        assert term[1].item() and obs["policy"][1, 1].item() > 0.0
        # only env1 carries forward; env0 is not re-flagged (no phantom loop)
        assert env._phantom_mask[1].item() and not env._phantom_mask[0].item()


# =============================================================================
# IsaacLabAdapter  (unit; fake inner env)
# =============================================================================
class TestIsaacLabAdapter:
    def test_init_copies_spaces(self):
        inner = _FakeInnerEnv(num_envs=3)
        adapter = IsaacLabAdapter(inner, distance_threshold=0.1)
        assert adapter.num_envs == 3
        assert adapter.single_observation_space is inner.single_observation_space
        assert adapter.single_action_space is inner.single_action_space
        assert adapter.observation_space is inner.observation_space
        assert adapter.action_space is inner.action_space
        assert adapter.distance_threshold == 0.1

    def test_reset_and_step_forwarded(self):
        inner = _FakeInnerEnv()
        adapter = IsaacLabAdapter(inner)
        _, info = adapter.reset(seed=11)
        assert inner.reset_seed == 11 and info == {"reset": True}
        action = T.full((inner.num_envs, 2), 0.5)
        out = adapter.step(action)
        assert T.equal(inner.step_action, action)
        assert len(out) == 5 and out[4] == {"step": True}
        assert out[1].shape == (inner.num_envs,)

    def test_compute_reward_sparse_batched(self):
        adapter = IsaacLabAdapter(_FakeInnerEnv(), distance_threshold=0.05)
        achieved = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        desired = np.array([[0.0, 0.02], [0.0, 0.0]], dtype=np.float32)
        reward = adapter.compute_reward(achieved, desired)
        assert np.allclose(reward, [0.0, -1.0])  # within / outside threshold
        assert reward.dtype == np.float32

    def test_spec_and_close_forwarded(self):
        inner = _FakeInnerEnv()
        adapter = IsaacLabAdapter(inner)
        assert adapter.spec == "fake-spec"
        adapter.close()
        assert inner.closed


# =============================================================================
# IsaacSimWrapper  (unit; _initialize_env mocked, no Isaac boot)
# =============================================================================
def _make_isaac_wrapper(monkeypatch, *, num_envs=2, obs_dim=3, act_dim=4, obs_key="policy",
                        goal_key=None, ach_goal_key=None, act_low=-1.0, act_high=1.0):
    fake = _FakeIsaacVecEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim, obs_key=obs_key,
                            goal_key=goal_key, ach_goal_key=ach_goal_key, act_low=act_low, act_high=act_high)

    def fake_initialize_env(self):
        self.app = _FakeApp()
        return fake

    monkeypatch.setattr(IsaacSimWrapper, "_initialize_env", fake_initialize_env)
    wrapper = IsaacSimWrapper(cfg="mod:Cls", num_envs=num_envs, obs_key=obs_key, goal_key=goal_key,
                              ach_goal_key=ach_goal_key, render_mode="headless", seed=7,
                              distance_threshold=0.05)
    return wrapper, fake


class TestIsaacSimWrapper:
    def test_unbounded_action_space_is_clipped(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch, act_low=-np.inf, act_high=np.inf, act_dim=4)
        space = wrapper.single_action_space
        assert np.all(space.low == -1.0) and np.all(space.high == 1.0)
        assert wrapper.action_space.shape == (2, 4)

    def test_bounded_action_space_preserved(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch, act_low=-2.0, act_high=2.0, act_dim=4)
        space = wrapper.single_action_space
        assert np.all(space.low == -2.0) and np.all(space.high == 2.0)

    def test_reset_returns_observation(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch, obs_dim=3)
        obs = wrapper.reset(seed=1)
        assert isinstance(obs, Observation)
        assert obs.states.shape == (2, 3)
        assert T.equal(obs.states, T.zeros((2, 3)))
        assert obs.goals is None and obs.ach_goals is None

    def test_step_returns_observation(self, monkeypatch):
        wrapper, fake = _make_isaac_wrapper(monkeypatch, obs_dim=3)
        action = T.zeros((2, 4))
        obs = wrapper.step(action)
        assert isinstance(obs, Observation)
        assert T.equal(obs.states, T.ones((2, 3)))
        assert T.equal(obs.rewards, T.ones(2))
        assert obs.terminations.shape == (2,) and not obs.terminations.any()
        assert obs.truncations.shape == (2,) and not obs.truncations.any()
        assert T.equal(fake.last_action, action)

    def test_goal_keys_extracted(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(
            monkeypatch, obs_key="policy", goal_key="desired_goal", ach_goal_key="achieved_goal"
        )
        obs = wrapper.reset()
        assert obs.goals is not None and obs.goals.shape == (2, 2)
        assert obs.ach_goals is not None and obs.ach_goals.shape == (2, 2)

    def test_format_actions(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch)
        out = wrapper.format_actions(np.zeros((2, 4), dtype=np.float32))
        assert isinstance(out, T.Tensor) and out.dtype == T.float32
        tensor = T.ones((2, 4))
        assert wrapper.format_actions(tensor) is tensor

    def test_space_properties_forwarded(self, monkeypatch):
        wrapper, fake = _make_isaac_wrapper(monkeypatch, act_low=-2.0, act_high=2.0)
        assert wrapper.observation_space is fake.observation_space
        assert wrapper.action_space is fake.action_space
        assert wrapper.single_action_space is fake.single_action_space
        assert wrapper.single_observation_space is fake.single_observation_space

    def test_config_and_json_roundtrip(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch)
        cfg = wrapper.config
        assert cfg["type"] == "isaacsim"
        assert cfg["config"]["cfg"] == "mod:Cls"
        assert cfg["config"]["num_envs"] == 2
        assert cfg["config"]["obs_key"] == "policy"
        assert cfg["config"]["distance_threshold"] == 0.05

        clone = IsaacSimWrapper.from_json(wrapper.to_json())
        assert clone.env_id == wrapper.env_id
        assert clone.num_envs == wrapper.num_envs
        assert clone.obs_key == wrapper.obs_key
        assert clone.distance_threshold == wrapper.distance_threshold

    def test_close_closes_env_and_app(self, monkeypatch):
        wrapper, _ = _make_isaac_wrapper(monkeypatch)
        wrapper.close()
        assert wrapper.app.closed


# =============================================================================
# IsaacSim integration  (real Isaac Sim + CUDA; auto-skipped otherwise)
# =============================================================================
# NOTE: gate only on ``isaaclab`` (+ CUDA). ``isaaclab_tasks`` (which provides
# FrankaReachEnvCfg) is only put on ``sys.path`` by the extension system once the
# Kit app boots, so it is undetectable here; a genuinely missing tasks package is
# caught as ``ModuleNotFoundError`` in the fixture and skipped there instead.
_ISAAC_AVAILABLE = (importlib.util.find_spec("isaaclab") is not None) and T.cuda.is_available()


@pytest.mark.isaac
@pytest.mark.skipif(not _ISAAC_AVAILABLE, reason="Isaac Sim / isaaclab + CUDA not available")
class TestIsaacSimIntegration:
    FRANKA_CFG = (
        "isaaclab_tasks.manager_based.manipulation.reach.config.franka."
        "joint_pos_env_cfg:FrankaReachEnvCfg"
    )

    @pytest.fixture(scope="class")
    def env(self, request):
        # Two Isaac-under-pytest hazards, both handled only while the app boots:
        #   1. Kit parses sys.argv and hard-crashes on pytest flags -> clean argv.
        #   2. Kit grabs the console handles, which pytest capture invalidates
        #      (WinError 6) -> suspend capture (a no-op under ``-s``).
        # This lets a plain ``pytest -m isaac`` launch Isaac without extra flags.
        def capture_disabled():
            cap = request.config.pluginmanager.getplugin("capturemanager")
            return cap.global_and_fixture_disabled() if cap else contextlib.nullcontext()

        try:
            with capture_disabled(), _isaac_boot_safe_argv():
                wrapper = IsaacSimWrapper(
                    cfg=self.FRANKA_CFG, num_envs=2, obs_key="policy",
                    render_mode="headless", seed=42, distance_threshold=0.05,
                )
        except (ModuleNotFoundError, ImportError, FileNotFoundError,
                ConnectionError, TimeoutError) as exc:
            # Genuinely environmental (deps/assets/network). Anything else - e.g. a
            # real wiring regression - propagates as a failure instead of hiding.
            pytest.skip(f"Isaac deps/assets unavailable, cannot launch env: {exc}")
        yield wrapper
        # Deliberately do NOT call wrapper.close() -> app.close(): Isaac's
        # SimulationApp.close() hard-exits the process (os._exit) during Kit
        # shutdown, which would kill pytest before it prints its summary, writes
        # reports, or sets the real exit code (a failing test could then report
        # exit 0). The app is a process-wide singleton torn down automatically at
        # interpreter exit; we only release the sim env here.
        with capture_disabled(), contextlib.suppress(Exception):
            wrapper.env.close()

    def test_env_wiring(self, env):
        assert isinstance(env.env, IsaacLabAdapter)
        base = env.env._env
        assert type(base).__name__ == "NextStepManagerBasedRLEnv"
        assert hasattr(base, "_phantom_mask")
        assert base._phantom_mask.shape == (2,)

    def test_action_space_finite(self, env):
        space = env.single_action_space
        assert np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))

    def test_reset_observation(self, env):
        obs = env.reset(seed=42)
        assert isinstance(obs, Observation)
        assert obs.states.ndim == 2 and obs.states.shape[0] == 2
        assert T.isfinite(obs.states).all()

    def test_step_shapes_and_finite(self, env):
        env.reset(seed=42)
        for _ in range(3):
            action = T.as_tensor(env.action_space.sample(), dtype=T.float32, device="cuda")
            obs = env.step(action)
            assert isinstance(obs, Observation)
            assert obs.states.shape[0] == 2
            assert obs.terminations.shape[0] == 2
            assert obs.truncations.shape[0] == 2
            assert T.isfinite(obs.states).all()
            assert T.isfinite(obs.rewards).all()

    @pytest.mark.slow
    def test_truncation_then_phantom(self, env):
        """End-to-end NextStep check on the real env.

        Franka reach only ever ends via ``time_out`` (truncation), so the
        ``NextStepManagerBasedRLEnv`` conversion must (1) return the *terminal*
        observation on the truncation step, then (2) emit a clean phantom on the
        very next step - no done flags, fresh reset obs - which the trainer
        masks out downstream (phantom rewards pass through the env unzeroed;
        see the matching note in TestNextStepManagerBasedRLEnv). This is the
        one path the CPU fake base can't prove.
        """
        env.reset(seed=42)
        act_dim = int(np.prod(env.single_action_space.shape))
        zero = T.zeros((env.num_envs, act_dim), dtype=T.float32, device="cuda")

        terminal, done = None, None
        for _ in range(2000):
            obs = env.step(zero)
            done = (obs.terminations | obs.truncations)
            if bool(done.any()):
                done = done.clone()
                terminal = obs.states.clone()  # spliced terminal obs (pre-reset)
                assert T.isfinite(obs.rewards).all()
                break
        assert terminal is not None, "env never finished an episode within the step budget"

        phantom = env.step(zero)  # the masked phantom step for the done envs
        assert not phantom.terminations[done].any()
        assert not phantom.truncations[done].any()
        # Phantom rewards pass through finite (masked downstream, not zeroed
        # in the env - same contract as the CPU fake-base test).
        assert T.isfinite(phantom.rewards[done]).all()
        # terminal obs (pre-reset) must differ from the phantom reset obs
        assert not T.allclose(terminal[done], phantom.states[done])


# =============================================================================
# GymnasiumWrapper                              (planned - add Test* here)
# =============================================================================

# =============================================================================
# EnvPoolWrapper / EnvPoolAdapter               (planned)
# =============================================================================

# =============================================================================
# EnvWrapper base + serialization + helpers     (planned)
# =============================================================================

# =============================================================================
# VectorNStepReward + wrapper registry          (planned)
# =============================================================================

# =============================================================================
# Observation / Action dataclasses              (planned)
# =============================================================================
