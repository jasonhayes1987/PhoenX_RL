"""Shared pytest fixtures/helpers for the PhoenX test suite.

Provides:
    * ``src/`` on ``sys.path`` so ``import app.X`` works from any CWD.
    * Registration of synthetic Gymnasium test envs used across test files:
        - ``PhoenXMultiModal-v0``          Dict obs (uint8 image + float vector),
                                           continuous Box(2,) actions.
        - ``PhoenXMultiModalDiscrete-v0``  Same obs, Discrete(3) actions.
        - ``PhoenXGoal-v0``                Goal-conditioned Dict obs
                                           (observation / desired_goal /
                                           achieved_goal), Box(2,) actions.
        - ``PhoenXMemory-v0``              Observe-then-recall memory task where
                                           a feedforward policy cannot exceed
                                           chance (used by recurrent smoke tests).
    * ``force_cpu`` fixture — patches ``app.torch_utils.get_device`` to CPU for
      the duration of a test (same mechanism the existing suites use inline).

These are *real* Gymnasium environments (not mocks) so wrappers, vectorization
and spaces behave exactly as in training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# =============================================================================
# Synthetic environments
# =============================================================================

class MultiModalTestEnv(gym.Env):
    """Deterministic-ish multi-modal env: uint8 image + float vector obs.

    The vector observation's first element encodes the episode step so tests
    can verify temporal ordering; the image encodes the step in its mean
    brightness so image plumbing is verifiable too.
    """

    metadata = {"render_modes": []}

    def __init__(self, img_size: int = 16, vec_dim: int = 7,
                 episode_len: int = 20, discrete: bool = False,
                 render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
            "vec": spaces.Box(-np.inf, np.inf, (vec_dim,), np.float32),
        })
        if discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._episode_len = episode_len
        self._t = 0

    def _obs(self):
        img_space = self.observation_space["rgb"]
        vec_space = self.observation_space["vec"]
        base = (self._t * 10) % 255
        rgb = np.full(img_space.shape, base, dtype=np.uint8)
        vec = self.np_random.normal(size=vec_space.shape).astype(np.float32)
        vec[0] = float(self._t)
        return {"rgb": rgb, "vec": vec}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        # Strongly action-dependent reward (maximized at action = +1 on every
        # dim) so learning smoke tests have a clear gradient to follow.
        if isinstance(self.action_space, spaces.Discrete):
            reward = float(action) * 0.1
        else:
            reward = float(np.sum(np.asarray(action, dtype=np.float64)))
        terminated = False
        truncated = self._t >= self._episode_len
        return self._obs(), reward, terminated, truncated, {}


class GoalTestEnv(gym.Env):
    """Tiny goal-conditioned env with the gymnasium-robotics Dict layout."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 6, goal_dim: int = 3, episode_len: int = 15,
                 render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, (goal_dim,), np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (goal_dim,), np.float64),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._episode_len = episode_len
        self._t = 0
        self._goal = np.zeros(goal_dim)
        self._pos = np.zeros(goal_dim)

    def _obs(self):
        obs = self.np_random.normal(size=self.observation_space["observation"].shape)
        return {
            "observation": obs,
            "desired_goal": self._goal.copy(),
            "achieved_goal": self._pos.copy(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(d > 0.05).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._goal = self.np_random.uniform(-1, 1, size=self._goal.shape)
        self._pos = self.np_random.uniform(-1, 1, size=self._pos.shape)
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        self._pos = self._pos + 0.1 * np.resize(np.asarray(action, dtype=np.float64), self._pos.shape)
        reward = float(self.compute_reward(self._pos, self._goal))
        terminated = bool(np.linalg.norm(self._pos - self._goal) <= 0.05)
        truncated = self._t >= self._episode_len
        return self._obs(), reward, terminated, truncated, {}


class MemoryTestEnv(gym.Env):
    """Observe-then-recall task: the cue is only visible on step 0.

    Episode: at t=0 the agent observes a cue in {-1, +1}; on later steps the
    cue channel is zeroed. At the final step the agent must output the action
    matching the cue to receive +1 reward (0 otherwise). A memoryless policy
    can only achieve 0.5 average success; a recurrent policy can achieve 1.0.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 4, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.action_space = spaces.Discrete(2)
        self._episode_len = episode_len
        self._t = 0
        self._cue = 1

    def _obs(self):
        cue = float(self._cue) if self._t == 0 else 0.0
        # channel 1: normalized time; channel 2: constant bias
        return np.array([cue, self._t / self._episode_len, 1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._cue = 1 if self.np_random.random() < 0.5 else -1
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._episode_len
        reward = 0.0
        if terminated:
            wanted = 1 if self._cue == 1 else 0
            reward = 1.0 if int(action) == wanted else 0.0
        return self._obs(), reward, terminated, False, {}


def _register(env_id: str, entry_point, **kwargs) -> None:
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=entry_point, kwargs=kwargs)


_register("PhoenXMultiModal-v0", MultiModalTestEnv)
_register("PhoenXMultiModalDiscrete-v0", MultiModalTestEnv, discrete=True)
_register("PhoenXGoal-v0", GoalTestEnv)
_register("PhoenXMemory-v0", MemoryTestEnv)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def force_cpu(monkeypatch):
    """Patch ``app.torch_utils.get_device`` (and the re-exported references in
    modules that imported it) to always return CPU for this test."""
    import torch as T
    from app import torch_utils as tu

    def _cpu(device_spec=None):
        return T.device("cpu")

    monkeypatch.setattr(tu, "get_device", _cpu)
    # Modules that did ``from .torch_utils import get_device`` hold their own
    # reference; patch the ones the model/env stack uses.
    import app.env_wrapper as ew
    import app.models as models
    import app.buffer as buffer_mod
    import app.normalizer as norm_mod
    monkeypatch.setattr(ew, "get_device", _cpu)
    monkeypatch.setattr(models, "get_device", _cpu)
    monkeypatch.setattr(buffer_mod, "get_device", _cpu)
    monkeypatch.setattr(norm_mod, "get_device", _cpu)
    yield
