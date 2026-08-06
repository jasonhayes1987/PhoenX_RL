"""Unit tests for ``Trainer.step``'s "best episode" bookkeeping in
``src/phoenx/trainer.py``.

Pins the one-line fix that sets ``episode_logs[-1]['best'] = True`` whenever a
completed training episode pushes the running average reward above the
previous best.

``Trainer.step`` is driven directly through minimal stand-in ``agent``/``env``/
``buffer`` objects that script a fixed reward/termination per call, rather
than a real Gymnasium env or agent — fast, deterministic, no I/O.
"""

from __future__ import annotations

from collections import deque

import torch as T

from phoenx.env_wrapper import Action, Observation
from phoenx.trainer import Trainer, TrainingSchedule


class _FakeAgent:
    """Minimal agent stand-in: no normalizers, no intrinsic motivation, and an
    ``act`` that always returns a fixed action regardless of the observation.
    """

    def __init__(self):
        self.device = "cpu"
        self.intrinsic_motivation = None

    def act(self, states, *_args, **_kwargs):
        n = states.shape[0]
        return Action(actions=T.zeros(n, 1))


class _FakeEnv:
    """Single-env stand-in whose ``step`` always returns a scripted reward and
    termination, independent of the action it receives.
    """

    def __init__(self, reward: float, done: bool = True):
        self.num_envs = 1
        self.goal_key = None
        self._reward = reward
        self._done = done

    def _find_nstep_wrapper(self):
        return None

    def step(self, _actions):
        return Observation(
            states=T.zeros(1, 4),
            rewards=T.tensor([self._reward]),
            terminations=T.tensor([self._done]),
            truncations=T.zeros(1, dtype=T.bool),
        )


class _FakeBuffer:
    def record(self, *args, **kwargs):
        pass


def _make_trainer(reward: float, best_reward: float, tmp_path, score_history=None) -> Trainer:
    """Build a real ``Trainer`` wired to the fakes above, with the internal
    episode-tracking state ``_initialize_run`` would normally set pre-seeded
    by hand so the test can drive ``step`` without booting a real env/agent.
    """
    trainer = Trainer(
        agent=_FakeAgent(),
        env=_FakeEnv(reward=reward),
        schedule=TrainingSchedule(),
        success_criterion=None,
        buffer=_FakeBuffer(),
        renderer=None,
        callbacks=None,
        save_dir=str(tmp_path),
    )
    trainer._step = 0
    trainer._prev_obs = Observation(states=T.zeros(1, 4))
    trainer._prev_done = T.zeros(1, dtype=T.bool)
    trainer._best_reward = best_reward
    trainer._episode_steps = T.zeros(1, dtype=T.int32)
    trainer._completed_episodes = T.zeros(1, dtype=T.int32)
    trainer._episode_scores = T.zeros(1, dtype=T.float32)
    trainer._score_history = deque(score_history or [], maxlen=100)
    return trainer


def test_step_marks_new_best_episode_and_saves(tmp_path, monkeypatch):
    """A completed training episode whose avg reward beats the running best
    carries `best: True` on its episode log and triggers `Trainer.save()`."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert episode_log.get("best") is True
    assert saved == [True]
    assert trainer._best_reward == 5.0


def test_step_omits_best_key_when_average_not_improved(tmp_path, monkeypatch):
    """A completed episode that does NOT beat the running best omits the
    `best` key entirely (absence, not `False`) and never calls `save()`."""
    trainer = _make_trainer(reward=1.0, best_reward=100.0, tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert "best" not in episode_log
    assert saved == []
    assert trainer._best_reward == 100.0


def test_step_testing_mode_never_marks_best_or_saves(tmp_path, monkeypatch):
    """`training=False` suppresses both the `best` flag and the save, even
    though the average would otherwise beat the (still `-inf`) best."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=False)

    episode_log = result["episode_logs"][-1]
    assert "best" not in episode_log
    assert saved == []
    assert trainer._best_reward == float("-inf")


def test_step_no_completed_episode_leaves_logs_and_best_untouched(tmp_path, monkeypatch):
    """When no env reaches a terminal/truncated state, `episode_logs` stays
    empty and neither the best-tracking nor `save()` fires."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    trainer.env._done = False  # scripted env: no termination this step
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    assert result["episode_logs"] == []
    assert saved == []
    assert trainer._best_reward == float("-inf")
