"""Which checkpoint counts as the best: ``TrainingSchedule.save_best_by`` and
``save_best_level_key``.

By default the best is the rolling average reward. ``success_rate`` picks the
best by the share of the last 100 episodes the ``SuccessCriterion`` counted
as a success, and a level key makes a higher curriculum level always win.
The reasons: reward from different curriculum levels cannot be compared, and
a long run whose average peaked early kept only that early policy.

The trainer is driven through ``step`` with a scripted one-env stand-in whose
infos carry ``is_success`` and an Isaac Lab style ``log`` with the level.
"""

from __future__ import annotations

from collections import deque

import pytest
import torch as T

from phoenx.config_schema import cross_field_problems
from phoenx.env_wrapper import Action, Observation
from phoenx.trainer import SuccessCriterion, Trainer, TrainingSchedule

LEVEL_KEY = "Curriculum/robustness/level"


class _Agent:
    device = "cpu"
    intrinsic_motivation = None

    def act(self, states, *_args, **_kwargs):
        return Action(actions=T.zeros(states.shape[0], 1))

    def get_config(self):
        return {"type": "FakeAgent"}

    def save_state(self, path):
        pass


class _Env:
    """One env; every step ends an episode with the scripted reward, success and level."""

    num_envs = 1
    goal_key = None
    config = {"type": "FakeEnv"}

    def __init__(self):
        self.reward = 0.0
        self.success = False
        self.level = 0.0

    def _find_nstep_wrapper(self):
        return None

    def step(self, _actions):
        return Observation(
            states=T.zeros(1, 4),
            rewards=T.tensor([self.reward]),
            terminations=T.tensor([True]),
            truncations=T.zeros(1, dtype=T.bool),
            # A new dict each step, as Isaac Lab builds one on each reset.
            infos={"is_success": T.tensor([self.success]), "log": {LEVEL_KEY: self.level}},
        )

    def close(self):
        pass


class _Buffer:
    def record(self, *args, **kwargs):
        pass

    def get_config(self):
        return None


def _trainer(tmp_path, **schedule) -> tuple[Trainer, _Env]:
    env = _Env()
    trainer = Trainer(
        agent=_Agent(),
        env=env,
        schedule=TrainingSchedule(save_every=0, **schedule),
        success_criterion=SuccessCriterion(metric="info_flag"),
        buffer=_Buffer(),
        save_dir=str(tmp_path),
    )
    trainer._step = 0
    trainer._prev_obs = Observation(states=T.zeros(1, 4))
    trainer._prev_done = T.zeros(1, dtype=T.bool)
    trainer._best_reward = float("-inf")
    trainer._episode_steps = T.zeros(1, dtype=T.int32)
    trainer._completed_episodes = T.zeros(1, dtype=T.int32)
    trainer._episode_scores = T.zeros(1, dtype=T.float32)
    trainer._score_history = deque(maxlen=100)
    trainer._success_tracker = deque(maxlen=100)
    trainer._last_save = None
    trainer._best_pending = False
    return trainer, env


def _play(trainer, env, n, *, reward=0.0, success=False, level=None):
    """Play ``n`` one-step episodes; return how many steps were marked best."""
    env.reward, env.success = reward, success
    if level is not None:
        env.level = level
    best = 0
    for _ in range(n):
        # Each step is a whole episode: count its reward (step() skips the
        # step after a done, which is the reset step in a real env).
        trainer._prev_done = T.zeros(1, dtype=T.bool)
        result = trainer.step(training=True)
        trainer._step += 1
        trainer._episode_scores.zero_()
        best += bool(result["episode_logs"][-1].get("best"))
    return best


# -- schedule and config -------------------------------------------------------


def test_schedule_defaults_and_config_round_trip():
    schedule = TrainingSchedule()
    assert schedule.save_best_by == "avg_reward"
    assert schedule.save_best_level_key is None
    config = TrainingSchedule(save_best_by="success_rate", save_best_level_key=LEVEL_KEY).get_config()
    assert config["save_best_by"] == "success_rate"
    assert TrainingSchedule(**config).save_best_level_key == LEVEL_KEY


def test_schedule_rejects_unknown_choice_and_empty_key():
    with pytest.raises(ValueError, match="save_best_by"):
        TrainingSchedule(save_best_by="landing_rate")
    with pytest.raises(ValueError, match="save_best_level_key"):
        TrainingSchedule(save_best_level_key="  ")


def test_success_rate_needs_a_success_criterion(tmp_path):
    with pytest.raises(ValueError, match="success_criterion"):
        Trainer(agent=_Agent(), env=_Env(), schedule=TrainingSchedule(save_best_by="success_rate"),
                save_dir=str(tmp_path))


def test_config_check_flags_success_rate_without_criterion():
    config = {"schedule": {"save_best_by": "success_rate"}}
    assert [p.split(":")[0] for p in cross_field_problems(config)] == ["schedule.save_best_by"]
    config["success_criterion"] = {"metric": "info_flag"}
    assert cross_field_problems(config) == []
    assert cross_field_problems({"schedule": {}}) == []


# -- success rate --------------------------------------------------------------


def test_success_rate_waits_for_a_full_window(tmp_path):
    """One lucky landing is not a 100% best: nothing counts before 100 episodes."""
    trainer, env = _trainer(tmp_path, save_best_by="success_rate")
    assert _play(trainer, env, 99, success=True) == 0
    assert _play(trainer, env, 1, success=True) == 1
    assert trainer._best_key == (0.0, 1.0, 0.0)


def test_success_rate_beats_reward(tmp_path):
    """A higher landing rate wins even when its average reward is lower."""
    trainer, env = _trainer(tmp_path, save_best_by="success_rate")
    _play(trainer, env, 100, reward=50.0, success=False)
    first = trainer._best_key
    assert first[1] == 0.0
    # Success climbs while reward drops: every step is a new best.
    assert _play(trainer, env, 10, reward=-10.0, success=True) == 10
    assert trainer._best_key[1] == pytest.approx(0.10)
    assert trainer._best_reward < 50.0
    # Same rate (the oldest episodes were failures too), less reward: no new best.
    assert _play(trainer, env, 5, reward=-1000.0, success=False) == 0


def test_success_rate_tie_goes_to_higher_reward(tmp_path):
    trainer, env = _trainer(tmp_path, save_best_by="success_rate")
    _play(trainer, env, 100, reward=1.0, success=True)
    assert _play(trainer, env, 1, reward=1.0, success=True) == 0  # same rate, same reward
    assert _play(trainer, env, 1, reward=200.0, success=True) == 1  # same rate, more reward
    assert trainer._best_key[1] == 1.0


# -- curriculum level ------------------------------------------------------------


def test_higher_level_beats_higher_success(tmp_path):
    trainer, env = _trainer(tmp_path, save_best_by="success_rate", save_best_level_key=LEVEL_KEY)
    _play(trainer, env, 101, success=True, level=0.5)
    assert trainer._best_key[:2] == (0.5, 1.0)
    # Promoted (this step's resets move the level; it was played at 0.5).
    _play(trainer, env, 1, success=True, level=0.6)
    # 40% at 0.6 beats 100% at 0.5, once the window is all 0.6 episodes.
    best = 0
    for n in range(100):
        best += _play(trainer, env, 1, success=n % 5 < 2)
    assert best == 1
    assert trainer._best_key[0] == 0.6
    assert trainer._best_key[1] == pytest.approx(0.4)


def test_level_change_waits_for_episodes_at_the_new_level(tmp_path):
    trainer, env = _trainer(tmp_path, save_best_by="success_rate", save_best_level_key=LEVEL_KEY)
    _play(trainer, env, 101, success=False, level=0.0)
    assert trainer._best_key[0] == 0.0
    # The step whose resets move the level was played at the old level.
    assert _play(trainer, env, 1, success=False, level=0.1) == 0
    assert trainer._level == 0.1 and trainer._episodes_at_level == 0
    assert _play(trainer, env, 99, success=True) == 0
    assert _play(trainer, env, 1, success=True) == 1
    assert trainer._best_key[:2] == (0.1, 1.0)


def test_level_key_with_mlflow_prefix(tmp_path):
    """``env/Curriculum/...``, as MLflow shows the key, is read the same."""
    trainer, env = _trainer(tmp_path, save_best_level_key="env/" + LEVEL_KEY)
    _play(trainer, env, 2, level=0.3)
    assert trainer._level == 0.3


def test_level_key_with_avg_reward(tmp_path):
    """A level key also works with the average reward as the metric."""
    trainer, env = _trainer(tmp_path, save_best_level_key=LEVEL_KEY)
    _play(trainer, env, 101, reward=100.0, level=0.0)
    _play(trainer, env, 1, reward=-50.0, level=0.2)
    assert _play(trainer, env, 100, reward=-50.0) >= 1
    assert trainer._best_key[0] == 0.2
    assert trainer._best_reward == pytest.approx(-50.0)


# -- default unchanged, log line, resume -------------------------------------------


def test_default_rule_is_the_rolling_average(tmp_path):
    """The default saves on the very first episode, as before, and keeps no key."""
    trainer, env = _trainer(tmp_path)
    assert _play(trainer, env, 1, reward=3.0) == 1
    assert _play(trainer, env, 1, reward=1.0) == 0
    assert trainer._best_key is None
    assert trainer._best_text() == "avg_reward=3.00"


def test_checkpoint_line_names_success_and_level(tmp_path):
    trainer, env = _trainer(tmp_path, save_best_by="success_rate", save_best_level_key=LEVEL_KEY)
    _play(trainer, env, 101, reward=2.0, success=True, level=0.6)
    assert trainer._best_text() == "success_rate=1.000, level=0.6, avg_reward=2.00"


def test_resume_keeps_best_only_under_the_same_rule(tmp_path):
    trainer, env = _trainer(tmp_path, save_best_by="success_rate", save_best_level_key=LEVEL_KEY)
    _play(trainer, env, 101, success=True, level=0.6)
    state = trainer._trainer_state()

    same, _ = _trainer(tmp_path, save_best_by="success_rate", save_best_level_key=LEVEL_KEY)
    same._resume_state = dict(state)
    same._apply_resume_state("train")
    assert same._best_key == trainer._best_key
    assert same._level == 0.6

    other, _ = _trainer(tmp_path, save_best_by="success_rate")
    other._resume_state = dict(state)
    other._apply_resume_state("train")
    assert other._best_key is None
