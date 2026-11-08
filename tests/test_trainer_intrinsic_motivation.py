"""Trainer-level regression tests for the intrinsic-motivation wiring in
``Trainer.step`` and ``VectorNStepReward.step``.

Pins the fix for two silent bugs (found Oct 21, 2026; fixed in ``fba84dc``):

1. **Double rollout call.** In off-policy runs ``IntrinsicMotivation.compute_rollout_reward``
   ran twice per env step: once inside ``VectorNStepReward.step`` on raw states
   (that bonus went to the replay buffer) and again in ``Trainer.step`` on
   normalized states (that bonus went to ``Observation.intrinsic_rewards``).
   ``EpisodicNovelty`` appended both to memory, so memory grew by 2 per step,
   mixed raw- and normalized-space embeddings, and the logged bonus never
   matched the stored one.
2. **Memory never cleared.** Nothing in the Trainer called ``on_episode_end``,
   so episodic memory carried over across episodes (on- and off-policy).

Neither bug raised or logged anything; training ran and simply learned from a
wrong bonus. The unit tests in ``test_intrinsic_motivation.py`` exercise
``EpisodicNovelty`` in isolation and could not see either one, because the
contract lives *between* ``Trainer.step``, ``Trainer._initialize_run`` and
``VectorNStepReward``. These tests drive a real ``Trainer`` (through the real
``_initialize_run``) over a real CartPole vector env, with and without the
real ``VectorNStepReward`` wrapper, and replace only the agent, its state
normalizer, the IM module and the buffer with recording stand-ins.

Invariants pinned:

* exactly one ``compute_rollout_reward`` call per *training* step, made by the
  wrapper when one owns the IM module and by the Trainer otherwise;
* the IM sees ``state_normalizer.normalize(raw)`` for both states and
  next states;
* the bonus on ``Observation.intrinsic_rewards`` is the bonus from that one
  call, and (off-policy) the same values reach the n-step window the replay
  buffer stores;
* evaluation steps never call the IM, and training resumes calling it after;
* ``on_episode_end`` is called on exactly the steps where envs finish, with
  exactly those env indices, after that step's bonus has been computed, and
  never during evaluation.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch as T

from phoenx.env_wrapper import Action, GymnasiumWrapper
from phoenx.trainer import Trainer, TrainingSchedule

NUM_ENVS = 3


# =============================================================================
# Recording stand-ins
# =============================================================================

class _RecordingIM:
    """Stand-in IM module that records every call and returns bonuses that are
    unique per call and per env, so any mix-up between calls is detectable.
    """

    def __init__(self):
        self.device = T.device("cpu")
        self.reward_calls: list[dict] = []
        self.episode_end_calls: list[list[int]] = []
        # Ordered log of "reward" / "end" events across both methods.
        self.events: list[str] = []

    def compute_rollout_reward(self, states, next_states, actions=None, env_indices=None):
        caller = os.path.basename(sys._getframe(1).f_code.co_filename)
        k = len(self.reward_calls)
        n = next_states.shape[0]
        bonus = 1000.0 * (k + 1) + T.arange(n, dtype=T.float32)
        self.reward_calls.append({
            "caller": caller,
            "states": states.detach().clone(),
            "next_states": next_states.detach().clone(),
            "bonus": bonus.clone(),
        })
        self.events.append("reward")
        return bonus

    def on_episode_end(self, env_indices):
        self.episode_end_calls.append(sorted(int(i) for i in env_indices.flatten().tolist()))
        self.events.append("end")


class _AffineNormalizer:
    """State normalizer stand-in. ``normalize`` is a fixed affine map that is
    deliberately *not* the identity, so a raw input is distinguishable from a
    normalized one. Its stats never change, so the expected value is exact.
    """

    device = T.device("cpu")

    def normalize(self, x):
        return (x - 1.0) * 2.0

    def add(self, *_args, **_kwargs):
        pass

    def update(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class _StubAgent:
    """Agent stand-in: always pushes cart left (CartPole episodes end within
    ~10 steps), owns the recording IM and the affine state normalizer.
    """

    def __init__(self):
        self.device = T.device("cpu")
        self.intrinsic_motivation = _RecordingIM()
        self.state_normalizer = _AffineNormalizer()

    def act(self, states, *_args, **_kwargs):
        return Action(actions=T.zeros(states.shape[0], dtype=T.long))


class _RecordingBuffer:
    """Buffer stand-in that keeps every ``record`` call's arguments."""

    def __init__(self):
        self.records: list[dict] = []

    def record(self, observation, prev_observation=None, actions=None, prev_dones=None):
        self.records.append({
            "observation": observation,
            "prev_observation": prev_observation,
            "prev_dones": prev_dones.clone() if prev_dones is not None else None,
        })


def _make_trainer(tmp_path, with_nstep: bool):
    """Real ``Trainer`` over a real CartPole vector env, initialized through
    the real ``_initialize_run("train")`` so the IM/normalizer hand-off to the
    n-step wrapper is exercised too.
    """
    wrappers = ([{"type": "VectorNStepReward", "params": {"n": 1}}]
                if with_nstep else None)
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=NUM_ENVS, seed=0, wrappers=wrappers)
    agent = _StubAgent()
    buffer = _RecordingBuffer()
    trainer = Trainer(
        agent=agent,
        env=env,
        schedule=TrainingSchedule(seed=0),
        buffer=buffer,
        save_dir=str(tmp_path),
    )
    trainer._initialize_run("train")
    # Keep the test about IM wiring: never write checkpoints.
    trainer.save = lambda *a, **k: None
    return trainer, agent.intrinsic_motivation, agent.state_normalizer, buffer


def _dones(observation) -> T.Tensor:
    return T.logical_or(T.as_tensor(observation.terminations),
                        T.as_tensor(observation.truncations)).cpu()


_PATHS = pytest.mark.parametrize(
    "with_nstep", [True, False], ids=["off_policy_nstep_wrapper", "on_policy_no_wrapper"]
)


# =============================================================================
# Bug 1: one IM call per training step, on normalized states
# =============================================================================

@_PATHS
def test_intrinsic_reward_computed_once_per_training_step_on_normalized_states(
        tmp_path, force_cpu, with_nstep):
    trainer, im, norm, buffer = _make_trainer(tmp_path, with_nstep)
    expected_caller = "env_wrapper.py" if with_nstep else "trainer.py"

    def train_steps(n_steps):
        for _ in range(n_steps):
            calls_before = len(im.reward_calls)
            prev_done = trainer._prev_done.clone().cpu()
            trainer.step(training=True)

            assert len(im.reward_calls) == calls_before + 1, (
                "compute_rollout_reward must run exactly once per training step")
            call = im.reward_calls[-1]
            assert call["caller"] == expected_caller

            rec = buffer.records[-1]
            obs, prev_obs = rec["observation"], rec["prev_observation"]

            # The IM saw normalized states, not raw ones.
            T.testing.assert_close(call["next_states"].cpu(),
                                   norm.normalize(T.as_tensor(obs.states).float().cpu()))
            T.testing.assert_close(call["states"].cpu(),
                                   norm.normalize(T.as_tensor(prev_obs.states).float().cpu()))

            # The logged bonus is the bonus from that single call.
            T.testing.assert_close(T.as_tensor(obs.intrinsic_rewards).float().cpu().view(-1),
                                   call["bonus"])

            # Off-policy: the same bonus reaches the n-step window the replay
            # buffer stores (n=1, envs whose previous step was done are skipped).
            if with_nstep:
                if obs.n_step_trajectory is None:
                    assert bool(prev_done.all())
                else:
                    stored = obs.n_step_trajectory["intrinsic_rewards"].cpu()
                    T.testing.assert_close(stored[:, 0], call["bonus"][~prev_done])

    train_steps(30)

    # Evaluation must never call the IM.
    calls_before_eval = len(im.reward_calls)
    for _ in range(30):
        trainer.step(training=False)
    assert len(im.reward_calls) == calls_before_eval, (
        "compute_rollout_reward must not run during evaluation steps")

    # And training must call it again afterwards (the train/eval switch is
    # not stuck off).
    train_steps(10)


# =============================================================================
# Bug 2: episodic memory cleared for exactly the envs that finished
# =============================================================================

@_PATHS
def test_on_episode_end_called_with_exactly_the_finished_envs(
        tmp_path, force_cpu, with_nstep):
    trainer, im, _norm, buffer = _make_trainer(tmp_path, with_nstep)

    episodes_ended = 0
    for _ in range(60):
        ends_before = len(im.episode_end_calls)
        trainer.step(training=True)

        finished = _dones(buffer.records[-1]["observation"]).nonzero().flatten().tolist()
        if finished:
            episodes_ended += len(finished)
            assert len(im.episode_end_calls) == ends_before + 1, (
                "on_episode_end must be called once on a step where envs finish")
            assert im.episode_end_calls[-1] == sorted(finished)
            # Memory is cleared only after this step's bonus was computed.
            assert im.events[-2:] == ["reward", "end"]
        else:
            assert len(im.episode_end_calls) == ends_before, (
                "on_episode_end must not be called on a step where no env finished")

    # Sanity: the run must actually have exercised episode ends.
    assert episodes_ended >= NUM_ENVS
    assert sum(len(c) for c in im.episode_end_calls) == episodes_ended

    # Evaluation never touches episodic memory, even when episodes end.
    ends_before_eval = len(im.episode_end_calls)
    eval_episodes_ended = 0
    for _ in range(30):
        result = trainer.step(training=False)
        eval_episodes_ended += len(result["episode_logs"])
    assert eval_episodes_ended > 0, "eval phase must include finished episodes"
    assert len(im.episode_end_calls) == ends_before_eval, (
        "on_episode_end must not be called during evaluation steps")