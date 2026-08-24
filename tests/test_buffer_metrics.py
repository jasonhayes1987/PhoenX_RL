"""Behavior-contract tests for the buffer-occupancy / optimizer-activity metrics
that replace the ``offpolicy/buffer_fill`` and ``offpolicy/replay_ratio``
categorization defect.

Covers ``Buffer.fill`` (and its four concrete formulas), ``env_steps``,
``ModularModel.optimizer_steps``, the ``buffer/*`` and ``optim/*``
``MetricsManager`` keys' capability gating and values, the ``buffer.pt``
save/load key set, and where in ``Trainer.train()`` each of these metrics is
and is not computed. Each test below is an independent oracle: either a
closed-form value computed here, or a behavioral assertion transcribed
directly from the specification's normative text.

Two construction strategies are used, chosen per contract:

    * Real buffer / model / agent construction, driven directly (mirrors
      ``tests/test_buffer.py``'s and ``tests/test_metrics.py``'s head-factory
      pattern): used for the ``fill`` formulas, ``env_steps``,
      ``optimizer_steps``, and the ``buffer.pt`` key-set contracts.
    * Minimal ``Trainer``/``agent``/``env``/``buffer`` stand-ins (mirrors
      ``tests/test_trainer.py``'s ``_FakeAgent`` / ``_FakeEnv`` pattern) and,
      for the two contracts that are explicitly about *where in the loop* a
      value is computed, a real ``Trainer.train()`` loop with a recording
      callback -- driving ``step()`` or ``learn()`` directly cannot detect a
      metric recorded at the wrong point in ``train()``'s iteration order.
"""

from __future__ import annotations

import os
from collections import deque

import pytest
import torch as T

from phoenx.buffer import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer, TrajectoryBuffer
from phoenx.builder import build_trainer_from_config
from phoenx.env_wrapper import Action, GymnasiumWrapper, Observation
from phoenx.metrics import METRIC_REGISTRY, MetricsManager
from phoenx.models import ContinuousQHead, ModularModel, StochasticContinuousHead, StochasticDiscreteHead, ValueHead
from phoenx.rl_agents import ActorCritic, PPO, SAC
from phoenx.trainer import Trainer, TrainingSchedule

DEVICE = "cpu"
T.manual_seed(0)

LC = [
    {"type": "dense", "params": {"units": 8, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


def _d_policy(env, **kw):
    return StochasticDiscreteHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def _c_policy(env, **kw):
    return StochasticContinuousHead(env, layer_config=LC, output_config=OC,
                                     distribution="normal", device=DEVICE, **kw)


def _value_head(env, **kw):
    return ValueHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def _q_head(env, **kw):
    return ContinuousQHead(env, layer_config=LC, merged_config=LC, output_config=OC, device=DEVICE, **kw)


def _rollout_sample(env, traj_len=6, discrete=True):
    E = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    total = traj_len * E
    if discrete:
        actions = T.randint(0, env.single_action_space.n, (traj_len, E, 1)).float()
        raw_actions = actions.clone()
    else:
        act_dim = env.single_action_space.shape[0]
        actions = (T.rand(traj_len, E, act_dim) * 2 - 1)
        raw_actions = T.randn(traj_len, E, act_dim) * 0.5
    return {
        "states": T.randn(traj_len, E, obs_dim),
        "actions": actions,
        "raw_actions": raw_actions,
        "rewards": T.randn(traj_len, E),
        "intrinsic_rewards": T.zeros(traj_len, E),
        "next_states": T.randn(traj_len, E, obs_dim),
        "terminations": T.zeros(traj_len, E, dtype=T.bool),
        "truncations": T.zeros(traj_len, E, dtype=T.bool),
        "log_probs": T.randn(traj_len, E) * 0.1,
        "first_steps": T.zeros(traj_len, E, dtype=T.bool),
        "valid_indices": T.arange(total).unsqueeze(-1),
        "state_achieved_goals": None,
        "next_state_achieved_goals": None,
        "desired_goals": None,
    }


def _replay_sample(env, B=16, N=1):
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    actions = T.rand(B, N, act_dim) * 2 - 1
    return {
        "states": T.randn(B, N, obs_dim),
        "actions": actions,
        "raw_actions": T.randn_like(actions) * 0.1,
        "log_probs": T.randn(B, N) * 0.1,
        "rewards": T.randn(B, N),
        "intrinsic_rewards": T.zeros(B, N),
        "next_states": T.randn(B, N, obs_dim),
        "terminations": T.zeros(B, N, dtype=T.bool),
        "truncations": T.zeros(B, N, dtype=T.bool),
        "trajectory_lengths": T.full((B,), N, dtype=T.long),
        "state_achieved_goals": None,
        "next_state_achieved_goals": None,
        "desired_goals": None,
    }


def _nstep_batch(env, N, B):
    """Synthetic N-step batch for ``ReplayBuffer.add`` / ``PrioritizedReplayBuffer.add``,
    shaped ``(B, N, ...)`` like ``VectorNStepReward`` emits. Drives ``add()``
    directly (not ``record()``): ``record()`` only reaches ``add()`` when an
    n-step trajectory is present on the observation.
    """
    obs_dim = env.single_observation_space.shape[0]
    if hasattr(env.single_action_space, "n"):
        actions = T.randint(0, env.single_action_space.n, (B, N))
    else:
        actions = T.randn(B, N, *env.single_action_space.shape)
    return {
        "states": T.randn(B, N, obs_dim),
        "actions": actions,
        "rewards": T.randn(B, N),
        "next_states": T.randn(B, N, obs_dim),
        "terminations": T.zeros(B, N, dtype=T.bool),
        "truncations": T.zeros(B, N, dtype=T.bool),
        "trajectory_lengths": T.full((B,), N, dtype=T.long),
    }


def _step_batch(env, *, terminations=None):
    """Synthetic single-timestep batch for ``RolloutBuffer.add`` / ``TrajectoryBuffer.add``,
    shaped ``(num_envs, ...)``.
    """
    E = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    if hasattr(env.single_action_space, "n"):
        actions = T.randint(0, env.single_action_space.n, (E,))
    else:
        actions = T.randn(E, *env.single_action_space.shape)
    if terminations is None:
        terminations = T.zeros(E, dtype=T.bool)
    return {
        "states": T.randn(E, obs_dim),
        "actions": actions,
        "rewards": T.randn(E),
        "next_states": T.randn(E, obs_dim),
        "terminations": terminations,
        "truncations": T.zeros(E, dtype=T.bool),
    }


def _drive_record_steps(buf, env, n):
    """Step a real env ``n`` times, feeding each transition through ``buf.record()``
    (as opposed to ``add()`` directly) -- required for the ``env_steps``
    contracts, since ``env_steps`` is only touched inside ``record()``.
    """
    obs = env.reset(seed=0)
    prev_dones = T.zeros(env.num_envs, dtype=T.bool)
    for _ in range(n):
        a = T.as_tensor(env.action_space.sample())
        act = Action(actions=a, log_probs=T.zeros(env.num_envs))
        next_obs = env.step(a)
        buf.record(next_obs, obs, act, prev_dones)
        prev_dones = T.logical_or(next_obs.terminations, next_obs.truncations)
        obs = next_obs


def _record_direct(buf, env, n):
    """Call ``buf.record()`` ``n`` times with directly constructed ``Observation``
    / ``Action`` objects -- no live env stepping. ``env_steps`` increments
    unconditionally at the top of every subclass's ``record()``, before any
    n-step-trajectory or ``add()`` branch, so a bare ``n_step_trajectory=None``
    observation (for the replay buffers) or a fully populated one-step batch
    (for the on-policy buffers, whose ``record()`` unconditionally calls
    ``add()``) is enough; this needs no live env and therefore sidesteps
    ``VectorNStepReward``'s ``set_action()`` requirement entirely.
    """
    E = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    if hasattr(env.single_action_space, "n"):
        actions = T.zeros(E, dtype=T.long)
    else:
        actions = T.zeros(E, *env.single_action_space.shape)
    prev_dones = T.zeros(E, dtype=T.bool)
    for _ in range(n):
        prev_obs = Observation(states=T.zeros(E, obs_dim))
        cur_obs = Observation(
            states=T.zeros(E, obs_dim),
            rewards=T.zeros(E),
            terminations=T.zeros(E, dtype=T.bool),
            truncations=T.zeros(E, dtype=T.bool),
            n_step_trajectory=None,
        )
        act = Action(actions=actions, log_probs=T.zeros(E))
        buf.record(cur_obs, prev_obs, act, prev_dones)


def _safe_close(env):
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass


# =============================================================================
# Real envs (module-scoped: cheap architecture-only construction, reused
# read-only-ish across the fill / env_steps / optimizer_steps contracts).
# =============================================================================
@pytest.fixture(scope="module")
def plain_env1():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=1, seed=0)
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def plain_env3():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=3, seed=0)
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def nstep_env1():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=1, seed=0,
                            wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}])
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def nstep_env3():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=3, seed=0,
                            wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}])
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def cartpole_env2():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=2, seed=0)
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def pendulum_env_nstep2():
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0,
                            wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}])
    yield env
    _safe_close(env)


# =============================================================================
# CONTRACT: ReplayBuffer.fill is samples_added/buffer_size, saturating at 1.0.
# =============================================================================
def test_replay_buffer_fill_is_samples_added_over_buffer_size_saturating_at_one(nstep_env1):
    buf = ReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    assert buf.fill == pytest.approx(0.0)

    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    assert buf.samples_added == 3
    assert buf.fill == pytest.approx(0.3)

    # Accumulate across several in-capacity add() calls rather than a single
    # over-capacity batch: ReplayBuffer.add's index arithmetic wraps and
    # yields only buffer_size unique slots, so a single batch larger than
    # buffer_size raises a shape mismatch -- a pre-existing property of
    # add() itself, unrelated to fill.
    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    assert buf.samples_added == 13
    assert buf.fill == pytest.approx(1.0)
    assert buf.fill <= 1.0


# =============================================================================
# CONTRACT: PrioritizedReplayBuffer.fill inherits ReplayBuffer's formula.
# =============================================================================
def test_prioritized_replay_buffer_fill_inherits_replay_buffer_formula(nstep_env1):
    buf = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    assert buf.fill == pytest.approx(0.0)

    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    assert buf.fill == pytest.approx(0.3)

    # Accumulate across several in-capacity add() calls rather than a single
    # over-capacity batch (see the ReplayBuffer test above for why).
    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    assert buf.samples_added == 13
    assert buf.fill == pytest.approx(1.0)
    assert buf.fill <= 1.0


# =============================================================================
# CONTRACT: RolloutBuffer.fill tracks the largest write pointer and returns to
#           0.0 after sample().
# =============================================================================
def test_rollout_buffer_fill_tracks_largest_write_pointer_and_resets_after_sample(plain_env1):
    buf = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    assert buf.fill == pytest.approx(0.0)

    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    assert buf.fill == pytest.approx(0.25)

    buf.sample()
    assert buf.fill == pytest.approx(0.0)


# =============================================================================
# CONTRACT: TrajectoryBuffer.fill does NOT return to 0.0 after sample(), and
#           drops per-env only when that env finishes an episode.
# =============================================================================
def test_trajectory_buffer_fill_does_not_reset_after_sample_drops_only_on_episode_end(plain_env1):
    buf = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)

    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    assert buf.fill == pytest.approx(0.375)  # 3 / 8

    buf.sample()
    assert buf.fill == pytest.approx(0.375), (
        "TrajectoryBuffer.fill must not drop to 0.0 after sample() -- unlike "
        "RolloutBuffer, sample() does not reset its write pointer"
    )

    buf.add(**_step_batch(plain_env1, terminations=T.tensor([True])))
    assert buf.fill == pytest.approx(0.0), (
        "cur_idx[i] is zeroed only when env i's episode completes"
    )


# =============================================================================
# CONTRACT: fill never raises and never divides by zero.
# =============================================================================
def test_fill_never_raises_when_buffer_size_is_zero(nstep_env1, plain_env1):
    rb = ReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    rb.buffer_size = 0
    assert rb.fill == 0.0

    per = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    per.buffer_size = 0
    assert per.fill == 0.0

    rob = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    rob.buffer_size = 0
    assert rob.fill == 0.0

    trb = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    trb.buffer_size = 0
    assert trb.fill == 0.0


def test_fill_never_raises_when_cur_idx_is_empty(plain_env1):
    rob = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    rob.cur_idx = T.zeros(0, dtype=T.long, device=rob.device)
    assert rob.fill == 0.0

    trb = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    trb.cur_idx = T.zeros(0, dtype=T.long, device=trb.device)
    assert trb.fill == 0.0


# =============================================================================
# CONTRACT: fill does not mutate the write pointer.
# =============================================================================
def test_fill_does_not_mutate_or_alias_the_write_pointer(plain_env1):
    buf = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    snapshot = buf.cur_idx.clone()

    first = buf.fill
    second = buf.fill

    assert T.equal(buf.cur_idx, snapshot), "reading .fill must not mutate cur_idx"
    assert not isinstance(first, T.Tensor), ".fill must not return a tensor view"
    assert type(first) is float
    assert first == pytest.approx(0.25)
    assert second == pytest.approx(0.25)


# =============================================================================
# CONTRACT: env_steps is maintained by every buffer subclass.
# =============================================================================
def test_env_steps_maintained_by_every_buffer_subclass(nstep_env3, plain_env3):
    n = 4
    k = 3  # num_envs on both fixtures below

    rb = ReplayBuffer(env=nstep_env3, buffer_size=100, N=1, device=DEVICE)
    _record_direct(rb, nstep_env3, n)
    assert rb.env_steps == n * k

    per = PrioritizedReplayBuffer(env=nstep_env3, buffer_size=100, N=1, device=DEVICE)
    _record_direct(per, nstep_env3, n)
    assert per.env_steps == n * k

    rob = RolloutBuffer(env=plain_env3, buffer_size=100, device=DEVICE)
    _record_direct(rob, plain_env3, n)
    assert rob.env_steps == n * k

    trb = TrajectoryBuffer(env=plain_env3, buffer_size=100, device=DEVICE)
    _record_direct(trb, plain_env3, n)
    assert trb.env_steps == n * k


# =============================================================================
# CONTRACT: env_steps is lifetime-scoped for the on-policy buffers.
# =============================================================================
def test_env_steps_lifetime_scoped_for_rollout_buffer_survives_sample_reset(plain_env1):
    buf = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    _drive_record_steps(buf, plain_env1, 3)
    assert buf.env_steps == 3

    buf.sample()  # resets cur_idx -- must NOT reset env_steps
    assert buf.env_steps == 3

    _drive_record_steps(buf, plain_env1, 2)
    assert buf.env_steps == 5


def test_env_steps_lifetime_scoped_for_trajectory_buffer_survives_sample(plain_env1):
    """Companion to the RolloutBuffer case above: the contract title says "the
    on-policy buffers" (plural) though its VERIFY line names only
    RolloutBuffer; TrajectoryBuffer's sample() also must not zero env_steps.
    """
    buf = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    _drive_record_steps(buf, plain_env1, 3)
    assert buf.env_steps == 3

    buf.sample()
    assert buf.env_steps == 3


# =============================================================================
# CONTRACT: optimizer_steps counts ModularModel.step() calls, not per-module
#           optimizer .step() calls.
# =============================================================================
def test_optimizer_steps_counts_modular_model_step_calls_not_per_module_optimizer_steps(cartpole_env2):
    model = ModularModel(
        env=cartpole_env2,
        branches={
            "policy": _d_policy(cartpole_env2),
            "value": _value_head(cartpole_env2),
            "value_b": _value_head(cartpole_env2),
        },
        optimizer_params=OPT,
        device=DEVICE,
    )
    assert len(model.optimizers) >= 3, "test setup invariant: need >= 3 modules owning optimizers"
    assert model.optimizer_steps == 0

    model.step()
    assert model.optimizer_steps == 1, "one step() call (which steps 3 optimizers) must count as 1"

    model.step(modules=[])  # empty resolution: zero optimizers actually stepped
    assert model.optimizer_steps == 2, (
        "the counter increments unconditionally on entry, even when "
        "_resolve_optimizer_names resolves to an empty list"
    )


# =============================================================================
# CONTRACT: The temperature optimizer does not count as an optimizer step, for
#           on-policy agents as well as SAC.
# =============================================================================
def test_temperature_optimizer_not_counted_for_sac_auto_entropy(pendulum_env_nstep2, monkeypatch):
    sac = SAC(policy=_c_policy(pendulum_env_nstep2), critic=_q_head(pendulum_env_nstep2),
              optimizer_params=OPT, auto_entropy_tuning=True, entropy_lr=1e-3,
              target_entropy_scale=1.0, device=DEVICE)
    assert hasattr(sac, "entropy_optimizer"), "test setup invariant: auto_entropy_tuning=True builds one"

    step_calls = []
    original_step = sac.model.step

    def _counting_step(modules=None):
        step_calls.append(modules)
        return original_step(modules)

    monkeypatch.setattr(sac.model, "step", _counting_step)

    entropy_step_calls = []
    original_entropy_step = sac.entropy_optimizer.step

    def _counting_entropy_step(*a, **k):
        entropy_step_calls.append(True)
        return original_entropy_step(*a, **k)

    monkeypatch.setattr(sac.entropy_optimizer, "step", _counting_entropy_step)

    before = sac.model.optimizer_steps
    sample = _replay_sample(pendulum_env_nstep2, B=16, N=1)
    sac.learn(0, sample)
    after = sac.model.optimizer_steps

    assert len(entropy_step_calls) >= 1, "test setup invariant: entropy_optimizer.step() must fire"
    assert after - before == len(step_calls), (
        "optimizer_steps must advance by exactly the number of observed "
        "ModularModel.step() calls; the entropy optimizer's step() must not "
        "contribute to it"
    )


def test_temperature_optimizer_not_counted_for_actor_critic_auto_entropy(cartpole_env2, monkeypatch):
    ac = ActorCritic(policy=_d_policy(cartpole_env2), value=_value_head(cartpole_env2),
                      optimizer_params=OPT, auto_entropy_tuning=True, entropy_lr=1e-3,
                      target_entropy_scale=1.0, device=DEVICE)
    assert hasattr(ac, "entropy_optimizer"), "test setup invariant: auto_entropy_tuning=True builds one"

    step_calls = []
    original_step = ac.model.step

    def _counting_step(modules=None):
        step_calls.append(modules)
        return original_step(modules)

    monkeypatch.setattr(ac.model, "step", _counting_step)

    entropy_step_calls = []
    original_entropy_step = ac.entropy_optimizer.step

    def _counting_entropy_step(*a, **k):
        entropy_step_calls.append(True)
        return original_entropy_step(*a, **k)

    monkeypatch.setattr(ac.entropy_optimizer, "step", _counting_entropy_step)

    before = ac.model.optimizer_steps
    sample = _rollout_sample(cartpole_env2, discrete=True)
    ac.learn(0, sample)
    after = ac.model.optimizer_steps

    assert len(entropy_step_calls) >= 1, "test setup invariant: entropy_optimizer.step() must fire"
    assert after - before == len(step_calls), (
        "optimizer_steps must advance by exactly the number of observed "
        "ModularModel.step() calls; the entropy optimizer's step() must not "
        "contribute to it"
    )


# =============================================================================
# Minimal Trainer/agent/env/buffer stand-ins (mirrors tests/test_trainer.py's
# _FakeAgent / _FakeEnv pattern) for the step()/train()-level contracts below.
# =============================================================================
class _StubModularModel:
    """Minimal duck-typed stand-in for ``ModularModel`` exposing ``optimizer_steps``
    and the ``.train()`` / ``.eval()`` / ``.logger`` surface ``_initialize_run``
    touches unconditionally when an agent has a ``model`` attribute.
    """

    class _NoOpLogger:
        def debug(self, *a, **k):
            pass

    def __init__(self):
        self.optimizer_steps = 0
        self.branches = {}
        self.is_recurrent = False
        self.logger = self._NoOpLogger()

    def step(self, modules=None):
        self.optimizer_steps += 1

    def train(self):
        pass

    def eval(self):
        pass


class _ScriptedAgent:
    """Agent stand-in whose ``act`` is fixed and whose ``learn`` calls
    ``model.step()`` a scripted number of times per call (consuming one entry
    of ``learn_step_counts`` per ``learn()`` invocation, 0 once exhausted).
    """

    def __init__(self, num_envs=1, model=None, learn_step_counts=None):
        self.device = "cpu"
        self.intrinsic_motivation = None
        self.model = model
        self._learn_step_counts = list(learn_step_counts) if learn_step_counts else []
        self._learn_calls = 0

    def act(self, states, *_args, **_kwargs):
        n = states.shape[0]
        return Action(actions=T.zeros(n, 1))

    def learn(self, step, sample, **kwargs):
        idx = self._learn_calls
        self._learn_calls += 1
        n = self._learn_step_counts[idx] if idx < len(self._learn_step_counts) else 0
        if self.model is not None:
            for _ in range(n):
                self.model.step()
        return {}

    def get_config(self):
        return {"type": "ScriptedAgent"}

    def save_state(self, path):
        os.makedirs(path, exist_ok=True)


class _ScriptedEnv:
    """Env stand-in that never terminates and returns fixed-shape zero states."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.goal_key = None
        self.config = {"type": "ScriptedEnv"}

    def _find_nstep_wrapper(self):
        return None

    def reset(self, seed=None):
        return Observation(states=T.zeros(self.num_envs, 4))

    def step(self, _actions):
        return Observation(
            states=T.zeros(self.num_envs, 4),
            rewards=T.zeros(self.num_envs),
            terminations=T.zeros(self.num_envs, dtype=T.bool),
            truncations=T.zeros(self.num_envs, dtype=T.bool),
        )

    def close(self):
        pass


class _AlwaysReadyBuffer:
    """Off-policy-shaped buffer stand-in exposing ``buffer_size`` / ``samples_added``
    / ``fill`` (the ``buffer_capacity`` duck-type) and an ``is_ready`` gated on
    ``samples_added``.
    """

    def __init__(self, buffer_size=100, samples_added=0):
        self.buffer_size = buffer_size
        self.samples_added = samples_added
        self.fill = (min(samples_added, buffer_size) / buffer_size) if buffer_size else 0.0

    def record(self, *a, **k):
        pass

    def is_ready(self, samples):
        return self.samples_added >= samples

    def sample(self, samples):
        return {"states": T.zeros(samples, 1)}

    def get_config(self):
        return None


class _OnPolicyStubBuffer:
    """On-policy-shaped buffer stand-in: always ready, independent of any
    ``samples`` threshold.
    """

    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.fill = 0.5

    def record(self, *a, **k):
        pass

    def is_ready(self, **kwargs):
        return True

    def sample(self, **kwargs):
        return {"states": T.zeros(1, 1)}

    def get_config(self):
        return None


class _RecordingCallback:
    """Callback stand-in that records every ``on_train_step_end`` logs dict."""

    def __init__(self):
        self.step_logs = []

    def get_config(self):
        return {"type": "RecordingCallback"}

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_step_end(self, step, logs):
        self.step_logs.append(dict(logs) if logs is not None else {})

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass


def _make_stub_trainer(env, buffer, tmp_path, agent, **schedule_kwargs):
    """Build a real ``Trainer`` with pre-seeded episode-tracking state, exactly
    as ``tests/test_trainer.py``'s ``_make_trainer`` does, so ``step()`` /
    ``learn()`` can be driven without booting a real env/agent.
    """
    schedule = TrainingSchedule(**schedule_kwargs)
    trainer = Trainer(
        agent=agent, env=env, schedule=schedule, success_criterion=None,
        buffer=buffer, renderer=None, callbacks=None, save_dir=str(tmp_path),
    )
    n = env.num_envs
    trainer._step = 0
    trainer._grad_step = 0
    trainer._prev_obs = Observation(states=T.zeros(n, 4))
    trainer._prev_done = T.zeros(n, dtype=T.bool)
    trainer._best_reward = float("-inf")
    trainer._last_save = None
    trainer._best_pending = False
    trainer._episode_steps = T.zeros(n, dtype=T.int32)
    trainer._completed_episodes = T.zeros(n, dtype=T.int32)
    trainer._episode_scores = T.zeros(n, dtype=T.float32)
    trainer._score_history = deque(maxlen=100)
    trainer._last_learn = 0
    return trainer


# =============================================================================
# CONTRACT: A stub agent without a `model`, or a stub buffer without `fill`,
#           does not crash Trainer.step()/_initialize_run.
# =============================================================================
def test_stub_agent_without_model_and_stub_buffer_without_fill_do_not_crash_trainer(tmp_path):
    class _NoModelAgent:
        def __init__(self):
            self.device = "cpu"
            self.intrinsic_motivation = None

        def act(self, states, *_a, **_k):
            return Action(actions=T.zeros(states.shape[0], 1))

        def get_config(self):
            return {"type": "NoModelAgent"}

        def save_state(self, path):
            os.makedirs(path, exist_ok=True)

    class _NoFillBuffer:
        def record(self, *a, **k):
            pass

        def get_config(self):
            return None

    agent = _NoModelAgent()
    env = _ScriptedEnv(num_envs=1)
    buffer = _NoFillBuffer()
    schedule = TrainingSchedule(stop_unit="timestep", stop_units=2, seed=0)
    trainer = Trainer(agent=agent, env=env, schedule=schedule, buffer=buffer, save_dir=str(tmp_path))

    # Drive _initialize_run and step() directly -- not train(), which also
    # calls learn(), and Trainer.learn() has always called
    # self.buffer.is_ready(samples=...) even before this change, so a stub
    # buffer reaching that path must still implement is_ready regardless of
    # the buffer/* metric sites under test here.
    trainer._initialize_run("train")  # must not raise
    result = trainer.step(training=True)  # must not raise

    assert trainer._step == 0  # step() itself does not advance _step; train()'s loop does
    step_log = result["step_log"]
    assert not any(k.startswith("buffer/") for k in step_log)
    assert not any(k.startswith("optim/") for k in step_log)
    collected = trainer.metrics.collect()
    assert not any(k.startswith("buffer/") for k in collected)
    assert not any(k.startswith("optim/") for k in collected)


# =============================================================================
# CONTRACT: buffer/is_ready reports the buffer gate, which is independent of
#           the schedule warmup gate.
# =============================================================================
@pytest.mark.parametrize("warmup_steps", [0, 1000])
def test_buffer_is_ready_below_threshold_is_zero_independent_of_warmup(tmp_path, warmup_steps):
    env = _ScriptedEnv(num_envs=1)
    agent = _ScriptedAgent(num_envs=1)
    buffer = _AlwaysReadyBuffer(buffer_size=100, samples_added=5)
    trainer = _make_stub_trainer(
        env, buffer, tmp_path, agent=agent, stop_unit="timestep", stop_units=1,
        updates_per_learn=1, batch_size=10, warmup_steps=warmup_steps,
    )
    trainer.metrics.bind(agent=agent, env=env, buffer=buffer)

    trainer.step(training=True)
    collected = trainer.metrics.collect()
    assert collected.get("buffer/is_ready") == pytest.approx(0.0)


def test_buffer_is_ready_above_threshold_is_one_even_during_warmup(tmp_path):
    env = _ScriptedEnv(num_envs=1)
    agent = _ScriptedAgent(num_envs=1)
    buffer = _AlwaysReadyBuffer(buffer_size=100, samples_added=20)
    trainer = _make_stub_trainer(
        env, buffer, tmp_path, agent=agent, stop_unit="timestep", stop_units=1,
        updates_per_learn=1, batch_size=10, warmup_steps=1000,
    )
    trainer.metrics.bind(agent=agent, env=env, buffer=buffer)

    trainer.step(training=True)
    collected = trainer.metrics.collect()
    assert collected.get("buffer/is_ready") == pytest.approx(1.0), (
        "buffer/is_ready must report the buffer gate alone -- a correct "
        "implementation can emit 1.0 during warmup once enough transitions exist"
    )


@pytest.mark.parametrize("buffer_cls", [RolloutBuffer, TrajectoryBuffer])
def test_buffer_is_ready_always_one_for_on_policy_buffer(tmp_path, buffer_cls):
    """Drives a real ``RolloutBuffer`` / ``TrajectoryBuffer`` (not a stub whose
    ``is_ready`` hardcodes ``True``), so a regression that gates
    ``RolloutBuffer.is_ready`` on something other than an unconditional
    ``True`` -- e.g. on ``completed_trajectories``, a concept
    ``TrajectoryBuffer`` alone defines -- would be caught for both classes,
    including confirming ``TrajectoryBuffer`` inherits the same
    unconditional gate rather than defining its own.
    """
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=1, seed=0)
    try:
        buffer = buffer_cls(env=env, buffer_size=8, device=DEVICE)
        agent = _ScriptedAgent(num_envs=1)
        # CartPole is Discrete(2); _ScriptedAgent.act's default float zeros
        # fails the real env's action_space.contains() check, unlike
        # _ScriptedEnv (used elsewhere in this module) which never validates.
        agent.act = lambda states, *a, **k: Action(actions=T.zeros(states.shape[0], dtype=T.long))
        trainer = _make_stub_trainer(env, buffer, tmp_path, agent=agent, stop_unit="timestep", stop_units=1)
        trainer._prev_obs = env.reset(seed=0)
        # _make_stub_trainer pre-seeds CPU tensors; a real env (unlike
        # _ScriptedEnv) resolves to CUDA when available, so the episode
        # counters must follow it or step() raises a device mismatch.
        real_device = trainer._prev_obs.states.device
        trainer._prev_done = trainer._prev_done.to(real_device)
        trainer._episode_steps = trainer._episode_steps.to(real_device)
        trainer._completed_episodes = trainer._completed_episodes.to(real_device)
        trainer._episode_scores = trainer._episode_scores.to(real_device)
        trainer.metrics.bind(agent=agent, env=env, buffer=buffer)

        trainer.step(training=True)
        collected = trainer.metrics.collect()
        assert collected.get("buffer/is_ready") == pytest.approx(1.0), (
            f"{buffer_cls.__name__}.is_ready() must report 1.0 through "
            f"buffer/is_ready unconditionally -- on-policy buffers are always "
            f"ready once bound, independent of samples_added or warmup"
        )
    finally:
        _safe_close(env)


# =============================================================================
# CONTRACT: buffer/pending_trajectories counts entries in the list, and is
#           gated to TrajectoryBuffer.
# =============================================================================
def test_buffer_pending_trajectories_counts_completed_trajectories_list_length(tmp_path, plain_env1):
    buf = TrajectoryBuffer(env=plain_env1, buffer_size=50, hindsight=None, device=DEVICE)
    for _ in range(3):
        buf.add(**_step_batch(plain_env1, terminations=T.tensor([True])))
    assert len(buf.completed_trajectories) == 3, "test setup invariant: 3 completed episodes"

    env = _ScriptedEnv(num_envs=1)
    agent = _ScriptedAgent(num_envs=1)
    trainer = _make_stub_trainer(env, buf, tmp_path, agent=agent, stop_unit="timestep", stop_units=1)
    trainer.metrics.bind(agent=agent, env=env, buffer=buf)

    trainer.step(training=True)  # a 4th, non-terminating step; must not harvest a 4th trajectory
    collected = trainer.metrics.collect()
    assert collected.get("buffer/pending_trajectories") == pytest.approx(3.0)


def test_buffer_pending_trajectories_absent_when_rollout_buffer_bound(plain_env1):
    buf = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    agent = _ScriptedAgent(num_envs=1)
    mm = MetricsManager()
    mm.bind(agent=agent, env=plain_env1, buffer=buf)
    assert "buffer/pending_trajectories" not in mm.enabled_keys()


def test_buffer_pending_trajectories_absent_when_replay_buffer_bound(nstep_env1):
    buf = ReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    agent = _ScriptedAgent(num_envs=1)
    mm = MetricsManager()
    mm.bind(agent=agent, env=nstep_env1, buffer=buf)
    assert "buffer/pending_trajectories" not in mm.enabled_keys()


# =============================================================================
# CONTRACT: The buffer.pt key set is unchanged.
#
# Literal key sets below are hand-derived from the prior commit's
# ``Buffer.save_state`` (buckets every ``T.Tensor`` / dict-of-tensors
# attribute into "tensors", every int/float/bool attribute into "scalars",
# and a ``SumTree`` (if present) into "extra"), read via `git show` against
# the pinned base commit -- not by importing or diffing the working tree.
# No stored state is added by this change, so these sets must still match.
# =============================================================================
REPLAY_BUFFER_TENSOR_KEYS = frozenset({
    "states", "actions", "raw_actions", "log_probs", "rewards",
    "intrinsic_rewards", "next_states", "terminations", "truncations",
    "trajectory_lengths",
})
REPLAY_BUFFER_SCALAR_KEYS = frozenset({
    "buffer_size", "env_steps", "_storage_dtypes_synced", "N", "samples_added",
})
REPLAY_BUFFER_EXTRA_KEYS = frozenset()

PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS = REPLAY_BUFFER_TENSOR_KEYS
PRIORITIZED_REPLAY_BUFFER_SCALAR_KEYS = REPLAY_BUFFER_SCALAR_KEYS | frozenset({
    "alpha", "beta_start", "beta_iter", "beta_update_freq", "sort_freq",
    "epsilon", "beta", "_sample_calls", "_beta_updates", "_samples_since_sort",
})
PRIORITIZED_REPLAY_BUFFER_EXTRA_KEYS = frozenset({"sum_tree_tree", "sum_tree_max_priority"})

ROLLOUT_BUFFER_TENSOR_KEYS = frozenset({
    "cur_idx", "states", "actions", "raw_actions", "log_probs", "rewards",
    "intrinsic_rewards", "next_states", "terminations", "truncations", "first_steps",
})
ROLLOUT_BUFFER_SCALAR_KEYS = frozenset({"buffer_size", "env_steps", "_storage_dtypes_synced"})
ROLLOUT_BUFFER_EXTRA_KEYS = frozenset()

TRAJECTORY_BUFFER_TENSOR_KEYS = ROLLOUT_BUFFER_TENSOR_KEYS
TRAJECTORY_BUFFER_SCALAR_KEYS = ROLLOUT_BUFFER_SCALAR_KEYS
TRAJECTORY_BUFFER_EXTRA_KEYS = frozenset()


def test_buffer_pt_key_set_unchanged_for_replay_buffer(nstep_env1, tmp_path):
    buf = ReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    buf.add(**_nstep_batch(nstep_env1, N=1, B=3))
    path = tmp_path / "buffer.pt"
    buf.save_state(path)
    payload = T.load(path, map_location="cpu", weights_only=False)

    assert frozenset(payload["tensors"].keys()) == REPLAY_BUFFER_TENSOR_KEYS
    assert frozenset(payload["scalars"].keys()) == REPLAY_BUFFER_SCALAR_KEYS
    assert frozenset(payload["extra"].keys()) == REPLAY_BUFFER_EXTRA_KEYS

    restored = ReplayBuffer(env=nstep_env1, buffer_size=10, N=1, device=DEVICE)
    restored.load_state(path)
    for key in REPLAY_BUFFER_TENSOR_KEYS:
        assert T.equal(getattr(restored, key), getattr(buf, key)), key


def test_buffer_pt_key_set_unchanged_for_prioritized_replay_buffer(nstep_env1, tmp_path):
    buf = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, priority="proportional", device=DEVICE)
    buf.add(**_nstep_batch(nstep_env1, N=1, B=3))
    path = tmp_path / "buffer.pt"
    buf.save_state(path)
    payload = T.load(path, map_location="cpu", weights_only=False)

    assert frozenset(payload["tensors"].keys()) == PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS
    assert frozenset(payload["scalars"].keys()) == PRIORITIZED_REPLAY_BUFFER_SCALAR_KEYS
    assert frozenset(payload["extra"].keys()) == PRIORITIZED_REPLAY_BUFFER_EXTRA_KEYS

    restored = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, priority="proportional", device=DEVICE)
    restored.load_state(path)
    for key in PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS:
        assert T.equal(getattr(restored, key), getattr(buf, key)), key
    assert T.equal(restored.sum_tree.tree, buf.sum_tree.tree)


def test_buffer_pt_key_set_unchanged_for_rollout_buffer(plain_env1, tmp_path):
    buf = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    path = tmp_path / "buffer.pt"
    buf.save_state(path)
    payload = T.load(path, map_location="cpu", weights_only=False)

    assert frozenset(payload["tensors"].keys()) == ROLLOUT_BUFFER_TENSOR_KEYS
    assert frozenset(payload["scalars"].keys()) == ROLLOUT_BUFFER_SCALAR_KEYS
    assert frozenset(payload["extra"].keys()) == ROLLOUT_BUFFER_EXTRA_KEYS

    restored = RolloutBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    restored.load_state(path)
    for key in ROLLOUT_BUFFER_TENSOR_KEYS:
        assert T.equal(getattr(restored, key), getattr(buf, key)), key


def test_buffer_pt_key_set_unchanged_for_trajectory_buffer(plain_env1, tmp_path):
    buf = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    buf.add(**_step_batch(plain_env1))
    buf.add(**_step_batch(plain_env1))
    path = tmp_path / "buffer.pt"
    buf.save_state(path)
    payload = T.load(path, map_location="cpu", weights_only=False)

    assert frozenset(payload["tensors"].keys()) == TRAJECTORY_BUFFER_TENSOR_KEYS
    assert frozenset(payload["scalars"].keys()) == TRAJECTORY_BUFFER_SCALAR_KEYS
    assert frozenset(payload["extra"].keys()) == TRAJECTORY_BUFFER_EXTRA_KEYS

    restored = TrajectoryBuffer(env=plain_env1, buffer_size=8, device=DEVICE)
    restored.load_state(path)
    for key in TRAJECTORY_BUFFER_TENSOR_KEYS:
        assert T.equal(getattr(restored, key), getattr(buf, key)), key


# =============================================================================
# CONTRACT: No metric key beginning "offpolicy/" remains.
# =============================================================================
def test_no_metric_key_beginning_offpolicy_remains_in_registry():
    assert not [k for k in METRIC_REGISTRY if k.startswith("offpolicy/")]


# =============================================================================
# CONTRACT: buffer/fill IS enabled for an on-policy agent bound with a real
#           buffer.
# =============================================================================
def test_buffer_fill_enabled_for_ppo_bound_with_real_rollout_buffer(cartpole_env2):
    ppo = PPO(policy=_d_policy(cartpole_env2), value=_value_head(cartpole_env2),
              optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)
    buf = RolloutBuffer(env=cartpole_env2, buffer_size=8, device=DEVICE)
    mm = MetricsManager()
    mm.bind(agent=ppo, env=cartpole_env2, buffer=buf)
    assert "buffer/fill" in mm.enabled_keys()
    assert "offpolicy/buffer_fill" not in mm.enabled_keys()


# =============================================================================
# CONTRACT: buffer/fill is emitted for on-policy and off-policy agents alike.
#
# learn_every is set far beyond stop_units so agent.learn() is never called --
# irrelevant to this contract (buffer/fill lives in step()'s unconditional
# `if training:` block) and keeps the run to a single env.step() so it stays
# in the fast subset. Must NOT be marked slow/isaac.
# =============================================================================
def _tiny_ppo_config(save_dir):
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = {"type": "dense", "params": {"units": 4, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 1, "learn_every_unit": "timestep",
            "learn_every": 10 ** 9, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 1, "learning_epochs": 1, "warmup_steps": 0, "seed": 3,
        },
        "agent": {"type": "PPO", "config": {
            "name": "PPO",
            "model": {"branches": {
                "policy": {"type": "StochasticDiscreteHead", "layer_config": [dense, relu],
                           "output_config": out, "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                           "distribution": "categorical", "device": dev},
                "value": {"type": "ValueHead", "layer_config": [dense, relu], "output_config": out,
                          "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
            }},
            "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
            "entropy_coefficient": 0.01, "policy_clip": 0.2, "value_clip": 0.2,
            "policy_grad_clip": 1.0, "value_grad_clip": 1.0, "value_coef": 0.5,
            "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "CartPole-v1", "num_envs": 1, "obs_key": None, "goal_key": None,
            "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 3}},
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 2}},
    }


def _tiny_reinforce_config(save_dir):
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = {"type": "dense", "params": {"units": 4, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 1, "learn_every_unit": "timestep",
            "learn_every": 10 ** 9, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 1, "learning_epochs": 1, "warmup_steps": 0, "seed": 3,
        },
        "agent": {"type": "Reinforce", "config": {
            "name": "Reinforce",
            "model": {"branches": {
                "policy": {"type": "StochasticDiscreteHead", "layer_config": [dense, relu],
                           "output_config": out, "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                           "distribution": "categorical", "device": dev},
                "value": {"type": "ValueHead", "layer_config": [dense, relu], "output_config": out,
                          "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
            }},
            "discount": 0.99, "auto_entropy_tuning": False, "entropy_coefficient": 0.01,
            "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "CartPole-v1", "num_envs": 1, "obs_key": None, "goal_key": None,
            "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 3}},
        "buffer": {"type": "TrajectoryBuffer", "config": {"buffer_size": 2}},
    }


def _tiny_sac_config(save_dir):
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = {"type": "dense", "params": {"units": 4, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 1, "learn_every_unit": "timestep",
            "learn_every": 10 ** 9, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 1, "learning_epochs": 1, "warmup_steps": 0, "seed": 3,
        },
        "agent": {"type": "SAC", "config": {
            "name": "SAC",
            "model": {"branches": {
                "policy": {"type": "StochasticContinuousHead", "layer_config": [dense, relu],
                           "output_config": out, "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                           "distribution": "normal", "device": dev},
                "critic": {"type": "ContinuousQHead", "layer_config": [dense, relu],
                           "merged_config": [dense, relu], "output_config": out,
                           "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
            }},
            "discount": 0.99, "tau": 0.01, "auto_entropy_tuning": False,
            "policy_grad_clip": 10.0, "critic_grad_clip": 10.0, "critic_huber_delta": 1.0,
            "N": 1, "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "Pendulum-v1", "num_envs": 1, "obs_key": None, "goal_key": None,
            "ach_goal_key": None, "wrappers": [{"type": "VectorNStepReward", "params": {"n": 1}}],
            "render_mode": None, "seed": 3}},
        "buffer": {"type": "ReplayBuffer", "config": {"buffer_size": 10, "N": 1}},
    }


@pytest.mark.parametrize(
    "config_builder",
    [_tiny_ppo_config, _tiny_reinforce_config, _tiny_sac_config],
    ids=["ppo_rollout_buffer", "reinforce_trajectory_buffer", "sac_replay_buffer"],
)
def test_buffer_fill_emitted_for_onpolicy_and_offpolicy_agents_alike(tmp_path, config_builder):
    config = config_builder(str(tmp_path / "run") + os.sep)
    trainer = build_trainer_from_config(config)
    cb = _RecordingCallback()
    trainer.callbacks = [cb]

    trainer.train()

    assert cb.step_logs, "on_train_step_end was never called"
    seen_keys = set()
    for log in cb.step_logs:
        seen_keys.update(log.keys())
    assert "buffer/fill" in seen_keys
    assert not any(k.startswith("offpolicy/") for k in seen_keys), (
        f"an 'offpolicy/' key leaked into on_train_step_end: {sorted(seen_keys)}"
    )


# =============================================================================
# CONTRACT: optim/updates_per_env_step reaches the callback with post-learn,
#           post-step-increment values.
# =============================================================================
def test_optim_updates_per_env_step_reflects_current_iteration_post_learn_post_increment(tmp_path):
    model = _StubModularModel()
    env = _ScriptedEnv(num_envs=2)
    agent = _ScriptedAgent(model=model, learn_step_counts=[3, 5])
    buffer = _AlwaysReadyBuffer(buffer_size=100, samples_added=10)
    schedule = TrainingSchedule(
        stop_unit="timestep", stop_units=4, learn_every_unit="timestep", learn_every=1,
        updates_per_learn=1, batch_size=1, warmup_steps=0, seed=0,
    )
    cb = _RecordingCallback()
    trainer = Trainer(agent=agent, env=env, schedule=schedule, buffer=buffer,
                       callbacks=[cb], save_dir=str(tmp_path))

    trainer.train()

    assert len(cb.step_logs) == 2, "test setup invariant: 2 iterations (stop_units=4, num_envs=2)"
    assert cb.step_logs[0]["optim/updates_per_env_step"] == pytest.approx(3 / 2)
    assert cb.step_logs[1]["optim/updates_per_env_step"] == pytest.approx(8 / 4), (
        "if this metric were recorded inside step() -- before this iteration's "
        "own learn() and its _step increment -- iteration 2 would report the "
        "PREVIOUS iteration's value (3/2 == 1.5) instead of the current one "
        "(8/4 == 2.0)"
    )


# =============================================================================
# CONTRACT: optim/updates_per_env_step is correct across a checkpoint resume,
# for BOTH halves of its formula -- the run-segment env-step denominator and
# the run-segment optimizer-step numerator -- via the real Trainer.save() ->
# Trainer.load() round trip (not by injecting _resume_state directly).
# =============================================================================
def test_optim_updates_per_env_step_after_resume_uses_run_segment_not_absolute_counters(tmp_path, monkeypatch):
    # A prior real run (real SAC agent/model/env/buffer, not stand-ins) that
    # reaches a known _step via Trainer.save() -- config.json, agent weights,
    # trainer_state.pt, rng.pt.
    save_dir = str(tmp_path / "run") + os.sep
    config = _tiny_sac_config(save_dir)
    config["schedule"]["learn_every"] = 1
    config["schedule"]["stop_units"] = 3
    trainer1 = build_trainer_from_config(config)
    trainer1.train()
    trainer1.save()
    assert trainer1._step == 3, "test setup invariant"

    # Resume via the real Trainer.load() -> _apply_resume_state path (not a
    # hand-built _trainer_state() dict poked onto a stub): this is the only
    # way a save() that started persisting _run_start_*, or a load() that
    # failed to stage _resume_state, would be caught.
    trainer2 = Trainer.load(save_dir, load_buffer=True)

    # optimizer_steps is never persisted -- ModularModel.__init__ always
    # starts a freshly built model at 0, so a real load() alone cannot make
    # _run_start_optimizer_steps non-zero. Setting it by hand here, before
    # any further training, is the only way to make the anchored formula
    # (updates = optimizer_steps - (self._run_start_optimizer_steps or 0))
    # differ from the un-anchored one (updates = optimizer_steps): this is
    # exactly what a fresh model whose optimizer_steps starts at 0 cannot
    # exercise, which is why that case left the numerator anchor untested.
    trainer2.agent.model.optimizer_steps = 50

    step_calls = []
    original_step = trainer2.agent.model.step

    def _counting_step(modules=None):
        step_calls.append(modules)
        return original_step(modules)

    monkeypatch.setattr(trainer2.agent.model, "step", _counting_step)

    trainer2.schedule.stop_units = 8  # 5 further env steps past the resumed _step=3
    cb = _RecordingCallback()
    trainer2.callbacks = [cb]

    trainer2.train()

    assert trainer2._step == 8, "resume did not restore/advance _step as scripted"
    new_env_steps = trainer2._step - 3
    new_updates = len(step_calls)  # observed ModularModel.step() calls, not a predicted count
    assert new_updates > 0, "test setup invariant: at least one further learn() must have stepped the model"

    expected_ratio = new_updates / new_env_steps
    unanchored_ratio = (50 + new_updates) / new_env_steps
    assert abs(expected_ratio - unanchored_ratio) > 0.5, (
        "test setup invariant: the anchored and un-anchored formulas must "
        "differ substantially, or this test cannot distinguish them"
    )

    last_logged = cb.step_logs[-1]["optim/updates_per_env_step"]
    assert last_logged == pytest.approx(expected_ratio), (
        f"expected new_updates / new_env_steps == {new_updates}/{new_env_steps} == "
        f"{expected_ratio}; a numerator missing the "
        f"- (self._run_start_optimizer_steps or 0) anchor would instead report "
        f"the baseline-inflated (50 + {new_updates}) / {new_env_steps} == "
        f"{unanchored_ratio:.6f}"
    )


# =============================================================================
# Supplementary coverage for structural gaps the contract tests above could
# not anticipate, added after reading the implementation.
# =============================================================================


# -----------------------------------------------------------------------------
# RolloutBuffer.fill / TrajectoryBuffer.fill with num_envs > 1 and differing
# per-env cur_idx: the property is defined against the max, not the mean, and
# every buffer.fill case above drives all envs identically, so none of them
# can tell a max-based implementation apart from a mean-based one.
# -----------------------------------------------------------------------------
def test_rollout_buffer_fill_uses_max_not_mean_across_envs(cartpole_env2):
    buf = RolloutBuffer(env=cartpole_env2, buffer_size=8, device=DEVICE)
    buf.cur_idx = T.tensor([1, 5], dtype=T.long, device=buf.device)

    assert buf.fill == pytest.approx(5 / 8), (
        "fill must report max(cur_idx)/buffer_size == 5/8; mean(cur_idx)/"
        "buffer_size == 3/8 would silently pass any test that never drives "
        "per-env cur_idx apart"
    )


def test_trajectory_buffer_fill_uses_max_not_mean_across_envs(plain_env3):
    buf = TrajectoryBuffer(env=plain_env3, buffer_size=8, device=DEVICE)

    # env 0 finishes an episode after 1 step (cur_idx[0] resets to 0 inside
    # add()); envs 1 and 2 keep accumulating uninterrupted.
    buf.add(**_step_batch(plain_env3, terminations=T.tensor([True, False, False])))
    buf.add(**_step_batch(plain_env3, terminations=T.tensor([False, False, False])))
    buf.add(**_step_batch(plain_env3, terminations=T.tensor([False, False, False])))

    assert T.equal(buf.cur_idx, T.tensor([2, 3, 3], device=buf.device)), (
        "test setup invariant: env 0 at 2 (reset once then 2 more steps), "
        "envs 1/2 at 3 (uninterrupted)"
    )
    assert buf.fill == pytest.approx(3 / 8), (
        "fill must report max(cur_idx)/buffer_size == 3/8; "
        "mean(cur_idx)/buffer_size == (8/3)/8 would silently pass any test "
        "that never drives per-env cur_idx apart"
    )


# -----------------------------------------------------------------------------
# PrioritizedReplayBuffer with priority="rank": the proportional-mode test
# above covered "proportional" only, and rank mode carries different
# attributes (a dense `priorities` tensor and `max_priority_rank`/`_rank_Z`
# scalars/tensors instead of a `SumTree`), which changes both the buffer.pt
# key set and the extra bucket.
# -----------------------------------------------------------------------------
def test_prioritized_replay_buffer_fill_inherits_replay_buffer_formula_in_rank_mode(nstep_env1):
    buf = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, priority="rank", device=DEVICE)
    assert buf.fill == pytest.approx(0.0)

    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=1))
    assert buf.fill == pytest.approx(0.3)

    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    buf.add(**_nstep_batch(nstep_env1, N=1, B=5))
    assert buf.samples_added == 13
    assert buf.fill == pytest.approx(1.0)
    assert buf.fill <= 1.0


RANK_PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS = REPLAY_BUFFER_TENSOR_KEYS | frozenset({
    "priorities", "max_priority_rank", "_rank_Z",
})
RANK_PRIORITIZED_REPLAY_BUFFER_SCALAR_KEYS = REPLAY_BUFFER_SCALAR_KEYS | frozenset({
    "alpha", "beta_start", "beta_iter", "beta_update_freq", "sort_freq",
    "epsilon", "beta", "_sample_calls", "_beta_updates", "_samples_since_sort",
})
RANK_PRIORITIZED_REPLAY_BUFFER_EXTRA_KEYS = frozenset()


def test_buffer_pt_key_set_for_prioritized_replay_buffer_rank_mode(nstep_env1, tmp_path):
    buf = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, priority="rank", device=DEVICE)
    buf.add(**_nstep_batch(nstep_env1, N=1, B=3))
    path = tmp_path / "buffer.pt"
    buf.save_state(path)
    payload = T.load(path, map_location="cpu", weights_only=False)

    assert frozenset(payload["tensors"].keys()) == RANK_PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS, (
        "rank mode's sorted_indices/_seg_starts/_seg_ends stay None until a "
        "sample() call and so must NOT appear -- save_state only buckets a "
        "dict/tensor value, never a bare None"
    )
    assert frozenset(payload["scalars"].keys()) == RANK_PRIORITIZED_REPLAY_BUFFER_SCALAR_KEYS
    assert frozenset(payload["extra"].keys()) == RANK_PRIORITIZED_REPLAY_BUFFER_EXTRA_KEYS, (
        "rank mode has no sum_tree, so the SumTree-specific extra bucket "
        "must be empty (unlike proportional mode)"
    )

    restored = PrioritizedReplayBuffer(env=nstep_env1, buffer_size=10, N=1, priority="rank", device=DEVICE)
    restored.load_state(path)
    for key in RANK_PRIORITIZED_REPLAY_BUFFER_TENSOR_KEYS:
        assert T.equal(getattr(restored, key), getattr(buf, key)), key


# -----------------------------------------------------------------------------
# optimizer_steps is an instance attribute: two ModularModel instances must
# not share a counter (relevant to running multiple Ray Tune trials, each
# owning its own model, in one process).
# -----------------------------------------------------------------------------
def test_optimizer_steps_is_a_per_instance_attribute_not_shared_across_models(cartpole_env2):
    model_a = ModularModel(env=cartpole_env2, branches={"policy": _d_policy(cartpole_env2)},
                            optimizer_params=OPT, device=DEVICE)
    model_b = ModularModel(env=cartpole_env2, branches={"policy": _d_policy(cartpole_env2)},
                            optimizer_params=OPT, device=DEVICE)

    assert model_a.optimizer_steps == 0
    assert model_b.optimizer_steps == 0

    model_a.step()
    model_a.step()
    assert model_a.optimizer_steps == 2
    assert model_b.optimizer_steps == 0, (
        "optimizer_steps must be a per-instance counter; a class-level "
        "counter would leak model_a's steps into model_b"
    )

    model_b.step()
    assert model_a.optimizer_steps == 2
    assert model_b.optimizer_steps == 1
    assert "optimizer_steps" not in type(model_a).__dict__, (
        "optimizer_steps must not be declared as a class attribute"
    )


# -----------------------------------------------------------------------------
# The buffer_capacity capability predicate requires the fill API itself, not
# merely a truthy buffer_size: a stub carrying only buffer_size must not
# enable buffer/fill, buffer/env_steps, or buffer/is_ready.
# -----------------------------------------------------------------------------
def test_buffer_capacity_capability_not_enabled_for_stub_with_buffer_size_but_no_fill():
    class _BufferSizeOnlyBuffer:
        def __init__(self):
            self.buffer_size = 100

        def record(self, *a, **k):
            pass

        def get_config(self):
            return None

    mm = MetricsManager()
    agent = _ScriptedAgent(num_envs=1)
    env = _ScriptedEnv(num_envs=1)
    buf = _BufferSizeOnlyBuffer()
    mm.bind(agent=agent, env=env, buffer=buf)

    assert "buffer_capacity" not in mm.capabilities()
    assert "buffer/fill" not in mm.enabled_keys()
    assert "buffer/env_steps" not in mm.enabled_keys()
    assert "buffer/is_ready" not in mm.enabled_keys()


# -----------------------------------------------------------------------------
# buffer/env_steps and buffer/fill reaching on_train_step_end with correct
# VALUES (not merely presence, which the cross-agent contract test above
# already covers) through a real Trainer.train() loop.
# -----------------------------------------------------------------------------
class _CountingBuffer:
    """Off-policy-shaped buffer stand-in whose ``env_steps``/``fill`` update on
    every ``record()`` call, mirroring the real subclasses' ``env_steps +=
    env.num_envs`` so the expected values can be computed independently
    (``iteration * num_envs``) rather than by re-deriving the production
    formula.
    """

    def __init__(self, buffer_size, num_envs):
        self.buffer_size = buffer_size
        self.env_steps = 0
        self._num_envs = num_envs

    def record(self, *a, **k):
        self.env_steps += self._num_envs

    @property
    def fill(self):
        return min(self.env_steps, self.buffer_size) / self.buffer_size

    def is_ready(self, samples):
        return False  # never learn -- irrelevant to this contract

    def get_config(self):
        return None


def test_buffer_env_steps_and_fill_reach_on_train_step_end_with_correct_values(tmp_path):
    num_envs = 2
    buffer_size = 10
    env = _ScriptedEnv(num_envs=num_envs)
    agent = _ScriptedAgent(num_envs=num_envs)
    buffer = _CountingBuffer(buffer_size=buffer_size, num_envs=num_envs)
    schedule = TrainingSchedule(
        stop_unit="timestep", stop_units=3 * num_envs, learn_every_unit="timestep",
        learn_every=10 ** 9, updates_per_learn=1, batch_size=1, warmup_steps=0, seed=0,
    )
    cb = _RecordingCallback()
    trainer = Trainer(agent=agent, env=env, schedule=schedule, buffer=buffer,
                       callbacks=[cb], save_dir=str(tmp_path))

    trainer.train()

    assert len(cb.step_logs) == 3, "test setup invariant: 3 iterations (stop_units=3*num_envs, num_envs=2)"
    for i, log in enumerate(cb.step_logs, start=1):
        expected_env_steps = i * num_envs
        expected_fill = min(expected_env_steps, buffer_size) / buffer_size
        assert log["buffer/env_steps"] == pytest.approx(expected_env_steps)
        assert log["buffer/fill"] == pytest.approx(expected_fill)


# -----------------------------------------------------------------------------
# Interaction with emit_every > 1: buffer/fill and buffer/env_steps are
# registered Reduction.LAST, so a collected window spanning several record()
# calls must report the final value, not e.g. a mean.
# -----------------------------------------------------------------------------
def test_buffer_fill_and_env_steps_use_last_reduction_under_emit_every_greater_than_one():
    mm = MetricsManager(emit_every=2)
    agent = _ScriptedAgent(num_envs=1)
    env = _ScriptedEnv(num_envs=1)
    buf = _AlwaysReadyBuffer(buffer_size=10, samples_added=1)
    mm.bind(agent=agent, env=env, buffer=buf)
    assert "buffer/fill" in mm.enabled_keys() and "buffer/env_steps" in mm.enabled_keys()

    mm.record("buffer/fill", 0.1)
    mm.record("buffer/env_steps", 1.0)
    assert mm.collect() == {}, "emit_every=2: the first collected window must be suppressed"

    mm.record("buffer/fill", 0.9)
    mm.record("buffer/env_steps", 2.0)
    collected = mm.collect()
    assert collected["buffer/fill"] == pytest.approx(0.9), (
        "Reduction.LAST must report the final value recorded across the "
        "2-call window (0.9); a MEAN reduction would report 0.5 instead"
    )
    assert collected["buffer/env_steps"] == pytest.approx(2.0), (
        "Reduction.LAST must report the final value (2.0); a MEAN "
        "reduction would report 1.5 instead"
    )


# -----------------------------------------------------------------------------
# The off_policy capability tag: confirm whether any METRIC_REGISTRY spec
# still requires it.
# -----------------------------------------------------------------------------
def test_off_policy_capability_tag_is_not_required_by_any_registered_metric_spec():
    tags_required = {tag for spec in METRIC_REGISTRY.values() for tag in spec.requires}
    assert "off_policy" not in tags_required, (
        "if this fails, some spec now gates on 'off_policy' and the "
        "surrounding claim that no spec requires it is stale"
    )
