"""Behavior-contract tests for ``phoenx.metrics`` (Phase 1 centralized metrics).

Written PHASE A, from ``.cursor/plans/metrics-manager-phase1.md`` and the pinned
public signatures only -- ``src/phoenx/metrics.py`` did not exist yet and was
never read while writing this file. Every test below is an independent oracle:
either a closed-form value computed in the test, or a behavioral assertion
transcribed directly from the plan's normative text / tables.

Two construction strategies are used, chosen per contract:

    * Direct agent construction (mirrors ``tests/test_agents_composite.py``'s
      head-factory / synthetic-sample pattern): fast, no real env stepping,
      used for capability gating, the no-Trainer NullRecorder default, and the
      frozen-legacy-key-set contract (driving ``agent.learn()`` once and
      feeding its return dict through ``record_many`` + ``collect`` -- the
      dual-emit / unmapped-bucket design guarantees every key ``agent.learn()``
      returns reaches ``collect()`` one way or another, so this is equivalent
      to the same assertion after a full ``Trainer.train()`` loop for this
      specific property, without retuning six different buffer-fill timings).
    * Minimal ``Trainer``/``agent``/``env``/``buffer`` stand-ins (mirrors
      ``tests/test_trainer.py``'s ``_FakeAgent`` / ``_FakeEnv`` pattern):
      used to drive ``Trainer.step`` / ``Trainer.learn`` / ``Trainer.train``
      directly for the termination/truncation, episode-accounting,
      evaluation-isolation, and updates-per-learn-mean contracts.
    * One real, proven-shape ``Trainer.train()`` run (identical proportions to
      ``tests/test_roundtrip.py``'s ``_ppo_config``) for the "collect() only
      ever returns floats" contract, and a construct-save-load (no training)
      run for the config.json backward-compatibility contract.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import textwrap
import weakref
from collections import deque

import numpy as np
import pytest
import torch as T

from phoenx.builder import build_metrics, build_trainer_from_config
from phoenx.env_wrapper import Action, GymnasiumWrapper, Observation
from phoenx.metrics import (
    METRIC_REGISTRY,
    NULL_RECORDER,
    MetricsManager,
    MetricSpec,
    NullRecorder,
    Reduction,
)
from phoenx.models import (
    ContinuousQHead,
    DeterministicActorHead,
    StochasticContinuousHead,
    StochasticDiscreteHead,
    ValueHead,
)
from phoenx.noise import NormalNoise
from phoenx.rl_agents import ActorCritic, DDPG, PPO, SAC, TD3, Reinforce
from phoenx.trainer import Trainer, TrainingSchedule

DEVICE = "cpu"
T.manual_seed(0)

LC = [
    {"type": "dense", "params": {"units": 16, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


# =============================================================================
# Frozen legacy key sets (mandatory members only -- the plan marks some keys
# optional per-agent with parentheses; those are omitted here unless the
# fixture agent below is specifically configured to trigger them).
# =============================================================================
REINFORCE_FROZEN_KEYS = {
    "policy_loss", "value_loss", "advantages", "returns", "entropy",
    "entropy_coefficient", "policy_learning_rate", "value_learning_rate",
    "temperature",  # Reinforce's policy is always StochasticDiscreteHead (categorical)
}
ACTOR_CRITIC_FROZEN_KEYS = REINFORCE_FROZEN_KEYS | {"temporal_difference"}
PPO_FROZEN_KEYS = {
    "policy_loss", "value_loss", "entropy", "kl", "prob_ratio",
    "temporal_difference", "advantages", "returns", "policy_clip", "value_clip",
    "entropy_coefficient", "kl_coefficient", "policy_learning_rate",
    "value_learning_rate", "temperature",
}
DDPG_FROZEN_KEYS = {
    "extrinsic_rewards", "policy_loss", "critic_loss", "td_error",
    "policy_predictions", "critic_predictions", "target_policy_predictions",
    "target_critic_predictions", "policy_learning_rate", "critic_learning_rate",
}
TD3_FROZEN_KEYS = DDPG_FROZEN_KEYS | {"critic_b_learning_rate"}
SAC_FROZEN_KEYS = {
    "extrinsic_rewards", "policy_loss", "critic_loss", "td_error",
    "policy_predictions", "critic_predictions", "target_critic_predictions",
    "entropy_coefficient", "entropy", "policy_learning_rate",
    "critic_learning_rate", "critic_b_learning_rate",
}


# =============================================================================
# Head factories + synthetic buffer-shaped samples
# (mirrors tests/test_agents_composite.py's own house pattern)
# =============================================================================
def _d_policy(env, **kw):
    return StochasticDiscreteHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def _c_policy(env, **kw):
    return StochasticContinuousHead(env, layer_config=LC, output_config=OC,
                                     distribution="normal", device=DEVICE, **kw)


def _actor(env, **kw):
    return DeterministicActorHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


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


def _replay_sample(env, B=16, N=1, discrete=False):
    obs_dim = env.single_observation_space.shape[0]
    if discrete:
        actions = T.randint(0, env.single_action_space.n, (B, N, 1)).float()
    else:
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


# =============================================================================
# Real envs + real agents (module-scoped: cheap architecture-only construction,
# reused read-only-ish across gating / NullRecorder-default / frozen-key tests)
# =============================================================================
@pytest.fixture(scope="module")
def cartpole_env():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def pendulum_env():
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def pendulum_env_nstep():
    """Separate from ``pendulum_env``: off-policy buffers (``ReplayBuffer`` /
    ``PrioritizedReplayBuffer``) require a ``VectorNStepReward``-wrapped env."""
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0,
                            wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}])
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def reinforce_agent(cartpole_env):
    return Reinforce(policy=_d_policy(cartpole_env), value=_value_head(cartpole_env),
                      optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)


@pytest.fixture(scope="module")
def actor_critic_agent(cartpole_env):
    return ActorCritic(policy=_d_policy(cartpole_env), value=_value_head(cartpole_env),
                        optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)


@pytest.fixture(scope="module")
def ppo_agent(cartpole_env):
    return PPO(policy=_d_policy(cartpole_env), value=_value_head(cartpole_env),
               optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)


@pytest.fixture(scope="module")
def ddpg_agent(pendulum_env):
    return DDPG(policy=_actor(pendulum_env), critic=_q_head(pendulum_env),
                noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                optimizer_params=OPT, device=DEVICE)


@pytest.fixture(scope="module")
def td3_agent(pendulum_env):
    return TD3(policy=_actor(pendulum_env), critic=_q_head(pendulum_env),
               noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
               target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE),
               optimizer_params=OPT, device=DEVICE)


@pytest.fixture(scope="module")
def sac_agent(pendulum_env):
    return SAC(policy=_c_policy(pendulum_env), critic=_q_head(pendulum_env),
               optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)


ALL_AGENT_FIXTURES = [
    "reinforce_agent", "actor_critic_agent", "ppo_agent",
    "ddpg_agent", "td3_agent", "sac_agent",
]


# =============================================================================
# Minimal Trainer/agent/env/buffer stand-ins (mirrors tests/test_trainer.py's
# _FakeAgent / _FakeEnv pattern) for the step()/train()-level contracts.
# =============================================================================
class _StubModel:
    """Minimal duck-typed stand-in for the composite ``ModularModel``."""

    def __init__(self, branch_names):
        self.branches = {name: None for name in branch_names}
        self.is_recurrent = False


class _ScriptedFakeAgent:
    """Agent stand-in whose ``act`` is fixed and whose ``learn`` is scriptable."""

    def __init__(self, num_envs=1, learn_returns=None):
        self.device = "cpu"
        self.intrinsic_motivation = None
        self._learn_calls = 0
        self._learn_returns = learn_returns  # callable(call_index) -> dict, or None

    def act(self, states, *_args, **_kwargs):
        n = states.shape[0]
        return Action(actions=T.zeros(n, 1))

    def learn(self, step, sample, **kwargs):
        idx = self._learn_calls
        self._learn_calls += 1
        return self._learn_returns(idx) if self._learn_returns is not None else {}

    def get_config(self):
        return {"type": "FakeAgent"}

    def save_state(self, path):
        import os
        os.makedirs(path, exist_ok=True)


class _ScriptedFakeEnv:
    """Env stand-in whose per-env rewards/terminations/truncations are scripted."""

    def __init__(self, num_envs, rewards, terminations, truncations, goal_key=None):
        self.num_envs = num_envs
        self.goal_key = goal_key
        self.config = {"type": "FakeEnv"}
        self._rewards = rewards
        self._terminations = terminations
        self._truncations = truncations

    def _find_nstep_wrapper(self):
        return None

    def reset(self, seed=None):
        # Only needed by tests that drive a real `_initialize_run` (e.g. the
        # F6 resume regression), as opposed to `_make_stub_trainer`'s
        # pre-seeded-state shortcut, which never calls it.
        return Observation(states=T.zeros(self.num_envs, 4))

    def step(self, _actions):
        return Observation(
            states=T.zeros(self.num_envs, 4),
            rewards=self._rewards,
            terminations=self._terminations,
            truncations=self._truncations,
        )

    def close(self):
        pass


class _RecordingBuffer:
    """Buffer stand-in that records every ``record()`` call's args/kwargs."""

    def __init__(self):
        self.record_calls = []

    def record(self, *args, **kwargs):
        self.record_calls.append((args, kwargs))

    def get_config(self):
        return None


class _LearnableFakeBuffer:
    """Buffer stand-in that is always ready and yields zero-valued samples."""

    def __init__(self):
        self.record_calls = []

    def record(self, *args, **kwargs):
        # Trainer.step() calls this unconditionally when training=True
        # (BASE_REF behavior, unchanged by the metrics feature); a buffer
        # stand-in used to drive step()/train() must implement it.
        self.record_calls.append((args, kwargs))

    def is_ready(self, samples):
        return True

    def sample(self, samples):
        return {"states": T.zeros(samples, 1)}

    def get_config(self):
        return None


class _RecordingCallback:
    """Callback stand-in that records every ``on_train_step_end`` logs dict."""

    def __init__(self):
        self.step_logs = []

    def get_config(self):
        # Trainer.get_config() (called from a real _initialize_run, as
        # opposed to _make_stub_trainer's pre-seeded-state shortcut, which
        # skips it) iterates every attached callback's get_config().
        return {"type": "RecordingCallback"}

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_step_end(self, step, logs):
        self.step_logs.append(dict(logs) if logs is not None else {})

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass


def _make_stub_trainer(env, buffer, tmp_path, agent=None, **schedule_kwargs):
    """Build a real ``Trainer`` with pre-seeded episode-tracking state, exactly
    as ``tests/test_trainer.py``'s ``_make_trainer`` does, so ``step()`` /
    ``learn()`` / ``train()`` can be driven without booting a real env/agent.
    """
    agent = agent if agent is not None else _ScriptedFakeAgent(num_envs=env.num_envs)
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


def _ppo_trainer_config(save_dir: str) -> dict:
    """Tiny PPO trainer config; identical proportions to the proven
    ``tests/test_roundtrip.py::_ppo_config`` (learn_every=16, buffer_size=8,
    num_envs=2, stop_units=96), chosen so buffer-fill/learn-cadence timing is
    known-good rather than retuned from scratch for this file.

    Device follows the machine default (mirroring
    ``tests/test_roundtrip.py::_ppo_config``'s own comment): the ``Trainer``
    allocates its episode counters on ``agent.device`` while ``GymnasiumWrapper``
    follows the machine default, so a hardcoded ``"cpu"`` here would raise a
    cross-device ``RuntimeError`` inside ``Trainer.step()`` on a CUDA machine --
    unrelated to anything under test.
    """
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                                     "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 96, "learn_every_unit": "timestep",
            "learn_every": 16, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 8, "learning_epochs": 2, "warmup_steps": 0, "seed": 7,
        },
        "agent": {"type": "PPO", "config": {
            "name": "PPO",
            "model": {
                "branches": {
                    "policy": {"type": "StochasticDiscreteHead",
                               "layer_config": [dense(16), relu], "output_config": out,
                               "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                               "distribution": "categorical", "device": dev},
                    "value": {"type": "ValueHead",
                              "layer_config": [dense(16), relu], "output_config": out,
                              "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
                },
            },
            "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
            "entropy_coefficient": 0.01, "policy_clip": 0.2, "value_clip": 0.2,
            "policy_grad_clip": 1.0, "value_grad_clip": 1.0, "value_coef": 0.5,
            "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
            "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 7}},
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
    }


def _actor_critic_auto_entropy_trainer_config(save_dir: str) -> dict:
    """Same on-policy CartPole/RolloutBuffer proportions as ``_ppo_trainer_config``
    (proven-good buffer-fill/learn-cadence timing), swapped to ``ActorCritic``
    with ``auto_entropy_tuning: True`` for the F1 end-to-end regression."""
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                                     "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 96, "learn_every_unit": "timestep",
            "learn_every": 16, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 8, "learning_epochs": 2, "warmup_steps": 0, "seed": 7,
        },
        "agent": {"type": "ActorCritic", "config": {
            "name": "ActorCritic",
            "model": {
                "branches": {
                    "policy": {"type": "StochasticDiscreteHead",
                               "layer_config": [dense(16), relu], "output_config": out,
                               "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                               "distribution": "categorical", "device": dev},
                    "value": {"type": "ValueHead",
                              "layer_config": [dense(16), relu], "output_config": out,
                              "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
                },
            },
            "discount": 0.99, "gae_coefficient": 0.95,
            "auto_entropy_tuning": True, "entropy_lr": 1e-3, "target_entropy_scale": 1.0,
            "policy_grad_clip": 1.0, "value_grad_clip": 1.0, "value_coef": 0.5,
            "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
            "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 7}},
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
    }


def _sac_auto_entropy_trainer_config(save_dir: str) -> dict:
    """Tiny off-policy Pendulum/ReplayBuffer config, proportions borrowed from
    ``tests/test_roundtrip.py``'s SAC shrink (``ReplayBuffer``, ``N=1``,
    ``VectorNStepReward``), with ``auto_entropy_tuning: True`` (already the
    default in ``tests/fixtures/sac.yml``) for the F1 end-to-end regression."""
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                                     "kernel_params": {"gain": 1.41421356}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 20, "learn_every_unit": "timestep",
            "learn_every": 1, "updates_per_learn": 1, "batch_size": 4,
            "mini_batch_size": 4, "learning_epochs": 1, "warmup_steps": 0, "seed": 7,
        },
        "agent": {"type": "SAC", "config": {
            "name": "SAC",
            "model": {
                "branches": {
                    "policy": {"type": "StochasticContinuousHead",
                               "layer_config": [dense(16), relu], "output_config": out,
                               "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                               "distribution": "normal", "device": dev},
                    "critic": {"type": "ContinuousQHead",
                               "layer_config": [dense(16), relu], "merged_config": [dense(16), relu],
                               "output_config": out,
                               "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
                },
            },
            "discount": 0.99, "tau": 0.01,
            "auto_entropy_tuning": True, "entropy_lr": 1e-3, "target_entropy_scale": 1.0,
            "policy_grad_clip": 10.0, "critic_grad_clip": 10.0, "critic_huber_delta": 1.0,
            "N": 1, "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": "Pendulum-v1", "num_envs": 1, "obs_key": None, "goal_key": None,
            "ach_goal_key": None,
            "wrappers": [{"type": "VectorNStepReward", "params": {"n": 1}}],
            "render_mode": None, "seed": 7}},
        "buffer": {"type": "ReplayBuffer", "config": {"buffer_size": 100, "N": 1}},
    }


# =============================================================================
# CONTRACT: metrics.py has no import path to rl_callbacks/rl_agents/trainer/
#           buffer/models; `import phoenx` still succeeds.
# =============================================================================
def test_metrics_module_has_no_import_path_to_coupled_modules():
    script = textwrap.dedent(
        """
        import sys
        import phoenx.metrics
        forbidden = {
            "phoenx.rl_callbacks", "phoenx.rl_agents", "phoenx.trainer",
            "phoenx.buffer", "phoenx.models",
        }
        leaked = sorted(m for m in sys.modules if m in forbidden)
        assert not leaked, f"phoenx.metrics pulled in: {leaked}"
        print("METRICS_ISOLATED_OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METRICS_ISOLATED_OK" in result.stdout


def test_importing_phoenx_metrics_then_phoenx_package_still_succeeds():
    script = textwrap.dedent(
        """
        import phoenx.metrics
        import phoenx
        print("PHOENX_IMPORT_OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PHOENX_IMPORT_OK" in result.stdout


# =============================================================================
# CONTRACT: An Agent constructed without a Trainer has a working no-op recorder.
# =============================================================================
@pytest.mark.parametrize("agent_fixture_name", ALL_AGENT_FIXTURES)
def test_agent_without_trainer_has_working_noop_recorder(agent_fixture_name, request):
    agent = request.getfixturevalue(agent_fixture_name)
    agent.metrics.record("x", 1.0)  # must not raise
    assert agent.metrics.is_enabled("x") is False


def test_null_recorder_singleton_is_fully_inert():
    assert isinstance(NULL_RECORDER, NullRecorder)
    NULL_RECORDER.record("x", 1.0)
    NULL_RECORDER.record_many({"x": 1.0, "y": 2.0})
    assert NULL_RECORDER.is_enabled("x") is False
    assert NULL_RECORDER.should_compute("x") is False
    assert NULL_RECORDER.collect() == {}
    NULL_RECORDER.bind(agent=object(), env=object())
    NULL_RECORDER.set_context(env_step=5)
    NULL_RECORDER.reset()


# =============================================================================
# CONTRACT: collect() returns a flat dict of floats; no value is a dict, list,
#           tensor, bool, or None. (Real short train on a proven-shape config.)
# =============================================================================
@pytest.mark.slow
def test_collect_outputs_are_always_plain_floats_never_bool_or_container(tmp_path):
    config = _ppo_trainer_config(str(tmp_path / "run") + os.sep)
    trainer = build_trainer_from_config(config)

    collected_dicts = []
    original_collect = trainer.metrics.collect

    def _spy_collect():
        result = original_collect()
        collected_dicts.append(dict(result))
        return result

    trainer.metrics.collect = _spy_collect
    trainer.train()

    assert collected_dicts, "collect() was never invoked during training"
    assert any(d for d in collected_dicts), "collect() never emitted a non-empty dict"
    for d in collected_dicts:
        for key, value in d.items():
            assert not isinstance(value, bool), f"{key!r} is a bool: {value!r}"
            assert not isinstance(value, (dict, list, tuple, T.Tensor)), (
                f"{key!r} is a container/tensor: {type(value)!r}"
            )
            assert value is not None, f"{key!r} is None"
            assert type(value) in (float, int), f"{key!r} has type {type(value)!r}"
            if key not in ("env_step", "grad_step"):
                assert type(value) is float, f"{key!r} has type {type(value)!r}, expected float"


# =============================================================================
# CONTRACT: record() never retains a tensor that requires grad.
# =============================================================================
def test_record_never_retains_a_tensor_requiring_grad():
    mm = MetricsManager()
    x = T.zeros((), requires_grad=True).exp()
    ref = weakref.ref(x)
    mm.record("policy/entropy", x)
    del x
    gc.collect()
    assert ref() is None, "MetricsManager retained a strong reference to the tensor"
    collected = mm.collect()
    assert type(collected["policy/entropy"]) is float
    assert collected["policy/entropy"] == pytest.approx(1.0)


# =============================================================================
# CONTRACT: A non-0-dim tensor passed to record() or record_many() never
#           reaches collect(), and does not raise.
# =============================================================================
def test_non_0dim_tensor_dropped_by_record_without_raising():
    mm = MetricsManager()
    mm.record("critic/td_error_mean", T.randn(64))  # must not raise
    collected = mm.collect()
    assert "critic/td_error_mean" not in collected


def test_non_0dim_tensor_dropped_by_record_many_without_raising():
    mm = MetricsManager()
    mm.record_many({"td_errors": T.randn(64), "policy_loss": 1.0})  # must not raise
    collected = mm.collect()
    assert "td_errors" not in collected
    assert collected["loss/policy"] == pytest.approx(1.0)


# =============================================================================
# CONTRACT: A metric whose capability or agent gate is unmet is absent from
#           enabled_keys() and from collect().
# =============================================================================
def test_ppo_capability_gating_excludes_offpolicy_and_sac_and_twin_critic_keys(ppo_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    enabled = mm.enabled_keys()
    assert "sac/alpha" not in enabled
    # Bound with buffer=None, so the buffer_capacity gate is unmet. Asserting on
    # a deleted key here would pass no matter what the gate did.
    assert "buffer/fill" not in enabled
    assert "optim/lr/critic_b" not in enabled


def test_sac_capability_gating_excludes_onpolicy_and_ppo_keys(sac_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=sac_agent, env=pendulum_env, buffer=None)
    enabled = mm.enabled_keys()
    assert "ppo/ratio_mean" not in enabled
    assert "policy/kl" not in enabled


def test_td3_capability_gating_includes_shared_ddpg_td3_exploration_key(td3_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=td3_agent, env=pendulum_env, buffer=None)
    assert "exploration/noise_anneal" in mm.enabled_keys()


def test_include_allowlist_cannot_force_a_capability_disabled_key(ppo_agent, cartpole_env):
    """STRENGTHENED (F3): the original assertion (only "sac/alpha" absent)
    cannot fail, because it holds under both an allow-list reading of
    `include` and a reading where `include` merely bypasses the tier gate --
    "sac/alpha" is disabled by the agent-tag gate either way. Pin the actual
    allow-list semantics instead: since `include` REPLACES the tier gate,
    and the bound agent (PPO) satisfies none of `include`'s entries once
    capability/agent gating is applied, `enabled_keys()` must be EMPTY, not
    the ~14-key tier-0 set a tier-bypassing (non-allow-list) `include` would
    still leave enabled."""
    mm = MetricsManager(include=["sac/alpha"])
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    assert mm.enabled_keys() == frozenset()


def test_exclude_removes_an_otherwise_enabled_key_from_enabled_and_collect():
    mm = MetricsManager(exclude=["policy/entropy"])
    assert "policy/entropy" not in mm.enabled_keys()
    mm.record("policy/entropy", 1.0)
    assert "policy/entropy" not in mm.collect()


def test_should_compute_mirrors_is_enabled_in_phase1(ppo_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    for key in ("loss/policy", "sac/alpha", "policy/kl", "policy/entropy"):
        assert mm.should_compute(key) == mm.is_enabled(key)


def test_bind_succeeds_against_a_minimal_partial_stub_agent():
    mm = MetricsManager()
    mm.bind(agent=object(), env=object(), buffer=None)  # must not raise
    mm.enabled_keys()  # must not raise


# =============================================================================
# CONTRACT: A legacy key that aliases a DISABLED spec still reaches collect()
#           under its own name.
# =============================================================================
def test_legacy_alias_of_disabled_spec_still_reaches_collect_under_own_name():
    mm = MetricsManager()
    agent = type("Reinforce", (), {"model": _StubModel({"policy"})})()  # no "value" branch
    mm.bind(agent=agent, env=object(), buffer=None)
    assert "loss/value" not in mm.enabled_keys()

    mm.record_many({"value_loss": 0.5})
    collected = mm.collect()
    assert collected["value_loss"] == pytest.approx(0.5)
    assert "loss/value" not in collected


# =============================================================================
# CONTRACT: Every legacy key currently reaching on_train_step_end still reaches
#           it (frozen key set), driven via a real agent.learn() call.
# =============================================================================
def test_reinforce_frozen_legacy_keys_reach_collect(reinforce_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=reinforce_agent, env=cartpole_env, buffer=None)
    trajectories = [
        {"states": T.randn(8, 4), "actions": T.randint(0, 2, (8, 1)), "rewards": T.randn(8)}
        for _ in range(2)
    ]
    metrics = reinforce_agent.learn(0, trajectories)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = REINFORCE_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_actor_critic_frozen_legacy_keys_reach_collect(actor_critic_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=actor_critic_agent, env=cartpole_env, buffer=None)
    sample = _rollout_sample(cartpole_env, discrete=True)
    metrics = actor_critic_agent.learn(0, sample)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = ACTOR_CRITIC_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_ppo_frozen_legacy_keys_reach_collect(ppo_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    sample = _rollout_sample(cartpole_env, discrete=True)
    metrics = ppo_agent.learn(0, sample, learning_epochs=2, mini_batch_size=4)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = PPO_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_ddpg_frozen_legacy_keys_reach_collect(ddpg_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=ddpg_agent, env=pendulum_env, buffer=None)
    sample = _replay_sample(pendulum_env, B=16, N=1)
    metrics = ddpg_agent.learn(0, sample)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = DDPG_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_td3_frozen_legacy_keys_reach_collect(td3_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=td3_agent, env=pendulum_env, buffer=None)
    sample = _replay_sample(pendulum_env, B=16, N=1)
    metrics = td3_agent.learn(0, sample)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = TD3_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_sac_frozen_legacy_keys_reach_collect(sac_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=sac_agent, env=pendulum_env, buffer=None)
    sample = _replay_sample(pendulum_env, B=16, N=1)
    metrics = sac_agent.learn(0, sample)
    mm.record_many(metrics)
    collected = mm.collect()
    missing = SAC_FROZEN_KEYS - collected.keys()
    assert not missing, f"missing frozen keys: {missing}"


def test_td_errors_never_reaches_collect_even_when_present_in_learn_output():
    """``_apply_per_update`` pops ``td_errors`` before any logger sees it; a
    bare MetricsManager fed it directly (as if that pop were skipped) must
    still never expose it, since it is a non-0-dim per-sample tensor."""
    mm = MetricsManager()
    mm.record_many({"td_errors": T.randn(32), "td_error": 0.1})
    collected = mm.collect()
    assert "td_errors" not in collected
    assert collected["td_error"] == pytest.approx(0.1)


# =============================================================================
# CONTRACT: With updates_per_learn > 1, the value delivered TO THE CALLBACK is
#           the mean over all updates, not the last update's value.
# =============================================================================
def test_updates_per_learn_mean_delivered_to_collect_and_callback(tmp_path):
    call_indices = []

    def _learn_returns(idx):
        call_indices.append(idx)
        return {"policy_loss": float(idx)}

    agent = _ScriptedFakeAgent(num_envs=1, learn_returns=_learn_returns)
    env = _ScriptedFakeEnv(num_envs=1, rewards=T.tensor([0.0]),
                            terminations=T.tensor([False]), truncations=T.tensor([False]))
    buffer = _LearnableFakeBuffer()
    trainer = _make_stub_trainer(
        env, buffer, tmp_path, agent=agent,
        stop_unit="timestep", stop_units=1, learn_every_unit="timestep", learn_every=1,
        updates_per_learn=4, batch_size=1, warmup_steps=0,
    )
    trainer._initialized = True  # skip real _initialize_run; state above is pre-seeded
    cb = _RecordingCallback()
    trainer.callbacks = [cb]

    trainer.train()

    assert call_indices == [0, 1, 2, 3]
    assert len(cb.step_logs) == 1
    logs = cb.step_logs[0]
    assert logs["loss/policy"] == pytest.approx(1.5)
    assert logs["policy_loss"] == pytest.approx(1.5)


# =============================================================================
# CONTRACT: Termination and truncation are recorded distinctly; a step with
#           both flags counts as a termination only.
# =============================================================================
def test_termination_and_truncation_recorded_distinctly_across_two_envs(tmp_path):
    env = _ScriptedFakeEnv(
        num_envs=2, rewards=T.zeros(2),
        terminations=T.tensor([True, False]), truncations=T.tensor([False, True]),
    )
    trainer = _make_stub_trainer(env, _RecordingBuffer(), tmp_path,
                                  stop_unit="timestep", stop_units=1)
    trainer.step(training=True)
    collected = trainer.metrics.collect()
    assert collected["rollout/termination_rate"] == pytest.approx(0.5)
    assert collected["rollout/truncation_rate"] == pytest.approx(0.5)


def test_termination_and_truncation_both_flags_counts_as_termination_only(tmp_path):
    env = _ScriptedFakeEnv(
        num_envs=1, rewards=T.zeros(1),
        terminations=T.tensor([True]), truncations=T.tensor([True]),
    )
    trainer = _make_stub_trainer(env, _RecordingBuffer(), tmp_path,
                                  stop_unit="timestep", stop_units=1)
    trainer.step(training=True)
    collected = trainer.metrics.collect()
    assert collected["rollout/termination_rate"] == pytest.approx(1.0)
    assert collected["rollout/truncation_rate"] == pytest.approx(0.0)


# =============================================================================
# CONTRACT: Episode-completion control flow is unchanged by the
#           termination/truncation split.
# =============================================================================
def test_episode_completion_control_flow_unchanged_for_truncation_only(tmp_path):
    buffer = _RecordingBuffer()
    env = _ScriptedFakeEnv(
        num_envs=1, rewards=T.tensor([2.0]),
        terminations=T.tensor([False]), truncations=T.tensor([True]),
    )
    trainer = _make_stub_trainer(env, buffer, tmp_path, stop_unit="timestep", stop_units=1)

    result1 = trainer.step(training=True)
    assert int(trainer._completed_episodes[0].item()) == 1
    assert bool(trainer._prev_done[0].item()) is True
    assert len(trainer._score_history) == 1
    assert result1["episode_logs"] != []

    # Second step: env no longer terminates/truncates. The buffer.record call
    # for THIS step must have received prev_dones reflecting env 0's
    # truncated-from-step-1 done state.
    env._terminations = T.tensor([False])
    env._truncations = T.tensor([False])
    trainer.step(training=True)

    _, kwargs = buffer.record_calls[-1]
    prev_dones = kwargs["prev_dones"]
    assert bool(prev_dones[0]) is True


# =============================================================================
# CONTRACT: Evaluation steps do not contribute to the next training window and
#           do not grow manager state without bound.
#
# VERIFY (amended -- see plan): part (b) must respect the pre-existing
# `valid_steps = ~self._prev_done` mask. A stub env that finishes an episode
# on EVERY eval step leaves `_prev_done` True, which makes the immediately
# following training step a phantom step whose reward is deliberately not
# accumulated into `_episode_scores` -- that is BASE_REF episode-accounting
# behavior, not a metrics-system property, and must not be asserted against.
# So the eval phase here ends with one eval step that does NOT finish an
# episode, leaving `_prev_done` False before the training phase begins.
# =============================================================================
def test_evaluation_steps_do_not_contribute_to_next_training_window(tmp_path):
    env = _ScriptedFakeEnv(
        num_envs=1, rewards=T.tensor([999.0]),
        terminations=T.tensor([True]), truncations=T.tensor([False]),
    )
    trainer = _make_stub_trainer(env, _RecordingBuffer(), tmp_path,
                                  stop_unit="timestep", stop_units=1)

    # (a) Eval steps that finish episodes with a sentinel return; the LAST
    # one deliberately does NOT finish an episode so _prev_done ends False.
    for is_last_eval_step in (False, False, True):
        if is_last_eval_step:
            env._terminations = T.tensor([False])
            env._truncations = T.tensor([False])
        result = trainer.step(training=False)
        for log in result["episode_logs"]:
            trainer._episode_scores[log["env"]] = 0
            trainer._episode_steps[log["env"]] = 0

    collected = trainer.metrics.collect()
    assert not any(k.startswith("rollout/") for k in collected), (
        f"eval steps leaked rollout keys: {collected}"
    )
    assert bool(trainer._prev_done[0].item()) is False, (
        "test setup invariant violated: the last eval step must not finish "
        "an episode, or the training step below would be a phantom step"
    )

    # (b) One complete training episode from this clean, non-phantom state;
    # its real return must reach collect(), not the eval sentinel.
    env._rewards = T.tensor([7.0])
    env._terminations = T.tensor([True])
    result = trainer.step(training=True)
    assert result["episode_logs"], "training step did not finish an episode"
    for log in result["episode_logs"]:
        trainer._episode_scores[log["env"]] = 0

    collected = trainer.metrics.collect()
    assert collected.get("rollout/return_mean") == pytest.approx(7.0)


# =============================================================================
# CONTRACT: Suppressed collects retain window state.
# =============================================================================
def test_suppressed_collects_retain_window_state_until_emit_every_elapses():
    mm = MetricsManager(emit_every=3)
    mm.record("policy/entropy", 1.0)
    assert mm.collect() == {}
    mm.record("policy/entropy", 2.0)
    assert mm.collect() == {}
    mm.record("policy/entropy", 3.0)
    collected = mm.collect()
    assert collected["policy/entropy"] == pytest.approx(2.0)
    assert collected["entropy"] == pytest.approx(2.0)  # dual-emit alias


# =============================================================================
# CONTRACT: get_config() is bind-independent and JSON-round-trips.
# =============================================================================
def test_get_config_matches_pinned_default_schema_shape():
    mm = MetricsManager()
    assert mm.get_config() == {
        "tiers": [0, 1, 2], "emit_every": 1, "include": None, "exclude": [], "dual_emit": True,
    }


def test_get_config_bind_independent_and_json_round_trips():
    mm = MetricsManager(tiers=(1, 0), emit_every=3, include=["policy/entropy"],
                         exclude=["health/non_finite"], dual_emit=False)
    before = mm.get_config()

    stub_agent = type("PPO", (), {})()
    stub_env = type("Env", (), {})()
    mm.bind(agent=stub_agent, env=stub_env, buffer=None)
    assert mm.get_config() == before

    assert json.loads(json.dumps(before)) == before

    rebuilt = MetricsManager.load(before)
    assert rebuilt.get_config() == before


def test_load_rejects_unknown_config_keys():
    with pytest.raises(TypeError):
        MetricsManager.load({"tiers": [0], "bogus_kwarg": 1})


# =============================================================================
# CONTRACT: A config.json written before this change still loads.
# =============================================================================
def test_trainer_load_without_metrics_key_uses_default_manager(tmp_path):
    run_dir = tmp_path / "run"
    config = _ppo_trainer_config(str(run_dir) + os.sep)
    trainer = build_trainer_from_config(config)
    trainer.save()

    config_path = run_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert "metrics" in saved, "get_config() did not gain a 'metrics' key"
    del saved["metrics"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved, f)

    loaded = Trainer.load(str(run_dir))
    assert isinstance(loaded.metrics, MetricsManager)
    assert loaded.metrics.get_config() == MetricsManager().get_config()


def test_trainer_always_sets_a_non_none_metrics_manager(tmp_path):
    config = _ppo_trainer_config(str(tmp_path / "run") + os.sep)
    trainer = build_trainer_from_config(config)
    assert trainer.metrics is not None
    assert isinstance(trainer.metrics, MetricsManager)


# =============================================================================
# CONTRACT: METRIC_REGISTRY is not mutated by bind().
# =============================================================================
def test_metric_registry_not_mutated_by_binding_two_different_agents():
    snapshot_keys = dict(METRIC_REGISTRY)
    identities = {k: id(v) for k, v in METRIC_REGISTRY.items()}

    agent_a = type("PPO", (), {"model": _StubModel({"policy", "value"})})()
    agent_b = type("SAC", (), {"model": _StubModel({"policy", "critic", "critic_b"})})()
    mm_a, mm_b = MetricsManager(), MetricsManager()
    mm_a.bind(agent=agent_a, env=object(), buffer=None)
    mm_b.bind(agent=agent_b, env=object(), buffer=None)
    mm_a.enabled_keys()
    mm_b.enabled_keys()

    assert dict(METRIC_REGISTRY) == snapshot_keys
    assert {k: id(v) for k, v in METRIC_REGISTRY.items()} == identities


# =============================================================================
# CONTRACT: WINDOW_STD and RATE keys are absent rather than NaN when
#           underdetermined.
# =============================================================================
def test_window_std_absent_with_single_value_but_mean_present():
    mm = MetricsManager()
    mm.record("rollout/return_mean", 5.0)
    mm.record("rollout/return_std", 5.0)
    collected = mm.collect()
    assert "rollout/return_std" not in collected
    assert collected["rollout/return_mean"] == pytest.approx(5.0)


def test_window_std_present_with_two_or_more_values():
    mm = MetricsManager()
    for v in (2.0, 4.0):
        mm.record("rollout/return_std", v)
    collected = mm.collect()
    assert "rollout/return_std" in collected
    assert collected["rollout/return_std"] == pytest.approx(1.0)  # population std, ddof=0


def test_rate_absent_on_first_emitting_collect():
    mm = MetricsManager()
    mm.record("perf/sps", 100.0)
    collected = mm.collect()
    assert "perf/sps" not in collected


# =============================================================================
# Bonus coverage implied (but not separately stated) by the normative text:
# record() vs record_many()'s differing treatment of unrecognized keys, the
# reset()/collect() window-vs-persistent-state distinction, the pinned
# Reduction/MetricSpec/METRIC_REGISTRY shapes, and build_metrics().
# =============================================================================
def test_record_on_a_key_outside_enabled_keys_is_a_silent_noop():
    mm = MetricsManager()
    mm.record("sac/alpha", 1.0)  # disabled pre-bind: requires an agent tag
    assert "sac/alpha" not in mm.collect()


def test_record_does_not_perform_alias_or_unmapped_resolution():
    mm = MetricsManager()
    mm.record("totally_bogus_key_zzz", 5.0)  # not a registered spec key
    assert "totally_bogus_key_zzz" not in mm.collect()


def test_record_many_places_unrecognized_key_in_unmapped_bucket():
    mm = MetricsManager()
    mm.record_many({"totally_bogus_key_zzz": 5.0})
    assert mm.collect()["totally_bogus_key_zzz"] == pytest.approx(5.0)


def test_record_many_alias_resolution_prefers_exact_key_over_alias():
    """record_many's algorithm checks 'k equals an enabled spec's key' BEFORE
    'k is an alias of exactly one enabled spec'; recording under the new key
    name directly must not be treated as an unmapped/alias name."""
    mm = MetricsManager()
    mm.record_many({"loss/policy": 3.0})
    collected = mm.collect()
    assert collected["loss/policy"] == pytest.approx(3.0)
    assert collected["policy_loss"] == pytest.approx(3.0)  # dual-emit


def _ppo_like_stub():
    return type("PPO", (), {"model": _StubModel({"policy", "value"})})()


def _sac_like_stub():
    return type("SAC", (), {"model": _StubModel({"policy", "critic", "critic_b"})})()


def test_record_many_disambiguates_entropy_coefficient_by_bound_agent_tag():
    """sac/alpha and onpolicy/entropy_coef both alias entropy_coefficient but
    are never simultaneously enabled (agent:sac vs on_policy are disjoint)."""
    mm_ppo = MetricsManager()
    mm_ppo.bind(agent=_ppo_like_stub(), env=object(), buffer=None)
    mm_ppo.record_many({"entropy_coefficient": 0.3})
    collected_ppo = mm_ppo.collect()
    assert collected_ppo.get("onpolicy/entropy_coef") == pytest.approx(0.3)
    assert "sac/alpha" not in collected_ppo

    mm_sac = MetricsManager()
    mm_sac.bind(agent=_sac_like_stub(), env=object(), buffer=None)
    mm_sac.record_many({"entropy_coefficient": 0.3})
    collected_sac = mm_sac.collect()
    assert collected_sac.get("sac/alpha") == pytest.approx(0.3)
    assert "onpolicy/entropy_coef" not in collected_sac


def test_collect_preserves_rolling_deque_across_calls_but_reset_clears_it():
    mm = MetricsManager()
    mm.record("rollout/return_rolling_100", 3.0)
    first = mm.collect()
    assert first["rollout/return_rolling_100"] == pytest.approx(3.0)

    # Nothing recorded this window; ROLLING still emits from the preserved deque.
    second = mm.collect()
    assert second.get("rollout/return_rolling_100") == pytest.approx(3.0)

    mm.reset()
    third = mm.collect()
    assert "rollout/return_rolling_100" not in third


def test_rate_anchor_preserved_across_collects_but_reset_clears_it():
    mm = MetricsManager()
    mm.record("perf/sps", 0.0)
    first = mm.collect()
    assert "perf/sps" not in first  # no prior anchor

    mm.record("perf/sps", 10.0)
    second = mm.collect()
    assert "perf/sps" in second

    mm.reset()
    mm.record("perf/sps", 20.0)
    third = mm.collect()
    assert "perf/sps" not in third  # anchor cleared by reset()


def test_reduction_enum_has_exactly_the_pinned_members():
    assert {member.value for member in Reduction} == {
        "mean", "last", "first", "max", "min", "sum", "rate", "rolling", "window_std",
        "rolling_std",
    }


def test_metric_spec_is_frozen_with_pinned_defaults():
    spec = MetricSpec(key="x/y", reduction=Reduction.MEAN)
    assert spec.key == "x/y"
    assert spec.reduction == Reduction.MEAN
    assert spec.requires == frozenset()
    assert spec.agents == frozenset()
    assert spec.tier == 0
    assert spec.window == 100
    assert spec.aliases == ()
    with pytest.raises(Exception):
        spec.key = "other"  # frozen dataclass


@pytest.mark.parametrize("key,expected_reduction,expected_aliases,expected_requires", [
    ("rollout/return_mean", Reduction.MEAN, frozenset(), frozenset()),
    ("rollout/return_rolling_100", Reduction.ROLLING, frozenset(), frozenset()),
    ("rollout/return_std", Reduction.ROLLING_STD, frozenset(), frozenset()),
    ("rollout/ep_length_mean", Reduction.MEAN, frozenset(), frozenset()),
    ("rollout/ep_length_std", Reduction.ROLLING_STD, frozenset(), frozenset()),
    ("rollout/termination_rate", Reduction.MEAN, frozenset(), frozenset()),
    ("rollout/truncation_rate", Reduction.MEAN, frozenset(), frozenset()),
    ("perf/sps", Reduction.RATE, frozenset(), frozenset()),
    ("buffer/fill", Reduction.LAST, frozenset(), frozenset({"buffer_capacity"})),
    ("buffer/env_steps", Reduction.LAST, frozenset(), frozenset({"buffer_capacity"})),
    ("buffer/is_ready", Reduction.LAST, frozenset(), frozenset({"buffer_capacity"})),
    ("buffer/pending_trajectories", Reduction.LAST, frozenset(), frozenset({"buffer_trajectories"})),
    ("optim/updates_per_env_step", Reduction.LAST, frozenset(), frozenset({"optimizer_steps"})),
    ("health/non_finite", Reduction.MAX, frozenset(), frozenset()),
    ("per/beta", Reduction.LAST, frozenset({"PER/beta"}), frozenset({"per"})),
    ("per/sampled_priority_mean", Reduction.MEAN, frozenset({"PER/sampled_priority_mean"}), frozenset({"per"})),
    ("per/is_weight_max", Reduction.MAX, frozenset(), frozenset({"per"})),
    ("per/priority_entropy", Reduction.MEAN, frozenset(), frozenset({"per"})),
    ("loss/policy", Reduction.MEAN, frozenset({"policy_loss"}), frozenset()),
    ("loss/value", Reduction.MEAN, frozenset({"value_loss"}), frozenset({"has_value"})),
    ("loss/critic", Reduction.MEAN, frozenset({"critic_loss"}), frozenset({"has_critic"})),
    ("critic/td_error_mean", Reduction.MEAN, frozenset({"temporal_difference", "td_error"}), frozenset()),
    ("critic/value_pred_mean", Reduction.MEAN, frozenset({"critic_predictions"}), frozenset({"has_critic"})),
    ("critic/return_mean_batch", Reduction.MEAN, frozenset({"returns"}), frozenset()),
    ("policy/advantage_mean", Reduction.MEAN, frozenset({"advantages"}), frozenset({"on_policy"})),
    ("policy/entropy", Reduction.MEAN, frozenset({"entropy"}), frozenset()),
    ("policy/kl", Reduction.MEAN, frozenset({"kl"}), frozenset({"on_policy"})),
    ("ppo/ratio_mean", Reduction.MEAN, frozenset({"prob_ratio"}), frozenset()),
    ("ppo/policy_clip", Reduction.LAST, frozenset({"policy_clip"}), frozenset()),
    ("ppo/value_clip", Reduction.LAST, frozenset({"value_clip"}), frozenset()),
    ("onpolicy/entropy_coef", Reduction.LAST, frozenset({"entropy_coefficient"}), frozenset({"on_policy"})),
    ("onpolicy/kl_coef", Reduction.LAST, frozenset({"kl_coefficient"}), frozenset()),
    ("optim/lr/policy", Reduction.LAST, frozenset({"policy_learning_rate"}), frozenset()),
    ("optim/lr/value", Reduction.LAST, frozenset({"value_learning_rate"}), frozenset({"has_value"})),
    ("optim/lr/critic", Reduction.LAST, frozenset({"critic_learning_rate"}), frozenset({"has_critic"})),
    ("optim/lr/critic_b", Reduction.LAST, frozenset({"critic_b_learning_rate"}), frozenset({"twin_critic"})),
    ("sac/alpha", Reduction.LAST, frozenset({"entropy_coefficient"}), frozenset()),
    ("exploration/noise_anneal", Reduction.LAST, frozenset({"noise_anneal"}), frozenset()),
    ("td3/target_noise_anneal", Reduction.LAST, frozenset({"target_noise_anneal"}), frozenset()),
    ("policy/temperature", Reduction.LAST, frozenset({"temperature"}), frozenset({"discrete"})),
])
def test_registry_entry_matches_pinned_table(key, expected_reduction, expected_aliases, expected_requires):
    assert key in METRIC_REGISTRY, f"{key!r} missing from METRIC_REGISTRY"
    spec = METRIC_REGISTRY[key]
    assert spec.key == key
    assert spec.reduction == expected_reduction
    assert frozenset(spec.aliases) == expected_aliases
    assert frozenset(spec.requires) == expected_requires


def test_build_metrics_absent_block_returns_default_manager():
    mm = build_metrics({"agent": {}, "env": {}})
    assert mm.get_config() == MetricsManager().get_config()


def test_build_metrics_wires_constructor_kwargs_from_top_level_metrics_block():
    mm = build_metrics({"metrics": {
        "tiers": [0, 1], "emit_every": 5, "include": ["policy/entropy"],
        "exclude": ["health/non_finite"], "dual_emit": False,
    }})
    cfg = mm.get_config()
    assert cfg["tiers"] == [0, 1]
    assert cfg["emit_every"] == 5
    assert cfg["include"] == ["policy/entropy"]
    assert cfg["exclude"] == ["health/non_finite"]
    assert cfg["dual_emit"] is False


# =============================================================================
# PHASE B -- supplementary coverage, written after reading
# src/phoenx/metrics.py, src/phoenx/trainer.py, src/phoenx/rl_agents.py, and
# src/phoenx/builder.py. Targets internal branches and cross-module
# integration Phase A could not anticipate from the contracts alone.
# =============================================================================

# --------------------------------------------------------------------------- #
# 1. _coerce() value-type edge cases
# --------------------------------------------------------------------------- #
def test_coerce_drops_none_value_without_raising():
    mm = MetricsManager()
    mm.record("policy/entropy", None)
    assert "policy/entropy" not in mm.collect()


def test_coerce_drops_non_numeric_string():
    mm = MetricsManager()
    mm.record("policy/entropy", "not-a-number")
    assert "policy/entropy" not in mm.collect()


def test_coerce_accepts_a_numeric_string():
    mm = MetricsManager()
    mm.record("policy/entropy", "3.5")
    assert mm.collect()["policy/entropy"] == pytest.approx(3.5)


def test_coerce_accepts_a_numpy_scalar():
    mm = MetricsManager()
    mm.record("policy/entropy", np.float32(2.5))
    collected = mm.collect()
    assert type(collected["policy/entropy"]) is float
    assert collected["policy/entropy"] == pytest.approx(2.5)


def test_coerce_drops_nan_and_inf_for_a_mean_key():
    mm = MetricsManager()
    mm.record("rollout/return_mean", float("nan"))
    mm.record("rollout/return_mean", float("inf"))
    assert "rollout/return_mean" not in mm.collect()


def test_coerce_drops_single_element_nondim0_tensor():
    """AMENDED (F1): the original assertion pinned the pre-amendment reading
    of contract #2 (drop every non-0-dim tensor, independent of element
    count). That reading is what made ``entropy_coefficient`` -- shape
    ``(1,)`` from ``setup_auto_entropy``'s ``log_alpha.exp()`` -- silently
    vanish for every ``auto_entropy_tuning`` run. Rule 1 (amended) now
    coerces ANY tensor with ``numel() == 1``, regardless of shape, via
    ``float(value.detach().reshape(()).item())``; only ``numel() != 1``
    tensors are still dropped (see the follow-up test below)."""
    mm = MetricsManager()
    mm.record("policy/entropy", T.tensor([1.0]))
    collected = mm.collect()
    assert collected["policy/entropy"] == pytest.approx(1.0)
    assert type(collected["policy/entropy"]) is float


def test_coerce_drops_a_genuinely_multi_element_tensor():
    """Companion to the amended test above: tensors with ``numel() != 1``
    (as opposed to merely ``dim() != 0``) remain dropped -- e.g. the
    per-sample ``td_errors`` tensor must never reach ``collect()``."""
    mm = MetricsManager()
    mm.record("policy/entropy", T.randn(64))
    assert "policy/entropy" not in mm.collect()

    mm2 = MetricsManager()
    mm2.record("policy/entropy", T.zeros(2))
    assert "policy/entropy" not in mm2.collect()


@pytest.mark.skipif(not T.cuda.is_available(), reason="requires a CUDA device")
def test_coerce_accepts_0dim_tensor_on_a_non_default_device():
    mm = MetricsManager()
    mm.record("policy/entropy", T.tensor(4.0, device="cuda"))
    collected = mm.collect()
    assert type(collected["policy/entropy"]) is float
    assert collected["policy/entropy"] == pytest.approx(4.0)


def test_coerce_via_record_many_unmapped_bucket_same_rules_apply():
    """The unmapped-bucket path in ``record_many`` is documented to be
    "subject to the same value coercion and dropping rules as ``record()``";
    confirm a non-0-dim tensor is dropped there too, not just for enabled specs."""
    mm = MetricsManager()
    mm.record_many({"some_custom_stat": T.randn(8), "another_custom_stat": float("nan")})
    collected = mm.collect()
    assert "some_custom_stat" not in collected
    assert "another_custom_stat" not in collected


# --------------------------------------------------------------------------- #
# 2. record_many alias resolution: both-present and forced ambiguity
# --------------------------------------------------------------------------- #
def test_record_many_when_canonical_key_and_its_alias_both_appear_together():
    """Both entries resolve to the same enabled spec's window (branch 1 for
    the canonical key, branch 2 for the alias), so MEAN averages across both
    -- an independently-computed expected value, not a self-check."""
    mm = MetricsManager()
    mm.record_many({"loss/policy": 1.0, "policy_loss": 5.0})
    collected = mm.collect()
    assert collected["loss/policy"] == pytest.approx(3.0)
    assert collected["policy_loss"] == pytest.approx(3.0)


def test_record_many_ambiguous_alias_falls_through_to_unmapped_bucket():
    """``entropy_coefficient`` aliases both ``onpolicy/entropy_coef`` and
    ``sac/alpha``, but the plan guarantees (Table B note) they are never
    simultaneously enabled since ``agent:sac`` and ``on_policy`` are disjoint
    capability tags -- so no real ``bind()`` call can reach the
    ``len(candidates) != 1`` fallback the algorithm's step 3 pins. Force it
    via direct private-state override (there is no legitimate agent/env
    combination that produces this state) purely to exercise that branch."""
    mm = MetricsManager()
    mm._enabled_keys = frozenset({"onpolicy/entropy_coef", "sac/alpha"})
    mm.record_many({"entropy_coefficient": 0.7})
    collected = mm.collect()
    assert collected.get("entropy_coefficient") == pytest.approx(0.7)
    assert "onpolicy/entropy_coef" not in collected
    assert "sac/alpha" not in collected


# --------------------------------------------------------------------------- #
# 3. Every Reduction member's arithmetic, and health/non_finite transitions
# --------------------------------------------------------------------------- #
def test_reduce_first_min_sum_max_arithmetic_via_synthetic_spec():
    """No Phase 1 registry spec uses FIRST, MIN, or SUM (confirmed by
    inspecting every entry in ``METRIC_REGISTRY``), so their arithmetic is
    otherwise unreachable through the public record()/collect() surface with
    a real key. Exercise ``_reduce`` directly against a synthetic spec/window;
    MEAN/LAST/ROLLING/WINDOW_STD/RATE are already covered via real registry
    keys elsewhere in this file."""
    mm = MetricsManager()
    mm._window_values["synthetic/x"] = [5.0, 1.0, 9.0, 3.0]
    assert mm._reduce(MetricSpec(key="synthetic/x", reduction=Reduction.FIRST)) == pytest.approx(5.0)
    assert mm._reduce(MetricSpec(key="synthetic/x", reduction=Reduction.MIN)) == pytest.approx(1.0)
    assert mm._reduce(MetricSpec(key="synthetic/x", reduction=Reduction.SUM)) == pytest.approx(18.0)
    assert mm._reduce(MetricSpec(key="synthetic/x", reduction=Reduction.MAX)) == pytest.approx(9.0)


def test_health_non_finite_absent_then_zero_then_one():
    """Table A: absent with no loss ingested this window, 0.0 after a finite
    loss, 1.0 after a non-finite one -- each collect() call starts a fresh
    window for this MAX-reduced key (only ROLLING/RATE state persists)."""
    mm = MetricsManager()
    assert "health/non_finite" not in mm.collect()

    mm.record("loss/policy", 1.0)
    collected = mm.collect()
    assert collected["health/non_finite"] == pytest.approx(0.0)

    mm.record("loss/policy", float("nan"))
    collected = mm.collect()
    assert collected["health/non_finite"] == pytest.approx(1.0)


def test_health_non_finite_max_over_a_mixed_window():
    """One non-finite loss among several finite ones this window still MAXes
    to 1.0 (not overwritten back to 0.0 by a later finite record), and the
    dropped non-finite value never pollutes the MEAN window for its own key."""
    mm = MetricsManager()
    mm.record("loss/policy", 1.0)
    mm.record("loss/policy", float("nan"))
    mm.record("loss/policy", 2.0)
    collected = mm.collect()
    assert collected["health/non_finite"] == pytest.approx(1.0)
    assert collected["loss/policy"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# 4. bind() capability sets for each of the six real agents, exactly
# --------------------------------------------------------------------------- #
def test_bind_capabilities_exact_for_reinforce(reinforce_agent, cartpole_env):
    # buffer=None: bool(getattr(None, "buffer_size", 0)) is False (short-circuits
    # buffer_capacity) and hasattr(None, "completed_trajectories") is False
    # (buffer_trajectories), so neither buffer tag is added regardless of agent.
    # optimizer_steps is added for every agent below: hasattr(agent.model,
    # "optimizer_steps") is True for any real ModularModel, independent of
    # on/off-policy status.
    mm = MetricsManager()
    mm.bind(agent=reinforce_agent, env=cartpole_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:reinforce", "on_policy", "has_value", "discrete", "optimizer_steps"}
    )


def test_bind_capabilities_exact_for_actor_critic(actor_critic_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=actor_critic_agent, env=cartpole_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:actorcritic", "on_policy", "has_value", "discrete", "optimizer_steps"}
    )


def test_bind_capabilities_exact_for_ppo(ppo_agent, cartpole_env):
    mm = MetricsManager()
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:ppo", "on_policy", "has_value", "discrete", "optimizer_steps"}
    )


def test_bind_capabilities_exact_for_ddpg(ddpg_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=ddpg_agent, env=pendulum_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:ddpg", "off_policy", "has_critic", "continuous", "optimizer_steps"}
    )


def test_bind_capabilities_exact_for_td3(td3_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=td3_agent, env=pendulum_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:td3", "off_policy", "has_critic", "twin_critic", "continuous", "optimizer_steps"}
    )


def test_bind_capabilities_exact_for_sac(sac_agent, pendulum_env):
    mm = MetricsManager()
    mm.bind(agent=sac_agent, env=pendulum_env, buffer=None)
    assert mm.capabilities() == frozenset(
        {"agent:sac", "off_policy", "has_critic", "twin_critic", "continuous", "optimizer_steps"}
    )


def test_bind_optim_lr_critic_b_enabled_for_both_twin_critic_agents(td3_agent, sac_agent, pendulum_env):
    """``optim/lr/critic_b`` requires ``twin_critic``, not ``agent:td3`` --
    SAC also auto-clones a second critic, so it must be enabled there too
    (an AND-vs-OR gate regression in the same spirit as the TD3
    ``exploration/noise_anneal`` contract, but for a `requires` tag shared by
    two different agent classes rather than an `agents` allow-list)."""
    mm_td3 = MetricsManager()
    mm_td3.bind(agent=td3_agent, env=pendulum_env, buffer=None)
    assert "optim/lr/critic_b" in mm_td3.enabled_keys()

    mm_sac = MetricsManager()
    mm_sac.bind(agent=sac_agent, env=pendulum_env, buffer=None)
    assert "optim/lr/critic_b" in mm_sac.enabled_keys()


# --------------------------------------------------------------------------- #
# 5. Table A2 PER path end-to-end: real PrioritizedReplayBuffer
# --------------------------------------------------------------------------- #
def _make_per_buffer(env, buffer_size, n_transitions, priority="proportional"):
    """Build a real ``PrioritizedReplayBuffer`` seeded with ``n_transitions``
    synthetic transitions via ``add()`` directly (bypassing the env-driven
    ``VectorNStepReward``/``record()`` path, which needs a live rollout)."""
    from phoenx.buffer import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(env=env, buffer_size=buffer_size, priority=priority, N=1)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    buf.add(
        states=T.randn(n_transitions, 1, obs_dim),
        actions=T.rand(n_transitions, 1, act_dim) * 2 - 1,
        rewards=T.randn(n_transitions, 1),
        next_states=T.randn(n_transitions, 1, obs_dim),
        terminations=T.zeros(n_transitions, 1, dtype=T.bool),
        truncations=T.zeros(n_transitions, 1, dtype=T.bool),
        trajectory_lengths=T.ones(n_transitions, dtype=T.long),
    )
    return buf


@pytest.mark.slow
def test_per_table_a2_all_new_and_legacy_keys_reach_collect(pendulum_env_nstep, ddpg_agent, tmp_path):
    buffer = _make_per_buffer(pendulum_env_nstep, buffer_size=16, n_transitions=8)
    trainer = _make_stub_trainer(pendulum_env_nstep, buffer, tmp_path, agent=ddpg_agent)
    trainer.metrics.bind(agent=ddpg_agent, env=pendulum_env_nstep, buffer=buffer)
    assert "per" in trainer.metrics.capabilities()

    sample = buffer.sample(4)
    learn_metrics = {"policy_loss": 1.0, "td_errors": T.rand(4)}
    trainer._apply_per_update(sample, learn_metrics)
    assert "td_errors" not in learn_metrics, "_apply_per_update must pop td_errors before any logger sees it"

    collected = trainer.metrics.collect()
    new_keys = {
        "per/beta", "per/sampled_priority_mean", "per/sampled_priority_max",
        "per/buffer_priority_mean", "per/buffer_priority_max", "per/is_weight_mean",
        "per/is_weight_std", "per/prob_mean", "per/prob_max", "per/is_weight_max",
        "per/priority_entropy",
    }
    legacy_keys = {
        "PER/beta", "PER/sampled_priority_mean", "PER/sampled_priority_max",
        "PER/buffer_priority_mean", "PER/buffer_priority_max", "PER/weight_mean",
        "PER/weight_std", "PER/prob_mean", "PER/prob_max",
    }
    missing_new = new_keys - collected.keys()
    missing_legacy = legacy_keys - collected.keys()
    assert not missing_new, f"missing new per/ keys: {missing_new}"
    assert not missing_legacy, f"missing legacy PER/ keys: {missing_legacy}"
    assert "td_errors" not in collected


@pytest.mark.slow
def test_per_priority_entropy_absent_when_actual_size_below_2(pendulum_env_nstep, ddpg_agent, tmp_path):
    buffer = _make_per_buffer(pendulum_env_nstep, buffer_size=4, n_transitions=1)
    trainer = _make_stub_trainer(pendulum_env_nstep, buffer, tmp_path, agent=ddpg_agent)
    trainer.metrics.bind(agent=ddpg_agent, env=pendulum_env_nstep, buffer=buffer)

    sample = buffer.sample(1)
    learn_metrics = {"td_errors": T.rand(1)}
    trainer._apply_per_update(sample, learn_metrics)

    collected = trainer.metrics.collect()
    assert "per/priority_entropy" not in collected
    assert "per/is_weight_max" in collected  # unaffected by the actual_size guard


# --------------------------------------------------------------------------- #
# 7. offpolicy/buffer_fill and offpolicy/replay_ratio exact formulas
#
# RETIRED: both keys were deleted from METRIC_REGISTRY with no aliases (they
# were gated on agent class name, so no on-policy run ever logged anything
# about its buffer -- the categorization defect this replaces). Their formula
# coverage now lives elsewhere under the new buffer/*-namespaced, duck-typed
# gating: the saturating-at-1.0 fill formula is
# test_replay_buffer_fill_is_samples_added_over_buffer_size_saturating_at_one,
# and the fixed-numerator ratio is
# test_optim_updates_per_env_step_reflects_current_iteration_post_learn_post_increment
# and test_optim_updates_per_env_step_after_resume_uses_run_segment_not_absolute_counters,
# all in tests/test_buffer_metrics.py.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 8. Trainer.get_config() -> save -> load metrics-block round trip
# (complements tests/test_roundtrip.py, which predates the "metrics" key)
# --------------------------------------------------------------------------- #
def test_trainer_metrics_block_round_trips_byte_for_byte_through_save_load(tmp_path):
    non_default_metrics = {
        "tiers": [0, 1], "emit_every": 2, "include": ["policy/entropy"],
        "exclude": ["health/non_finite"], "dual_emit": False,
    }
    config = _ppo_trainer_config(str(tmp_path / "run") + os.sep)
    config["metrics"] = non_default_metrics
    trainer = build_trainer_from_config(config)
    assert trainer.metrics.get_config() == non_default_metrics

    before = json.dumps(trainer.get_config()["metrics"], sort_keys=True)
    trainer.save()

    loaded = Trainer.load(str(tmp_path / "run"))
    after = json.dumps(loaded.get_config()["metrics"], sort_keys=True)
    assert after == before
    assert loaded.metrics.get_config() == non_default_metrics


# --------------------------------------------------------------------------- #
# 9. RayTuneCallback._accumulate integration (no source change to
# rl_callbacks.py; the input it receives from collect() changed)
# --------------------------------------------------------------------------- #
def test_raytune_accumulate_aggregates_a_coerced_float_that_was_previously_a_tensor():
    """Plan's "Accepted behavior change #1": ``entropy_coefficient`` used to
    arrive at ``_accumulate`` as a live 0-dim tensor (silently skipped by its
    ``isinstance(value, (int, float))`` check) and now arrives as the plain
    float ``collect()`` always produces, so it is aggregated."""
    from phoenx.rl_callbacks import RayTuneCallback

    cb = RayTuneCallback(every=1, unit="timestep")
    cb._accumulate({"onpolicy/entropy_coef": 0.3})
    cb._accumulate({"onpolicy/entropy_coef": 0.5})
    assert cb._step_sums["onpolicy/entropy_coef"] == pytest.approx(0.8)
    assert cb._step_counts["onpolicy/entropy_coef"] == 2


def test_raytune_accumulate_still_skips_a_live_tensor_unchanged_by_this_feature():
    """``rl_callbacks.py`` has no code changes (plan: "No changes"); a value
    that is still literally a live tensor is skipped exactly as before. The
    behavior change is entirely in what ``metrics.record()`` now hands this
    callback, never in ``_accumulate`` itself."""
    from phoenx.rl_callbacks import RayTuneCallback

    cb = RayTuneCallback(every=1, unit="timestep")
    cb._accumulate({"still_a_tensor": T.tensor(0.3)})
    assert "still_a_tensor" not in cb._step_sums


# =============================================================================
# REGRESSION TESTS -- six findings accepted from two independent reviewers.
# Each test below FAILS against the current implementation and reproduces
# the defect described in the amended `.cursor/plans/metrics-manager-phase1.md`.
# No src/ file is touched here; these tests are written to the amended
# contracts, not to what the implementation happens to do today.
# =============================================================================

# --------------------------------------------------------------------------- #
# F1 (BLOCKER) -- one-element tensors (e.g. log_alpha.exp(), shape (1,)) are
# dropped by `record`'s dim()!=0 check instead of being coerced.
# --------------------------------------------------------------------------- #
def test_f1_record_many_coerces_log_alpha_exp_instead_of_dropping_it(sac_agent, pendulum_env):
    """Plan VERIFY: log_alpha = torch.zeros(1, requires_grad=True);
    record_many({"entropy_coefficient": log_alpha.exp()}) on a manager bound
    to a SAC agent; assert collect()["sac/alpha"] == 1.0 and
    collect()["entropy_coefficient"] == 1.0."""
    log_alpha = T.zeros(1, requires_grad=True)
    mm = MetricsManager()
    mm.bind(agent=sac_agent, env=pendulum_env, buffer=None)

    mm.record_many({"entropy_coefficient": log_alpha.exp(), "policy_loss": 0.5})
    collected = mm.collect()

    assert collected.get("sac/alpha") == pytest.approx(1.0), (
        f"'sac/alpha' missing/wrong -- log_alpha.exp() has shape (1,), dim()==1, "
        f"numel()==1; a dim()!=0 coercion check drops it entirely. collected={collected}"
    )
    assert collected.get("entropy_coefficient") == pytest.approx(1.0)
    # Sanity: the unrelated key in the same record_many call is unaffected.
    assert collected.get("loss/policy") == pytest.approx(0.5)


@pytest.mark.slow
def test_f1_entropy_coefficient_reaches_on_train_step_end_for_sac(tmp_path):
    """Plan VERIFY (end-to-end half): a real short SAC train with
    auto_entropy_tuning=True must deliver "entropy_coefficient" to
    on_train_step_end. PPO is deliberately excluded (pre-existing,
    unrelated RuntimeError at rl_agents.py:2004 when auto_entropy_tuning is
    combined with multiple PPO learning epochs -- out of scope here)."""
    config = _sac_auto_entropy_trainer_config(str(tmp_path / "run") + os.sep)
    trainer = build_trainer_from_config(config)
    cb = _RecordingCallback()
    trainer.callbacks = [cb]
    trainer.train()

    seen_keys = set()
    for log in cb.step_logs:
        seen_keys.update(log.keys())

    assert cb.step_logs, "on_train_step_end was never called"
    assert "entropy_coefficient" in seen_keys, (
        "'entropy_coefficient' never reached on_train_step_end for a real "
        "SAC auto_entropy_tuning run"
    )
    assert "sac/alpha" in seen_keys


@pytest.mark.slow
def test_f1_entropy_coefficient_reaches_on_train_step_end_for_actor_critic(tmp_path):
    """Plan VERIFY (end-to-end half), ActorCritic side."""
    config = _actor_critic_auto_entropy_trainer_config(str(tmp_path / "run") + os.sep)
    trainer = build_trainer_from_config(config)
    cb = _RecordingCallback()
    trainer.callbacks = [cb]
    trainer.train()

    seen_keys = set()
    for log in cb.step_logs:
        seen_keys.update(log.keys())

    assert cb.step_logs, "on_train_step_end was never called"
    assert "entropy_coefficient" in seen_keys, (
        "'entropy_coefficient' never reached on_train_step_end for a real "
        "ActorCritic auto_entropy_tuning run"
    )
    assert "onpolicy/entropy_coef" in seen_keys


# --------------------------------------------------------------------------- #
# F2 (MAJOR) -- `exclude` renames an excluded spec's alias into the unmapped
# bucket instead of suppressing it, when driven through record_many (record()
# has no alias resolution, so the pre-existing exclude test using record()
# cannot see this).
# --------------------------------------------------------------------------- #
def test_f2_exclude_suppresses_the_alias_too_when_driven_through_record_many():
    """Plan VERIFY: MetricsManager(exclude=["loss/policy"]);
    record_many({"policy_loss": 0.5}); assert collect() == {} apart from
    health/non_finite, and specifically that "policy_loss" is absent."""
    mm = MetricsManager(exclude=["loss/policy"])
    mm.record_many({"policy_loss": 0.5})
    collected = mm.collect()

    assert "policy_loss" not in collected, (
        f"'policy_loss' leaked through the unmapped bucket instead of being "
        f"suppressed: excluding 'loss/policy' must suppress its alias too, "
        f"not rename the metric to its legacy alias. collected={collected}"
    )
    assert "loss/policy" not in collected
    extra = set(collected) - {"health/non_finite"}
    assert not extra, f"unexpected keys survived exclude: {extra}"


# --------------------------------------------------------------------------- #
# F3 (MAJOR) -- `include` only ever bypassed the tier gate (a no-op in
# Phase 1, since every spec is tier 0 and the default tiers already contains
# 0), instead of replacing it as a true allow-list.
# --------------------------------------------------------------------------- #
def test_f3_include_is_a_true_allowlist_that_replaces_the_tier_gate():
    """Plan VERIFY: MetricsManager(include=["policy/entropy"]).enabled_keys()
    == {"policy/entropy"}."""
    mm = MetricsManager(include=["policy/entropy"])
    enabled = mm.enabled_keys()
    assert enabled == frozenset({"policy/entropy"}), (
        f"expected exactly {{'policy/entropy'}}, got {len(enabled)} keys: "
        f"{sorted(enabled)} -- an `include` that only bypasses the tier "
        f"check is a no-op in Phase 1 (every spec is tier 0)."
    )


def test_f3_include_sac_alpha_bound_to_ppo_enables_nothing(ppo_agent, cartpole_env):
    """Plan VERIFY: MetricsManager(include=["sac/alpha"]) bound to a PPO
    agent enables nothing (it cannot force a disabled spec onto the wrong
    agent, AND it must not fall back to widening the tier gate for every
    other tier-0 spec)."""
    mm = MetricsManager(include=["sac/alpha"])
    mm.bind(agent=ppo_agent, env=cartpole_env, buffer=None)
    enabled = mm.enabled_keys()
    assert enabled == frozenset(), (
        f"expected enabled_keys() to be empty, got {len(enabled)} keys: "
        f"{sorted(enabled)}"
    )


# --------------------------------------------------------------------------- #
# F4 (MAJOR) -- a canonical registry key that is not enabled bypasses gating
# via record_many, landing in the unmapped bucket (Reduction.LAST) instead of
# being dropped.
# --------------------------------------------------------------------------- #
def test_f4_record_many_drops_a_disabled_canonical_key_instead_of_unmapping_it():
    """Plan VERIFY: MetricsManager() (unbound, so sac/alpha disabled);
    record_many({"sac/alpha": 0.2}); assert "sac/alpha" not in collect()."""
    mm = MetricsManager()
    assert "sac/alpha" not in mm.enabled_keys()  # precondition: unbound, disabled

    mm.record_many({"sac/alpha": 0.2})
    collected = mm.collect()
    assert "sac/alpha" not in collected, (
        f"a disabled canonical key bypassed capability gating via "
        f"record_many's unmapped-bucket fallback: collected={collected}"
    )


# --------------------------------------------------------------------------- #
# F5 (MAJOR) -- rollout/return_std and rollout/ep_length_std are backed by
# WINDOW_STD, which needs >=2 values recorded in the SAME collect() window;
# at the default emit_every=1 a window is one env step, making the metric
# structurally unreachable on a single-env run (it would need >=2 parallel
# envs finishing episodes on the same step). The amended plan introduces
# Reduction.ROLLING_STD, backed by a deque that survives collect().
# --------------------------------------------------------------------------- #
def test_f5_rollout_return_std_reachable_on_a_single_env_run_across_episodes(tmp_path):
    """Plan VERIFY: drive a single-env stub trainer through 3 complete
    episodes with differing returns, calling collect() after each; assert
    rollout/return_std is present and equals the population std of the
    returns seen so far. Also assert absence after only one episode.

    Each episode after the first needs one leading "phantom" step (the
    pre-existing valid_steps = ~self._prev_done mask, unrelated to this
    metric) that does NOT finish an episode, so the episode's own return is
    the one actually accumulated -- exactly the pattern already used by
    test_evaluation_steps_do_not_contribute_to_next_training_window."""
    env = _ScriptedFakeEnv(num_envs=1, rewards=T.tensor([0.0]),
                            terminations=T.tensor([False]), truncations=T.tensor([False]))
    trainer = _make_stub_trainer(env, _RecordingBuffer(), tmp_path,
                                  stop_unit="timestep", stop_units=1)

    returns_seen = []
    for i, ret in enumerate((5.0, 9.0, 1.0)):
        if i > 0:
            # Leading phantom step: does not finish an episode, so the
            # subsequent real step starts from a clean (non-phantom) state.
            env._rewards = T.tensor([0.0])
            env._terminations = T.tensor([False])
            env._truncations = T.tensor([False])
            trainer.step(training=True)

        env._rewards = T.tensor([ret])
        env._terminations = T.tensor([True])
        env._truncations = T.tensor([False])
        result = trainer.step(training=True)
        assert result["episode_logs"], f"episode {i} did not finish as scripted"
        for log in result["episode_logs"]:
            trainer._episode_scores[log["env"]] = 0
            trainer._episode_steps[log["env"]] = 0
        returns_seen.append(ret)

        collected = trainer.metrics.collect()
        if len(returns_seen) < 2:
            assert "rollout/return_std" not in collected, (
                "rollout/return_std must be absent after only one episode"
            )
        else:
            expected = statistics.pstdev(returns_seen)
            assert "rollout/return_std" in collected, (
                f"rollout/return_std absent after {len(returns_seen)} episodes "
                f"({returns_seen}) -- WINDOW_STD needs >=2 values in the SAME "
                f"collect() window, which is structurally unreachable here "
                f"since only one value is recorded between collect() calls; "
                f"the amended plan requires a deque-backed ROLLING_STD that "
                f"survives collect() instead."
            )
            assert collected["rollout/return_std"] == pytest.approx(expected), (
                f"rollout/return_std == {collected.get('rollout/return_std')}, "
                f"expected population std of {returns_seen} == {expected}"
            )


# --------------------------------------------------------------------------- #
# F6 (MAJOR) -- offpolicy/replay_ratio divided by the absolute self._step,
# which IS restored on a checkpoint resume, while self._grad_step was
# deliberately NOT restored -- so the ratio was depressed by however many
# steps the run had already taken before the resume.
#
# RETIRED: offpolicy/replay_ratio no longer exists, but the defect class
# (a resume-sensitive counter divided by a resume-restored absolute step)
# is guarded against for its successor, optim/updates_per_env_step, by
# test_optim_updates_per_env_step_after_resume_uses_run_segment_not_absolute_counters
# in tests/test_buffer_metrics.py. That successor was confirmed to actually
# detect this defect class: temporarily replacing its trainer-side
# denominator (`self._step - (self._run_start_step or 0)`) with the
# absolute `self._step` made it fail with
#   AssertionError: expected new_updates / new_env_steps == 10/5 == 2.0; ...
#   Obtained: 0.009950248756218905 (== 10/1005, the un-anchored ratio)
#   Expected: 2.0
# confirming it fails for exactly this reason (depressed by the restored
# absolute step) rather than by accident, before the diagnostic edit was
# reverted.
# --------------------------------------------------------------------------- #
