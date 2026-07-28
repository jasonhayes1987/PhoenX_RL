"""Unit tests for ``src/phoenx/buffer.py``.

These tests import the **real** classes from the PhoenX API
(``SumTree``, ``Buffer``, ``ReplayBuffer``, ``PrioritizedReplayBuffer``,
``RolloutBuffer`` and ``TrajectoryBuffer``) and exercise them directly, so any
refactor of the API is automatically validated by simply re-running this file
(no test-side mock copies of the classes).

Goal of the suite:
    Passing every test here means "I can trust that all of my buffer classes in
    ``src/phoenx/buffer.py`` are behaving correctly."

Coverage:
    * SumTree            - capacity rounding, sum invariant, exact cumulative
                           lookup, priority floor / max tracking, statistical
                           proportionality of sampling.
    * ReplayBuffer       - storage round-trip, circular overwrite, dtypes,
                           sampling bounds, reset / is_ready, config round-trip,
                           goal storage, and an end-to-end ``record`` smoke test.
    * PrioritizedReplay  - both ``proportional`` and ``rank`` strategies:
                           new-transition max priority, importance-weight maths,
                           that high-|TD| transitions dominate sampling
                           (proportional) / high-rank items dominate (rank),
                           beta annealing, NaN-safe priority updates, config.
    * RolloutBuffer      - per-env circular writes, phantom (first-step) masking,
                           sample-then-reset semantics.
    * TrajectoryBuffer   - completed-trajectory capture on episode boundaries,
                           phantom filtering, sample clears storage.
    * Buffer.create_instance factory + HER-related constructor validation.

Real Gymnasium environments are used as lightweight dependencies:
    * ``CartPole-v1``   - Discrete(2) actions, Box(4,) obs.
    * ``Pendulum-v1``   - Box(1,) actions, Box(3,) obs.
    * ``FetchReach-v4`` - Dict goal obs (skipped if MuJoCo is unavailable).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch as T
import gymnasium as gym

from phoenx.buffer import (
    Buffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    SumTree,
    TrajectoryBuffer,
)
from phoenx.env_wrapper import Action, GymnasiumWrapper
from phoenx.her import AchievedGoalPool, HindsightRelabeler

DEVICE = T.device("cpu")

# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
T.manual_seed(0)
np.random.seed(0)


# =============================================================================
# Environment fixtures (real, CPU, fast)
# =============================================================================
@pytest.fixture(scope="module")
def discrete_env() -> GymnasiumWrapper:
    """CartPole-v1 with the N-step wrapper (required by the off-policy buffers)."""
    env = GymnasiumWrapper(
        cfg="CartPole-v1",
        num_envs=2,
        seed=0,
        wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}],
    )
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def continuous_env() -> GymnasiumWrapper:
    """Pendulum-v1 with the N-step wrapper (continuous-action coverage)."""
    env = GymnasiumWrapper(
        cfg="Pendulum-v1",
        num_envs=2,
        seed=0,
        wrappers=[{"type": "VectorNStepReward", "params": {"n": 1}}],
    )
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def plain_env() -> GymnasiumWrapper:
    """CartPole-v1 with NO N-step wrapper, 3 envs (for on-policy buffers)."""
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=3, seed=0)
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def goal_env() -> GymnasiumWrapper:
    """FetchReach-v4 (Dict goal obs). Skipped if MuJoCo / robotics is missing."""
    try:
        env = GymnasiumWrapper(
            cfg="FetchReach-v4",
            num_envs=2,
            seed=0,
            obs_key="observation",
            goal_key="desired_goal",
            ach_goal_key="achieved_goal",
            wrappers=[{
                "type": "VectorNStepReward",
                "params": {
                    "n": 1,
                    "obs_key": "observation",
                    "goal_key": "desired_goal",
                    "ach_goal_key": "achieved_goal",
                },
            }],
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"goal env (FetchReach-v4) unavailable: {exc}")
    yield env
    _safe_close(env)


@pytest.fixture(scope="module")
def nstep_env_n4() -> GymnasiumWrapper:
    """Single CartPole env with an N=4 N-step wrapper, for window-structure tests."""
    env = GymnasiumWrapper(
        cfg="CartPole-v1",
        num_envs=1,
        seed=0,
        wrappers=[{"type": "VectorNStepReward", "params": {"n": 4}}],
    )
    yield env
    _safe_close(env)


def _safe_close(env: GymnasiumWrapper) -> None:
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass


# =============================================================================
# Helpers
# =============================================================================
def _is_discrete(env: GymnasiumWrapper) -> bool:
    return hasattr(env.single_action_space, "n")


def _obs_shape(env: GymnasiumWrapper) -> tuple[int, ...]:
    space = env.single_observation_space
    if isinstance(space, gym.spaces.Dict):
        return space[env.obs_key].shape
    return space.shape


def _goal_shape(env: GymnasiumWrapper) -> tuple[int, ...] | None:
    if env.goal_key is None:
        return None
    return env.single_observation_space[env.goal_key].shape


def make_nstep_batch(
    env: GymnasiumWrapper,
    N: int,
    B: int,
    *,
    state_fill: float | None = None,
    dones: bool = False,
) -> Dict[str, T.Tensor]:
    """Build a synthetic N-step batch shaped exactly like the env emits.

    Mirrors ``VectorNStepReward``'s trajectory dict: states/next_states are
    ``(B, N, *obs)``, discrete actions are ``(B, N)`` and continuous are
    ``(B, N, *act)``, scalar fields are ``(B, N)`` and ``trajectory_lengths``
    is ``(B,)``. Goal fields are added only for goal-aware envs.
    """
    obs_shape = _obs_shape(env)
    if state_fill is None:
        states = T.randn(B, N, *obs_shape, device=DEVICE)
        next_states = T.randn(B, N, *obs_shape, device=DEVICE)
    else:
        states = T.full((B, N, *obs_shape), float(state_fill), device=DEVICE)
        next_states = T.full((B, N, *obs_shape), float(state_fill), device=DEVICE)

    if _is_discrete(env):
        n_actions = int(env.single_action_space.n)
        actions = T.randint(0, n_actions, (B, N), device=DEVICE)
    else:
        act_shape = env.single_action_space.shape
        actions = T.randn(B, N, *act_shape, device=DEVICE)

    done_val = bool(dones)
    batch: Dict[str, T.Tensor] = {
        "states": states,
        "actions": actions,
        "rewards": T.randn(B, N, device=DEVICE),
        "next_states": next_states,
        "terminations": T.full((B, N), done_val, dtype=T.bool, device=DEVICE),
        "truncations": T.zeros(B, N, dtype=T.bool, device=DEVICE),
        "log_probs": T.zeros(B, N, device=DEVICE),
        "intrinsic_rewards": T.zeros(B, N, device=DEVICE),
        "trajectory_lengths": T.full((B,), N, dtype=T.int64, device=DEVICE),
    }

    goal_shape = _goal_shape(env)
    if goal_shape is not None:
        batch["state_achieved_goals"] = T.randn(B, N, *goal_shape, device=DEVICE)
        batch["next_state_achieved_goals"] = T.randn(B, N, *goal_shape, device=DEVICE)
        batch["desired_goals"] = T.randn(B, N, *goal_shape, device=DEVICE)
    return batch


def make_step_batch(
    env: GymnasiumWrapper,
    *,
    terminations: T.Tensor | None = None,
    first_steps: T.Tensor | None = None,
) -> Dict[str, T.Tensor]:
    """Build a single per-env step batch (shape ``(num_envs, *feat)``) for the
    on-policy buffers' ``add``."""
    E = env.num_envs
    obs_shape = _obs_shape(env)
    if _is_discrete(env):
        n_actions = int(env.single_action_space.n)
        actions = T.randint(0, n_actions, (E,), device=DEVICE)
    else:
        actions = T.randn(E, *env.single_action_space.shape, device=DEVICE)

    if terminations is None:
        terminations = T.zeros(E, dtype=T.bool, device=DEVICE)
    return {
        "states": T.randn(E, *obs_shape, device=DEVICE),
        "actions": actions,
        "rewards": T.randn(E, device=DEVICE),
        "next_states": T.randn(E, *obs_shape, device=DEVICE),
        "terminations": terminations,
        "truncations": T.zeros(E, dtype=T.bool, device=DEVICE),
        "first_steps": first_steps,
    }


def _drive_nstep(env: GymnasiumWrapper):
    """Take one random step through an N-step-wrapped env and return the
    resulting ``Observation`` (with its ``n_step_trajectory`` populated)."""
    a = T.as_tensor(env.action_space.sample())
    act = Action(actions=a, log_probs=T.zeros(env.num_envs, device=DEVICE))
    env._find_nstep_wrapper().set_action(act)
    return env.step(a)


def _make_goal_episode(env: GymnasiumWrapper, T_ep: int) -> Dict[str, T.Tensor]:
    """Synthetic completed episode for a goal env, with deterministic, well-separated
    achieved goals so HER reward recomputation is unambiguous.

    ``next_state_achieved_goals[t] = [t, 0, 0]`` — consecutive goals are 1.0 apart,
    far beyond FetchReach's 0.05 success threshold, so a window only scores a
    success (reward 0) when its relabeled desired goal exactly matches its own
    achieved goal.
    """
    obs_dim = _obs_shape(env)[0]
    goal_dim = _goal_shape(env)[0]
    act_dim = env.single_action_space.shape[0]

    next_ach = T.zeros(T_ep, goal_dim, device=DEVICE)
    next_ach[:, 0] = T.arange(T_ep, dtype=T.float32, device=DEVICE)
    state_ach = T.zeros(T_ep, goal_dim, device=DEVICE)
    state_ach[:, 0] = T.arange(T_ep, dtype=T.float32, device=DEVICE) - 0.5
    return {
        "states": T.randn(T_ep, obs_dim, device=DEVICE),
        "actions": T.randn(T_ep, act_dim, device=DEVICE),
        "rewards": -T.ones(T_ep, device=DEVICE),
        "next_states": T.randn(T_ep, obs_dim, device=DEVICE),
        "terminations": T.zeros(T_ep, dtype=T.bool, device=DEVICE),
        "truncations": T.zeros(T_ep, dtype=T.bool, device=DEVICE),
        "state_achieved_goals": state_ach,
        "next_state_achieved_goals": next_ach,
        "desired_goals": T.randn(T_ep, goal_dim, device=DEVICE),
    }


# =============================================================================
# SumTree
# =============================================================================
class TestSumTree:
    def test_capacity_rounds_up_to_power_of_two(self):
        # 100 -> 128, 5 -> 8, 8 -> 8, 1 -> 1.
        assert SumTree(100, DEVICE).capacity == 128
        assert SumTree(5, DEVICE).capacity == 8
        assert SumTree(8, DEVICE).capacity == 8
        assert SumTree(1, DEVICE).capacity == 1

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            SumTree(0, DEVICE)
        with pytest.raises(ValueError):
            SumTree(-3, DEVICE)

    def test_empty_total_priority_is_zero(self):
        assert SumTree(16, DEVICE).total_priority == 0.0

    def test_total_priority_is_sum_of_leaves(self):
        tree = SumTree(8, DEVICE)
        idx = T.tensor([0, 1, 2, 3], device=DEVICE)
        pri = T.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        tree.update(idx, pri)
        assert tree.total_priority == pytest.approx(10.0, abs=1e-5)

        # Overwriting a leaf updates the running total correctly.
        tree.update(T.tensor([1], device=DEVICE), T.tensor([10.0], device=DEVICE))
        assert tree.total_priority == pytest.approx(18.0, abs=1e-5)

    def test_priority_floor_applied(self):
        tree = SumTree(4, DEVICE)
        tree.update(T.tensor([0], device=DEVICE), T.tensor([0.0], device=DEVICE))
        # 0 is floored to 1e-6 rather than stored as a zero-probability leaf.
        assert tree.total_priority == pytest.approx(1e-6, abs=1e-9)

    def test_max_priority_tracking(self):
        tree = SumTree(4, DEVICE)
        assert tree.max_priority.item() == pytest.approx(1.0)
        tree.update(T.tensor([0], device=DEVICE), T.tensor([5.0], device=DEVICE))
        assert tree.max_priority.item() == pytest.approx(5.0)
        # A smaller subsequent priority must not lower the running max.
        tree.update(T.tensor([1], device=DEVICE), T.tensor([2.0], device=DEVICE))
        assert tree.max_priority.item() == pytest.approx(5.0)

    def test_get_exact_cumulative_lookup(self):
        """Cumulative lookup must land in the correct leaf and return its value.

        Priorities [1,2,3,4] -> leaf intervals
            leaf0=[0,1), leaf1=[1,3), leaf2=[3,6), leaf3=[6,10).
        """
        tree = SumTree(4, DEVICE)
        tree.update(
            T.tensor([0, 1, 2, 3], device=DEVICE),
            T.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE),
        )
        p = T.tensor([0.5, 2.0, 4.5, 8.0], device=DEVICE)
        data_idx, pri = tree.get(p)
        assert data_idx.tolist() == [0, 1, 2, 3]
        assert pri.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_get_handles_rounded_capacity(self):
        # capacity 5 -> 8 leaves; data index 4 (last real one) must be reachable.
        tree = SumTree(5, DEVICE)
        tree.update(T.arange(5, device=DEVICE), T.tensor([1.0, 1, 1, 1, 5.0], device=DEVICE))
        assert tree.total_priority == pytest.approx(9.0, abs=1e-5)
        # A value inside leaf-4's interval [4, 9) must return data index 4.
        data_idx, pri = tree.get(T.tensor([6.0], device=DEVICE))
        assert data_idx.item() == 4
        assert pri.item() == pytest.approx(5.0)

    def test_sampling_is_proportional_to_priority(self):
        """Frequency of sampled leaves must track their priority share."""
        T.manual_seed(123)
        tree = SumTree(4, DEVICE)
        priorities = T.tensor([1.0, 1.0, 1.0, 97.0], device=DEVICE)  # total 100
        tree.update(T.arange(4, device=DEVICE), priorities)

        n = 20_000
        p = T.rand(n, device=DEVICE) * tree.total_priority
        data_idx, _ = tree.get(p)
        freq = T.bincount(data_idx, minlength=4).float() / n
        # Dominant leaf should be sampled ~97% of the time.
        assert freq[3].item() > 0.93
        # Each minor leaf is ~1%.
        for i in range(3):
            assert freq[i].item() == pytest.approx(0.01, abs=0.01)


# =============================================================================
# ReplayBuffer
# =============================================================================
class TestReplayBuffer:
    def test_requires_nstep_wrapper(self, plain_env):
        with pytest.raises(ValueError, match="VectorNStepReward"):
            ReplayBuffer(env=plain_env, buffer_size=100, N=1, device=DEVICE)

    def test_add_increments_counter_and_stores(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=100, N=1, device=DEVICE)
        assert buf.samples_added == 0
        batch = make_nstep_batch(discrete_env, N=1, B=10, state_fill=0.5)
        buf.add(**batch)
        assert buf.samples_added == 10
        # Stored states match what we put in.
        assert T.allclose(buf.states[:10], T.full_like(buf.states[:10], 0.5))

    def test_action_dtype_is_long_for_discrete(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=50, N=1, device=DEVICE)
        buf.add(**make_nstep_batch(discrete_env, N=1, B=5))
        assert buf.actions.dtype == T.long

    def test_action_dtype_is_float_for_continuous(self, continuous_env):
        buf = ReplayBuffer(env=continuous_env, buffer_size=50, N=1, device=DEVICE)
        buf.add(**make_nstep_batch(continuous_env, N=1, B=5))
        assert buf.actions.dtype == T.float32

    def test_nstep_dimension_preserved(self, discrete_env):
        N = 3
        buf = ReplayBuffer(env=discrete_env, buffer_size=20, N=N, device=DEVICE)
        buf.add(**make_nstep_batch(discrete_env, N=N, B=4))
        assert buf.states.shape[1] == N
        assert buf.rewards.shape == (20, N)

    def test_circular_overwrite(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=4, N=1, device=DEVICE)
        # Fill completely with marker 1.0.
        buf.add(**make_nstep_batch(discrete_env, N=1, B=4, state_fill=1.0))
        assert buf.samples_added == 4
        # Add 2 more with marker 9.0 -> wraps to indices 0 and 1.
        buf.add(**make_nstep_batch(discrete_env, N=1, B=2, state_fill=9.0))
        assert buf.samples_added == 6
        assert T.allclose(buf.states[0], T.full_like(buf.states[0], 9.0))
        assert T.allclose(buf.states[1], T.full_like(buf.states[1], 9.0))
        # Indices 2 and 3 still hold the original marker.
        assert T.allclose(buf.states[2], T.full_like(buf.states[2], 1.0))

    def test_sample_shapes_and_bounds(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=100, N=1, device=DEVICE)
        buf.add(**make_nstep_batch(discrete_env, N=1, B=30))
        sample = buf.sample(16)
        assert sample["states"].shape[0] == 16
        assert sample["rewards"].shape[0] == 16
        # No priority bookkeeping in the vanilla buffer.
        assert "weights" not in sample and "indices" not in sample

    def test_sample_only_draws_from_filled_region(self, discrete_env):
        """A partially-filled buffer must never return zeroed (unwritten) rows."""
        buf = ReplayBuffer(env=discrete_env, buffer_size=1000, N=1, device=DEVICE)
        buf.add(**make_nstep_batch(discrete_env, N=1, B=5, state_fill=3.0))
        sample = buf.sample(50)
        assert T.allclose(sample["states"], T.full_like(sample["states"], 3.0))

    def test_sample_empty_raises(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=10, N=1, device=DEVICE)
        with pytest.raises(ValueError, match="empty buffer"):
            buf.sample(4)

    def test_is_ready(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=100, N=1, device=DEVICE)
        assert not buf.is_ready(8)
        buf.add(**make_nstep_batch(discrete_env, N=1, B=8))
        assert buf.is_ready(8)

    def test_reset(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=20, N=1, device=DEVICE)
        buf.add(**make_nstep_batch(discrete_env, N=1, B=10, state_fill=2.0))
        buf.reset()
        assert buf.samples_added == 0
        assert T.all(buf.states == 0)
        assert T.all(buf.rewards == 0)

    def test_get_config_round_trip(self, discrete_env):
        buf = ReplayBuffer(env=discrete_env, buffer_size=123, N=2, device=DEVICE)
        cfg = buf.get_config()
        assert cfg["type"] == "ReplayBuffer"
        assert cfg["config"]["buffer_size"] == 123
        assert cfg["config"]["N"] == 2
        # env is serialized as a JSON string and must be parseable.
        env_cfg = json.loads(cfg["config"]["env"])
        assert env_cfg["config"]["cfg"] == "CartPole-v1"

    def test_record_integration(self, discrete_env):
        """Drive the real env through ``record`` and confirm storage + sampling."""
        buf = ReplayBuffer(env=discrete_env, buffer_size=2000, N=1, device=DEVICE)
        obs = discrete_env.reset(seed=0)
        prev_dones = T.zeros(discrete_env.num_envs, dtype=T.bool, device=DEVICE)
        nstep = discrete_env._find_nstep_wrapper()
        for _ in range(30):
            a = T.as_tensor(discrete_env.action_space.sample())
            act = Action(actions=a, log_probs=T.zeros(discrete_env.num_envs))
            nstep.set_action(act)
            next_obs = discrete_env.step(a)
            buf.record(next_obs, obs, act, prev_dones)
            prev_dones = T.logical_or(next_obs.terminations, next_obs.truncations)
            obs = next_obs
        assert buf.samples_added > 0
        sample = buf.sample(min(8, buf.samples_added))
        assert sample["states"].shape[0] == min(8, buf.samples_added)


# =============================================================================
# PrioritizedReplayBuffer - shared behaviour + construction
# =============================================================================
class TestPrioritizedReplayBufferCommon:
    def test_invalid_priority_raises(self, discrete_env):
        with pytest.raises(ValueError, match="Invalid priority"):
            PrioritizedReplayBuffer(
                env=discrete_env, buffer_size=100, priority="nonsense", device=DEVICE
            )

    def test_requires_nstep_wrapper(self, plain_env):
        with pytest.raises(ValueError, match="VectorNStepReward"):
            PrioritizedReplayBuffer(env=plain_env, buffer_size=100, device=DEVICE)

    @pytest.mark.parametrize("priority", ["proportional", "rank"])
    def test_sample_returns_priority_metadata(self, discrete_env, priority):
        buf = PrioritizedReplayBuffer(
            env=discrete_env, buffer_size=200, priority=priority, device=DEVICE
        )
        buf.add(**make_nstep_batch(discrete_env, N=1, B=64))
        sample = buf.sample(16)
        for key in ("weights", "probs", "indices"):
            assert key in sample, f"{priority}: missing {key}"
            assert sample[key].shape[0] == 16
        assert T.all(sample["weights"] > 0)
        # Importance weights are normalized so the max equals 1.
        assert sample["weights"].max().item() == pytest.approx(1.0, abs=1e-5)
        # Indices stay inside the filled region.
        size = min(buf.samples_added, buf.buffer_size)
        assert int(sample["indices"].min()) >= 0
        assert int(sample["indices"].max()) < size

    @pytest.mark.parametrize("priority", ["proportional", "rank"])
    def test_beta_anneals_to_one(self, discrete_env, priority):
        buf = PrioritizedReplayBuffer(
            env=discrete_env,
            buffer_size=200,
            priority=priority,
            beta_start=0.4,
            beta_iter=50,
            beta_update_freq=1,
            device=DEVICE,
        )
        buf.add(**make_nstep_batch(discrete_env, N=1, B=64))
        assert buf.beta == pytest.approx(0.4)
        for _ in range(60):
            buf.sample(8)
        assert buf.beta == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("priority", ["proportional", "rank"])
    def test_update_priorities_is_nan_safe(self, discrete_env, priority):
        buf = PrioritizedReplayBuffer(
            env=discrete_env, buffer_size=64, priority=priority, device=DEVICE
        )
        buf.add(**make_nstep_batch(discrete_env, N=1, B=64))
        idx = T.arange(8, device=DEVICE)
        td = T.tensor([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0], device=DEVICE)
        buf.update_priorities(idx, td)
        # No NaN should leak into the priority bookkeeping.
        if priority == "proportional":
            assert not np.isnan(buf.sum_tree.total_priority)
            assert buf.sum_tree.total_priority > 0
        else:
            assert not T.isnan(buf.priorities).any()
        # Sampling still works afterwards.
        buf.sample(8)

    def test_update_priorities_accepts_python_sequences(self, discrete_env):
        buf = PrioritizedReplayBuffer(
            env=discrete_env, buffer_size=64, priority="proportional", device=DEVICE
        )
        buf.add(**make_nstep_batch(discrete_env, N=1, B=64))
        buf.update_priorities([0, 1, 2], [0.5, 1.5, 2.5])  # lists, not tensors
        assert buf.sum_tree.total_priority > 0

    def test_get_config(self, discrete_env):
        buf = PrioritizedReplayBuffer(
            env=discrete_env,
            buffer_size=100,
            alpha=0.7,
            beta_start=0.5,
            priority="rank",
            device=DEVICE,
        )
        cfg = buf.get_config()
        assert cfg["type"] == "PrioritizedReplayBuffer"
        c = cfg["config"]
        for key in ("alpha", "beta_start", "beta_iter", "beta_update_freq",
                    "priority", "epsilon", "sort_freq", "N"):
            assert key in c
        assert c["alpha"] == 0.7
        assert c["priority"] == "rank"


# =============================================================================
# PrioritizedReplayBuffer - proportional strategy correctness
# =============================================================================
class TestPrioritizedProportional:
    def _full_buffer(self, env, size, **kw):
        buf = PrioritizedReplayBuffer(
            env=env, buffer_size=size, priority="proportional", device=DEVICE, **kw
        )
        buf.add(**make_nstep_batch(env, N=1, B=size))
        return buf

    def test_new_transitions_get_max_priority(self, discrete_env):
        buf = self._full_buffer(discrete_env, 16)
        # Right after add (no TD update yet) every leaf carries max_priority,
        # so sampling is uniform and every IS-weight equals 1.
        sample = buf.sample(16)
        assert T.allclose(sample["weights"], T.ones_like(sample["weights"]), atol=1e-5)

    def test_high_td_error_dominates_sampling(self, discrete_env):
        """With alpha=1 the sampling probability is proportional to |TD|, so a
        single huge-error transition should dominate the sampled indices."""
        T.manual_seed(7)
        buf = self._full_buffer(
            discrete_env, 16, alpha=1.0, beta_update_freq=10_000
        )
        td = T.full((16,), 0.01, device=DEVICE)
        td[5] = 50.0
        buf.update_priorities(T.arange(16, device=DEVICE), td)

        counts = T.zeros(16, device=DEVICE)
        for _ in range(200):
            counts += T.bincount(buf.sample(16)["indices"], minlength=16).float()
        frac = counts / counts.sum()
        assert frac[5].item() > 0.9
        assert int(frac.argmax()) == 5

    def test_importance_weights_match_formula(self, discrete_env):
        """weights == (size * probs) ** (-beta), normalized so max == 1."""
        T.manual_seed(11)
        buf = self._full_buffer(
            discrete_env, 32, alpha=0.6, beta_start=0.5, beta_update_freq=10_000
        )
        td = T.abs(T.randn(32, device=DEVICE)) + 0.1
        buf.update_priorities(T.arange(32, device=DEVICE), td)

        sample = buf.sample(16)
        size = min(buf.samples_added, buf.buffer_size)
        expected = (size * sample["probs"]).pow(-buf.beta)
        expected = expected / expected.max()
        assert T.allclose(sample["weights"], expected, atol=1e-5)
        # Monotonicity: the highest-probability sample carries the lowest weight.
        assert int(sample["weights"].argmin()) == int(sample["probs"].argmax())

    def test_sampling_frequency_tracks_all_priorities(self, discrete_env):
        """The strongest proportional check: with alpha=1 the empirical sampling
        frequency of every transition must match its priority share, validating
        the full add -> update_priorities -> SumTree -> stratified-sample path."""
        T.manual_seed(99)
        size = 8
        buf = self._full_buffer(discrete_env, size, alpha=1.0, beta_update_freq=10_000)
        # priorities proportional to 1..8 (epsilon negligible at this scale).
        td = T.arange(1, size + 1, dtype=T.float32, device=DEVICE)
        buf.update_priorities(T.arange(size, device=DEVICE), td)

        expected = td / td.sum()
        counts = T.zeros(size, device=DEVICE)
        for _ in range(400):
            counts += T.bincount(buf.sample(size)["indices"], minlength=size).float()
        freq = counts / counts.sum()
        assert T.allclose(freq, expected, atol=0.03), f"{freq.tolist()} vs {expected.tolist()}"

    def test_nstep_shapes_preserved(self, discrete_env):
        N = 3
        buf = PrioritizedReplayBuffer(
            env=discrete_env, buffer_size=64, N=N, priority="proportional", device=DEVICE
        )
        buf.add(**make_nstep_batch(discrete_env, N=N, B=20))
        sample = buf.sample(8)
        assert sample["states"].shape[:2] == (8, N)
        assert sample["rewards"].shape == (8, N)

    def test_probs_sum_consistent(self, discrete_env):
        buf = self._full_buffer(discrete_env, 32, beta_update_freq=10_000)
        buf.update_priorities(
            T.arange(32, device=DEVICE), T.abs(T.randn(32, device=DEVICE)) + 0.1
        )
        sample = buf.sample(8)
        # Each prob is in (0, 1].
        assert T.all(sample["probs"] > 0)
        assert T.all(sample["probs"] <= 1.0 + 1e-6)


# =============================================================================
# PrioritizedReplayBuffer - rank strategy correctness
# =============================================================================
class TestPrioritizedRank:
    def _full_buffer(self, env, size, **kw):
        buf = PrioritizedReplayBuffer(
            env=env, buffer_size=size, priority="rank", device=DEVICE, **kw
        )
        buf.add(**make_nstep_batch(env, N=1, B=size))
        return buf

    def test_high_rank_items_sampled_more(self, discrete_env):
        """Rank-based sampling favours high-priority (low-rank) transitions."""
        T.manual_seed(21)
        size = 32
        buf = self._full_buffer(size=size, env=discrete_env, alpha=0.6,
                                beta_update_freq=10_000)
        # Give each index a distinct priority; index i has priority i, so the
        # top quartile by priority is indices 24..31.
        td = T.arange(size, dtype=T.float32, device=DEVICE) + 1.0
        buf.update_priorities(T.arange(size, device=DEVICE), td)

        counts = T.zeros(size, device=DEVICE)
        for _ in range(300):
            counts += T.bincount(buf.sample(size)["indices"], minlength=size).float()
        frac = counts / counts.sum()
        top_quartile = frac[24:].sum().item()
        bottom_quartile = frac[:8].sum().item()
        # Power-law over ranks concentrates mass on the top items.
        assert top_quartile > 0.35
        assert top_quartile > bottom_quartile

    def test_new_transition_gets_max_rank_priority(self, discrete_env):
        """A freshly added transition should be (near) the top of the sort so it
        is eligible for early replay."""
        T.manual_seed(3)
        buf = self._full_buffer(env=discrete_env, size=16, beta_update_freq=10_000)
        # Push existing priorities down low.
        buf.update_priorities(
            T.arange(16, device=DEVICE), T.full((16,), 0.01, device=DEVICE)
        )
        # Overwrite index 0 with a new transition (gets max_priority_rank).
        single = make_nstep_batch(discrete_env, N=1, B=1)
        buf.samples_added = 0  # force the next add to land at index 0
        buf.add(**single)
        buf._maybe_resort(16)
        # Index 0 now has the maximum stored priority.
        assert int(buf.priorities[:16].argmax()) == 0

    def test_rank_weights_normalized(self, discrete_env):
        buf = self._full_buffer(env=discrete_env, size=32, beta_update_freq=10_000)
        buf.update_priorities(
            T.arange(32, device=DEVICE), T.abs(T.randn(32, device=DEVICE)) + 0.1
        )
        sample = buf.sample(16)
        assert sample["weights"].max().item() == pytest.approx(1.0, abs=1e-5)
        assert T.all(sample["weights"] > 0)


# =============================================================================
# RolloutBuffer
# =============================================================================
class TestRolloutBuffer:
    def test_add_advances_per_env_index(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        for _ in range(5):
            buf.add(**make_step_batch(plain_env))
        assert int(buf.cur_idx.max()) == 5

    def test_sample_shapes_and_reset(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        steps = 6
        for _ in range(steps):
            buf.add(**make_step_batch(plain_env))
        sample = buf.sample()
        assert sample["states"].shape[0] == steps
        assert sample["states"].shape[1] == plain_env.num_envs
        # RolloutBuffer.sample resets the rollout in place.
        assert int(buf.cur_idx.max()) == 0

    def test_phantom_first_steps_excluded(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        E = plain_env.num_envs
        # Step 0: env 0 is a phantom (first) step.
        first = T.zeros(E, dtype=T.bool, device=DEVICE)
        first[0] = True
        buf.add(**make_step_batch(plain_env, first_steps=first))
        buf.add(**make_step_batch(plain_env))
        sample = buf.sample()
        total_slots = sample["states"].shape[0] * E
        # Exactly one phantom step was flagged and excluded.
        assert sample["valid_indices"].shape[0] == total_slots - 1

    def test_is_ready_always_true(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        assert buf.is_ready() is True

    def test_sample_empty_raises(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        with pytest.raises(ValueError, match="empty buffer"):
            buf.sample()

    def test_record_integration(self, plain_env):
        buf = RolloutBuffer(env=plain_env, buffer_size=200, device=DEVICE)
        obs = plain_env.reset(seed=1)
        prev_dones = T.zeros(plain_env.num_envs, dtype=T.bool, device=DEVICE)
        for _ in range(8):
            a = T.as_tensor(plain_env.action_space.sample())
            act = Action(actions=a, log_probs=T.zeros(plain_env.num_envs))
            next_obs = plain_env.step(a)
            buf.record(next_obs, obs, act, prev_dones)
            prev_dones = T.logical_or(next_obs.terminations, next_obs.truncations)
            obs = next_obs
        sample = buf.sample()
        assert sample["states"].shape[0] == 8


# =============================================================================
# TrajectoryBuffer
# =============================================================================
class TestTrajectoryBuffer:
    def test_completed_trajectory_captured_on_done(self, plain_env):
        buf = TrajectoryBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        E = plain_env.num_envs
        # Two ongoing steps, then env 0 terminates.
        buf.add(**make_step_batch(plain_env))
        buf.add(**make_step_batch(plain_env))
        term = T.zeros(E, dtype=T.bool, device=DEVICE)
        term[0] = True
        buf.add(**make_step_batch(plain_env, terminations=term))

        trajectories = buf.sample()
        assert len(trajectories) == 1
        # The completed trajectory holds env 0's 3 steps.
        assert trajectories[0]["states"].shape[0] == 3
        # cur_idx for env 0 was reset after capture.
        assert int(buf.cur_idx[0]) == 0

    def test_sample_clears_completed(self, plain_env):
        buf = TrajectoryBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        term = T.zeros(plain_env.num_envs, dtype=T.bool, device=DEVICE)
        term[0] = True
        buf.add(**make_step_batch(plain_env, terminations=term))
        assert len(buf.sample()) == 1
        # Second sample is empty - storage was cleared.
        assert buf.sample() == []

    def test_phantom_only_trajectory_discarded(self, plain_env):
        """A 'trajectory' consisting solely of a phantom first step that
        immediately terminates yields no usable data."""
        buf = TrajectoryBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        E = plain_env.num_envs
        first = T.zeros(E, dtype=T.bool, device=DEVICE)
        first[0] = True
        term = T.zeros(E, dtype=T.bool, device=DEVICE)
        term[0] = True
        buf.add(**make_step_batch(plain_env, terminations=term, first_steps=first))
        assert buf.sample() == []
        assert int(buf.cur_idx[0]) == 0

    def test_reset_clears(self, plain_env):
        buf = TrajectoryBuffer(env=plain_env, buffer_size=50, device=DEVICE)
        buf.add(**make_step_batch(plain_env))
        buf.reset()
        assert int(buf.cur_idx.max()) == 0
        assert buf.completed_trajectories == []


# =============================================================================
# Buffer factory + HER-related construction validation
# =============================================================================
class TestBufferFactory:
    def test_create_instance_dispatches(self, discrete_env, plain_env):
        rb = Buffer.create_instance(
            "ReplayBuffer", env=discrete_env, buffer_size=100, N=1, device=DEVICE
        )
        assert isinstance(rb, ReplayBuffer)
        pr = Buffer.create_instance(
            "PrioritizedReplayBuffer", env=discrete_env, buffer_size=100, device=DEVICE
        )
        assert isinstance(pr, PrioritizedReplayBuffer)
        ro = Buffer.create_instance(
            "RolloutBuffer", env=plain_env, buffer_size=50, device=DEVICE
        )
        assert isinstance(ro, RolloutBuffer)

    def test_create_instance_unknown_raises(self, discrete_env):
        with pytest.raises(ValueError, match="not a subclass of Buffer"):
            Buffer.create_instance("Nope", env=discrete_env, buffer_size=10)


class TestHindsightValidation:
    """Buffer-side validation of HER relabeler compatibility (goal env)."""

    def test_replaybuffer_rejects_flat_relabeler(self, goal_env):
        relabeler = HindsightRelabeler(
            goal_env, strategy="final", output_format="flat", device=DEVICE
        )
        with pytest.raises(ValueError, match="output_format = 'n_step'"):
            ReplayBuffer(
                env=goal_env, buffer_size=100, N=1, hindsight=relabeler, device=DEVICE
            )

    def test_replaybuffer_rejects_mismatched_N(self, goal_env):
        relabeler = HindsightRelabeler(
            goal_env, strategy="future", output_format="n_step", N=2, device=DEVICE
        )
        with pytest.raises(ValueError, match="N value must match"):
            ReplayBuffer(
                env=goal_env, buffer_size=100, N=1, hindsight=relabeler, device=DEVICE
            )

    def test_trajectorybuffer_rejects_nstep_relabeler(self, goal_env):
        relabeler = HindsightRelabeler(
            goal_env, strategy="future", output_format="n_step", N=1, device=DEVICE
        )
        with pytest.raises(ValueError, match="output_format = 'flat'"):
            TrajectoryBuffer(
                env=goal_env, buffer_size=100, hindsight=relabeler, device=DEVICE
            )


# =============================================================================
# Goal-aware storage (Dict observation envs)
# =============================================================================
class TestGoalBuffers:
    def test_replaybuffer_stores_goals(self, goal_env):
        N = 1
        buf = ReplayBuffer(env=goal_env, buffer_size=100, N=N, device=DEVICE)
        assert hasattr(buf, "desired_goals")
        buf.add(**make_nstep_batch(goal_env, N=N, B=10))
        assert buf.samples_added == 10
        sample = buf.sample(8)
        goal_shape = _goal_shape(goal_env)
        assert sample["desired_goals"] is not None
        assert sample["desired_goals"].shape == (8, N, *goal_shape)
        assert sample["state_achieved_goals"].shape == (8, N, *goal_shape)

    def test_replaybuffer_missing_goal_data_raises(self, goal_env):
        buf = ReplayBuffer(env=goal_env, buffer_size=100, N=1, device=DEVICE)
        batch = make_nstep_batch(goal_env, N=1, B=4)
        del batch["desired_goals"]
        del batch["state_achieved_goals"]
        del batch["next_state_achieved_goals"]
        with pytest.raises(ValueError, match="Goal data must be provided"):
            buf.add(**batch)

    def test_per_stores_goals(self, goal_env):
        N = 1
        buf = PrioritizedReplayBuffer(
            env=goal_env, buffer_size=100, N=N, priority="proportional", device=DEVICE
        )
        buf.add(**make_nstep_batch(goal_env, N=N, B=10))
        sample = buf.sample(8)
        goal_shape = _goal_shape(goal_env)
        assert sample["desired_goals"].shape == (8, N, *goal_shape)
        assert "weights" in sample


# =============================================================================
# N-step window generation (VectorNStepReward) and its interaction with buffers
# =============================================================================
class TestNStepWindows:
    """Validate the N-step values the env feeds into the off-policy buffers.

    The sliding window emitted each step is what ``ReplayBuffer.record`` stores,
    so these properties are exactly the N-step semantics the buffers rely on.
    """

    def test_window_length_grows_then_clamps(self, nstep_env_n4):
        env = nstep_env_n4
        env.reset(seed=0)
        seq = []
        for _ in range(6):
            obs = _drive_nstep(env)
            if obs.n_step_trajectory:
                seq.append(obs.n_step_trajectory["trajectory_lengths"].tolist())
        # CartPole cannot terminate in the first handful of steps, so the window
        # length grows 1->2->3->4 and then stays clamped at N=4.
        assert seq[:4] == [[1], [2], [3], [4]]
        assert seq[4] == [4] and seq[5] == [4]

    def test_chronological_continuity(self, nstep_env_n4):
        """Within every window, next_states[k] must equal states[k+1] — i.e. the
        window is a real contiguous slice of the trajectory."""
        env = nstep_env_n4
        env.reset(seed=0)
        for _ in range(10):
            obs = _drive_nstep(env)
            tr = obs.n_step_trajectory
            if not tr:
                continue
            for row in range(tr["states"].shape[0]):
                L = int(tr["trajectory_lengths"][row])
                states = tr["states"][row]
                next_states = tr["next_states"][row]
                for k in range(L - 1):
                    assert T.allclose(states[k + 1], next_states[k]), (
                        f"discontinuity at row {row}, step {k}"
                    )

    def test_repeat_and_zero_padding(self, nstep_env_n4):
        """States past the valid length are repeat-padded; rewards are zero-padded."""
        env = nstep_env_n4
        env.reset(seed=0)
        _drive_nstep(env)               # length 1
        obs = _drive_nstep(env)         # length 2 (< N=4)
        tr = obs.n_step_trajectory
        assert tr["trajectory_lengths"].tolist() == [2]
        states = tr["states"][0]
        rewards = tr["rewards"][0]
        # Padding slots (indices 2,3) repeat the last valid state (index 1)...
        assert T.allclose(states[2], states[1])
        assert T.allclose(states[3], states[1])
        # ...and carry zero reward.
        assert rewards[2].item() == 0.0
        assert rewards[3].item() == 0.0

    def test_newest_next_state_matches_env(self, nstep_env_n4):
        """The most recent transition in the window must match the env's last step."""
        env = nstep_env_n4
        env.reset(seed=0)
        for _ in range(5):
            obs = _drive_nstep(env)
            if bool((obs.terminations | obs.truncations).any()):
                continue
            tr = obs.n_step_trajectory
            if not tr:
                continue
            L = int(tr["trajectory_lengths"][0])
            assert T.allclose(tr["next_states"][0][L - 1], obs.states[0])

    def test_done_emits_descending_tail(self, nstep_env_n4):
        """On episode end every step becomes its own anchor, so the emitted batch
        carries windows of lengths L, L-1, ..., 1 (contiguous)."""
        env = nstep_env_n4
        env.reset(seed=2)
        for _ in range(400):
            obs = _drive_nstep(env)
            done = bool((obs.terminations | obs.truncations).any())
            if done and obs.n_step_trajectory:
                lengths = sorted(obs.n_step_trajectory["trajectory_lengths"].tolist())
                L = max(lengths)
                assert lengths == list(range(1, L + 1))
                return
        pytest.fail("no episode completed within the step cap")

    def test_reset_clears_window_state(self, nstep_env_n4):
        env = nstep_env_n4
        env.reset(seed=0)
        for _ in range(3):
            _drive_nstep(env)
        env.reset(seed=0)
        obs = _drive_nstep(env)
        # After reset the accumulator is empty, so the first window has length 1.
        assert obs.n_step_trajectory["trajectory_lengths"].tolist() == [1]

    def test_nstep_windows_feed_replaybuffer(self, nstep_env_n4):
        """End-to-end: N-step windows recorded into a ReplayBuffer keep their
        chronological structure after storage and sampling."""
        buf = ReplayBuffer(env=nstep_env_n4, buffer_size=5000, N=4, device=DEVICE)
        nstep_env_n4.reset(seed=0)
        prev_dones = T.zeros(nstep_env_n4.num_envs, dtype=T.bool, device=DEVICE)
        obs = None
        for _ in range(40):
            a = T.as_tensor(nstep_env_n4.action_space.sample())
            act = Action(actions=a, log_probs=T.zeros(nstep_env_n4.num_envs, device=DEVICE))
            nstep_env_n4._find_nstep_wrapper().set_action(act)
            nxt = nstep_env_n4.step(a)
            buf.record(nxt, obs if obs is not None else nxt, act, prev_dones)
            prev_dones = T.logical_or(nxt.terminations, nxt.truncations)
            obs = nxt
        assert buf.samples_added > 0
        sample = buf.sample(min(16, buf.samples_added))
        # Continuity must survive storage for full-length (L==N) windows.
        for row in range(sample["states"].shape[0]):
            L = int(sample["trajectory_lengths"][row])
            states = sample["states"][row]
            next_states = sample["next_states"][row]
            for k in range(L - 1):
                assert T.allclose(states[k + 1], next_states[k])


# =============================================================================
# HER — goal-sampling logic (white-box, per-strategy)
# =============================================================================
class TestHERGoalSampling:
    def test_final_uses_final_goal_for_every_step(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="final", output_format="n_step", N=1, device=DEVICE
        )
        ep = _make_goal_episode(goal_env, 5)
        starts, goals = rl._sample_per_step_goals(ep, 5)
        assert starts.tolist() == [0, 1, 2, 3, 4]
        final = ep["next_state_achieved_goals"][4]
        assert T.allclose(goals, final.expand_as(goals))

    def test_future_inclusive_respects_causality(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="future", num_goals=3, output_format="n_step",
            N=1, future_lo="inclusive", device=DEVICE,
        )
        T.manual_seed(0)
        ep = _make_goal_episode(goal_env, 6)
        starts, goals = rl._sample_per_step_goals(ep, 6)
        # Inclusive: each of the 6 steps yields 3 future goals.
        assert starts.numel() == 18
        sampled_idx = goals[:, 0]  # achieved-goal coord encodes its time index
        assert bool((sampled_idx >= starts.float()).all())   # never from the past
        assert bool((sampled_idx < 6).all())

    def test_future_exclusive_drops_last_step(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="future", num_goals=2, output_format="n_step",
            N=1, future_lo="exclusive", device=DEVICE,
        )
        T.manual_seed(0)
        ep = _make_goal_episode(goal_env, 5)
        starts, goals = rl._sample_per_step_goals(ep, 5)
        # Exclusive: the final step has no strictly-future goal -> dropped.
        assert starts.numel() == (5 - 1) * 2
        assert int(starts.max()) == 3
        assert bool((goals[:, 0] > starts.float()).all())    # strictly future

    def test_episode_samples_from_whole_episode(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="episode", num_goals=2, output_format="n_step",
            N=1, device=DEVICE,
        )
        T.manual_seed(0)
        ep = _make_goal_episode(goal_env, 5)
        starts, goals = rl._sample_per_step_goals(ep, 5)
        assert starts.numel() == 5 * 2
        idx = goals[:, 0]
        assert bool(((idx >= 0) & (idx < 5)).all())


# =============================================================================
# HER — n_step relabeling (off-policy output) + buffer hand-off
# =============================================================================
class TestHERRelabelNStep:
    def _relabeler(self, env, **kw):
        kw.setdefault("strategy", "final")
        kw.setdefault("N", 1)
        return HindsightRelabeler(
            env, output_format="n_step", device=DEVICE, **kw
        )

    def test_final_injects_success_reward(self, goal_env):
        rl = self._relabeler(goal_env)
        ep = _make_goal_episode(goal_env, 5)
        out = rl.relabel_episode(ep)
        rewards = out["rewards"].view(-1)
        # Only the window anchored at the final step achieves its relabeled goal.
        assert abs(rewards[-1].item()) < 1e-6           # success -> 0
        assert T.all(rewards[:-1] == -1.0)              # all others fail
        # Every window's desired goal is the episode's final achieved goal.
        final = ep["next_state_achieved_goals"][4]
        assert T.allclose(out["desired_goals"], final.view(1, 1, 3).expand_as(out["desired_goals"]))

    def test_window_count_equals_episode_length(self, goal_env):
        rl = self._relabeler(goal_env)
        out = rl.relabel_episode(_make_goal_episode(goal_env, 7))
        assert out["states"].shape[0] == 7

    def test_nstep_padding_matches_env_convention(self, goal_env):
        rl = self._relabeler(goal_env, N=3)
        ep = _make_goal_episode(goal_env, 5)
        out = rl.relabel_episode(ep)
        assert out["states"].shape == (5, 3, 10)
        # Last window (anchored at final step) has only 1 valid step.
        assert int(out["trajectory_lengths"][-1]) == 1
        assert T.allclose(out["states"][-1, 2], out["states"][-1, 0])  # repeat pad
        assert out["rewards"][-1, 1].item() == 0.0                     # zero pad
        assert out["rewards"][-1, 2].item() == 0.0
        # First window spans the full N steps.
        assert int(out["trajectory_lengths"][0]) == 3

    def test_relabel_terminations_marks_achievement(self, goal_env):
        rl = self._relabeler(goal_env, relabel_terminations=True)
        ep = _make_goal_episode(goal_env, 5)
        out = rl.relabel_episode(ep)
        term = out["terminations"].view(-1)
        assert bool(term[-1])               # final window achieves -> terminal
        assert not bool(term[:-1].any())    # earlier windows do not

    def test_relabeled_output_feeds_per_buffer(self, goal_env):
        rl = self._relabeler(goal_env)
        out = rl.relabel_episode(_make_goal_episode(goal_env, 6))
        buf = PrioritizedReplayBuffer(
            env=goal_env, buffer_size=100, N=1, hindsight=rl,
            priority="proportional", device=DEVICE,
        )
        buf.add(**out)
        assert buf.samples_added == 6
        sample = buf.sample(4)
        assert sample["desired_goals"].shape == (4, 1, 3)
        assert "weights" in sample

    def test_empty_episode_returns_none(self, goal_env):
        rl = self._relabeler(goal_env)
        empty = {k: v[:0] for k, v in _make_goal_episode(goal_env, 3).items()}
        assert rl.relabel_episode(empty) is None


# =============================================================================
# HER — flat relabeling (on-policy output)
# =============================================================================
class TestHERRelabelFlat:
    def test_final_flat_single_trajectory(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="final", output_format="flat", device=DEVICE
        )
        ep = _make_goal_episode(goal_env, 5)
        out = rl.relabel_episode(ep)
        assert isinstance(out, list) and len(out) == 1
        traj = out[0]
        assert traj["states"].shape == (5, 10)          # full trajectory, not windows
        final = ep["next_state_achieved_goals"][4]
        assert T.allclose(traj["desired_goals"], final.view(1, 3).expand(5, 3))
        rewards = traj["rewards"]
        assert abs(rewards[-1].item()) < 1e-6
        assert T.all(rewards[:-1] == -1.0)

    def test_episode_flat_returns_num_goals_copies(self, goal_env):
        rl = HindsightRelabeler(
            goal_env, strategy="episode", num_goals=3, output_format="flat",
            device=DEVICE,
        )
        out = rl.relabel_episode(_make_goal_episode(goal_env, 5))
        assert len(out) == 3
        for traj in out:
            assert traj["states"].shape == (5, 10)
            assert traj["desired_goals"].shape == (5, 3)


# =============================================================================
# AchievedGoalPool (backs HER strategy='random')
# =============================================================================
class TestAchievedGoalPool:
    def test_add_and_size_caps_at_capacity(self):
        pool = AchievedGoalPool(5, (3,), DEVICE)
        pool.add(T.zeros(3, 3, device=DEVICE))
        assert pool.size == 3
        pool.add(T.ones(4, 3, device=DEVICE))
        assert pool.size == 5

    def test_fifo_overwrite(self):
        pool = AchievedGoalPool(4, (3,), DEVICE)
        pool.add(T.zeros(4, 3, device=DEVICE))
        pool.add(T.ones(2, 3, device=DEVICE))   # wraps, overwrites the 2 oldest
        assert int((pool.buffer == 1).all(dim=1).sum()) == 2

    def test_batch_larger_than_capacity_keeps_last(self):
        pool = AchievedGoalPool(3, (3,), DEVICE)
        goals = T.arange(15, dtype=T.float32, device=DEVICE).view(5, 3)
        pool.add(goals)
        assert pool.size == 3
        assert T.allclose(pool.buffer, goals[-3:])

    def test_sample_only_returns_added_goals(self):
        pool = AchievedGoalPool(10, (3,), DEVICE)
        vals = T.tensor([[1.0, 0, 0], [2, 0, 0], [3, 0, 0]], device=DEVICE)
        pool.add(vals)
        sampled = pool.sample(20)
        assert sampled.shape == (20, 3)
        assert set(sampled[:, 0].tolist()).issubset({1.0, 2.0, 3.0})


class TestHERRandomStrategy:
    def test_random_samples_from_pool(self, goal_env):
        pool = AchievedGoalPool(10, (3,), DEVICE)
        pool.add(T.tensor([[100.0, 0, 0], [200.0, 0, 0]], device=DEVICE))
        rl = HindsightRelabeler(
            goal_env, strategy="random", num_goals=2, output_format="n_step",
            N=1, goal_pool=pool, device=DEVICE,
        )
        out = rl.relabel_episode(_make_goal_episode(goal_env, 5))
        assert out["states"].shape[0] == 5 * 2
        coords = set(out["desired_goals"][..., 0].unique().tolist())
        assert coords.issubset({100.0, 200.0})

    def test_random_empty_pool_skips_episode(self, goal_env):
        pool = AchievedGoalPool(10, (3,), DEVICE)
        rl = HindsightRelabeler(
            goal_env, strategy="random", num_goals=2, output_format="n_step",
            N=1, goal_pool=pool, device=DEVICE,
        )
        assert rl.relabel_episode(_make_goal_episode(goal_env, 5)) is None


# =============================================================================
# HER <-> buffer integration (driven through real env episodes)
# =============================================================================
class TestHERBufferIntegration:
    def test_offpolicy_record_adds_relabeled_transitions(self, goal_env):
        """Drive real FetchReach episodes through PER.record with a HER relabeler;
        on episode completion the buffer must gain relabeled transitions beyond the
        normal per-step additions."""
        rl = HindsightRelabeler(
            goal_env, strategy="final", output_format="n_step", N=1, device=DEVICE
        )
        buf = PrioritizedReplayBuffer(
            env=goal_env, buffer_size=20_000, N=1, hindsight=rl,
            priority="proportional", device=DEVICE,
        )
        obs = goal_env.reset(seed=0)
        prev = T.zeros(goal_env.num_envs, dtype=T.bool, device=DEVICE)
        nstep = goal_env._find_nstep_wrapper()
        her_added = False
        for _ in range(140):
            a = T.as_tensor(goal_env.action_space.sample())
            act = Action(actions=a, log_probs=T.zeros(goal_env.num_envs, device=DEVICE))
            nstep.set_action(act)
            nxt = goal_env.step(a)
            before = buf.samples_added
            buf.record(nxt, obs, act, prev)
            delta = buf.samples_added - before
            real_b = nxt.n_step_trajectory["states"].shape[0] if nxt.n_step_trajectory else 0
            if bool(prev.any()) and delta > real_b:
                her_added = True
                break
            prev = T.logical_or(nxt.terminations, nxt.truncations)
            obs = nxt
        assert her_added, "HER relabeling did not add transitions on episode completion"
        sample = buf.sample(8)
        assert sample["desired_goals"] is not None

    def test_onpolicy_trajectory_buffer_relabels(self, goal_env):
        """TrajectoryBuffer + flat HER: completed episodes yield both the real
        trajectory and a relabeled copy whose final reward is a success (0)."""
        rl = HindsightRelabeler(
            goal_env, strategy="final", output_format="flat", device=DEVICE
        )
        buf = TrajectoryBuffer(
            env=goal_env, buffer_size=200, hindsight=rl, device=DEVICE
        )
        obs = goal_env.reset(seed=0)
        prev = T.zeros(goal_env.num_envs, dtype=T.bool, device=DEVICE)
        nstep = goal_env._find_nstep_wrapper()
        for _ in range(140):
            a = T.as_tensor(goal_env.action_space.sample())
            act = Action(actions=a, log_probs=T.zeros(goal_env.num_envs, device=DEVICE))
            nstep.set_action(act)
            nxt = goal_env.step(a)
            buf.record(nxt, obs, act, prev)
            prev = T.logical_or(nxt.terminations, nxt.truncations)
            obs = nxt
            if len(buf.completed_trajectories) >= 2:
                break
        trajectories = buf.sample()
        # At least one real + one relabeled trajectory.
        assert len(trajectories) >= 2
        # The relabeled 'final' copy guarantees a terminal success reward of 0.
        assert any(abs(float(tr["rewards"][-1])) < 1e-6 for tr in trajectories)
