"""End-to-end Dict (multi-modal) observation data path tests.

Uses the REAL production stack on the synthetic multi-modal env registered in
``conftest.py`` (``PhoenXMultiModal-v0``: uint8 image + float vector obs):

    * ``GymnasiumWrapper.extract_states_goals`` with ``obs_key=None`` returns
      dict states with per-modality dtypes preserved;
    * ``VectorNStepReward`` emits dict-states n-step trajectories that match a
      naive per-env reference record exactly (temporal order, padding, dtypes);
    * all four buffers store/sample dict observations correctly
      (dtype preservation: uint8 in storage, reference roundtrips);
    * the ``raw_actions`` (B, N) vs (B, N, 1) shape contract works end to end
      for discrete envs (audit item 10.2.2 — asserted, not worked around);
    * ``DictNormalizer`` / ``ImageScale`` route per key;
    * a full Trainer-shaped loop: env -> RolloutBuffer -> PPO.learn with dict
      obs and a multi-root composite model updates and stays finite;
    * SAC + ReplayBuffer on the same env (off-policy dict path);
    * HER relabeling with dict states (n_step + flat output formats).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch as T

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.buffer import (  # noqa: E402
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    TrajectoryBuffer,
)
from app.env_wrapper import Action, GymnasiumWrapper  # noqa: E402
from app.her import HindsightRelabeler  # noqa: E402
from app.models import (  # noqa: E402
    ContinuousQHead,
    StochasticContinuousHead,
    SubNetwork,
    ValueHead,
)
from app.normalizer import DictNormalizer, ImageScale, RunningNorm, create_normalizer  # noqa: E402
from app.obs_utils import flatten_leading, flatten_obs, tree_index  # noqa: E402
from app.rl_agents import PPO, SAC  # noqa: E402

DEVICE = "cpu"
T.manual_seed(0)
np.random.seed(0)

LC = [{"type": "dense", "params": {"units": 16}}, {"type": "relu"}]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


@pytest.fixture(scope="module")
def mm_env():
    """Multi-modal env WITHOUT n-step wrapper (on-policy path)."""
    env = GymnasiumWrapper(cfg="PhoenXMultiModal-v0", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def mm_nstep_env():
    """Multi-modal env WITH the VectorNStepReward wrapper (off-policy path)."""
    env = GymnasiumWrapper(
        cfg="PhoenXMultiModal-v0", num_envs=2, seed=0,
        wrappers=[{"type": "VectorNStepReward", "params": {"n": 3}}],
    )
    yield env
    try:
        env.close()
    except Exception:
        pass


def _step_env(env, action_t=None):
    a = env.action_space.sample() if action_t is None else action_t
    a_t = T.as_tensor(np.asarray(a))
    nstep = env._find_nstep_wrapper()
    if nstep is not None:
        nstep.set_action(Action(actions=a_t, log_probs=T.zeros(env.num_envs)))
    return a_t, env.step(a_t)


def _mm_roots():
    return {
        "cnn": SubNetwork([
            {"type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "stride": 2}},
            {"type": "relu"}, {"type": "flatten"},
        ], input_keys=["rgb"], name="cnn"),
        "vec": SubNetwork(LC, input_keys=["vec"], name="vec"),
    }


# =============================================================================
# Env wrapper: dict extraction + VectorNStepReward emission ground truth
# =============================================================================
class TestDictExtraction:
    def test_reset_and_step_return_dict_states(self, mm_env):
        obs = mm_env.reset(seed=3)
        assert isinstance(obs.states, dict)
        assert set(obs.states.keys()) == {"rgb", "vec"}
        assert obs.states["rgb"].dtype == T.uint8
        assert obs.states["rgb"].shape == (2, 16, 16, 3)
        assert obs.states["vec"].dtype in (T.float32, T.float64)
        assert obs.goals is None and obs.ach_goals is None

        _, next_obs = _step_env(mm_env)
        assert isinstance(next_obs.states, dict)
        assert next_obs.rewards.shape == (2,)

    def test_nstep_wrapper_emits_dict_trajectories(self, mm_nstep_env):
        """The emitted windows must match a naive per-env reference record —
        the dict-obs analogue of the ground-truth emission tests in
        test_agent_utils.py."""
        obs = mm_nstep_env.reset(seed=5)
        cur = {k: v.clone() for k, v in obs.states.items()}
        episodes = [[] for _ in range(mm_nstep_env.num_envs)]
        prev_done = [False] * mm_nstep_env.num_envs
        n = 3
        checked_windows = 0

        for _ in range(30):
            a_t, next_obs = _step_env(mm_nstep_env)
            dones = T.logical_or(next_obs.terminations, next_obs.truncations)
            for e in range(mm_nstep_env.num_envs):
                if not prev_done[e]:
                    episodes[e].append({
                        "state": {k: v[e].clone() for k, v in cur.items()},
                        "next_state": {k: v[e].clone() for k, v in next_obs.states.items()},
                        "reward": float(next_obs.rewards[e]),
                    })
            traj = next_obs.n_step_trajectory
            if traj is not None and len(traj.get("trajectory_lengths", [])) > 0:
                assert isinstance(traj["states"], dict)
                assert traj["states"]["rgb"].dtype == T.uint8  # dtype preserved
                row = 0
                for e in range(mm_nstep_env.num_envs):
                    L = min(len(episodes[e]), n)
                    if L == 0:
                        continue
                    window = episodes[e][-L:]
                    assert int(traj["trajectory_lengths"][row]) == L
                    for k_step in range(L):
                        for key in ("rgb", "vec"):
                            assert T.equal(
                                traj["states"][key][row, k_step].cpu(),
                                window[k_step]["state"][key].cpu(),
                            ), f"env {e} step {k_step} key {key}: state mismatch"
                            assert T.equal(
                                traj["next_states"][key][row, k_step].cpu(),
                                window[k_step]["next_state"][key].cpu(),
                            )
                        assert abs(float(traj["rewards"][row, k_step]) - window[k_step]["reward"]) < 1e-5
                    checked_windows += 1
                    row += 1
            for e in range(mm_nstep_env.num_envs):
                prev_done[e] = bool(dones[e])
                if dones[e]:
                    episodes[e] = []
            cur = {k: v.clone() for k, v in next_obs.states.items()}

        assert checked_windows >= 10


# =============================================================================
# Buffers with dict observations
# =============================================================================
class TestDictBuffers:
    def _mm_step_batch(self, env, E=None):
        E = E or env.num_envs
        return {
            "states": {"rgb": T.randint(0, 255, (E, 16, 16, 3), dtype=T.uint8),
                       "vec": T.randn(E, 7)},
            "actions": T.randn(E, 2),
            "rewards": T.randn(E),
            "next_states": {"rgb": T.randint(0, 255, (E, 16, 16, 3), dtype=T.uint8),
                            "vec": T.randn(E, 7)},
            "terminations": T.zeros(E, dtype=T.bool),
            "truncations": T.zeros(E, dtype=T.bool),
        }

    def test_rollout_buffer_dict_roundtrip(self, mm_env):
        buf = RolloutBuffer(env=mm_env, buffer_size=10, device=DEVICE)
        assert isinstance(buf.states, dict)
        assert buf.states["rgb"].dtype == T.uint8  # uint8 storage
        batches = [self._mm_step_batch(mm_env) for _ in range(4)]
        for b in batches:
            buf.add(**b)
        sample = buf.sample()
        assert isinstance(sample["states"], dict)
        assert sample["states"]["rgb"].shape == (4, 2, 16, 16, 3)
        assert sample["states"]["rgb"].dtype == T.uint8
        # exact roundtrip vs reference
        for t, b in enumerate(batches):
            for key in ("rgb", "vec"):
                stored = sample["states"][key][t]
                expected = b["states"][key].to(stored.dtype)
                assert T.allclose(stored.float(), expected.float(), atol=1e-6)

    def test_replay_buffer_dict_roundtrip(self, mm_nstep_env):
        buf = ReplayBuffer(env=mm_nstep_env, buffer_size=64, N=3, device=DEVICE)
        assert isinstance(buf.states, dict)
        mm_nstep_env.reset(seed=11)
        prev_done = T.zeros(mm_nstep_env.num_envs, dtype=T.bool)
        obs = None
        for _ in range(12):
            a_t, next_obs = _step_env(mm_nstep_env)
            buf.record(next_obs, prev_observation=obs,
                       actions=Action(actions=a_t, log_probs=T.zeros(mm_nstep_env.num_envs)),
                       prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs
        assert buf.samples_added > 0
        sample = buf.sample(8)
        assert isinstance(sample["states"], dict)
        assert sample["states"]["rgb"].shape == (8, 3, 16, 16, 3)
        assert sample["states"]["rgb"].dtype == T.uint8
        assert T.isfinite(sample["states"]["vec"].float()).all()

    def test_per_buffer_dict_sampling(self, mm_nstep_env):
        buf = PrioritizedReplayBuffer(env=mm_nstep_env, buffer_size=32, N=3,
                                      priority="proportional", device=DEVICE)
        mm_nstep_env.reset(seed=13)
        prev_done = T.zeros(mm_nstep_env.num_envs, dtype=T.bool)
        obs = None
        for _ in range(10):
            a_t, next_obs = _step_env(mm_nstep_env)
            buf.record(next_obs, prev_observation=obs,
                       actions=Action(actions=a_t, log_probs=T.zeros(mm_nstep_env.num_envs)),
                       prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs
        sample = buf.sample(6)
        assert isinstance(sample["states"], dict)
        assert sample["weights"].shape[0] == 6
        buf.update_priorities(sample["indices"], T.rand(6) + 0.1)
        buf.sample(6)

    def test_buffer_save_load_dict_state(self, mm_nstep_env, tmp_path):
        buf = ReplayBuffer(env=mm_nstep_env, buffer_size=16, N=3, device=DEVICE)
        mm_nstep_env.reset(seed=17)
        prev_done = T.zeros(mm_nstep_env.num_envs, dtype=T.bool)
        obs = None
        for _ in range(6):
            a_t, next_obs = _step_env(mm_nstep_env)
            buf.record(next_obs, prev_observation=obs,
                       actions=Action(actions=a_t, log_probs=T.zeros(mm_nstep_env.num_envs)),
                       prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs
        path = tmp_path / "buffer.pt"
        buf.save_state(path)
        buf2 = ReplayBuffer(env=mm_nstep_env, buffer_size=16, N=3, device=DEVICE)
        buf2.load_state(path)
        assert buf2.samples_added == buf.samples_added
        for key in ("rgb", "vec"):
            assert T.equal(buf.states[key], buf2.states[key])

    def test_raw_actions_shape_contract_discrete(self):
        """Audit item 10.2.2: the (B, N) discrete raw_actions emitted by
        VectorNStepReward must roundtrip through ReplayBuffer.add unchanged
        (stored as (B, N, 1)) — asserted rather than worked around."""
        env = GymnasiumWrapper(
            cfg="CartPole-v1", num_envs=2, seed=0,
            wrappers=[{"type": "VectorNStepReward", "params": {"n": 3}}],
        )
        try:
            buf = ReplayBuffer(env=env, buffer_size=64, N=3, device=DEVICE)
            env.reset(seed=0)
            prev_done = T.zeros(env.num_envs, dtype=T.bool)
            obs = None
            fed = False
            for _ in range(20):
                a_t, next_obs = _step_env(env)
                traj = next_obs.n_step_trajectory
                if traj is not None and len(traj.get("trajectory_lengths", [])) > 0:
                    assert traj["raw_actions"].ndim == 2  # (rows, N) emission
                    buf.add(**traj)  # must NOT raise; includes raw_actions
                    fed = True
                prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
                obs = next_obs
            assert fed
            sample = buf.sample(4)
            assert sample["raw_actions"].shape == (4, 3, 1)
        finally:
            try:
                env.close()
            except Exception:
                pass


# =============================================================================
# Storage dtype sync: spaces that lie about observation dtypes
# =============================================================================
class TestStorageDtypeSyncLyingSpace:
    """IsaacLab manager-based envs declare float32 Boxes for EVERY observation
    group — including uint8 camera images. Buffer storage dtypes must follow
    the actual data (reconciled on first add), not the space: trusting the
    space stores uint8 frames as float 0..255, which bypasses the models'
    uint8 -> [0,1] input scaling and made learn-time camera inputs 255x the
    rollout-time inputs (the PPO-camera NaN bug)."""

    class _LyingSpaceEnv:
        import gymnasium as gym
        num_envs = 2
        obs_key = None
        goal_key = None
        ach_goal_key = None
        # The space CLAIMS float32 for rgb; the env actually emits uint8.
        single_observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(-np.inf, np.inf, (8, 8, 3), np.float32),
            "vec": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        })
        observation_space = single_observation_space
        single_action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        action_space = single_action_space

        def _find_nstep_wrapper(self):
            return None

    def _rollout_batch(self, E=2):
        return {
            "states": {"rgb": T.randint(0, 255, (E, 8, 8, 3), dtype=T.uint8),
                       "vec": T.randn(E, 5)},
            "actions": T.randn(E, 2),
            "rewards": T.randn(E),
            "next_states": {"rgb": T.randint(0, 255, (E, 8, 8, 3), dtype=T.uint8),
                            "vec": T.randn(E, 5)},
            "terminations": T.zeros(E, dtype=T.bool),
            "truncations": T.zeros(E, dtype=T.bool),
        }

    def test_rollout_buffer_storage_follows_data(self):
        env = self._LyingSpaceEnv()
        buf = RolloutBuffer(env=env, buffer_size=4, device=DEVICE)
        # Pre-add: storage trusts the (lying) space.
        assert buf.states["rgb"].dtype == T.float32
        batches = [self._rollout_batch() for _ in range(2)]
        for b in batches:
            buf.add(**b)
        # Post-add: storage and spec reconciled to the real uint8 dtype.
        assert buf.states["rgb"].dtype == T.uint8
        assert buf.next_states["rgb"].dtype == T.uint8
        assert buf.obs_spec["rgb"][1] == T.uint8
        assert buf.states["vec"].dtype == T.float32  # honest keys untouched
        sample = buf.sample()
        assert sample["states"]["rgb"].dtype == T.uint8
        for t, b in enumerate(batches):
            assert T.equal(sample["states"]["rgb"][t].cpu(), b["states"]["rgb"])
            assert T.allclose(sample["states"]["vec"][t].cpu(), b["states"]["vec"])

    def test_replay_buffer_storage_follows_data(self):
        env = self._LyingSpaceEnv()
        buf = ReplayBuffer(env=env, buffer_size=16, N=1, device=DEVICE)
        assert buf.states["rgb"].dtype == T.float32
        E = 2
        traj = {
            "states": {"rgb": T.randint(0, 255, (E, 1, 8, 8, 3), dtype=T.uint8),
                       "vec": T.randn(E, 1, 5)},
            "actions": T.randn(E, 1, 2),
            "rewards": T.randn(E, 1),
            "next_states": {"rgb": T.randint(0, 255, (E, 1, 8, 8, 3), dtype=T.uint8),
                            "vec": T.randn(E, 1, 5)},
            "terminations": T.zeros(E, 1, dtype=T.bool),
            "truncations": T.zeros(E, 1, dtype=T.bool),
            "trajectory_lengths": T.ones(E, dtype=T.long),
        }
        buf.add(**traj)
        assert buf.states["rgb"].dtype == T.uint8
        assert buf.next_states["rgb"].dtype == T.uint8
        assert buf.obs_spec["rgb"][1] == T.uint8
        sample = buf.sample(2)
        assert sample["states"]["rgb"].dtype == T.uint8


# =============================================================================
# DictNormalizer / ImageScale
# =============================================================================
class TestDictNormalizer:
    def test_routes_per_key_and_passthrough(self):
        norm = DictNormalizer(per_key={
            "vec": {"type": "RunningNorm", "config": {"num_features": 7, "clip_value": 5.0}},
            "rgb": {"type": "ImageScale", "config": {}},
        }, device=DEVICE)
        data = {"rgb": T.full((4, 2, 2, 3), 255, dtype=T.uint8),
                "vec": T.randn(4, 7) * 3 + 5,
                "extra": T.ones(4, 2)}
        for _ in range(5):
            norm.add({"vec": T.randn(64, 7) * 3 + 5})
            norm.update()
        out = norm.normalize(data)
        assert T.allclose(out["rgb"], T.ones(4, 2, 2, 3))  # 255/255
        assert out["vec"].abs().mean() < 2.0  # roughly standardized
        assert T.equal(out["extra"], data["extra"])  # passthrough

    def test_config_roundtrip_and_state(self, tmp_path):
        norm = DictNormalizer(per_key={
            "vec": {"type": "RunningNorm", "config": {"num_features": 3}},
            "rgb": {"type": "ImageScale", "config": {}},
        }, device=DEVICE)
        norm.add({"vec": T.randn(32, 3) + 7})
        norm.update()
        cfg = norm.get_config()
        assert cfg["type"] == "DictNormalizer"
        rebuilt = create_normalizer({"type": cfg["type"],
                                     "config": {"per_key": cfg["config"]["per_key"],
                                                "device": DEVICE}})
        path = tmp_path / "norm.pt"
        norm.save_state(path)
        rebuilt.load_state(path)
        assert T.allclose(rebuilt.normalizers["vec"].running_mean,
                          norm.normalizers["vec"].running_mean)

    def test_train_eval_propagates(self):
        norm = DictNormalizer(per_key={
            "vec": {"type": "BatchNorm", "config": {"num_features": 3}},
        }, device=DEVICE)
        norm.eval()
        assert not norm.normalizers["vec"].training
        norm.train()
        assert norm.normalizers["vec"].training


class TestRunningNormMinStd:
    """min_std floors the running std so near-constant features (fixed sim
    joints, constant command dims early in training) cannot catastrophically
    amplify later drift (RSL-RL EmpiricalNormalization convention)."""

    def test_default_floor_matches_legacy(self):
        norm = RunningNorm(num_features=2, device=DEVICE)
        assert norm.min_std == pytest.approx(1e-4)
        # constant feature -> variance 0 -> std clamps at the floor
        norm.add(T.ones(64, 2) * 3.0)
        norm.update()
        assert T.allclose(norm.running_std, T.full((2,), 1e-4))

    def test_min_std_floors_amplification(self):
        norm = RunningNorm(num_features=1, min_std=1e-2, clip_value=1e9, device=DEVICE)
        norm.add(T.ones(64, 1) * 3.0)  # constant -> std floor
        norm.update()
        assert T.allclose(norm.running_std, T.full((1,), 1e-2))
        # A 0.05 drift normalizes to 5.0 (drift / min_std), not 500.
        out = norm.normalize(T.full((1, 1), 3.05))
        assert out.item() == pytest.approx(5.0, rel=1e-3)

    def test_min_std_config_roundtrip(self):
        norm = RunningNorm(num_features=3, min_std=0.01, device=DEVICE)
        cfg = norm.get_config()
        assert cfg["config"]["min_std"] == pytest.approx(0.01)
        rebuilt = create_normalizer(cfg)
        assert rebuilt.min_std == pytest.approx(0.01)


# =============================================================================
# obs_utils invariants
# =============================================================================
class TestObsUtils:
    def test_flatten_leading_preserves_feature_shape(self):
        obs = {"rgb": T.randint(0, 255, (4, 2, 8, 8, 3), dtype=T.uint8),
               "vec": T.randn(4, 2, 7)}
        flat = flatten_leading(obs, 2)
        assert flat["rgb"].shape == (8, 8, 8, 3)  # image dims NOT flattened
        assert flat["vec"].shape == (8, 7)

    def test_flatten_obs_scales_uint8_and_concats(self):
        obs = {"a": T.full((3, 2), 255, dtype=T.uint8), "b": T.ones(3, 4)}
        flat = flatten_obs(obs)
        assert flat.shape == (3, 6)
        assert T.allclose(flat[:, :2], T.ones(3, 2))  # 255 -> 1.0
        assert T.allclose(flat[:, 2:], T.ones(3, 4))

    def test_tree_index_tuple(self):
        obs = {"a": T.arange(24).reshape(2, 3, 4)}
        out = tree_index(obs, (slice(None), 0))
        assert out["a"].shape == (2, 4)
        assert T.equal(out["a"], T.arange(24).reshape(2, 3, 4)[:, 0])


# =============================================================================
# End-to-end agents on dict observations
# =============================================================================
class TestDictAgents:
    def test_ppo_multi_root_learns_on_dict_obs(self, mm_env):
        agent = PPO(
            roots=_mm_roots(),
            trunk=SubNetwork(LC, name="trunk"),
            policy=StochasticContinuousHead(mm_env, layer_config=LC, output_config=OC,
                                            distribution="normal", device=DEVICE),
            value=ValueHead(mm_env, layer_config=LC, output_config=OC, device=DEVICE),
            optimizer_params=OPT, auto_entropy_tuning=False,
            state_normalizer=DictNormalizer(per_key={
                "vec": {"type": "RunningNorm", "config": {"num_features": 7}},
            }, device=DEVICE),
            device=DEVICE,
        )
        # Drive the real env through the real RolloutBuffer (trainer-shaped loop)
        buf = RolloutBuffer(env=mm_env, buffer_size=8, device=DEVICE)
        obs = mm_env.reset(seed=21)
        prev_done = T.zeros(mm_env.num_envs, dtype=T.bool)
        for _ in range(8):
            action = agent.act(obs.states, context="train")
            next_obs = mm_env.step(action.actions)
            # intrinsic_rewards field expected by the buffer record path
            from dataclasses import replace
            next_obs = replace(next_obs, intrinsic_rewards=T.zeros_like(next_obs.rewards))
            buf.record(next_obs, prev_observation=obs, actions=action, prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs

        sample = buf.sample()
        assert isinstance(sample["states"], dict)
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample, learning_epochs=2, mini_batch_size=4)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any(k.startswith("roots.cnn.") for k in changed)
        assert any(k.startswith("roots.vec.") for k in changed)
        assert any(k.startswith("trunk.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)
        assert any(k.startswith("branches.value.") for k in changed)

    def test_sac_multi_root_learns_on_dict_obs(self, mm_nstep_env):
        agent = SAC(
            roots=_mm_roots(),
            trunk=SubNetwork(LC, name="trunk"),
            policy=StochasticContinuousHead(mm_nstep_env, layer_config=LC, output_config=OC,
                                            distribution="normal", device=DEVICE),
            critic=ContinuousQHead(mm_nstep_env, layer_config=LC, merged_config=LC,
                                   output_config=OC, device=DEVICE),
            optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE,
        )
        buf = ReplayBuffer(env=mm_nstep_env, buffer_size=64, N=3, device=DEVICE)
        obs = mm_nstep_env.reset(seed=23)
        prev_done = T.zeros(mm_nstep_env.num_envs, dtype=T.bool)
        nstep = mm_nstep_env._find_nstep_wrapper()
        for step_i in range(16):
            action = agent.act(obs.states, context="train", step=step_i, warmup=-1)
            nstep.set_action(action)
            next_obs = mm_nstep_env.step(action.actions)
            buf.record(next_obs, prev_observation=obs, actions=action, prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs

        sample = buf.sample(8)
        assert isinstance(sample["states"], dict)
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any(k.startswith("roots.") for k in changed)      # critic owns shared body
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)


# =============================================================================
# HER with dict states
# =============================================================================
class TestHERDictStates:
    def _dict_episode(self, T_ep=6, goal_dim=3):
        return {
            "states": {"rgb": T.randint(0, 255, (T_ep, 8, 8, 3), dtype=T.uint8),
                       "vec": T.randn(T_ep, 5)},
            "actions": T.randn(T_ep, 2),
            "rewards": -T.ones(T_ep),
            "next_states": {"rgb": T.randint(0, 255, (T_ep, 8, 8, 3), dtype=T.uint8),
                            "vec": T.randn(T_ep, 5)},
            "terminations": T.zeros(T_ep, dtype=T.bool),
            "truncations": T.zeros(T_ep, dtype=T.bool),
            "state_achieved_goals": T.randn(T_ep, goal_dim),
            "next_state_achieved_goals": T.randn(T_ep, goal_dim),
            "desired_goals": T.randn(T_ep, goal_dim),
        }

    @pytest.fixture()
    def goal_env(self):
        env = GymnasiumWrapper(
            cfg="PhoenXGoal-v0", num_envs=2, seed=0,
            obs_key="observation", goal_key="desired_goal", ach_goal_key="achieved_goal",
        )
        yield env
        try:
            env.close()
        except Exception:
            pass

    def test_nstep_relabel_gathers_dict_states(self, goal_env):
        relabeler = HindsightRelabeler(goal_env, strategy="final", output_format="n_step",
                                       N=3, device=DEVICE)
        episode = self._dict_episode(T_ep=6)
        out = relabeler.relabel_episode(episode)
        assert out is not None
        assert isinstance(out["states"], dict)
        assert out["states"]["rgb"].shape == (6, 3, 8, 8, 3)
        assert out["states"]["rgb"].dtype == T.uint8
        # repeat padding: the final window's steps past the episode end repeat
        # the last real step
        assert T.equal(out["states"]["vec"][5, 1], episode["states"]["vec"][5])
        assert T.equal(out["states"]["vec"][5, 2], episode["states"]["vec"][5])
        # rewards recomputed against the final achieved goal: the last window's
        # first step compares next_ach[5] to itself -> success (reward 0)
        assert float(out["rewards"][5, 0]) == pytest.approx(0.0)

    def test_flat_relabel_passes_dict_states_through(self, goal_env):
        relabeler = HindsightRelabeler(goal_env, strategy="final", output_format="flat",
                                       device=DEVICE)
        episode = self._dict_episode(T_ep=5)
        out = relabeler.relabel_episode(episode)
        assert isinstance(out, list) and len(out) == 1
        traj = out[0]
        assert isinstance(traj["states"], dict)
        for key in ("rgb", "vec"):
            assert T.equal(traj["states"][key].cpu(), episode["states"][key].cpu())
        # desired goals rewritten to the final achieved goal
        assert T.allclose(traj["desired_goals"][0],
                          episode["next_state_achieved_goals"][4].to(traj["desired_goals"].dtype))
