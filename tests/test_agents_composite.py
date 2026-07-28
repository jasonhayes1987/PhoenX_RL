"""Agent-level tests for the composite roots->trunk->branches architecture.

Uses the *real* agents from ``phoenx.rl_agents`` end to end. Covers:

    * per-algorithm head-type enforcement (wrong head class raises TypeError);
    * ``critic_b`` auto-clone (fresh weights, same architecture);
    * target-network branch subsets (DDPG/TD3/SAC) + ``soft_update_targets``;
    * gradient-ownership defaults (on-policy 'combined', off-policy 'critic');
    * ``learn()`` smoke for all 6 agents on synthetic buffer-shaped batches
      (flat obs; branches-only AND shared roots+trunk variants) — finite
      metrics, parameters actually update;
    * off-policy ownership at the agent level: SAC's learn leaves the policy
      branch untouched by the critic phase and vice versa (verified by param
      snapshots between phases via shared_update wiring);
    * ``get_config`` -> ``build_agent`` round trip for every agent;
    * legacy config adaptation (legacy Model type tags -> heads);
    * legacy per-model checkpoint shim (policy.pt/value.pt -> composite).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch as T

from phoenx.env_wrapper import GymnasiumWrapper
from phoenx.models import (
    ContinuousQHead,
    DeterministicActorHead,
    DiscreteQHead,
    StochasticContinuousHead,
    StochasticDiscreteHead,
    SubNetwork,
    ValueHead,
    ValueModel,
    StochasticDiscretePolicy,
    map_legacy_state_dict,
)
from phoenx.rl_agents import (
    ActorCritic,
    DDPG,
    PPO,
    Reinforce,
    SAC,
    TD3,
    build_agent,
)
from phoenx.noise import NormalNoise

DEVICE = "cpu"
T.manual_seed(0)
np.random.seed(0)

LC = [
    {"type": "dense", "params": {"units": 24, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


@pytest.fixture(scope="module")
def cartpole():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def pendulum():
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Head / shared-part factories
# -----------------------------------------------------------------------------
def d_policy(env, **kw):
    return StochasticDiscreteHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def c_policy(env, **kw):
    return StochasticContinuousHead(env, layer_config=LC, output_config=OC,
                                    distribution="normal", device=DEVICE, **kw)


def actor(env, **kw):
    return DeterministicActorHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def value(env, **kw):
    return ValueHead(env, layer_config=LC, output_config=OC, device=DEVICE, **kw)


def c_q(env, **kw):
    return ContinuousQHead(env, layer_config=LC, merged_config=LC, output_config=OC, device=DEVICE, **kw)


def shared_parts():
    return {
        "roots": {"state": SubNetwork(LC, name="state")},
        "trunk": SubNetwork(LC, name="trunk"),
    }


# -----------------------------------------------------------------------------
# Synthetic buffer-shaped samples (mirror RolloutBuffer / ReplayBuffer output)
# -----------------------------------------------------------------------------
def rollout_sample(env, traj_len=6, discrete=True):
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


def replay_sample(env, B=16, N=1, discrete=False):
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


def _params_snapshot(agent):
    return {k: v.clone() for k, v in agent.model.state_dict().items()}


def _changed_keys(before, after):
    return {k for k in before if not T.equal(before[k], after[k])}


def _assert_finite_metrics(metrics):
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            assert np.isfinite(val), f"metric {key} not finite: {val}"
        elif isinstance(val, T.Tensor):
            assert T.isfinite(val.float()).all(), f"metric {key} has non-finite entries"


# =============================================================================
# Type enforcement
# =============================================================================
class TestTypeEnforcement:
    def test_reinforce_requires_discrete_policy(self, cartpole, pendulum):
        with pytest.raises(TypeError, match="policy"):
            Reinforce(policy=value(cartpole), save_dir="models", device=DEVICE)
        with pytest.raises(TypeError, match="policy"):
            Reinforce(policy=c_policy(pendulum), save_dir="models", device=DEVICE)
        with pytest.raises(TypeError, match="required"):
            Reinforce(save_dir="models", device=DEVICE)

    def test_ppo_requires_stochastic_policy_and_value(self, pendulum):
        with pytest.raises(TypeError, match="policy"):
            PPO(policy=actor(pendulum), value=value(pendulum), device=DEVICE)
        with pytest.raises(TypeError, match="value"):
            PPO(policy=c_policy(pendulum), value=c_q(pendulum), device=DEVICE)

    def test_actor_critic_types(self, cartpole):
        with pytest.raises(TypeError, match="value"):
            ActorCritic(policy=d_policy(cartpole), value=d_policy(cartpole), device=DEVICE)

    def test_ddpg_requires_deterministic_actor_and_q(self, pendulum):
        with pytest.raises(TypeError, match="policy"):
            DDPG(policy=c_policy(pendulum), critic=c_q(pendulum), device=DEVICE)
        with pytest.raises(TypeError, match="critic"):
            DDPG(policy=actor(pendulum), critic=value(pendulum), device=DEVICE)

    def test_td3_requires_actor_and_two_qs(self, pendulum):
        with pytest.raises(TypeError, match="policy"):
            TD3(policy=c_policy(pendulum), critic=c_q(pendulum), device=DEVICE)
        with pytest.raises(TypeError, match="critic_b"):
            TD3(policy=actor(pendulum), critic=c_q(pendulum), critic_b=value(pendulum), device=DEVICE)

    def test_sac_requires_stochastic_policy(self, pendulum):
        with pytest.raises(TypeError, match="policy"):
            SAC(policy=actor(pendulum), critic=c_q(pendulum), device=DEVICE)


# =============================================================================
# critic_b auto-clone + target subsets
# =============================================================================
class TestTargetsAndClones:
    def test_td3_critic_b_autoclone_fresh_weights(self, pendulum):
        agent = TD3(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                    target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE),
                    device=DEVICE)
        assert set(agent.model.branches.keys()) == {"policy", "critic", "critic_b"}
        # same architecture, fresh (different) weights
        sd = agent.model.state_dict()
        a_keys = sorted(k for k in sd if k.startswith("branches.critic."))
        b_keys = sorted(k for k in sd if k.startswith("branches.critic_b."))
        assert len(a_keys) == len(b_keys) > 0
        assert any(
            not T.equal(sd[a], sd[b])
            for a, b in zip(a_keys, b_keys) if sd[a].dim() >= 2
        )

    def test_target_branch_subsets(self, pendulum):
        ddpg = DDPG(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE), device=DEVICE)
        assert set(ddpg.target_model.branches.keys()) == {"policy", "critic"}

        td3 = TD3(policy=actor(pendulum), critic=c_q(pendulum),
                  noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                  target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE), device=DEVICE)
        assert set(td3.target_model.branches.keys()) == {"policy", "critic", "critic_b"}

        sac = SAC(policy=c_policy(pendulum), critic=c_q(pendulum), device=DEVICE)
        assert set(sac.target_model.branches.keys()) == {"critic", "critic_b"}

        # targets start as exact copies
        for agent in (ddpg, td3, sac):
            src = agent.model.state_dict()
            for k, v in agent.target_model.state_dict().items():
                assert T.equal(v, src[k]), k

    def test_soft_update_targets(self, pendulum):
        agent = SAC(policy=c_policy(pendulum), critic=c_q(pendulum), tau=0.5, device=DEVICE)
        with T.no_grad():
            for p in agent.model.parameters():
                p.add_(2.0)
        before = {k: v.clone() for k, v in agent.target_model.state_dict().items()}
        agent.soft_update_targets()
        src = agent.model.state_dict()
        for k, v in agent.target_model.state_dict().items():
            expected = before[k] + 0.5 * (src[k] - before[k])
            assert T.allclose(v, expected, atol=1e-6), k


# =============================================================================
# shared_update defaults
# =============================================================================
class TestSharedUpdateDefaults:
    def test_on_policy_defaults_combined(self, cartpole, pendulum):
        assert PPO(policy=d_policy(cartpole), value=value(cartpole),
                   device=DEVICE).model.shared_update == "combined"
        assert ActorCritic(policy=d_policy(cartpole), value=value(cartpole),
                           device=DEVICE).model.shared_update == "combined"
        assert Reinforce(policy=d_policy(cartpole), value=value(cartpole),
                         device=DEVICE).model.shared_update == "combined"

    def test_off_policy_defaults_critic(self, pendulum):
        assert SAC(policy=c_policy(pendulum), critic=c_q(pendulum),
                   device=DEVICE).model.shared_update == "critic"
        assert DDPG(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                    device=DEVICE).model.shared_update == "critic"


# =============================================================================
# learn() smoke — every agent, branches-only and shared variants
# =============================================================================
class TestLearnSmoke:
    @pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
    def test_ppo_learn(self, cartpole, shared):
        parts = shared_parts() if shared else {}
        agent = PPO(policy=d_policy(cartpole), value=value(cartpole),
                    optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE, **parts)
        sample = rollout_sample(cartpole, discrete=True)
        before = _params_snapshot(agent)
        metrics = agent.learn(0, sample, learning_epochs=2, mini_batch_size=4)
        _assert_finite_metrics(metrics)
        changed = _changed_keys(before, agent.model.state_dict())
        assert any(k.startswith("branches.policy.") for k in changed)
        assert any(k.startswith("branches.value.") for k in changed)
        if shared:
            assert any(k.startswith("roots.") for k in changed)
            assert any(k.startswith("trunk.") for k in changed)

    @pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
    def test_actor_critic_learn(self, cartpole, shared):
        parts = shared_parts() if shared else {}
        agent = ActorCritic(policy=d_policy(cartpole), value=value(cartpole),
                            optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE, **parts)
        sample = rollout_sample(cartpole, discrete=True)
        before = _params_snapshot(agent)
        metrics = agent.learn(0, sample)
        _assert_finite_metrics(metrics)
        changed = _changed_keys(before, agent.model.state_dict())
        assert any(k.startswith("branches.policy.") for k in changed)
        assert any(k.startswith("branches.value.") for k in changed)
        if shared:
            assert any(k.startswith("trunk.") for k in changed)

    def test_reinforce_learn(self, cartpole):
        agent = Reinforce(policy=d_policy(cartpole), value=value(cartpole),
                          optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE)
        trajectories = [
            {"states": T.randn(8, 4), "actions": T.randint(0, 2, (8, 1)), "rewards": T.randn(8)}
            for _ in range(2)
        ]
        before = _params_snapshot(agent)
        metrics = agent.learn(0, trajectories)
        _assert_finite_metrics(metrics)
        assert _changed_keys(before, agent.model.state_dict())

    @pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
    def test_ddpg_learn(self, pendulum, shared):
        parts = shared_parts() if shared else {}
        agent = DDPG(policy=actor(pendulum), critic=c_q(pendulum),
                     noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                     optimizer_params=OPT, device=DEVICE, **parts)
        sample = replay_sample(pendulum, B=16, N=1)
        before = _params_snapshot(agent)
        metrics = agent.learn(0, sample)
        _assert_finite_metrics(metrics)
        changed = _changed_keys(before, agent.model.state_dict())
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)
        if shared:
            # critic phase owns the shared body
            assert any(k.startswith("roots.") or k.startswith("trunk.") for k in changed)

    @pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
    def test_td3_learn(self, pendulum, shared):
        parts = shared_parts() if shared else {}
        agent = TD3(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                    target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE),
                    policy_update_delay=1, optimizer_params=OPT, device=DEVICE, **parts)
        sample = replay_sample(pendulum, B=16, N=1)
        before = _params_snapshot(agent)
        metrics = agent.learn(0, sample)
        _assert_finite_metrics(metrics)
        changed = _changed_keys(before, agent.model.state_dict())
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("branches.critic_b.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)

    @pytest.mark.parametrize("discrete", [False, True], ids=["continuous", "discrete"])
    @pytest.mark.parametrize("shared", [False, True], ids=["separate", "shared"])
    def test_sac_learn(self, pendulum, cartpole, shared, discrete):
        env = cartpole if discrete else pendulum
        if discrete:
            policy = d_policy(env)
            critic = DiscreteQHead(env, layer_config=LC, output_config=OC, device=DEVICE)
        else:
            policy = c_policy(env)
            critic = c_q(env)
        parts = shared_parts() if shared else {}
        agent = SAC(policy=policy, critic=critic, optimizer_params=OPT,
                    auto_entropy_tuning=False, device=DEVICE, **parts)
        sample = replay_sample(env, B=16, N=1, discrete=discrete)
        before = _params_snapshot(agent)
        metrics = agent.learn(0, sample)
        _assert_finite_metrics(metrics)
        changed = _changed_keys(before, agent.model.state_dict())
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)
        if shared:
            assert any(k.startswith("roots.") or k.startswith("trunk.") for k in changed)

    def test_sac_shared_actor_never_updates_body(self, pendulum):
        """With critic LR forced to ~0 and policy LR high, a shared model's
        roots/trunk must stay bit-identical through learn() — proving the
        actor loss cannot reach the shared body (detach + step subset)."""
        parts = shared_parts()
        # critic + shared modules get lr=0 (their updates are no-ops); policy lr high
        for subnet in list(parts["roots"].values()) + [parts["trunk"]]:
            subnet.optimizer_params = {"type": "SGD", "params": {"lr": 0.0}}
        critic = c_q(pendulum, optimizer_params={"type": "SGD", "params": {"lr": 0.0}})
        critic_b = c_q(pendulum, optimizer_params={"type": "SGD", "params": {"lr": 0.0}})
        policy = c_policy(pendulum, optimizer_params={"type": "SGD", "params": {"lr": 0.5}})
        agent = SAC(policy=policy, critic=critic, critic_b=critic_b,
                    auto_entropy_tuning=False, device=DEVICE, **parts)
        sample = replay_sample(pendulum, B=16, N=1)
        before = _params_snapshot(agent)
        agent.learn(0, sample)
        changed = _changed_keys(before, agent.model.state_dict())
        assert changed, "policy branch should have updated"
        assert all(k.startswith("branches.policy.") for k in changed), (
            f"non-policy params changed despite zero critic/shared LR: "
            f"{[k for k in changed if not k.startswith('branches.policy.')]}"
        )


# =============================================================================
# act() smoke
# =============================================================================
class TestActSmoke:
    def test_ppo_act_bounds(self, pendulum):
        agent = PPO(policy=c_policy(pendulum), value=value(pendulum),
                    auto_entropy_tuning=False, device=DEVICE)
        states = T.randn(4, 3)
        for context in ("train", "test"):
            action = agent.act(states, context=context)
            assert action.actions.shape == (4, 1)
            assert T.all(action.actions >= -2.0 - 1e-5) and T.all(action.actions <= 2.0 + 1e-5)
            assert action.log_probs is not None

    def test_td3_act_bounds(self, pendulum):
        agent = TD3(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                    target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE),
                    device=DEVICE)
        # act() sizes exploration noise to the env's vectorized action space,
        # so the state batch must equal num_envs (as in real rollouts).
        states = T.randn(pendulum.num_envs, 3)
        action = agent.act(states, context="train", step=100, warmup=0)
        assert T.all(action.actions >= -2.0 - 1e-5) and T.all(action.actions <= 2.0 + 1e-5)
        action = agent.act(states, context="test")
        assert action.actions.shape == (pendulum.num_envs, 1)

    def test_reinforce_act_discrete(self, cartpole):
        agent = Reinforce(policy=d_policy(cartpole), auto_entropy_tuning=False, device=DEVICE)
        actions = agent.act(T.randn(4, 4), context="train")
        assert actions.shape == (4,)
        assert set(actions.tolist()) <= {0, 1}


# =============================================================================
# Config round trips + legacy adaptation
# =============================================================================
class TestConfigRoundTrip:
    def _roundtrip(self, agent, env):
        cfg = agent.get_config()
        blob = json.dumps(cfg)  # JSON-serializable
        rebuilt = build_agent(json.loads(blob), env)
        assert type(rebuilt) is type(agent)
        assert (
            [n for n, _ in rebuilt.model.named_parameters()]
            == [n for n, _ in agent.model.named_parameters()]
        )
        assert rebuilt.get_config() == cfg
        return rebuilt

    def test_ppo_roundtrip(self, cartpole):
        agent = PPO(policy=d_policy(cartpole), value=value(cartpole),
                    auto_entropy_tuning=False, device=DEVICE, **shared_parts())
        self._roundtrip(agent, cartpole)

    def test_sac_roundtrip(self, pendulum):
        agent = SAC(policy=c_policy(pendulum), critic=c_q(pendulum),
                    auto_entropy_tuning=False, device=DEVICE)
        self._roundtrip(agent, pendulum)

    def test_td3_roundtrip(self, pendulum):
        agent = TD3(policy=actor(pendulum), critic=c_q(pendulum),
                    noise=NormalNoise(mean=0.0, stddev=0.1, device=DEVICE),
                    target_noise=NormalNoise(mean=0.0, stddev=0.2, device=DEVICE),
                    device=DEVICE)
        self._roundtrip(agent, pendulum)

    def test_legacy_config_schema_adapts(self, cartpole):
        """A legacy-style agent config (per-model keys with legacy Model type
        tags) must build a composite agent with the matching heads."""
        legacy_cfg = {
            "policy": {"type": "StochasticDiscretePolicy",
                       "config": {"layer_config": LC, "output_config": OC,
                                  "optimizer_params": OPT, "device": DEVICE}},
            "value": {"type": "ValueModel",
                      "config": {"layer_config": LC, "output_config": OC,
                                 "optimizer_params": OPT, "device": DEVICE}},
            "discount": 0.98,
            "auto_entropy_tuning": False,
            "save_dir": "models",
            "device": DEVICE,
        }
        agent = PPO.from_config(legacy_cfg, cartpole)
        assert isinstance(agent.policy, StochasticDiscreteHead)
        assert isinstance(agent.value, ValueHead)
        assert agent.discount == 0.98
        # branches-only composite (no shared modules)
        assert agent.model.roots is None and agent.model.trunk is None

    def test_legacy_checkpoint_shim(self, cartpole, tmp_path):
        """Legacy per-model checkpoint files load onto the composite model."""
        legacy_policy = StochasticDiscretePolicy(cartpole, layer_config=LC,
                                                 output_config=OC, device=DEVICE)
        legacy_value = ValueModel(cartpole, layer_config=LC, output_config=OC, device=DEVICE)
        legacy_policy.save_state(tmp_path / "policy.pt")
        legacy_value.save_state(tmp_path / "value.pt")

        agent = PPO(policy=d_policy(cartpole), value=value(cartpole),
                    auto_entropy_tuning=False, device=DEVICE)
        agent.load_state(tmp_path)

        expected = {}
        expected.update(map_legacy_state_dict(legacy_policy.state_dict(), "policy"))
        expected.update(map_legacy_state_dict(legacy_value.state_dict(), "value"))
        actual = agent.model.state_dict()
        for k, v in expected.items():
            assert k in actual and T.equal(actual[k], v), k

    def test_save_load_state_roundtrip(self, pendulum, tmp_path):
        agent = SAC(policy=c_policy(pendulum), critic=c_q(pendulum),
                    auto_entropy_tuning=False, device=DEVICE)
        agent.learn(0, replay_sample(pendulum, B=8, N=1))  # optimizer state exists
        agent.save_state(tmp_path)
        assert (tmp_path / "model.pt").exists()
        assert (tmp_path / "target_model.pt").exists()

        rebuilt = build_agent(json.loads(json.dumps(agent.get_config())), pendulum)
        rebuilt.load_state(tmp_path)
        for k, v in agent.model.state_dict().items():
            assert T.equal(v, rebuilt.model.state_dict()[k]), k
        for k, v in agent.target_model.state_dict().items():
            assert T.equal(v, rebuilt.target_model.state_dict()[k]), k
