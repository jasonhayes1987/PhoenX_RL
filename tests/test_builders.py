"""Builder-contract tests for the ``agent.config.model`` migration.

Covers what the removal of the legacy per-model YAML paths (``create_policy``
/ ``create_actor`` / ``create_value`` / ``create_critic``, and the legacy
branch of ``apply_model_config``) and the ``Agent.intrinsic_motivation`` /
``Trainer.step`` fixes introduced:

    * a dependency-free lint over every bundled example config, proving each
      one already speaks the canonical ``agent.config.model``
      roots/trunk/branches schema;
    * every per-algorithm builder rejecting a legacy-shaped (or missing
      ``model:``) config with a ``ValueError`` that names the algorithm;
    * ``apply_model_config`` treating ``model: null`` / ``model: {}`` as
      absent;
    * the Reinforce / ActorCritic builders actually wiring
      ``state_normalizer`` / ``advantage_normalizer`` / ``reward_normalizer``
      / ``entropy_schedule`` (plus ``goal_normalizer`` for ActorCritic) onto
      the built agent, instead of silently dropping them;
    * ``Agent.intrinsic_motivation`` defaulting to ``None`` on agents with an
      empty ``IM_ATTRS`` (Reinforce, ActorCritic), and ``Trainer.step`` no
      longer raising ``AttributeError`` while reading it.
"""

from __future__ import annotations

import pytest
import torch as T

from phoenx.builder import (
    apply_model_config,
    available_example_configs,
    build_agent,
    build_trainer_from_config,
    create_env,
    load_config,
)
from phoenx.env_wrapper import GymnasiumWrapper
from phoenx.models import StochasticDiscreteHead, ValueHead
from phoenx.normalizer import BatchNorm, RewardNorm, RunningNorm
from phoenx.rl_agents import ActorCritic, Reinforce
from phoenx.schedulers import ScheduleWrapper

DEV = "cuda" if T.cuda.is_available() else "cpu"

DENSE = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                               "kernel_params": {"gain": 1.41421356}}}
RELU = {"type": "relu"}
OUT = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 3e-4}}

CARTPOLE_ENV_CONFIG = {"type": "gymnasium", "config": {
    "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
    "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 3}}


# =============================================================================
# 1. Bundled-config schema lint (dependency-free: yaml + importlib.resources)
# =============================================================================
class TestBundledConfigSchema:
    """Every bundled example config must already speak the canonical schema."""

    def test_available_example_configs_non_empty(self):
        """The enumeration this lint drives off of is not accidentally empty."""
        names = available_example_configs()
        assert names, "available_example_configs() returned no bundled configs"

    @pytest.mark.parametrize("name", available_example_configs())
    def test_bundled_config_uses_modular_schema(self, name):
        """Each bundled config declares a non-empty, explicitly-typed
        ``agent.config.model`` and carries none of the deleted legacy keys."""
        config = load_config(name)
        agent_cfg = config["agent"]["config"]

        model = agent_cfg.get("model")
        assert model, f"{name}: agent.config.model is missing or empty"

        # `apply_model_config` accepts both the flat model form and the
        # nested `{"type", "config"}` form; mirror that here.
        inner = model.get("config", model)
        branches = inner.get("branches") or {}
        assert branches, f"{name}: agent.config.model has no branches"
        for role, branch in branches.items():
            head_type = branch.get("type") if isinstance(branch, dict) else None
            assert isinstance(branch, dict) and head_type, (
                f"{name}: branch {role!r} has no explicit 'type'"
            )

        assert "models" not in config, f"{name}: stale top-level 'models:' key"
        assert "normalizers" not in config, f"{name}: stale top-level 'normalizers:' key"
        stale_lr_schedule_keys = [k for k in config if str(k).endswith("_lr_schedule")]
        assert not stale_lr_schedule_keys, (
            f"{name}: stale top-level '*_lr_schedule' key(s): {stale_lr_schedule_keys}"
        )
        for legacy_key in ("policy", "value", "critic", "critic_b"):
            assert legacy_key not in agent_cfg, (
                f"{name}: flat legacy '{legacy_key}:' head dict directly under agent.config"
            )


# =============================================================================
# 2. Every builder requires agent.config.model
# =============================================================================
LEGACY_AGENT_CONFIGS = {
    "PPO": {"name": "PPO", "policy": {"layer_config": []}, "value": {"layer_config": []}},
    "SAC": {"name": "SAC", "policy": {"layer_config": []}, "critic": {"layer_config": []}},
    "DDPG": {"name": "DDPG", "policy": {"layer_config": []}, "critic": {"layer_config": []}},
    "TD3": {"name": "TD3", "policy": {"layer_config": []}, "critic": {"layer_config": []}},
    "Reinforce": {"name": "Reinforce", "policy": {"layer_config": []}, "value": {"layer_config": []}},
    "ActorCritic": {"name": "ActorCritic", "policy": {"layer_config": []}, "value": {"layer_config": []}},
}


class TestModelSchemaRequired:
    """``apply_model_config`` (and every builder that calls it first) must
    reject configs without a non-empty ``agent.config.model``."""

    @pytest.mark.parametrize("algo", sorted(LEGACY_AGENT_CONFIGS))
    def test_builder_rejects_legacy_shaped_config(self, algo):
        """Driving a legacy flat-head-dict config through
        ``phoenx.builder.build_agent`` (the actual dispatch path each
        algorithm's real training run uses) raises a ``ValueError`` naming
        both the algorithm and ``'agent.config.model'``. ``env=None`` is
        safe here: ``apply_model_config`` raises before touching ``env``."""
        config = {"agent": {"type": algo, "config": dict(LEGACY_AGENT_CONFIGS[algo])}}
        with pytest.raises(ValueError) as excinfo:
            build_agent(config, env=None)
        message = str(excinfo.value)
        assert algo in message
        assert "agent.config.model" in message

    @pytest.mark.parametrize("algo", sorted(LEGACY_AGENT_CONFIGS))
    @pytest.mark.parametrize("model_value", [None, {}], ids=["null", "empty"])
    def test_builder_rejects_null_or_empty_model(self, algo, model_value):
        """``model: null`` / ``model: {}`` are as-absent through the same
        real dispatch path as the legacy-shape test above."""
        config = {"agent": {"type": algo, "config": {"name": algo, "model": model_value}}}
        with pytest.raises(ValueError, match="agent.config.model"):
            build_agent(config, env=None)

    def test_apply_model_config_direct_raises_and_names_algo(self):
        """A direct ``apply_model_config`` call (no builder indirection)
        raises the same message shape."""
        with pytest.raises(ValueError) as excinfo:
            apply_model_config({}, env=None, algo="SAC")
        message = str(excinfo.value)
        assert "SAC" in message
        assert "agent.config.model" in message

    @pytest.mark.parametrize("model_value", [None, {}], ids=["null", "empty"])
    def test_apply_model_config_treats_null_and_empty_as_absent(self, model_value):
        """``model: null`` / ``model: {}`` are both falsy, so
        ``apply_model_config`` treats them identically to a missing key."""
        with pytest.raises(ValueError, match="agent.config.model"):
            apply_model_config({"model": model_value}, env=None, algo="PPO")


# =============================================================================
# 3. Reinforce / ActorCritic: normalizers + entropy schedule reach the agent
# =============================================================================
def _normalizer_regression_config(agent_type: str, policy_branch: dict, env_cfg: dict) -> dict:
    """Build an ``agent.config.model`` config declaring every builder-owned
    normalizer plus an entropy schedule, for the silent-drop regression guard.

    Args:
        agent_type: ``"Reinforce"`` or ``"ActorCritic"``.
        policy_branch: Branch dict for the ``policy`` role (head type varies
            with the env's action space).
        env_cfg: Top-level ``env`` config section.

    Returns:
        Full training config with ``agent`` and ``env`` sections set.
    """
    return {
        "agent": {"type": agent_type, "config": {
            "name": agent_type,
            "model": {"branches": {
                "policy": policy_branch,
                "value": {"type": "ValueHead", "layer_config": [DENSE(16), RELU],
                          "output_config": OUT, "optimizer_params": OPT, "device": DEV},
            }},
            "state_normalizer": {"type": "RunningNorm", "config": {"clip_value": 10.0, "device": DEV}},
            "advantage_normalizer": {"type": "BatchNorm", "config": {"clip_value": 10.0, "device": DEV}},
            "reward_normalizer": {"type": "RewardNorm",
                                  "config": {"gamma": 0.99, "clip_value": 10.0, "device": DEV}},
            "entropy_schedule": {"schedule_type": "linear", "steps": 1000,
                                 "start_value": 0.05, "end_value": 0.0},
            "discount": 0.99, "auto_entropy_tuning": False, "device": DEV, "log_level": "ERROR",
        }},
        "env": env_cfg,
    }


class TestReinforceActorCriticNormalizersReachAgent:
    """``builders/reinforce.py`` and ``builders/actor_critic.py`` must read
    normalizers / ``entropy_schedule`` from ``agent.config`` (not the deleted
    top-level ``config['normalizers']`` / ``config['entropy_schedule']``),
    and must force ``num_features: 1`` on the advantage normalizer."""

    def test_reinforce_normalizers_and_entropy_schedule_land_on_agent(self):
        config = _normalizer_regression_config(
            "Reinforce",
            {"type": "StochasticDiscreteHead", "layer_config": [DENSE(16), RELU],
             "output_config": OUT, "distribution": "categorical",
             "optimizer_params": OPT, "device": DEV},
            CARTPOLE_ENV_CONFIG,
        )
        env = create_env(config)
        try:
            agent = build_agent(config, env)
            assert isinstance(agent.state_normalizer, RunningNorm)
            assert isinstance(agent.advantage_normalizer, BatchNorm)
            assert agent.advantage_normalizer.num_features == 1
            assert isinstance(agent.reward_normalizer, RewardNorm)
            assert isinstance(agent.entropy_schedule, ScheduleWrapper)
        finally:
            env.close()

    def test_actor_critic_normalizers_and_entropy_schedule_land_on_agent(self):
        """Uses the ``PhoenXGoal-v0`` synthetic env (registered in
        ``conftest.py``) so ``goal_normalizer`` coverage is cheap: it is a
        goal-conditioned Dict env with a real ``desired_goal`` key."""
        env_cfg = {"type": "gymnasium", "config": {
            "cfg": "PhoenXGoal-v0", "num_envs": 2, "obs_key": "observation",
            "goal_key": "desired_goal", "ach_goal_key": "achieved_goal",
            "wrappers": [], "render_mode": None, "seed": 3}}
        config = _normalizer_regression_config(
            "ActorCritic",
            {"type": "StochasticContinuousHead", "layer_config": [DENSE(16), RELU],
             "output_config": OUT, "distribution": "normal",
             "optimizer_params": OPT, "device": DEV},
            env_cfg,
        )
        config["agent"]["config"]["goal_normalizer"] = {
            "type": "RunningNorm", "config": {"clip_value": 10.0, "device": DEV},
        }
        env = create_env(config)
        try:
            agent = build_agent(config, env)
            assert isinstance(agent.state_normalizer, RunningNorm)
            assert isinstance(agent.goal_normalizer, RunningNorm)
            assert isinstance(agent.advantage_normalizer, BatchNorm)
            assert agent.advantage_normalizer.num_features == 1
            assert isinstance(agent.reward_normalizer, RewardNorm)
            assert isinstance(agent.entropy_schedule, ScheduleWrapper)
        finally:
            env.close()


# =============================================================================
# 4. intrinsic_motivation defaults to None; Trainer.step tolerates it
# =============================================================================
@pytest.fixture(scope="module")
def cartpole():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


class TestIntrinsicMotivationDefaultsToNone:
    """``Agent.__init__`` now sets ``self.intrinsic_motivation = None``
    unconditionally, so subclasses with an empty ``IM_ATTRS`` (Reinforce,
    ActorCritic) still expose the attribute instead of omitting it."""

    def test_reinforce_intrinsic_motivation_defaults_to_none(self, cartpole):
        policy = StochasticDiscreteHead(cartpole, layer_config=[DENSE(16), RELU],
                                        output_config=OUT, device=DEV)
        agent = Reinforce(policy=policy, auto_entropy_tuning=False, device=DEV)
        assert agent.intrinsic_motivation is None

    def test_actor_critic_intrinsic_motivation_defaults_to_none(self, cartpole):
        policy = StochasticDiscreteHead(cartpole, layer_config=[DENSE(16), RELU],
                                        output_config=OUT, device=DEV)
        value = ValueHead(cartpole, layer_config=[DENSE(16), RELU], output_config=OUT, device=DEV)
        agent = ActorCritic(policy=policy, value=value, auto_entropy_tuning=False, device=DEV)
        assert agent.intrinsic_motivation is None


class TestTrainerStepToleratesMissingIntrinsicMotivation:
    """``Trainer.step`` hoists ``im = getattr(self.agent, 'intrinsic_motivation',
    None)`` and tests ``im is not None`` for its log line, instead of reading
    ``self.agent.intrinsic_motivation`` directly (which raised
    ``AttributeError`` pre-fix for any agent with empty ``IM_ATTRS``)."""

    @pytest.mark.parametrize("agent_type", ["Reinforce", "ActorCritic"])
    def test_step_does_not_raise_and_logs_zero_intrinsic_reward(self, agent_type, tmp_path):
        config = {
            "save_dir": str(tmp_path / agent_type) + "/",
            "log_level": "ERROR",
            "schedule": {"stop_unit": "timestep", "stop_units": 4, "learn_every_unit": "timestep",
                        "learn_every": 999_999, "updates_per_learn": 1, "batch_size": 1,
                        "warmup_steps": 0, "seed": 3},
            "agent": {"type": agent_type, "config": {
                "name": agent_type,
                "model": {"branches": {
                    "policy": {"type": "StochasticDiscreteHead", "layer_config": [DENSE(16), RELU],
                               "output_config": OUT, "distribution": "categorical",
                               "optimizer_params": OPT, "device": DEV},
                    "value": {"type": "ValueHead", "layer_config": [DENSE(16), RELU],
                              "output_config": OUT, "optimizer_params": OPT, "device": DEV},
                }},
                "discount": 0.99, "auto_entropy_tuning": False, "device": DEV, "log_level": "ERROR",
            }},
            "env": CARTPOLE_ENV_CONFIG,
            "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
        }
        trainer = build_trainer_from_config(config)
        try:
            trainer._initialize_run(context="train")
            result = trainer.step(training=True)  # must not raise AttributeError
            assert result["step_log"]["step_intrinsic_reward"] == 0.0
        finally:
            trainer.env.close()
