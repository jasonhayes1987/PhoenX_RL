"""Tests for the single-source n-step horizon and unknown-config-key rejection.

Covers:
    * ``resolve_n_step``: ``agent.config.N`` fans out to the replay buffer,
      the ``VectorNStepReward`` wrapper (added when missing, as the outermost
      wrapper, inheriting env obs/goal keys) and an ``n_step`` HER relabeler;
      older configs that repeat a consistent N still load; a mismatch raises.
    * End to end: a SAC config that sets N only under ``agent.config`` builds
      a trainer whose agent, buffer and wrapper all agree.
    * ``reject_unknown_keys`` and every construction site that used to accept
      a misspelled key silently: agent kwargs, normalizer kwargs, the n-step
      wrapper, heads, ``roots`` / ``trunk`` subnetworks, the ``model`` block,
      and layer params.
    * ``context_length`` is now a real Agent parameter that survives
      ``get_config`` (it used to be lost on reload).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from phoenx.config_utils import reject_unknown_keys, resolve_n_step

FIXTURES = Path(__file__).parent / "fixtures"


def _cpu(obj):
    """Recursively replace device strings so fixtures run on CPU-only hosts."""
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu(v) for v in obj]
    return "cpu" if obj == "cuda" else obj


def _load(name: str) -> dict:
    return _cpu(yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8")))


def _sac_cfg(**agent_extra) -> dict:
    return {
        "agent": {"type": "SAC", "config": {**agent_extra}},
        "env": {"type": "gymnasium", "config": {"cfg": "Pendulum-v1", "obs_key": None,
                                                "goal_key": None, "ach_goal_key": None}},
        "buffer": {"type": "ReplayBuffer", "config": {"buffer_size": 100}},
    }


# =============================================================================
# resolve_n_step
# =============================================================================

class TestResolveNStep:
    def test_agent_n_fans_out_to_buffer_and_new_wrapper(self):
        cfg = resolve_n_step(_sac_cfg(N=3))
        assert cfg["buffer"]["config"]["N"] == 3
        (wrapper,) = cfg["env"]["config"]["wrappers"]
        assert wrapper == {"type": "VectorNStepReward",
                           "params": {"n": 3, "obs_key": None, "goal_key": None, "ach_goal_key": None}}

    def test_default_is_one_and_is_written_to_the_agent(self):
        cfg = resolve_n_step(_sac_cfg())
        assert cfg["agent"]["config"]["N"] == 1
        assert cfg["buffer"]["config"]["N"] == 1

    def test_wrapper_is_appended_last_so_it_is_outermost(self):
        cfg = _sac_cfg(N=2)
        cfg["env"]["config"]["wrappers"] = [{"type": "VectorOneHotObservation", "params": {}}]
        resolve_n_step(cfg)
        assert [w["type"] for w in cfg["env"]["config"]["wrappers"]] == [
            "VectorOneHotObservation", "VectorNStepReward"]

    def test_wrapper_keys_come_from_env_config(self):
        cfg = _sac_cfg(N=2)
        cfg["env"]["config"].update(obs_key="observation", goal_key="desired_goal",
                                    ach_goal_key="achieved_goal")
        resolve_n_step(cfg)
        params = cfg["env"]["config"]["wrappers"][0]["params"]
        assert params["obs_key"] == "observation"
        assert params["goal_key"] == "desired_goal"
        assert params["ach_goal_key"] == "achieved_goal"

    def test_existing_wrapper_gets_n_and_keeps_explicit_keys(self):
        cfg = _sac_cfg(N=4)
        cfg["env"]["config"]["obs_key"] = "env_level"
        cfg["env"]["config"]["wrappers"] = [
            {"type": "VectorNStepReward", "params": {"obs_key": "explicit"}}]
        resolve_n_step(cfg)
        (wrapper,) = cfg["env"]["config"]["wrappers"]
        assert wrapper["params"]["n"] == 4
        assert wrapper["params"]["obs_key"] == "explicit"

    def test_legacy_config_with_agreeing_copies_still_loads(self):
        cfg = _load("sac.yml")  # still spells N in three places, all equal
        resolve_n_step(cfg)
        assert cfg["buffer"]["config"]["N"] == cfg["agent"]["config"]["N"]

    def test_n_found_only_on_buffer_is_adopted(self):
        cfg = _sac_cfg()
        cfg["buffer"]["config"]["N"] = 5
        resolve_n_step(cfg)
        assert cfg["agent"]["config"]["N"] == 5
        assert cfg["env"]["config"]["wrappers"][0]["params"]["n"] == 5

    @pytest.mark.parametrize("where", ["buffer", "wrapper", "hindsight"])
    def test_mismatch_raises_and_names_both_places(self, where):
        cfg = _sac_cfg(N=3)
        if where == "buffer":
            cfg["buffer"]["config"]["N"] = 2
        elif where == "wrapper":
            cfg["env"]["config"]["wrappers"] = [{"type": "VectorNStepReward", "params": {"n": 2}}]
        else:
            cfg["buffer"]["config"]["hindsight"] = {"output_format": "n_step", "N": 2}
        with pytest.raises(ValueError, match="agent.config.N=3") as exc:
            resolve_n_step(cfg)
        assert "=2" in str(exc.value)

    def test_n_step_hindsight_gets_n_and_flat_hindsight_does_not(self):
        cfg = _sac_cfg(N=3)
        cfg["buffer"]["config"]["hindsight"] = {"output_format": "n_step"}
        resolve_n_step(cfg)
        assert cfg["buffer"]["config"]["hindsight"]["N"] == 3

        flat = {"agent": {"type": "PPO", "config": {}},
                "buffer": {"type": "TrajectoryBuffer",
                           "config": {"hindsight": {"output_format": "flat"}}}}
        resolve_n_step(flat)
        assert "N" not in flat["buffer"]["config"]["hindsight"]

    def test_on_policy_config_is_left_alone(self):
        cfg = {"agent": {"type": "PPO", "config": {"discount": 0.99}},
               "env": {"type": "gymnasium", "config": {"cfg": "CartPole-v1", "wrappers": []}},
               "buffer": {"type": "RolloutBuffer", "config": {}}}
        before = copy.deepcopy(cfg)
        assert resolve_n_step(cfg) == before

    def test_single_source_config_builds_consistent_trainer(self, tmp_path):
        from phoenx.builder import build_trainer_from_config

        cfg = _load("smoke_sac_pendulum.yml")
        assert "N" not in cfg["buffer"]["config"]
        assert cfg["env"]["config"]["wrappers"] == []
        cfg["agent"]["config"]["N"] = 3
        cfg["save_dir"] = str(tmp_path)
        trainer = build_trainer_from_config(cfg)
        assert trainer.agent.N == 3
        assert trainer.buffer.N == 3
        assert trainer.env._find_nstep_wrapper().n == 3


# =============================================================================
# Unknown-key rejection
# =============================================================================

class TestRejectUnknownKeys:
    def test_suggests_close_match(self):
        with pytest.raises(TypeError, match=r"'polcy_clip' \(did you mean 'policy_clip'\?\)"):
            reject_unknown_keys("PPO", ["polcy_clip"], ["policy_clip", "value_clip"])

    def test_private_keys_allowed_by_default(self):
        reject_unknown_keys("PPO", ["_diag_freq"], ["policy_clip"])

    def test_private_keys_rejected_when_disallowed(self):
        with pytest.raises(TypeError):
            reject_unknown_keys("layer", ["_x"], ["units"], allow_private=False)


def _ppo_cfg(tmp_path) -> dict:
    cfg = _load("smoke_ppo_cartpole.yml")
    cfg["save_dir"] = str(tmp_path)
    return cfg


def _policy_branch(cfg: dict) -> dict:
    return cfg["agent"]["config"]["model"]["branches"]["policy"]


class TestTyposStopTheBuild:
    """Each case used to train silently on a default value."""

    def _build(self, cfg):
        from phoenx.builder import build_trainer_from_config
        return build_trainer_from_config(cfg)

    def test_baseline_fixture_builds(self, tmp_path):
        self._build(_ppo_cfg(tmp_path))

    def test_agent_kwarg_typo(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        cfg["agent"]["config"]["polcy_clip"] = 0.1
        with pytest.raises(TypeError, match="did you mean 'policy_clip'"):
            self._build(cfg)

    def test_agent_private_knob_still_allowed(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        cfg["agent"]["config"]["_diag_freq"] = 5
        assert self._build(cfg).agent._diag_freq == 5

    def test_layer_param_typo(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        layer = next(l for l in _policy_branch(cfg)["layer_config"] if l["type"] == "dense")
        layer["params"]["kernal"] = "orthogonal"
        with pytest.raises(TypeError, match="did you mean 'kernel'"):
            self._build(cfg)

    def test_head_key_typo(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        branch = _policy_branch(cfg)
        branch["layer_confg"] = branch.pop("layer_config")
        with pytest.raises(TypeError, match="did you mean 'layer_config'"):
            self._build(cfg)

    def test_model_block_typo(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        model = cfg["agent"]["config"]["model"]
        model["branchs"] = model.pop("branches")
        with pytest.raises(TypeError, match="did you mean 'branches'"):
            self._build(cfg)

    def test_normalizer_typo(self, tmp_path):
        cfg = _ppo_cfg(tmp_path)
        cfg["agent"]["config"]["state_normalizer"] = {
            "type": "RunningNorm", "config": {"clip_val": 5.0, "device": "cpu"}}
        with pytest.raises(TypeError, match="did you mean 'clip_value'"):
            self._build(cfg)


class TestDirectConstruction:
    def test_subnetwork_typo(self):
        from phoenx.models import ModularModel

        with pytest.raises(TypeError, match="did you mean 'input_keys'"):
            ModularModel._subnet_from_config({"layer_config": [], "input_key": ["rgb"]}, "rgb")

    def test_nstep_wrapper_typo(self):
        import gymnasium as gym
        from phoenx.env_wrapper import VectorNStepReward

        env = gym.make_vec("CartPole-v1", num_envs=1)
        try:
            with pytest.raises(TypeError, match="did you mean 'obs_key'"):
                VectorNStepReward(env, n=1, obs_ky="x")
        finally:
            env.close()

    def test_strict_layer_unknown_param(self):
        from phoenx.models import build_layer

        with pytest.raises(TypeError, match="did you mean 'hidden_size'"):
            build_layer("lstm", {"hiden_size": 8})

    def test_passthrough_layer_still_uses_torch_validation(self):
        from phoenx.models import build_layer

        build_layer("dropout", {"p": 0.1})
        with pytest.raises(TypeError):
            build_layer("dropout", {"q": 0.1})


def test_context_length_is_a_real_parameter_and_round_trips(tmp_path):
    from phoenx.builder import build_trainer_from_config

    cfg = _ppo_cfg(tmp_path)
    cfg["agent"]["config"]["context_length"] = 4
    agent = build_trainer_from_config(cfg).agent
    assert agent.context_length == 4
    assert agent.get_config()["config"]["context_length"] == 4
