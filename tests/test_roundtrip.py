"""Round-trip validation for the unified save/load architecture.

Pytest port of ``src/scripts/roundtrip_test.py`` (which it supersedes),
extended to the composite roots->trunk->branches models. Exercises the
``get_config`` / ``from_config`` / ``save_state`` / ``load_state`` contract
end to end with the REAL production stack:

    1. SAC: build from the production ``src/Configs/sac.yml`` (legacy schema,
       exercising the config adapter), shrink to a tiny CPU/GPU Pendulum run,
       train ~120 steps, ``Trainer.save``, ``Trainer.load`` and assert:
        - the rebuilt config is byte-identical,
        - every model/target tensor and per-module optimizer state matches,
        - entropy temperature and normalizer statistics match,
        - the replay buffer is restored,
        - training resumes from the saved step counter.
    2. PPO: same full-trainer round trip on the on-policy path
       (RolloutBuffer, CartPole) built from an inline legacy-schema config.
    3. DDPG: agent-level contract (build -> save_state -> build_agent +
       load_state -> tensor equality) including the target model.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch as T
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.rl_agents import build_agent  # noqa: E402
from app.trainer import Trainer  # noqa: E402
from scripts.agent import build_trainer_from_config  # noqa: E402

CONFIGS_DIR = _SRC / "Configs"

T.manual_seed(0)


# --------------------------------------------------------------------------- #
# helpers (ported verbatim in spirit from the script)
# --------------------------------------------------------------------------- #
def _shrink(config: dict, save_dir: str, env_id: str = "Pendulum-v1") -> dict:
    """Turn a production config into a tiny, fast run."""
    config = copy.deepcopy(config)
    config["save_dir"] = save_dir
    config["log_level"] = "ERROR"
    config.pop("callbacks", None)
    config.pop("renderer", None)

    sched = config.setdefault("schedule", {})
    sched.update({
        "stop_unit": "timestep", "stop_units": 120,
        "learn_every_unit": "timestep", "learn_every": 1,
        "updates_per_learn": 1, "batch_size": 32,
        "warmup_steps": 16, "seed": 7,
    })

    env_cfg = config["env"]["config"]
    env_cfg.update({"cfg": env_id, "num_envs": 1, "render_mode": None, "seed": 7})

    if config.get("buffer") is not None:
        config["buffer"] = {"type": "ReplayBuffer", "config": {"buffer_size": 500, "N": 1}}

    agent_cfg = config["agent"]["config"]
    agent_cfg["log_level"] = "ERROR"
    agent_cfg["intrinsic_motivation"] = None
    for model_key in ("policy", "critic", "critic_b"):
        model = agent_cfg.get(model_key)
        if not isinstance(model, dict):
            continue
        for layers_key in ("layer_config", "merged_config"):
            layers = model.get(layers_key)
            if layers:
                for layer in layers:
                    if layer.get("type") == "dense":
                        layer.setdefault("params", {})["units"] = 16
    return config


def _named(module) -> dict:
    return {name: p.detach().cpu() for name, p in module.named_parameters()}


def _params_match(a, b, label: str) -> None:
    pa, pb = _named(a), _named(b)
    assert pa.keys() == pb.keys(), f"{label}: parameter names differ"
    for name in pa:
        assert T.allclose(pa[name], pb[name]), f"{label}: '{name}' weights differ"


def _optimizer_states_match(model_a, model_b, label: str) -> None:
    assert set(model_a.optimizers) == set(model_b.optimizers), f"{label}: optimizer modules differ"
    for mod_name in model_a.optimizers:
        sa = model_a.optimizers[mod_name].state_dict()["state"]
        sb = model_b.optimizers[mod_name].state_dict()["state"]
        assert set(sa.keys()) == set(sb.keys()), f"{label}:{mod_name}: optimizer param set differs"
        for k in sa:
            for field, val in sa[k].items():
                if isinstance(val, T.Tensor):
                    assert T.allclose(val.cpu(), sb[k][field].cpu()), (
                        f"{label}:{mod_name}: optimizer state[{k}].{field} differs"
                    )


def _arch(cfg: dict) -> dict:
    """Strip run-location noise (save_dir path normalization) before compare."""
    cfg = dict(cfg)
    cfg.pop("save_dir", None)
    return cfg


# --------------------------------------------------------------------------- #
# 1. SAC: full Trainer round trip from the production sac.yml (legacy schema)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_sac_trainer_round_trip(tmp_path):
    run_dir = str(tmp_path / "sac_run") + os.sep
    raw = yaml.safe_load((CONFIGS_DIR / "sac.yml").read_text(encoding="utf-8"))
    config = _shrink(raw, run_dir)

    trainer = build_trainer_from_config(config)
    trainer.train()
    saved_step = trainer._step
    ref_config = json.dumps(_arch(trainer.get_config()), sort_keys=True)

    trainer.save(run_dir, save_buffer=True)
    trainer.env.close()

    trainer2 = Trainer.load(run_dir, load_weights=True, load_buffer=True)

    # 1) config is the single source of truth and survives the round trip.
    got = json.dumps(_arch(trainer2.get_config()), sort_keys=True)
    assert got == ref_config, "config (architecture) did not round-trip byte-for-byte"

    # 2) every tensor family matches (composite model + target subset).
    _params_match(trainer.agent.model, trainer2.agent.model, "model")
    _params_match(trainer.agent.target_model, trainer2.agent.target_model, "target_model")
    _optimizer_states_match(trainer.agent.model, trainer2.agent.model, "model")

    # 3) auto-entropy temperature.
    if getattr(trainer.agent, "auto_entropy_tuning", False):
        assert T.allclose(trainer.agent.log_alpha.detach().cpu(),
                          trainer2.agent.log_alpha.detach().cpu()), "log_alpha differs"

    # 4) observation normalizer running stats.
    sn1, sn2 = trainer.agent.state_normalizer, trainer2.agent.state_normalizer
    if sn1 is not None:
        assert T.allclose(sn1.running_mean.cpu(), sn2.running_mean.cpu())
        assert T.allclose(sn1.running_std.cpu(), sn2.running_std.cpu())

    # 5) replay buffer restored.
    assert trainer.buffer.samples_added == trainer2.buffer.samples_added
    assert T.allclose(trainer.buffer.states.cpu(), trainer2.buffer.states.cpu())

    # 6) resume: continue training from the saved step counter.
    trainer2.schedule.stop_units = saved_step + 24
    trainer2.train()
    assert trainer2._step >= saved_step + 24, (
        f"resume did not advance past saved step ({saved_step} -> {trainer2._step})"
    )
    trainer2.env.close()


# --------------------------------------------------------------------------- #
# 2. PPO: full Trainer round trip on the on-policy path (RolloutBuffer)
# --------------------------------------------------------------------------- #
def _ppo_config(save_dir: str) -> dict:
    # The Trainer allocates its counters on the agent's device while the env
    # wrapper follows the machine default, so the config device must match it.
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
            "policy": {"layer_config": [dense(16), relu], "output_config": out,
                       "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                       "distribution": "categorical", "device": dev},
            "value": {"layer_config": [dense(16), relu], "output_config": out,
                      "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": dev},
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


def test_ppo_trainer_round_trip(tmp_path):
    run_dir = str(tmp_path / "ppo_run") + os.sep
    trainer = build_trainer_from_config(_ppo_config(run_dir))
    trainer.train()
    saved_step = trainer._step
    ref_config = json.dumps(_arch(trainer.get_config()), sort_keys=True)

    trainer.save(run_dir)
    trainer.env.close()

    # load_buffer=True rebuilds the (stateless) RolloutBuffer from its config.
    trainer2 = Trainer.load(run_dir, load_weights=True, load_buffer=True)
    assert json.dumps(_arch(trainer2.get_config()), sort_keys=True) == ref_config
    _params_match(trainer.agent.model, trainer2.agent.model, "model")
    _optimizer_states_match(trainer.agent.model, trainer2.agent.model, "model")

    # Resume training from the saved step counter.
    trainer2.schedule.stop_units = saved_step + 32
    trainer2.train()
    assert trainer2._step >= saved_step + 32
    trainer2.env.close()


def test_normalizer_stats_update_after_learn(tmp_path):
    """Running-stat normalizers must NOT fold in new statistics between a
    rollout and the learn() that consumes it: agent.learn re-normalizes the
    stored raw rollout, so the stats it uses must be the ones the policy
    acted under (collection-time normalization, SB3/RSL-RL semantics).
    Regression test for the IsaacSim camera-PPO NaN: stats updated right
    before learn whipsawed the inputs and collapsed the policy sigma."""
    cfg = _ppo_config(str(tmp_path / "ppo_norm_run") + os.sep)
    cfg["agent"]["config"]["state_normalizer"] = {
        "type": "RunningNorm",
        "config": {"name": "PPO.SN", "clip_value": 5.0,
                   "device": cfg["agent"]["config"]["device"]},
    }
    trainer = build_trainer_from_config(cfg)

    events = []
    orig_update = trainer.update_normalizers
    orig_learn = trainer.agent.learn

    def spy_update():
        events.append("update")
        return orig_update()

    def spy_learn(*a, **k):
        events.append("learn")
        return orig_learn(*a, **k)

    trainer.update_normalizers = spy_update
    trainer.agent.learn = spy_learn
    trainer.train()
    trainer.env.close()

    assert events.count("learn") >= 2, f"expected >=2 learns, got {events}"
    # Every stats update must directly FOLLOW a learn — never precede one
    # within the same learn gate (which would re-normalize the rollout with
    # statistics the policy never acted under).
    for i, ev in enumerate(events):
        if ev == "update":
            assert i > 0 and events[i - 1] == "learn", (
                f"normalizer stats updated before the learn that consumes the "
                f"rollout: {events}"
            )


# --------------------------------------------------------------------------- #
# 3. DDPG: agent-level save_state/load_state contract (incl. target model)
# --------------------------------------------------------------------------- #
def _ddpg_config(save_dir: str, env_id: str = "Pendulum-v1") -> dict:
    dev = "cuda" if T.cuda.is_available() else "cpu"
    dense = lambda u: {"type": "dense", "params": {"units": u, "kernel": "default", "kernel_params": {}}}
    relu = {"type": "relu"}
    out = [{"type": "dense", "params": {"kernel": "default", "kernel_params": {}}}]
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 60, "learn_every_unit": "timestep",
            "learn_every": 1, "updates_per_learn": 1, "batch_size": 32,
            "warmup_steps": 8, "seed": 7,
        },
        "agent": {"type": "DDPG", "config": {
            "name": "DDPG",
            "policy": {"layer_config": [dense(16), relu], "output_config": out,
                       "optimizer_params": {"type": "Adam", "params": {"lr": 1e-3}}, "device": dev},
            "critic": {"layer_config": [dense(16)], "merged_config": [dense(16)], "output_config": out,
                       "optimizer_params": {"type": "Adam", "params": {"lr": 1e-3}}, "device": dev},
            "discount": 0.99, "tau": 0.05, "action_epsilon": 0.0,
            "state_normalizer": {"type": "RunningNorm",
                                 "config": {"name": "DDPG.SN", "clip_value": 10.0, "device": dev}},
            "noise": {"type": "NormalNoise", "config": {"stddev": 0.2}},
            "noise_clip": 0.3, "policy_grad_clip": 40.0, "critic_grad_clip": 40.0, "N": 1,
            "device": dev, "log_level": "ERROR",
        }},
        "env": {"type": "gymnasium", "config": {
            "cfg": env_id, "num_envs": 1, "obs_key": None, "goal_key": None, "ach_goal_key": None,
            "wrappers": [{"type": "VectorNStepReward",
                          "params": {"n": 1, "obs_key": None, "goal_key": None, "ach_goal_key": None}}],
            "render_mode": None, "seed": 7}},
        "buffer": {"type": "ReplayBuffer", "config": {"buffer_size": 500, "N": 1}},
    }


def test_ddpg_agent_contract_round_trip(tmp_path):
    config = _ddpg_config(str(tmp_path / "ddpg_run") + os.sep)
    trainer = build_trainer_from_config(config)
    try:
        agent = trainer.agent
        # Nudge weights so we are not comparing identical fresh inits by luck.
        with T.no_grad():
            for p in agent.model.parameters():
                p.add_(T.randn_like(p) * 0.01)

        state_dir = tmp_path / "ddpg_state"
        agent.save_state(state_dir)

        rebuilt = build_agent(agent.get_config(), trainer.env)
        rebuilt.load_state(state_dir, load_weights=True)

        for key in agent.MODEL_ATTRS + agent.TARGET_ATTRS:
            m1, m2 = getattr(agent, key, None), getattr(rebuilt, key, None)
            if m1 is not None and m2 is not None:
                _params_match(m1, m2, f"DDPG:{key}")
    finally:
        try:
            trainer.env.close()
        except Exception:
            pass
