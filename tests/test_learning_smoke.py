"""Learning smoke tests (marked ``slow``) — the refactored stack must not just
run, it must LEARN.

    * feedforward PPO on CartPole reaches a solid average reward;
    * recurrent (LSTM) PPO solves the observe-then-recall memory task from
      conftest (``PhoenXMemory-v0``) where a memoryless policy provably cannot
      exceed ~0.5 average success;
    * multi-modal (dict-obs, multi-root) PPO on the synthetic multi-modal env
      shifts its policy toward the reward-maximizing actions.

Run explicitly with:  pytest tests/test_learning_smoke.py -m slow
"""

from __future__ import annotations

import pytest
import torch as T

from phoenx.builder import build_trainer_from_config

DEV = "cuda" if T.cuda.is_available() else "cpu"

DENSE = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                               "kernel_params": {"gain": 1.41421356}}}
RELU = {"type": "relu"}
OUT_POLICY = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OUT_VALUE = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 1.0}}}]


def _base_config(save_dir: str) -> dict:
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {
            "stop_unit": "timestep", "stop_units": 20_000,
            "learn_every_unit": "timestep", "learn_every": 1024,
            "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 256, "learning_epochs": 4,
            "warmup_steps": 0, "seed": 3,
        },
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 512}},
    }


def _avg_reward(trainer) -> float:
    history = trainer._score_history
    return sum(history) / len(history) if history else float("nan")


@pytest.mark.slow
def test_ppo_cartpole_learns(tmp_path):
    config = _base_config(str(tmp_path) + "/")
    config["agent"] = {"type": "PPO", "config": {
        "name": "PPO",
        "policy": {"layer_config": [DENSE(64), RELU, DENSE(64), RELU],
                   "output_config": OUT_POLICY, "distribution": "categorical",
                   "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": DEV},
        "value": {"layer_config": [DENSE(64), RELU, DENSE(64), RELU],
                  "output_config": OUT_VALUE,
                  "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}}, "device": DEV},
        "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
        "entropy_coefficient": 0.01, "policy_clip": 0.2, "value_clip": 0.2,
        "policy_grad_clip": 0.5, "value_grad_clip": 0.5, "value_coef": 0.5,
        "device": DEV, "log_level": "ERROR",
    }}
    config["env"] = {"type": "gymnasium", "config": {
        "cfg": "CartPole-v1", "num_envs": 8, "obs_key": None, "goal_key": None,
        "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 3}}
    config["schedule"]["stop_units"] = 40_000

    trainer = build_trainer_from_config(config)
    trainer.train()
    avg = _avg_reward(trainer)
    trainer.env.close()
    # random CartPole ≈ 20; a learning agent comfortably exceeds 100
    assert avg > 100.0, f"PPO failed to learn CartPole (avg reward {avg:.1f})"


@pytest.mark.slow
def test_recurrent_ppo_solves_memory_task(tmp_path):
    """PhoenXMemory-v0: the cue is only visible on step 0 and the reward comes
    at the final step — a memoryless policy cannot exceed ~0.5 success. The
    LSTM-trunk PPO must clearly beat that bound."""
    config = _base_config(str(tmp_path) + "/")
    config["schedule"].update({
        "stop_units": 60_000, "learn_every": 512,
        # recurrent PPO: mini_batch_size is in env units
        "mini_batch_size": 8, "learning_epochs": 4, "seed": 5,
    })
    config["buffer"] = {"type": "RolloutBuffer", "config": {"buffer_size": 64}}
    config["agent"] = {"type": "PPO", "config": {
        "name": "PPO",
        "model": {"type": "ModularModel", "config": {
            "roots": {"state": {"layer_config": [DENSE(32), RELU]}},
            "trunk": {"layer_config": [{"type": "lstm", "params": {"hidden_size": 32}}]},
            "branches": {
                "policy": {"type": "StochasticDiscreteHead",
                           "config": {"layer_config": [DENSE(32), RELU],
                                      "output_config": OUT_POLICY, "device": DEV}},
                "value": {"type": "ValueHead",
                          "config": {"layer_config": [DENSE(32), RELU],
                                     "output_config": OUT_VALUE, "device": DEV}},
            },
            "optimizer_params": {"type": "Adam", "params": {"lr": 5e-4}},
            "shared_update": "combined",
            "device": DEV,
        }},
        "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
        "entropy_coefficient": 0.01, "policy_clip": 0.2, "value_clip": 0.2,
        "policy_grad_clip": 0.5, "value_grad_clip": 0.5, "value_coef": 0.5,
        "device": DEV, "log_level": "ERROR",
    }}
    config["env"] = {"type": "gymnasium", "config": {
        "cfg": "PhoenXMemory-v0", "num_envs": 8, "obs_key": None, "goal_key": None,
        "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 5}}

    trainer = build_trainer_from_config(config)
    assert trainer.agent.model.is_recurrent
    trainer.train()
    avg = _avg_reward(trainer)  # per-episode reward == success (0 or 1)
    trainer.env.close()
    assert avg > 0.75, (
        f"recurrent PPO failed the memory task (avg success {avg:.2f}; "
        f"memoryless bound ≈ 0.5)"
    )


@pytest.mark.slow
def test_multimodal_ppo_learns_on_dict_obs(tmp_path):
    """Multi-root PPO on the synthetic dict-obs env: the reward increases with
    the (continuous) action sum, so a learning policy pushes its mean action
    up from ~0 toward +1 on both dims."""
    config = _base_config(str(tmp_path) + "/")
    config["schedule"].update({"stop_units": 40_000, "learn_every": 512,
                               "mini_batch_size": 256, "learning_epochs": 4, "seed": 7})
    config["buffer"] = {"type": "RolloutBuffer", "config": {"buffer_size": 64}}
    config["agent"] = {"type": "PPO", "config": {
        "name": "PPO",
        "model": {"type": "ModularModel", "config": {
            "roots": {
                "cnn": {"input_keys": ["rgb"], "layer_config": [
                    {"type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "stride": 2}},
                    RELU, {"type": "flatten"},
                    DENSE(32), RELU,
                ]},
                "vec": {"input_keys": ["vec"], "layer_config": [DENSE(32), RELU]},
            },
            "trunk": {"layer_config": [DENSE(64), RELU]},
            "branches": {
                "policy": {"type": "StochasticContinuousHead",
                           "config": {"layer_config": [DENSE(32), RELU],
                                      "output_config": OUT_POLICY,
                                      "distribution": "normal", "device": DEV}},
                "value": {"type": "ValueHead",
                          "config": {"layer_config": [DENSE(32), RELU],
                                     "output_config": OUT_VALUE, "device": DEV}},
            },
            "optimizer_params": {"type": "Adam", "params": {"lr": 1e-3}},
            "shared_update": "combined",
            "device": DEV,
        }},
        "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
        "entropy_coefficient": 0.005, "policy_clip": 0.2, "value_clip": 0.2,
        "policy_grad_clip": 0.5, "value_grad_clip": 0.5, "value_coef": 0.5,
        "advantage_normalizer": {"type": "BatchNorm",
                                 "config": {"name": "PPO.Adv", "clip_value": 10.0,
                                            "device": DEV}},
        "device": DEV, "log_level": "ERROR",
    }}
    config["env"] = {"type": "gymnasium", "config": {
        "cfg": "PhoenXMultiModal-v0", "num_envs": 8, "obs_key": None, "goal_key": None,
        "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 7}}

    trainer = build_trainer_from_config(config)
    trainer.train()

    # After training the test-time (mean) action should be clearly positive.
    obs = trainer.env.reset(seed=99)
    with T.no_grad():
        action = trainer.agent.act(obs.states, context="test")
    mean_action = float(action.actions.float().mean())
    trainer.env.close()
    assert mean_action > 0.3, (
        f"multi-modal PPO did not shift toward reward-maximizing actions "
        f"(mean action {mean_action:.3f})"
    )
