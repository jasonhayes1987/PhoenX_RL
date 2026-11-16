"""The env wrapper puts its tensors on the configured device.

A ``device: cpu`` run on a machine with a GPU used to crash: the agent was on
the CPU but the env wrapper always picked CUDA, so their tensors met on two
devices in ``Trainer.step``.
"""

from __future__ import annotations

import torch as T

from phoenx.builder import create_env
from phoenx.env_wrapper import GymnasiumWrapper


def _config(agent_device=None, env_device=None):
    env_cfg = {
        "cfg": "CartPole-v1",
        "num_envs": 2,
        "seed": 0,
        "wrappers": [{"type": "VectorNStepReward", "params": {"n": 3}}],
    }
    if env_device is not None:
        env_cfg["device"] = env_device
    agent_cfg = {} if agent_device is None else {"device": agent_device}
    return {"env": {"type": "gymnasium", "config": env_cfg}, "agent": {"config": agent_cfg}}


def test_env_follows_the_agent_device():
    env = create_env(_config(agent_device="cpu"))
    try:
        assert env.device == T.device("cpu")
        assert env.config["config"]["device"] == "cpu"
        assert env._find_nstep_wrapper().device == T.device("cpu")
        obs = env.reset()
        assert obs.states.device.type == "cpu"
    finally:
        env.close()


def test_explicit_env_device_wins():
    env = create_env(_config(agent_device="cuda", env_device="cpu"))
    try:
        assert env.device == T.device("cpu")
    finally:
        env.close()


def test_device_survives_json_roundtrip():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=1, seed=0, device="cpu")
    try:
        clone = GymnasiumWrapper.from_json(env.to_json())
        try:
            assert clone.device == T.device("cpu")
        finally:
            clone.close()
    finally:
        env.close()
