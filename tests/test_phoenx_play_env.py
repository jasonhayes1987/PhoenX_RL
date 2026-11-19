"""The ``phoenx_play`` env type: retro games exported from PhoenX Play.

PhoenX Play itself (libretro cores, ROMs) is not needed here: a stand-in
``phoenx_play.env.make_vec_env`` returns a CartPole vector env, so these tests
check PhoenX RL's side of the contract only. PhoenX Play's own tests run the
real emulator through this wrapper.
"""

from __future__ import annotations

import json
import sys
import types

import gymnasium as gym
import pytest

from phoenx.builder import create_env
from phoenx.config_schema import build_schema
from phoenx.env_wrapper import EnvWrapper, PhoenXPlayWrapper


@pytest.fixture
def fake_play(monkeypatch):
    """Install a stand-in ``phoenx_play.env``; returns the calls it received."""
    calls: list[tuple] = []

    def make_vec_env(integration, num_envs, *, wrappers=None, **kwargs):
        calls.append((integration, num_envs, len(wrappers or []), kwargs))
        return gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync",
                            wrappers=wrappers or [])

    pkg = types.ModuleType("phoenx_play")
    mod = types.ModuleType("phoenx_play.env")
    mod.make_vec_env = make_vec_env
    monkeypatch.setitem(sys.modules, "phoenx_play", pkg)
    monkeypatch.setitem(sys.modules, "phoenx_play.env", mod)
    return calls


def test_create_env_builds_the_phoenx_play_wrapper(fake_play):
    config = {"agent": {"config": {"device": "cpu"}},
              "env": {"type": "phoenx_play",
                      "config": {"cfg": "exports/ff3_battle", "num_envs": 2, "rom": "roms",
                                 "wrappers": [{"type": "TimeLimit",
                                               "params": {"max_episode_steps": 5}}]}}}
    env = create_env(config)
    try:
        assert isinstance(env, PhoenXPlayWrapper)
        assert fake_play == [("exports/ff3_battle", 2, 1, {"rom": "roms"})]
        assert env.device.type == "cpu"  # filled from agent.config.device
        assert env.finite_horizon
        obs = env.reset()
        assert tuple(obs.states.shape) == (2, 4)
        step = env.step(env.format_actions(env.action_space.sample()))
        assert tuple(step.rewards.shape) == (2,)
    finally:
        env.close()


def test_config_round_trips_through_json(fake_play):
    env = PhoenXPlayWrapper("exports/x", num_envs=1, seed=7, device="cpu")
    try:
        doc = json.loads(env.to_json())
        assert doc == {"type": "phoenx_play",
                       "config": {"cfg": "exports/x", "num_envs": 1, "wrappers": None,
                                  "render_mode": None, "seed": 7, "device": "cpu", "rom": None}}
        again = EnvWrapper.from_json(env.to_json())
        assert isinstance(again, PhoenXPlayWrapper) and again.config == env.config
        again.close()
    finally:
        env.close()


def test_without_phoenx_play_installed_it_says_how(monkeypatch):
    monkeypatch.setitem(sys.modules, "phoenx_play.env", None)  # import fails
    with pytest.raises(ImportError, match="pip install phoenx-play"):
        PhoenXPlayWrapper("exports/x", device="cpu")


def test_schema_offers_phoenx_play_with_its_settings():
    env = build_schema()["properties"]["env"]
    assert "phoenx_play" in env["properties"]["type"]["enum"]
    variant = next(b["then"] for b in env["allOf"]
                   if b["if"]["properties"]["type"].get("const") == "phoenx_play")
    props = variant["properties"]["config"]["properties"]
    assert {"cfg", "num_envs", "rom", "device", "seed"} <= set(props)
    assert "integration folder" in props["cfg"]["description"]
