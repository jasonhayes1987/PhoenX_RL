import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.buffer import Buffer
from app.env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper, EnvPoolWrapper
from app.renderer import Renderer
from app.rl_callbacks import load as callback_load
from app.trainer import TrainingSchedule, OnPolicyTrainer, OffPolicyTrainer


def load_config(config_file: str | Path) -> dict:
    with open(config_file, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def infer_dim(env: EnvWrapper, key: str | None = None) -> int:
    space = env.single_observation_space
    if isinstance(space, gym.spaces.Dict):
        if key is None:
            raise ValueError(
                "Observation space is Dict, but no key was provided. "
                f"Available keys: {list(space.spaces.keys())}"
            )
        if key not in space.spaces:
            raise KeyError(
                f"Key '{key}' not in observation space. "
                f"Available keys: {list(space.spaces.keys())}"
            )
        return int(np.prod(space.spaces[key].shape))
    return int(np.prod(space.shape))


def create_env(config: dict) -> EnvWrapper:
    env_type = config["env"]["type"]
    env_config = dict(config["env"]["config"])
    if env_type == "isaacsim":
        return IsaacSimWrapper(**env_config)
    if env_type == "gymnasium":
        return GymnasiumWrapper(**env_config)
    if env_type == "envpool":
        return EnvPoolWrapper(**env_config)
    raise ValueError(f"Invalid environment type: {env_type}")


def build_callbacks(config: dict) -> list | None:
    callbacks = config.get("callbacks")
    if not callbacks:
        return None
    return [callback_load(callback_config) for callback_config in callbacks]


def build_renderer(config: dict) -> Renderer | None:
    renderer_config = config.get("renderer")
    if renderer_config is None:
        return None

    if "config" in renderer_config:
        renderer_kwargs = dict(renderer_config["config"])
    else:
        renderer_kwargs = dict(renderer_config)

    renderer_kwargs.setdefault("save_dir", config["save_dir"])
    return Renderer(**renderer_kwargs)


def build_buffer(config: dict, env: EnvWrapper) -> Buffer:
    buffer_spec = config.get("buffer")
    if not buffer_spec:
        raise ValueError("Config is missing the required 'buffer' section.")

    buffer_kwargs = dict(buffer_spec.get("config", {}))
    buffer_kwargs["env"] = env
    return Buffer.create_instance(buffer_spec["type"], **buffer_kwargs)


def build_schedule(config: dict):
    schedule_spec = config.get("schedule")
    if not schedule_spec:
        raise ValueError("Config is missing the required 'schedule' section.")

    return TrainingSchedule(**schedule_spec)
    
def build_agent(config: dict, env: EnvWrapper):
    agent_type = config["agent"]["type"]
    if agent_type == "ActorCritic":
        from scripts.actor_critic import build
    elif agent_type == "Reinforce":
        from scripts.reinforce import build
    elif agent_type == "PPO":
        from scripts.ppo import build
    elif agent_type == "DDPG":
        from scripts.ddpg import build
    elif agent_type == "TD3":
        from scripts.td3 import build
    else:
        raise NotImplementedError(f"Agent builder for '{agent_type}' is not implemented yet.")
    return build(config, env)


def build_trainer_from_config(config: dict):
    env = create_env(config)
    agent = build_agent(config, env)
    buffer = build_buffer(config, env)
    schedule = build_schedule(config)
    callbacks = build_callbacks(config)
    renderer = build_renderer(config)

    agent_type = config["agent"]["type"]

    if agent_type in {"ActorCritic", "Reinforce", "PPO"}:
        return OnPolicyTrainer(
            agent=agent,
            env=env,
            buffer=buffer,
            schedule=schedule,
            renderer=renderer,
            callbacks=callbacks,
            log_level=config.get('log_level', 'INFO'),
        )
    
    elif agent_type in {"DDPG", "TD3", "SAC"}:
        return OffPolicyTrainer(
            agent=agent,
            env=env,
            buffer=buffer,
            schedule=schedule,
            renderer=renderer,
            callbacks=callbacks,
            log_level=config.get('log_level', 'INFO'),
        )

    raise ValueError(f"Unsupported trainer type for agent '{agent_type}'.")


def build_trainer_from_config_path(config_path: str | Path):
    config = load_config(config_path)
    return build_trainer_from_config(config)