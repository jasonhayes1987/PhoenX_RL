from importlib import resources
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from phoenx.models import (
    StochasticContinuousHead, StochasticDiscreteHead, DeterministicActorHead,
    ValueHead, DiscreteQHead, ContinuousQHead, modular_parts_from_config,
)
from phoenx.buffer import Buffer
from phoenx.her import HindsightRelabeler
from phoenx.env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper, EnvPoolWrapper
from phoenx.rl_agents import Agent
from phoenx.renderer import Renderer
from phoenx.rl_callbacks import load as callback_load
from phoenx.trainer import TrainingSchedule, Trainer, SuccessCriterion
from phoenx.intrinsic_motivation import IntrinsicMotivation
from phoenx.normalizer import create_normalizer as normalizer_factory, RunningNorm, BatchNorm, RewardNorm
from phoenx.schedulers import ScheduleWrapper
from phoenx.torch_utils import set_seed


def available_example_configs() -> list[str]:
    """List bundled example config names under ``phoenx.examples``.

    Recursively walks ``phoenx/examples/configs/`` and returns every ``.yml``
    file as a sorted forward-slash path relative to that root. Returns an
    empty list when the packaged configs directory is absent.

    Returns:
        Sorted list of relative config paths, e.g.
        ``["LunarLander-v3/reinforce.yml", ...]``.

    Example:
        >>> from phoenx.builder import available_example_configs
        >>> available_example_configs()  # doctest: +SKIP
        ['LunarLander-v3/reinforce.yml', ...]
    """
    configs_root = resources.files("phoenx.examples").joinpath("configs")
    if not configs_root.is_dir():
        return []

    names: list[str] = []

    def _walk(node, prefix: str) -> None:
        for entry in node.iterdir():
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                _walk(entry, rel)
            elif entry.is_file() and entry.name.endswith(".yml"):
                names.append(rel)

    _walk(configs_root, "")
    return sorted(names)


def load_config(config_file: str | Path) -> dict:
    """Load a YAML training config from disk or from bundled examples.

    An existing on-disk path always wins. If the path does not exist and is
    not absolute, fall back to the packaged copy under
    ``phoenx/examples/configs/``.

    Args:
        config_file: Filesystem path or bundled example name
            (e.g. ``LunarLanderContinuous-v3/sac.yml``). Forward or
            backslash separators are accepted.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError: If neither an on-disk path nor a bundled example
            resolves. The message names the request and lists available
            bundled examples.

    Example:
        >>> from phoenx.builder import load_config
        >>> cfg = load_config("LunarLanderContinuous-v3/sac.yml")  # doctest: +SKIP
        >>> isinstance(cfg, dict)
        True
    """
    path = Path(config_file)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj)

    if not path.is_absolute():
        parts = path.as_posix().split("/")
        packaged = resources.files("phoenx.examples").joinpath("configs", *parts)
        if packaged.is_file():
            return yaml.safe_load(packaged.read_text(encoding="utf-8"))

    available = available_example_configs()
    available_msg = ", ".join(available) if available else "(none)"
    raise FileNotFoundError(
        f"Config not found: {config_file!s}. "
        f"Bundled examples: {available_msg}"
    )



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


def apply_model_config(agent_cfg: dict, env: EnvWrapper) -> bool:
    """Handle the canonical ``model:`` schema (roots/trunk/branches).

    If ``agent_cfg`` carries a ``model`` entry it is decomposed into live
    ``roots`` / ``trunk`` / per-role branch heads (plus model-wide
    ``optimizer_params`` / ``lr_scheduler`` / ``shared_update``) written back
    into ``agent_cfg``, and True is returned. Legacy per-model configs
    (``policy:`` / ``value:`` / ...) return False and are handled by the
    per-algorithm builders below.
    """
    model_cfg = agent_cfg.pop('model', None)
    if not model_cfg:
        return False
    inner = model_cfg.get('config', model_cfg)
    parts = modular_parts_from_config(inner, env)
    agent_cfg['roots'] = parts['roots']
    agent_cfg['trunk'] = parts['trunk']
    for role, head in parts['branches'].items():
        agent_cfg[role] = head
    if parts['optimizer_params'] is not None:
        agent_cfg.setdefault('optimizer_params', parts['optimizer_params'])
    if parts['lr_scheduler'] is not None:
        agent_cfg.setdefault('lr_scheduler', parts['lr_scheduler'])
    if parts['shared_update']:
        agent_cfg.setdefault('shared_update', parts['shared_update'])
    return True


def create_policy(config: dict, env: EnvWrapper) -> StochasticContinuousHead | StochasticDiscreteHead:
    """Create a stochastic policy head from a config.
    
    Args:
        config (dict): The config to create the policy from.
        env (EnvWrapper): The environment to create the policy for.
        
    Returns:
        StochasticContinuousHead | StochasticDiscreteHead: The created policy head.
    """
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    if config['distribution'] in ['categorical']:
        config['temperature_schedule'] = ScheduleWrapper(**config['temperature_schedule']) if config.get('temperature_schedule', None) else None
        return StochasticDiscreteHead(**config)
    elif config['distribution'] in ['beta', 'kumaraswamy', 'normal']:
        return StochasticContinuousHead(**config)
    else:
        raise ValueError(f"Invalid distribution: {config['distribution']}")


def create_actor(config: dict, env: EnvWrapper) -> DeterministicActorHead:
    """Create a deterministic actor head from a config.
    
    Args:
        config (dict): The config to create the actor from.
        env (EnvWrapper): The environment to create the actor for.
        
    Returns:
        DeterministicActorHead: The created actor head.
    """
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    return DeterministicActorHead(**config)


def create_value(config: dict, env: EnvWrapper) -> ValueHead:
    """Create a value head from a config."""
    config = dict(config)
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    return ValueHead(**config)


def create_critic(config: dict, env: EnvWrapper) -> DiscreteQHead | ContinuousQHead:
    """Create a Q-head from a config.
    
    Args:
        config (dict): The config to create the critic from.
        env (EnvWrapper): The environment to create the critic for.
        
    Returns:
        DiscreteQHead | ContinuousQHead: The created Q head.
    """
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        return DiscreteQHead(**config)
    elif isinstance(env.single_action_space, gym.spaces.Box):
        return ContinuousQHead(**config)
    else:
        raise ValueError(f"Invalid action space: {env.single_action_space}")


def create_normalizer(config: dict, env: EnvWrapper, key: str | None = None) -> RunningNorm | BatchNorm | RewardNorm:
    """Create a normalizer from a config.

    Args:
        config (dict): The config to create the normalizer from.
        env (EnvWrapper): The environment to create the normalizer for.
        key (str | None): If env observation space is a dict, the key to create the normalizer for.

    Returns:
        RunningNorm | BatchNorm | RewardNorm: The created normalizer.
    """
    # RewardNorm needs no feature dim; DictNormalizer/ImageScale configure
    # their (per-key) dims themselves.
    if config['type'] not in ('RewardNorm', 'DictNormalizer', 'ImageScale'):
        num_features = infer_dim(env, key)
        config['config'].update({'num_features': num_features})
    return normalizer_factory(config)
    

def create_intrinsic_motivation(config: dict, env: EnvWrapper, key: str | None = None) -> IntrinsicMotivation:
    """Create an intrinsic motivation from a config.

    Args:
        config (dict): The config to create the intrinsic motivation from.
        env (EnvWrapper): The environment to create the intrinsic motivation for.
        key (str | None): If using an observation normalizer and the env observation space is a dict, the key to create the normalizer for.

    Returns:
        IntrinsicMotivation: The created intrinsic motivation.
    """
    im_type = config['type']
    im_config = config['config']
    im_config.update({
        'env': env,
        'reward_scheduler': ScheduleWrapper(**im_config['reward_scheduler']) if im_config.get('reward_scheduler', None) else None,
        'reward_normalizer': create_normalizer(im_config['reward_normalizer'], env) if im_config.get('reward_normalizer', None) else None,
    })
    return IntrinsicMotivation.create_instance(im_type, **im_config)


def build_callbacks(config: dict) -> list | None:
    callbacks = config.get("callbacks")
    if not callbacks:
        return None
    return [callback_load(callback_config) for callback_config in callbacks]


def build_renderer(config: dict) -> Renderer | None:
    renderer_config = config.get("renderer", None)
    if renderer_config is None:
        return None

    renderer_kwargs = dict(renderer_config)
    return Renderer(**renderer_kwargs)


def build_buffer(config: dict, env: EnvWrapper) -> Buffer:
    buffer_spec = config.get("buffer")
    if not buffer_spec:
        raise ValueError("Config is missing the required 'buffer' section.")

    buffer_kwargs = dict(buffer_spec.get("config", {}))
    buffer_kwargs["env"] = env
    # Instantiate HindsightRelabeler if hindsight in config
    hindsight_spec = buffer_kwargs.get("hindsight", None)
    if hindsight_spec is not None:
        hindsight_spec["env"] = env
        buffer_kwargs["hindsight"] = HindsightRelabeler(**hindsight_spec)

    return Buffer.create_instance(buffer_spec["type"], **buffer_kwargs)


def build_schedule(config: dict) -> TrainingSchedule:
    schedule_spec = config.get("schedule")
    if not schedule_spec:
        raise ValueError("Config is missing the required 'schedule' section.")

    return TrainingSchedule(**schedule_spec)

def build_success_criterion(config:dict) -> SuccessCriterion | None:
    success_spec = config.get("success_criterion", None)
    if success_spec is None:
        return None
    return SuccessCriterion(**success_spec)


def build_agent(config: dict, env: EnvWrapper) -> Agent:
    agent_type = config["agent"]["type"]
    if agent_type == "ActorCritic":
        from phoenx.builders.actor_critic import build
    elif agent_type == "Reinforce":
        from phoenx.builders.reinforce import build
    elif agent_type == "PPO":
        from phoenx.builders.ppo import build
    elif agent_type == "DDPG":
        from phoenx.builders.ddpg import build
    elif agent_type == "TD3":
        from phoenx.builders.td3 import build
    elif agent_type == "SAC":
        from phoenx.builders.sac import build
    else:
        raise NotImplementedError(f"Agent builder for '{agent_type}' is not implemented yet.")
    return build(config, env)


def build_trainer_from_config(config: dict, log_level: str | None = None) -> Trainer:
    seed = config.get('schedule', {}).get('seed', None)
    if seed is not None:
        set_seed(seed)

    env = create_env(config)
    agent = build_agent(config, env)
    buffer = build_buffer(config, env)
    schedule = build_schedule(config)
    callbacks = build_callbacks(config)
    renderer = build_renderer(config)
    success_criterion = build_success_criterion(config)
    return Trainer(
            agent=agent,
            env=env,
            buffer=buffer,
            schedule=schedule,
            success_criterion=success_criterion,
            renderer=renderer,
            callbacks=callbacks,
            log_level=log_level if log_level is not None else config.get('log_level', 'INFO'),
            save_dir=config.get('save_dir', 'models/'),
        )


def build_trainer_from_config_path(config_path: str | Path, log_level: str | None = None) -> Trainer:
    config = load_config(config_path)
    return build_trainer_from_config(config, log_level)