"""YAML-to-object factories for envs, agents, buffers, and trainers.

Training configs are plain YAML mappings. These helpers turn sections of that
mapping into live objects: ``create_env`` / ``build_agent`` / ``build_buffer``
and friends construct the pieces, and ``build_trainer_from_config`` (or
``build_trainer_from_config_path``) wires them into a ``Trainer``. Head and
normalizer helpers (``create_policy``, ``create_normalizer``, …) are shared by
the per-algorithm builders under ``phoenx.builders``.

Most factories take either the full config or a sub-dict already extracted by
the caller. When an optional section is absent they typically return ``None``
rather than raising, so callers can pass the result straight through.
"""

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
    """Infer a flat feature count from the env's single-observation space.

    For a ``Box`` (or other non-Dict) space, returns the product of
    ``space.shape``. For a ``Dict`` space, returns the product of the named
    subspace's shape.

    Args:
        env: Environment whose ``single_observation_space`` is inspected.
        key: Observation-dict key to measure. Required when the space is a
            ``Dict``; ignored otherwise.

    Returns:
        Flattened feature dimension as an ``int``.

    Raises:
        ValueError: If the space is a ``Dict`` and ``key`` is ``None``.
        KeyError: If ``key`` is not among the Dict space's keys.
    """
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
    """Build an env wrapper from the top-level ``env`` config section.

    Args:
        config: Full training config. Reads ``config["env"]["type"]`` and
            ``config["env"]["config"]``. Accepted ``type`` values are
            ``"isaacsim"``, ``"gymnasium"``, and ``"envpool"``; the ``config``
            sub-dict is forwarded as keyword arguments to the matching wrapper
            constructor (``IsaacSimWrapper``, ``GymnasiumWrapper``, or
            ``EnvPoolWrapper``).

    Returns:
        Constructed environment wrapper.

    Raises:
        ValueError: If ``env.type`` is not one of the three accepted values.
        KeyError: If the ``env`` section or its ``type`` / ``config`` keys are
            missing.
    """
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

    Args:
        agent_cfg: Agent sub-dict (typically ``config["agent"]["config"]``).
            On success this function pops ``model`` and writes ``roots``,
            ``trunk``, per-role head objects, and optionally
            ``optimizer_params``, ``lr_scheduler``, and ``shared_update``.
        env: Environment passed to ``modular_parts_from_config``.

    Returns:
        ``True`` if a ``model`` entry was applied; ``False`` if absent or empty
        (legacy schema left untouched).
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
    """Create a stochastic policy head from a head config sub-dict.

    Args:
        config: Policy head mapping. Must include ``distribution``. Accepted
            values: ``"categorical"`` (builds ``StochasticDiscreteHead``, and
            wraps optional ``temperature_schedule``), or ``"beta"``,
            ``"kumaraswamy"``, ``"normal"`` (builds ``StochasticContinuousHead``).
            Optional ``lr_scheduler`` is wrapped in ``ScheduleWrapper`` when
            present. Mutated in place: injects ``env`` and the wrapped
            schedules before unpacking into the head constructor.
        env: Environment bound onto the head as ``config["env"]``.

    Returns:
        Stochastic discrete or continuous policy head.

    Raises:
        ValueError: If ``distribution`` is not one of the accepted values.
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
    """Create a deterministic actor head from a head config sub-dict.

    Args:
        config: Actor head mapping unpacked into ``DeterministicActorHead``.
            Optional ``lr_scheduler`` is wrapped in ``ScheduleWrapper`` when
            present. Mutated in place: injects ``env`` and the wrapped
            schedule.
        env: Environment bound onto the head as ``config["env"]``.

    Returns:
        Deterministic actor head.
    """
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    return DeterministicActorHead(**config)


def create_value(config: dict, env: EnvWrapper) -> ValueHead:
    """Create a value head from a head config sub-dict.

    Args:
        config: Value head mapping. Copied before mutation, then unpacked into
            ``ValueHead``. Optional ``lr_scheduler`` is wrapped in
            ``ScheduleWrapper`` when present.
        env: Environment bound onto the head as ``config["env"]``.

    Returns:
        Value head.
    """
    config = dict(config)
    config['env'] = env
    config['lr_scheduler'] = ScheduleWrapper(**config['lr_scheduler']) if config.get('lr_scheduler', None) else None
    return ValueHead(**config)


def create_critic(config: dict, env: EnvWrapper) -> DiscreteQHead | ContinuousQHead:
    """Create a Q-head from a head config sub-dict.

    Dispatches on ``env.single_action_space``: ``Discrete`` → ``DiscreteQHead``,
    ``Box`` → ``ContinuousQHead``.

    Args:
        config: Critic head mapping unpacked into the chosen Q-head class.
            Optional ``lr_scheduler`` is wrapped in ``ScheduleWrapper`` when
            present. Mutated in place: injects ``env`` and the wrapped
            schedule.
        env: Environment used for action-space dispatch and bound as
            ``config["env"]``.

    Returns:
        Discrete or continuous Q-head.

    Raises:
        ValueError: If the action space is neither ``Discrete`` nor ``Box``.
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
    """Create a normalizer from a ``{type, config}`` mapping.

    Args:
        config: Normalizer spec with ``type`` and ``config`` keys. ``type`` is
            dispatched by ``phoenx.normalizer.create_normalizer``; registered
            names include ``RunningNorm``, ``BatchNorm``, ``RewardNorm``,
            ``ImageScale``, and ``DictNormalizer``. For types other than
            ``RewardNorm``, ``DictNormalizer``, and ``ImageScale``, this
            factory writes ``num_features`` (from [infer_dim][phoenx.builder.infer_dim]) into
            ``config["config"]`` before dispatch.
        env: Environment used to infer ``num_features`` when needed.
        key: Observation-dict key passed to [infer_dim][phoenx.builder.infer_dim] when the
            observation space is a ``Dict``.

    Returns:
        Constructed normalizer instance.
    """
    # RewardNorm needs no feature dim; DictNormalizer/ImageScale configure
    # their (per-key) dims themselves.
    if config['type'] not in ('RewardNorm', 'DictNormalizer', 'ImageScale'):
        num_features = infer_dim(env, key)
        config['config'].update({'num_features': num_features})
    return normalizer_factory(config)
    

def create_intrinsic_motivation(config: dict, env: EnvWrapper, key: str | None = None) -> IntrinsicMotivation:
    """Create an intrinsic-motivation module from a ``{type, config}`` mapping.

    Args:
        config: Spec with ``type`` and ``config`` keys. ``type`` is dispatched
            by ``IntrinsicMotivation.create_instance``; accepted values are
            ``"ICM"``, ``"RND"``, and ``"EpisodicNovelty"``. Inside
            ``config["config"]``, optional ``reward_scheduler`` is wrapped in
            ``ScheduleWrapper`` and optional ``reward_normalizer`` is built via
            [create_normalizer][phoenx.builder.create_normalizer]. ``env`` is injected before dispatch.
        env: Environment injected into the IM config and used when building an
            optional reward normalizer.
        key: Present for call-site symmetry with [create_normalizer][phoenx.builder.create_normalizer];
            not read by this factory (reward normalizers are built without a
            key).

    Returns:
        Constructed intrinsic-motivation module.
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
    """Build the optional callback list from the top-level config.

    Args:
        config: Full training config. Reads ``config["callbacks"]``, a list of
            per-callback mappings consumed by ``phoenx.rl_callbacks.load``
            (each needs a registered ``type`` plus a ``config`` sub-dict).

    Returns:
        List of callback instances, or ``None`` when ``callbacks`` is absent
        or empty.
    """
    callbacks = config.get("callbacks")
    if not callbacks:
        return None
    return [callback_load(callback_config) for callback_config in callbacks]


def build_renderer(config: dict) -> Renderer | None:
    """Build the optional renderer from the top-level config.

    Args:
        config: Full training config. Reads ``config["renderer"]`` and
            forwards that mapping as keyword arguments to ``Renderer``.

    Returns:
        ``Renderer`` instance, or ``None`` when ``renderer`` is absent.
    """
    renderer_config = config.get("renderer", None)
    if renderer_config is None:
        return None

    renderer_kwargs = dict(renderer_config)
    return Renderer(**renderer_kwargs)


def build_buffer(config: dict, env: EnvWrapper) -> Buffer:
    """Build the replay / rollout buffer from the top-level config.

    Args:
        config: Full training config. Requires ``config["buffer"]`` with
            ``type`` and optional ``config`` keys. Accepted ``type`` values
            are ``"ReplayBuffer"``, ``"PrioritizedReplayBuffer"``,
            ``"RolloutBuffer"``, and ``"TrajectoryBuffer"``. The inner
            ``config`` mapping is forwarded to ``Buffer.create_instance`` with
            ``env`` injected. If that mapping contains ``hindsight``, it is
            replaced by a ``HindsightRelabeler(**hindsight_spec)`` (after
            injecting ``env`` into the hindsight spec).
        env: Environment injected into the buffer kwargs (and into any
            ``hindsight`` spec before constructing ``HindsightRelabeler``).

    Returns:
        Constructed buffer instance.

    Raises:
        ValueError: If the ``buffer`` section is missing or empty.
    """
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
    """Build the training schedule from the top-level config.

    Args:
        config: Full training config. Requires ``config["schedule"]``, which is
            unpacked as keyword arguments to ``TrainingSchedule``.

    Returns:
        Constructed ``TrainingSchedule``.

    Raises:
        ValueError: If the ``schedule`` section is missing or empty.
    """
    schedule_spec = config.get("schedule")
    if not schedule_spec:
        raise ValueError("Config is missing the required 'schedule' section.")

    return TrainingSchedule(**schedule_spec)

def build_success_criterion(config:dict) -> SuccessCriterion | None:
    """Build the optional success criterion from the top-level config.

    Args:
        config: Full training config. Reads ``config["success_criterion"]`` and
            unpacks it as keyword arguments to ``SuccessCriterion``.

    Returns:
        ``SuccessCriterion`` instance, or ``None`` when
        ``success_criterion`` is absent.
    """
    success_spec = config.get("success_criterion", None)
    if success_spec is None:
        return None
    return SuccessCriterion(**success_spec)


def build_agent(config: dict, env: EnvWrapper) -> Agent:
    """Build an agent by dispatching to the per-algorithm builder.

    Args:
        config: Full training config. Reads ``config["agent"]["type"]`` and
            passes the whole ``config`` plus ``env`` to the matching builder
            under ``phoenx.builders``. Accepted ``type`` values are
            ``"ActorCritic"``, ``"Reinforce"``, ``"PPO"``, ``"DDPG"``,
            ``"TD3"``, and ``"SAC"``.
        env: Environment passed through to the per-algorithm builder.

    Returns:
        Constructed agent.

    Raises:
        NotImplementedError: If ``agent.type`` has no registered builder.
        KeyError: If the ``agent`` section or its ``type`` key is missing.
    """
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
    """Assemble a ``Trainer`` from a full parsed training config.

    Seeds from ``config["schedule"]["seed"]`` when present, then builds env,
    agent, buffer, schedule, callbacks, renderer, and success criterion via
    the factories in this module.

    Args:
        config: Full training config. Uses top-level keys ``env``, ``agent``,
            ``buffer``, and ``schedule`` (required by their factories);
            optional ``callbacks``, ``renderer``, ``success_criterion``,
            ``log_level``, and ``save_dir`` (default ``"models/"``).
        log_level: Override for the trainer's log level. When ``None``, falls
            back to ``config.get("log_level", "INFO")``.

    Returns:
        Fully wired ``Trainer``.
    """
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
    """Load a YAML config path (or bundled example name) and build a trainer.

    Args:
        config_path: Filesystem path or bundled example name, as accepted by
            [load_config][phoenx.builder.load_config].
        log_level: Optional log-level override forwarded to
            [build_trainer_from_config][phoenx.builder.build_trainer_from_config].

    Returns:
        Fully wired ``Trainer``.
    """
    config = load_config(config_path)
    return build_trainer_from_config(config, log_level)
