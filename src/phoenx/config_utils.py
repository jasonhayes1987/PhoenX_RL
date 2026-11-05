"""Config helpers that run before (or while) objects are built.

Everything here works on plain parsed-YAML dicts and imports no torch, so the
config schema tooling and GUI can use it without loading the training stack.

* [resolve_n_step][phoenx.config_utils.resolve_n_step] makes
  ``agent.config.N`` the single place the n-step horizon is set and fans it
  out to the replay buffer, the ``VectorNStepReward`` wrapper and any HER
  relabeler.
* [reject_unknown_keys][phoenx.config_utils.reject_unknown_keys] turns a
  misspelled config key into an error instead of a silently ignored value.
"""

from __future__ import annotations

import difflib
import inspect
import logging
from typing import Any, Iterable

logger = logging.getLogger("phoenx.config")


def init_param_names(cls: type) -> set[str]:
    """Names accepted by ``cls.__init__`` and every ``__init__`` up its MRO.

    Constructors in PhoenX forward ``**kwargs`` to their base class, so a key
    is valid if any class in the chain names it.

    Args:
        cls: Class to inspect.

    Returns:
        Set of keyword parameter names (``self`` and ``*args`` /
        ``**kwargs`` excluded).
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters.values()
        except (TypeError, ValueError):
            continue
        for p in params:
            if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            names.add(p.name)
    return names


def reject_unknown_keys(
    owner: str,
    keys: Iterable[str],
    valid: Iterable[str],
    allow_private: bool = True,
) -> None:
    """Raise if any config key is not one the owner actually reads.

    Several PhoenX constructors accept ``**kwargs`` and ``setattr`` the
    leftovers, and several factories read their config with ``dict.get``.
    Without this check a typo such as ``polcy_clip: 0.1`` creates an unused
    attribute (or is dropped) and training silently runs on the default.

    Keys starting with ``_`` (diagnostic knobs such as ``_diag_freq``) are
    always allowed when ``allow_private`` is set.

    Args:
        owner: Name used in the error message, e.g. ``"PPO"`` or
            ``"layer 'dense'"``.
        keys: Keys supplied by the config.
        valid: Keys the owner understands.
        allow_private: Accept any key that starts with ``_``.

    Raises:
        TypeError: Listing each unknown key, with a "did you mean" suggestion
            when a close match exists.

    Example:
        >>> reject_unknown_keys("PPO", ["policy_clip", "_diag_freq"], ["policy_clip"])
        >>> reject_unknown_keys("PPO", ["polcy_clip"], ["policy_clip"])
        Traceback (most recent call last):
        ...
        TypeError: PPO got an unknown config key: 'polcy_clip' (did you mean 'policy_clip'?). ...
    """
    valid = sorted(set(valid))
    unknown = [
        k for k in keys
        if k not in valid and not (allow_private and isinstance(k, str) and k.startswith("_"))
    ]
    if not unknown:
        return
    parts = []
    for k in unknown:
        close = difflib.get_close_matches(str(k), valid, n=1, cutoff=0.6)
        parts.append(f"{k!r} (did you mean {close[0]!r}?)" if close else repr(k))
    noun = "key" if len(unknown) == 1 else "keys"
    raise TypeError(
        f"{owner} got an unknown config {noun}: {', '.join(parts)}. "
        f"Valid keys: {', '.join(valid)}."
        + (" Private diagnostic keys must start with '_'." if allow_private else "")
    )

#: Agents whose constructor takes the n-step horizon ``N``.
N_STEP_AGENTS = frozenset({"DDPG", "TD3", "SAC"})

#: Buffers that store ``(B, N, ...)`` windows and need ``VectorNStepReward``.
N_STEP_BUFFERS = frozenset({"ReplayBuffer", "PrioritizedReplayBuffer"})

NSTEP_WRAPPER = "VectorNStepReward"

_KEY_FIELDS = ("obs_key", "goal_key", "ach_goal_key")


def _section_config(config: dict, section: str) -> dict | None:
    """Return ``config[section]["config"]`` when that mapping exists."""
    block = config.get(section)
    if not isinstance(block, dict):
        return None
    inner = block.get("config")
    return inner if isinstance(inner, dict) else None


def resolve_n_step(config: dict) -> dict:
    """Make ``agent.config.N`` the single source of the n-step horizon.

    The horizon has to agree in up to four places: the agent's return
    computation (``agent.config.N``), the replay buffer's stored window
    (``buffer.config.N``), the ``VectorNStepReward`` env wrapper's ring length
    (``params.n``) and an n-step HER relabeler (``buffer.config.hindsight.N``).
    Users now set it once, under ``agent.config.N``, and this pass copies it
    to the others:

    * ``buffer.config.N`` is filled for ``ReplayBuffer`` /
      ``PrioritizedReplayBuffer``.
    * A ``VectorNStepReward`` wrapper is appended to ``env.config.wrappers``
      when a replay buffer is used and none is listed (appended last, so it is
      the outermost vector wrapper and sees fully processed observations). An
      existing one gets its ``n`` set. Missing ``obs_key`` / ``goal_key`` /
      ``ach_goal_key`` wrapper params are taken from ``env.config``.
    * An ``n_step`` HER relabeler gets ``N``.

    Older configs that still spell the value in several places keep working
    as long as every copy agrees. If ``agent.config.N`` is absent, an agreed
    value found elsewhere is adopted; otherwise the default is ``1``.

    Args:
        config: Full parsed training config. Modified in place.

    Returns:
        The same ``config`` object, for chaining.

    Raises:
        ValueError: If two places name different horizons.

    Example:
        >>> cfg = {
        ...     "agent": {"type": "SAC", "config": {"N": 3}},
        ...     "env": {"type": "gymnasium", "config": {"cfg": "Pendulum-v1"}},
        ...     "buffer": {"type": "ReplayBuffer", "config": {"buffer_size": 1000}},
        ... }
        >>> resolve_n_step(cfg)["buffer"]["config"]["N"]
        3
        >>> cfg["env"]["config"]["wrappers"][0]["params"]["n"]
        3
    """
    agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    agent_type = agent.get("type")
    agent_cfg = _section_config(config, "agent")

    buffer = config.get("buffer") if isinstance(config.get("buffer"), dict) else {}
    buffer_type = buffer.get("type")
    uses_replay = buffer_type in N_STEP_BUFFERS

    if agent_type not in N_STEP_AGENTS and not uses_replay:
        return config

    buffer_cfg = _section_config(config, "buffer")
    if uses_replay and buffer_cfg is None:
        buffer_cfg = {}
        buffer["config"] = buffer_cfg

    env_cfg = _section_config(config, "env")
    wrappers: list | None = None
    nstep_wrapper: dict | None = None
    if env_cfg is not None:
        wrappers = env_cfg.get("wrappers")
        if wrappers is None:
            wrappers = []
        for w in wrappers:
            if isinstance(w, dict) and w.get("type") == NSTEP_WRAPPER:
                nstep_wrapper = w
                break

    hindsight = buffer_cfg.get("hindsight") if buffer_cfg else None
    hindsight_n_step = (
        isinstance(hindsight, dict)
        and hindsight.get("output_format", "n_step") == "n_step"
    )

    # Every place the horizon may already be spelled out.
    declared: dict[str, Any] = {}
    if agent_cfg is not None and agent_cfg.get("N") is not None:
        declared["agent.config.N"] = agent_cfg["N"]
    if buffer_cfg is not None and buffer_cfg.get("N") is not None:
        declared["buffer.config.N"] = buffer_cfg["N"]
    if nstep_wrapper is not None and (nstep_wrapper.get("params") or {}).get("n") is not None:
        declared[f"env.config.wrappers[{NSTEP_WRAPPER}].params.n"] = nstep_wrapper["params"]["n"]
    if hindsight_n_step and hindsight.get("N") is not None:
        declared["buffer.config.hindsight.N"] = hindsight["N"]

    if len(set(declared.values())) > 1:
        listing = ", ".join(f"{k}={v}" for k, v in declared.items())
        raise ValueError(
            f"N-step horizon mismatch: {listing}. Set it once, as agent.config.N, "
            "and remove the other copies; PhoenX fills in the buffer, the "
            f"{NSTEP_WRAPPER} wrapper and HER from it."
        )

    n = next(iter(declared.values()), 1)
    redundant = [k for k in declared if k != "agent.config.N"]
    if redundant:
        logger.info(
            "N-step horizon is also set at %s; only agent.config.N is needed now.",
            ", ".join(redundant),
        )

    if agent_cfg is not None and agent_type in N_STEP_AGENTS:
        agent_cfg["N"] = n

    if uses_replay:
        buffer_cfg["N"] = n
        if env_cfg is not None:
            if nstep_wrapper is None:
                nstep_wrapper = {"type": NSTEP_WRAPPER, "params": {}}
                wrappers.append(nstep_wrapper)
            env_cfg["wrappers"] = wrappers

    if nstep_wrapper is not None:
        params = nstep_wrapper.get("params")
        if not isinstance(params, dict):
            params = {}
            nstep_wrapper["params"] = params
        params["n"] = n
        for key in _KEY_FIELDS:
            if key not in params and env_cfg is not None:
                params[key] = env_cfg.get(key)

    if hindsight_n_step:
        hindsight["N"] = n

    return config
