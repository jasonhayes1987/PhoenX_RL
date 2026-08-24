"""Centralized, logger-agnostic metrics collection for a training run.

``MetricsManager`` is the single place that owns metric namespacing, windowed
reduction, cadence, and capability gating for a ``Trainer`` run. Producers
(the trainer's own rollout bookkeeping, and each agent's ``learn()`` return
dict) *push* raw values into the manager via ``record`` / ``record_many``;
``collect()`` reduces the current window into a flat ``dict[str, float]`` that
the trainer merges into ``step_log`` before handing it to callbacks.

This module has no import path to ``phoenx.rl_agents``, ``phoenx.trainer``,
``phoenx.rl_callbacks``, ``phoenx.buffer``, or ``phoenx.models`` — it may
import only the standard library, ``torch``, and ``phoenx.logging_config``.
Every piece of agent/env/buffer capability detection is duck-typed (see
``MetricsManager.bind``) so that ``Agent.metrics = NULL_RECORDER`` (the
default before a ``Trainer`` attaches a real manager) never creates an import
cycle.

Current scope: rollout, buffer occupancy/readiness, prioritized-replay, and
agent ``learn()``-return metrics, emitted every collected window
(``emit_every`` defaults to ``1``), plus one optimizer-activity counter
(``optim/updates_per_env_step``, a ratio of ``ModularModel.step()`` calls to
environment steps -- not a gradient norm and not per-module telemetry). There
is as yet no per-module gradient-norm telemetry, no recurrent-specific
metric, and no persistence of ROLLING / ROLLING_STD deques or RATE anchors
across a resume — after ``Trainer.load``, ``rollout/return_rolling_100``
starts from an empty window even though ``Trainer._score_history`` is
restored from ``trainer_state.pt``, so the two will disagree for the first
window after a resume.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, Sequence

import torch as T

from .logging_config import get_logger

logger = get_logger("metrics")


class Reduction(str, Enum):
    """How a metric's recorded values this window collapse into one float."""

    MEAN = "mean"
    LAST = "last"
    FIRST = "first"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    RATE = "rate"
    ROLLING = "rolling"
    WINDOW_STD = "window_std"
    ROLLING_STD = "rolling_std"


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one metric in ``METRIC_REGISTRY``.

    Attributes:
        key: Canonical (new-namespace) metric name, e.g. ``"loss/policy"``.
        reduction: How recorded values collapse into one float at ``collect()``.
        requires: Capability tags that must ALL be present (AND) for this spec
            to be enabled. See ``MetricsManager.bind`` for the predicate table.
        agents: Agent-class-name tags (lowercase, no ``"agent:"`` prefix) of
            which the bound agent's tag must be ONE (OR); empty means any
            agent.
        tier: Coarse cost tier; enabled when in ``MetricsManager``'s ``tiers``
            (or explicitly named in ``include``). Every registered spec is
            currently tier 0.
        window: Deque length for ``Reduction.ROLLING`` / ``Reduction.ROLLING_STD``
            specs; unused otherwise.
        aliases: Legacy flat key names dual-emitted alongside ``key`` at
            ``collect()`` (when ``dual_emit`` is ``True``) and consulted by
            ``MetricsManager.record_many`` for alias resolution.
    """

    key: str
    reduction: Reduction
    requires: frozenset[str] = frozenset()
    agents: frozenset[str] = frozenset()
    tier: int = 0
    window: int = 100
    aliases: tuple[str, ...] = ()


def _spec(
    key: str,
    reduction: Reduction,
    *,
    requires: Sequence[str] = (),
    agents: Sequence[str] = (),
    tier: int = 0,
    window: int = 100,
    aliases: Sequence[str] = (),
) -> MetricSpec:
    """Build one ``MetricSpec`` with sequence args coerced to frozenset/tuple."""
    return MetricSpec(
        key=key,
        reduction=reduction,
        requires=frozenset(requires),
        agents=frozenset(agents),
        tier=tier,
        window=window,
        aliases=tuple(aliases),
    )


# --------------------------------------------------------------------------- #
# Metric registry, in three groups: metrics computed in trainer.py
# (rollout / perf), prioritized-replay diagnostics, and metrics alias-ingested
# from agent.learn() return dicts.
# --------------------------------------------------------------------------- #
_SPECS: tuple[MetricSpec, ...] = (
    # Table A -- computed in trainer.py
    _spec("rollout/return_mean", Reduction.MEAN),
    _spec("rollout/return_rolling_100", Reduction.ROLLING, window=100),
    _spec("rollout/return_std", Reduction.ROLLING_STD, window=100),
    _spec("rollout/ep_length_mean", Reduction.MEAN),
    _spec("rollout/ep_length_std", Reduction.ROLLING_STD, window=100),
    _spec("rollout/termination_rate", Reduction.MEAN),
    _spec("rollout/truncation_rate", Reduction.MEAN),
    _spec("perf/sps", Reduction.RATE),
    _spec("buffer/fill", Reduction.LAST, requires=("buffer_capacity",)),
    _spec("buffer/env_steps", Reduction.LAST, requires=("buffer_capacity",)),
    _spec("buffer/is_ready", Reduction.LAST, requires=("buffer_capacity",)),
    _spec("buffer/pending_trajectories", Reduction.LAST, requires=("buffer_trajectories",)),
    _spec("optim/updates_per_env_step", Reduction.LAST, requires=("optimizer_steps",)),
    _spec("health/non_finite", Reduction.MAX),
    # Table A2 -- PER, exact old -> new map
    _spec("per/beta", Reduction.LAST, requires=("per",), aliases=("PER/beta",)),
    _spec("per/sampled_priority_mean", Reduction.MEAN, requires=("per",), aliases=("PER/sampled_priority_mean",)),
    _spec("per/sampled_priority_max", Reduction.MAX, requires=("per",), aliases=("PER/sampled_priority_max",)),
    _spec("per/buffer_priority_mean", Reduction.MEAN, requires=("per",), aliases=("PER/buffer_priority_mean",)),
    _spec("per/buffer_priority_max", Reduction.MAX, requires=("per",), aliases=("PER/buffer_priority_max",)),
    _spec("per/is_weight_mean", Reduction.MEAN, requires=("per",), aliases=("PER/weight_mean",)),
    _spec("per/is_weight_std", Reduction.MEAN, requires=("per",), aliases=("PER/weight_std",)),
    _spec("per/prob_mean", Reduction.MEAN, requires=("per",), aliases=("PER/prob_mean",)),
    _spec("per/prob_max", Reduction.MAX, requires=("per",), aliases=("PER/prob_max",)),
    _spec("per/is_weight_max", Reduction.MAX, requires=("per",)),
    _spec("per/priority_entropy", Reduction.MEAN, requires=("per",)),
    # Table B -- alias-ingested from agent.learn() returns
    _spec("loss/policy", Reduction.MEAN, aliases=("policy_loss",)),
    _spec("loss/value", Reduction.MEAN, requires=("has_value",), aliases=("value_loss",)),
    _spec("loss/critic", Reduction.MEAN, requires=("has_critic",), aliases=("critic_loss",)),
    _spec("critic/td_error_mean", Reduction.MEAN, aliases=("temporal_difference", "td_error")),
    _spec("critic/value_pred_mean", Reduction.MEAN, requires=("has_critic",), aliases=("critic_predictions",)),
    _spec("critic/return_mean_batch", Reduction.MEAN, aliases=("returns",)),
    _spec("policy/advantage_mean", Reduction.MEAN, requires=("on_policy",), aliases=("advantages",)),
    _spec("policy/entropy", Reduction.MEAN, aliases=("entropy",)),
    _spec("policy/kl", Reduction.MEAN, requires=("on_policy",), aliases=("kl",)),
    _spec("ppo/ratio_mean", Reduction.MEAN, agents=("ppo",), aliases=("prob_ratio",)),
    _spec("ppo/policy_clip", Reduction.LAST, agents=("ppo",), aliases=("policy_clip",)),
    _spec("ppo/value_clip", Reduction.LAST, agents=("ppo",), aliases=("value_clip",)),
    _spec("onpolicy/entropy_coef", Reduction.LAST, requires=("on_policy",), aliases=("entropy_coefficient",)),
    _spec("onpolicy/kl_coef", Reduction.LAST, agents=("ppo",), aliases=("kl_coefficient",)),
    _spec("optim/lr/policy", Reduction.LAST, aliases=("policy_learning_rate",)),
    _spec("optim/lr/value", Reduction.LAST, requires=("has_value",), aliases=("value_learning_rate",)),
    _spec("optim/lr/critic", Reduction.LAST, requires=("has_critic",), aliases=("critic_learning_rate",)),
    _spec("optim/lr/critic_b", Reduction.LAST, requires=("twin_critic",), aliases=("critic_b_learning_rate",)),
    _spec("sac/alpha", Reduction.LAST, agents=("sac",), aliases=("entropy_coefficient",)),
    _spec("exploration/noise_anneal", Reduction.LAST, agents=("ddpg", "td3"), aliases=("noise_anneal",)),
    _spec("td3/target_noise_anneal", Reduction.LAST, agents=("td3",), aliases=("target_noise_anneal",)),
    _spec("policy/temperature", Reduction.LAST, requires=("discrete",), aliases=("temperature",)),
)

#: Canonical key -> ``MetricSpec``. Immutable after import; never mutated by
#: ``bind()`` (capability gating produces per-instance enabled-key sets, not
#: registry edits).
METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType({spec.key: spec for spec in _SPECS})


def _build_alias_index(specs: tuple[MetricSpec, ...]) -> Mapping[str, tuple[str, ...]]:
    """Return legacy alias name -> tuple of canonical spec keys declaring it."""
    index: dict[str, list[str]] = {}
    for spec in specs:
        for alias in spec.aliases:
            index.setdefault(alias, []).append(spec.key)
    return MappingProxyType({alias: tuple(keys) for alias, keys in index.items()})


_ALIAS_INDEX: Mapping[str, tuple[str, ...]] = _build_alias_index(_SPECS)


class MetricsRecorder(Protocol):
    """Minimal push-based metrics interface shared by every recorder.

    Implemented by both ``NullRecorder`` (the no-op default) and
    ``MetricsManager``. Callbacks never see this interface directly; a
    ``Trainer`` reads ``collect()`` and hands the resulting flat dict to
    ``callback.on_train_step_end``.
    """

    def record(self, key: str, value: Any) -> None:
        """Push one scalar-like value under a canonical metric key."""
        ...

    def record_many(self, values: Mapping[str, Any]) -> None:
        """Push a batch of legacy-or-canonical-keyed values in one call."""
        ...

    def is_enabled(self, key: str) -> bool:
        """Return whether ``key`` currently resolves to an enabled spec."""
        ...

    def should_compute(self, key: str) -> bool:
        """Return whether a caller should bother computing ``key`` at all."""
        ...


class NullRecorder(MetricsRecorder):
    """No-op recorder used as ``Agent.metrics`` before a ``Trainer`` binds one.

    Every method is a no-op (or returns the "nothing enabled / nothing
    collected" value), so any code written against ``MetricsRecorder`` works
    unchanged on a standalone agent that was never attached to a ``Trainer``.

    Example:
        >>> from phoenx.metrics import NULL_RECORDER
        >>> NULL_RECORDER.record("loss/policy", 1.0)
        >>> NULL_RECORDER.is_enabled("loss/policy")
        False
    """

    def record(self, key: str, value: Any) -> None:
        """No-op.

        Args:
            key: Ignored.
            value: Ignored.
        """
        return None

    def record_many(self, values: Mapping[str, Any]) -> None:
        """No-op.

        Args:
            values: Ignored.
        """
        return None

    def is_enabled(self, key: str) -> bool:
        """Always ``False``: a null recorder has no enabled specs.

        Args:
            key: Ignored.

        Returns:
            ``False``.
        """
        return False

    def should_compute(self, key: str) -> bool:
        """Always ``False`` (mirrors ``is_enabled``).

        Args:
            key: Ignored.

        Returns:
            ``False``.
        """
        return False

    def bind(self, *, agent: Any, env: Any, buffer: Any = None) -> None:
        """No-op: a null recorder never gains capabilities.

        Args:
            agent: Ignored.
            env: Ignored.
            buffer: Ignored.
        """
        return None

    def set_context(self, *, env_step: int, grad_step: int | None = None) -> None:
        """No-op.

        Args:
            env_step: Ignored.
            grad_step: Ignored.
        """
        return None

    def collect(self) -> dict[str, float]:
        """Always ``{}``.

        Returns:
            An empty dict.
        """
        return {}

    def reset(self) -> None:
        """No-op."""
        return None


#: Module-level singleton; every ``Agent`` starts with ``metrics = NULL_RECORDER``.
NULL_RECORDER: NullRecorder = NullRecorder()


class MetricsManager(MetricsRecorder):
    """Owns metric namespacing, windowed reduction, cadence, and gating.

    Producers call ``record`` / ``record_many`` as values become available
    (rollout bookkeeping in ``Trainer.step``, PER diagnostics, each agent's
    ``learn()`` return dict); ``collect()`` reduces the current window into a
    flat ``dict[str, float]`` and clears window state (ROLLING / ROLLING_STD
    deques and RATE anchors survive). Capability gating (``bind``) decides
    which of the specs in ``METRIC_REGISTRY`` are enabled for this run; recording or
    reading a disabled key is always safe and always a no-op.

    Example:
        >>> mm = MetricsManager()
        >>> mm.record("rollout/return_mean", 0.5)
        >>> mm.collect()
        {'rollout/return_mean': 0.5}
    """

    def __init__(
        self,
        *,
        tiers: Sequence[int] = (0, 1, 2),
        emit_every: int = 1,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        dual_emit: bool = True,
    ) -> None:
        """Configure gating and cadence; no capability is enabled until ``bind()``.

        Args:
            tiers: Cost tiers enabled by default. Every registered spec is
                currently tier 0.
            emit_every: Number of ``collect()`` calls between emitting calls;
                ``1`` emits every call (today's per-step behavior).
            include: Optional exact-key allow-list applied AFTER capability
                gating; it cannot force a disabled spec (wrong agent/missing
                capability) to enable.
            exclude: Exact keys to force-disable regardless of tier/capability.
            dual_emit: When ``True``, ``collect()`` also emits every alias of
                an enabled spec under its legacy flat name.
        """
        self._tiers = tuple(tiers)
        self.emit_every = int(emit_every)
        self._include = tuple(include) if include is not None else None
        self._exclude = tuple(exclude)
        self.dual_emit = bool(dual_emit)

        self._agent_tag = ""
        self._capabilities: frozenset[str] = frozenset()
        self._enabled_keys: frozenset[str] = self._compute_enabled_keys()
        self._suppressed_keys: frozenset[str] = self._compute_suppressed_keys()

        self._window_values: dict[str, list[float]] = {}
        self._rolling: dict[str, deque] = {}
        self._rate_anchors: dict[str, tuple[float, float]] = {}
        self._since_emit = 0
        self._env_step: int | None = None
        self._grad_step: int | None = None

    # ------------------------------------------------------------------ #
    # Capability gating
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe(predicate) -> bool:
        """Evaluate a zero-arg capability predicate, treating a missing or wrong-shaped attribute as ``False``.

        Catches both ``AttributeError`` (a duck-typed attribute is simply
        absent, e.g. no ``agent.model``) and ``TypeError`` (an attribute is
        present but ``None`` or otherwise the wrong shape for the predicate,
        e.g. ``'value' in agent.model.branches`` when ``branches`` is
        ``None`` on a partial stub), so ``bind()`` never raises against a
        partial stub agent/env/buffer.
        """
        try:
            return bool(predicate())
        except (AttributeError, TypeError):
            return False

    def bind(self, *, agent: Any, env: Any, buffer: Any = None) -> None:
        """Duck-type ``agent`` / ``env`` / ``buffer`` into a capability set.

        Recomputes ``capabilities()`` and ``enabled_keys()``. Every predicate
        is wrapped so a missing attribute yields ``False`` rather than
        raising, so this succeeds against a partial stub agent/env/buffer
        (e.g. in tests). Never mutates ``METRIC_REGISTRY``.

        Args:
            agent: Live agent instance; only ``type(agent).__name__`` and a
                handful of duck-typed attributes (including ``agent.model``)
                are read.
            env: Live environment instance; only ``goal_key`` is read.
            buffer: Optional buffer instance; reads ``hasattr(buffer,
                'update_priorities')`` (PER), ``buffer_size`` and
                ``hasattr(buffer, 'fill')`` together (buffer occupancy /
                readiness), and ``hasattr(buffer, 'completed_trajectories')``
                (pending-trajectory count).
        """
        agent_class_name = type(agent).__name__
        self._agent_tag = agent_class_name.lower()

        caps: set[str] = {f"agent:{self._agent_tag}"}

        def _add(tag: str, predicate) -> None:
            if self._safe(predicate):
                caps.add(tag)

        _add("on_policy", lambda: agent_class_name in {"Reinforce", "ActorCritic", "PPO"})
        _add("off_policy", lambda: agent_class_name in {"DDPG", "TD3", "SAC"})
        _add("has_value", lambda: "value" in agent.model.branches)
        _add("has_critic", lambda: "critic" in agent.model.branches)
        _add("twin_critic", lambda: "critic_b" in agent.model.branches)
        _add("continuous", lambda: getattr(agent.policy, "distribution", None) != "categorical")
        _add("discrete", lambda: getattr(agent.policy, "distribution", None) == "categorical")
        _add("recurrent", lambda: bool(getattr(agent.model, "is_recurrent", False)))
        _add("per", lambda: hasattr(buffer, "update_priorities"))
        # `buffer_capacity` requires the `fill` API itself, not merely a
        # truthy `buffer_size`: a stub carrying only `buffer_size` would
        # otherwise enable three keys and then raise on the reads.
        _add("buffer_capacity", lambda: bool(getattr(buffer, "buffer_size", 0)) and hasattr(buffer, "fill"))
        _add("buffer_trajectories", lambda: hasattr(buffer, "completed_trajectories"))
        _add("optimizer_steps", lambda: hasattr(agent.model, "optimizer_steps"))
        _add("auto_entropy", lambda: bool(getattr(agent, "auto_entropy_tuning", False)))
        _add("intrinsic_motivation", lambda: getattr(agent, "intrinsic_motivation", None) is not None)
        _add("goal_conditioned", lambda: getattr(env, "goal_key", None) is not None)

        self._capabilities = frozenset(caps)
        self._enabled_keys = self._compute_enabled_keys()

    def capabilities(self) -> frozenset[str]:
        """Return the capability tag set from the last ``bind()`` call.

        Returns:
            Empty ``frozenset`` before ``bind()`` has ever run.
        """
        return self._capabilities

    def _compute_enabled_keys(self) -> frozenset[str]:
        """Recompute which registry keys pass tier-or-include/requires/agents/exclude gating.

        A spec is enabled iff ALL of: ``requires`` is a subset of
        ``capabilities()``; ``agents`` is empty or the bound agent tag is one
        of them; its key is not in ``exclude``; and -- when ``include`` is
        ``None`` -- its ``tier`` is in ``tiers``, or -- when ``include`` is
        not ``None`` -- its key is in ``include``.
        """
        tiers = set(self._tiers)
        include = set(self._include) if self._include is not None else None
        exclude = set(self._exclude)
        caps = self._capabilities
        agent_tag = self._agent_tag

        enabled: set[str] = set()
        for key, spec in METRIC_REGISTRY.items():
            if include is None:
                if spec.tier not in tiers:
                    continue
            elif key not in include:
                continue
            if not spec.requires <= caps:
                continue
            if spec.agents and agent_tag not in spec.agents:
                continue
            if key in exclude:
                continue
            enabled.add(key)
        return frozenset(enabled)

    def _compute_suppressed_keys(self) -> frozenset[str]:
        """Return the canonical keys and aliases of every ``exclude``d spec.

        Distinct from capability/agent/tier-disabled keys: a name in this
        set must be dropped outright by ``record_many`` rather than falling
        through to the unmapped bucket, so ``exclude`` suppresses a metric
        (and its legacy alias) instead of merely renaming it.
        """
        suppressed: set[str] = set()
        for key in self._exclude:
            spec = METRIC_REGISTRY.get(key)
            if spec is None:
                continue
            suppressed.add(spec.key)
            suppressed.update(spec.aliases)
        return frozenset(suppressed)

    def enabled_keys(self) -> frozenset[str]:
        """Return the canonical keys currently enabled by tier/capability/agent gating."""
        return self._enabled_keys

    def is_enabled(self, key: str) -> bool:
        """Return whether ``key`` is a currently-enabled canonical spec key.

        Args:
            key: Canonical ``METRIC_REGISTRY`` key to check.

        Returns:
            ``True`` iff ``key`` is in ``enabled_keys()``.
        """
        return key in self._enabled_keys

    def should_compute(self, key: str) -> bool:
        """Return whether a producer should bother computing ``key`` at all.

        Identical to ``is_enabled`` today: neither ``emit_every`` nor cadence
        affects computation gating, only emission.

        Args:
            key: Canonical ``METRIC_REGISTRY`` key to check.

        Returns:
            ``True`` iff ``key`` is in ``enabled_keys()``.
        """
        return self.is_enabled(key)

    # ------------------------------------------------------------------ #
    # Value coercion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(value: Any) -> float | None:
        """Coerce a recordable value to ``float``, or ``None`` if it must be dropped.

        A ``torch.Tensor`` with ``numel() == 1`` -- of ANY shape, including
        ``()``, ``(1,)``, and ``(1, 1)`` -- is detached and converted
        immediately via ``float(value.detach().reshape(()).item())`` (never
        retained). This is deliberately not a ``dim() == 0`` check:
        ``setup_auto_entropy`` builds ``log_alpha = T.zeros(1,
        requires_grad=True)``, so ``entropy_coefficient =
        self.log_alpha.exp()`` has shape ``(1,)``, not ``()``, and a
        dim-based check would silently drop it. A ``bool`` becomes ``float``.
        A tensor with ``numel() != 1`` (e.g. the per-sample ``td_errors``
        tensor), a ``Mapping``, a ``list``/``tuple``/``set``, ``None``, or a
        value that raises on ``float()`` (including a NumPy array with more
        than one element) is dropped.
        """
        if value is None:
            return None
        if isinstance(value, T.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().reshape(()).item())
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record(self, key: str, value: Any) -> None:
        """Push one value under a canonical (already-resolved) metric key.

        A silent no-op when ``key`` is not currently enabled. Never raises:
        an unrecordable ``value`` (wrong shape, wrong type, or non-finite) is
        dropped and logged at debug level. A non-finite value recorded under
        a ``loss/*`` key also sets the ``health/non_finite`` flag for this
        window (via a nested ``record`` call, so it is subject to the same
        rules).

        Args:
            key: Canonical ``METRIC_REGISTRY`` key (not a legacy alias --
                use ``record_many`` for alias resolution).
            value: Scalar-like value; see ``_coerce`` for accepted shapes.

        Example:
            >>> mm = MetricsManager()
            >>> mm.record("rollout/return_mean", 1.0)
            >>> mm.collect()
            {'rollout/return_mean': 1.0}
        """
        if key not in self._enabled_keys:
            return
        spec = METRIC_REGISTRY[key]
        coerced = self._coerce(value)
        if coerced is None:
            logger.debug("metrics: dropping unrecordable value for %r (%s)", key, type(value).__name__)
            return
        finite = math.isfinite(coerced)

        if spec.reduction in (Reduction.ROLLING, Reduction.ROLLING_STD):
            if finite:
                self._rolling.setdefault(key, deque(maxlen=spec.window)).append(coerced)
            else:
                logger.debug("metrics: dropping non-finite value for %r", key)
            if key.startswith("loss/"):
                self.record("health/non_finite", 0.0 if finite else 1.0)
            return

        if not finite:
            logger.debug("metrics: dropping non-finite value for %r", key)
            if key.startswith("loss/"):
                self.record("health/non_finite", 1.0)
            return

        self._window_values.setdefault(key, []).append(coerced)
        if key.startswith("loss/"):
            self.record("health/non_finite", 0.0)

    def record_many(self, values: Mapping[str, Any]) -> None:
        """Push a batch of legacy-or-canonical-keyed values, resolving aliases.

        For each ``(k, v)`` in ``values``, in iteration order:

        1. If ``k`` equals the key of an **enabled** spec -> records directly.
        2. Else if ``k`` equals the key of any *registered* spec (a canonical
           key that is not enabled) -> dropped. A canonical key must not
           bypass capability gating by arriving through ``record_many``, nor
           be emitted with ``LAST`` in place of its declared reduction.
        3. Else if ``k`` is suppressed -- the key or an alias of a spec named
           in ``exclude`` -- -> dropped. This is what makes ``exclude``
           actually suppress a metric instead of merely renaming it to its
           legacy alias.
        4. Else if ``k`` is an alias of exactly one enabled spec -> records
           under that spec's key (dual-emit then re-surfaces it under ``k``
           at ``collect()``).
        5. Else -> stores ``k`` into the unmapped bucket (``Reduction.LAST``
           semantics, same coercion/dropping rules as ``record()``), so a
           legacy name that aliases a spec disabled by *capability or agent*
           gating still reaches ``collect()`` under its own name instead of
           being silently dropped.

        Args:
            values: Mapping of legacy or canonical metric names to
                scalar-like values (typically an ``agent.learn()`` return
                dict).

        Example:
            >>> mm = MetricsManager()
            >>> mm.record_many({"policy_loss": 0.5, "some_custom_stat": 3.0})
            >>> sorted(mm.collect().items())
            [('health/non_finite', 0.0), ('loss/policy', 0.5), ('policy_loss', 0.5), ('some_custom_stat', 3.0)]
        """
        enabled = self._enabled_keys
        for k, v in values.items():
            if k in enabled:
                self.record(k, v)
                continue
            if k in METRIC_REGISTRY:
                logger.debug("metrics: dropping disabled canonical key %r via record_many", k)
                continue
            if k in self._suppressed_keys:
                logger.debug("metrics: dropping excluded key/alias %r via record_many", k)
                continue
            candidates = [spec_key for spec_key in _ALIAS_INDEX.get(k, ()) if spec_key in enabled]
            if len(candidates) == 1:
                self.record(candidates[0], v)
                continue
            coerced = self._coerce(v)
            if coerced is None:
                logger.debug("metrics: dropping unmapped value for %r (%s)", k, type(v).__name__)
                continue
            if not math.isfinite(coerced):
                logger.debug("metrics: dropping non-finite unmapped value for %r", k)
                continue
            self._window_values.setdefault(k, []).append(coerced)

    # ------------------------------------------------------------------ #
    # Context and collection
    # ------------------------------------------------------------------ #
    def set_context(self, *, env_step: int, grad_step: int | None = None) -> None:
        """Record the run-progress counters emitted alongside the next ``collect()``.

        Args:
            env_step: Cumulative environment steps so far this run.
            grad_step: Cumulative ``agent.learn()`` calls so far this run, or
                ``None`` to leave ``grad_step`` absent from ``collect()``.
        """
        self._env_step = int(env_step)
        self._grad_step = int(grad_step) if grad_step is not None else None

    def _reduce_rate(self, key: str, value: float) -> float | None:
        """Compute ``Δvalue / Δwall_clock`` against the persisted anchor for ``key``."""
        now = time.perf_counter()
        anchor = self._rate_anchors.get(key)
        self._rate_anchors[key] = (value, now)
        if anchor is None:
            return None
        prev_value, prev_time = anchor
        dt = now - prev_time
        if dt <= 0:
            return None
        return (value - prev_value) / dt

    def _reduce(self, spec: MetricSpec) -> float | None:
        """Reduce this window's recorded values for ``spec`` into one float."""
        if spec.reduction is Reduction.ROLLING:
            dq = self._rolling.get(spec.key)
            if not dq:
                return None
            return sum(dq) / len(dq)
        if spec.reduction is Reduction.ROLLING_STD:
            dq = self._rolling.get(spec.key)
            if dq is None or len(dq) < 2:
                return None
            return statistics.pstdev(dq)

        values = self._window_values.get(spec.key)
        if not values:
            return None

        if spec.reduction is Reduction.MEAN:
            return sum(values) / len(values)
        if spec.reduction is Reduction.LAST:
            return values[-1]
        if spec.reduction is Reduction.FIRST:
            return values[0]
        if spec.reduction is Reduction.MAX:
            return max(values)
        if spec.reduction is Reduction.MIN:
            return min(values)
        if spec.reduction is Reduction.SUM:
            return sum(values)
        if spec.reduction is Reduction.WINDOW_STD:
            if len(values) < 2:
                return None
            return statistics.pstdev(values)
        if spec.reduction is Reduction.RATE:
            return self._reduce_rate(spec.key, values[-1])
        return None

    def collect(self) -> dict[str, float]:
        """Reduce the current window into a flat metrics dict, then advance cadence.

        Returns ``{}`` when ``emit_every`` collected windows have not yet
        elapsed since the last emitting call; window state (everything except
        ROLLING / ROLLING_STD deques and RATE anchors, which persist) is
        retained across suppressed calls, not discarded. On an emitting call,
        returns every enabled spec with at least one recorded value this
        window (reduced per its ``Reduction``), every alias of those specs
        when ``dual_emit`` is ``True``, every unmapped-bucket key (as
        ``Reduction.LAST``), and ``env_step`` / ``grad_step`` (as ``int``)
        from the last ``set_context`` call, if any. A key with no value this
        window is absent -- never ``0.0``, never ``None``. ROLLING keys emit
        while their deque is non-empty; ROLLING_STD keys (population std,
        ``ddof=0``) emit while their deque holds at least two values.

        Returns:
            Flat mapping of metric name to float (plus the two integer
                ``env_step`` / ``grad_step`` context keys, when set).

        Example:
            >>> mm = MetricsManager(emit_every=2)
            >>> mm.record("rollout/return_mean", 1.0)
            >>> mm.collect()
            {}
            >>> mm.record("rollout/return_mean", 3.0)
            >>> mm.collect()
            {'rollout/return_mean': 2.0}
        """
        self._since_emit += 1
        if self._since_emit < self.emit_every:
            return {}
        self._since_emit = 0

        result: dict[str, float] = {}
        for key in self._enabled_keys:
            spec = METRIC_REGISTRY[key]
            value = self._reduce(spec)
            if value is None:
                continue
            result[spec.key] = value
            if self.dual_emit:
                for alias in spec.aliases:
                    result[alias] = value

        for key, values in self._window_values.items():
            if key in self._enabled_keys or not values:
                continue
            result[key] = values[-1]

        if self._env_step is not None:
            result["env_step"] = int(self._env_step)
        if self._grad_step is not None:
            result["grad_step"] = int(self._grad_step)

        self._window_values.clear()
        return result

    def reset(self) -> None:
        """Clear all window state, including ROLLING/ROLLING_STD deques and RATE anchors.

        Unlike an emitting ``collect()`` (which preserves ROLLING/ROLLING_STD/
        RATE state), this clears everything. Capability bindings and the last
        ``set_context`` values are untouched.
        """
        self._window_values.clear()
        self._rolling.clear()
        self._rate_anchors.clear()
        self._since_emit = 0

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def get_config(self) -> dict:
        """Return a JSON-safe, bind-independent constructor-kwarg mapping.

        Identical before and after ``bind()``, and identical across a
        save/load round trip -- contains no enums, no frozensets, and no
        derived (capability-dependent) key sets.

        Returns:
            Mapping with ``tiers`` (sorted ``list[int]``), ``emit_every``
                (``int``), ``include`` (``None`` or ``list[str]``), ``exclude``
                (sorted ``list[str]``), and ``dual_emit`` (``bool``).

        Example:
            >>> import json
            >>> mm = MetricsManager()
            >>> json.loads(json.dumps(mm.get_config())) == mm.get_config()
            True
        """
        return {
            "tiers": sorted(self._tiers),
            "emit_every": self.emit_every,
            "include": list(self._include) if self._include is not None else None,
            "exclude": sorted(self._exclude),
            "dual_emit": self.dual_emit,
        }

    @classmethod
    def load(cls, config: dict) -> "MetricsManager":
        """Rebuild a ``MetricsManager`` from a ``get_config()`` mapping.

        Args:
            config: Mapping as returned by ``get_config()``. Unpacked
                directly as constructor keyword arguments.

        Returns:
            A fresh, unbound ``MetricsManager``.

        Raises:
            TypeError: If ``config`` contains a key that is not a constructor
                parameter (normal keyword-argument behavior; unknown keys are
                never silently ignored).

        Example:
            >>> mm = MetricsManager(emit_every=4)
            >>> MetricsManager.load(mm.get_config()).get_config() == mm.get_config()
            True
        """
        return cls(**config)
