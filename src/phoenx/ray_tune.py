"""Ray Tune hyperparameter sweeps: schema resolution and the phase driver.

A sweep is one standalone YAML file, separate from the training config it
sweeps. It never edits that base config on disk; every trial resolves a
fresh, fully-materialized training config in memory. Top-level keys:

- ``base_config``: bundled or on-disk training YAML (see
  [load_config][phoenx.builder.load_config]).
- ``overrides``: dotted-path -> constant value, applied to every trial of
  every phase (e.g. ``{"schedule.stop_units": 20000000}``).
- ``defaults``: inherited by every phase, key-by-key; a phase's own key wins.
- ``blocks``: a reusable library of small fixed layer templates
  (``{block_name: {layers: [...], max_count?: int}}``) referenced by
  ``architecture`` sections. See [parse_block_library][phoenx.ray_tune.parse_block_library].
- ``phases``: a list of ``{name, search_space?, architecture?, optimizers?,
  tune?, ...}`` mappings run in series (see
  [normalize_phases][phoenx.ray_tune.normalize_phases]). A sweep with no
  ``phases`` key but a top-level ``search_space`` / ``tune`` / ``architecture``
  / ``optimizers`` is treated as one implicit phase.

Distributions use a ``dist:`` key rather than ``type:``, because ``type:`` is
already the discriminator for envs, agents, buffers, layers, heads, and
optimizers elsewhere in a training config; reusing it inside a search spec
would be ambiguous. See [parse_search_spec][phoenx.ray_tune.parse_search_spec]
for the full list of supported ``dist`` values, and
[build_search_space][phoenx.ray_tune.build_search_space] for how a phase's
``search_space`` / ``architecture`` / ``optimizers`` sections become a flat
Ray Tune search space.

The module is organized in two layers. The resolution layer handles
loading/validating a sweep, dotted-path get/set, search-spec parsing,
search-space construction, block-grammar architecture assembly, the
per-module ``optimizers:`` writer, ``auto_learn_every``, and trial-config
constraint validation. The driver layer builds searchers, schedulers and
stoppers, runs each phase as one ``tune.Tuner``, promotes the winning trial's
resolved config into the next phase, and writes the runnable
``best_config.yml``. The ``phoenx-tune`` CLI in
``phoenx.cli.tune`` drives it.
"""

from __future__ import annotations

import copy
import re
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import yaml
from ray import tune

from .builder import load_config
from .logging_config import get_logger

logger = get_logger("ray_tune")

#: Path prefix under which the roots/trunk/branches network schema lives in
#: a training config (see ``agent.config.model`` in ``builder.apply_model_config``).
_MODEL_PATH_PREFIX = "agent.config.model"

#: Buffer ``type`` names that use the on-policy ``(buffer_size, num_envs, ...)``
#: rollout layout (``RolloutBuffer`` and its ``TrajectoryBuffer`` subclass).
_ROLLOUT_BUFFER_TYPES = ("RolloutBuffer", "TrajectoryBuffer")

#: Layer registry names that may only appear in the trunk.
_TEMPORAL_LAYER_TYPES = frozenset({"lstm", "gru"})

#: Supported ``phase['promote']['mode']`` values. Currently just the
#: winner-becomes-next-base-config mode described in the module docstring;
#: any other value (including a capitalization typo) is rejected outright
#: by [validate_sweep_config][phoenx.ray_tune.validate_sweep_config] rather
#: than silently producing no promotion.
_SUPPORTED_PROMOTE_MODES = frozenset({"best"})


# =============================================================================
# Sweep / phase loading and validation
# =============================================================================

def available_example_sweeps() -> list[str]:
    """List bundled example sweep names under ``phoenx.examples/sweeps``.

    Recursively walks ``phoenx/examples/sweeps/`` and returns every ``.yml``
    file as a sorted forward-slash path relative to that root. Mirrors
    [available_example_configs][phoenx.builder.available_example_configs].

    Returns:
        Sorted list of relative sweep paths, e.g.
        ``["isaac_franka_cube_lift.yml", "lunarlander_ppo.yml"]``. Empty
        when the packaged ``sweeps`` directory is absent.

    Example:
        >>> from phoenx.ray_tune import available_example_sweeps
        >>> available_example_sweeps()  # doctest: +SKIP
        ['isaac_franka_cube_lift.yml', 'lunarlander_ppo.yml']
    """
    sweeps_root = resources.files("phoenx.examples").joinpath("sweeps")
    if not sweeps_root.is_dir():
        return []

    names: list[str] = []

    def _walk(node, prefix: str) -> None:
        for entry in node.iterdir():
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                _walk(entry, rel)
            elif entry.is_file() and entry.name.endswith(".yml"):
                names.append(rel)

    _walk(sweeps_root, "")
    return sorted(names)


def load_sweep_config(sweep_file: str | Path) -> dict:
    """Load a sweep YAML from disk or from bundled examples.

    Resolution order mirrors [load_config][phoenx.builder.load_config]: an
    existing on-disk path always wins, else a bundled copy under
    ``phoenx/examples/sweeps/`` is tried.

    Args:
        sweep_file: Filesystem path or bundled example name (e.g.
            ``LunarLanderContinuous-v3/ppo_sweep.yml``). Forward or backslash
            separators are accepted.

    Returns:
        Parsed YAML mapping (not yet validated; call
        [validate_sweep_config][phoenx.ray_tune.validate_sweep_config]).

    Raises:
        FileNotFoundError: If neither an on-disk path nor a bundled example
            resolves. The message names the request and lists available
            bundled sweeps.
    """
    path = Path(sweep_file)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj)

    if not path.is_absolute():
        parts = path.as_posix().split("/")
        packaged = resources.files("phoenx.examples").joinpath("sweeps", *parts)
        if packaged.is_file():
            return yaml.safe_load(packaged.read_text(encoding="utf-8"))

    available = available_example_sweeps()
    available_msg = ", ".join(available) if available else "(none)"
    raise FileNotFoundError(
        f"Sweep config not found: {sweep_file!s}. Bundled examples: {available_msg}"
    )


def normalize_phases(sweep: dict) -> list[dict]:
    """Expand a sweep into its ordered list of fully-inherited phase dicts.

    A sweep with a ``phases:`` list returns those phases, each merged with
    ``sweep['defaults']`` (defaults first, phase keys win). A sweep with no
    ``phases:`` key but a top-level ``search_space`` / ``tune`` /
    ``architecture`` / ``optimizers`` key is treated as a single implicit
    phase named ``"phase_0"``, so simple single-phase sweeps do not need to
    write a ``phases:`` wrapper.

    Args:
        sweep: Parsed sweep config (see [load_sweep_config][phoenx.ray_tune.load_sweep_config]).

    Returns:
        Non-empty list of phase dicts, each carrying a unique, non-empty
        ``"name"`` plus whichever of ``search_space`` / ``architecture`` /
        ``optimizers`` / ``tune`` / ``promote`` / ... apply after inheriting
        ``defaults``.

    Raises:
        ValueError: If ``phases`` is present but empty/non-list, a phase is
            not a mapping, a phase name is missing/empty/non-string, a phase
            name is duplicated, or neither ``phases`` nor any implicit-phase
            key is present.

    Example:
        >>> from phoenx.ray_tune import normalize_phases
        >>> sweep = {"search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        >>> names = [p["name"] for p in normalize_phases(sweep)]
        >>> names
        ['phase_0']
    """
    defaults = sweep.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"sweep['defaults'] must be a mapping, got {defaults!r}")

    if "phases" in sweep:
        raw_phases = sweep["phases"]
        if not isinstance(raw_phases, list) or not raw_phases:
            raise ValueError("sweep['phases'] must be a non-empty list")
        phases: list[dict] = []
        seen_names: set[str] = set()
        for i, raw in enumerate(raw_phases):
            if not isinstance(raw, dict):
                raise ValueError(f"sweep['phases'][{i}] must be a mapping, got {raw!r}")
            name = raw.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"sweep['phases'][{i}] must have a non-empty string 'name'")
            if name in seen_names:
                raise ValueError(f"Duplicate phase name: '{name}'")
            seen_names.add(name)
            merged = copy.deepcopy(defaults)
            merged.update(raw)
            phases.append(merged)
        return phases

    implicit_keys = ("search_space", "tune", "architecture", "optimizers")
    if any(k in sweep for k in implicit_keys):
        implicit = dict(defaults)
        for k in implicit_keys:
            if k in sweep:
                implicit[k] = sweep[k]
        implicit.setdefault("name", "phase_0")
        return [implicit]

    raise ValueError(
        "Sweep config has no 'phases' list and none of "
        f"{implicit_keys} at the top level; there is nothing to sweep."
    )


def validate_sweep_config(sweep: dict) -> None:
    """Validate a sweep config's structure before any compute happens.

    Checks ``base_config`` / ``overrides`` / ``defaults`` shape, parses the
    ``blocks`` library, normalizes phases (name uniqueness/non-emptiness),
    and — for every phase — builds its search space via
    [build_search_space][phoenx.ray_tune.build_search_space]. That last step
    transitively validates every ``dist:`` spec, every architecture block
    reference, and rejects a temporal block (``lstm``/``gru``/causal
    ``transformer_encoder``) declared for a module other than ``trunk``.

    Args:
        sweep: Parsed sweep config.

    Raises:
        ValueError: On any structural problem: missing/invalid
            ``base_config``, non-mapping ``overrides``/``defaults``/``tune``,
            a non-mapping ``promote``, an unsupported ``promote.mode``, an
            unknown ``promote`` key, ``promote.seed_next`` set without
            ``promote.mode: "best"``, an invalid ``optimizers`` field, or
            any error surfaced while building a phase's search space
            (block-library errors, unknown ``dist`` values,
            missing/unexpected spec keys, unknown block references, or a
            temporal block outside ``trunk``).
    """
    if not isinstance(sweep, dict):
        raise ValueError(f"Sweep config must be a mapping, got {type(sweep).__name__}")

    base_config = sweep.get("base_config")
    if not base_config or not isinstance(base_config, str):
        raise ValueError(
            f"Sweep config must declare a non-empty string 'base_config' path, got {base_config!r}"
        )
    if "overrides" in sweep and not isinstance(sweep["overrides"], dict):
        raise ValueError("sweep['overrides'] must be a mapping of dotted-path -> value")

    blocks = parse_block_library(sweep.get("blocks"))
    phases = normalize_phases(sweep)

    for phase in phases:
        name = phase["name"]
        if "tune" in phase and not isinstance(phase["tune"], dict):
            raise ValueError(f"Phase '{name}': 'tune' must be a mapping")
        promote_cfg = phase.get("promote")
        if promote_cfg is not None:
            if not isinstance(promote_cfg, dict):
                raise ValueError(f"Phase '{name}': 'promote' must be a mapping, got {promote_cfg!r}")
            unknown_keys = set(promote_cfg) - {"mode", "seed_next"}
            if unknown_keys:
                raise ValueError(
                    f"Phase '{name}': promote has unknown key(s) {sorted(unknown_keys)}; the only "
                    "supported keys are 'mode' and 'seed_next'"
                )
            promote_mode = promote_cfg.get("mode")
            if promote_mode is not None and promote_mode not in _SUPPORTED_PROMOTE_MODES:
                raise ValueError(
                    f"Phase '{name}': promote.mode={promote_mode!r} is not supported (supported: "
                    f"{sorted(_SUPPORTED_PROMOTE_MODES)}); an unsupported value silently skips "
                    "promotion, so the next phase would train against the un-tuned base config "
                    "with no other signal that anything went wrong"
                )
            if "seed_next" in promote_cfg and promote_mode != "best":
                raise ValueError(
                    f"Phase '{name}': promote.seed_next is set without promote.mode: 'best'; "
                    "seeding the next phase's searcher without also promoting the winning trial's "
                    "config is a confusing half-applied state. Add mode: 'best', or drop seed_next."
                )
        for module, fields in (phase.get("optimizers") or {}).items():
            if not isinstance(fields, dict):
                raise ValueError(f"Phase '{name}': optimizers['{module}'] must be a mapping")
            for field, value in fields.items():
                if is_search_spec(value):
                    parse_search_spec(value)
        try:
            build_search_space(phase, blocks)
        except Exception as exc:
            raise ValueError(f"Phase '{name}': invalid search space: {exc}") from exc


# =============================================================================
# Dotted-path get / set
# =============================================================================

_BRACKET_RE = re.compile(r"\[(-?\d+)\]")
_INT_RE = re.compile(r"^-?\d+$")


def _split_path(path: str) -> list[str]:
    """Normalize ``a.b[0].c`` and ``a.b.0.c`` into the same segment list."""
    normalized = _BRACKET_RE.sub(r".\1", path)
    segments = [s for s in normalized.split(".") if s != ""]
    if not segments:
        raise ValueError(f"Path '{path}' has no segments")
    return segments


def get_by_path(config: dict, path: str) -> Any:
    """Read a nested value by dotted path, with list-index support.

    List indices are accepted in either ``a.b.0.c`` or ``a.b[0].c`` form
    (both normalize to the same traversal). A missing intermediate segment
    **raises** rather than returning ``None`` — a typo'd path would
    otherwise silently read the wrong (base) value and look legitimate.

    Args:
        config: Mapping (typically a full training config) to read from.
        path: Dotted path, e.g. ``"agent.config.model.trunk.layer_config.0.params.units"``
            or the equivalent ``"agent.config.model.trunk.layer_config[0].params.units"``.

    Returns:
        The value found at ``path``.

    Raises:
        KeyError: If a dict segment is missing, or a segment tries to
            descend into a non-container value.
        ValueError: If a list segment is not a valid/in-range index, or
            ``path`` has no segments.

    Example:
        >>> from phoenx.ray_tune import get_by_path
        >>> cfg = {"schedule": {"learn_every": 8192}}
        >>> get_by_path(cfg, "schedule.learn_every")
        8192
    """
    segments = _split_path(path)
    node: Any = config
    for seg in segments:
        if isinstance(node, list):
            if not _INT_RE.match(seg):
                raise ValueError(f"Path '{path}': segment '{seg}' is not a valid list index")
            idx = int(seg)
            if not (-len(node) <= idx < len(node)):
                raise ValueError(f"Path '{path}': index '{seg}' out of range (len={len(node)})")
            node = node[idx]
        elif isinstance(node, dict):
            if seg not in node:
                raise KeyError(f"Path '{path}': missing segment '{seg}'")
            node = node[seg]
        else:
            raise KeyError(
                f"Path '{path}': cannot descend into segment '{seg}' "
                f"(parent is a {type(node).__name__}, not a dict or list)"
            )
    return node


def set_by_path(config: dict, path: str, value: Any, *, create: bool = False) -> None:
    """Write a nested value by dotted path, with list-index support, in place.

    Mirrors [get_by_path][phoenx.ray_tune.get_by_path]'s path syntax. By
    default (``create=False``) every segment — including the final one —
    must already exist, so this is a strict "update an existing value"
    operation that catches a typo'd path immediately. ``create=True`` is the
    opt-in used only by [apply_optimizers][phoenx.ray_tune.apply_optimizers],
    where intermediate **dicts** (never list elements) may legitimately not
    exist yet.

    Args:
        config: Mapping to mutate in place.
        path: Dotted path (same syntax as ``get_by_path``).
        value: Value to write at ``path``.
        create: When ``True``, create missing intermediate dict keys (and
            the final dict key) instead of raising. List indices are never
            created (out-of-range still raises) because appending at an
            arbitrary index is ambiguous.

    Raises:
        KeyError: If ``create`` is ``False`` and a dict segment (including
            the final one) is missing, or a segment tries to descend into a
            non-container value.
        ValueError: If a list segment is not a valid/in-range index, or
            ``path`` has no segments.

    Example:
        >>> from phoenx.ray_tune import get_by_path, set_by_path
        >>> cfg = {"schedule": {"learn_every": 8192}}
        >>> set_by_path(cfg, "schedule.learn_every", 4096)
        >>> get_by_path(cfg, "schedule.learn_every")
        4096
    """
    segments = _split_path(path)
    node: Any = config
    for seg in segments[:-1]:
        if isinstance(node, list):
            if not _INT_RE.match(seg):
                raise ValueError(f"Path '{path}': segment '{seg}' is not a valid list index")
            idx = int(seg)
            if not (-len(node) <= idx < len(node)):
                raise ValueError(f"Path '{path}': index '{seg}' out of range (len={len(node)})")
            node = node[idx]
        elif isinstance(node, dict):
            if seg not in node:
                if not create:
                    raise KeyError(f"Path '{path}': missing segment '{seg}'")
                node[seg] = {}
            node = node[seg]
        else:
            raise KeyError(
                f"Path '{path}': cannot descend into segment '{seg}' "
                f"(parent is a {type(node).__name__}, not a dict or list)"
            )

    last = segments[-1]
    if isinstance(node, list):
        if not _INT_RE.match(last):
            raise ValueError(f"Path '{path}': segment '{last}' is not a valid list index")
        idx = int(last)
        if not (-len(node) <= idx < len(node)):
            raise ValueError(f"Path '{path}': index '{last}' out of range (len={len(node)})")
        node[idx] = value
    elif isinstance(node, dict):
        if last not in node and not create:
            raise KeyError(f"Path '{path}': missing segment '{last}'")
        node[last] = value
    else:
        raise KeyError(
            f"Path '{path}': cannot set segment '{last}' "
            f"(parent is a {type(node).__name__}, not a dict or list)"
        )


# =============================================================================
# ``dist:`` search-spec parsing
# =============================================================================

#: Non-sampler keys legal inside any search spec, stripped before validating
#: the dist-specific key set. ``share_across_slots`` is only meaningful
#: inside a block-library layer param (see ``build_layer_stack``), but is
#: accepted anywhere for simplicity — it is always stripped, never sampled.
_SPEC_META_KEYS = frozenset({"dist", "share_across_slots"})

#: Required / optional (non-meta) keys per supported ``dist`` value.
_DIST_KEYS: dict[str, dict[str, set[str]]] = {
    "uniform": {"required": {"low", "high"}, "optional": set()},
    "loguniform": {"required": {"low", "high"}, "optional": set()},
    "quniform": {"required": {"low", "high", "q"}, "optional": set()},
    "randint": {"required": {"lower", "upper"}, "optional": set()},
    "qrandint": {"required": {"lower", "upper", "q"}, "optional": set()},
    "lograndint": {"required": {"lower", "upper"}, "optional": {"base"}},
    "choice": {"required": {"values"}, "optional": set()},
    "grid_search": {"required": {"values"}, "optional": set()},
    "randn": {"required": set(), "optional": {"mean", "sd"}},
    "fixed": {"required": {"value"}, "optional": set()},
}


def is_search_spec(obj: Any) -> bool:
    """Return whether ``obj`` is a ``dist:`` search spec.

    A search spec is exactly a dict containing the key ``"dist"``. Nothing
    else counts, so a plain dict that happens to look like config (e.g. an
    optimizer's ``{"type": "Adam", "params": {...}}``) is never mistaken for
    one.

    Args:
        obj: Value to test.

    Returns:
        ``True`` iff ``obj`` is a ``dict`` and ``"dist" in obj``.
    """
    return isinstance(obj, dict) and "dist" in obj


def parse_search_spec(spec: dict) -> Any:
    """Parse one ``dist:`` spec into the Ray Tune sampler (or constant) it maps to.

    Supported ``dist`` values, mapping 1:1 onto ``ray.tune`` samplers:

    - ``uniform`` (``low``, ``high``) -> ``tune.uniform``
    - ``loguniform`` (``low``, ``high``) -> ``tune.loguniform``
    - ``quniform`` (``low``, ``high``, ``q``) -> ``tune.quniform``
    - ``randint`` (``lower``, ``upper``) -> ``tune.randint`` — **upper is
      exclusive**, matching Ray's own semantics.
    - ``qrandint`` (``lower``, ``upper``, ``q``) -> ``tune.qrandint`` —
      upper exclusive.
    - ``lograndint`` (``lower``, ``upper``, optional ``base``) ->
      ``tune.lograndint`` — upper exclusive.
    - ``choice`` (``values``) -> ``tune.choice``
    - ``grid_search`` (``values``) -> ``tune.grid_search``
    - ``randn`` (optional ``mean``, ``sd``) -> ``tune.randn``
    - ``fixed`` (``value``) -> the plain value itself (Ray treats any
      non-sampler as a constant, so no wrapper is needed).

    ``share_across_slots`` is accepted and silently stripped on any spec
    (see [is_search_spec][phoenx.ray_tune.is_search_spec]); it is only
    meaningful inside a block-library layer param, consumed by
    [build_search_space][phoenx.ray_tune.build_search_space] /
    [build_layer_stack][phoenx.ray_tune.build_layer_stack].

    Args:
        spec: A ``{"dist": ..., ...}`` mapping.

    Returns:
        A ``ray.tune`` sampler object, or the plain constant for ``fixed``.

    Raises:
        ValueError: If ``spec`` is not a search spec, ``dist`` is not one of
            the supported names, or the spec's keys (other than ``dist`` /
            ``share_across_slots``) do not exactly match that ``dist``'s
            required/optional key set.

    Example:
        >>> from phoenx.ray_tune import parse_search_spec
        >>> parse_search_spec({"dist": "fixed", "value": 42})
        42
    """
    if not is_search_spec(spec):
        raise ValueError(f"Not a search spec (missing 'dist' key): {spec!r}")
    dist = spec["dist"]
    key_spec = _DIST_KEYS.get(dist)
    if key_spec is None:
        raise ValueError(
            f"Unknown dist '{dist}' in search spec {spec!r}; supported: {sorted(_DIST_KEYS)}"
        )

    present = set(spec) - _SPEC_META_KEYS
    missing = key_spec["required"] - present
    allowed = key_spec["required"] | key_spec["optional"]
    unexpected = present - allowed
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing required key(s) {sorted(missing)}")
        if unexpected:
            problems.append(f"unexpected key(s) {sorted(unexpected)}")
        raise ValueError(f"Invalid '{dist}' search spec {spec!r}: {'; '.join(problems)}")

    if dist == "fixed":
        return spec["value"]
    if dist == "uniform":
        return tune.uniform(spec["low"], spec["high"])
    if dist == "loguniform":
        return tune.loguniform(spec["low"], spec["high"])
    if dist == "quniform":
        return tune.quniform(spec["low"], spec["high"], spec["q"])
    if dist == "randint":
        return tune.randint(spec["lower"], spec["upper"])
    if dist == "qrandint":
        return tune.qrandint(spec["lower"], spec["upper"], spec["q"])
    if dist == "lograndint":
        if "base" in spec:
            return tune.lograndint(spec["lower"], spec["upper"], base=spec["base"])
        return tune.lograndint(spec["lower"], spec["upper"])
    if dist == "choice":
        return tune.choice(spec["values"])
    if dist == "grid_search":
        return tune.grid_search(spec["values"])
    if dist == "randn":
        return tune.randn(spec.get("mean", 0.0), spec.get("sd", 1.0))
    raise AssertionError(f"unreachable: dist '{dist}' passed key validation but has no handler")


def _max_depth_from_spec(module: str, depth_spec: dict) -> int:
    """Compute the largest int an architecture ``depth`` search spec can produce.

    Used to enumerate ``slot{i}`` keys under the flat max-depth encoding
    (see [build_search_space][phoenx.ray_tune.build_search_space]). For
    ``qrandint``/``lograndint`` this returns the same exclusive-upper-bound
    ceiling as ``randint`` (``upper - 1``), which is always a safe (possibly
    loose) upper bound even though ``q``-rounding may make it unreachable —
    unused high slots are simply never referenced when the stack is built.
    """
    dist = depth_spec.get("dist")
    if dist == "fixed":
        return int(depth_spec["value"])
    if dist in ("choice", "grid_search"):
        values = depth_spec.get("values") or []
        if not values:
            raise ValueError(f"architecture['{module}'].depth spec {depth_spec!r} has no values")
        return int(max(values))
    if dist in ("randint", "qrandint", "lograndint"):
        return int(depth_spec["upper"]) - 1
    raise ValueError(
        f"architecture['{module}'].depth dist='{dist}' cannot produce an integer depth; "
        "use randint/qrandint/lograndint/choice/grid_search/fixed"
    )


# =============================================================================
# Generic search-spec tree walk (shared by search-space building and
# layer-stack assembly)
# =============================================================================

def _iter_search_specs(node: Any, prefix: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(dotted_path, spec)`` for every search spec nested in ``node``.

    Walks dicts and lists recursively (covering ``params``, ``kernel_params``,
    and a layer's ``type`` itself, e.g. ``{"type": {"dist": "choice", ...}}``).
    Recursion stops at a search spec — its own keys (``dist``, ``low``, ...)
    are never themselves scanned.
    """
    if is_search_spec(node):
        yield prefix, node
        return
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_search_specs(v, child)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            child = f"{prefix}.{i}" if prefix else str(i)
            yield from _iter_search_specs(v, child)


def _substitute_search_specs(node: Any, prefix: str, resolver) -> Any:
    """Deep-copy ``node``, replacing every search spec with ``resolver(path, spec)``."""
    if is_search_spec(node):
        return resolver(prefix, node)
    if isinstance(node, dict):
        return {
            k: _substitute_search_specs(v, f"{prefix}.{k}" if prefix else str(k), resolver)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [
            _substitute_search_specs(v, f"{prefix}.{i}" if prefix else str(i), resolver)
            for i, v in enumerate(node)
        ]
    return node


# =============================================================================
# Block library / architecture search-space + assembly
# =============================================================================

def _spec_string_candidates(value: Any) -> set[str]:
    """String values a ``type``-like field could literally take on."""
    if isinstance(value, str):
        return {value}
    if is_search_spec(value):
        if value.get("dist") in ("choice", "grid_search"):
            return {v for v in value.get("values", []) if isinstance(v, str)}
    return set()


def _spec_could_be_truthy(value: Any) -> bool:
    """Whether a (possibly swept) field could evaluate truthy at trial time."""
    if is_search_spec(value):
        dist = value.get("dist")
        if dist == "fixed":
            return bool(value.get("value"))
        if dist in ("choice", "grid_search"):
            return any(bool(v) for v in value.get("values", []))
        return True  # any other sampled dist could land on a truthy value
    return bool(value)


def _layer_is_temporal(layer: dict) -> bool:
    """Whether a raw (possibly swept) layer template could be temporal.

    Conservative: if the layer's ``type`` is itself a ``choice``/``grid_search``
    spec, any candidate value of ``lstm``/``gru`` (or ``transformer_encoder``
    whose ``causal`` could be truthy) marks the whole layer as temporal, since
    a phase could sample into it.
    """
    candidates = _spec_string_candidates(layer.get("type"))
    if candidates & _TEMPORAL_LAYER_TYPES:
        return True
    if "transformer_encoder" in candidates:
        causal = (layer.get("params") or {}).get("causal", False)
        return _spec_could_be_truthy(causal)
    return False


def parse_block_library(blocks: dict | None) -> dict:
    """Parse a sweep's ``blocks:`` library into a lookup keyed by block name.

    Each block is ``{"layers": [...], "max_count"?: int}``, where ``layers``
    is a non-empty list of raw ``{"type", "params"?}`` layer templates that
    may embed ``dist:`` search specs anywhere (including inside ``params``,
    ``kernel_params``, or as the ``type`` itself).

    Args:
        blocks: The sweep's ``blocks`` mapping, or ``None``.

    Returns:
        ``{block_name: {"layers": [...], "max_count": int | None,
        "is_temporal": bool}}``. Empty dict when ``blocks`` is ``None``/empty.

    Raises:
        ValueError: If a block is not a mapping, or its ``layers`` is not a
            non-empty list.
    """
    if not blocks:
        return {}
    parsed: dict[str, dict] = {}
    for name, spec in blocks.items():
        if not isinstance(spec, dict) or "layers" not in spec:
            raise ValueError(f"Block '{name}' must be a mapping with a 'layers' list")
        layers = spec["layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"Block '{name}': 'layers' must be a non-empty list")
        parsed[name] = {
            "layers": layers,
            "max_count": spec.get("max_count"),
            "is_temporal": any(_layer_is_temporal(layer) for layer in layers if isinstance(layer, dict)),
        }
    return parsed


def _validate_arch_spec_shape(module: str, arch_spec: Any, blocks: dict) -> None:
    """Validate one module's ``architecture`` entry shape and block references.

    Raises:
        ValueError: If ``arch_spec`` is not a mapping, ``depth``/``blocks``
            is missing/empty, a referenced block name is unknown, or a
            referenced block contains a temporal layer while ``module`` is
            not ``"trunk"``.
    """
    if not isinstance(arch_spec, dict):
        raise ValueError(f"architecture['{module}'] must be a mapping, got {arch_spec!r}")
    if "depth" not in arch_spec:
        raise ValueError(f"architecture['{module}'] is missing required key 'depth'")
    block_names = arch_spec.get("blocks")
    if not block_names or not isinstance(block_names, list):
        raise ValueError(f"architecture['{module}'] is missing a non-empty 'blocks' list")

    is_trunk = module == "trunk"
    for block_name in block_names:
        block = blocks.get(block_name)
        if block is None:
            raise ValueError(
                f"architecture['{module}']: unknown block '{block_name}' "
                f"(declare it under the sweep's 'blocks:' library)"
            )
        if block["is_temporal"] and not is_trunk:
            raise ValueError(
                f"architecture['{module}']: block '{block_name}' contains a temporal layer "
                "(lstm/gru/causal transformer_encoder), which is only legal for module 'trunk'"
            )


def build_search_space(phase: dict, blocks: dict | None = None) -> dict:
    """Build the flat Ray Tune search space for one phase.

    Combines three sources into one flat ``dict[str, Any]``:

    - Dotted config paths from ``phase['search_space']``.
    - Synthetic architecture keys from ``phase['architecture']``, using the
      flat max-depth encoding: ``arch.<module>.depth``,
      ``arch.<module>.slot{i}.block``,
      ``arch.<module>.slot{i}.<block_name>.<param_path>`` (or, for a param
      marked ``share_across_slots: true``, the slot-free
      ``arch.<module>.<block_name>.<param_path>``), and
      ``arch.<module>.suffix.{i}.<param_path>`` /
      ``arch.<module>.prefix.{i}.<param_path>`` for suffix/prefix layers.
      Slots are enumerated up to the maximum depth the ``depth`` spec can
      produce (never ``tune.sample_from``, which only ``BasicVariantGenerator``
      supports).
    - Optimizer keys from ``phase['optimizers']``: ``opt.<module>.<field>``
      (e.g. ``opt.branches.policy.lr``).

    Every key string is constructed by string-formatting the declared module
    name (``roots.<name>`` / ``trunk`` / ``branches.<role>``), never by
    parsing it, so a module name containing dots is unambiguous.

    Args:
        phase: One normalized phase (see [normalize_phases][phoenx.ray_tune.normalize_phases]).
        blocks: Parsed block library (see [parse_block_library][phoenx.ray_tune.parse_block_library]);
            required when the phase has an ``architecture`` section, otherwise unused.

    Returns:
        Flat mapping from search-space key to a ``ray.tune`` sampler (or
        plain constant for a ``fixed`` spec / a literal optimizer field).

    Raises:
        ValueError: If any ``search_space`` value is not a search spec, an
            architecture module's shape is invalid (see
            ``_validate_arch_spec_shape``), a referenced block is unknown, or
            any embedded ``dist:`` spec fails to parse.

    Example:
        >>> from phoenx.ray_tune import build_search_space
        >>> phase = {"name": "refine", "search_space": {
        ...     "agent.config.discount": {"dist": "uniform", "low": 0.95, "high": 0.999}}}
        >>> space = build_search_space(phase)
        >>> "agent.config.discount" in space
        True
    """
    blocks = blocks or {}
    space: dict[str, Any] = {}

    for path, spec in (phase.get("search_space") or {}).items():
        if not is_search_spec(spec):
            raise ValueError(
                f"search_space['{path}'] must be a {{dist: ...}} spec, got {spec!r}"
            )
        space[path] = parse_search_spec(spec)

    for module, arch_spec in (phase.get("architecture") or {}).items():
        _validate_arch_spec_shape(module, arch_spec, blocks)
        depth_spec = arch_spec["depth"]
        block_names = list(arch_spec["blocks"])
        if is_search_spec(depth_spec):
            space[f"arch.{module}.depth"] = parse_search_spec(depth_spec)
            max_depth = _max_depth_from_spec(module, depth_spec)
        else:
            try:
                max_depth = int(depth_spec)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"architecture['{module}'].depth must be an int or a search spec, "
                    f"got {depth_spec!r}"
                ) from exc

        for i in range(max_depth):
            space[f"arch.{module}.slot{i}.block"] = tune.choice(block_names)

        for block_name in block_names:
            block = blocks[block_name]
            for param_path, spec in _iter_search_specs(block["layers"]):
                if spec.get("share_across_slots"):
                    space[f"arch.{module}.{block_name}.{param_path}"] = parse_search_spec(spec)
                else:
                    for i in range(max_depth):
                        space[f"arch.{module}.slot{i}.{block_name}.{param_path}"] = parse_search_spec(spec)

        for section in ("prefix", "suffix"):
            for i, layer in enumerate(arch_spec.get(section) or []):
                for param_path, spec in _iter_search_specs(layer):
                    space[f"arch.{module}.{section}.{i}.{param_path}"] = parse_search_spec(spec)

    for module, fields in (phase.get("optimizers") or {}).items():
        for field, value in (fields or {}).items():
            if is_search_spec(value):
                space[f"opt.{module}.{field}"] = parse_search_spec(value)

    return space


def _select_block_respecting_max_count(
    chosen: str, block_names: list[str], blocks: dict, counts: dict[str, int], module: str, slot_idx: int
) -> str:
    """Return the block to actually use for a slot, honoring ``max_count``.

    Fallback strategy: if ``chosen`` (the sampled block for this slot) has
    already been used ``max_count`` times, walk the declared ``block_names``
    order starting right after ``chosen``'s position (wrapping around) and
    use the first block whose count is still under its own ``max_count``
    (unlimited when unset). Raises if every declared block is exhausted.
    """
    if chosen not in block_names:
        raise ValueError(
            f"architecture['{module}'] slot {slot_idx}: sampled block '{chosen}' is not "
            f"among the declared blocks {block_names}"
        )
    n = len(block_names)
    start = block_names.index(chosen)
    for offset in range(n):
        candidate = block_names[(start + offset) % n]
        max_count = blocks.get(candidate, {}).get("max_count")
        if max_count is None or counts.get(candidate, 0) < max_count:
            return candidate
    raise ValueError(
        f"architecture['{module}'] slot {slot_idx}: every declared block {block_names} "
        "has reached its 'max_count'; cannot fill this slot"
    )


def build_layer_stack(module: str, arch_spec: dict, sampled: dict, blocks: dict) -> list[dict]:
    """Assemble a concrete ``layer_config`` list for one module from sampled values.

    For each slot ``i`` in ``range(depth)`` (``depth`` read from
    ``sampled[f'arch.{module}.depth']`` when swept, else the literal
    ``arch_spec['depth']``), looks up the slot's chosen block (falling back
    per ``_select_block_respecting_max_count``
    when the sampled block has exhausted its ``max_count``), deep-copies that
    block's layer templates, substitutes every embedded search spec with its
    sampled value, and appends the result. ``prefix`` entries (if any) are
    emitted before the slots; ``suffix`` entries after.

    Args:
        module: Canonical module name (``roots.<name>`` / ``trunk`` /
            ``branches.<role>``), used verbatim (never parsed) to build
            ``sampled`` lookup keys.
        arch_spec: This module's ``{"depth", "blocks", "suffix"?, "prefix"?}``
            architecture spec.
        sampled: Flat ``{search_space_key: concrete_value}`` mapping — a
            resolved Ray Tune trial config (or a hand-built dict of the same
            shape).
        blocks: Parsed block library (see [parse_block_library][phoenx.ray_tune.parse_block_library]).

    Returns:
        A ``layer_config`` list ready to write to
        ``agent.config.model.<module>.layer_config``.

    Raises:
        KeyError: If a required key is missing from ``sampled`` (e.g. a
            slot's block choice, or a swept layer param's sampled value).
        ValueError: If ``depth`` cannot be resolved, or ``max_count``
            exhausts every declared block for a slot.

    Example:
        >>> from phoenx.ray_tune import build_layer_stack, parse_block_library
        >>> blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 64}}, {"type": "relu"}]}})
        >>> stack = build_layer_stack("trunk", {"depth": 1, "blocks": ["dense_block"]}, {"arch.trunk.slot0.block": "dense_block"}, blocks)
        >>> types = [l["type"] for l in stack]
        >>> types
        ['dense', 'relu']
    """
    block_names = list(arch_spec["blocks"])
    depth_key = f"arch.{module}.depth"
    if depth_key in sampled:
        depth = int(sampled[depth_key])
    else:
        depth_spec = arch_spec["depth"]
        if is_search_spec(depth_spec):
            raise KeyError(
                f"build_layer_stack: '{depth_key}' not found in 'sampled' but "
                f"architecture['{module}'].depth is a search spec"
            )
        depth = int(depth_spec)

    stack: list[dict] = []

    def _emit_section(entries: list, section: str) -> None:
        for i, layer in enumerate(entries):
            def resolver(path: str, spec: dict, _i=i, _section=section) -> Any:
                key = f"arch.{module}.{_section}.{_i}.{path}" if path else f"arch.{module}.{_section}.{_i}"
                if key not in sampled:
                    raise KeyError(f"build_layer_stack: missing sampled key '{key}'")
                return sampled[key]
            stack.append(_substitute_search_specs(layer, "", resolver))

    _emit_section(arch_spec.get("prefix") or [], "prefix")

    counts: dict[str, int] = {name: 0 for name in block_names}
    for i in range(depth):
        block_key = f"arch.{module}.slot{i}.block"
        if block_key in sampled:
            chosen = sampled[block_key]
        elif len(block_names) == 1:
            chosen = block_names[0]
        else:
            raise KeyError(
                f"build_layer_stack: missing sampled key '{block_key}' "
                f"(needed to choose among {block_names})"
            )
        block_name = _select_block_respecting_max_count(chosen, block_names, blocks, counts, module, i)
        counts[block_name] += 1
        block = blocks.get(block_name)
        if block is None:
            raise ValueError(f"build_layer_stack: unknown block '{block_name}' for module '{module}'")

        def resolver(path: str, spec: dict, _slot=i, _block_name=block_name) -> Any:
            if spec.get("share_across_slots"):
                key = f"arch.{module}.{_block_name}.{path}"
            else:
                key = f"arch.{module}.slot{_slot}.{_block_name}.{path}"
            if key not in sampled:
                raise KeyError(f"build_layer_stack: missing sampled key '{key}'")
            return sampled[key]

        for layer_idx, layer in enumerate(block["layers"]):
            stack.append(_substitute_search_specs(layer, str(layer_idx), resolver))

    _emit_section(arch_spec.get("suffix") or [], "suffix")
    return stack


def apply_architecture(config: dict, phase: dict, sampled: dict, blocks: dict) -> None:
    """Assemble and write every swept module's ``layer_config`` into ``config``.

    For each module in ``phase['architecture']``, builds its layer stack via
    [build_layer_stack][phoenx.ray_tune.build_layer_stack] and writes it to
    ``agent.config.model.<module>.layer_config`` via
    [set_by_path][phoenx.ray_tune.set_by_path] with ``create=False`` — the
    module (and its existing ``layer_config`` key) must already exist in
    ``config`` (roots additionally need their ``input_keys`` already
    declared), since this is not the optimizers writer's create-on-demand
    exception.

    Args:
        config: Full training config, mutated in place.
        phase: The current phase (reads ``phase['architecture']``; a no-op
            when absent).
        sampled: Resolved Ray Tune trial values (see
            [build_layer_stack][phoenx.ray_tune.build_layer_stack]).
        blocks: Parsed block library.

    Raises:
        ValueError: If a module named in ``phase['architecture']`` does not
            already exist (with its ``layer_config`` key) in ``config``.
        KeyError: Propagated from ``build_layer_stack`` when a required
            ``sampled`` key is missing.
    """
    arch = phase.get("architecture")
    if not arch:
        return
    for module, arch_spec in arch.items():
        _validate_arch_spec_shape(module, arch_spec, blocks)
        stack = build_layer_stack(module, arch_spec, sampled, blocks)
        module_path = f"{_MODEL_PATH_PREFIX}.{module}"
        try:
            get_by_path(config, module_path)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"architecture['{module}']: module does not exist in the base config at "
                f"'{module_path}' (roots need their input_keys declared already): {exc}"
            ) from exc
        try:
            set_by_path(config, f"{module_path}.layer_config", stack, create=False)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"architecture['{module}']: '{module_path}.layer_config' does not exist in "
                f"the base config; architecture search replaces an existing layer_config, it "
                f"does not create one: {exc}"
            ) from exc


# =============================================================================
# Per-module ``optimizers:`` writer
# =============================================================================

def apply_optimizers(config: dict, phase: dict, sampled: dict) -> None:
    """Write per-module ``optimizer_params`` blocks from ``phase['optimizers']``.

    ``phase['optimizers']`` is ``{module: {field: spec_or_literal, ...}}``.
    For each module this becomes
    ``agent.config.model.<module>.optimizer_params = {'type': ..., 'params': {...}}``:

    - ``type`` (a plain string or a search spec) sets the optimizer type.
      When not swept for this module, it inherits — in order — the
      existing module-own ``optimizer_params['type']``, else the model-wide
      ``agent.config.model.optimizer_params['type']``, else defaults to
      ``'Adam'``.
    - Every other field (``lr``, ``weight_decay``, ``eps``, ...) is merged
      into ``params``, overriding any existing module-own value of the same
      name but leaving other existing ``params`` keys untouched.

    This is the one function in this module allowed to create missing keys
    (via [set_by_path][phoenx.ray_tune.set_by_path]'s ``create=True``): a
    module's own ``optimizer_params`` sub-dict commonly does not exist yet
    (only the model-wide default does). The module itself, however, must
    already exist in ``config`` — a typo'd module name still raises.

    Args:
        config: Full training config, mutated in place.
        phase: The current phase (reads ``phase['optimizers']``; a no-op
            when absent).
        sampled: Resolved Ray Tune trial values, keyed
            ``opt.<module>.<field>`` for any swept field (see
            [build_search_space][phoenx.ray_tune.build_search_space]).

    Raises:
        ValueError: If ``phase['optimizers'][module]`` is not a mapping, or
            ``module`` does not exist in ``config``.
        KeyError: If a field is a search spec but its ``opt.<module>.<field>``
            key is missing from ``sampled``.

    Example:
        >>> from phoenx.ray_tune import apply_optimizers, get_by_path
        >>> cfg = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        >>> phase = {"optimizers": {"trunk": {"lr": 0.0003}}}
        >>> apply_optimizers(cfg, phase, {})
        >>> get_by_path(cfg, "agent.config.model.trunk.optimizer_params")
        {'type': 'Adam', 'params': {'lr': 0.0003}}
    """
    optimizers_spec = phase.get("optimizers")
    if not optimizers_spec:
        return

    try:
        model_wide = get_by_path(config, f"{_MODEL_PATH_PREFIX}.optimizer_params")
    except (KeyError, ValueError):
        model_wide = None

    for module, fields in optimizers_spec.items():
        if not isinstance(fields, dict):
            raise ValueError(f"optimizers['{module}'] must be a mapping, got {fields!r}")

        module_path = f"{_MODEL_PATH_PREFIX}.{module}"
        try:
            get_by_path(config, module_path)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"optimizers['{module}']: module does not exist in the base config at "
                f"'{module_path}'"
            ) from exc

        try:
            existing = get_by_path(config, f"{module_path}.optimizer_params")
        except (KeyError, ValueError):
            existing = None
        existing_type = existing.get("type") if isinstance(existing, dict) else None
        existing_params = dict(existing.get("params") or {}) if isinstance(existing, dict) else {}

        type_field = fields.get("type")
        if type_field is not None:
            if is_search_spec(type_field):
                key = f"opt.{module}.type"
                if key not in sampled:
                    raise KeyError(f"apply_optimizers: missing sampled key '{key}'")
                resolved_type = sampled[key]
            else:
                resolved_type = type_field
        else:
            resolved_type = (
                existing_type
                or (model_wide.get("type") if isinstance(model_wide, dict) else None)
                or "Adam"
            )

        params = existing_params
        for field, value in fields.items():
            if field == "type":
                continue
            if is_search_spec(value):
                key = f"opt.{module}.{field}"
                if key not in sampled:
                    raise KeyError(f"apply_optimizers: missing sampled key '{key}'")
                params[field] = sampled[key]
            else:
                params[field] = value

        set_by_path(
            config, f"{module_path}.optimizer_params", {"type": resolved_type, "params": params}, create=True
        )


# =============================================================================
# ``auto_learn_every`` + trial-config constraint validation
# =============================================================================

def apply_auto_learn_every(config: dict) -> int | None:
    """Set ``schedule.learn_every`` to keep an on-policy rollout exactly full.

    ``RolloutBuffer``/``TrajectoryBuffer`` allocate ``(buffer_size, num_envs,
    ...)`` storage with no wrap-around and no bounds check, and
    ``learn_every`` counts global steps (incremented by ``num_envs`` per
    iteration). When the buffer is one of those two types and
    ``schedule.learn_every_unit == 'timestep'``, this sets
    ``schedule.learn_every = buffer.config.buffer_size * env.config.num_envs``
    in place, logs the computed value, and returns it — so sweeping either
    ``buffer_size`` or ``num_envs`` never produces a mid-run ``IndexError``
    (rollout overflow) or a silent partial-rollout run.

    Args:
        config: Full training config, mutated in place when applicable.

    Returns:
        The computed ``learn_every`` value, or ``None`` when this does not
        apply (an off-policy buffer, or ``learn_every_unit != 'timestep'``).
    """
    schedule = config.get("schedule") or {}
    if schedule.get("learn_every_unit") != "timestep":
        return None
    buffer_type = (config.get("buffer") or {}).get("type")
    if buffer_type not in _ROLLOUT_BUFFER_TYPES:
        return None

    try:
        buffer_size = get_by_path(config, "buffer.config.buffer_size")
        num_envs = get_by_path(config, "env.config.num_envs")
    except (KeyError, ValueError):
        return None

    computed = int(buffer_size) * int(num_envs)
    set_by_path(config, "schedule.learn_every", computed, create=False)
    logger.info(
        "auto_learn_every: schedule.learn_every=%d (buffer.config.buffer_size=%d * "
        "env.config.num_envs=%d)",
        computed, buffer_size, num_envs,
    )
    return computed


def _model_is_recurrent(config: dict) -> bool:
    """Detect recurrence from a temporal layer in ``agent.config.model.trunk.layer_config``."""
    try:
        trunk_layers = get_by_path(config, f"{_MODEL_PATH_PREFIX}.trunk.layer_config")
    except (KeyError, ValueError):
        return False
    if not isinstance(trunk_layers, list):
        return False
    for layer in trunk_layers:
        if not isinstance(layer, dict):
            continue
        layer_type = layer.get("type")
        if layer_type in _TEMPORAL_LAYER_TYPES:
            return True
        if layer_type == "transformer_encoder" and (layer.get("params") or {}).get("causal"):
            return True
    return False


def validate_trial_config(config: dict) -> None:
    """Run the resolved-config constraint checks before anything is built.

    Applied to one fully-resolved trial config (after architecture and
    optimizers have already been written in). Checks, in order:

    1. **Rollout capacity** (on-policy buffers, ``learn_every_unit ==
       'timestep'``): rejects ``learn_every / num_envs > buffer_size`` (the
       ``RolloutBuffer.add`` out-of-bounds case) and warns (does not raise)
       when it is strictly less (silent partial rollouts).
    2. **Feedforward zero-batch**: rejects ``mini_batch_size`` greater than
       the feedforward rollout batch (``buffer_size * num_envs``), since
       ``num_valid // mini_batch_size == 0`` means zero gradient updates.
       An explicitly-swept ``mini_batch_size: 0`` reaches this check too
       (``0`` is a real sampled value, not "not swept"); only a genuinely
       *absent* ``mini_batch_size`` short-circuits it.
    3. **Recurrent env-unit minibatches**: for a recurrent trunk (see
       ``_model_is_recurrent``), rejects a ``mini_batch_size`` that does not evenly divide
       ``num_envs`` or is non-positive (including an explicitly-swept
       ``0``, which is a real sampled value, not "not swept"), since the
       recurrent PPO path silently falls back to using every env
       otherwise.
    4. **SAC N-step triple**: when two or more of ``agent.config.N``,
       ``buffer.config.N``, and an ``env.config.wrappers`` entry of type
       ``VectorNStepReward``'s ``params.n`` are present, rejects a mismatch.

    Args:
        config: Fully-resolved trial config (post architecture/optimizers).

    Raises:
        ValueError: On any of the four constraint violations above, naming
            the offending values and the corrective action.
    """
    schedule = config.get("schedule") or {}
    env_cfg = (config.get("env") or {}).get("config") or {}
    buffer_section = config.get("buffer") or {}
    buffer_type = buffer_section.get("type")
    buffer_cfg = buffer_section.get("config") or {}

    num_envs = env_cfg.get("num_envs")
    mini_batch_size = schedule.get("mini_batch_size")
    learn_every = schedule.get("learn_every")
    learn_every_unit = schedule.get("learn_every_unit")
    buffer_size = buffer_cfg.get("buffer_size")

    is_rollout_buffer = buffer_type in _ROLLOUT_BUFFER_TYPES
    is_recurrent = _model_is_recurrent(config)

    if is_rollout_buffer and learn_every_unit == "timestep" and num_envs and buffer_size and learn_every:
        iterations = learn_every / num_envs
        if iterations > buffer_size:
            raise ValueError(
                f"schedule.learn_every={learn_every} / env.config.num_envs={num_envs} = "
                f"{iterations:g} iterations between learns, which exceeds "
                f"buffer.config.buffer_size={buffer_size}. RolloutBuffer.add has no wrap-around "
                "and no bounds check, so this raises IndexError mid-training. Lower "
                "schedule.learn_every, raise buffer.config.buffer_size, or enable auto_learn_every."
            )
        if iterations < buffer_size:
            logger.warning(
                "schedule.learn_every=%s / env.config.num_envs=%s = %g iterations between "
                "learns, under buffer.config.buffer_size=%s: trials will train on partial "
                "(%g-step) rollouts instead of the full %s-step buffer.",
                learn_every, num_envs, iterations, buffer_size, iterations, buffer_size,
            )

    if is_rollout_buffer and not is_recurrent and mini_batch_size is not None and buffer_size and num_envs:
        rollout_batch = buffer_size * num_envs
        if mini_batch_size <= 0:
            raise ValueError(
                f"schedule.mini_batch_size={mini_batch_size} must be positive: the "
                "feedforward path computes num_valid // mini_batch_size, which raises "
                "ZeroDivisionError at the first learn call. Raise schedule.mini_batch_size "
                "above 0 (or raise the lower bound of its search spec)."
            )
        if mini_batch_size > rollout_batch:
            raise ValueError(
                f"schedule.mini_batch_size={mini_batch_size} exceeds the feedforward rollout "
                f"batch (buffer.config.buffer_size={buffer_size} * env.config.num_envs="
                f"{num_envs} = {rollout_batch}). num_valid // mini_batch_size == 0 means this "
                "trial would perform zero gradient updates per learn call. Lower "
                "schedule.mini_batch_size or raise the rollout batch."
            )

    if is_rollout_buffer and is_recurrent and mini_batch_size is not None and num_envs:
        if mini_batch_size <= 0 or mini_batch_size > num_envs or num_envs % mini_batch_size != 0:
            raise ValueError(
                f"schedule.mini_batch_size={mini_batch_size} is in env units for a recurrent "
                f"trunk and must evenly divide env.config.num_envs={num_envs} (and be in "
                f"(0, {num_envs}]); the recurrent PPO path silently falls back to using all "
                f"{num_envs} envs otherwise, making the swept value a no-op."
            )

    def _safe_get(path: str) -> Any:
        try:
            return get_by_path(config, path)
        except (KeyError, ValueError):
            return None

    wrapper_n = None
    for wrapper in env_cfg.get("wrappers") or []:
        if isinstance(wrapper, dict) and wrapper.get("type") == "VectorNStepReward":
            wrapper_n = (wrapper.get("params") or {}).get("n")
            break

    n_values = {
        k: v
        for k, v in (
            ("agent.config.N", _safe_get("agent.config.N")),
            ("buffer.config.N", _safe_get("buffer.config.N")),
            ("VectorNStepReward.params.n", wrapper_n),
        )
        if v is not None
    }
    if len(set(n_values.values())) > 1:
        raise ValueError(
            f"N-step mismatch across {n_values}: agent.config.N, buffer.config.N, and any "
            "VectorNStepReward wrapper's params.n must all agree, or SAC's N-step return "
            "computation, the replay buffer's stored window, and the reward wrapper disagree "
            "about the window length."
        )


# =============================================================================
# Trial-config resolution
# =============================================================================

def resolve_trial_config(sweep: dict, phase: dict, sampled: dict, base_config: dict | None = None) -> dict:
    """Resolve one trial's fully-materialized training config.

    Pipeline, in order (order matters: architecture and optimizers must land
    before the constraint check runs):

    1. Deep-copy ``base_config`` (or ``load_config(sweep['base_config'])``
       when not given).
    2. Apply ``sweep['overrides']`` (dotted path -> constant, the same for
       every trial of every phase).
    3. Apply ``phase['search_space']``'s dotted paths using the matching
       values from ``sampled``.
    4. [apply_architecture][phoenx.ray_tune.apply_architecture].
    5. [apply_optimizers][phoenx.ray_tune.apply_optimizers].
    6. [apply_auto_learn_every][phoenx.ray_tune.apply_auto_learn_every], when
       ``phase.get('auto_learn_every', True)`` (the sweep's
       ``defaults.auto_learn_every`` flows into this via
       [normalize_phases][phoenx.ray_tune.normalize_phases]'s key-by-key
       inheritance).
    7. [validate_trial_config][phoenx.ray_tune.validate_trial_config].

    Args:
        sweep: Parsed sweep config (used for ``base_config``, ``overrides``,
            and ``blocks``).
        phase: The current normalized phase.
        sampled: Flat ``{search_space_key: concrete_value}`` mapping for this
            trial — a resolved Ray Tune trial ``config`` (or a hand-built
            dict of the same shape), matching the keys
            [build_search_space][phoenx.ray_tune.build_search_space] would
            produce for this phase.
        base_config: Optional pre-loaded base config to deep-copy instead of
            loading ``sweep['base_config']`` from disk (useful for tests, or
            when a previous phase's promoted config is the new base).

    Returns:
        The fully-resolved, trial-ready training config dict.

    Raises:
        KeyError: If a dotted ``search_space`` path has no matching entry in
            ``sampled``, or a required architecture/optimizer sampled key is
            missing.
        ValueError: Propagated from any resolution step (invalid path,
            invalid architecture/optimizer spec, or a failed constraint
            check).

    Example:
        >>> from phoenx.ray_tune import get_by_path, resolve_trial_config
        >>> sweep = {"base_config": "LunarLanderContinuous-v3/ppo.yml"}
        >>> phase = {"name": "refine", "search_space": {
        ...     "agent.config.discount": {"dist": "uniform", "low": 0.95, "high": 0.999}}}
        >>> sampled = {"agent.config.discount": 0.97}
        >>> resolved = resolve_trial_config(sweep, phase, sampled)  # doctest: +SKIP
        >>> get_by_path(resolved, "agent.config.discount")  # doctest: +SKIP
        0.97
    """
    if base_config is not None:
        config = copy.deepcopy(base_config)
    else:
        config = copy.deepcopy(load_config(sweep["base_config"]))

    for path, value in (sweep.get("overrides") or {}).items():
        set_by_path(config, path, value, create=False)

    for path in (phase.get("search_space") or {}):
        if path not in sampled:
            raise KeyError(f"resolve_trial_config: missing sampled value for search_space key '{path}'")
        set_by_path(config, path, sampled[path], create=False)

    blocks = parse_block_library(sweep.get("blocks"))
    apply_architecture(config, phase, sampled, blocks)
    apply_optimizers(config, phase, sampled)

    if phase.get("auto_learn_every", True):
        apply_auto_learn_every(config)

    validate_trial_config(config)
    return config


# =============================================================================
# Phase driver: searcher / scheduler / stopper factories, the trainable,
# per-phase Tuner runs (with winner promotion + points_to_evaluate seeding),
# artifacts, and write_best_config.
# =============================================================================

import json
import math
import warnings

import numpy as np
import ray
from ray.tune.schedulers import create_scheduler
from ray.tune.search import create_searcher
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Domain
from ray.tune.stopper import (
    CombinedStopper,
    ExperimentPlateauStopper,
    MaximumIterationStopper,
    Stopper,
    TimeoutStopper,
    TrialPlateauStopper,
)

from .builder import build_trainer_from_config

#: ``search_alg.type`` values verified (per Ray's own constructors) to accept
#: a ``points_to_evaluate`` keyword at construction time: the random
#: generator, Optuna, and HyperOpt. Any other type gets a logged warning and
#: no seeding rather than a silently-dropped or crashing kwarg.
_POINTS_TO_EVALUATE_SUPPORTED = frozenset({"random", "optuna", "hyperopt"})

#: Friendly scheduler aliases not registered under that name in Ray's own
#: ``create_scheduler`` alias table.
_SCHEDULER_ALIASES: dict[str, str] = {"asha": "async_hyperband"}

#: ``scheduler.type`` names that mean Population Based Training, rejected
#: with a clear message since PBT needs checkpoint plumbing not implemented
#: in this delivery pass (see [build_scheduler][phoenx.ray_tune.build_scheduler]).
_PBT_SCHEDULER_TYPES = frozenset({"pbt", "population_based_training"})

#: ``stop:`` ``{"type": ...}`` names mapped onto the Ray stopper classes
#: this module supports (see [build_stopper][phoenx.ray_tune.build_stopper]).
_STOPPER_CLASSES: dict[str, type] = {
    "TrialPlateauStopper": TrialPlateauStopper,
    "ExperimentPlateauStopper": ExperimentPlateauStopper,
    "MaximumIterationStopper": MaximumIterationStopper,
    "TimeoutStopper": TimeoutStopper,
    "CombinedStopper": CombinedStopper,
}

#: Callback ``type`` names [rl_trainable][phoenx.ray_tune.rl_trainable] fully
#: owns per trial (removed from the base config's own ``callbacks`` list
#: before re-adding exactly the right set; see ``_inject_trial_callbacks``).
_TRIAL_MANAGED_CALLBACK_TYPES = frozenset({"RayTuneCallback", "WandbCallback"})

#: Callback ``type`` names considered trial-runtime-only artifacts, stripped
#: by [write_best_config][phoenx.ray_tune.write_best_config] since they are
#: meaningless (or actively wrong) outside the trial that produced them.
_TRIAL_RUNTIME_CALLBACK_TYPES = frozenset({"RayTuneCallback"})


def build_search_alg(phase: dict, points_to_evaluate: list[dict] | None = None) -> Any:
    """Build the Ray Tune search algorithm for one phase.

    ``phase['tune']['search_alg']`` is ``{"type": <name>, ...extras}``
    (default ``{"type": "random"}`` when absent). ``type: "random"`` builds
    a ``BasicVariantGenerator`` directly; every other name delegates to
    ``ray.tune.search.create_searcher``, which reaches all of Ray's
    registered searchers (``ax``, ``bayesopt``, ``bohb``, ``hebo``,
    ``hyperopt``, ``nevergrad``, ``optuna``, ``zoopt``, plus
    ``variant_generator``) and raises Ray's own error when a
    searcher-specific required extra (e.g. Ax's ``space``) is missing.

    Args:
        phase: One normalized phase; reads ``phase['tune']['search_alg']``.
        points_to_evaluate: Optional list of concrete sampled-key dicts to
            seed the searcher with (see [run_sweep][phoenx.ray_tune.run_sweep]'s
            promotion/``seed_next``). Passed through only for search-alg
            types verified to accept it at construction (``random``,
            ``optuna``, ``hyperopt``); for any other type, a warning is
            logged and seeding is skipped for this phase rather than
            raising. Every point is expected to already be construction-safe
            for this ``phase``'s searcher type — for anything but
            ``"random"``, that means each point covers the phase's entire
            search space, since ``OptunaSearch``/``HyperOptSearch`` raise
            ``ValueError`` on a partial point rather than accepting it; see
            ``_filter_seed_points``, the caller responsible for that
            filtering.

    Returns:
        A ``ray.tune.search.Searcher`` (or ``SearchAlgorithm``) instance
        ready to hand to ``TuneConfig(search_alg=...)``.

    Raises:
        ValueError: If ``phase['tune']['search_alg']`` is present but not a
            mapping, or (propagated from Ray) a searcher-specific required
            extra is missing.

    Example:
        >>> from phoenx.ray_tune import build_search_alg
        >>> alg = build_search_alg({"tune": {"search_alg": {"type": "random"}}})
        >>> type(alg).__name__
        'BasicVariantGenerator'
    """
    tune_cfg = phase.get("tune") or {}
    search_cfg = tune_cfg.get("search_alg") or {"type": "random"}
    if not isinstance(search_cfg, dict):
        raise ValueError(f"phase['tune']['search_alg'] must be a mapping, got {search_cfg!r}")

    extras = dict(search_cfg)
    alg_type = str(extras.pop("type", "random")).lower()

    if points_to_evaluate:
        if alg_type in _POINTS_TO_EVALUATE_SUPPORTED:
            extras["points_to_evaluate"] = points_to_evaluate
        else:
            logger.warning(
                "build_search_alg: search_alg type '%s' is not verified to accept "
                "points_to_evaluate; skipping seeding of %d point(s) for this phase.",
                alg_type, len(points_to_evaluate),
            )

    if alg_type == "random":
        return BasicVariantGenerator(**extras)
    return create_searcher(alg_type, **extras)


def build_scheduler(phase: dict) -> Any:
    """Build the Ray Tune trial scheduler for one phase.

    ``phase['tune']['scheduler']`` is ``{"type": <alias>, ...extras}``,
    delegating to ``ray.tune.schedulers.create_scheduler``. ``asha`` is
    accepted as a friendly alias for Ray's own ``async_hyperband``
    (``AsyncHyperBandScheduler``, i.e. ASHA), since Ray does not register it
    under that name itself.

    Args:
        phase: One normalized phase; reads ``phase['tune']['scheduler']``.
            Absent (or ``None``) means no scheduler (Ray's default FIFO
            ordering).

    Returns:
        A ``ray.tune.schedulers.TrialScheduler`` instance, or ``None`` when
        the phase declares no scheduler.

    Raises:
        ValueError: If ``phase['tune']['scheduler']`` is present but not a
            mapping, is missing the required ``type`` key, or ``type`` is
            ``"pbt"``/``"population_based_training"`` — Population Based
            Training needs checkpoint plumbing (``Trainer.load(config=...)``,
            periodic saves via ``RayTuneCallback.bind``, mutated-LR
            re-application onto optimizer ``param_groups``) that lands in a
            later delivery pass, so it is deliberately rejected here rather
            than silently mis-training.

    Example:
        >>> from phoenx.ray_tune import build_scheduler
        >>> build_scheduler({"tune": {}})  # no scheduler declared
    """
    tune_cfg = phase.get("tune") or {}
    sched_cfg = tune_cfg.get("scheduler")
    if sched_cfg is None:
        return None
    if not isinstance(sched_cfg, dict):
        raise ValueError(f"phase['tune']['scheduler'] must be a mapping, got {sched_cfg!r}")

    extras = dict(sched_cfg)
    sched_type = extras.pop("type", None)
    if not sched_type:
        raise ValueError("phase['tune']['scheduler'] is missing required key 'type'")

    normalized = str(sched_type).lower()
    if normalized in _PBT_SCHEDULER_TYPES:
        raise ValueError(
            f"scheduler.type '{sched_type}' (Population Based Training) is not supported yet: "
            "it needs checkpoint plumbing (Trainer.load(config=...), periodic saves via "
            "RayTuneCallback.bind, mutated-LR re-application onto optimizer param_groups) that "
            "lands in a later delivery pass. Use 'asha' (async_hyperband) or another scheduler "
            "for now."
        )
    normalized = _SCHEDULER_ALIASES.get(normalized, normalized)
    return create_scheduler(normalized, **extras)


class _MetricThresholdStopper(Stopper):
    """Per-trial stopper for a plain ``{metric: threshold}`` mapping.

    Reproduces Ray's own dict-based ``RunConfig(stop=...)`` semantics
    (``ray.tune.experiment.trial.Trial.should_stop``): a trial stops once
    *any* listed metric's latest reported value is ``>=`` its threshold.
    Only needed to fold a plain metric mapping into a ``CombinedStopper``
    alongside named stopper classes; a lone plain mapping is returned as-is
    by [build_stopper][phoenx.ray_tune.build_stopper] instead, since
    ``RunConfig(stop=dict)`` already implements this directly.
    """

    def __init__(self, thresholds: dict):
        """Store the metric -> threshold mapping to check on every report.

        Args:
            thresholds: Mapping of reported metric key to the value at
                which (``>=``) a trial should stop.
        """
        self._thresholds = thresholds

    def __call__(self, trial_id: str, result: dict) -> bool:
        """Return whether any tracked metric's current value meets its threshold.

        Args:
            trial_id: Unused; accepted for ``Stopper`` interface compatibility.
            result: The trial's latest reported metrics dict.

        Returns:
            ``True`` if any ``self._thresholds`` metric present in ``result``
            is ``>=`` its threshold.
        """
        return any(
            metric in result and result[metric] >= threshold
            for metric, threshold in self._thresholds.items()
        )

    def stop_all(self) -> bool:
        """Return ``False`` always; this stopper is per-trial only.

        Returns:
            ``False``, since a plain metric-threshold mapping never stops
            the whole experiment.
        """
        return False


def _build_one_stopper(spec: dict) -> Any:
    """Build one stopper (or return a plain metric dict) from ``spec``.

    Args:
        spec: A plain ``{metric: threshold}`` mapping, or a
            ``{"type": <StopperName>, ...kwargs}`` mapping.

    Returns:
        The plain ``spec`` dict unchanged (no ``"type"`` key), or a
        constructed stopper instance.

    Raises:
        ValueError: If ``spec["type"]`` does not name a supported stopper
            class.
    """
    if "type" not in spec:
        return _MetricThresholdStopper(spec)
    extras = dict(spec)
    name = extras.pop("type")
    stopper_class = _STOPPER_CLASSES.get(name)
    if stopper_class is None:
        raise ValueError(f"Unknown stopper type '{name}'; valid: {sorted(_STOPPER_CLASSES)}")
    return stopper_class(**extras)


def build_stopper(phase: dict) -> Any:
    """Build the Ray Tune stop criterion for one phase.

    ``phase['tune']['stop']`` accepts:

    - A plain metric mapping, e.g. ``{"avg_reward": 250}``, returned as-is
      since ``RunConfig(stop=...)`` already implements "stop this trial
      once any listed metric's value is ``>=`` its threshold" for a plain
      dict.
    - A single ``{"type": <StopperName>, ...kwargs}`` mapping, built into
      the named Ray stopper class.
    - A list of either, combined via ``CombinedStopper`` (OR semantics: any
      member stopping a trial stops it); a plain metric mapping inside a
      list is wrapped in a small private ``Stopper`` that reproduces the
      same ``>=`` semantics, since ``CombinedStopper`` requires ``Stopper``
      instances, not plain dicts.

    Args:
        phase: One normalized phase; reads ``phase['tune']['stop']``.
            Absent means no stop criterion beyond the schedule's own
            ``stop_units``.

    Returns:
        ``None`` when absent, a plain ``dict`` (a lone plain metric
        mapping), or a ``ray.tune.stopper.Stopper`` instance — all valid
        for ``RunConfig(stop=...)``.

    Raises:
        ValueError: If ``stop`` is not a mapping or list, a list is empty,
            a list element is not a mapping, or a ``{"type": ...}`` name is
            not one of ``TrialPlateauStopper`` / ``ExperimentPlateauStopper``
            / ``MaximumIterationStopper`` / ``TimeoutStopper`` /
            ``CombinedStopper``.

    Example:
        >>> from phoenx.ray_tune import build_stopper
        >>> build_stopper({"tune": {"stop": {"avg_reward": 250}}})
        {'avg_reward': 250}
    """
    tune_cfg = phase.get("tune") or {}
    stop_cfg = tune_cfg.get("stop")
    if stop_cfg is None:
        return None

    if isinstance(stop_cfg, dict):
        if "type" not in stop_cfg:
            return stop_cfg
        return _build_one_stopper(stop_cfg)

    if isinstance(stop_cfg, list):
        if not stop_cfg:
            raise ValueError("phase['tune']['stop'] list must not be empty")
        stoppers = []
        for i, item in enumerate(stop_cfg):
            if not isinstance(item, dict):
                raise ValueError(f"phase['tune']['stop'][{i}] must be a mapping, got {item!r}")
            stoppers.append(_build_one_stopper(item))
        return CombinedStopper(*stoppers)

    raise ValueError(f"phase['tune']['stop'] must be a mapping or list, got {stop_cfg!r}")


def sample_search_space(space: dict, seed: int | None = None) -> dict:
    """Draw one concrete sample from a built search space, no Ray cluster needed.

    Used by [validate_only][phoenx.ray_tune.validate_only] to sanity-check a
    phase's resolution without going through ``tune.Tuner``.
    ``ray.tune.search.sample.Domain`` values (``tune.uniform`` /
    ``tune.choice`` / ...) are sampled via ``.sample()``; a
    ``{"grid_search": [...]}`` dict takes its first value (Ray's own grid
    walk would eventually visit every value — one deterministic point is
    all a smoke test needs); anything else (a literal constant, e.g. from a
    ``fixed`` spec or an unswept optimizer field) passes through unchanged.

    Args:
        space: Flat search space as returned by
            [build_search_space][phoenx.ray_tune.build_search_space].
        seed: Optional seed for reproducible sampling.

    Returns:
        Flat ``{key: concrete_value}`` mapping with the same keys as
        ``space``.

    Example:
        >>> from ray import tune
        >>> from phoenx.ray_tune import sample_search_space
        >>> space = {"agent.config.discount": tune.uniform(0.9, 0.999), "fixed_key": 42}
        >>> sampled = sample_search_space(space, seed=0)
        >>> sorted(sampled)
        ['agent.config.discount', 'fixed_key']
        >>> sampled["fixed_key"]
        42
    """
    random_state = np.random.RandomState(seed) if seed is not None else None
    sampled: dict[str, Any] = {}
    for key, value in space.items():
        if isinstance(value, dict) and "grid_search" in value:
            sampled[key] = value["grid_search"][0]
        elif isinstance(value, Domain):
            sampled[key] = value.sample(random_state=random_state) if random_state is not None else value.sample()
        else:
            sampled[key] = value
    return sampled


def _to_plain(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python types.

    Some searchers (and, defensively, numpy-backed samplers in general) can
    hand back ``numpy`` scalar types (e.g. ``numpy.int64``) that ``yaml``'s
    safe dumper and ``json`` do not know how to serialize. Applied before
    every write in this module (``best_config.yml``, ``phase_summary.json``)
    so a "best config" is guaranteed to round-trip.

    Args:
        obj: Any (possibly nested) value — dict, list, tuple, numpy scalar,
            or plain Python value.

    Returns:
        An equivalent structure with every ``numpy.generic`` scalar replaced
        by its native Python ``.item()`` value; everything else is returned
        unchanged (dicts/lists are rebuilt, not mutated in place).
    """
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _ensure_ray_initialized(sweep: dict) -> None:
    """Call ``ray.init`` with the sweep's optional ``ray_init:`` section, once.

    Only calls ``ray.init`` when Ray is not already initialized (e.g. inside
    a notebook, or a caller that manages its own Ray cluster), so running
    several phases (or several sweeps) in one process never double-inits.

    Args:
        sweep: Parsed sweep config; reads the optional top-level
            ``ray_init`` mapping (e.g. ``{"runtime_env": {"env_vars": {...}}}``)
            and forwards it verbatim as ``ray.init(**sweep['ray_init'])``, so
            e.g. ``ISAACLAB_PATH`` / ``PYTHONPATH`` reach Ray workers.

    Raises:
        ValueError: If ``sweep['ray_init']`` is present but not a mapping.
    """
    if ray.is_initialized():
        return
    ray_init_kwargs = sweep.get("ray_init") or {}
    if not isinstance(ray_init_kwargs, dict):
        raise ValueError(f"sweep['ray_init'] must be a mapping, got {ray_init_kwargs!r}")
    ray.init(**ray_init_kwargs)


def _inject_trial_callbacks(config: dict, sweep: dict, phase: dict, sampled: dict) -> None:
    """Replace this trial's ``callbacks:`` list with exactly the right set.

    Rules (see [rl_trainable][phoenx.ray_tune.rl_trainable]):

    - Always inject exactly one ``RayTuneCallback`` built from
      ``phase['report']`` (``{every, unit}``; defaults ``every=50000``,
      ``unit="timestep"`` when absent), replacing any the base config
      already declares.
    - If ``phase['wandb']`` is present, ensure exactly one ``WandbCallback``
      with ``project_name`` from ``phase['wandb']['project']``, an explicit
      ``run_name`` identifying the trial (sweep name, phase name, Ray trial
      id), an explicit ``group`` shared by every trial of the phase, tags
      including the sweep and phase name, and ``sweep_params=sampled`` (so
      swept values land under ``sweep/*`` in the W&B config), passed
      through [``_to_plain``][phoenx.ray_tune._to_plain] first so a numpy
      scalar from a searcher cannot break a later plain ``json.dump`` of
      the trainer config (e.g. ``Trainer.save()``).
    - If ``phase['wandb']`` is absent, strip any ``WandbCallback`` the base
      config declares, so trials do not spam W&B with default naming.

    Args:
        config: This trial's fully-resolved training config, mutated in
            place (``config['callbacks']`` is replaced).
        sweep: The parsed sweep config; reads the private ``_sweep_name``
            key [run_phase][phoenx.ray_tune.run_phase] stashes on it for
            this call (falls back to ``"sweep"`` when absent).
        phase: The current normalized phase (reads ``phase['report']`` /
            ``phase['wandb']`` and ``phase['name']``).
        sampled: This trial's sampled dict, passed through (after
            [``_to_plain``][phoenx.ray_tune._to_plain] sanitization) as
            ``WandbCallback(sweep_params=...)`` when W&B is enabled.

    Raises:
        ValueError: If ``phase['wandb']`` is present but missing the
            required ``project`` key.
    """
    callbacks = [
        cb for cb in (config.get("callbacks") or [])
        if not (isinstance(cb, dict) and cb.get("type") in _TRIAL_MANAGED_CALLBACK_TYPES)
    ]

    report_cfg = phase.get("report") or {}
    callbacks.append({
        "type": "RayTuneCallback",
        "config": {
            "every": report_cfg.get("every", 50000),
            "unit": report_cfg.get("unit", "timestep"),
        },
    })

    wandb_cfg = phase.get("wandb")
    if wandb_cfg:
        project = wandb_cfg.get("project")
        if not project:
            raise ValueError(
                f"phase['wandb'] for phase '{phase.get('name')}' is missing required key 'project'"
            )
        sweep_name = sweep.get("_sweep_name", "sweep")
        phase_name = phase.get("name", "phase")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trial_id = tune.get_context().get_trial_id() or "local"
        callbacks.append({
            "type": "WandbCallback",
            "config": {
                "project_name": project,
                "run_name": f"{sweep_name}-{phase_name}-{trial_id}",
                "group": f"{sweep_name}-{phase_name}",
                "tags": [sweep_name, phase_name],
                "sweep_params": _to_plain(sampled),
            },
        })

    config["callbacks"] = callbacks


def _log_trial_failure(phase: dict, sampled: dict, resolved: dict | None) -> None:
    """Log this trial's swept dotted-path values before an exception propagates.

    A bare Ray traceback loses which trial config failed (Ray reports the
    exception, not the sampled/resolved values that produced it), so this
    is called from [rl_trainable][phoenx.ray_tune.rl_trainable]'s except
    block, before re-raising.

    Args:
        phase: The current normalized phase (for its ``search_space`` keys).
        sampled: This trial's raw sampled dict.
        resolved: This trial's fully-resolved config, or ``None`` when
            resolution itself failed before producing one.
    """
    lines = []
    for path in (phase.get("search_space") or {}):
        value = None
        found = False
        if resolved is not None:
            try:
                value = get_by_path(resolved, path)
                found = True
            except (KeyError, ValueError):
                pass
        if not found:
            value = sampled.get(path)
        lines.append(f"{path}={value!r}")
    for key in sorted(sampled):
        if key.startswith("arch.") or key.startswith("opt."):
            lines.append(f"{key}={sampled[key]!r}")
    logger.error(
        "Ray Tune trial failed. Swept values -> %s",
        "; ".join(lines) if lines else "(no swept values)",
    )


def rl_trainable(sampled: dict, *, sweep: dict, phase: dict, base_config: dict) -> None:
    """Per-trial Ray Tune entry point: resolve, build, and train one trial.

    Invoked by Ray with the sampled search-space dict as ``sampled``, via
    ``tune.with_parameters(rl_trainable, sweep=..., phase=phase,
    base_config=base_config)`` in [run_phase][phoenx.ray_tune.run_phase] —
    so ``sweep``/``phase``/``base_config`` are constant across every trial
    of the phase, and ``sampled`` is this trial's draw from the phase's
    search space.

    Resolves the trial's full training config via
    [resolve_trial_config][phoenx.ray_tune.resolve_trial_config] (which
    itself deep-copies ``base_config``), sets ``save_dir`` from
    ``tune.get_context().get_trial_dir()`` (never the process's current
    working directory, whose Ray-managed chdir behavior has shifted across
    Ray 2.x releases), injects this trial's callbacks (Ray Tune reporting,
    optional W&B — see ``_inject_trial_callbacks``; this is deliberately
    *not* done inside ``resolve_trial_config``, so promotion and
    [write_best_config][phoenx.ray_tune.write_best_config] see a clean,
    trial-agnostic config), builds a ``Trainer``, and calls
    ``trainer.train()``.

    Args:
        sampled: This trial's flat ``{search_space_key: concrete_value}``
            dict, supplied by Ray from the phase's search space.
        sweep: The parsed sweep config (for ``base_config``/``overrides``/
            ``blocks``; forwarded to
            [resolve_trial_config][phoenx.ray_tune.resolve_trial_config]).
        phase: The current normalized phase.
        base_config: The training config this trial resolves from — either
            the sweep's own ``base_config`` (first phase) or a previous
            phase's promoted winner (later phases).

    Raises:
        Exception: Whatever [resolve_trial_config][phoenx.ray_tune.resolve_trial_config],
            [build_trainer_from_config][phoenx.builder.build_trainer_from_config],
            or ``trainer.train()`` raise, after logging the trial's swept
            values (see ``_log_trial_failure``) so a bare Ray traceback does
            not lose which trial config failed.
    """
    resolved = None
    try:
        resolved = resolve_trial_config(sweep, phase, sampled, copy.deepcopy(base_config))

        trial_dir = tune.get_context().get_trial_dir()
        if trial_dir:
            resolved["save_dir"] = trial_dir

        _inject_trial_callbacks(resolved, sweep, phase, sampled)

        trainer = build_trainer_from_config(resolved)
        trainer.train()
    except Exception:
        _log_trial_failure(phase, sampled, resolved)
        raise


def _trial_dirname_creator(trial: Any) -> str:
    """Short per-trial directory name, mandatory on Windows.

    Ray's default trial dirname embeds every sampled param; with dotted
    config paths and architecture search-space keys, that routinely exceeds
    Windows' ``MAX_PATH`` mid-sweep. Uses just the trial's own zero-padded
    index suffix (e.g. trial id ``"3cf3e_00007"`` -> ``"trial_00007"``).

    Args:
        trial: The ``ray.tune.experiment.Trial`` Ray is naming a directory
            for.

    Returns:
        A short, unique directory name for this trial.
    """
    suffix = trial.trial_id.rsplit("_", 1)[-1]
    return f"trial_{suffix}"


def _top_k_sampled(result_grid: Any, metric: str, mode: str, k: int) -> list[dict]:
    """Return the top-``k`` non-errored trials' raw sampled dicts, best first.

    Args:
        result_grid: The ``ray.tune.ResultGrid`` returned by ``tuner.fit()``.
        metric: Metric key to rank trials on.
        mode: ``"max"`` or ``"min"``.
        k: Number of trials to return (clamped to ``>= 0``).

    Returns:
        Up to ``k`` sampled dicts (``result.metrics["config"]``), sorted
        best-first by ``metric``/``mode``. Trials that errored, or never
        reported a finite numeric value for ``metric``, are excluded.
    """
    scored: list[tuple[float, dict]] = []
    for i in range(len(result_grid)):
        result = result_grid[i]
        if result.error is not None:
            continue
        value = result.metrics.get(metric) if result.metrics else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        scored.append((float(value), result.metrics.get("config") or {}))
    scored.sort(key=lambda item: item[0], reverse=(mode == "max"))
    return [cfg for _, cfg in scored[: max(k, 0)]]


def run_phase(
    sweep: dict,
    phase: dict,
    base_config: dict,
    *,
    storage_path: str | Path,
    sweep_name: str,
    points_to_evaluate: list[dict] | None = None,
    num_samples: int | None = None,
    max_concurrent_trials: int | None = None,
    resume: bool = False,
    **kwargs: Any,
) -> dict:
    """Run one phase's ``tune.Tuner`` and write its promotion artifacts.

    Builds the phase's flat search space (dotted ``search_space`` paths,
    synthetic ``arch.*`` architecture keys, and ``opt.*`` per-module
    optimizer keys) via [build_search_space][phoenx.ray_tune.build_search_space],
    builds the searcher / scheduler / stop criterion via
    [build_search_alg][phoenx.ray_tune.build_search_alg] /
    [build_scheduler][phoenx.ray_tune.build_scheduler] /
    [build_stopper][phoenx.ray_tune.build_stopper], and runs exactly one
    ``tune.Tuner`` under ``<storage_path>/<sweep_name>/<phase['name']>/``.
    ``storage_path`` is resolved to an absolute path first, and every trial
    gets a short ``trial_XXXXX`` directory name (mandatory on Windows; see
    ``_trial_dirname_creator``).

    Args:
        sweep: Parsed sweep config (forwarded to the trainable, and used
            here for its ``blocks`` library and optional ``ray_init``
            section).
        phase: The normalized phase to run.
        base_config: The training config this phase's trials resolve
            from — either the sweep's own ``base_config`` (first phase) or
            a previous phase's promoted winner.
        storage_path: Root directory for Ray Tune experiment storage;
            resolved to an absolute path.
        sweep_name: Sweep identifier; this phase's artifacts land under
            ``<storage_path>/<sweep_name>/<phase['name']>/``.
        points_to_evaluate: Optional seed points for this phase's searcher
            (see [run_sweep][phoenx.ray_tune.run_sweep]'s ``seed_next``
            promotion), forwarded to
            [build_search_alg][phoenx.ray_tune.build_search_alg].
        num_samples: Override for ``phase['tune']['num_samples']`` (default
            ``1`` when neither is given).
        max_concurrent_trials: Override for ``phase['max_concurrent_trials']``.
        resume: When ``True`` and a matching experiment already exists on
            disk at this phase's storage directory, resume it via
            ``Tuner.restore`` instead of starting a fresh ``Tuner``. A no-op
            (starts fresh) when no matching experiment exists.
        **kwargs: Accepted and ignored, so [run_sweep][phoenx.ray_tune.run_sweep]
            can forward one shared overrides dict to every phase without
            each phase needing to declare every possible override.

    Returns:
        Dict with keys:

        - ``"phase"``: this phase's name.
        - ``"metric"`` / ``"mode"``: the metric/mode used to rank trials.
        - ``"num_trials"`` / ``"num_errors"``: trial counts from the
            ``ResultGrid``.
        - ``"best_metric_value"``: the winning trial's final value for
            ``metric``.
        - ``"best_sampled"``: the winning trial's raw sampled dict (Ray's
            reported ``config``).
        - ``"best_config"``: the winning trial's fully-resolved,
            trial-agnostic training config (see
            [resolve_trial_config][phoenx.ray_tune.resolve_trial_config]) —
            the same dict written to this phase's ``best_config.yml``.
        - ``"top_k_sampled"``: the top-``k`` trials' raw sampled dicts, best
            first (``k = max(phase['promote']['seed_next'], 1)`` when set,
            else ``1``), for seeding the next phase.
        - ``"phase_dir"``: absolute path (as ``str``) to this phase's
            artifact directory (holds ``best_config.yml`` and
            ``phase_summary.json``).

    Raises:
        ValueError: Propagated from [build_search_alg][phoenx.ray_tune.build_search_alg] /
            [build_scheduler][phoenx.ray_tune.build_scheduler] /
            [build_stopper][phoenx.ray_tune.build_stopper] (invalid or
            unsupported ``tune`` config, including ``scheduler.type: pbt``).
        RuntimeError: If every trial in the phase errored, or no trial
            reported the configured ``metric``.
    """
    _ensure_ray_initialized(sweep)

    phase_name = phase["name"]
    blocks = parse_block_library(sweep.get("blocks"))
    space = build_search_space(phase, blocks)

    tune_cfg = phase.get("tune") or {}
    metric = phase.get("metric", "avg_reward")
    mode = phase.get("mode", "max")
    resolved_num_samples = num_samples if num_samples is not None else tune_cfg.get("num_samples", 1)
    resolved_max_concurrent = (
        max_concurrent_trials if max_concurrent_trials is not None else phase.get("max_concurrent_trials")
    )
    resources = phase.get("resources") or {"cpu": 1}
    reuse_actors = tune_cfg.get("reuse_actors", False)
    time_budget_s = tune_cfg.get("time_budget_s")

    search_alg = build_search_alg(phase, points_to_evaluate)
    scheduler = build_scheduler(phase)
    stopper = build_stopper(phase)

    storage_path_abs = Path(storage_path).resolve()
    phase_dir = storage_path_abs / sweep_name / phase_name

    trial_sweep = dict(sweep)
    trial_sweep["_sweep_name"] = sweep_name
    trainable = tune.with_parameters(rl_trainable, sweep=trial_sweep, phase=phase, base_config=base_config)
    trainable = tune.with_resources(trainable, resources)

    tune_config = tune.TuneConfig(
        metric=metric,
        mode=mode,
        num_samples=resolved_num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        max_concurrent_trials=resolved_max_concurrent,
        reuse_actors=reuse_actors,
        time_budget_s=time_budget_s,
        trial_dirname_creator=_trial_dirname_creator,
    )
    run_config = tune.RunConfig(
        name=phase_name,
        storage_path=str(storage_path_abs / sweep_name),
        stop=stopper,
    )

    if resume and tune.Tuner.can_restore(phase_dir):
        logger.info("run_phase: resuming existing experiment at '%s'", phase_dir)
        tuner = tune.Tuner.restore(str(phase_dir), trainable, resume_errored=True)
    else:
        tuner = tune.Tuner(trainable, param_space=space, tune_config=tune_config, run_config=run_config)

    result_grid = tuner.fit()

    if result_grid.num_errors:
        logger.warning(
            "run_phase '%s': %d of %d trial(s) errored.",
            phase_name, result_grid.num_errors, len(result_grid),
        )
    if len(result_grid) == 0 or result_grid.num_errors == len(result_grid):
        first_error = result_grid.errors[0] if result_grid.errors else "(no trials ran)"
        raise RuntimeError(
            f"run_phase '{phase_name}': all {len(result_grid)} trial(s) errored; "
            f"first error: {first_error}"
        )

    try:
        best_result = result_grid.get_best_result(metric=metric, mode=mode)
    except Exception as exc:
        raise RuntimeError(f"run_phase '{phase_name}': could not determine a best result: {exc}") from exc

    best_sampled = _to_plain(best_result.metrics.get("config") or {})
    best_metric_value = best_result.metrics.get(metric)
    best_config = _to_plain(resolve_trial_config(sweep, phase, best_sampled, base_config))

    promote_cfg = phase.get("promote") or {}
    top_k = max(int(promote_cfg.get("seed_next", 1)), 1)
    top_k_sampled = _to_plain(_top_k_sampled(result_grid, metric, mode, top_k))

    phase_dir.mkdir(parents=True, exist_ok=True)
    write_best_config(best_config, phase_dir / "best_config.yml")
    summary = {
        "phase": phase_name,
        "metric": metric,
        "mode": mode,
        "num_trials": len(result_grid),
        "num_errors": result_grid.num_errors,
        "best_metric_value": best_metric_value,
        "best_sampled": best_sampled,
        "top_k_sampled": top_k_sampled,
    }
    with open(phase_dir / "phase_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, default=str)

    return {
        "phase": phase_name,
        "metric": metric,
        "mode": mode,
        "num_trials": len(result_grid),
        "num_errors": result_grid.num_errors,
        "best_metric_value": best_metric_value,
        "best_sampled": best_sampled,
        "best_config": best_config,
        "top_k_sampled": top_k_sampled,
        "phase_dir": str(phase_dir),
    }


def _search_alg_type(phase: dict) -> str:
    """Return the lowercased ``tune.search_alg.type`` a phase's searcher will use.

    Mirrors [build_search_alg][phoenx.ray_tune.build_search_alg]'s own
    default (``"random"`` when ``phase['tune']['search_alg']`` is absent),
    so any other code that needs to know a phase's searcher type ahead of
    actually building it — e.g. [_filter_seed_points][phoenx.ray_tune._filter_seed_points]'s
    per-searcher ``points_to_evaluate`` coverage check — reads it the same
    way instead of re-deriving its own notion of the default.

    Args:
        phase: One normalized phase; reads ``phase['tune']['search_alg']``.

    Returns:
        The lowercased ``type`` string, defaulting to ``"random"`` when
        ``phase['tune']['search_alg']`` is absent or not a mapping (an
        actually-invalid mapping is reported by
        [build_search_alg][phoenx.ray_tune.build_search_alg] itself when
        that phase runs).

    Example:
        >>> from phoenx.ray_tune import _search_alg_type
        >>> _search_alg_type({"tune": {"search_alg": {"type": "Optuna"}}})
        'optuna'
    """
    tune_cfg = phase.get("tune") or {}
    search_cfg = tune_cfg.get("search_alg") or {"type": "random"}
    if not isinstance(search_cfg, dict):
        return "random"
    return str(search_cfg.get("type", "random")).lower()


def _filter_seed_points(
    top_k_sampled: list[dict],
    phase: dict,
    next_phase: dict,
    blocks: dict,
    next_alg_type: str | None = None,
) -> list[dict] | None:
    """Filter promoted top-k sampled configs to keys the next phase re-searches.

    ``search_space`` and ``opt.*`` keys always carry over when the next
    phase's search space happens to include the same key. ``arch.*`` keys
    only carry over when ``phase``'s and ``next_phase``'s ``architecture``
    sections are identical (slot/block indices from a different
    architecture spec would not mean the same thing); otherwise a log line
    explains why they were skipped. A per-trial seed dict that ends up
    empty after filtering is dropped, not passed, and logged.

    ``OptunaSearch`` and ``HyperOptSearch`` (unlike ``BasicVariantGenerator``,
    i.e. ``type: "random"``) require every ``points_to_evaluate`` entry to
    cover the *entire* next phase's search space — a partial point raises
    ``ValueError`` from the searcher constructor when the next phase
    starts. So for any ``next_alg_type`` other than ``"random"``, a
    filtered point that does not cover every key of the next phase's
    search space is dropped (and logged) rather than passed through
    partially. When every point ends up dropped, ``None`` is returned (and
    logged) rather than an empty list, so a user who set ``seed_next`` can
    tell seeding did not happen instead of assuming it did.

    Args:
        top_k_sampled: The finishing phase's top-k sampled dicts (see
            [run_phase][phoenx.ray_tune.run_phase]'s return value).
        phase: The phase that just finished.
        next_phase: The phase about to run; its search space determines
            which keys are worth seeding.
        blocks: Parsed block library, needed to build the next phase's
            search space.
        next_alg_type: The next phase's lowercased ``tune.search_alg.type``
            (see [_search_alg_type][phoenx.ray_tune._search_alg_type]).
            Defaults to ``None``, which derives it from ``next_phase``
            itself — callers that already know it (e.g.
            [run_sweep][phoenx.ray_tune.run_sweep]) should still pass it
            explicitly rather than relying on this fallback.

    Returns:
        Filtered list of non-empty, (for non-``"random"`` searchers)
        fully-covering seed dicts, or ``None`` when every point ends up
        empty or insufficiently covering.
    """
    if next_alg_type is None:
        next_alg_type = _search_alg_type(next_phase)

    next_space_keys = set(build_search_space(next_phase, blocks).keys())
    arch_same = (phase.get("architecture") or {}) == (next_phase.get("architecture") or {})

    any_arch_key = any(key.startswith("arch.") for point in top_k_sampled for key in point)
    if any_arch_key and not arch_same:
        logger.info(
            "run_sweep: skipping arch.* seeding from phase '%s' into phase '%s': their "
            "'architecture' sections differ, so slot/block indices would not mean the same "
            "thing in the next phase's search space.",
            phase.get("name"), next_phase.get("name"),
        )

    requires_full_coverage = next_alg_type != "random"
    filtered: list[dict] = []
    dropped_partial = 0
    for point in top_k_sampled:
        seed = {
            key: value
            for key, value in point.items()
            if key in next_space_keys and (arch_same or not key.startswith("arch."))
        }
        if not seed:
            continue
        if requires_full_coverage and set(seed) != next_space_keys:
            dropped_partial += 1
            continue
        filtered.append(seed)

    if dropped_partial:
        logger.info(
            "run_sweep: dropped %d partial seed point(s) bound for phase '%s': search_alg type "
            "'%s' requires every points_to_evaluate entry to cover the full search space (keys: "
            "%s), and OptunaSearch/HyperOptSearch raise ValueError at phase start on a partial "
            "point instead of accepting it.",
            dropped_partial, next_phase.get("name"), next_alg_type, sorted(next_space_keys),
        )

    if not filtered:
        logger.info(
            "run_sweep: no seed points carried over from phase '%s' into phase '%s': every "
            "candidate point filtered down to an empty dict against the next phase's search "
            "space, or (search_alg type '%s') did not cover it fully. seed_next had no effect "
            "for this phase transition.",
            phase.get("name"), next_phase.get("name"), next_alg_type,
        )
        return None
    return filtered


def run_sweep(
    sweep: dict,
    *,
    storage_path: str | Path | None = None,
    sweep_name: str | None = None,
    from_phase: str | None = None,
    resume: bool = False,
    **kwargs: Any,
) -> dict:
    """Run every phase of a sweep in series, promoting winners between phases.

    Validates the whole sweep before any compute: structural validation via
    [validate_sweep_config][phoenx.ray_tune.validate_sweep_config], a
    resolution smoke test (one sampled config per phase, fully resolved via
    [resolve_trial_config][phoenx.ray_tune.resolve_trial_config] against the
    sweep's own ``base_config``), and — for every phase — constructing (then
    discarding) its searcher, scheduler, and stop criterion via
    [build_search_alg][phoenx.ray_tune.build_search_alg] /
    [build_scheduler][phoenx.ray_tune.build_scheduler] /
    [build_stopper][phoenx.ray_tune.build_stopper], so a typo'd dotted path,
    an unsupported scheduler (e.g. ``scheduler.type: pbt``), or an unknown
    stopper name fails in milliseconds instead of hours into a real run.
    Then runs each phase's ``tune.Tuner`` via [run_phase][phoenx.ray_tune.run_phase],
    in order.

    After a phase whose own ``phase['promote']['mode'] == 'best'``, that
    phase's fully-resolved winning config becomes the next phase's base
    config (anything the phase searched and the next phase does not
    therefore stays frozen at the winner's value; anything searched again
    is re-searched). ``phase['promote']['seed_next']: k`` additionally
    seeds the next phase's searcher with its top-``k`` sampled configs as
    ``points_to_evaluate``, filtered via ``_filter_seed_points`` to keys
    the next phase actually searches — and, for any next-phase searcher
    other than ``type: "random"`` (e.g. ``optuna``/``hyperopt``, which
    reject a partial point at phase start), further filtered to only
    points that cover the *entire* next phase's search space. A phase with
    no ``promote`` block at all is logged, since it means the next phase
    (if any) trains against the original, un-tuned base config.

    Writes the final phase's fully-resolved winning config to
    ``best_config.yml`` at the sweep root
    (``<storage_path>/<sweep_name>/best_config.yml``), directly runnable
    with ``phoenx-train --config``.

    Args:
        sweep: Parsed sweep config (see
            [load_sweep_config][phoenx.ray_tune.load_sweep_config]).
        storage_path: Root directory for Ray Tune experiment storage.
            Resolved to an absolute path. Defaults to ``"ray_results"``
            under the current working directory.
        sweep_name: Sweep identifier; artifacts land under
            ``<storage_path>/<sweep_name>/<phase_name>/``. Defaults to the
            sweep's ``base_config`` filename stem plus ``"_sweep"``.
        from_phase: When given, skip every phase before this name and load
            its base config from the *preceding* phase's promoted
            ``best_config.yml`` on disk (or the sweep's own ``base_config``
            when ``from_phase`` names the first phase).
        resume: When ``True``, attempt to resume the starting phase's own
            interrupted ``Tuner`` run from
            ``<storage_path>/<sweep_name>/<phase_name>/`` if a matching
            experiment already exists there (via ``Tuner.restore``); later
            phases in the same call always start fresh. A no-op if no
            matching experiment exists.
        **kwargs: Forwarded to every [run_phase][phoenx.ray_tune.run_phase]
            call (e.g. CLI overrides ``num_samples`` / ``max_concurrent_trials``).

    Returns:
        Dict with keys:

        - ``"phases"``: ``{phase_name: <run_phase's return dict>}`` for
            every phase actually run this call.
        - ``"final_config"``: the last-run phase's fully-resolved winning
            config (trial-agnostic; the same dict written to the sweep
            root's ``best_config.yml``).
        - ``"final_config_path"``: absolute path (as ``str``) to the sweep
            root's ``best_config.yml``.

    Raises:
        ValueError: From [validate_sweep_config][phoenx.ray_tune.validate_sweep_config],
            an unknown ``from_phase`` name, the pre-flight resolution smoke
            test, or the pre-flight searcher/scheduler/stopper construction
            check, naming the phase and offending key.
        FileNotFoundError: If ``from_phase`` is given and the preceding
            phase's ``best_config.yml`` is missing on disk.
    """
    validate_sweep_config(sweep)
    phases = normalize_phases(sweep)
    phase_names = [p["name"] for p in phases]
    blocks = parse_block_library(sweep.get("blocks"))

    sweep_base_config = load_config(sweep["base_config"])
    for phase in phases:
        space = build_search_space(phase, blocks)
        sampled = sample_search_space(space, seed=0)
        try:
            resolve_trial_config(sweep, phase, sampled, copy.deepcopy(sweep_base_config))
        except Exception as exc:
            raise ValueError(
                f"run_sweep: pre-flight resolution smoke test failed for phase "
                f"'{phase['name']}': {exc}"
            ) from exc
        try:
            build_search_alg(phase)
            build_scheduler(phase)
            build_stopper(phase)
        except Exception as exc:
            raise ValueError(
                f"run_sweep: pre-flight tune config check failed for phase "
                f"'{phase['name']}': {exc}"
            ) from exc

    storage_path_abs = Path(storage_path or "ray_results").resolve()
    sweep_name = sweep_name or f"{Path(str(sweep['base_config'])).stem}_sweep"

    if from_phase is not None:
        if from_phase not in phase_names:
            raise ValueError(f"from_phase='{from_phase}' is not one of this sweep's phases: {phase_names}")
        start_idx = phase_names.index(from_phase)
    else:
        start_idx = 0

    if start_idx == 0:
        base_config = copy.deepcopy(sweep_base_config)
    else:
        prev_phase_name = phase_names[start_idx - 1]
        prev_best_path = storage_path_abs / sweep_name / prev_phase_name / "best_config.yml"
        if not prev_best_path.is_file():
            raise FileNotFoundError(
                f"run_sweep: from_phase='{from_phase}' needs the preceding phase "
                f"'{prev_phase_name}''s promoted config at '{prev_best_path}', which does not "
                "exist. Run the sweep from the start, or from an earlier already-completed "
                "phase, first."
            )
        base_config = load_config(prev_best_path)

    results: dict[str, dict] = {}
    points_to_evaluate: list[dict] | None = None
    final_config = base_config

    for i in range(start_idx, len(phases)):
        phase = phases[i]
        phase_result = run_phase(
            sweep,
            phase,
            base_config,
            storage_path=storage_path_abs,
            sweep_name=sweep_name,
            points_to_evaluate=points_to_evaluate,
            resume=resume if i == start_idx else False,
            **kwargs,
        )
        results[phase["name"]] = phase_result
        final_config = phase_result["best_config"]

        promote_cfg = phase.get("promote") or {}
        if promote_cfg.get("mode") == "best":
            base_config = phase_result["best_config"]
        elif not promote_cfg:
            logger.info(
                "run_sweep: phase '%s' declares no 'promote' block; the next phase (if any) "
                "will train against the ORIGINAL base config, not this phase's winner.",
                phase["name"],
            )

        points_to_evaluate = None
        seed_next = promote_cfg.get("seed_next")
        if seed_next and i + 1 < len(phases):
            next_phase = phases[i + 1]
            points_to_evaluate = _filter_seed_points(
                phase_result["top_k_sampled"][:seed_next],
                phase,
                next_phase,
                blocks,
                _search_alg_type(next_phase),
            )

    sweep_root = storage_path_abs / sweep_name
    final_config_path = sweep_root / "best_config.yml"
    write_best_config(final_config, final_config_path)

    return {
        "phases": results,
        "final_config": final_config,
        "final_config_path": str(final_config_path),
    }


def write_best_config(config: dict, path: str | Path) -> None:
    """Write a trial-agnostic, directly-runnable training config YAML.

    Strips trial-runtime artifacts before writing: any ``RayTuneCallback``
    entry in ``config['callbacks']`` (a Ray Tune reporting seam that is
    meaningless outside a Tune trial) and a trial-specific ``save_dir``
    (Ray Tune's per-trial directory, which would not exist on a later plain
    ``phoenx-train`` run). Every value is passed through
    ``_to_plain`` first, so a numpy scalar slipping through from a
    searcher cannot break ``yaml.safe_dump``. The result is intended to
    round-trip: it must be loadable by
    [load_config][phoenx.builder.load_config] and buildable by
    [build_trainer_from_config][phoenx.builder.build_trainer_from_config].

    Args:
        config: A fully-resolved training config (e.g. a phase's winning
            trial's config, or the sweep's final promoted config).
        path: Destination file path; parent directories are created.

    Example:
        >>> from phoenx.ray_tune import write_best_config
        >>> write_best_config({"schedule": {}, "agent": {}}, "/tmp/best.yml")  # doctest: +SKIP
    """
    clean = _to_plain(copy.deepcopy(config))
    clean.pop("save_dir", None)
    if "callbacks" in clean:
        clean["callbacks"] = [
            cb for cb in clean["callbacks"]
            if not (isinstance(cb, dict) and cb.get("type") in _TRIAL_RUNTIME_CALLBACK_TYPES)
        ]
        if not clean["callbacks"]:
            del clean["callbacks"]

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(clean, file_obj, sort_keys=False)


def _diff_changed_paths(base_config: dict, resolved: dict, phase: dict) -> dict[str, tuple]:
    """Compute ``{dotted_path: (old_value, new_value)}`` for one resolved sample.

    Compares ``base_config`` against ``resolved`` at every dotted path this
    phase could have changed: ``phase['search_space']`` keys, each swept
    architecture module's ``layer_config`` path, each swept optimizer
    module's ``optimizer_params`` path, and ``schedule.learn_every`` when
    ``auto_learn_every`` applies. Used by
    [validate_only][phoenx.ray_tune.validate_only] to print a readable diff.

    Args:
        base_config: The phase's base training config (pre-resolution).
        resolved: The same trial's fully-resolved config.
        phase: The current normalized phase.

    Returns:
        Mapping of changed dotted path to ``(old_value, new_value)``; a path
        present in neither config, or unchanged, is omitted. A path missing
        from one side is represented as ``None`` on that side.
    """
    paths = list((phase.get("search_space") or {}).keys())
    for module in (phase.get("architecture") or {}):
        paths.append(f"{_MODEL_PATH_PREFIX}.{module}.layer_config")
    for module in (phase.get("optimizers") or {}):
        paths.append(f"{_MODEL_PATH_PREFIX}.{module}.optimizer_params")
    if phase.get("auto_learn_every", True):
        paths.append("schedule.learn_every")

    def _safe_get(config: dict, path: str) -> Any:
        try:
            return get_by_path(config, path)
        except (KeyError, ValueError):
            return None

    diff: dict[str, tuple] = {}
    for path in paths:
        old = _safe_get(base_config, path)
        new = _safe_get(resolved, path)
        if old != new:
            diff[path] = (old, new)
    return diff


def validate_only(sweep: dict, *, num_samples: int = 3) -> None:
    """Resolve sample trial configs per phase and print a diff, without training.

    Before printing anything, also builds (then discards) every phase's
    searcher, scheduler, and stop criterion via
    [build_search_alg][phoenx.ray_tune.build_search_alg] /
    [build_scheduler][phoenx.ray_tune.build_scheduler] /
    [build_stopper][phoenx.ray_tune.build_stopper], so an unsupported
    scheduler (e.g. ``scheduler.type: pbt``), an unknown stopper name, or a
    searcher whose optional dependency is missing is caught here too,
    matching [run_sweep][phoenx.ray_tune.run_sweep]'s own pre-flight.

    For each phase, draws ``num_samples`` concrete samples via
    [sample_search_space][phoenx.ray_tune.sample_search_space], fully
    resolves each one via [resolve_trial_config][phoenx.ray_tune.resolve_trial_config]
    (architecture assembly, per-module optimizers, ``auto_learn_every``, and
    the constraint validator all run exactly as they would for a real
    trial), and prints the changed dotted paths against the phase's base
    config. Never touches Ray's cluster (no ``ray.init``, no ``Tuner``), so
    it is safe to run against a sweep whose ``base_config`` needs hardware
    (e.g. Isaac Sim) that is not booted here.

    Args:
        sweep: Parsed sweep config (validated internally via
            [validate_sweep_config][phoenx.ray_tune.validate_sweep_config]
            before sampling).
        num_samples: Number of sample trial configs to resolve per phase.

    Raises:
        ValueError: Propagated from [validate_sweep_config][phoenx.ray_tune.validate_sweep_config],
            a phase's searcher/scheduler/stopper construction failure, or a
            phase's resolution failure, with the phase name and the
            offending key/path named in the message.
    """
    validate_sweep_config(sweep)
    phases = normalize_phases(sweep)
    blocks = parse_block_library(sweep.get("blocks"))
    base_config = load_config(sweep["base_config"])

    for phase in phases:
        name = phase["name"]
        try:
            build_search_alg(phase)
            build_scheduler(phase)
            build_stopper(phase)
        except Exception as exc:
            raise ValueError(f"Phase '{name}': invalid tune.search_alg/scheduler/stop config: {exc}") from exc

    for phase in phases:
        name = phase["name"]
        space = build_search_space(phase, blocks)
        print(f"\n=== Phase '{name}': {num_samples} sample trial(s) ===")
        for i in range(num_samples):
            sampled = sample_search_space(space, seed=i)
            try:
                resolved = resolve_trial_config(sweep, phase, sampled, copy.deepcopy(base_config))
            except Exception as exc:
                raise ValueError(f"Phase '{name}': sample {i} failed to resolve: {exc}") from exc
            diff = _diff_changed_paths(base_config, resolved, phase)
            print(f"  sample {i}: {len(diff)} changed path(s)")
            for path, (old, new) in diff.items():
                print(f"    {path}: {old!r} -> {new!r}")
