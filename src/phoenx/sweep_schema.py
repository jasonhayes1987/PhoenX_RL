"""The structure of a sweep config, without importing Ray or torch.

``phoenx.ray_tune`` imports these key sets to reject misspelled keys before a
sweep starts (a misspelled phase key such as ``serach_space`` used to be
ignored, so that phase ran with nothing to search). The config GUI's sweep
editor uses the same definitions.
"""

from __future__ import annotations

import copy

#: Keys a sweep file may have at the top level. The last four may also appear
#: at the top level instead of under ``phases`` (an implicit single phase).
SWEEP_KEYS: dict[str, str] = {
    "base_config": "Training config the sweep starts from: a bundled example name or a path.",
    "overrides": "Dotted config path -> constant, applied to every trial of every phase.",
    "defaults": "Phase settings inherited by every phase; a phase's own value wins.",
    "blocks": "Reusable layer templates for architecture search, by name.",
    "phases": "Phases run in order; each can promote its best config into the next.",
    "ray_init": "Keyword arguments forwarded to ray.init(), e.g. runtime_env.",
    "search_space": "Single-phase shorthand: dotted config path -> search spec.",
    "tune": "Single-phase shorthand: Ray Tune settings.",
    "architecture": "Single-phase shorthand: per-module depth / block search.",
    "optimizers": "Single-phase shorthand: per-module optimizer search.",
}

#: Keys a phase (or ``defaults``) may have.
PHASE_KEYS: dict[str, str] = {
    "name": "Phase name (unique); results are stored under it.",
    "search_space": "Dotted config path -> search spec, e.g. "
                    "agent.config.discount: {dist: uniform, low: 0.95, high: 0.999}.",
    "architecture": "Module (branches.policy, trunk, ...) -> {depth, blocks} to search layer stacks.",
    "optimizers": "Module -> optimizer fields to search, e.g. {lr: {dist: loguniform, ...}}.",
    "tune": "Ray Tune settings: num_samples, search_alg, scheduler, stop, ...",
    "promote": "What to carry into the next phase: {mode: best} promotes the best trial's config.",
    "metric": "Metric the search optimizes, e.g. avg_reward.",
    "mode": "max or min: whether a higher or lower metric is better.",
    "resources": "Resources per trial, e.g. {cpu: 1, gpu: 0.25}.",
    "max_concurrent_trials": "How many trials run at once.",
    "report": "How often trials report the metric, e.g. {every: 20000, unit: timestep}.",
    "auto_learn_every": "Derive schedule.learn_every from the trial's buffer / env sizes.",
    "wandb": "Log trials to Weights & Biases, e.g. {project: my-sweep}.",
}

#: Keys of a phase's ``tune`` block. ``search_alg``, ``scheduler`` and
#: ``stop`` contents are passed to Ray and are not checked here.
TUNE_KEYS: dict[str, str] = {
    "num_samples": "Number of trials to run in this phase.",
    "search_alg": "Search algorithm, e.g. {type: random} or {type: optuna}.",
    "scheduler": "Trial scheduler, e.g. {type: asha, time_attr: timestep, grace_period: ..., max_t: ...}.",
    "stop": "Stopping rules, e.g. [{type: TrialPlateauStopper, metric: avg_reward, std: 5.0}].",
    "time_budget_s": "Stop the phase after this many seconds.",
    "reuse_actors": "Reuse Ray actors between trials (faster startup).",
}

#: Search-spec distributions and their keys: (required, optional).
DISTRIBUTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "uniform": (("low", "high"), ()),
    "loguniform": (("low", "high"), ()),
    "quniform": (("low", "high", "q"), ()),
    "randint": (("lower", "upper"), ()),
    "qrandint": (("lower", "upper", "q"), ()),
    "lograndint": (("lower", "upper"), ("base",)),
    "choice": (("values",), ()),
    "grid_search": (("values",), ()),
    "randn": ((), ("mean", "sd")),
    "fixed": (("value",), ()),
}


def check_sweep_keys(sweep: dict) -> None:
    """Reject misspelled keys at the sweep, phase and ``tune`` levels.

    Keys starting with ``_`` are internal and always allowed.

    Args:
        sweep: Parsed sweep config.

    Raises:
        ValueError: Naming each unknown key with a "did you mean" suggestion.
    """
    from phoenx.config_utils import reject_unknown_keys

    def check(owner: str, mapping, valid) -> None:
        if not isinstance(mapping, dict):
            return
        try:
            reject_unknown_keys(owner, list(mapping), valid)
        except TypeError as exc:
            raise ValueError(str(exc)) from None

    check("Sweep config", sweep, SWEEP_KEYS)
    check("sweep['defaults']", sweep.get("defaults"), PHASE_KEYS)
    phase_like = [("top-level phase", sweep)] if "phases" not in sweep else []
    for i, phase in enumerate(sweep.get("phases") or []):
        if isinstance(phase, dict):
            label = f"Phase '{phase.get('name', i)}'"
            check(label, phase, PHASE_KEYS)
            phase_like.append((label, phase))
    for label, phase in phase_like + [("sweep['defaults']", sweep.get("defaults") or {})]:
        if isinstance(phase, dict):
            check(f"{label} tune", phase.get("tune"), TUNE_KEYS)


# =============================================================================
# JSON Schema for sweep files (drives the config GUI's Sweep tab)
# =============================================================================

#: Help text for search-spec fields.
SPEC_FIELD_DOCS: dict[str, str] = {
    "low": "Lowest value.",
    "high": "Highest value.",
    "q": "Round samples to a multiple of this.",
    "lower": "Lowest integer.",
    "upper": "Upper bound, EXCLUSIVE (Ray semantics): randint 1..4 draws 1, 2 or 3.",
    "base": "Log base.",
    "values": "List of options, e.g. [64, 128, 256] or [relu, tanh].",
    "value": "The constant to use.",
    "mean": "Mean of the normal distribution.",
    "sd": "Standard deviation.",
}

DIST_DOCS: dict[str, str] = {
    "uniform": "Float drawn evenly between low and high.",
    "loguniform": "Float drawn evenly in log space (good for learning rates).",
    "quniform": "Uniform float rounded to a multiple of q.",
    "randint": "Integer in [lower, upper).",
    "qrandint": "Integer in [lower, upper) rounded to a multiple of q.",
    "lograndint": "Integer drawn evenly in log space, [lower, upper).",
    "choice": "One of the listed values.",
    "grid_search": "Every listed value, each in its own trial.",
    "randn": "Normally distributed float.",
    "fixed": "A constant (not searched).",
}

SEARCH_ALGS = ("random", "optuna", "hyperopt", "bayesopt", "ax", "bohb", "hebo", "nevergrad",
               "zoopt")
SCHEDULERS = ("asha", "fifo", "hyperband", "median_stopping_rule", "hb_bohb")


def _spec_field_schema(dist: str, name: str) -> dict:
    integer = dist in ("randint", "qrandint", "lograndint") and name in ("lower", "upper", "q")
    if name == "values":
        s: dict = {"type": "array"}
    elif name == "value":
        s = {}
    else:
        s = {"type": "integer" if integer else "number"}
    s["description"] = SPEC_FIELD_DOCS[name]
    return s


def search_spec_schema() -> dict:
    """``{dist: <name>, ...}``: one searched value, its fields chosen by ``dist``."""
    from phoenx.config_schema import tagged_union

    variants = {}
    for dist, (required, optional) in DISTRIBUTIONS.items():
        props = {n: _spec_field_schema(dist, n) for n in (*required, *optional)}
        variants[dist] = {"type": "object", "properties": props, "required": list(required),
                          "additionalProperties": False, "description": DIST_DOCS[dist]}
    s = tagged_union(variants, config_key=None, tag="dist",
                     description="How the value is searched.")
    s["properties"]["dist"]["description"] = "Distribution to sample from."
    return s


def _phase_props(spec_ref: dict) -> dict:
    raw = {}  # free YAML
    tune = {
        "type": "object",
        "description": PHASE_KEYS["tune"],
        "properties": {
            "num_samples": {"type": "integer", "description": TUNE_KEYS["num_samples"]},
            "search_alg": {
                "type": "object", "description": TUNE_KEYS["search_alg"],
                "properties": {"type": {"type": "string", "examples": list(SEARCH_ALGS),
                                        "description": "Ray searcher name (random = Ray's "
                                                       "basic variant generator)."}},
                "additionalProperties": True},
            "scheduler": {
                "type": "object", "description": TUNE_KEYS["scheduler"],
                "required": ["type"],
                "properties": {"type": {"type": "string", "examples": list(SCHEDULERS),
                                        "description": "Ray scheduler name (PBT is not "
                                                       "supported)."}},
                "additionalProperties": True},
            "stop": {**raw, "description": TUNE_KEYS["stop"]},
            "time_budget_s": {"type": "number", "description": TUNE_KEYS["time_budget_s"]},
            "reuse_actors": {"type": "boolean", "default": False,
                             "description": TUNE_KEYS["reuse_actors"]},
        },
        "additionalProperties": False,
    }
    return {
        "name": {"type": "string", "description": PHASE_KEYS["name"]},
        "search_space": {"type": "object", "additionalProperties": spec_ref, "x-path-keys": True,
                         "description": PHASE_KEYS["search_space"]},
        "architecture": {
            "type": "object", "description": PHASE_KEYS["architecture"], "x-path-keys": True,
            "additionalProperties": {
                "type": "object", "required": ["depth", "blocks"],
                "properties": {
                    "depth": {**spec_ref, "description": "Number of blocks to stack, e.g. "
                              "{dist: randint, lower: 1, upper: 4}."},
                    "blocks": {"type": "array", "items": {"type": "string"},
                               "description": "Names from the sweep's blocks library to "
                                              "choose from, e.g. [dense_block]."},
                    "prefix": {"description": "Layers placed before the searched blocks."},
                    "suffix": {"description": "Layers placed after the searched blocks."},
                },
                "additionalProperties": False}},
        "optimizers": {
            "type": "object", "description": PHASE_KEYS["optimizers"], "x-path-keys": True,
            "additionalProperties": {
                "description": "Optimizer fields, e.g. {lr: {dist: loguniform, low: 1.0e-5, "
                               "high: 1.0e-3}} or {type: AdamW}."}},
        "tune": tune,
        "promote": {
            "type": "object", "description": PHASE_KEYS["promote"],
            "properties": {
                "mode": {"enum": ["best"], "description": "best: the next phase starts from "
                                                          "this phase's best trial config."},
                "seed_next": {"type": "integer", "description": "Also seed the next phase's "
                              "searcher with this many top trials (needs mode: best)."}},
            "additionalProperties": False},
        "metric": {"type": "string", "examples": ["avg_reward"], "description": PHASE_KEYS["metric"]},
        "mode": {"enum": ["max", "min"], "description": PHASE_KEYS["mode"]},
        "resources": {"description": PHASE_KEYS["resources"]},
        "max_concurrent_trials": {"type": "integer",
                                  "description": PHASE_KEYS["max_concurrent_trials"]},
        "report": {"description": PHASE_KEYS["report"]},
        "auto_learn_every": {"type": "boolean", "default": True,
                             "description": PHASE_KEYS["auto_learn_every"]},
        "wandb": {"description": PHASE_KEYS["wandb"]},
    }


def build_sweep_schema() -> dict:
    """JSON Schema (draft-07) for a sweep file, in the multi-phase form."""
    spec_ref = {"$ref": "#/definitions/search_spec"}
    phase_props = _phase_props(spec_ref)
    phase = {"type": "object", "required": ["name"], "properties": phase_props,
             "additionalProperties": False, "patternProperties": {"^_": {}},
             "description": "One phase of the sweep."}
    defaults = {"type": "object", "description": SWEEP_KEYS["defaults"],
                "properties": {k: v for k, v in phase_props.items() if k != "name"},
                "additionalProperties": False}
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "PhoenX sweep config",
        "type": "object",
        "x-keep-order": True,
        "required": ["base_config", "phases"],
        "properties": {
            "base_config": {"type": "string", "description": SWEEP_KEYS["base_config"]},
            "overrides": {"type": "object", "additionalProperties": {
                "description": "Value every trial uses at this path."},
                "x-path-keys": True, "description": SWEEP_KEYS["overrides"]},
            "defaults": defaults,
            "blocks": {"type": "object", "description": SWEEP_KEYS["blocks"],
                       "additionalProperties": {
                           "description": "A block: {layers: [...]}; layer fields may be "
                                          "search specs."}},
            "phases": {"type": "array", "items": phase, "description": SWEEP_KEYS["phases"]},
            "ray_init": {"description": SWEEP_KEYS["ray_init"]},
        },
        "additionalProperties": False,
        "patternProperties": {"^_": {}},
        "definitions": {"search_spec": search_spec_schema()},
    }


def to_phases_form(sweep: dict) -> tuple[dict, bool]:
    """Rewrite single-phase shorthand as a one-phase ``phases`` list.

    Top-level ``search_space`` / ``tune`` / ... become
    ``phases: [{name: phase_0, ...}]``, which runs identically.

    Returns:
        ``(sweep, changed)``; the input is not modified.
    """
    if "phases" in sweep:
        return sweep, False
    implicit = ("search_space", "tune", "architecture", "optimizers")
    if not any(k in sweep for k in implicit):
        return sweep, False
    out = {k: v for k, v in sweep.items() if k not in implicit}
    out["phases"] = [{"name": "phase_0", **{k: sweep[k] for k in implicit if k in sweep}}]
    return out, True


# =============================================================================
# Ray-free validation against the base config
# =============================================================================

_MISSING = object()


def _split(path: str) -> list:
    out: list = []
    for seg in str(path).split("."):
        out.append(int(seg) if seg.lstrip("-").isdigit() else seg)
    return out


def get_path(config: object, path: str) -> object:
    """Value at a dotted path (``a.b.0.c``), or ``_MISSING`` if any segment is absent."""
    node = config
    for seg in _split(path):
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        elif isinstance(node, list) and isinstance(seg, int) and -len(node) <= seg < len(node):
            node = node[seg]
        else:
            return _MISSING
    return node


def set_path(config: dict, path: str, value: object) -> None:
    """Set an existing dotted path (caller checked it exists)."""
    *parents, last = _split(path)
    node = config
    for seg in parents:
        node = node[seg]
    node[last] = value


def representative(spec: object) -> object:
    """One plausible sample from a search spec (its low end / first option)."""
    if not (isinstance(spec, dict) and "dist" in spec):
        return spec
    d = spec["dist"]
    if d in ("uniform", "loguniform", "quniform"):
        return spec.get("low")
    if d in ("randint", "qrandint", "lograndint"):
        return spec.get("lower")
    if d in ("choice", "grid_search"):
        vals = spec.get("values") or [None]
        return vals[0]
    if d == "fixed":
        return spec.get("value")
    if d == "randn":
        return spec.get("mean", 0.0)
    return None


def _spec_problems(spec: object, where: str) -> list[str]:
    if not (isinstance(spec, dict) and "dist" in spec):
        return []
    d, out = spec["dist"], []
    lo_hi = ("low", "high") if d in ("uniform", "loguniform", "quniform") else \
        ("lower", "upper") if d in ("randint", "qrandint", "lograndint") else None
    if lo_hi:
        lo, hi = spec.get(lo_hi[0]), spec.get(lo_hi[1])
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo >= hi:
            out.append(f"{where}: {lo_hi[0]} ({lo}) must be below {lo_hi[1]} ({hi})")
        if d in ("loguniform", "lograndint") and isinstance(lo, (int, float)) and lo <= 0:
            out.append(f"{where}: {d} needs {lo_hi[0]} > 0")
    if d in ("choice", "grid_search") and not spec.get("values"):
        out.append(f"{where}: values must be a non-empty list")
    return out


def validate_sweep(sweep: dict, base_config: dict | None = None) -> list[str]:
    """Check a sweep without Ray: structure, search specs, and the base config.

    With ``base_config`` (the parsed training config the sweep starts from),
    every ``overrides`` / ``search_space`` path must exist in it, as the sweep
    runner requires, and each phase's representative trial (each spec at its
    low end / first option) is validated with the training-config schema.

    Returns:
        ``path: message`` strings; empty when nothing is wrong.
    """
    from phoenx.config_schema import validate_config

    problems = list(validate_config(sweep, build_sweep_schema()))
    phases = [p for p in (sweep.get("phases") or []) if isinstance(p, dict)]
    names = [p.get("name") for p in phases]
    for n in {n for n in names if n and names.count(n) > 1}:
        problems.append(f"phases: duplicate phase name {n!r}")

    blocks = sweep.get("blocks") or {}
    for i, phase in enumerate(phases):
        label = f"phases.{i}"
        for path, spec in (phase.get("search_space") or {}).items():
            problems += _spec_problems(spec, f"{label}.search_space.{path}")
        for module, arch in (phase.get("architecture") or {}).items():
            if isinstance(arch, dict):
                problems += _spec_problems(arch.get("depth"), f"{label}.architecture.{module}.depth")
                for b in arch.get("blocks") or []:
                    if isinstance(blocks, dict) and b not in blocks:
                        problems.append(f"{label}.architecture.{module}.blocks: no block named "
                                        f"{b!r} in the sweep's blocks library")

    if base_config is None:
        return problems

    overrides = sweep.get("overrides") or {}
    trial_base = copy.deepcopy(base_config)
    for path, value in overrides.items():
        if get_path(trial_base, path) is _MISSING:
            problems.append(f"overrides.{path}: not in the base config (a sweep can only change "
                            "values its base config sets explicitly)")
        else:
            set_path(trial_base, path, value)
    for i, phase in enumerate(phases):
        trial = copy.deepcopy(trial_base)
        missing = False
        for path, spec in (phase.get("search_space") or {}).items():
            if get_path(trial, path) is _MISSING:
                problems.append(f"phases.{i}.search_space.{path}: not in the base config (a "
                                "sweep can only change values its base config sets explicitly)")
                missing = True
            else:
                set_path(trial, path, representative(spec))
        if missing:
            continue
        base_errors = validate_config(trial)
        for err in base_errors:
            problems.append(f"phases.{i} ({phase.get('name')}) sample trial: {err}")
        # Every option of a choice must be valid too, not just the first.
        for path, spec in (phase.get("search_space") or {}).items():
            if not (isinstance(spec, dict) and spec.get("dist") in ("choice", "grid_search")):
                continue
            for option in (spec.get("values") or [])[1:20]:
                alt = copy.deepcopy(trial)
                set_path(alt, path, option)
                for err in validate_config(alt):
                    if err not in base_errors:
                        problems.append(f"phases.{i}.search_space.{path}: option {option!r}: {err}")
    return problems


def suggest_spec(kind: str, value: object, name: str = "", choices: list | None = None) -> dict:
    """A starting search spec for a field, from its type and current value.

    Floats that look like rates or small coefficients get ``loguniform`` over
    a decade each side; discount-like values near 1 get ``uniform`` up to
    0.999; other numbers get ``uniform`` over half to double; integers get
    ``randint``, widths get ``choice`` of half / same / double; booleans and
    dropdowns get ``choice`` over their options.
    """
    lname = name.lower()
    if kind == "bool":
        return {"dist": "choice", "values": [True, False]}
    if kind == "enum" and choices:
        return {"dist": "choice", "values": list(choices)}
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        if lname in ("units", "hidden_size", "out_channels", "embed_dim", "d_model",
                     "dim_feedforward") and value > 1:
            return {"dist": "choice", "values": sorted({max(1, value // 2), value, value * 2})}
        lo = max(1, value // 2) if value > 0 else 0
        return {"dist": "randint", "lower": lo, "upper": max(value * 2, lo + 1) + 1}
    if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if lname in ("discount", "gamma", "gae_coefficient", "gae_lambda", "lambda") and v >= 0.9:
            return {"dist": "uniform", "low": 0.9, "high": 0.999}
        if v > 0 and ("lr" in lname.split("_") or "learning_rate" in lname or v < 0.01):
            return {"dist": "loguniform", "low": float(f"{v / 10:.3g}"), "high": float(f"{v * 10:.3g}")}
        if v == 0:
            return {"dist": "uniform", "low": 0.0, "high": 1.0}
        lo, hi = sorted((v * 0.5, v * 2))
        return {"dist": "uniform", "low": float(f"{lo:.3g}"), "high": float(f"{hi:.3g}")}
    return {"dist": "choice", "values": [value]}
