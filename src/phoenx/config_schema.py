"""Generate a JSON Schema for PhoenX training configs from the source code.

Config sections are passed straight into constructors as keyword arguments,
so the constructors already *are* the schema: their signatures give the
names, type hints and defaults, and their Google-style docstrings give the
descriptions. This module reads them by parsing the ``phoenx`` source with
``ast``. It never imports torch (or any other training dependency), so it is
fast and works in a bare editor environment.

Registries (``LAYER_REGISTRY``, ``HEAD_REGISTRY``, ``NORMALIZER_CLASSES``,
...) are read the same way, so dropdown choices track the code. Layer params,
which the registry lambdas read without a signature, come from
[LAYER_SPECS][phoenx.layer_specs.LAYER_SPECS].

The result is plain JSON Schema (draft-07, the dialect the VS Code YAML
extension speaks best). It powers editor autocomplete and
``phoenx-config validate``, and is the field list the config GUI renders.

Example:
    >>> from phoenx.config_schema import build_schema
    >>> schema = build_schema()
    >>> sorted(schema["properties"])[:3]
    ['agent', 'buffer', 'callbacks']
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from phoenx.layer_specs import KERNEL_NAMES, LAYER_SPECS, WEIGHTED_LAYERS

SRC = Path(__file__).resolve().parent

_MISSING = object()

# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _module(name: str) -> ast.Module:
    """Parse ``phoenx/<name>.py`` (dotted names map to sub-packages)."""
    path = SRC.joinpath(*name.split(".")).with_suffix(".py")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


_MODULES = (
    "rl_agents", "models", "buffer", "normalizer", "schedulers", "adaptive_kl",
    "noise", "intrinsic_motivation", "her", "env_wrapper", "trainer",
    "renderer", "metrics", "rl_callbacks",
)


@lru_cache(maxsize=None)
def _class_index() -> dict[str, ast.ClassDef]:
    """Every top-level class in the modules the schema covers, by name."""
    index: dict[str, ast.ClassDef] = {}
    for mod in _MODULES:
        for node in _module(mod).body:
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, node)
    return index


def registry_keys(module: str, name: str) -> list[str]:
    """String keys of a module-level dict literal, e.g. ``LAYER_REGISTRY``."""
    for node in _module(module).body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Dict):
            return [k.value for k in value.keys if isinstance(k, ast.Constant)]
    raise LookupError(f"No dict literal {name!r} in phoenx.{module}")


def registry_classes(module: str, name: str) -> dict[str, str]:
    """``{key: class name}`` for a registry whose values name classes.

    Handles both ``{"X": X}`` and ``{"X": {"cls": X, ...}}`` entries.
    """
    for node in _module(module).body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Dict):
            out = {}
            for k, v in zip(value.keys, value.values):
                if isinstance(v, ast.Dict):
                    for kk, vv in zip(v.keys, v.values):
                        if isinstance(kk, ast.Constant) and kk.value == "cls":
                            v = vv
                if isinstance(k, ast.Constant):
                    out[k.value] = ast.unparse(v).split(".")[-1]
            return out
    raise LookupError(f"No dict literal {name!r} in phoenx.{module}")


def subclasses_in(module: str, base: str) -> list[str]:
    """Names of classes in ``phoenx.<module>`` that derive (transitively) from ``base``."""
    classes = {n.name: n for n in _module(module).body if isinstance(n, ast.ClassDef)}
    def derives(name: str, seen: frozenset = frozenset()) -> bool:
        node = classes.get(name)
        if node is None or name in seen:
            return False
        bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
        return base in bases or any(derives(b, seen | {name}) for b in bases)
    return [n for n in classes if derives(n)]


@dataclass
class Param:
    """One constructor argument (or dataclass field) as read from source."""

    name: str
    annotation: ast.expr | None
    default: Any = _MISSING
    description: str = ""

    @property
    def required(self) -> bool:
        """True when the param has no default."""
        return self.default is _MISSING


def _parse_doc_args(doc: str | None) -> dict[str, str]:
    """Map arg name -> description from a Google ``Args:``/``Attributes:`` block."""
    if not doc:
        return {}
    out: dict[str, str] = {}
    lines = doc.splitlines()
    in_section, base, current = False, None, None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped in ("Args:", "Arguments:", "Attributes:", "Parameters:"):
            in_section, base, current = True, None, None
            continue
        if not in_section:
            continue
        if not stripped:
            continue
        if base is None:
            base = indent
        if indent < base or (indent == base and stripped.endswith(":") and " " not in stripped):
            in_section = False
            continue
        m = re.match(r"^\*{0,2}(\w+)\s*(\([^)]*\))?\s*:\s*(.*)$", stripped)
        if indent == base and m:
            current = m.group(1)
            out[current] = m.group(3)
        elif current is not None:
            out[current] = f"{out[current]} {stripped}".strip()
    return {k: re.sub(r"\s+", " ", v).strip() for k, v in out.items()}


def _literal(node: ast.expr | None) -> Any:
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return _UNREPRESENTABLE


_UNREPRESENTABLE = object()


def _init_of(cls: ast.ClassDef) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    return None


def _own_params(cls: ast.ClassDef) -> tuple[list[Param], bool]:
    """Params declared by ``cls`` itself, and whether it forwards ``**kwargs``.

    Returns ``(params, forwards)``. A class with no ``__init__`` that declares
    annotated fields is treated as a dataclass.
    """
    init = _init_of(cls)
    if init is not None:
        docs = _parse_doc_args(ast.get_docstring(init)) or _parse_doc_args(ast.get_docstring(cls))
        args = init.args
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        params = [
            Param(a.arg, a.annotation, _literal(d) if d is not None else _MISSING, docs.get(a.arg, ""))
            for a, d in zip(positional, defaults) if a.arg != "self"
        ]
        params += [
            Param(a.arg, a.annotation, _literal(d) if d is not None else _MISSING, docs.get(a.arg, ""))
            for a, d in zip(args.kwonlyargs, args.kw_defaults)
        ]
        return params, args.kwarg is not None
    fields = [n for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    if fields:
        docs = _parse_doc_args(ast.get_docstring(cls))
        return [
            Param(f.target.id, f.annotation, _literal(f.value), docs.get(f.target.id, ""))
            for f in fields
            if not (isinstance(f.annotation, ast.Subscript) and ast.unparse(f.annotation.value).endswith("ClassVar"))
        ], False
    return [], True  # no __init__: inherits the base constructor unchanged


def class_params(class_name: str) -> dict[str, Param]:
    """All keyword params ``class_name`` accepts, following base classes.

    A base class's params count when the subclass has no ``__init__`` or its
    ``__init__`` takes ``**kwargs`` (PhoenX constructors forward those to
    ``super().__init__``). The subclass's own spelling wins on conflicts.
    """
    index = _class_index()
    out: dict[str, Param] = {}
    name: str | None = class_name
    while name is not None and name in index:
        cls = index[name]
        own, forwards = _own_params(cls)
        for p in own:
            out.setdefault(p.name, p)
        if not forwards:
            break
        name = next((ast.unparse(b).split(".")[-1] for b in cls.bases
                     if ast.unparse(b).split(".")[-1] in index), None)
    return out


# --------------------------------------------------------------------------
# Annotation -> JSON Schema
# --------------------------------------------------------------------------

_PRIMITIVES = {
    "float": {"type": "number"}, "int": {"type": "integer"}, "bool": {"type": "boolean"},
    "str": {"type": "string"}, "dict": {"type": "object"}, "Dict": {"type": "object"},
    "list": {"type": "array"}, "List": {"type": "array"}, "tuple": {"type": "array"},
    "Tuple": {"type": "array"}, "None": {"type": "null"}, "NoneType": {"type": "null"},
    "Any": {}, "object": {}, "Path": {"type": "string"}, "device": {"type": "string"},
}

#: Annotation class names that are ``{type, config}`` objects in YAML.
_OBJECT_REFS = {
    "BaseNormalizer": "normalizer", "RunningNorm": "normalizer", "BatchNorm": "normalizer",
    "RewardNorm": "normalizer", "ImageScale": "normalizer", "DictNormalizer": "normalizer",
    "ScheduleWrapper": "schedule", "AdaptiveKL": "kl_adapter",
    "IntrinsicMotivation": "intrinsic_motivation", "Noise": "noise",
    "NormalNoise": "noise", "UniformNoise": "noise", "HindsightRelabeler": "hindsight",
}


def _simplify_union(options: list[dict]) -> dict:
    flat: list[dict] = []
    for o in options:
        flat.extend(o["anyOf"] if set(o) == {"anyOf"} else [o])
    if any(o == {} for o in flat):
        return {}
    if all(set(o) == {"type"} for o in flat):
        types: list[str] = []
        for o in flat:
            for t in (o["type"] if isinstance(o["type"], list) else [o["type"]]):
                if t not in types:
                    types.append(t)
        return {"type": types[0] if len(types) == 1 else types}
    uniq = []
    for o in flat:
        if o not in uniq:
            uniq.append(o)
    return uniq[0] if len(uniq) == 1 else {"anyOf": uniq}


def annotation_schema(node: ast.expr | None) -> dict:
    """Translate a type-hint AST node into a JSON Schema fragment."""
    if node is None:
        return {}
    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"type": "null"}
        if isinstance(node.value, str):  # forward reference
            try:
                return annotation_schema(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return {}
        return {}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _simplify_union([annotation_schema(node.left), annotation_schema(node.right)])
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = ast.unparse(node).split(".")[-1]
        if name in _OBJECT_REFS:
            return {"$ref": f"#/definitions/{_OBJECT_REFS[name]}"}
        return copy.deepcopy(_PRIMITIVES.get(name, {}))
    if isinstance(node, ast.Subscript):
        base = ast.unparse(node.value).split(".")[-1]
        inner = node.slice
        items = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
        if base == "Optional":
            return _simplify_union([annotation_schema(items[0]), {"type": "null"}])
        if base == "Union":
            return _simplify_union([annotation_schema(i) for i in items])
        if base == "Literal":
            values = [i.value for i in items if isinstance(i, ast.Constant)]
            return {"enum": values}
        if base in ("list", "List", "Sequence"):
            sub = annotation_schema(items[0])
            return {"type": "array", "items": sub} if sub else {"type": "array"}
        if base in ("tuple", "Tuple"):
            return {"type": "array"}
        if base in ("dict", "Dict", "Mapping"):
            sub = annotation_schema(items[1]) if len(items) > 1 else {}
            return {"type": "object", "additionalProperties": sub} if sub else {"type": "object"}
        return {}
    return {}


def param_schema(p: Param) -> dict:
    """Schema for one param: its type plus ``default`` and ``description``."""
    s = annotation_schema(p.annotation)
    # A ``None`` default makes null valid even when the hint leaves it out.
    if p.default is None and s:
        if "$ref" in s:
            s = {"anyOf": [s, {"type": "null"}]}
        elif set(s) == {"type"}:
            s = _simplify_union([s, {"type": "null"}])
        elif "enum" in s and None not in s["enum"]:
            s = {**s, "enum": [*s["enum"], None]}
    if p.default is not _MISSING and p.default is not _UNREPRESENTABLE:
        default = list(p.default) if isinstance(p.default, tuple) else p.default
        is_nan_or_inf = isinstance(default, float) and (default != default or abs(default) == float("inf"))
        if not is_nan_or_inf:
            s["default"] = default
    if p.description:
        s["description"] = p.description
    return s


def object_schema(
    class_name: str,
    exclude: set[str] = frozenset(),
    overrides: dict[str, dict] | None = None,
    allow_private: bool = True,
    extra: dict[str, dict] | None = None,
) -> dict:
    """Closed object schema for the keyword params of ``class_name``."""
    params = class_params(class_name)
    props = {n: param_schema(p) for n, p in params.items() if n not in exclude}
    props.update(extra or {})
    for k, v in (overrides or {}).items():
        if k in props:
            desc = props[k].get("description")
            props[k] = dict(v)
            if desc and "description" not in v:
                props[k]["description"] = desc
    schema: dict = {"type": "object", "properties": props, "additionalProperties": False}
    if allow_private:
        schema["patternProperties"] = {"^_": {"description": "Private diagnostic knob."}}
    required = [n for n, p in params.items() if p.required and n not in exclude]
    if required:
        schema["required"] = required
    return schema


def tagged_union(
    variants: dict[str, dict],
    config_key: str | None = "config",
    description: str = "",
    extra_props: dict[str, dict] | None = None,
) -> dict:
    """``{type: <choice>, config: {...}}`` with ``config`` chosen by ``type``.

    With ``config_key=None`` the variant's fields sit beside ``type`` (the
    flat form heads and layers use).
    """
    names = list(variants)
    s: dict = {
        "type": "object",
        "required": ["type"],
        "properties": {"type": {"enum": names}},
        "additionalProperties": False,
        # Read by the config GUI: where the type-specific fields live.
        "x-tagged": {"config_key": config_key},
    }
    if description:
        s["description"] = description
    if config_key is not None:
        s["properties"][config_key] = {"type": "object"}
    s["properties"].update(extra_props or {})
    branches = []
    for name, variant in variants.items():
        if config_key is not None:
            then = {"properties": {config_key: variant}}
        else:
            # Flat form: merge the variant's props into the outer object.
            then = {k: v for k, v in variant.items() if k in ("required",)}
            then["properties"] = variant.get("properties", {})
            for key in variant.get("properties", {}):
                s["properties"].setdefault(key, {})
        branches.append({"if": {"properties": {"type": {"const": name}}, "required": ["type"]},
                         "then": then})
    if branches:
        s["allOf"] = branches
    return s


# --------------------------------------------------------------------------
# Section definitions
# --------------------------------------------------------------------------

AGENT_CLASSES = ("Reinforce", "ActorCritic", "PPO", "DDPG", "TD3", "SAC")
ENV_CLASSES = {"gymnasium": "GymnasiumWrapper", "envpool": "EnvPoolWrapper", "isaacsim": "IsaacSimWrapper"}
BUFFER_CLASSES = ("ReplayBuffer", "PrioritizedReplayBuffer", "RolloutBuffer", "TrajectoryBuffer")
#: ``Noise.create_instance`` / ``IntrinsicMotivation.create_instance`` dispatch
#: on if/elif chains, not a dict literal; ``tests/test_config_schema.py``
#: checks these against the live factories.
NOISE_CLASSES = {"Normal": "NormalNoise", "NormalNoise": "NormalNoise",
                 "Uniform": "UniformNoise", "UniformNoise": "UniformNoise"}
INTRINSIC_CLASSES = ("ICM", "RND", "EpisodicNovelty")
OPTIMIZER_TYPES = ("Adam", "SGD", "RMSprop", "Adagrad")
SCHEDULE_TYPES = ("linear", "cosine", "exponential")

_INJECTED = {"env", "optimizer"}


def _ref(name: str) -> dict:
    return {"$ref": f"#/definitions/{name}"}


def _nullable(s: dict) -> dict:
    return {"anyOf": [s, {"type": "null"}]}


def _layer_def(require: bool) -> dict:
    variants = {}
    init_props = {
        "kernel": {"enum": list(KERNEL_NAMES), "description": "Weight initializer."},
        "kernel_params": {"type": "object", "description": "Keyword args for the initializer."},
        "kernel params": {"type": "object", "description": "Legacy spelling of kernel_params.",
                          "x-hidden": True},
    }
    for layer_type, spec in LAYER_SPECS.items():
        props = {}
        req = []
        for pname, pspec in spec["params"].items():
            ps = {k: v for k, v in pspec.items() if k not in ("required",)}
            if not require and pname in ("units", "out_features", "out_channels"):
                ps["x-hidden"] = True  # output widths come from the env
            props[pname] = ps
            if pspec.get("required") and require:
                req.append(pname)
        if layer_type in WEIGHTED_LAYERS:
            props.update(init_props)
        else:
            # Accepted (init skips weightless layers) but not worth showing.
            props.update({k: {**v, "x-hidden": True} for k, v in init_props.items()})
        params = {"type": "object", "properties": props,
                  "additionalProperties": "passthrough" in spec}
        if req:
            params["required"] = req
        if spec.get("doc") or spec.get("passthrough"):
            params["description"] = spec.get("doc") or f"Keyword args for {spec['passthrough']}."
        variants[layer_type] = params
    return tagged_union(variants, config_key="params",
                        description="One layer: a LAYER_REGISTRY type plus its params.")


def _head_variants() -> dict[str, dict]:
    heads = registry_classes("models", "HEAD_REGISTRY")
    variants = {}
    for key, cls in heads.items():
        variants[key] = object_schema(
            cls, exclude=_INJECTED | {"name"}, allow_private=False,
            overrides={
                "layer_config": {"type": "array", "items": _ref("layer")},
                "output_config": {"type": "array", "items": _ref("output_layer")},
                "merged_config": {"type": "array", "items": _ref("layer")},
                "optimizer_params": _nullable(_ref("optimizer")),
                "lr_scheduler": _nullable(_ref("schedule")),
                "temperature_schedule": _nullable(_ref("schedule")),
            },
        )
    return variants


def _definitions() -> dict:
    defs: dict[str, dict] = {}
    defs["layer"] = _layer_def(require=True)
    defs["output_layer"] = _layer_def(require=False)
    defs["output_layer"]["description"] = (
        "Init spec for an auto-created output layer (its width comes from the env).")
    defs["optimizer"] = {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"enum": list(OPTIMIZER_TYPES) + [t.lower() for t in OPTIMIZER_TYPES],
                     "description": "Optimizer name (case-insensitive)."},
            "params": {"type": "object", "description": "Keyword args, e.g. {lr: 3.0e-4}.",
                       "properties": {"lr": {"type": "number"}}},
        },
        "additionalProperties": False,
    }
    sched = object_schema("ScheduleWrapper", exclude=_INJECTED, allow_private=False)
    sched["properties"]["schedule_type"] = {
        "enum": list(SCHEDULE_TYPES), **{k: v for k, v in sched["properties"]["schedule_type"].items() if k == "description"}}
    sched["additionalProperties"] = True  # extra kwargs reach the torch scheduler
    defs["schedule"] = sched
    defs["kl_adapter"] = object_schema("AdaptiveKL", exclude=_INJECTED)

    norm_classes = registry_classes("normalizer", "NORMALIZER_CLASSES")
    defs["normalizer"] = tagged_union(
        {k: object_schema(c, exclude=_INJECTED | {"num_features"}) for k, c in norm_classes.items()},
        description="Normalizer: a NORMALIZER_CLASSES type plus its constructor config.")
    defs["noise"] = tagged_union(
        {k: object_schema(c, exclude=_INJECTED) for k, c in NOISE_CLASSES.items()})
    defs["intrinsic_motivation"] = tagged_union(
        {k: object_schema(k, exclude=_INJECTED, overrides={
            "reward_scheduler": _nullable(_ref("schedule")),
            "reward_normalizer": _nullable(_ref("normalizer")),
        }) for k in INTRINSIC_CLASSES})
    defs["hindsight"] = object_schema("HindsightRelabeler", exclude=_INJECTED)

    defs["subnetwork"] = {
        "type": "object",
        "properties": {
            "layer_config": {"type": "array", "items": _ref("layer")},
            "input_keys": {"type": ["array", "null"], "items": {"type": "string"},
                           "description": "Observation-dict keys this subnetwork reads."},
            "optimizer_params": _nullable(_ref("optimizer")),
            "lr_scheduler": _nullable(_ref("schedule")),
        },
        "additionalProperties": False,
    }
    defs["head"] = tagged_union(_head_variants(), config_key=None,
                                description="Branch head: a HEAD_REGISTRY type plus its settings.")
    model_body = {
        "device": {"type": ["string", "null"]},
        "roots": {"anyOf": [{"type": "object", "additionalProperties": _ref("subnetwork")},
                            {"type": "null"}],
                  "description": "Per-modality encoders, by name (null for flat observations)."},
        "trunk": {"anyOf": [_ref("subnetwork"), {"type": "null"}],
                  "description": "Shared fusion network (null for none)."},
        "branches": {"type": "object", "additionalProperties": _ref("head"),
                     "description": "Heads by role (policy, value, critic, ...)."},
        "optimizer_params": _nullable(_ref("optimizer")),
        "lr_scheduler": _nullable(_ref("schedule")),
        "shared_update": {"type": ["string", "null"]},
    }
    defs["model"] = {
        "type": "object",
        "properties": {"type": {"const": "ModularModel", "x-hidden": True},
                       "config": {"type": "object", "properties": model_body,
                                  "additionalProperties": False, "x-hidden": True},
                       **model_body},
        "additionalProperties": False,
    }

    wrapper_names = registry_keys("env_wrapper", "WRAPPER_REGISTRY")
    defs["env_wrapper"] = {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"type": "string", "examples": wrapper_names,
                     "description": "WRAPPER_REGISTRY name or any gymnasium wrapper class name."},
            "params": {"type": "object"},
        },
        "additionalProperties": False,
    }
    return defs


def _agent_variant(cls: str) -> dict:
    params = class_params(cls)
    # Heads, roots and trunk are built from `model`, not set directly.
    built = {n for n, p in params.items()
             if p.annotation is not None
             and re.search(r"Head|SubNetwork", ast.unparse(p.annotation))}
    s = object_schema(
        cls, exclude=built | {"roots", "trunk"},
        overrides={"optimizer_params": _nullable(_ref("optimizer")),
                   "lr_scheduler": _nullable(_ref("schedule"))},
        extra={"model": _ref("model")},
    )
    # Also settable under `model`, and the agent-level copy silently wins
    # (apply_model_config uses setdefault). Keep them valid for old configs
    # but out of the GUI unless a file already uses them.
    for dup in ("optimizer_params", "lr_scheduler", "shared_update"):
        if dup in s["properties"]:
            s["properties"][dup]["x-hidden"] = True
            s["properties"][dup]["description"] = (
                s["properties"][dup].get("description", "")
                + " Prefer setting this under agent.config.model.").strip()
    s["required"] = sorted(set(s.get("required", [])) | {"model"})
    if "N" in s["properties"]:
        s["properties"]["N"]["description"] = (
            "N-step return horizon. Set it only here: PhoenX copies it to the "
            "replay buffer, the VectorNStepReward wrapper and HER.")
    return s


@lru_cache(maxsize=1)
def _build_schema() -> dict:
    defs = _definitions()

    env_variants = {}
    for key, cls in ENV_CLASSES.items():
        env_variants[key] = object_schema(
            cls, exclude=_INJECTED,
            overrides={"wrappers": {"anyOf": [{"type": "array", "items": _ref("env_wrapper")},
                                              {"type": "null"}]}})

    buffer_variants = {
        cls: object_schema(cls, exclude=_INJECTED,
                           overrides={"hindsight": _nullable(_ref("hindsight"))})
        for cls in BUFFER_CLASSES
    }
    for cls in ("ReplayBuffer", "PrioritizedReplayBuffer"):
        n = buffer_variants[cls]["properties"].get("N")
        if n is not None:
            n["description"] = "Filled in from agent.config.N; you do not need to set it."

    callback_variants = {}
    # Callbacks register themselves via ``Callback.__init_subclass__``.
    for cls in subclasses_in("rl_callbacks", "Callback"):
        callback_variants[cls] = object_schema(cls, exclude=_INJECTED)

    schedule = object_schema("TrainingSchedule", allow_private=False)
    schedule.pop("required", None)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://github.com/jasonhayes1987/PhoenX_RL/phoenx-config.schema.json",
        "title": "PhoenX training config",
        "description": "Generated from the phoenx source by phoenx.config_schema. "
                       "Regenerate with `phoenx-config schema`.",
        "type": "object",
        "required": ["agent", "env", "buffer", "schedule"],
        "properties": {
            "device": {"type": "string", "description": "Torch device, e.g. cuda or cpu."},
            "save_dir": {"type": "string", "description": "Root directory for run outputs."},
            "log_level": {"enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          "description": "Trainer log level."},
            "schedule": schedule,
            "agent": tagged_union({c: _agent_variant(c) for c in AGENT_CLASSES},
                                  description="Algorithm and its hyperparameters."),
            "env": tagged_union(env_variants, description="Environment backend and settings."),
            "buffer": tagged_union(buffer_variants, description="Experience buffer."),
            "renderer": _nullable(object_schema("Renderer", allow_private=False)),
            "metrics": _nullable(object_schema("MetricsManager", allow_private=False)),
            "success_criterion": _nullable(object_schema("SuccessCriterion", allow_private=False)),
            "callbacks": {"anyOf": [{"type": "array", "items": tagged_union(callback_variants)},
                                    {"type": "null"}]},
        },
        # Extra top-level keys are allowed: configs park YAML anchors there
        # (``policy_lr: &policy_lr 3.0e-4``) and the builder ignores them.
        "additionalProperties": True,
        "definitions": defs,
    }


def build_schema() -> dict:
    """Return the full config JSON Schema (a fresh copy each call)."""
    return copy.deepcopy(_build_schema())


def validate_config(config: dict, schema: dict | None = None) -> list[str]:
    """Validate a parsed config against the schema.

    Requires the ``jsonschema`` package.

    Args:
        config: Parsed YAML mapping.
        schema: Schema to use; defaults to [build_schema][phoenx.config_schema.build_schema].

    Returns:
        Human-readable error strings, ``path: message``; empty when valid.
    """
    import jsonschema

    schema = schema or _build_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(config), key=lambda e: list(map(str, e.absolute_path))):
        err = _most_specific(err)
        path = ".".join(str(p) for p in err.absolute_path) or "<root>"
        msg = f"{path}: {_explain(err)}"
        if msg not in errors:
            errors.append(msg)
    return errors


def _most_specific(err):
    """For ``anyOf`` failures (nullable fields), report the non-null branch's error."""
    while err.validator in ("anyOf", "oneOf") and err.context:
        subs = [e for e in err.context
                if not (e.validator == "type" and e.validator_value == "null")]
        if not subs:
            break
        err = max(subs, key=lambda e: len(e.absolute_path))
    return err


def _explain(err) -> str:
    """Rewrite jsonschema's unknown-key messages into "did you mean" form."""
    import difflib

    if err.validator == "additionalProperties" and isinstance(err.instance, dict):
        known = list(err.schema.get("properties", {}))
        patterns = list(err.schema.get("patternProperties", {}))
        extras = [k for k in err.instance
                  if k not in known and not any(re.search(p, k) for p in patterns)]
        if extras:
            parts = []
            for k in extras:
                close = difflib.get_close_matches(k, known, n=1, cutoff=0.6)
                parts.append(f"{k!r} (did you mean {close[0]!r}?)" if close else repr(k))
            noun = "key" if len(extras) == 1 else "keys"
            return f"unknown {noun} {', '.join(parts)}"
    if err.validator == "enum":
        close = difflib.get_close_matches(str(err.instance), [str(v) for v in err.validator_value], n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        return f"{err.instance!r} is not one of the allowed values{hint}"
    return err.message
