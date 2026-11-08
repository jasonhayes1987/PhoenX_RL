"""Tests for the GUI-independent config form model (``phoenx.config_form``).

The editor is only trustworthy if loading a config and saving it back
changes nothing PhoenX would do differently. These tests check that for every
bundled config (and that the result still validates), then cover the pieces
the GUI drives: scalar parsing, defaults, type switching, list / map editing,
optional sections, unknown keys, and the YAML writer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from phoenx.config_form import (
    UNSET,
    ListNode,
    MapNode,
    ObjectNode,
    OptionalNode,
    ScalarNode,
    TaggedNode,
    build_form,
    dump_config,
)
from phoenx.config_schema import build_schema, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted(
    list((ROOT / "src/phoenx/examples/configs").rglob("*.yml"))
    + list((ROOT / "tests/fixtures").glob("*.yml"))
)


@pytest.fixture(scope="module")
def schema():
    return build_schema()


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _form(schema, config=None):
    form = build_form(schema)
    if config is not None:
        form.load(copy.deepcopy(config))
    return form


def _agent_cfg(form) -> ObjectNode:
    return form.children["agent"].variant


# =============================================================================
# Round trips over real configs
# =============================================================================

@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_load_save_round_trip_is_valid_and_stable(schema, path):
    form = _form(schema, _load(path))
    assert form.problems() == []
    minimal = form.value()
    assert validate_config(minimal) == []

    reloaded = yaml.safe_load(dump_config(minimal))
    assert reloaded == minimal
    assert _form(schema, reloaded).value() == minimal


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_open_and_save_changes_nothing(schema, path):
    """Every key a file had is written back unchanged, defaults and nulls included.

    Sweeps depend on this: a sweep can only change values its base config sets
    explicitly, so dropping an explicit default on save would break them.
    """
    original = _load(path)
    assert _form(schema, original).value() == original


def test_minimal_config_builds_the_same_trainer(tmp_path):
    """The saved file must train exactly like the hand-written one."""
    pytest.importorskip("torch")
    from phoenx.builder import build_trainer_from_config

    def cpu(o):
        if isinstance(o, dict):
            return {k: cpu(v) for k, v in o.items()}
        if isinstance(o, list):
            return [cpu(v) for v in o]
        return "cpu" if o == "cuda" else o

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k not in ("save_dir", "seed", "env")}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        return o

    for name in ("smoke_ppo_cartpole.yml", "smoke_sac_pendulum.yml", "smoke_td3_pendulum.yml"):
        original = cpu(_load(ROOT / "tests/fixtures" / name))
        minimal = _form(build_schema(), original).value()
        built = []
        for cfg in (copy.deepcopy(original), copy.deepcopy(minimal)):
            cfg["save_dir"] = str(tmp_path / name)
            t = build_trainer_from_config(cfg)
            built.append(json.loads(json.dumps(scrub({
                "agent": t.agent.get_config(),
                "buffer": t.buffer.get_config(),
                "schedule": vars(t.schedule),
            }), default=str)))
        assert built[0] == built[1], name


# =============================================================================
# Scalars
# =============================================================================

def _scalar(form, name) -> ScalarNode:
    node = _agent_cfg(form).children[name]
    assert isinstance(node, ScalarNode)
    return node


@pytest.mark.parametrize("text, expected", [
    ("3e-4", 3e-4), ("1e-06", 1e-6), ("1.0e-05", 1e-5), (".inf", float("inf")),
    ("inf", float("inf")), ("2", 2),
])
def test_number_parsing_accepts_python_style_floats(schema, text, expected):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    node = _scalar(form, "policy_clip")
    node.text = text
    assert node.value() == expected


def test_bad_number_is_reported_and_kept(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    node = _scalar(form, "policy_clip")
    node.text = "big"
    assert node.problems() == ["policy_clip: must be a number"]
    assert node.value() == "big"  # not silently dropped


def test_int_rejects_float_text(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    node = _scalar(form, "context_length")
    node.text = "4.5"
    assert node.problems()


def test_defaults_are_omitted_unless_set_or_requested(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {"discount": 0.99}}})
    agent = _agent_cfg(form)
    agent.children["gae_coefficient"].text = "0.9"   # typed, differs from the default
    agent.children["value_coef"].text = "0.5"        # typed, equals the default
    cfg = form.value()["agent"]["config"]
    assert cfg["discount"] == 0.99          # was in the file: kept even though default
    assert cfg["gae_coefficient"] == 0.9
    assert "value_coef" not in cfg          # never in the file and at its default
    agent.children["discount"].text = ""    # clearing a field un-sets it
    assert "discount" not in form.value()["agent"]["config"]
    full = form.value(include_defaults=True)["agent"]["config"]
    assert full["value_coef"] == 0.5 and full["discount"] == 0.99


def test_explicit_nulls_survive(schema):
    cfg = {"agent": {"type": "PPO", "config": {"model": {"roots": None, "trunk": None, "branches": {}},
                                               "entropy_schedule": None}},
           "env": {"type": "gymnasium", "config": {"cfg": "CartPole-v1", "wrappers": None}},
           "buffer": {"type": "RolloutBuffer", "config": {}},
           "schedule": {}}
    assert _form(schema, cfg).value() == cfg


def test_fields_start_at_the_code_default(schema):
    form = _form(schema)
    form.children["agent"].set_type("PPO")
    assert _scalar(form, "policy_clip").text == "0.2"
    assert form.value()["agent"]["config"] == {"model": {}}


# =============================================================================
# Structure
# =============================================================================

def test_switching_type_keeps_each_types_values(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {"policy_clip": 0.1}}})
    agent: TaggedNode = form.children["agent"]
    agent.set_type("SAC")
    _scalar(form, "tau").text = "0.02"
    assert "policy_clip" not in form.value()["agent"]["config"]
    agent.set_type("PPO")
    assert form.value()["agent"]["config"]["policy_clip"] == 0.1
    agent.set_type("SAC")
    assert form.value()["agent"]["config"]["tau"] == 0.02


def test_list_add_move_remove(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {"model": {"branches": {}}}}})
    branches: MapNode = _agent_cfg(form).children["model"].children["branches"]
    head = branches.add("policy")
    head.set_type("StochasticDiscreteHead")
    layers: ListNode = head.variant.children["layer_config"]
    a, b = layers.add(), layers.add()
    a.set_type("dense")
    a.variant.children["units"].text = "32"
    b.set_type("relu")
    layers.move(1, -1)
    value = form.value()["agent"]["config"]["model"]["branches"]["policy"]["layer_config"]
    assert value == [{"type": "relu"}, {"type": "dense", "params": {"units": 32}}]
    layers.remove(0)
    value = form.value()["agent"]["config"]["model"]["branches"]["policy"]["layer_config"]
    assert value == [{"type": "dense", "params": {"units": 32}}]


def test_map_add_rename_and_duplicate(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    branches: MapNode = _agent_cfg(form).children["model"].children["branches"]
    branches.add("policy").set_type("ValueHead")
    with pytest.raises(ValueError, match="already exists"):
        branches.add("policy")
    branches.rename("policy", "value")
    assert list(form.value()["agent"]["config"]["model"]["branches"]) == ["value"]


def test_optional_section_toggles(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    norm = _agent_cfg(form).children["state_normalizer"]
    assert isinstance(norm, OptionalNode)
    assert "state_normalizer" not in form.value()["agent"]["config"]
    norm.enabled = True
    norm.inner.set_type("RunningNorm")
    assert form.value()["agent"]["config"]["state_normalizer"] == {"type": "RunningNorm", "config": {}}
    norm.enabled = False
    assert "state_normalizer" not in form.value()["agent"]["config"]


def test_unknown_and_private_keys_are_kept_and_typos_reported(schema):
    cfg = {"agent": {"type": "PPO", "config": {"_diag_freq": 10, "polcy_clip": 0.1}}}
    form = _form(schema, cfg)
    out = form.value()["agent"]["config"]
    assert out["_diag_freq"] == 10 and out["polcy_clip"] == 0.1
    errors = validate_config(form.value())
    assert any("polcy_clip" in e for e in errors)
    assert not any("_diag_freq" in e for e in errors)


def test_hidden_internal_fields(schema):
    form = _form(schema, {"agent": {"type": "PPO", "config": {}}})
    model = _agent_cfg(form).children["model"]
    assert "type" not in [c.name for c in model.visible_children()]
    assert "optimizer_params" not in [c.name for c in _agent_cfg(form).visible_children()]
    # Shown again when a loaded file actually uses it.
    form = _form(schema, {"agent": {"type": "PPO", "config": {"shared_update": "combined"}}})
    assert "shared_update" in [c.name for c in _agent_cfg(form).visible_children()]


def test_layer_params_omitted_when_empty_but_config_blocks_kept(schema):
    form = _form(schema, {
        "agent": {"type": "PPO", "config": {
            "state_normalizer": {"type": "RunningNorm", "config": {}},
            "model": {"branches": {"value": {"type": "ValueHead",
                                             "layer_config": [{"type": "relu"}]}}}}}})
    cfg = form.value()["agent"]["config"]
    assert cfg["model"]["branches"]["value"]["layer_config"] == [{"type": "relu"}]
    assert cfg["state_normalizer"] == {"type": "RunningNorm", "config": {}}


# =============================================================================
# YAML writer
# =============================================================================

def test_dump_uses_blocks_for_sections_and_flow_for_leaves(schema):
    cfg = {"schedule": {"stop_units": 10, "seed": 1},
           "agent": {"type": "PPO", "config": {"model": {"branches": {"value": {
               "type": "ValueHead",
               "layer_config": [{"type": "dense", "params": {"units": 64}}]}}}}}}
    text = dump_config(cfg)
    assert "schedule:\n  stop_units: 10\n" in text
    assert "- {type: dense, params: {units: 64}}" in text
    assert yaml.safe_load(text) == cfg


# =============================================================================
# Every field the GUI shows has help text
# =============================================================================

def _visible_fields_without_help(schema) -> list[str]:
    """Walk the whole form as the GUI renders it, visiting every type choice and
    one fresh item of every list / map, and list fields that have no tooltip."""
    missing: list[str] = []
    seen_variants: set[tuple] = set()

    def check(node, path):
        if not node.hidden and not node.description:
            missing.append(path)

    def visit(node, path, depth=0):
        if depth > 12:
            return
        if isinstance(node, OptionalNode):
            check(node, path)
            visit_inner(node.inner, path, depth)
        elif isinstance(node, (ScalarNode,)):
            if node.name != "type":
                check(node, path)
        else:
            check(node, path)
            visit_inner(node, path, depth)

    def visit_inner(node, path, depth):
        if isinstance(node, ObjectNode):
            for child in node.visible_children():
                visit(child, f"{path}.{child.name}", depth + 1)
        elif isinstance(node, TaggedNode):
            for choice in node.choices:
                key = (id(node.variant_schemas[choice]),)
                if key in seen_variants:
                    continue
                seen_variants.add(key)
                node.set_type(choice)
                visit_inner(node.variant, f"{path}<{choice}>", depth + 1)
        elif isinstance(node, ListNode):
            visit_inner(node.new_item(), f"{path}[]", depth + 1)
        elif isinstance(node, MapNode):
            entry = __import__("phoenx.config_form", fromlist=["make_node"]).make_node(
                "x", node.value_schema, node.root, required=True)
            visit_inner(entry, f"{path}.<name>", depth + 1)

    form = build_form(schema)
    for child in form.visible_children():
        visit(child, child.name)
    return sorted(set(missing))


def test_every_visible_field_has_help_text(schema):
    assert _visible_fields_without_help(schema) == []
