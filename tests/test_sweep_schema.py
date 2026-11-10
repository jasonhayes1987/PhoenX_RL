"""Sweep configs reject misspelled keys (``phoenx.sweep_schema.check_sweep_keys``).

A misspelled phase key used to be merged into the phase and never read, so
for example ``serach_space:`` ran that phase with nothing to search.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from phoenx.sweep_schema import check_sweep_keys

SWEEPS = sorted((Path(__file__).resolve().parents[1] / "src/phoenx/examples/sweeps").glob("*.yml"))


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_bundled_sweeps_pass(path):
    check_sweep_keys(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.fixture()
def sweep():
    return yaml.safe_load((SWEEPS[0].parent / "lunarlander_ppo.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutate, expected", [
    (lambda s: s["phases"][2].__setitem__("serach_space", s["phases"][2].pop("search_space")),
     "did you mean 'search_space'"),
    (lambda s: s.__setitem__("overides", {}), "did you mean 'overrides'"),
    (lambda s: s["phases"][0]["tune"].__setitem__("num_sample", 3), "did you mean 'num_samples'"),
    (lambda s: s["defaults"].__setitem__("metrc", "x"), "did you mean 'metric'"),
])
def test_typos_rejected(sweep, mutate, expected):
    mutate(sweep)
    with pytest.raises(ValueError, match=expected):
        check_sweep_keys(sweep)


def test_single_phase_shorthand_and_private_keys():
    check_sweep_keys({"base_config": "x.yml", "search_space": {}, "tune": {"num_samples": 2},
                      "_sweep_name": "internal"})
    with pytest.raises(ValueError, match="num_samples"):
        check_sweep_keys({"base_config": "x.yml", "search_space": {}, "tune": {"num_sample": 2}})


def test_validate_sweep_config_uses_the_check(sweep):
    pytest.importorskip("ray")
    from phoenx.ray_tune import validate_sweep_config

    bad = copy.deepcopy(sweep)
    bad["phases"][0]["serach_space"] = {}
    with pytest.raises(ValueError, match="serach_space"):
        validate_sweep_config(bad)


# =============================================================================
# Sweep schema, Ray-free validation, and spec suggestions (the Sweep tab's core)
# =============================================================================

from phoenx.sweep_schema import (  # noqa: E402
    build_sweep_schema,
    get_path,
    representative,
    suggest_spec,
    to_phases_form,
    validate_sweep,
)


def _base(name="LunarLanderContinuous-v3/ppo.yml"):
    root = Path(__file__).resolve().parents[1] / "src/phoenx/examples/configs"
    return yaml.safe_load((root / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_bundled_sweeps_validate_against_their_base(path):
    sweep = yaml.safe_load(path.read_text(encoding="utf-8"))
    sweep, _ = to_phases_form(sweep)
    assert validate_sweep(sweep, _base(sweep["base_config"])) == []


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_bundled_sweeps_round_trip_through_the_form(path):
    from phoenx.config_form import build_form

    sweep, _ = to_phases_form(yaml.safe_load(path.read_text(encoding="utf-8")))
    form = build_form(build_sweep_schema())
    form.load(sweep)
    assert form.problems() == []
    assert form.value() == sweep


@pytest.mark.parametrize("mutate, expected", [
    (lambda s: s["phases"][2]["search_space"].__setitem__(
        "agent.config.polcy_clip", {"dist": "uniform", "low": 0.1, "high": 0.4}),
     "agent.config.polcy_clip: not in the base config"),
    (lambda s: s["overrides"].__setitem__("schedule.stop_unitz", 5), "overrides.schedule.stop_unitz"),
    (lambda s: s["phases"][2]["search_space"].__setitem__(
        "agent.config.discount", {"dist": "uniform", "low": 0.99, "high": 0.95}),
     "low (0.99) must be below high (0.95)"),
    (lambda s: s["phases"][2]["search_space"].__setitem__(
        "agent.config.discount", {"dist": "loguniform", "low": 0.0, "high": 0.9}),
     "loguniform needs low > 0"),
    (lambda s: s["phases"][2]["search_space"].__setitem__(
        "agent.config.discount", {"dist": "uniform", "low": 0.9}), "'high' is a required property"),
    (lambda s: s["phases"][0]["architecture"]["branches.policy"].__setitem__("blocks", ["nope"]),
     "no block named 'nope'"),
    (lambda s: s["phases"][1].__setitem__("search_space", {
        "agent.config.model.branches.policy.layer_config.1.type":
            {"dist": "choice", "values": ["tanh", "tanhh"]}}),
     "option 'tanhh'"),
    (lambda s: s["phases"][1].__setitem__("search_space", {
        "agent.config.gae_coefficient": {"dist": "choice", "values": ["high"]}}),
     "sample trial: agent.config.gae_coefficient: 'high' is not of type 'number'"),
    (lambda s: s["phases"].append(dict(s["phases"][0])), "duplicate phase name 'architecture'"),
])
def test_validate_sweep_catches(sweep, mutate, expected):
    mutate(sweep)
    problems = validate_sweep(sweep, _base())
    assert any(expected in p for p in problems), problems


def test_get_path_and_representative():
    cfg = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert get_path(cfg, "a.b.1.c") == 2
    assert get_path(cfg, "a.b.5.c") is not 2  # noqa: F632 (sentinel)
    assert representative({"dist": "loguniform", "low": 1e-5, "high": 1e-3}) == 1e-5
    assert representative({"dist": "randint", "lower": 1, "upper": 4}) == 1
    assert representative({"dist": "choice", "values": ["relu", "tanh"]}) == "relu"
    assert representative({"dist": "fixed", "value": 7}) == 7


def test_single_phase_shorthand_becomes_one_phase():
    sweep = {"base_config": "x.yml", "search_space": {"a": {"dist": "fixed", "value": 1}},
             "tune": {"num_samples": 2}}
    out, changed = to_phases_form(sweep)
    assert changed and out == {"base_config": "x.yml", "phases": [
        {"name": "phase_0", "search_space": sweep["search_space"], "tune": sweep["tune"]}]}


@pytest.mark.parametrize("kind, value, name, choices, expected", [
    ("number", 3e-4, "lr", None, {"dist": "loguniform", "low": 3e-5, "high": 3e-3}),
    ("number", 0.99, "discount", None, {"dist": "uniform", "low": 0.9, "high": 0.999}),
    ("number", 0.2, "policy_clip", None, {"dist": "uniform", "low": 0.1, "high": 0.4}),
    ("int", 64, "units", None, {"dist": "choice", "values": [32, 64, 128]}),
    ("int", 5, "learning_epochs", None, {"dist": "randint", "lower": 2, "upper": 11}),
    ("bool", True, "auto_entropy_tuning", None, {"dist": "choice", "values": [True, False]}),
    ("enum", "relu", "type", ["relu", "tanh"], {"dist": "choice", "values": ["relu", "tanh"]}),
])
def test_suggest_spec(kind, value, name, choices, expected):
    assert suggest_spec(kind, value, name, choices) == expected
