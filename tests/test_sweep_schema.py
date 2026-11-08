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
