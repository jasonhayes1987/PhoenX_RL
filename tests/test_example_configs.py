"""Unit tests for packaged example configs and ``load_config`` fallback.

Covers ``available_example_configs`` and the on-disk-then-bundled resolution
in ``phoenx.builder.load_config``. Packaged examples live under
``phoenx.examples`` and are shipped as package data; that every bundled name
resolves, parses, and speaks the current schema is covered by
``tests/test_builders.py::TestBundledConfigSchema``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phoenx.builder import available_example_configs, load_config

EXPECTED_EXAMPLE_CONFIGS = [
    "IsaacSim/franka/cube_lift/dense/ppo_camera.yml",
    "LunarLander-v3/reinforce.yml",
    "LunarLanderContinuous-v3/ppo.yml",
    "LunarLanderContinuous-v3/sac.yml",
]


def test_available_example_configs_lists_bundled_names():
    """available_example_configs returns the four bundled names, sorted, with /."""
    names = available_example_configs()
    assert names == EXPECTED_EXAMPLE_CONFIGS


def test_load_config_resolves_bundled_forward_slash(tmp_path, monkeypatch):
    """A bundled relative name with forward slashes loads a non-empty dict."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config("LunarLanderContinuous-v3/sac.yml")
    assert isinstance(cfg, dict)
    assert cfg


def test_load_config_resolves_bundled_backslash(tmp_path, monkeypatch):
    """Windows-style backslash separators resolve to the same bundled dict."""
    monkeypatch.chdir(tmp_path)
    forward = load_config("LunarLanderContinuous-v3/sac.yml")
    backslash = load_config("LunarLanderContinuous-v3\\sac.yml")
    assert backslash == forward


def test_on_disk_relative_path_wins_over_bundled(tmp_path, monkeypatch):
    """An existing on-disk relative path always wins over a same-shaped bundled name."""
    rel = Path("LunarLanderContinuous-v3") / "sac.yml"
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    disk_payload = {"marker": "from-disk-not-bundled", "unique": 42}
    target.write_text(yaml.safe_dump(disk_payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    loaded = load_config("LunarLanderContinuous-v3/sac.yml")
    assert loaded == disk_payload


def test_load_config_absolute_path(tmp_path):
    """An absolute on-disk path still loads YAML (pre-existing caller behavior)."""
    cfg_path = tmp_path / "standalone.yml"
    payload = {"hello": "absolute", "n": 7}
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert load_config(cfg_path) == payload
    assert load_config(str(cfg_path)) == payload


def test_missing_relative_name_raises_with_bundled_list(tmp_path, monkeypatch):
    """A missing relative name raises FileNotFoundError listing bundled examples."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config("TotallyMissing-v0/nope.yml")

    message = str(exc_info.value)
    assert "TotallyMissing-v0/nope.yml" in message
    assert "Bundled examples:" in message
    for name in EXPECTED_EXAMPLE_CONFIGS:
        assert name in message


def test_missing_absolute_path_does_not_resolve_bundled(tmp_path):
    """A missing absolute path raises FileNotFoundError and never uses package data."""
    missing = tmp_path / "LunarLanderContinuous-v3" / "sac.yml"
    assert missing.is_absolute()
    assert not missing.exists()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_config(missing)

    message = str(exc_info.value)
    assert str(missing) in message or missing.as_posix() in message
    assert "Bundled examples:" in message
