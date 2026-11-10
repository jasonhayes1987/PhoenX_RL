"""Tests for the generated config JSON Schema and the ``phoenx-config`` CLI.

The schema is built by parsing source, so the most important tests here pin
it to the *live* code: AST-read registries must equal the runtime
registries, and the params the schema lists for each agent must equal what
``inspect`` sees on the real class. If someone adds a parameter or registry
entry, the schema follows automatically; if the parser ever misreads the
source, these fail.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from phoenx.config_schema import (
    AGENT_CLASSES,
    BUFFER_CLASSES,
    INTRINSIC_CLASSES,
    NOISE_CLASSES,
    build_schema,
    class_params,
    registry_classes,
    registry_keys,
    subclasses_in,
    validate_config,
)
from phoenx.config_utils import init_param_names
from phoenx.layer_specs import LAYER_SPECS

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted(
    list((ROOT / "src/phoenx/examples/configs").rglob("*.yml"))
    + list((ROOT / "tests/fixtures").glob("*.yml"))
)


def _agent_variants(schema: dict) -> dict[str, dict]:
    return {b["if"]["properties"]["type"]["const"]: b["then"]["properties"]["config"]
            for b in schema["properties"]["agent"]["allOf"]}


# =============================================================================
# The schema matches the live code
# =============================================================================

class TestSchemaTracksCode:
    def test_builds_without_importing_torch(self):
        code = ("import sys; sys.modules['torch'] = None\n"
                "from phoenx.config_schema import build_schema\n"
                "assert build_schema()['properties']['agent']\n")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_layer_specs_cover_layer_registry_exactly(self):
        from phoenx.models import LAYER_REGISTRY
        assert set(LAYER_SPECS) == set(LAYER_REGISTRY)

    def test_registries_read_from_source_match_runtime(self):
        from phoenx.env_wrapper import WRAPPER_REGISTRY
        from phoenx.models import HEAD_REGISTRY, LAYER_REGISTRY
        from phoenx.normalizer import NORMALIZER_CLASSES

        assert registry_keys("models", "LAYER_REGISTRY") == list(LAYER_REGISTRY)
        assert registry_classes("models", "HEAD_REGISTRY") == {k: v.__name__ for k, v in HEAD_REGISTRY.items()}
        assert registry_classes("normalizer", "NORMALIZER_CLASSES") == {
            k: v.__name__ for k, v in NORMALIZER_CLASSES.items()}
        assert registry_keys("env_wrapper", "WRAPPER_REGISTRY") == list(WRAPPER_REGISTRY)

    def test_callback_types_match_runtime_registry(self):
        from phoenx.rl_callbacks import _CALLBACK_REGISTRY
        assert set(subclasses_in("rl_callbacks", "Callback")) == set(_CALLBACK_REGISTRY)

    def test_hardcoded_factories_match_runtime(self):
        from phoenx import buffer, intrinsic_motivation
        from phoenx.noise import Noise

        for key, cls_name in NOISE_CLASSES.items():
            kwargs = {"shape": (1,)} if cls_name == "UniformNoise" else {}
            assert type(Noise.create_instance(key, device="cpu", **kwargs)).__name__ == cls_name
        for name in INTRINSIC_CLASSES:
            assert hasattr(intrinsic_motivation, name)
        for name in BUFFER_CLASSES:
            assert hasattr(buffer, name)

    @pytest.mark.parametrize("cls_name", AGENT_CLASSES)
    def test_agent_params_match_inspect(self, cls_name):
        from phoenx import rl_agents

        live = init_param_names(getattr(rl_agents, cls_name))
        assert set(class_params(cls_name)) == live

    @pytest.mark.parametrize("cls_name", AGENT_CLASSES)
    def test_agent_schema_lists_every_settable_param(self, cls_name):
        from phoenx import rl_agents

        props = set(_agent_variants(build_schema())[cls_name]["properties"])
        live = init_param_names(getattr(rl_agents, cls_name))
        # Heads, roots and trunk are built from `model`; everything else is settable.
        built = live - props
        assert props - live == {"model"}
        assert built >= {"roots", "trunk"}
        assert all(n in ("roots", "trunk", "policy", "value", "critic", "critic_b", "actor")
                   for n in built), built


# =============================================================================
# Real configs validate; typos do not
# =============================================================================

@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_bundled_configs_validate(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_config(config) == []


@pytest.fixture(scope="module")
def ppo_config() -> dict:
    path = ROOT / "src/phoenx/examples/configs/LunarLanderContinuous-v3/ppo.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _policy(cfg):
    return cfg["agent"]["config"]["model"]["branches"]["policy"]


@pytest.mark.parametrize("mutate, expected", [
    (lambda c: c["agent"]["config"].update(polcy_clip=0.1),
     "agent.config: unknown key 'polcy_clip' (did you mean 'policy_clip'?)"),
    (lambda c: c["agent"]["config"].update(tau=0.01),
     "agent.config: unknown key 'tau'"),
    (lambda c: c["agent"]["config"].update(policy_clip="big"),
     "agent.config.policy_clip: 'big' is not of type 'number'"),
    (lambda c: c["agent"].update(type="PP0"),
     "agent.type: 'PP0' is not one of the allowed values (did you mean 'PPO'?)"),
    (lambda c: _policy(c)["layer_config"][0]["params"].update(kernal="orthogonal"),
     "unknown key 'kernal' (did you mean 'kernel'?)"),
    (lambda c: _policy(c)["layer_config"][0]["params"].pop("units"),
     "'units' is a required property"),
    (lambda c: _policy(c).update(layer_confg=[]),
     "unknown key 'layer_confg' (did you mean 'layer_config'?)"),
    (lambda c: c["agent"]["config"]["state_normalizer"]["config"].update(clip_val=1),
     "agent.config.state_normalizer.config: unknown key 'clip_val' (did you mean 'clip_value'?)"),
    (lambda c: c["agent"]["config"]["entropy_schedule"].update(schedule_type="cosin"),
     "(did you mean 'cosine'?)"),
    (lambda c: c["schedule"].update(learn_evry=1),
     "schedule: unknown key 'learn_evry' (did you mean 'learn_every'?)"),
    (lambda c: c["env"]["config"].update(num_env=1),
     "env.config: unknown key 'num_env' (did you mean 'num_envs'?)"),
])
def test_typos_are_reported(ppo_config, mutate, expected):
    cfg = copy.deepcopy(ppo_config)
    mutate(cfg)
    errors = validate_config(cfg)
    assert any(expected in e for e in errors), errors


def test_private_keys_and_top_level_anchor_holders_allowed(ppo_config):
    cfg = copy.deepcopy(ppo_config)
    cfg["agent"]["config"]["_anything"] = 1
    cfg["policy_lr"] = 3e-4  # anchors are often parked at the top level
    assert validate_config(cfg) == []


# =============================================================================
# CLI
# =============================================================================

def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "phoenx.cli.config", *args],
                          capture_output=True, text=True, cwd=cwd)


def test_cli_validate_ok_and_fail(tmp_path):
    ok = _cli("validate", *map(str, CONFIG_FILES))
    assert ok.returncode == 0, ok.stdout

    bad = tmp_path / "bad.yml"
    bad.write_text(yaml.safe_dump({
        "agent": {"type": "SAC", "config": {"N": 3, "model": {"branches": {}}}},
        "env": {"type": "gymnasium", "config": {"cfg": "Pendulum-v1"}},
        "buffer": {"type": "ReplayBuffer", "config": {"N": 2}},
        "schedule": {},
    }))
    res = _cli("validate", str(bad))
    assert res.returncode == 1
    assert "N-step horizon mismatch" in res.stdout


def test_cli_schema_and_vscode(tmp_path):
    res = _cli("schema", "-o", "-")
    assert json.loads(res.stdout)["title"] == "PhoenX training config"

    res = _cli("vscode", "--root", str(tmp_path))
    assert res.returncode == 0, res.stderr
    settings = json.loads((tmp_path / ".vscode/settings.json").read_text())
    assert "./phoenx-config.schema.json" in settings["yaml.schemas"]
    assert (tmp_path / "phoenx-config.schema.json").exists()
