"""Tests for the Ray Tune sweep engine (``src/phoenx/ray_tune.py``).

Covers the resolution layer (dotted paths, ``dist:`` spec parsing, search-space
construction, block-grammar architecture assembly, the ``optimizers:`` writer,
``auto_learn_every``, constraint validation, trial-config resolution) and the
driver layer (searcher/scheduler/stopper factories, phase promotion /
``points_to_evaluate`` seeding, ``write_best_config``, the two bundled example
sweeps) with no Ray cluster, no network, and no GPU requirement.

One end-to-end test (``TestEndToEndSweep``) is marked ``@pytest.mark.slow``: a
real two-phase CartPole sweep through ``run_sweep`` that actually starts a
local Ray cluster. Everything else runs Ray Tune's pure-Python factories
(samplers, searchers, schedulers, stoppers) without ``ray.init()``.
"""

from __future__ import annotations

import copy
import os

import pytest
import yaml
from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Domain
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper

from phoenx import models, ray_tune
from phoenx.builder import build_trainer_from_config, load_config
from phoenx.ray_tune import (
    apply_architecture,
    apply_auto_learn_every,
    apply_optimizers,
    available_example_sweeps,
    build_layer_stack,
    build_scheduler,
    build_search_alg,
    build_search_space,
    build_stopper,
    get_by_path,
    is_search_spec,
    load_sweep_config,
    normalize_phases,
    parse_block_library,
    parse_search_spec,
    resolve_trial_config,
    run_sweep,
    sample_search_space,
    set_by_path,
    validate_sweep_config,
    validate_trial_config,
    write_best_config,
)

DEV = "cpu"
DENSE = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                               "kernel_params": {"gain": 1.41421356}}}
RELU = {"type": "relu"}
OUT = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 3e-4}}

CARTPOLE_ENV_CONFIG = {"type": "gymnasium", "config": {
    "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
    "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 7}}


def _tiny_ppo_config(save_dir: str) -> dict:
    """A minimal, real, buildable feedforward CartPole PPO training config."""
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "device": DEV,
        "schedule": {
            "stop_unit": "timestep", "stop_units": 64, "learn_every_unit": "timestep",
            "learn_every": 16, "updates_per_learn": 1, "batch_size": 1,
            "mini_batch_size": 8, "learning_epochs": 2, "warmup_steps": 0, "seed": 7,
        },
        "agent": {"type": "PPO", "config": {
            "name": "PPO",
            "model": {
                "branches": {
                    "policy": {"type": "StochasticDiscreteHead",
                               "layer_config": [DENSE(16), RELU], "output_config": OUT,
                               "optimizer_params": OPT, "distribution": "categorical", "device": DEV},
                    "value": {"type": "ValueHead",
                              "layer_config": [DENSE(16), RELU], "output_config": OUT,
                              "optimizer_params": OPT, "device": DEV},
                },
            },
            "discount": 0.99, "gae_coefficient": 0.95, "auto_entropy_tuning": False,
            "entropy_coefficient": 0.01, "policy_clip": 0.2, "value_clip": 0.2,
            "policy_grad_clip": 1.0, "value_grad_clip": 1.0, "value_coef": 0.5,
            "device": DEV, "log_level": "ERROR",
        }},
        "env": CARTPOLE_ENV_CONFIG,
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
    }


# =============================================================================
# 1. Dotted-path get / set
# =============================================================================
class TestDottedPaths:
    def test_get_by_path_dict_and_bracket_forms_agree(self):
        cfg = {"a": {"b": [10, {"c": 5}]}}
        assert get_by_path(cfg, "a.b.1.c") == 5
        assert get_by_path(cfg, "a.b[1].c") == 5
        assert get_by_path(cfg, "a.b.0") == 10
        assert get_by_path(cfg, "a.b[0]") == 10

    def test_set_then_get_round_trips(self):
        cfg = {"a": {"b": [10, {"c": 5}]}}
        set_by_path(cfg, "a.b.1.c", 99)
        assert get_by_path(cfg, "a.b.1.c") == 99
        set_by_path(cfg, "a.b[0]", 42)
        assert cfg["a"]["b"][0] == 42

    def test_get_missing_dict_segment_raises_keyerror_with_message(self):
        cfg = {"a": {"b": 1}}
        with pytest.raises(KeyError, match=r"missing segment 'bogus'"):
            get_by_path(cfg, "a.bogus.c")

    def test_get_list_index_out_of_range_raises_valueerror(self):
        cfg = {"a": [1, 2]}
        with pytest.raises(ValueError, match="out of range"):
            get_by_path(cfg, "a.5")

    def test_get_list_index_non_integer_raises_valueerror(self):
        cfg = {"a": [1, 2]}
        with pytest.raises(ValueError, match="not a valid list index"):
            get_by_path(cfg, "a.oops")

    def test_get_descend_into_non_container_raises_keyerror(self):
        cfg = {"a": 5}
        with pytest.raises(KeyError, match="cannot descend"):
            get_by_path(cfg, "a.b")

    def test_empty_path_raises_valueerror(self):
        with pytest.raises(ValueError, match="no segments"):
            get_by_path({}, "")

    def test_negative_list_index(self):
        cfg = {"a": [1, 2, 3]}
        assert get_by_path(cfg, "a.-1") == 3

    def test_set_missing_final_key_without_create_raises_keyerror(self):
        cfg = {"a": {}}
        with pytest.raises(KeyError, match=r"missing segment 'b'"):
            set_by_path(cfg, "a.b", 1)

    def test_set_missing_intermediate_dict_without_create_raises_keyerror(self):
        cfg = {"a": {}}
        with pytest.raises(KeyError, match=r"missing segment 'b'"):
            set_by_path(cfg, "a.b.c", 1)

    def test_set_never_silently_creates_by_default(self):
        """The deliberate safety property: a typo'd path must raise, not
        silently create a sibling key and leave the real target unset."""
        cfg = {"schedule": {"learn_every": 8}}
        with pytest.raises(KeyError):
            set_by_path(cfg, "schedule.lear_every", 999)
        assert cfg == {"schedule": {"learn_every": 8}}

    def test_set_create_true_creates_missing_intermediate_dicts(self):
        cfg = {"a": {}}
        set_by_path(cfg, "a.b.c", 7, create=True)
        assert cfg == {"a": {"b": {"c": 7}}}

    def test_set_create_true_never_creates_list_elements(self):
        cfg = {"a": [1, 2]}
        with pytest.raises(ValueError, match="out of range"):
            set_by_path(cfg, "a.5", 1, create=True)

    def test_set_list_index_out_of_range_raises(self):
        cfg = {"a": [1, 2]}
        with pytest.raises(ValueError, match="out of range"):
            set_by_path(cfg, "a.5", 1)

    def test_set_descend_into_non_container_raises_keyerror(self):
        cfg = {"a": 5}
        with pytest.raises(KeyError, match="cannot"):
            set_by_path(cfg, "a.b", 1)


# =============================================================================
# 2. ``dist:`` search-spec parsing
# =============================================================================
class TestIsSearchSpec:
    @pytest.mark.parametrize("obj", [
        {"dist": "uniform", "low": 0, "high": 1},
        {"dist": "fixed", "value": 1},
    ])
    def test_true_for_dicts_with_dist_key(self, obj):
        assert is_search_spec(obj) is True

    @pytest.mark.parametrize("obj", [
        {"type": "Adam", "params": {"lr": 1e-4}},
        {"low": 0, "high": 1},
        "dense",
        42,
        None,
        [1, 2],
    ])
    def test_false_otherwise(self, obj):
        assert is_search_spec(obj) is False


class TestParseSearchSpec:
    def test_uniform(self):
        d = parse_search_spec({"dist": "uniform", "low": 0.5, "high": 1.6})
        assert isinstance(d, Domain)
        assert all(0.5 <= d.sample() <= 1.6 for _ in range(50))

    def test_loguniform(self):
        d = parse_search_spec({"dist": "loguniform", "low": 1e-5, "high": 1e-3})
        assert isinstance(d, Domain)
        assert all(1e-5 <= d.sample() <= 1e-3 for _ in range(50))

    def test_quniform(self):
        d = parse_search_spec({"dist": "quniform", "low": 0, "high": 10, "q": 2})
        samples = {d.sample() for _ in range(200)}
        assert samples <= {0.0, 2.0, 4.0, 6.0, 8.0, 10.0}

    def test_randint_upper_is_exclusive(self):
        d = parse_search_spec({"dist": "randint", "lower": 1, "upper": 4})
        samples = {d.sample() for _ in range(500)}
        assert samples == {1, 2, 3}
        assert 4 not in samples

    def test_qrandint_upper_is_exclusive(self):
        d = parse_search_spec({"dist": "qrandint", "lower": 0, "upper": 10, "q": 2})
        samples = {d.sample() for _ in range(500)}
        assert 10 not in samples
        assert samples <= {0, 2, 4, 6, 8}

    def test_lograndint_upper_is_exclusive(self):
        d = parse_search_spec({"dist": "lograndint", "lower": 1, "upper": 100})
        samples = [d.sample() for _ in range(500)]
        assert max(samples) < 100
        assert min(samples) >= 1

    def test_lograndint_accepts_optional_base(self):
        d = parse_search_spec({"dist": "lograndint", "lower": 1, "upper": 100, "base": 2})
        assert isinstance(d, Domain)

    def test_choice(self):
        d = parse_search_spec({"dist": "choice", "values": ["a", "b", "c"]})
        samples = {d.sample() for _ in range(200)}
        assert samples <= {"a", "b", "c"}

    def test_grid_search_returns_plain_dict_not_domain(self):
        result = parse_search_spec({"dist": "grid_search", "values": [1, 2, 3]})
        assert result == {"grid_search": [1, 2, 3]}
        assert not isinstance(result, Domain)

    def test_randn_defaults(self):
        d = parse_search_spec({"dist": "randn"})
        assert isinstance(d, Domain)
        d.sample()  # must not raise

    def test_randn_explicit_mean_sd(self):
        d = parse_search_spec({"dist": "randn", "mean": 1.0, "sd": 0.1})
        assert isinstance(d, Domain)

    def test_fixed_returns_plain_value(self):
        assert parse_search_spec({"dist": "fixed", "value": 42}) == 42
        assert parse_search_spec({"dist": "fixed", "value": "orthogonal"}) == "orthogonal"

    def test_share_across_slots_is_stripped_and_never_fails_validation(self):
        d = parse_search_spec({"dist": "uniform", "low": 0, "high": 1, "share_across_slots": True})
        assert isinstance(d, Domain)

    def test_not_a_spec_raises(self):
        with pytest.raises(ValueError, match="Not a search spec"):
            parse_search_spec({"low": 0, "high": 1})

    def test_unknown_dist_raises_and_lists_supported(self):
        with pytest.raises(ValueError, match="Unknown dist"):
            parse_search_spec({"dist": "bogus"})

    def test_missing_required_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            parse_search_spec({"dist": "uniform", "low": 0})

    def test_unexpected_key_raises(self):
        with pytest.raises(ValueError, match="unexpected key"):
            parse_search_spec({"dist": "uniform", "low": 0, "high": 1, "q": 2})

    def test_choice_missing_values_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            parse_search_spec({"dist": "choice"})


# =============================================================================
# 3. ``build_search_space``
# =============================================================================
class TestBuildSearchSpace:
    def test_search_space_only(self):
        phase = {"search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
        }}
        space = build_search_space(phase)
        assert set(space) == {"agent.config.discount"}
        assert isinstance(space["agent.config.discount"], Domain)

    def test_non_spec_search_space_value_raises(self):
        phase = {"search_space": {"agent.config.discount": 0.99}}
        with pytest.raises(ValueError, match="must be a"):
            build_search_space(phase)

    def test_exact_key_set_with_search_space_architecture_and_optimizers(self):
        blocks = parse_block_library({
            "dense_block": {"layers": [
                {"type": "dense", "params": {
                    "units": {"dist": "choice", "values": [64, 128], "share_across_slots": True},
                }},
                {"type": "relu"},
            ]},
        })
        phase = {
            "search_space": {
                "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
            },
            "architecture": {
                "trunk": {
                    "depth": {"dist": "randint", "lower": 1, "upper": 3},  # max_depth = 2
                    "blocks": ["dense_block"],
                    "suffix": [
                        {"type": "flatten"},
                        {"type": "dense", "params": {"units": {"dist": "choice", "values": [32, 64]}}},
                    ],
                },
            },
            "optimizers": {
                "trunk": {"lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-3}},
            },
        }
        space = build_search_space(phase, blocks)
        expected = {
            "agent.config.discount",
            "arch.trunk.depth",
            "arch.trunk.slot0.block",
            "arch.trunk.slot1.block",
            "arch.trunk.dense_block.0.params.units",
            "arch.trunk.suffix.1.params.units",
            "opt.trunk.lr",
        }
        assert set(space) == expected
        assert isinstance(space["arch.trunk.depth"], Domain)
        assert isinstance(space["arch.trunk.slot0.block"], Domain)
        assert isinstance(space["opt.trunk.lr"], Domain)

    def test_architecture_without_blocks_library_raises(self):
        phase = {"architecture": {"trunk": {"depth": 1, "blocks": ["dense_block"]}}}
        with pytest.raises(ValueError, match="unknown block"):
            build_search_space(phase)  # blocks=None -> {}

    def test_fixed_depth_int_produces_no_depth_key_but_slot_keys(self):
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"architecture": {"trunk": {"depth": 2, "blocks": ["dense_block"]}}}
        space = build_search_space(phase, blocks)
        assert "arch.trunk.depth" not in space
        assert set(space) == {"arch.trunk.slot0.block", "arch.trunk.slot1.block"}

    def test_slots_enumerate_to_max_of_choice_dist(self):
        blocks = parse_block_library({"b": {"layers": [{"type": "dense", "params": {"units": 8}}]}})
        phase = {"architecture": {"trunk": {"depth": {"dist": "choice", "values": [1, 3, 2]}, "blocks": ["b"]}}}
        space = build_search_space(phase, blocks)
        slot_keys = {k for k in space if k.startswith("arch.trunk.slot")}
        assert slot_keys == {"arch.trunk.slot0.block", "arch.trunk.slot1.block", "arch.trunk.slot2.block"}

    def test_temporal_block_outside_trunk_rejected(self):
        blocks = parse_block_library({
            "gru_block": {"max_count": 1, "layers": [
                {"type": "gru", "params": {"hidden_size": 128, "num_layers": 1}},
            ]},
        })
        phase = {"architecture": {"branches.policy": {"depth": 1, "blocks": ["gru_block"]}}}
        with pytest.raises(ValueError, match="only legal for module 'trunk'"):
            build_search_space(phase, blocks)

    def test_temporal_block_in_trunk_is_allowed(self):
        blocks = parse_block_library({"gru_block": {"layers": [{"type": "gru", "params": {"hidden_size": 64}}]}})
        phase = {"architecture": {"trunk": {"depth": 1, "blocks": ["gru_block"]}}}
        space = build_search_space(phase, blocks)
        assert "arch.trunk.slot0.block" in space

    def test_optimizers_literal_field_is_not_added_to_space(self):
        phase = {"optimizers": {"trunk": {"lr": 3e-4, "type": "Adam"}}}
        space = build_search_space(phase)
        assert space == {}


# =============================================================================
# 4. Block library and ``build_layer_stack``
# =============================================================================
class TestParseBlockLibrary:
    def test_empty_or_none_returns_empty_dict(self):
        assert parse_block_library(None) == {}
        assert parse_block_library({}) == {}

    def test_missing_layers_key_raises(self):
        with pytest.raises(ValueError, match="'layers'"):
            parse_block_library({"b": {"max_count": 1}})

    def test_empty_layers_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            parse_block_library({"b": {"layers": []}})

    def test_layers_not_a_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            parse_block_library({"b": {"layers": {"type": "dense"}}})

    @pytest.mark.parametrize("layer_type,is_temporal", [
        ("lstm", True),
        ("gru", True),
        ("dense", False),
        ("relu", False),
    ])
    def test_is_temporal_for_plain_layer_types(self, layer_type, is_temporal):
        blocks = parse_block_library({"b": {"layers": [{"type": layer_type, "params": {
            "hidden_size": 8} if layer_type in ("lstm", "gru") else {}}]}})
        assert blocks["b"]["is_temporal"] is is_temporal

    def test_transformer_encoder_causal_true_is_temporal(self):
        blocks = parse_block_library({"b": {"layers": [
            {"type": "transformer_encoder", "params": {"d_model": 8, "causal": True}},
        ]}})
        assert blocks["b"]["is_temporal"] is True

    def test_transformer_encoder_causal_false_is_not_temporal(self):
        blocks = parse_block_library({"b": {"layers": [
            {"type": "transformer_encoder", "params": {"d_model": 8, "causal": False}},
        ]}})
        assert blocks["b"]["is_temporal"] is False

    def test_swept_type_choice_including_gru_is_conservatively_temporal(self):
        blocks = parse_block_library({"b": {"layers": [
            {"type": {"dist": "choice", "values": ["dense", "gru"]}, "params": {"hidden_size": 8}},
        ]}})
        assert blocks["b"]["is_temporal"] is True


class TestBuildLayerStack:
    def test_simple_stack_matches_block_layers(self):
        blocks = parse_block_library({"dense_block": {"layers": [
            {"type": "dense", "params": {"units": 64}}, {"type": "relu"},
        ]}})
        stack = build_layer_stack(
            "trunk", {"depth": 1, "blocks": ["dense_block"]},
            {"arch.trunk.slot0.block": "dense_block"}, blocks,
        )
        assert [l["type"] for l in stack] == ["dense", "relu"]
        assert stack[0]["params"]["units"] == 64

    def test_single_declared_block_needs_no_slot_choice_key(self):
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        stack = build_layer_stack("trunk", {"depth": 2, "blocks": ["dense_block"]}, {}, blocks)
        assert [l["type"] for l in stack] == ["dense", "dense"]

    def test_missing_slot_key_with_multiple_blocks_raises_keyerror(self):
        blocks = parse_block_library({
            "a": {"layers": [{"type": "dense", "params": {"units": 8}}]},
            "b": {"layers": [{"type": "relu"}]},
        })
        with pytest.raises(KeyError):
            build_layer_stack("trunk", {"depth": 1, "blocks": ["a", "b"]}, {}, blocks)

    def test_prefix_and_suffix_ordering_with_flatten(self):
        blocks = parse_block_library({"conv_block": {"layers": [
            {"type": "conv2d", "params": {"out_channels": 32, "kernel_size": 3}},
            {"type": "relu"},
        ]}})
        arch_spec = {
            "depth": 1, "blocks": ["conv_block"],
            "suffix": [{"type": "flatten"}, {"type": "dense", "params": {"units": 64}}, {"type": "relu"}],
        }
        stack = build_layer_stack("roots.camera", arch_spec, {"arch.roots.camera.slot0.block": "conv_block"}, blocks)
        assert [l["type"] for l in stack] == ["conv2d", "relu", "flatten", "dense", "relu"]

    def test_prefix_emitted_before_slots(self):
        blocks = parse_block_library({"b": {"layers": [{"type": "dense", "params": {"units": 8}}]}})
        arch_spec = {"depth": 1, "blocks": ["b"], "prefix": [{"type": "flatten"}]}
        stack = build_layer_stack("trunk", arch_spec, {"arch.trunk.slot0.block": "b"}, blocks)
        assert [l["type"] for l in stack] == ["flatten", "dense"]

    def test_flatten_not_auto_inserted(self):
        blocks = parse_block_library({"b": {"layers": [
            {"type": "conv2d", "params": {}}, {"type": "dense", "params": {"units": 8}}]}})
        stack = build_layer_stack("trunk", {"depth": 1, "blocks": ["b"]},
                                   {"arch.trunk.slot0.block": "b"}, blocks)
        assert [l["type"] for l in stack] == ["conv2d", "dense"]

    def test_share_across_slots_uses_slot_free_key(self):
        blocks = parse_block_library({"dense_block": {"layers": [
            {"type": "dense", "params": {"units": {"dist": "choice", "values": [64], "share_across_slots": True}}},
        ]}})
        sampled = {
            "arch.trunk.slot0.block": "dense_block",
            "arch.trunk.slot1.block": "dense_block",
            "arch.trunk.dense_block.0.params.units": 128,
        }
        stack = build_layer_stack("trunk", {"depth": 2, "blocks": ["dense_block"]}, sampled, blocks)
        assert [l["params"]["units"] for l in stack] == [128, 128]

    def test_max_count_fallback_matches_verified_example(self):
        """Plan-verified example: gru_block(max_count=1) + dense_block (unlimited),
        sampling gru, gru, dense resolves to gru, dense, dense."""
        blocks = parse_block_library({
            "gru_block": {"max_count": 1, "layers": [
                {"type": "gru", "params": {"hidden_size": 128, "num_layers": 1}},
            ]},
            "dense_block": {"layers": [{"type": "dense", "params": {"units": 64}}]},
        })
        arch_spec = {"depth": 3, "blocks": ["gru_block", "dense_block"]}
        sampled = {
            "arch.trunk.slot0.block": "gru_block",
            "arch.trunk.slot1.block": "gru_block",
            "arch.trunk.slot2.block": "dense_block",
        }
        stack = build_layer_stack("trunk", arch_spec, sampled, blocks)
        assert [l["type"] for l in stack] == ["gru", "dense", "dense"]

    def test_max_count_exhausted_raises(self):
        blocks = parse_block_library({
            "gru_block": {"max_count": 1, "layers": [{"type": "gru", "params": {"hidden_size": 8}}]},
        })
        arch_spec = {"depth": 2, "blocks": ["gru_block"]}
        sampled = {"arch.trunk.slot0.block": "gru_block", "arch.trunk.slot1.block": "gru_block"}
        with pytest.raises(ValueError, match="max_count"):
            build_layer_stack("trunk", arch_spec, sampled, blocks)

    def test_missing_sampled_layer_param_raises_keyerror(self):
        blocks = parse_block_library({"dense_block": {"layers": [
            {"type": "dense", "params": {"units": {"dist": "choice", "values": [32]}}},
        ]}})
        with pytest.raises(KeyError):
            build_layer_stack(
                "trunk", {"depth": 1, "blocks": ["dense_block"]},
                {"arch.trunk.slot0.block": "dense_block"}, blocks,
            )  # missing arch.trunk.slot0.dense_block.0.params.units

    def test_depth_from_sampled_key_overrides_literal(self):
        blocks = parse_block_library({"b": {"layers": [{"type": "dense", "params": {"units": 8}}]}})
        stack = build_layer_stack("trunk", {"depth": 99, "blocks": ["b"]},
                                   {"arch.trunk.depth": 2, "arch.trunk.slot0.block": "b",
                                    "arch.trunk.slot1.block": "b"}, blocks)
        assert len(stack) == 2

    def test_depth_missing_for_search_spec_raises_keyerror(self):
        blocks = parse_block_library({"b": {"layers": [{"type": "dense", "params": {"units": 8}}]}})
        with pytest.raises(KeyError):
            build_layer_stack("trunk", {"depth": {"dist": "fixed", "value": 1}, "blocks": ["b"]}, {}, blocks)

    def test_generated_layers_use_real_registry_param_names(self):
        """Cross-check against ``LAYER_REGISTRY`` (models.py): every generated
        layer dict must actually construct via the real ``build_layer``."""
        blocks = parse_block_library({
            "mixed_block": {"layers": [
                {"type": "conv2d", "params": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0}},
                {"type": "relu"},
                {"type": "flatten"},
                {"type": "dense", "params": {"units": 32}},
            ]},
        })
        stack = build_layer_stack(
            "roots.camera", {"depth": 1, "blocks": ["mixed_block"]},
            {"arch.roots.camera.slot0.block": "mixed_block"}, blocks,
        )
        for layer in stack:
            assert layer["type"] in models.LAYER_REGISTRY
            models.build_layer(layer["type"], layer.get("params"))  # must not raise

        gru_blocks = parse_block_library({"gru_block": {"layers": [
            {"type": "gru", "params": {"hidden_size": 64, "num_layers": 1}},
        ]}})
        gru_stack = build_layer_stack(
            "trunk", {"depth": 1, "blocks": ["gru_block"]},
            {"arch.trunk.slot0.block": "gru_block"}, gru_blocks,
        )
        assert gru_stack[0]["params"] == {"hidden_size": 64, "num_layers": 1}
        models.build_layer(gru_stack[0]["type"], gru_stack[0]["params"])


class TestApplyArchitecture:
    def test_writes_layer_config_for_existing_module(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"architecture": {"trunk": {"depth": 1, "blocks": ["dense_block"]}}}
        apply_architecture(config, phase, {"arch.trunk.slot0.block": "dense_block"}, blocks)
        assert config["agent"]["config"]["model"]["trunk"]["layer_config"] == [
            {"type": "dense", "params": {"units": 32}},
        ]

    def test_no_op_when_phase_has_no_architecture(self):
        config = {"agent": {"config": {"model": {}}}}
        apply_architecture(config, {}, {}, {})
        assert config == {"agent": {"config": {"model": {}}}}

    def test_missing_module_raises_clear_error(self):
        config = {"agent": {"config": {"model": {}}}}
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"architecture": {"roots.state": {"depth": 1, "blocks": ["dense_block"]}}}
        with pytest.raises(ValueError, match="does not exist in the base config"):
            apply_architecture(config, phase, {"arch.roots.state.slot0.block": "dense_block"}, blocks)

    def test_missing_layer_config_key_on_existing_module_raises(self):
        config = {"agent": {"config": {"model": {"trunk": {}}}}}
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"architecture": {"trunk": {"depth": 1, "blocks": ["dense_block"]}}}
        with pytest.raises(ValueError, match="does not exist"):
            apply_architecture(config, phase, {"arch.trunk.slot0.block": "dense_block"}, blocks)

    def test_temporal_block_for_non_trunk_module_rejected_as_config_error(self):
        """Must fail fast at validation, naming module and block -- not 30
        seconds later inside ModularModel's own temporal check."""
        config = {"agent": {"config": {"model": {"branches": {"value": {"layer_config": []}}}}}}
        blocks = parse_block_library({"gru_block": {"layers": [
            {"type": "gru", "params": {"hidden_size": 64}}]}})
        phase = {"architecture": {"branches.value": {"depth": 1, "blocks": ["gru_block"]}}}
        with pytest.raises(ValueError) as excinfo:
            apply_architecture(config, phase, {"arch.branches.value.slot0.block": "gru_block"}, blocks)
        message = str(excinfo.value)
        assert "branches.value" in message
        assert "gru_block" in message


# =============================================================================
# 5. ``apply_optimizers``
# =============================================================================
class TestApplyOptimizers:
    def test_creates_new_block_with_default_adam(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        apply_optimizers(config, {"optimizers": {"trunk": {"lr": 3e-4}}}, {})
        assert config["agent"]["config"]["model"]["trunk"]["optimizer_params"] == {
            "type": "Adam", "params": {"lr": 3e-4},
        }

    def test_explicit_literal_type_wins(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        apply_optimizers(config, {"optimizers": {"trunk": {"lr": 1e-4, "type": "RMSprop"}}}, {})
        assert config["agent"]["config"]["model"]["trunk"]["optimizer_params"]["type"] == "RMSprop"

    def test_swept_type_reads_from_sampled(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        phase = {"optimizers": {"trunk": {"lr": 1e-4, "type": {"dist": "choice", "values": ["Adam", "RMSprop"]}}}}
        apply_optimizers(config, phase, {"opt.trunk.type": "RMSprop"})
        assert config["agent"]["config"]["model"]["trunk"]["optimizer_params"]["type"] == "RMSprop"

    def test_unswept_type_inherits_existing_module_own_type(self):
        config = {"agent": {"config": {"model": {"trunk": {
            "layer_config": [], "optimizer_params": {"type": "SGD", "params": {"momentum": 0.9}},
        }}}}}
        apply_optimizers(config, {"optimizers": {"trunk": {"lr": 1e-4}}}, {})
        params_block = config["agent"]["config"]["model"]["trunk"]["optimizer_params"]
        assert params_block["type"] == "SGD"
        # merge, not replace: pre-existing 'momentum' survives alongside new 'lr'.
        assert params_block["params"] == {"momentum": 0.9, "lr": 1e-4}

    def test_unswept_type_falls_back_to_model_wide_type(self):
        config = {"agent": {"config": {"model": {
            "optimizer_params": {"type": "RMSprop"},
            "trunk": {"layer_config": []},
        }}}}
        apply_optimizers(config, {"optimizers": {"trunk": {"lr": 1e-4}}}, {})
        assert config["agent"]["config"]["model"]["trunk"]["optimizer_params"]["type"] == "RMSprop"

    def test_unswept_type_defaults_to_adam_when_nothing_else(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        apply_optimizers(config, {"optimizers": {"trunk": {"lr": 1e-4}}}, {})
        assert config["agent"]["config"]["model"]["trunk"]["optimizer_params"]["type"] == "Adam"

    def test_no_op_when_phase_has_no_optimizers(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        apply_optimizers(config, {}, {})
        assert "optimizer_params" not in config["agent"]["config"]["model"]["trunk"]

    def test_missing_module_raises(self):
        config = {"agent": {"config": {"model": {}}}}
        with pytest.raises(ValueError, match="does not exist"):
            apply_optimizers(config, {"optimizers": {"branches.policy": {"lr": 1e-4}}}, {})

    def test_fields_not_a_mapping_raises(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        with pytest.raises(ValueError, match="must be a mapping"):
            apply_optimizers(config, {"optimizers": {"trunk": 1e-4}}, {})

    def test_missing_sampled_key_for_swept_field_raises_keyerror(self):
        config = {"agent": {"config": {"model": {"trunk": {"layer_config": []}}}}}
        phase = {"optimizers": {"trunk": {"lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-3}}}}
        with pytest.raises(KeyError):
            apply_optimizers(config, phase, {})

    def test_two_modules_get_independent_blocks(self):
        config = {"agent": {"config": {"model": {
            "trunk": {"layer_config": []}, "branches": {"policy": {"layer_config": []}},
        }}}}
        phase = {"optimizers": {
            "trunk": {"lr": 1e-4}, "branches.policy": {"lr": 3e-4, "type": "RMSprop"},
        }}
        apply_optimizers(config, phase, {})
        model = config["agent"]["config"]["model"]
        assert model["trunk"]["optimizer_params"] == {"type": "Adam", "params": {"lr": 1e-4}}
        assert model["branches"]["policy"]["optimizer_params"] == {"type": "RMSprop", "params": {"lr": 3e-4}}


# =============================================================================
# 6. ``apply_auto_learn_every`` + ``validate_trial_config``
# =============================================================================
class TestApplyAutoLearnEvery:
    def _config(self, buffer_type="RolloutBuffer", learn_every_unit="timestep",
                buffer_size=8, num_envs=4):
        return {
            # `learn_every` must already exist: `apply_auto_learn_every` writes
            # through `set_by_path(..., create=False)`, mirroring every real
            # bundled config, which always declares `schedule.learn_every`
            # explicitly even when `auto_learn_every` will overwrite it.
            "schedule": {"learn_every_unit": learn_every_unit, "learn_every": 0},
            "buffer": {"type": buffer_type, "config": {"buffer_size": buffer_size}},
            "env": {"config": {"num_envs": num_envs}},
        }

    def test_computes_and_sets_learn_every(self):
        config = self._config(buffer_size=8, num_envs=4)
        result = apply_auto_learn_every(config)
        assert result == 32
        assert config["schedule"]["learn_every"] == 32

    def test_trajectory_buffer_also_applies(self):
        config = self._config(buffer_type="TrajectoryBuffer", buffer_size=6, num_envs=3)
        assert apply_auto_learn_every(config) == 18

    def test_none_for_off_policy_buffer(self):
        config = self._config(buffer_type="ReplayBuffer")
        assert apply_auto_learn_every(config) is None
        assert config["schedule"]["learn_every"] == 0  # untouched placeholder

    def test_none_when_learn_every_unit_is_episode(self):
        config = self._config(learn_every_unit="episode")
        assert apply_auto_learn_every(config) is None

    def test_none_when_buffer_size_or_num_envs_missing(self):
        config = {"schedule": {"learn_every_unit": "timestep"}, "buffer": {"type": "RolloutBuffer", "config": {}},
                   "env": {"config": {}}}
        assert apply_auto_learn_every(config) is None


class TestValidateTrialConfig:
    def _base(self, **overrides):
        config = {
            "schedule": {"learn_every_unit": "timestep", "learn_every": 16, "mini_batch_size": 8},
            "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
            "env": {"config": {"num_envs": 2}},
            "agent": {"config": {"model": {"trunk": {"layer_config": [{"type": "dense", "params": {"units": 8}}]}}}},
        }
        for path_str, value in overrides.items():
            set_by_path(config, path_str, value, create=True)
        return config

    def test_valid_feedforward_config_does_not_raise(self):
        validate_trial_config(self._base())

    def test_rollout_overflow_raises(self):
        config = self._base()
        config["schedule"]["learn_every"] = 40  # 40/2=20 iters > buffer_size=8
        with pytest.raises(ValueError, match="IndexError"):
            validate_trial_config(config)

    def test_rollout_underflow_warns_but_does_not_raise(self, monkeypatch):
        warnings_seen = []
        monkeypatch.setattr(ray_tune.logger, "warning", lambda *a, **k: warnings_seen.append((a, k)))
        config = self._base()
        config["schedule"]["learn_every"] = 8  # 8/2=4 iters < buffer_size=8
        validate_trial_config(config)  # must not raise
        assert warnings_seen, "expected a logged warning for partial rollouts"

    def test_feedforward_zero_batch_raises(self):
        config = self._base()
        config["schedule"]["mini_batch_size"] = 64  # > buffer_size(8)*num_envs(2)=16
        with pytest.raises(ValueError, match="zero gradient updates"):
            validate_trial_config(config)

    def test_feedforward_mini_batch_within_rollout_batch_ok(self):
        config = self._base()
        config["schedule"]["mini_batch_size"] = 16  # == rollout batch, not > it
        validate_trial_config(config)

    def _recurrent_config(self, mini_batch_size, num_envs=8):
        config = self._base()
        config["env"]["config"]["num_envs"] = num_envs
        config["agent"]["config"]["model"]["trunk"]["layer_config"] = [
            {"type": "gru", "params": {"hidden_size": 32}},
        ]
        config["schedule"]["mini_batch_size"] = mini_batch_size
        config["schedule"]["learn_every"] = num_envs * 8  # keep rollout capacity check happy
        return config

    def test_recurrent_mini_batch_not_dividing_num_envs_raises(self):
        with pytest.raises(ValueError, match="evenly divide"):
            validate_trial_config(self._recurrent_config(mini_batch_size=3, num_envs=8))

    def test_recurrent_mini_batch_dividing_num_envs_ok(self):
        validate_trial_config(self._recurrent_config(mini_batch_size=4, num_envs=8))

    def test_recurrent_mini_batch_exceeding_num_envs_raises(self):
        with pytest.raises(ValueError, match="evenly divide"):
            validate_trial_config(self._recurrent_config(mini_batch_size=16, num_envs=8))

    def test_recurrent_negative_mini_batch_raises(self):
        with pytest.raises(ValueError, match="evenly divide"):
            validate_trial_config(self._recurrent_config(mini_batch_size=-3, num_envs=8))

    def test_recurrent_zero_mini_batch_raises(self):
        with pytest.raises(ValueError, match="evenly divide"):
            validate_trial_config(self._recurrent_config(mini_batch_size=0, num_envs=8))

    def test_feedforward_zero_mini_batch_raises(self):
        """A sampled `mini_batch_size: 0` (Ray's `randint` lower bound is
        inclusive, so `{dist: randint, lower: 0, upper: 64}` reaches 0
        easily) must be rejected here, before it reaches rl_agents.py's
        `num_valid // mini_batch_size`, which would raise a bare
        ZeroDivisionError mid-training instead of this actionable message."""
        config = self._base()
        config["schedule"]["mini_batch_size"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            validate_trial_config(config)

    def test_recurrent_skips_feedforward_zero_batch_check(self):
        """A recurrent trunk's mini_batch_size is env-units, so even a value
        that would blow the feedforward rollout-batch check is fine as long
        as it divides num_envs."""
        config = self._recurrent_config(mini_batch_size=8, num_envs=8)  # == num_envs, divides evenly
        validate_trial_config(config)  # must not raise

    def test_sac_n_triple_mismatch_raises(self):
        config = self._base(**{"agent.config.N": 3})
        set_by_path(config, "buffer.config.N", 5, create=True)
        with pytest.raises(ValueError, match="N-step mismatch"):
            validate_trial_config(config)

    def test_sac_n_triple_agreement_does_not_raise(self):
        config = self._base(**{"agent.config.N": 3})
        set_by_path(config, "buffer.config.N", 3, create=True)
        set_by_path(config, "env.config.wrappers", [{"type": "VectorNStepReward", "params": {"n": 3}}], create=True)
        validate_trial_config(config)

    def test_sac_n_wrapper_mismatch_raises(self):
        config = self._base(**{"agent.config.N": 3})
        set_by_path(config, "env.config.wrappers", [{"type": "VectorNStepReward", "params": {"n": 5}}], create=True)
        with pytest.raises(ValueError, match="N-step mismatch"):
            validate_trial_config(config)

    def test_single_n_value_present_never_mismatches(self):
        config = self._base(**{"agent.config.N": 3})
        validate_trial_config(config)  # only one of the three present -> trivially consistent

    def test_no_n_values_present_never_mismatches(self):
        validate_trial_config({})


# =============================================================================
# 7. Searcher / scheduler / stopper factories
# =============================================================================
class TestBuildSearchAlg:
    def test_default_is_random(self):
        alg = build_search_alg({})
        assert isinstance(alg, BasicVariantGenerator)

    def test_explicit_random_type(self):
        alg = build_search_alg({"tune": {"search_alg": {"type": "random"}}})
        assert isinstance(alg, BasicVariantGenerator)

    def test_random_with_points_to_evaluate(self):
        points = [{"agent.config.discount": 0.97}]
        alg = build_search_alg(
            {"tune": {"search_alg": {"type": "random"}}},
            points_to_evaluate=points,
        )
        assert isinstance(alg, BasicVariantGenerator)
        # The point of this test is the seeding kwarg; assert it actually
        # landed on the constructed searcher, not just that a searcher
        # of the right type came back.
        assert alg._points_to_evaluate == points

    def test_optuna_type_builds_optuna_searcher(self):
        alg = build_search_alg({"tune": {"search_alg": {"type": "optuna"}}})
        assert type(alg).__name__ == "OptunaSearch"

    def test_hyperopt_type_builds_hyperopt_searcher(self):
        alg = build_search_alg({"tune": {"search_alg": {"type": "hyperopt"}}})
        assert type(alg).__name__ == "HyperOptSearch"

    def test_unsupported_points_to_evaluate_type_warns_and_skips_seeding(self, monkeypatch):
        warnings_seen = []
        monkeypatch.setattr(ray_tune.logger, "warning", lambda *a, **k: warnings_seen.append((a, k)))
        # 'variant_generator' is a real create_searcher alias but is NOT in
        # _POINTS_TO_EVALUATE_SUPPORTED, so seeding must be skipped, not crash.
        alg = build_search_alg(
            {"tune": {"search_alg": {"type": "variant_generator"}}},
            points_to_evaluate=[{"x": 1}],
        )
        # `alg is not None` cannot fail: build_search_alg either returns an
        # instance or raises. Assert the concrete type it actually falls
        # back to instead.
        assert isinstance(alg, BasicVariantGenerator)
        assert warnings_seen

    def test_non_mapping_search_alg_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            build_search_alg({"tune": {"search_alg": "random"}})


class TestBuildScheduler:
    def test_absent_returns_none(self):
        assert build_scheduler({"tune": {}}) is None
        assert build_scheduler({}) is None

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            build_scheduler({"tune": {"scheduler": "asha"}})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required key 'type'"):
            build_scheduler({"tune": {"scheduler": {}}})

    def test_asha_alias_builds_async_hyperband(self):
        sched = build_scheduler({"tune": {"scheduler": {
            "type": "asha", "time_attr": "timestep", "grace_period": 100,
        }}})
        assert isinstance(sched, AsyncHyperBandScheduler)

    def test_fifo_scheduler(self):
        sched = build_scheduler({"tune": {"scheduler": {"type": "fifo"}}})
        assert isinstance(sched, FIFOScheduler)

    @pytest.mark.parametrize("pbt_name", ["pbt", "population_based_training", "PBT"])
    def test_pbt_types_are_rejected(self, pbt_name):
        with pytest.raises(ValueError, match="not supported yet"):
            build_scheduler({"tune": {"scheduler": {"type": pbt_name}}})


class TestBuildStopper:
    def test_absent_returns_none(self):
        assert build_stopper({"tune": {}}) is None

    def test_lone_plain_metric_dict_returned_as_is(self):
        result = build_stopper({"tune": {"stop": {"avg_reward": 250}}})
        assert result == {"avg_reward": 250}

    def test_single_named_stopper(self):
        result = build_stopper({"tune": {"stop": {
            "type": "TrialPlateauStopper", "metric": "avg_reward", "std": 2.0, "num_results": 8,
        }}})
        assert isinstance(result, TrialPlateauStopper)

    def test_list_combines_via_combined_stopper(self):
        result = build_stopper({"tune": {"stop": [
            {"avg_reward": 250},
            {"type": "MaximumIterationStopper", "max_iter": 100},
        ]}})
        assert isinstance(result, CombinedStopper)

    def test_unknown_stopper_name_raises_and_lists_valid(self):
        with pytest.raises(ValueError, match="Unknown stopper type"):
            build_stopper({"tune": {"stop": {"type": "NotAStopper"}}})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_stopper({"tune": {"stop": []}})

    def test_non_mapping_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a mapping or list"):
            build_stopper({"tune": {"stop": "avg_reward"}})

    def test_list_item_not_a_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            build_stopper({"tune": {"stop": [42]}})


class TestSampleSearchSpace:
    def test_domain_grid_and_constant_all_handled(self):
        from ray import tune
        space = {
            "a": tune.uniform(0.0, 1.0),
            "b": {"grid_search": [1, 2, 3]},
            "c": 42,
        }
        sampled = sample_search_space(space, seed=0)
        assert set(sampled) == {"a", "b", "c"}
        assert 0.0 <= sampled["a"] <= 1.0
        assert sampled["b"] == 1
        assert sampled["c"] == 42

    def test_seeded_sampling_is_reproducible(self):
        from ray import tune
        space = {"a": tune.uniform(0.0, 1.0)}
        s1 = sample_search_space(space, seed=5)
        s2 = sample_search_space(space, seed=5)
        assert s1 == s2


# =============================================================================
# 8. Sweep-schema validation + ``normalize_phases``
# =============================================================================
class TestNormalizePhases:
    def test_defaults_not_a_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            normalize_phases({"defaults": "oops", "phases": [{"name": "p"}]})

    def test_phases_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            normalize_phases({"phases": []})

    def test_phases_not_a_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            normalize_phases({"phases": {"name": "p"}})

    def test_phase_not_a_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            normalize_phases({"phases": ["not-a-dict"]})

    @pytest.mark.parametrize("name", [None, "", 42])
    def test_phase_missing_or_invalid_name_raises(self, name):
        with pytest.raises(ValueError, match="non-empty string 'name'"):
            normalize_phases({"phases": [{"name": name}]})

    def test_duplicate_phase_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate phase name"):
            normalize_phases({"phases": [{"name": "p"}, {"name": "p"}]})

    def test_defaults_inherited_key_by_key_phase_wins(self):
        sweep = {
            "defaults": {"metric": "avg_reward", "mode": "max", "max_concurrent_trials": 4},
            "phases": [{"name": "p1", "mode": "min"}],
        }
        phases = normalize_phases(sweep)
        assert phases[0]["metric"] == "avg_reward"
        assert phases[0]["mode"] == "min"
        assert phases[0]["max_concurrent_trials"] == 4

    def test_implicit_single_phase_from_top_level_search_space(self):
        sweep = {"search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        phases = normalize_phases(sweep)
        assert [p["name"] for p in phases] == ["phase_0"]
        assert "agent.config.discount" in phases[0]["search_space"]

    def test_implicit_phase_inherits_defaults(self):
        sweep = {
            "defaults": {"metric": "avg_reward"},
            "tune": {"num_samples": 5},
        }
        phases = normalize_phases(sweep)
        assert phases[0]["metric"] == "avg_reward"
        assert phases[0]["tune"]["num_samples"] == 5

    def test_no_phases_and_no_implicit_key_raises(self):
        with pytest.raises(ValueError, match="nothing to sweep"):
            normalize_phases({"base_config": "x.yml"})

    def test_defaults_deep_copied_mutating_one_phase_does_not_leak(self):
        """Each phase's inherited `defaults` must be its own copy: mutating
        a nested dict inside one returned phase must not affect another
        phase's copy, nor the original `sweep['defaults']` dict."""
        sweep = {
            "defaults": {
                "resources": {"cpu": 1, "gpu": 0},
                "report": {"every": 50000, "unit": "timestep"},
            },
            "phases": [{"name": "p1"}, {"name": "p2"}],
        }
        original_defaults = copy.deepcopy(sweep["defaults"])
        phases = normalize_phases(sweep)

        phases[0]["resources"]["cpu"] = 999
        phases[0]["report"]["every"] = 1

        assert phases[1]["resources"]["cpu"] == 1
        assert phases[1]["report"]["every"] == 50000
        assert sweep["defaults"] == original_defaults


class TestValidateSweepConfig:
    def _valid_sweep(self, **overrides):
        sweep = {
            "base_config": "LunarLanderContinuous-v3/ppo.yml",
            "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
        }
        sweep.update(overrides)
        return sweep

    def test_valid_sweep_does_not_raise(self):
        validate_sweep_config(self._valid_sweep())

    def test_non_mapping_sweep_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_sweep_config([])

    def test_missing_base_config_raises(self):
        with pytest.raises(ValueError, match="base_config"):
            validate_sweep_config({"search_space": {}})

    def test_non_string_base_config_raises(self):
        with pytest.raises(ValueError, match="base_config"):
            validate_sweep_config(self._valid_sweep(base_config=123))

    def test_non_mapping_overrides_raises(self):
        with pytest.raises(ValueError, match="overrides"):
            validate_sweep_config(self._valid_sweep(overrides="x"))

    def test_invalid_blocks_library_propagates(self):
        with pytest.raises(ValueError, match="'layers'"):
            validate_sweep_config(self._valid_sweep(blocks={"b": {}}))

    def test_phase_tune_not_a_mapping_raises(self):
        sweep = self._valid_sweep()
        sweep["tune"] = "oops"
        with pytest.raises(ValueError, match="'tune' must be a mapping"):
            validate_sweep_config(sweep)

    def test_phase_optimizers_field_not_a_mapping_raises(self):
        sweep = self._valid_sweep()
        sweep["optimizers"] = {"trunk": "not-a-dict"}
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_sweep_config(sweep)

    def test_phase_optimizer_invalid_search_spec_raises(self):
        sweep = self._valid_sweep()
        sweep["optimizers"] = {"trunk": {"lr": {"dist": "bogus"}}}
        with pytest.raises(ValueError):
            validate_sweep_config(sweep)

    def test_invalid_search_space_is_wrapped_with_phase_name(self):
        sweep = self._valid_sweep()
        sweep["search_space"] = {"agent.config.discount": {"dist": "bogus"}}
        with pytest.raises(ValueError, match="invalid search space"):
            validate_sweep_config(sweep)

    def _sweep_with_promote(self, promote_cfg):
        """A sweep using the explicit 'phases' list (unlike `_valid_sweep`'s
        implicit single phase, `normalize_phases` never carries a top-level
        'promote' key into the implicit phase, so 'promote' coverage needs
        an explicit phase)."""
        return {
            "base_config": "LunarLanderContinuous-v3/ppo.yml",
            "phases": [{
                "name": "p1",
                "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
                "promote": promote_cfg,
            }],
        }

    def test_promote_not_a_mapping_raises(self):
        with pytest.raises(ValueError, match="'promote' must be a mapping"):
            validate_sweep_config(self._sweep_with_promote("best"))

    def test_promote_unsupported_mode_raises(self):
        """A capitalization typo like `mode: 'Best'` must be loud, not a
        silent no-op that trains the next phase against the un-tuned base
        config while looking entirely successful."""
        with pytest.raises(ValueError, match="not supported"):
            validate_sweep_config(self._sweep_with_promote({"mode": "Best"}))

    def test_promote_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown key"):
            validate_sweep_config(self._sweep_with_promote({"mode": "best", "bogus": 1}))

    def test_promote_seed_next_without_mode_best_raises(self):
        """Seeding the next phase's searcher without also promoting the
        winning trial's config is a confusing half-applied state."""
        with pytest.raises(ValueError, match="seed_next"):
            validate_sweep_config(self._sweep_with_promote({"seed_next": 3}))

    def test_promote_valid_mode_best_with_seed_next_does_not_raise(self):
        validate_sweep_config(self._sweep_with_promote({"mode": "best", "seed_next": 3}))

    def test_promote_absent_does_not_raise(self):
        sweep = self._sweep_with_promote({"mode": "best"})
        del sweep["phases"][0]["promote"]
        validate_sweep_config(sweep)  # must not raise


# =============================================================================
# 9. ``resolve_trial_config`` pipeline
# =============================================================================
class TestResolveTrialConfig:
    def _base_config(self):
        return {
            "schedule": {"learn_every_unit": "timestep", "learn_every": 16, "mini_batch_size": 8},
            "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
            "env": {"config": {"num_envs": 2}},
            "agent": {"config": {"discount": 0.9, "model": {
                "trunk": {"layer_config": [{"type": "dense", "params": {"units": 8}}]},
            }}},
        }

    def test_applies_overrides_search_space_and_returns_new_config(self):
        sweep = {"base_config": "x.yml", "overrides": {"agent.config.discount": 0.5}}
        phase = {"search_space": {"schedule.mini_batch_size": {"dist": "fixed", "value": 8}}}
        resolved = resolve_trial_config(sweep, phase, {"schedule.mini_batch_size": 8}, self._base_config())
        assert resolved["agent"]["config"]["discount"] == 0.5
        assert resolved["schedule"]["mini_batch_size"] == 8

    def test_does_not_mutate_base_config(self):
        base = self._base_config()
        base_copy = copy.deepcopy(base)
        sweep = {"base_config": "x.yml"}
        phase = {"search_space": {"agent.config.discount": {"dist": "fixed", "value": 0.5}}}
        resolve_trial_config(sweep, phase, {"agent.config.discount": 0.5}, base)
        assert base == base_copy

    def test_result_is_independent_of_base_after_return(self):
        base = self._base_config()
        sweep = {"base_config": "x.yml"}
        resolved = resolve_trial_config(sweep, {}, {}, base)
        resolved["schedule"]["mini_batch_size"] = -1
        assert base["schedule"]["mini_batch_size"] != -1

    def test_missing_sampled_value_for_search_space_key_raises(self):
        sweep = {"base_config": "x.yml"}
        phase = {"search_space": {"agent.config.discount": {"dist": "fixed", "value": 0.5}}}
        with pytest.raises(KeyError, match="missing sampled value"):
            resolve_trial_config(sweep, phase, {}, self._base_config())

    def test_typo_in_search_space_path_raises_instead_of_silent_no_op(self):
        sweep = {"base_config": "x.yml"}
        phase = {"search_space": {"agent.config.discont": {"dist": "fixed", "value": 0.5}}}  # typo
        with pytest.raises(KeyError):
            resolve_trial_config(sweep, phase, {"agent.config.discont": 0.5}, self._base_config())

    def test_runs_architecture_then_optimizers_then_constraints(self):
        sweep = {"base_config": "x.yml", "blocks": {
            "dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]},
        }}
        phase = {
            "architecture": {"trunk": {"depth": 1, "blocks": ["dense_block"]}},
            "optimizers": {"trunk": {"lr": 1e-4}},
        }
        sampled = {"arch.trunk.slot0.block": "dense_block"}
        resolved = resolve_trial_config(sweep, phase, sampled, self._base_config())
        model = resolved["agent"]["config"]["model"]
        assert model["trunk"]["layer_config"] == [{"type": "dense", "params": {"units": 32}}]
        assert model["trunk"]["optimizer_params"] == {"type": "Adam", "params": {"lr": 1e-4}}

    def test_constraint_violation_raises(self):
        sweep = {"base_config": "x.yml"}
        phase = {"search_space": {"schedule.mini_batch_size": {"dist": "fixed", "value": 999}}}
        with pytest.raises(ValueError, match="zero gradient updates"):
            resolve_trial_config(sweep, phase, {"schedule.mini_batch_size": 999}, self._base_config())

    def test_auto_learn_every_defaults_to_true(self):
        sweep = {"base_config": "x.yml"}
        base = self._base_config()
        base["schedule"]["learn_every"] = 0  # placeholder; auto_learn_every overwrites it
        resolved = resolve_trial_config(sweep, {}, {}, base)
        assert resolved["schedule"]["learn_every"] == 16  # buffer_size(8) * num_envs(2)

    def test_auto_learn_every_disabled_via_phase_flag(self):
        sweep = {"base_config": "x.yml"}
        base = self._base_config()
        # A valid-but-different-from-the-auto-computed-value(16) learn_every,
        # so leaving it untouched is distinguishable from recomputing it.
        base["schedule"]["learn_every"] = 8
        resolved = resolve_trial_config(sweep, {"auto_learn_every": False}, {}, base)
        assert resolved["schedule"]["learn_every"] == 8  # left untouched

    def test_does_not_inject_save_dir_or_callbacks(self):
        base = self._base_config()
        sweep = {"base_config": "x.yml"}
        resolved = resolve_trial_config(sweep, {}, {}, base)
        assert "save_dir" not in resolved
        assert "callbacks" not in resolved


# =============================================================================
# 10. ``write_best_config`` round trip
# =============================================================================
class TestWriteBestConfigRoundTrip:
    def test_strips_save_dir_and_raytune_callback(self, tmp_path):
        config = {
            "save_dir": "/tmp/trial_00007",
            "agent": {"config": {"discount": 0.99}},
            "callbacks": [
                {"type": "RayTuneCallback", "config": {"every": 50000, "unit": "timestep"}},
                {"type": "WandbCallback", "config": {"project_name": "p"}},
            ],
        }
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)
        loaded = load_config(dest)
        assert "save_dir" not in loaded
        assert [cb["type"] for cb in loaded["callbacks"]] == ["WandbCallback"]

    def test_drops_callbacks_key_entirely_when_only_raytune_present(self, tmp_path):
        config = {"callbacks": [{"type": "RayTuneCallback", "config": {"every": 1, "unit": "timestep"}}]}
        dest = tmp_path / "best.yml"
        write_best_config(config, dest)
        loaded = load_config(dest)
        assert "callbacks" not in loaded

    def test_numpy_scalars_are_converted_before_dump(self, tmp_path):
        import numpy as np
        config = {"agent": {"config": {"discount": np.float64(0.97), "n": np.int64(3)}}}
        dest = tmp_path / "best.yml"
        write_best_config(config, dest)  # must not raise on yaml.safe_dump
        loaded = load_config(dest)
        assert loaded["agent"]["config"]["discount"] == pytest.approx(0.97)
        assert loaded["agent"]["config"]["n"] == 3

    def test_round_trip_produces_a_buildable_trainer(self, tmp_path, force_cpu):
        config = _tiny_ppo_config(str(tmp_path / "run") + os.sep)
        config["callbacks"] = [{"type": "RayTuneCallback", "config": {"every": 8, "unit": "timestep"}}]
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)

        loaded = load_config(dest)
        assert "save_dir" not in loaded  # every save_dir is stripped, trial or not
        assert "callbacks" not in loaded  # RayTuneCallback was the only callback

        # A real `phoenx-train --config best_config.yml` run supplies its own
        # save_dir (via the CLI or the config file itself); build_trainer_from_config
        # falls back to 'models/' when absent, so give it a tmp_path-scoped one
        # here to avoid writing into the repo's working directory.
        loaded["save_dir"] = str(tmp_path / "run2") + os.sep
        trainer = build_trainer_from_config(loaded)
        try:
            assert trainer is not None
        finally:
            trainer.env.close()


# =============================================================================
# 11. Bundled example sweeps
# =============================================================================
class TestBundledExampleSweeps:
    def test_available_example_sweeps_returns_both_bundled_names(self):
        names = available_example_sweeps()
        assert names == sorted(names)
        assert set(names) == {"isaac_franka_cube_lift.yml", "lunarlander_ppo.yml"}

    @pytest.mark.parametrize("name", ["lunarlander_ppo.yml", "isaac_franka_cube_lift.yml"])
    def test_bundled_sweep_loads_and_validates(self, name):
        sweep = load_sweep_config(name)
        validate_sweep_config(sweep)  # must not raise

    @pytest.mark.parametrize("name", ["lunarlander_ppo.yml", "isaac_franka_cube_lift.yml"])
    def test_bundled_sweep_resolves_one_sampled_trial_per_phase(self, name):
        """Pure dict resolution -- the Isaac sweep's base_config is loaded via
        plain YAML parsing (``load_config``), never booting Isaac Sim."""
        sweep = load_sweep_config(name)
        validate_sweep_config(sweep)
        blocks = parse_block_library(sweep.get("blocks"))
        phases = normalize_phases(sweep)
        assert phases, "expected at least one phase"
        for phase in phases:
            space = build_search_space(phase, blocks)
            sampled = sample_search_space(space, seed=0)
            resolved = resolve_trial_config(sweep, phase, sampled)  # base_config=None -> loads from disk
            assert isinstance(resolved, dict)
            assert resolved.get("agent", {}).get("config") is not None

    def test_load_sweep_config_unknown_name_raises_and_lists_bundled(self):
        with pytest.raises(FileNotFoundError, match="lunarlander_ppo.yml"):
            load_sweep_config("does_not_exist.yml")

    @pytest.mark.parametrize("name", ["lunarlander_ppo.yml", "isaac_franka_cube_lift.yml"])
    def test_bundled_sweep_every_phase_tune_config_is_fully_constructible(self, name):
        """The regression this whole class was missing: both bundled sweeps
        once declared an ASHA ``scheduler.grace_period`` with no explicit
        ``max_t``, so ``AsyncHyperBandScheduler``'s default ``max_t=100``
        made ``build_scheduler`` raise ``AssertionError: grace_period must
        be <= max_t!`` for every phase using it -- a real
        ``phoenx-tune --config <name>`` run would have crashed on phase
        one. Neither ``validate_sweep_config`` nor ``resolve_trial_config``
        ever construct a searcher/scheduler/stopper, so this went
        undetected. Iterate every phase via ``normalize_phases``, not just
        the first."""
        sweep = load_sweep_config(name)
        validate_sweep_config(sweep)
        phases = normalize_phases(sweep)
        assert phases, f"{name}: no phases resolved"
        for phase in phases:
            build_search_alg(phase)  # must not raise
            build_scheduler(phase)  # must not raise
            build_stopper(phase)  # must not raise


# =============================================================================
# 12. Phase promotion and ``points_to_evaluate`` seeding
# =============================================================================
class TestFilterSeedPoints:
    def test_search_space_keys_carry_over_when_present_in_next_phase(self):
        phase = {"name": "p1", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = {"name": "p2", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        top_k = [{"agent.config.discount": 0.95}]
        result = ray_tune._filter_seed_points(top_k, phase, next_phase, {})
        assert result == [{"agent.config.discount": 0.95}]

    def test_keys_not_in_next_phase_space_are_dropped(self):
        phase = {"name": "p1", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = {"name": "p2", "search_space": {"agent.config.policy_clip": {"dist": "uniform", "low": 0.1, "high": 0.3}}}
        top_k = [{"agent.config.discount": 0.95}]
        result = ray_tune._filter_seed_points(top_k, phase, next_phase, {})
        assert result is None  # every point ended up empty after filtering

    def test_arch_keys_carry_over_when_architecture_sections_identical(self):
        arch = {"trunk": {"depth": {"dist": "randint", "lower": 1, "upper": 3}, "blocks": ["dense_block"]}}
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"name": "p1", "architecture": arch}
        next_phase = {"name": "p2", "architecture": copy.deepcopy(arch)}
        top_k = [{"arch.trunk.depth": 2, "arch.trunk.slot0.block": "dense_block"}]
        result = ray_tune._filter_seed_points(top_k, phase, next_phase, blocks)
        assert result == top_k

    def test_arch_keys_dropped_when_architecture_sections_differ(self, monkeypatch):
        infos_seen = []
        monkeypatch.setattr(ray_tune.logger, "info", lambda *a, **k: infos_seen.append((a, k)))
        blocks = parse_block_library({"dense_block": {"layers": [{"type": "dense", "params": {"units": 32}}]}})
        phase = {"name": "p1", "architecture": {"trunk": {"depth": 1, "blocks": ["dense_block"]}}}
        next_phase = {"name": "p2", "architecture": {"trunk": {"depth": 2, "blocks": ["dense_block"]}}}
        top_k = [{"arch.trunk.slot0.block": "dense_block"}]
        result = ray_tune._filter_seed_points(top_k, phase, next_phase, blocks)
        assert result is None
        assert infos_seen, "expected a log line explaining the skipped arch.* seeding"

    def test_mixed_point_keeps_only_carried_over_keys(self):
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
        }}
        next_phase = {"name": "p2", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
        }}
        top_k = [{"agent.config.discount": 0.95, "opt.trunk.lr": 1e-4}]
        result = ray_tune._filter_seed_points(top_k, phase, next_phase, {})
        assert result == [{"agent.config.discount": 0.95}]

    def test_empty_top_k_returns_none(self):
        result = ray_tune._filter_seed_points([], {"name": "p1"}, {"name": "p2"}, {})
        assert result is None


class TestRunSweepPromotionAndSeeding:
    """Drives ``run_sweep`` end to end but with ``run_phase`` monkeypatched to
    a canned, non-Ray fake, so promotion / seeding / ``from_phase`` /
    ``write_best_config`` are all exercised without a Ray cluster."""

    BASE_CONFIG = {
        "schedule": {"stop_units": 1000},
        "agent": {"config": {"discount": 0.9}},
        "env": {"config": {"num_envs": 4}},
    }

    def _fake_load_config(self, real_load_config):
        def _loader(path):
            if str(path) == "fake_base.yml":
                return copy.deepcopy(self.BASE_CONFIG)
            return real_load_config(path)
        return _loader

    def test_promotes_winner_and_forwards_filtered_seed_points(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))

        calls = []

        def fake_run_phase(sweep, phase, base_config, *, storage_path, sweep_name, points_to_evaluate=None, **kwargs):
            calls.append({
                "phase": phase["name"],
                "base_discount": base_config["agent"]["config"]["discount"],
                "points_to_evaluate": points_to_evaluate,
            })
            winner_discount = 0.95 if phase["name"] == "p1" else 0.90
            best_config = copy.deepcopy(base_config)
            best_config["agent"]["config"]["discount"] = winner_discount
            phase_dir = tmp_path / sweep_name / phase["name"]
            phase_dir.mkdir(parents=True, exist_ok=True)
            return {
                "phase": phase["name"], "metric": "avg_reward", "mode": "max",
                "num_trials": 2, "num_errors": 0, "best_metric_value": 1.0,
                "best_sampled": {"agent.config.discount": winner_discount},
                "best_config": best_config,
                "top_k_sampled": [
                    {"agent.config.discount": winner_discount},
                    {"agent.config.discount": winner_discount - 0.01},
                ],
                "phase_dir": str(phase_dir),
            }

        monkeypatch.setattr(ray_tune, "run_phase", fake_run_phase)

        sweep = {
            "base_config": "fake_base.yml",
            "phases": [
                {"name": "p1", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
                 "promote": {"mode": "best", "seed_next": 2}},
                {"name": "p2", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
                 "promote": {"mode": "best"}},
            ],
        }

        result = run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e")

        assert [c["phase"] for c in calls] == ["p1", "p2"]
        # p1 starts from the sweep's own base config.
        assert calls[0]["base_discount"] == 0.9
        assert calls[0]["points_to_evaluate"] is None
        # p2's base config is p1's PROMOTED winner (0.95), not the original base.
        assert calls[1]["base_discount"] == 0.95
        # p2 is seeded with p1's filtered top-2 sampled points.
        assert calls[1]["points_to_evaluate"] == [
            {"agent.config.discount": 0.95}, {"agent.config.discount": 0.94},
        ]

        # Final result reflects p2 (the last phase run)'s promoted winner.
        assert result["final_config"]["agent"]["config"]["discount"] == 0.90
        assert result["phases"]["p1"]["best_config"]["agent"]["config"]["discount"] == 0.95
        final_path = tmp_path / "e2e" / "best_config.yml"
        assert result["final_config_path"] == str(final_path)
        assert final_path.is_file()
        assert load_config(final_path)["agent"]["config"]["discount"] == 0.90

    def test_from_phase_skips_earlier_phases_and_loads_promoted_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))

        # Pre-write phase p1's promoted best_config.yml, as if it already ran.
        p1_dir = tmp_path / "e2e" / "p1"
        p1_best = copy.deepcopy(self.BASE_CONFIG)
        p1_best["agent"]["config"]["discount"] = 0.95
        write_best_config(p1_best, p1_dir / "best_config.yml")

        calls = []

        def fake_run_phase(sweep, phase, base_config, *, storage_path, sweep_name, points_to_evaluate=None, **kwargs):
            calls.append(phase["name"])
            best_config = copy.deepcopy(base_config)
            phase_dir = tmp_path / sweep_name / phase["name"]
            phase_dir.mkdir(parents=True, exist_ok=True)
            return {
                "phase": phase["name"], "metric": "avg_reward", "mode": "max",
                "num_trials": 1, "num_errors": 0, "best_metric_value": 1.0,
                "best_sampled": {}, "best_config": best_config, "top_k_sampled": [{}],
                "phase_dir": str(phase_dir),
            }

        monkeypatch.setattr(ray_tune, "run_phase", fake_run_phase)

        sweep = {
            "base_config": "fake_base.yml",
            "phases": [
                {"name": "p1", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
                {"name": "p2", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
            ],
        }

        run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e", from_phase="p2")

        assert calls == ["p2"]  # p1 was skipped entirely

    def test_from_phase_unknown_name_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))
        sweep = {
            "base_config": "fake_base.yml",
            "phases": [{"name": "p1", "search_space": {
                "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}],
        }
        with pytest.raises(ValueError, match="not one of this sweep's phases"):
            run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e", from_phase="bogus")

    def test_from_phase_missing_preceding_best_config_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))
        sweep = {
            "base_config": "fake_base.yml",
            "phases": [
                {"name": "p1", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
                {"name": "p2", "search_space": {"agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
            ],
        }
        with pytest.raises(FileNotFoundError):
            run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e", from_phase="p2")

    def test_preflight_resolution_failure_raises_before_any_run_phase_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))
        calls = []
        monkeypatch.setattr(ray_tune, "run_phase", lambda *a, **k: calls.append(1))

        sweep = {
            "base_config": "fake_base.yml",
            "phases": [{"name": "p1", "search_space": {
                # Typo'd path: 'agent.config.bogus_key' does not exist in BASE_CONFIG.
                "agent.config.bogus_key": {"dist": "uniform", "low": 0.0, "high": 1.0},
            }}],
        }
        with pytest.raises(ValueError, match="pre-flight resolution smoke test failed"):
            run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e")
        assert calls == []

    def test_preflight_tune_config_failure_raises_before_any_run_phase_call(self, monkeypatch, tmp_path):
        """A THIRD phase (or, here, second -- the same principle) with an
        unsupported ``scheduler.type: pbt`` must be caught by run_sweep's
        pre-flight ``build_search_alg``/``build_scheduler``/``build_stopper``
        construction loop -- before phase one's ``run_phase`` (which could
        run for hours) is ever called. Without this pre-flight check, PBT
        rejection lives only inside ``build_scheduler``, which ``run_phase``
        only reaches once that phase actually starts."""
        monkeypatch.setattr(ray_tune, "load_config", self._fake_load_config(load_config))
        calls = []
        monkeypatch.setattr(ray_tune, "run_phase", lambda *a, **k: calls.append(1))

        sweep = {
            "base_config": "fake_base.yml",
            "phases": [
                {"name": "p1", "search_space": {
                    "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
                {"name": "p2", "search_space": {
                    "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
                 "tune": {"scheduler": {"type": "pbt"}}},
            ],
        }
        with pytest.raises(ValueError, match="p2"):
            run_sweep(sweep, storage_path=tmp_path, sweep_name="e2e")
        assert calls == [], "run_phase must never be called when pre-flight tune config check fails"


# =============================================================================
# 13. End-to-end: a real two-phase CartPole sweep through a local Ray cluster
# =============================================================================
class TestEndToEndSweep:
    @pytest.mark.slow
    def test_two_phase_cartpole_sweep_promotes_and_writes_runnable_best_config(self, tmp_path, force_cpu, monkeypatch):
        import ray

        # `force_cpu` only monkeypatches `get_device` in THIS process; Ray
        # spawns real OS worker processes for each trial, which inherit
        # os.environ but not monkeypatches. Hide any local GPU from those
        # workers too, so `torch.cuda.is_available()` is False in the actor
        # regardless of the host machine. `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`
        # stops Ray from re-exposing the GPU on top of our CUDA_VISIBLE_DEVICES
        # override for tasks/actors that request 0 GPUs (Ray's current default
        # behavior for num_gpus=0/None; see the FutureWarning it prints otherwise).
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

        base_path = tmp_path / "base.yml"
        with open(base_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_tiny_ppo_config(str(tmp_path / "trials") + os.sep), f, sort_keys=False)

        sweep = {
            "base_config": str(base_path),
            "overrides": {"schedule.stop_units": 200, "log_level": "ERROR"},
            "ray_init": {"num_cpus": 2, "include_dashboard": False, "logging_level": "ERROR"},
            "defaults": {
                "metric": "avg_reward", "mode": "max", "resources": {"cpu": 1, "gpu": 0},
                "max_concurrent_trials": 1, "report": {"every": 8, "unit": "timestep"},
                "auto_learn_every": True,
            },
            "phases": [
                {
                    "name": "phase1",
                    "search_space": {"agent.config.entropy_coefficient": {"dist": "uniform", "low": 0.0, "high": 0.05}},
                    "tune": {"num_samples": 2, "search_alg": {"type": "random"}},
                    "promote": {"mode": "best"},
                },
                {
                    "name": "phase2",
                    "search_space": {"agent.config.policy_clip": {"dist": "uniform", "low": 0.1, "high": 0.3}},
                    "tune": {"num_samples": 2, "search_alg": {"type": "random"}},
                    "promote": {"mode": "best"},
                },
            ],
        }

        try:
            result = run_sweep(sweep, storage_path=tmp_path / "results", sweep_name="e2e_cartpole")
        finally:
            if ray.is_initialized():
                ray.shutdown()

        assert set(result["phases"]) == {"phase1", "phase2"}
        assert result["phases"]["phase1"]["num_trials"] == 2
        assert result["phases"]["phase2"]["num_trials"] == 2

        # Phase 2 started from phase 1's promoted config: the entropy
        # coefficient phase 2 never re-searches stays frozen at phase 1's
        # winning value all the way through to the final config.
        phase1_entropy = result["phases"]["phase1"]["best_config"]["agent"]["config"]["entropy_coefficient"]
        final_entropy = result["final_config"]["agent"]["config"]["entropy_coefficient"]
        assert final_entropy == phase1_entropy

        final_path = result["final_config_path"]
        assert os.path.isfile(final_path)
        loaded = load_config(final_path)
        assert "save_dir" not in loaded
        loaded["save_dir"] = str(tmp_path / "final_run") + os.sep
        trainer = build_trainer_from_config(loaded)
        try:
            assert trainer is not None
        finally:
            trainer.env.close()
