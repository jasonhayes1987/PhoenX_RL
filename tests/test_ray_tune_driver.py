"""Tests for the *driver layer* of ``phoenx.ray_tune``.

Covers searcher/scheduler/stopper factories, search-space sampling, trial
callback injection, phase promotion / ``points_to_evaluate`` seeding,
``write_best_config`` round-trip, and the bundled example sweeps. The
resolution layer (``load_sweep_config`` / ``validate_sweep_config`` /
``normalize_phases`` / ``build_search_space`` / ``resolve_trial_config`` /
``get_by_path`` / ``set_by_path``) is exercised here only as a *given* — it
is covered by ``tests/test_ray_tune.py``.

Every test in this file (except the one marked ``slow``) runs with no Ray
cluster (no ``ray.init``), no network, no GPU, and no Isaac Sim.
"""

from __future__ import annotations

import copy
import json
import logging
import re

import numpy as np
import pytest
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Domain
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper

from phoenx.builder import build_trainer_from_config, load_config
from phoenx.ray_tune import (
    _filter_seed_points,
    _inject_trial_callbacks,
    _search_alg_type,
    available_example_sweeps,
    build_scheduler,
    build_search_alg,
    build_search_space,
    build_stopper,
    load_sweep_config,
    normalize_phases,
    parse_block_library,
    resolve_trial_config,
    run_sweep,
    sample_search_space,
    validate_only,
    validate_sweep_config,
    write_best_config,
)

BUNDLED_SWEEPS = ("lunarlander_ppo.yml", "isaac_franka_cube_lift.yml")


# =============================================================================
# 1. build_search_alg
# =============================================================================
class TestBuildSearchAlg:
    def test_random_type_returns_basic_variant_generator(self):
        alg = build_search_alg({"tune": {"search_alg": {"type": "random"}}})
        assert isinstance(alg, BasicVariantGenerator)

    def test_missing_search_alg_defaults_to_random(self):
        """``phase['tune']['search_alg']`` absent defaults to ``{"type": "random"}``."""
        alg = build_search_alg({"tune": {}})
        assert isinstance(alg, BasicVariantGenerator)

    def test_missing_tune_key_defaults_to_random(self):
        alg = build_search_alg({})
        assert isinstance(alg, BasicVariantGenerator)

    def test_optuna_delegates_to_create_searcher(self):
        from ray.tune.search.optuna import OptunaSearch

        alg = build_search_alg({"tune": {"search_alg": {"type": "optuna"}}})
        assert isinstance(alg, OptunaSearch)

    def test_hyperopt_delegates_to_create_searcher(self):
        from ray.tune.search.hyperopt import HyperOptSearch

        alg = build_search_alg({"tune": {"search_alg": {"type": "hyperopt"}}})
        assert isinstance(alg, HyperOptSearch)

    def test_missing_extra_yields_rays_own_error(self):
        """A searcher whose optional package is not installed (bayesopt needs
        ``bayesian-optimization``, not a PhoenX dependency) surfaces Ray's own
        error, not a PhoenX-specific one."""
        with pytest.raises(Exception):
            build_search_alg({"tune": {"search_alg": {"type": "bayesopt"}}})

    def test_search_alg_not_a_mapping_raises(self):
        with pytest.raises(ValueError):
            build_search_alg({"tune": {"search_alg": "random"}})

    @pytest.mark.parametrize("alg_type", ["random", "optuna", "hyperopt"])
    def test_points_to_evaluate_passed_through_for_supported_types(self, alg_type):
        """`alg is not None` cannot fail (`build_search_alg` either returns
        an instance or raises), so it can never catch a dropped
        `points_to_evaluate` kwarg. Assert on the searcher's own
        `_points_to_evaluate` (verified, for all three of these classes on
        the installed Ray version, to hold the exact list passed at
        construction) instead, which is the assertion that would fail if
        the kwarg were silently dropped."""
        points = [{"agent.config.discount": 0.97}]
        alg = build_search_alg({"tune": {"search_alg": {"type": alg_type}}}, points_to_evaluate=points)
        assert alg._points_to_evaluate == points

    def test_points_to_evaluate_unsupported_type_logs_warning_and_does_not_raise(self, caplog):
        """``variant_generator`` is reachable via ``create_searcher`` (no extra
        package needed) but is not in the verified points_to_evaluate set, so
        seeding must be skipped with a warning rather than crashing the sweep
        between phases."""
        points = [{"agent.config.discount": 0.97}]
        with caplog.at_level(logging.WARNING):
            alg = build_search_alg(
                {"tune": {"search_alg": {"type": "variant_generator"}}}, points_to_evaluate=points
            )
        assert isinstance(alg, BasicVariantGenerator)
        assert any("points_to_evaluate" in rec.message for rec in caplog.records)

    def test_no_points_to_evaluate_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            build_search_alg({"tune": {"search_alg": {"type": "variant_generator"}}})
        assert not any("points_to_evaluate" in rec.message for rec in caplog.records)


# =============================================================================
# 2. build_scheduler
# =============================================================================
class TestBuildScheduler:
    def test_absent_scheduler_yields_none(self):
        assert build_scheduler({"tune": {}}) is None
        assert build_scheduler({}) is None

    def test_asha_aliases_async_hyperband(self):
        from ray.tune.schedulers import AsyncHyperBandScheduler

        sched = build_scheduler(
            {"tune": {"scheduler": {"type": "asha", "time_attr": "timestep",
                                     "grace_period": 10, "max_t": 1000}}}
        )
        assert isinstance(sched, AsyncHyperBandScheduler)

    def test_async_hyperband_direct_name_also_works(self):
        from ray.tune.schedulers import AsyncHyperBandScheduler

        sched = build_scheduler({"tune": {"scheduler": {"type": "async_hyperband"}}})
        assert isinstance(sched, AsyncHyperBandScheduler)

    @pytest.mark.parametrize("pbt_name", ["pbt", "population_based_training"])
    def test_pbt_raises_informative_value_error(self, pbt_name):
        """The plan's own example sweep (refine phase) uses PBT; a user will
        hit this, so the message must clearly say PBT is not supported yet."""
        with pytest.raises(ValueError) as excinfo:
            build_scheduler({"tune": {"scheduler": {"type": pbt_name}}})
        message = str(excinfo.value)
        assert "Population Based Training" in message or "PBT" in message
        assert "not supported yet" in message or "checkpoint" in message

    def test_scheduler_missing_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            build_scheduler({"tune": {"scheduler": {"time_attr": "timestep"}}})

    def test_scheduler_not_a_mapping_raises(self):
        with pytest.raises(ValueError):
            build_scheduler({"tune": {"scheduler": "asha"}})


# =============================================================================
# 3. build_stopper
# =============================================================================
class TestBuildStopper:
    def test_absent_stop_yields_none(self):
        assert build_stopper({"tune": {}}) is None

    def test_plain_metric_dict_returned_as_is(self):
        stop = build_stopper({"tune": {"stop": {"avg_reward": 250}}})
        assert stop == {"avg_reward": 250}
        assert isinstance(stop, dict)

    def test_single_typed_stopper_form(self):
        stop = build_stopper(
            {"tune": {"stop": {"type": "TrialPlateauStopper", "metric": "avg_reward",
                                "std": 2.0, "num_results": 8}}}
        )
        assert isinstance(stop, TrialPlateauStopper)

    def test_combined_list_folds_plain_dict_and_behaves_like_or(self):
        stop = build_stopper({"tune": {"stop": [
            {"avg_reward": 250},
            {"type": "TrialPlateauStopper", "metric": "avg_reward", "std": 2.0, "num_results": 8},
        ]}})
        assert isinstance(stop, CombinedStopper)
        # The folded plain-dict member should stop once avg_reward >= 250.
        assert stop("trial-1", {"avg_reward": 300}) is True
        assert stop("trial-1", {"avg_reward": 10}) is False

    def test_unknown_stopper_type_raises_and_lists_valid_names(self):
        with pytest.raises(ValueError) as excinfo:
            build_stopper({"tune": {"stop": {"type": "NoSuchStopper"}}})
        message = str(excinfo.value)
        assert "NoSuchStopper" in message
        assert "TrialPlateauStopper" in message

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_stopper({"tune": {"stop": []}})

    def test_list_element_not_a_mapping_raises(self):
        with pytest.raises(ValueError):
            build_stopper({"tune": {"stop": ["avg_reward"]}})

    def test_stop_not_mapping_or_list_raises(self):
        with pytest.raises(ValueError):
            build_stopper({"tune": {"stop": 250}})


# =============================================================================
# 4. sample_search_space
# =============================================================================
class TestSampleSearchSpace:
    def _space(self):
        return {
            "agent.config.discount": tune.uniform(0.9, 0.999),
            "schedule.learning_epochs": tune.randint(3, 11),
            "agent.config.model.branches.policy.output_config.0.params.kernel_params.gain":
                tune.choice([0.01, 0.1, 1.0]),
            "arch.trunk.slot0.block": tune.grid_search(["dense_block", "gru_block"]),
            "fixed_key": 42,
        }

    def test_every_returned_value_is_concrete(self):
        sampled = sample_search_space(self._space(), seed=0)
        assert set(sampled) == set(self._space())
        for value in sampled.values():
            assert not isinstance(value, Domain)
            assert not (isinstance(value, dict) and "grid_search" in value)

    def test_grid_search_takes_first_value(self):
        sampled = sample_search_space(self._space(), seed=0)
        assert sampled["arch.trunk.slot0.block"] == "dense_block"

    def test_constant_passes_through_unchanged(self):
        sampled = sample_search_space(self._space(), seed=0)
        assert sampled["fixed_key"] == 42

    def test_seed_makes_sampling_reproducible(self):
        space = self._space()
        first = sample_search_space(space, seed=7)
        second = sample_search_space(space, seed=7)
        assert first == second

    def test_different_seeds_can_differ(self):
        space = self._space()
        first = sample_search_space(space, seed=1)
        second = sample_search_space(space, seed=2)
        assert first != second

    def test_no_seed_still_returns_concrete_values(self):
        sampled = sample_search_space(self._space(), seed=None)
        for value in sampled.values():
            assert not isinstance(value, Domain)


# =============================================================================
# 5. Trial callback injection (_inject_trial_callbacks, called by rl_trainable)
# =============================================================================
class _FakeTrialContext:
    """Stand-in for ``tune.get_context()`` outside any real Ray session."""

    def __init__(self, trial_id):
        self._trial_id = trial_id

    def get_trial_id(self):
        return self._trial_id


class TestInjectTrialCallbacks:
    def test_default_injects_exactly_one_raytune_callback_no_wandb(self):
        config = {"callbacks": []}
        sweep = {"_sweep_name": "my_sweep"}
        phase = {"name": "refine", "report": {"every": 20000, "unit": "timestep"}}
        _inject_trial_callbacks(config, sweep, phase, sampled={"agent.config.discount": 0.97})

        callbacks = config["callbacks"]
        raytune_cbs = [cb for cb in callbacks if cb.get("type") == "RayTuneCallback"]
        wandb_cbs = [cb for cb in callbacks if cb.get("type") == "WandbCallback"]
        assert len(raytune_cbs) == 1
        assert raytune_cbs[0]["config"] == {"every": 20000, "unit": "timestep"}
        assert len(wandb_cbs) == 0

    def test_report_defaults_used_when_phase_report_absent(self):
        config = {}
        sweep = {"_sweep_name": "my_sweep"}
        phase = {"name": "refine"}
        _inject_trial_callbacks(config, sweep, phase, sampled={})
        raytune_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "RayTuneCallback"]
        assert len(raytune_cbs) == 1
        assert raytune_cbs[0]["config"] == {"every": 50000, "unit": "timestep"}

    def test_replaces_existing_raytune_callback_from_base_config(self):
        config = {"callbacks": [{"type": "RayTuneCallback", "config": {"every": 1, "unit": "episode"}}]}
        sweep = {"_sweep_name": "my_sweep"}
        phase = {"name": "refine", "report": {"every": 20000, "unit": "timestep"}}
        _inject_trial_callbacks(config, sweep, phase, sampled={})

        raytune_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "RayTuneCallback"]
        assert len(raytune_cbs) == 1
        assert raytune_cbs[0]["config"] == {"every": 20000, "unit": "timestep"}

    def test_wandb_absent_strips_wandb_callback_from_base_config(self):
        """A sweep trial must never spam W&B with default naming; if the
        phase declares no ``wandb`` section, any WandbCallback the base
        training config declares is stripped."""
        config = {"callbacks": [{"type": "WandbCallback", "config": {"project_name": "base-project"}}]}
        sweep = {"_sweep_name": "my_sweep"}
        phase = {"name": "refine"}
        _inject_trial_callbacks(config, sweep, phase, sampled={})

        wandb_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "WandbCallback"]
        assert len(wandb_cbs) == 0
        raytune_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "RayTuneCallback"]
        assert len(raytune_cbs) == 1

    def test_wandb_present_injects_exactly_one_with_explicit_fields(self, monkeypatch):
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(ray_tune_mod.tune, "get_context", lambda: _FakeTrialContext("abcd_00003"))

        config = {"callbacks": []}
        sweep = {"_sweep_name": "my_sweep"}
        phase = {"name": "architecture", "wandb": {"project": "franka-sweeps"}}
        sampled = {"agent.config.discount": 0.97}
        _inject_trial_callbacks(config, sweep, phase, sampled)

        wandb_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "WandbCallback"]
        assert len(wandb_cbs) == 1
        wcfg = wandb_cbs[0]["config"]
        assert wcfg["project_name"] == "franka-sweeps"
        assert "abcd_00003" in wcfg["run_name"]
        assert "my_sweep" in wcfg["run_name"] and "architecture" in wcfg["run_name"]
        assert wcfg["group"] == "my_sweep-architecture"
        assert set(wcfg["tags"]) >= {"my_sweep", "architecture"}
        assert wcfg["sweep_params"] == sampled

        # Exactly one RayTuneCallback too (not stripped by the wandb branch).
        raytune_cbs = [cb for cb in config["callbacks"] if cb.get("type") == "RayTuneCallback"]
        assert len(raytune_cbs) == 1

    def test_wandb_group_shared_across_trials_but_run_name_differs(self, monkeypatch):
        """Every trial of a phase must share one ``group`` (for W&B's grouped
        view) while getting its own distinguishing ``run_name``."""
        import phoenx.ray_tune as ray_tune_mod

        results = []
        for trial_id in ("t1_00001", "t1_00002"):
            monkeypatch.setattr(ray_tune_mod.tune, "get_context", lambda tid=trial_id: _FakeTrialContext(tid))
            config = {"callbacks": []}
            phase = {"name": "optimizers", "wandb": {"project": "p"}}
            _inject_trial_callbacks(config, {"_sweep_name": "s"}, phase, sampled={})
            wcfg = next(cb["config"] for cb in config["callbacks"] if cb["type"] == "WandbCallback")
            results.append(wcfg)

        assert results[0]["group"] == results[1]["group"]
        assert results[0]["run_name"] != results[1]["run_name"]

    def test_wandb_missing_project_raises(self):
        # A non-empty but project-less wandb dict: `if phase['wandb']:` is
        # truthy, so the branch runs and must raise on the missing key.
        config = {}
        phase = {"name": "architecture", "wandb": {"tags": ["extra"]}}
        with pytest.raises(ValueError, match="project"):
            _inject_trial_callbacks(config, {"_sweep_name": "s"}, phase, sampled={})

    def test_falls_back_to_local_trial_id_outside_a_tune_session(self):
        """Calling this directly (as this whole test class does) never has a
        real Ray Tune session behind it; the trial id must fall back to
        ``'local'`` rather than raising or leaving run_name blank."""
        config = {}
        phase = {"name": "refine", "wandb": {"project": "p"}}
        _inject_trial_callbacks(config, {"_sweep_name": "s"}, phase, sampled={})
        wcfg = next(cb["config"] for cb in config["callbacks"] if cb["type"] == "WandbCallback")
        assert "local" in wcfg["run_name"]

    def test_sweep_name_defaults_to_sweep_when_absent(self):
        config = {}
        phase = {"name": "refine", "wandb": {"project": "p"}}
        _inject_trial_callbacks(config, {}, phase, sampled={})
        wcfg = next(cb["config"] for cb in config["callbacks"] if cb["type"] == "WandbCallback")
        assert wcfg["group"] == "sweep-refine"

    def test_sweep_params_numpy_scalars_arrive_as_native_python_types(self):
        """`WandbCallback.get_config()` returns `sweep_params` verbatim, and
        `Trainer.save()` (which fires on every new best-reward episode)
        does a plain `json.dump` with no custom encoder, so a numpy scalar
        sampled value would crash a checkpoint save mid-run. `sampled` must
        reach the injected callback's config already passed through
        `_to_plain`."""
        config = {}
        phase = {"name": "refine", "wandb": {"project": "p"}}
        sampled = {
            "agent.config.discount": np.float64(0.97),
            "schedule.learning_epochs": np.int64(4),
        }
        _inject_trial_callbacks(config, {"_sweep_name": "s"}, phase, sampled)
        wcfg = next(cb["config"] for cb in config["callbacks"] if cb["type"] == "WandbCallback")
        sweep_params = wcfg["sweep_params"]

        assert isinstance(sweep_params["agent.config.discount"], float)
        assert isinstance(sweep_params["schedule.learning_epochs"], int)
        assert not isinstance(sweep_params["agent.config.discount"], np.generic)
        assert not isinstance(sweep_params["schedule.learning_epochs"], np.generic)
        json.dumps(wcfg)  # must not raise (a numpy scalar would break this)


# =============================================================================
# 6. Phase promotion + points_to_evaluate seeding (_filter_seed_points)
# =============================================================================
class TestFilterSeedPoints:
    def test_search_space_keys_carry_over(self):
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = {"name": "p2", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.95, "high": 0.999}}}
        top_k = [{"agent.config.discount": 0.97}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result == [{"agent.config.discount": 0.97}]

    def test_opt_keys_carry_over_when_next_phase_searches_same_module_field(self):
        phase = {"name": "p1", "optimizers": {"branches.policy": {
            "lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-3}}}}
        next_phase = {"name": "p2", "optimizers": {"branches.policy": {
            "lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-3}}}}
        top_k = [{"opt.branches.policy.lr": 3e-4}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result == [{"opt.branches.policy.lr": 3e-4}]

    def test_opt_key_dropped_when_next_phase_does_not_search_it(self):
        phase = {"name": "p1", "optimizers": {"branches.policy": {
            "lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-3}}}}
        next_phase = {"name": "p2", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        top_k = [{"opt.branches.policy.lr": 3e-4}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result is None

    def _arch_spec(self):
        return {
            "trunk": {"depth": {"dist": "randint", "lower": 1, "upper": 3}, "blocks": ["dense_block"]},
        }

    def _blocks(self):
        return parse_block_library({"dense_block": {"layers": [
            {"type": "dense", "params": {"units": {"dist": "choice", "values": [64, 128]}}},
        ]}})

    def test_arch_keys_carry_over_when_architecture_identical(self):
        arch = self._arch_spec()
        phase = {"name": "p1", "architecture": arch}
        next_phase = {"name": "p2", "architecture": copy.deepcopy(arch)}
        top_k = [{"arch.trunk.depth": 2, "arch.trunk.slot0.block": "dense_block"}]
        result = _filter_seed_points(top_k, phase, next_phase, self._blocks())
        assert result == top_k

    def test_arch_keys_dropped_when_architecture_differs(self, caplog):
        phase = {"name": "p1", "architecture": self._arch_spec()}
        different_arch = {
            "trunk": {"depth": {"dist": "randint", "lower": 1, "upper": 4}, "blocks": ["dense_block"]},
        }
        next_phase = {"name": "p2", "architecture": different_arch}
        top_k = [{"arch.trunk.depth": 2, "arch.trunk.slot0.block": "dense_block"}]
        with caplog.at_level(logging.INFO):
            result = _filter_seed_points(top_k, phase, next_phase, self._blocks())
        assert result is None
        assert any("skipping arch" in rec.message for rec in caplog.records)

    def test_seed_dict_that_ends_up_empty_is_dropped_not_passed(self):
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = {"name": "p2", "search_space": {
            "agent.config.gae_coefficient": {"dist": "uniform", "low": 0.9, "high": 0.99}}}
        top_k = [{"agent.config.discount": 0.97}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result is None

    def test_mixed_top_k_keeps_only_nonempty_points(self):
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = {"name": "p2", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        top_k = [
            {"agent.config.discount": 0.97},
            {"agent.config.unrelated_key": 1.0},
        ]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result == [{"agent.config.discount": 0.97}]

    def test_empty_top_k_returns_none(self):
        phase = {"name": "p1", "search_space": {}}
        next_phase = {"name": "p2", "search_space": {}}
        assert _filter_seed_points([], phase, next_phase, blocks={}) is None

    def test_disjoint_search_spaces_logs_both_phase_names(self, caplog):
        """Every candidate point filtering down to empty (true of both
        bundled example sweeps' phase transitions) must log one line naming
        both the finishing and the next phase, not fail silently."""
        phase = {"name": "explore", "search_space": {
            "agent.config.entropy_coefficient": {"dist": "uniform", "low": 0.0, "high": 0.05}}}
        next_phase = {"name": "refine", "search_space": {
            "agent.config.policy_clip": {"dist": "uniform", "low": 0.1, "high": 0.3}}}
        top_k = [{"agent.config.entropy_coefficient": 0.02}]
        with caplog.at_level(logging.INFO):
            result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result is None
        assert any("explore" in rec.message and "refine" in rec.message for rec in caplog.records)

    def _two_key_next_phase(self, name="p2"):
        return {"name": name, "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
            "agent.config.gae_coefficient": {"dist": "uniform", "low": 0.9, "high": 0.99},
        }}

    def test_partial_point_dropped_for_non_random_next_searcher(self, caplog):
        """`OptunaSearch`/`HyperOptSearch` raise `ValueError` on a partial
        `points_to_evaluate` entry rather than accepting it (verified live
        against the installed Ray version), so a point covering only one
        of the next phase's two search-space keys must be dropped, and the
        drop logged, when the next phase's searcher is not `'random'`."""
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = self._two_key_next_phase()
        top_k = [{"agent.config.discount": 0.97}]
        with caplog.at_level(logging.INFO):
            result = _filter_seed_points(top_k, phase, next_phase, blocks={}, next_alg_type="optuna")
        assert result is None
        assert any("partial" in rec.message.lower() for rec in caplog.records)

    def test_partial_point_kept_for_random_next_searcher(self):
        """`BasicVariantGenerator` (`type: 'random'`) accepts a partial
        point, so the full-coverage requirement must not apply to it."""
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = self._two_key_next_phase()
        top_k = [{"agent.config.discount": 0.97}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={}, next_alg_type="random")
        assert result == [{"agent.config.discount": 0.97}]

    def test_exact_coverage_point_kept_for_non_random_searcher(self):
        phase = self._two_key_next_phase("p1")
        next_phase = self._two_key_next_phase("p2")
        top_k = [{"agent.config.discount": 0.97, "agent.config.gae_coefficient": 0.95}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={}, next_alg_type="optuna")
        assert result == top_k

    def test_next_alg_type_none_derives_random_default_from_next_phase(self):
        """`next_alg_type=None` (the default) must derive the same
        `'random'` default `_search_alg_type` itself uses when the next
        phase declares no `tune.search_alg`, so a partial point is kept."""
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = self._two_key_next_phase()  # no tune.search_alg -> derives "random"
        top_k = [{"agent.config.discount": 0.97}]
        result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result == [{"agent.config.discount": 0.97}]

    def test_next_alg_type_none_derives_non_random_from_next_phase_tune(self, caplog):
        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = self._two_key_next_phase()
        next_phase["tune"] = {"search_alg": {"type": "optuna"}}
        top_k = [{"agent.config.discount": 0.97}]
        with caplog.at_level(logging.INFO):
            result = _filter_seed_points(top_k, phase, next_phase, blocks={})
        assert result is None
        assert any("partial" in rec.message.lower() for rec in caplog.records)

    def test_filtered_seed_points_feed_a_real_optuna_search_without_raising(self):
        """The regression assertion: feed the exact points `_filter_seed_points`
        returns for a non-random next searcher into a real `OptunaSearch`
        via `set_search_properties` with the next phase's real search
        space. Checked against the pre-fix source (bare truthiness /
        no full-coverage check), the partial point in `top_k` below would
        have survived filtering and this call would raise
        `ValueError: Dim of point ... and parameter_names ... do not match`."""
        from ray.tune.search.optuna import OptunaSearch

        phase = {"name": "p1", "search_space": {
            "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}}
        next_phase = self._two_key_next_phase()
        next_phase["tune"] = {"search_alg": {"type": "optuna"}}
        top_k = [
            {"agent.config.discount": 0.97, "agent.config.gae_coefficient": 0.95},  # full coverage
            {"agent.config.discount": 0.93},  # partial -> must be dropped
        ]

        seeds = _filter_seed_points(top_k, phase, next_phase, blocks={}, next_alg_type="optuna")
        assert seeds == [{"agent.config.discount": 0.97, "agent.config.gae_coefficient": 0.95}]

        next_space = build_search_space(next_phase, blocks={})
        searcher = OptunaSearch(points_to_evaluate=seeds)
        searcher.set_search_properties(metric="avg_reward", mode="max", config=next_space)  # must not raise


# =============================================================================
# 6b. _search_alg_type
# =============================================================================
class TestSearchAlgType:
    def test_defaults_to_random_when_tune_key_absent(self):
        assert _search_alg_type({}) == "random"

    def test_defaults_to_random_when_search_alg_key_absent(self):
        assert _search_alg_type({"tune": {}}) == "random"

    def test_defaults_to_random_when_search_alg_not_a_mapping(self):
        assert _search_alg_type({"tune": {"search_alg": "oops"}}) == "random"

    def test_reads_and_lowercases_configured_type(self):
        assert _search_alg_type({"tune": {"search_alg": {"type": "Optuna"}}}) == "optuna"


# =============================================================================
# 7. write_best_config round-trip
# =============================================================================
DENSE = lambda u: {"type": "dense", "params": {"units": u, "kernel": "orthogonal",
                                               "kernel_params": {"gain": 1.41421356}}}
RELU = {"type": "relu"}
OUT = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 3e-4}}
CARTPOLE_ENV_CONFIG = {"type": "gymnasium", "config": {
    "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
    "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 3}}


def _tiny_cartpole_config(save_dir: str) -> dict:
    """Minimal ActorCritic + RolloutBuffer config, inline-dict style
    (mirrors ``tests/test_builders.py``), with a numpy scalar mixed in to
    exercise ``_to_plain``'s numpy -> native conversion on write."""
    return {
        "save_dir": save_dir,
        "log_level": "ERROR",
        "schedule": {"stop_unit": "timestep", "stop_units": 4, "learn_every_unit": "timestep",
                     "learn_every": 999_999, "updates_per_learn": 1, "batch_size": 1,
                     "warmup_steps": 0, "seed": 3},
        "agent": {"type": "ActorCritic", "config": {
            "name": "ActorCritic",
            "model": {"branches": {
                "policy": {"type": "StochasticDiscreteHead", "layer_config": [DENSE(16), RELU],
                           "output_config": OUT, "distribution": "categorical",
                           "optimizer_params": OPT, "device": "cpu"},
                "value": {"type": "ValueHead", "layer_config": [DENSE(16), RELU],
                          "output_config": OUT, "optimizer_params": OPT, "device": "cpu"},
            }},
            "discount": np.float64(0.99), "auto_entropy_tuning": False, "device": "cpu",
            "log_level": "ERROR",
        }},
        "env": CARTPOLE_ENV_CONFIG,
        "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
        "callbacks": [{"type": "RayTuneCallback", "config": {"every": 1, "unit": "timestep"}}],
    }


class TestWriteBestConfig:
    def test_strips_raytune_callback_and_save_dir(self, tmp_path):
        config = _tiny_cartpole_config(save_dir=str(tmp_path / "trial_00007") + "/")
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)

        raw = dest.read_text(encoding="utf-8")
        assert "save_dir" not in raw
        assert "RayTuneCallback" not in raw

    def test_numpy_scalars_do_not_leak_as_python_object_tags(self, tmp_path):
        config = _tiny_cartpole_config(save_dir=str(tmp_path / "trial_00007") + "/")
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)

        raw = dest.read_text(encoding="utf-8")
        assert "!!python/object" not in raw
        assert "numpy" not in raw
        loaded = load_config(dest)
        assert loaded["agent"]["config"]["discount"] == pytest.approx(0.99)
        assert isinstance(loaded["agent"]["config"]["discount"], float)

    def test_wandb_callback_is_not_stripped_only_raytune_is(self, tmp_path):
        """Only ``RayTuneCallback`` is a trial-runtime-only artifact;
        ``WandbCallback`` legitimately describes a real W&B run and must
        survive into the written config."""
        config = _tiny_cartpole_config(save_dir=str(tmp_path / "t") + "/")
        config["callbacks"].append({"type": "WandbCallback", "config": {"project_name": "p"}})
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)

        loaded = load_config(dest)
        cb_types = [cb["type"] for cb in loaded.get("callbacks", [])]
        assert "WandbCallback" in cb_types
        assert "RayTuneCallback" not in cb_types

    def test_creates_parent_directories(self, tmp_path):
        config = _tiny_cartpole_config(save_dir=str(tmp_path / "t") + "/")
        dest = tmp_path / "nested" / "dir" / "best_config.yml"
        write_best_config(config, dest)
        assert dest.is_file()

    def test_round_trip_builds_a_working_trainer(self, tmp_path, force_cpu):
        """The single most user-visible failure mode of this whole feature
        is a 'best config' that cannot actually be run. Dump it, read it
        back with ``phoenx.builder.load_config``, and assert
        ``build_trainer_from_config`` builds a working trainer from it."""
        config = _tiny_cartpole_config(save_dir=str(tmp_path / "trial_00001") + "/")
        dest = tmp_path / "best_config.yml"
        write_best_config(config, dest)

        loaded = load_config(dest)
        assert "save_dir" not in loaded
        # A real subsequent phoenx-train run supplies its own save_dir (the
        # trial-specific one is intentionally stripped); mirror that here.
        loaded["save_dir"] = str(tmp_path / "rerun") + "/"

        trainer = build_trainer_from_config(loaded)
        try:
            assert trainer.agent is not None
            assert trainer.env is not None
        finally:
            trainer.env.close()


# =============================================================================
# 8. Bundled example sweeps
# =============================================================================
class TestBundledExampleSweeps:
    def test_available_example_sweeps_returns_exactly_the_two_bundled_names(self):
        assert available_example_sweeps() == sorted(BUNDLED_SWEEPS)

    def test_load_sweep_config_resolves_packaged_name(self):
        sweep = load_sweep_config("lunarlander_ppo.yml")
        assert sweep["base_config"] == "LunarLanderContinuous-v3/ppo.yml"

    def test_load_sweep_config_ondisk_path_wins_over_packaged_name(self, tmp_path):
        """An on-disk path always wins, even if a file with the exact same
        relative name also happens to be a bundled example."""
        on_disk = tmp_path / "lunarlander_ppo.yml"
        on_disk.write_text("base_config: LunarLander-v3/reinforce.yml\nsearch_space: {}\n",
                            encoding="utf-8")
        sweep = load_sweep_config(str(on_disk))
        assert sweep["base_config"] == "LunarLander-v3/reinforce.yml"

    def test_load_sweep_config_missing_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_sweep_config("does_not_exist_anywhere.yml")

    @pytest.mark.parametrize("sweep_name", BUNDLED_SWEEPS)
    def test_bundled_sweep_passes_validate_sweep_config(self, sweep_name):
        sweep = load_sweep_config(sweep_name)
        validate_sweep_config(sweep)  # must not raise

    @pytest.mark.parametrize("sweep_name", BUNDLED_SWEEPS)
    def test_bundled_sweep_resolves_one_trial_per_phase(self, sweep_name):
        """Pure dict resolution: no Ray cluster, no GPU, and for the Isaac
        sweep specifically, no Isaac Sim boot. Also the regression guard for
        a typo'd dotted path in a shipped example, since ``set_by_path``
        raises on a missing intermediate segment."""
        sweep = load_sweep_config(sweep_name)
        phases = normalize_phases(sweep)
        blocks = parse_block_library(sweep.get("blocks"))
        base_config = load_config(sweep["base_config"])

        assert phases, f"{sweep_name}: no phases resolved"
        for phase in phases:
            space = build_search_space(phase, blocks)
            sampled = sample_search_space(space, seed=0)
            resolved = resolve_trial_config(sweep, phase, sampled, copy.deepcopy(base_config))
            assert isinstance(resolved, dict)
            assert "agent" in resolved and "env" in resolved

    def test_isaac_sweep_base_config_name_has_no_isaac_marker_needed(self):
        """The Isaac sweep resolves via pure dict manipulation; nothing here
        should require the `isaac` marker or a GPU."""
        sweep = load_sweep_config("isaac_franka_cube_lift.yml")
        assert "IsaacSim" in sweep["base_config"]
        # Resolution (see test above) already proves no Isaac Sim boot is
        # needed; this test just documents the expectation explicitly.

    @pytest.mark.parametrize("sweep_name", BUNDLED_SWEEPS)
    def test_bundled_sweep_every_phase_tune_config_is_fully_constructible(self, sweep_name):
        """Structurally valid is not the same as constructible: both bundled
        sweeps once declared ``scheduler: {type: asha, grace_period: ...}``
        with no ``max_t``, so ``AsyncHyperBandScheduler``'s implicit
        ``max_t=100`` default made ``build_scheduler`` raise
        ``AssertionError: grace_period must be <= max_t!`` for every phase
        using it -- a real ``phoenx-tune`` run would have crashed on phase
        one, and neither ``validate_sweep_config`` nor
        ``resolve_trial_config`` ever construct a searcher, scheduler, or
        stopper to catch it. Iterate every phase via ``normalize_phases``,
        not just the first."""
        sweep = load_sweep_config(sweep_name)
        validate_sweep_config(sweep)
        phases = normalize_phases(sweep)
        assert phases, f"{sweep_name}: no phases resolved"
        for phase in phases:
            build_search_alg(phase)  # must not raise
            build_scheduler(phase)  # must not raise
            build_stopper(phase)  # must not raise


# =============================================================================
# 9. validate_only
# =============================================================================
class TestValidateOnly:
    @pytest.mark.parametrize("sweep_name", BUNDLED_SWEEPS)
    def test_completes_on_a_bundled_sweep_and_prints_a_diff(self, sweep_name, capsys):
        sweep = load_sweep_config(sweep_name)
        validate_only(sweep, num_samples=2)
        out = capsys.readouterr().out
        assert "Phase" in out
        assert "sample" in out

    def test_raises_clearly_on_a_typo_d_path(self, capsys):
        sweep = {
            "base_config": "LunarLander-v3/reinforce.yml",
            "search_space": {
                "agent.config.nonexistent_typo_key": {"dist": "uniform", "low": 0.9, "high": 0.999},
            },
        }
        with pytest.raises(ValueError) as excinfo:
            validate_only(sweep, num_samples=2)
        message = str(excinfo.value)
        assert "phase_0" in message
        assert "nonexistent_typo_key" in message

    def test_num_samples_controls_number_of_printed_samples(self, capsys):
        sweep = {
            "base_config": "LunarLander-v3/reinforce.yml",
            "search_space": {
                "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999},
            },
        }
        validate_only(sweep, num_samples=3)
        out = capsys.readouterr().out
        assert out.count("sample 0") == 1
        assert out.count("sample 1") == 1
        assert out.count("sample 2") == 1
        assert "sample 3" not in out

    def test_never_touches_ray_cluster(self, monkeypatch):
        """No ``ray.init`` / ``Tuner`` anywhere in the call path."""
        import phoenx.ray_tune as ray_tune_mod

        def _boom(*args, **kwargs):
            raise AssertionError("validate_only must never construct a Tuner")

        monkeypatch.setattr(ray_tune_mod.tune, "Tuner", _boom)
        sweep = load_sweep_config("lunarlander_ppo.yml")
        validate_only(sweep, num_samples=1)  # must not raise

    def _two_phase_sweep(self, bad_tune):
        """Phase one is unremarkable; phase two carries the bad
        ``tune`` config, so any pre-flight failure must be attributable to
        phase two specifically, and must happen before phase one's diff
        (which would otherwise print first) is ever emitted."""
        return {
            "base_config": "LunarLander-v3/reinforce.yml",
            "phases": [
                {"name": "p1", "search_space": {
                    "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}}},
                {"name": "p2", "search_space": {
                    "agent.config.discount": {"dist": "uniform", "low": 0.9, "high": 0.999}},
                 "tune": bad_tune},
            ],
        }

    def test_second_phase_pbt_scheduler_raises_naming_phase_before_any_diff_prints(self, capsys):
        sweep = self._two_phase_sweep({"scheduler": {"type": "pbt"}})
        with pytest.raises(ValueError) as excinfo:
            validate_only(sweep, num_samples=2)
        assert "p2" in str(excinfo.value)
        # Phase one's diff must never be printed: the pre-flight
        # searcher/scheduler/stopper construction loop runs over every
        # phase, and must fail, before the (separate) diff-printing loop
        # starts on phase one.
        assert capsys.readouterr().out == ""

    def test_second_phase_unknown_stopper_name_raises_naming_phase_before_any_diff_prints(self, capsys):
        sweep = self._two_phase_sweep({"stop": {"type": "NotARealStopperClass"}})
        with pytest.raises(ValueError) as excinfo:
            validate_only(sweep, num_samples=2)
        assert "p2" in str(excinfo.value)
        assert capsys.readouterr().out == ""

    def test_second_phase_unknown_scheduler_alias_raises_naming_phase_before_any_diff_prints(self, capsys):
        sweep = self._two_phase_sweep({"scheduler": {"type": "not_a_real_scheduler_alias"}})
        with pytest.raises(ValueError) as excinfo:
            validate_only(sweep, num_samples=2)
        assert "p2" in str(excinfo.value)
        assert capsys.readouterr().out == ""


# =============================================================================
# 10. End-to-end: two-phase CartPole sweep (slow, real Ray + real training)
# =============================================================================
_CARTPOLE_PPO_BASE_YAML = """
log_level: ERROR
schedule:
  stop_unit: timestep
  stop_units: 2000
  learn_every_unit: timestep
  learn_every: 64
  mini_batch_size: 16
  learning_epochs: 2
  seed: 3
agent:
  type: PPO
  config:
    model:
      device: cpu
      roots: null
      trunk: null
      branches:
        policy:
          type: StochasticDiscreteHead
          layer_config:
            - {type: dense, params: {units: 16, kernel: orthogonal, kernel_params: {gain: 1.41421356}}}
            - {type: relu}
          output_config:
            - {type: dense, params: {kernel: orthogonal, kernel_params: {gain: 0.01}}}
          optimizer_params: {type: Adam, params: {lr: 3.0e-4}}
          distribution: categorical
        value:
          type: ValueHead
          layer_config:
            - {type: dense, params: {units: 16, kernel: orthogonal, kernel_params: {gain: 1.41421356}}}
            - {type: relu}
          output_config:
            - {type: dense, params: {kernel: orthogonal, kernel_params: {gain: 0.01}}}
          optimizer_params: {type: Adam, params: {lr: 3.0e-4}}
    discount: 0.99
    gae_coefficient: 0.95
    entropy_coefficient: 0.02
    auto_entropy_tuning: false
    policy_clip: 0.2
    policy_grad_clip: 0.5
    value_clip: 0.2
    value_grad_clip: 0.5
    value_coef: 0.5
    reward_clip: .inf
    bootstrap_truncations: true
    log_level: ERROR
    device: cpu
env:
  type: gymnasium
  config:
    cfg: CartPole-v1
    num_envs: 4
    obs_key: null
    goal_key: null
    ach_goal_key: null
    wrappers: []
    render_mode: null
    seed: 3
buffer:
  type: RolloutBuffer
  config:
    buffer_size: 16
    device: cpu
"""


@pytest.mark.slow
class TestEndToEndTwoPhaseCartpoleSweep:
    def test_two_phases_run_and_promotion_and_best_config_load(self, tmp_path, force_cpu, monkeypatch):
        import ray

        # `env_wrapper.EnvWrapper` always calls the bare `get_device()` (no
        # per-config override; see env_wrapper.py:140), and Ray trial actors
        # are separate processes the test's `force_cpu` monkeypatch cannot
        # reach. On a GPU-equipped machine the only way to keep trial actors
        # on CPU is to hide CUDA in the runtime env; Ray itself clobbers a
        # CUDA_VISIBLE_DEVICES override back to "" for num_gpus=0 requests
        # unless RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 is set first, and an
        # empty string does not reliably hide the GPU from torch on this
        # platform, so "-1" is used instead.
        monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

        base_cfg_path = tmp_path / "cartpole_ppo_base.yml"
        base_cfg_path.write_text(_CARTPOLE_PPO_BASE_YAML, encoding="utf-8")

        sweep = {
            "base_config": str(base_cfg_path),
            "overrides": {"log_level": "ERROR"},
            "ray_init": {"runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": "-1"}}},
            "defaults": {
                "metric": "avg_reward", "mode": "max",
                "resources": {"cpu": 1, "gpu": 0},
                "report": {"every": 500, "unit": "timestep"},
            },
            "phases": [
                {
                    "name": "explore",
                    "search_space": {
                        "agent.config.entropy_coefficient": {"dist": "uniform", "low": 0.0, "high": 0.05},
                    },
                    "tune": {"num_samples": 2, "search_alg": {"type": "random"}},
                    "promote": {"mode": "best"},
                },
                {
                    "name": "refine",
                    "search_space": {
                        "agent.config.policy_clip": {"dist": "uniform", "low": 0.1, "high": 0.3},
                    },
                    "tune": {"num_samples": 2, "search_alg": {"type": "random"}},
                    "promote": {"mode": "best"},
                },
            ],
        }

        try:
            result = run_sweep(
                sweep,
                storage_path=str(tmp_path / "storage"),
                sweep_name="e2e_cartpole",
                max_concurrent_trials=1,
            )
        finally:
            ray.shutdown()

        assert set(result["phases"]) == {"explore", "refine"}
        for phase_name in ("explore", "refine"):
            phase_result = result["phases"][phase_name]
            assert phase_result["num_trials"] == 2
            assert phase_result["num_errors"] == 0

        # Phase two's base config carried phase one's promoted (winning)
        # value for a key phase two does not itself search.
        explore_winner_entropy = result["phases"]["explore"]["best_config"]["agent"]["config"][
            "entropy_coefficient"
        ]
        refine_base_entropy = result["phases"]["refine"]["best_config"]["agent"]["config"][
            "entropy_coefficient"
        ]
        assert refine_base_entropy == pytest.approx(explore_winner_entropy)

        final_config_path = result["final_config_path"]
        loaded_final = load_config(final_config_path)
        # The tautological `load_config(x) == load_config(x)` this replaces
        # can never fail. Assert something real instead: the final config
        # carries "refine" (the last-run phase)'s promoted winning value,
        # and the reloaded config actually builds a working trainer --
        # the single most user-visible failure mode of this whole feature
        # is a "best config" that cannot actually be run.
        refine_winner_policy_clip = result["phases"]["refine"]["best_config"]["agent"]["config"]["policy_clip"]
        assert loaded_final["agent"]["config"]["policy_clip"] == pytest.approx(refine_winner_policy_clip)
        assert "save_dir" not in loaded_final
        loaded_final["save_dir"] = str(tmp_path / "final_rerun") + "/"
        trainer = build_trainer_from_config(loaded_final)
        try:
            assert trainer.agent is not None
        finally:
            trainer.env.close()

        # Short trial_XXXXX dir names (mandatory on Windows: Ray's default
        # embeds every sampled param and blows past MAX_PATH mid-sweep).
        explore_phase_dir = tmp_path / "storage" / "e2e_cartpole" / "explore"
        trial_dirs = [p.name for p in explore_phase_dir.iterdir() if p.is_dir()]
        assert trial_dirs, f"no trial directories found under {explore_phase_dir}"
        for name in trial_dirs:
            assert re.fullmatch(r"trial_\d+", name), f"unexpected trial dir name: {name}"
