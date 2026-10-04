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
from types import SimpleNamespace

import numpy as np
import pytest
from ray import tune
from ray.tune.experiment import Experiment
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Domain
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper

from phoenx.builder import build_trainer_from_config, load_config
from phoenx.ray_tune import (
    _filter_seed_points,
    _format_trial_ordinal,
    _inject_trial_callbacks,
    _phase_trial_total,
    _PhaseTrialLedger,
    _resolve_trial_ordinal,
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
    rl_trainable,
    run_phase,
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
        """PBT is a natural scheduler choice for a sweep's refine phase; a user
        will hit this, so the message must clearly say PBT is not supported yet."""
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
# 4b. Per-phase trial ordinal helpers (_format_trial_ordinal, _PhaseTrialLedger,
# _resolve_trial_ordinal) -- no Ray cluster.
# =============================================================================
class TestTrialOrdinalHelpers:
    @pytest.mark.parametrize(
        "ordinal,total,expected",
        [
            (7, 10, "0007of0010"),
            (7, None, "0007"),
            (7, True, "0007"),
            (7, False, "0007"),
            (7, 0, "0007"),
            (7, -1, "0007"),
            (7, "10", "0007"),
            (7, 10.0, "0007"),
            (12345, 10, "12345of0010"),
            (7, 100000, "000007of100000"),
        ],
    )
    def test_format_trial_ordinal_pads_and_appends_total_only_for_positive_nonbool_int(
        self, ordinal, total, expected
    ):
        """C1: width is max(4, len(str(total))); 'ofTOTAL' is appended only when total is a positive, non-bool int."""
        assert _format_trial_ordinal(ordinal, total) == expected

    def test_ledger_ordinal_is_one_based_memoized_and_unique_per_trial_id(self):
        """C2: `_PhaseTrialLedger.ordinal` is 1-based, memoized per trial id, and issues each ordinal to exactly one id."""
        ledger = _PhaseTrialLedger()
        first = [ledger.ordinal(f"t{i}") for i in range(50)]
        assert first == list(range(1, 51))
        assert [ledger.ordinal(f"t{i}") for i in range(50)] == first
        assert ledger.issued() == 50
        assert ledger.ordinal("t0") == 1

    def test_resolve_ordinal_never_contacts_actor_when_phase_counter_absent(self, monkeypatch):
        """C3(a-d): with no `_phase_counter` on the sweep dict, resolution never touches ray.get_actor and falls back straight to the trial id's own `_NNNNN` index + 1, or None for a searcher-shaped id."""
        import phoenx.ray_tune as ray_tune_mod

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("must not contact the counter actor when _phase_counter is absent")

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _must_not_be_called)

        assert _resolve_trial_ordinal({}, "optimizers", "2c7b5_00006") == 7
        assert _resolve_trial_ordinal({}, "optimizers", "2c7b5_00000") == 1
        assert _resolve_trial_ordinal({}, "optimizers", "c6a5bab2") is None
        assert _resolve_trial_ordinal({}, "optimizers", "local") is None

    def test_resolve_ordinal_falls_back_and_logs_warning_when_actor_lookup_raises(self, monkeypatch, caplog):
        """C3(e): a raising `ray.get_actor` never escapes, and -- once a phase HAS a counter -- resolution must return `None` for BOTH id shapes, never falling through to the trial-id `_NNNNN` index. Actor ordinals are trial *start* order; the trial-id index is Ray's *creation* order. Mixing them on a dead counter collides: a trial that started third already holds actor ordinal 3, and a surviving trial `x_00002` would independently compute `int("00002")+1 == 3` from its own index -- two different runs both named `...-0003of0010`. The index fallback is legitimate ONLY when the phase has no counter at all (see the sibling test above), where it is the phase's only numbering and so is collision-free. Still logs a WARNING naming the phase and the trial id."""
        import phoenx.ray_tune as ray_tune_mod

        def _raise_runtime_error(name):
            raise RuntimeError("actor is gone")

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _raise_runtime_error)
        sweep = {"_phase_counter": "n"}

        with caplog.at_level(logging.WARNING):
            assert _resolve_trial_ordinal(sweep, "optimizers", "2c7b5_00006") is None
            assert _resolve_trial_ordinal(sweep, "optimizers", "c6a5bab2") is None

        warnings_ = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("optimizers" in msg and "2c7b5_00006" in msg for msg in warnings_)
        assert any("optimizers" in msg and "c6a5bab2" in msg for msg in warnings_)

    def test_resolve_ordinal_uses_stub_actor_result_looked_up_by_stashed_name(self, monkeypatch):
        """C3(f): when `_phase_counter` is present and the actor resolves, its (int) answer wins, and `ray.get_actor` is called with the exact stashed name."""
        import phoenx.ray_tune as ray_tune_mod

        get_actor_calls = []

        class _StubOrdinalRemote:
            def remote(self, trial_id):
                return ("ordinal-future", trial_id)

        class _StubActor:
            ordinal = _StubOrdinalRemote()

        def _fake_get_actor(name):
            get_actor_calls.append(name)
            return _StubActor()

        def _fake_ray_get(obj, timeout=None):
            assert obj == ("ordinal-future", "c6a5bab2")
            return 3

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _fake_get_actor)
        monkeypatch.setattr(ray_tune_mod.ray, "get", _fake_ray_get)

        result = _resolve_trial_ordinal({"_phase_counter": "n"}, "optimizers", "c6a5bab2")

        assert result == 3
        assert get_actor_calls == ["n"]

    def test_resolve_ordinal_converts_stub_actor_string_result_to_int(self, monkeypatch):
        """C3(g): a stub actor answering with the string "4" still resolves to the int 4."""
        import phoenx.ray_tune as ray_tune_mod

        class _StubOrdinalRemote:
            def remote(self, trial_id):
                return "ordinal-future"

        class _StubActor:
            ordinal = _StubOrdinalRemote()

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", lambda name: _StubActor())
        monkeypatch.setattr(ray_tune_mod.ray, "get", lambda obj, timeout=None: "4")

        result = _resolve_trial_ordinal({"_phase_counter": "n"}, "optimizers", "c6a5bab2")

        assert result == 4
        assert isinstance(result, int)

    # -------------------------------------------------------------------
    # C3b: _resolve_trial_ordinal never raises, for hostile input. The
    # with-counter cases below assert `None` rather than a trial-id-index
    # fallback.
    # -------------------------------------------------------------------
    @pytest.mark.parametrize(
        "trial_id,expected",
        [
            (None, None),
            (12345, None),
            (12.5, None),
            (["a"], None),
            (b"x", None),
            ("", None),
            ("___", None),
            ("_00001", 2),
            ("abc_", None),
            ("abc_\u00b2", None),  # str.isdigit() is True for U+00B2 but int() raises; the ASCII guard must reject it
            ("abc_" + "9" * 40, int("9" * 40) + 1),  # huge suffix: no overflow
        ],
    )
    def test_c3b_no_counter_hostile_trial_ids_never_raise(self, monkeypatch, trial_id, expected):
        """C3b: with no `_phase_counter` on the sweep, every one of these hostile `trial_id` shapes -- non-`str` types, all-underscore ids, a Unicode digit `int()` rejects, and a suffix too large to worry about overflow -- resolves without raising, returning `None` unless the trial-id's own `_NNNNN` index is genuinely present and ASCII-digit."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(
            ray_tune_mod.ray, "get_actor",
            lambda name: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        assert _resolve_trial_ordinal({}, "optimizers", trial_id) == expected

    def test_c3b_non_string_phase_counter_with_failing_lookup_returns_none(self, monkeypatch, caplog):
        """C3b: `sweep["_phase_counter"]` holding a non-string (e.g. a stray int from a malformed sweep dict) is still merely truthy to the `if counter_name:` check; a real `ray.get_actor` would reject it, that rejection must be swallowed (not propagate as a `TypeError`), and -- because the phase HAS a counter -- the result must be `None`, not a trial-id-index guess."""
        import phoenx.ray_tune as ray_tune_mod

        def _fake_get_actor(name):
            raise TypeError(f"actor name must be a string, got {type(name).__name__}")

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _fake_get_actor)

        with caplog.at_level(logging.WARNING):
            result = _resolve_trial_ordinal({"_phase_counter": 12345}, "optimizers", "2c7b5_00006")

        assert result is None

    def test_c3b_actor_ordinal_remote_call_itself_raising_returns_none(self, monkeypatch):
        """C3b: distinct from C3(e) (which raises inside `ray.get_actor`), this raises while evaluating `actor.ordinal.remote(trial_id)` itself -- i.e. before `ray.get` is even reached -- and must be caught by the same `except Exception` and resolve to `None`, not the trial-id index."""
        import phoenx.ray_tune as ray_tune_mod

        class _BrokenOrdinalRemote:
            def remote(self, trial_id):
                raise RuntimeError("remote call exploded synchronously")

        class _StubActor:
            ordinal = _BrokenOrdinalRemote()

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", lambda name: _StubActor())

        result = _resolve_trial_ordinal({"_phase_counter": "n"}, "optimizers", "abcd_00003")

        assert result is None

    def test_c3b_actor_answer_not_int_convertible_returns_none(self, monkeypatch):
        """C3b: the actor answering with a value `int()` rejects (e.g. a non-numeric string) must resolve to `None`, not fall back to the trial-id index -- once a phase has a counter, "the actor said something unusable" means "position unknown", not "ask the trial id instead"."""
        import phoenx.ray_tune as ray_tune_mod

        class _StubOrdinalRemote:
            def remote(self, trial_id):
                return "ordinal-future"

        class _StubActor:
            ordinal = _StubOrdinalRemote()

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", lambda name: _StubActor())
        monkeypatch.setattr(ray_tune_mod.ray, "get", lambda obj, timeout=None: "not-a-number")

        result = _resolve_trial_ordinal({"_phase_counter": "n"}, "optimizers", "abcd_00003")

        assert result is None

    # -------------------------------------------------------------------
    # `_format_trial_ordinal` when the ordinal is WIDER than the total --
    # an errored trial still consumed an ordinal, so ordinal > total is a
    # legitimate runtime state, not just an input-validation curiosity.
    # -------------------------------------------------------------------
    @pytest.mark.parametrize(
        "ordinal,total,expected",
        [
            (15, 10, "0015of0010"),
            (15000, 10, "15000of0010"),
        ],
    )
    def test_format_trial_ordinal_wider_than_total_is_not_truncated(self, ordinal, total, expected):
        """An ordinal wider than `total` renders in full (Python's `d` format only pads to a MINIMUM width; it never truncates), so a phase whose planned total undershoots reality still gets a sane, readable run-name suffix instead of a truncated or raising one."""
        assert _format_trial_ordinal(ordinal, total) == expected


# =============================================================================
# 4c. _phase_trial_total
# =============================================================================
class TestPhaseTrialTotal:
    @pytest.mark.parametrize(
        "space,num_samples,expand_grid,num_seed_points,expected",
        [
            ({}, 10, True, 0, 10),
            ({"a": {"grid_search": [1, 2, 3]}}, 10, True, 0, 30),
            ({"a": {"grid_search": [1, 2]}, "b": {"grid_search": [1, 2, 3]}}, 5, True, 0, 30),
            ({"a": {"grid_search": [1, 2]}, "b": {"grid_search": [1, 2, 3]}}, 5, False, 0, 5),
            ({}, -1, True, 0, None),
            ({}, 0, True, 0, None),
            ({}, True, True, 0, None),
            ({}, None, True, 0, None),
            ({"a": 0.5}, 10, True, 0, 10),
            ({"a": {"grid_search": []}}, 10, True, 0, 10),
            ({"a": {"grid_search": [1, 2], "other": 1}}, 10, True, 0, 10),
            ({}, 10, True, 10, 10),
            ({}, 10, True, 11, None),
            ({"a": {"grid_search": [1, 2, 3]}}, 10, True, 11, None),
        ],
    )
    def test_phase_trial_total_matrix(self, space, num_samples, expand_grid, num_seed_points, expected):
        """C4: `_phase_trial_total` expands grid_search dimensions only when `expand_grid`, refuses a non-positive/non-int `num_samples`, and refuses to bound a phase whose seed points can overshoot `num_samples`."""
        assert _phase_trial_total(
            space, num_samples, expand_grid=expand_grid, num_seed_points=num_seed_points
        ) == expected

    @pytest.mark.parametrize(
        "grid_search_value",
        [5, None, 3.14, True],
    )
    def test_unsized_grid_search_value_is_skipped_not_raising(self, grid_search_value):
        """A malformed `grid_search` value with no `len()` (an int/None/float/bool instead of a list) hits the `except TypeError: continue` branch -- the matrix above only covers proper lists (including the empty-list case); a scalar `grid_search:` value is a plausible hand-authored YAML typo this must not crash on."""
        space = {"a": {"grid_search": grid_search_value}}
        assert _phase_trial_total(space, 10, expand_grid=True, num_seed_points=0) == 10


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
        """The Ray trial id is no longer embedded in `run_name` (it moves to `extra_config["ray/trial_id"]`); `group`/`tags`/`sweep_params` are unchanged."""
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
        # "abcd_00003" has 0-based suffix 3 -> 1-based ordinal 4; no
        # "_phase_trial_total" on this sweep, so the name carries a bare
        # ordinal with no "of<total>" suffix.
        assert wcfg["run_name"] == "my_sweep-architecture-0004"
        assert wcfg["extra_config"]["ray/trial_id"] == "abcd_00003"
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

    def _wandb_config_for(self, monkeypatch, trial_id, sweep, phase_name="optimizers"):
        """Run `_inject_trial_callbacks` under a fake trial id and return the injected WandbCallback's config dict."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(ray_tune_mod.tune, "get_context", lambda: _FakeTrialContext(trial_id))
        config = {}
        phase = {"name": phase_name, "wandb": {"project": "p"}}
        _inject_trial_callbacks(config, sweep, phase, sampled={})
        return next(cb["config"] for cb in config["callbacks"] if cb["type"] == "WandbCallback")

    def test_c5_run_name_is_sweep_phase_ordinal_and_degrades_to_trial_id_only_when_no_ordinal(self, monkeypatch):
        """C5: `run_name` is `f"{sweep}-{phase}-{ordinal}"` with nothing after the ordinal, degrading to the raw trial id only when no ordinal resolves."""
        sweep_with_total = {"_sweep_name": "ppo_sweep", "_phase_trial_total": 10}
        assert (
            self._wandb_config_for(monkeypatch, "2c7b5_00006", sweep_with_total)["run_name"]
            == "ppo_sweep-optimizers-0007of0010"
        )
        assert (
            self._wandb_config_for(monkeypatch, "c6a5bab2", sweep_with_total)["run_name"]
            == "ppo_sweep-optimizers-c6a5bab2"
        )

        sweep_no_total = {"_sweep_name": "ppo_sweep"}
        assert (
            self._wandb_config_for(monkeypatch, "2c7b5_00006", sweep_no_total)["run_name"]
            == "ppo_sweep-optimizers-0007"
        )

    def test_c5b_searcher_shaped_trial_id_still_gets_an_ordinal_via_the_counter_actor(self, monkeypatch):
        """C5b (THE HEADLINE CONTRACT): a searcher-shaped trial id, which carries no `_NNNNN` index, still gets the ordinal name -- proving `_inject_trial_callbacks` actually reaches the counter actor rather than only ever parsing the trial id. The actor's answer must win over the trial id's own index, unswapped."""
        import phoenx.ray_tune as ray_tune_mod

        class _StubOrdinalRemote:
            def remote(self, trial_id):
                return "ordinal-future"

        class _StubActor:
            ordinal = _StubOrdinalRemote()

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", lambda name: _StubActor())
        monkeypatch.setattr(ray_tune_mod.ray, "get", lambda obj, timeout=None: 3)

        sweep = {"_sweep_name": "ppo_sweep", "_phase_counter": "actor-name", "_phase_trial_total": 10}

        wcfg = self._wandb_config_for(monkeypatch, "c6a5bab2", sweep)
        assert wcfg["run_name"] == "ppo_sweep-optimizers-0003of0010"
        assert wcfg["extra_config"]["ray/trial_ordinal"] == 3
        assert wcfg["extra_config"]["ray/trial_id"] == "c6a5bab2"

        # A random-shaped id under the SAME stub still yields the actor's
        # answer -- the actor wins, not silently swapped for the id's index.
        wcfg2 = self._wandb_config_for(monkeypatch, "2c7b5_00099", sweep)
        assert wcfg2["run_name"] == "ppo_sweep-optimizers-0003of0010"

    def test_c6_group_and_tags_unchanged_in_both_the_ordinal_and_fallback_case(self, monkeypatch):
        """C6: `group`/`tags` are exact-equality unchanged by the ordinal feature, in both the ordinal case and the fallback case."""
        sweep_name, phase_name = "s", "optimizers"

        wcfg_fallback = self._wandb_config_for(monkeypatch, "c6a5bab2", {"_sweep_name": sweep_name}, phase_name)
        assert wcfg_fallback["group"] == f"{sweep_name}-{phase_name}"
        assert wcfg_fallback["tags"] == [sweep_name, phase_name]

        sweep_with_ordinal = {"_sweep_name": sweep_name, "_phase_trial_total": 10}
        wcfg_ordinal = self._wandb_config_for(monkeypatch, "2c7b5_00006", sweep_with_ordinal, phase_name)
        assert wcfg_ordinal["group"] == f"{sweep_name}-{phase_name}"
        assert wcfg_ordinal["tags"] == [sweep_name, phase_name]

    def test_c7_extra_config_always_carries_ray_id_and_phase_ordinal_only_when_resolved(self, monkeypatch):
        """C7: `extra_config` always carries `ray/trial_id` and `ray/phase`; `ray/trial_ordinal` (an int, not a bool) is present only when an ordinal resolved. Nothing lands under `sweep/`, and the whole config is JSON-serializable."""
        sweep = {"_sweep_name": "ppo_sweep"}

        wcfg = self._wandb_config_for(monkeypatch, "2c7b5_00006", sweep)
        extra = wcfg["extra_config"]
        assert extra == {"ray/trial_id": "2c7b5_00006", "ray/phase": "optimizers", "ray/trial_ordinal": 7}
        assert isinstance(extra["ray/trial_ordinal"], int)
        assert not isinstance(extra["ray/trial_ordinal"], bool)
        assert not any(key.startswith("sweep/") for key in extra)
        json.dumps(wcfg)  # must not raise

        wcfg2 = self._wandb_config_for(monkeypatch, "c6a5bab2", sweep)
        extra2 = wcfg2["extra_config"]
        assert "ray/trial_ordinal" not in extra2
        assert extra2["ray/trial_id"] == "c6a5bab2"
        assert not any(key.startswith("sweep/") for key in extra2)
        json.dumps(wcfg2)  # must not raise

    def test_c12_phase_with_no_wandb_block_never_contacts_the_counter_actor(self, monkeypatch):
        """C12: a phase declaring no `wandb:` block never calls `ray.get_actor` at all, even when the sweep has a live `_phase_counter`."""
        import phoenx.ray_tune as ray_tune_mod

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("must not contact the counter actor without a wandb: block")

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _must_not_be_called)

        config = {"callbacks": []}
        sweep = {"_sweep_name": "s", "_phase_counter": "some-actor"}
        phase = {"name": "refine"}
        _inject_trial_callbacks(config, sweep, phase, sampled={})

        assert [cb for cb in config["callbacks"] if cb.get("type") == "WandbCallback"] == []
        assert len([cb for cb in config["callbacks"] if cb.get("type") == "RayTuneCallback"]) == 1

    def test_c13_dead_counter_degrades_without_raising_and_logs_a_warning(self, monkeypatch, caplog):
        """C13: a dead/unreachable actor never fails a trial, and -- because the phase HAS a counter -- BOTH id shapes must degrade all the way to the raw trial id, with NO `ray/trial_ordinal`. A random-shaped id must NOT fall back to its own `_NNNNN` index once a counter is configured: actor ordinals are trial *start* order and the trial-id index is Ray's *creation* order, so inventing a number from the id after the actor has died would collide with start-order ordinals the actor already issued to other trials before it died (contrast C5, where NO counter is configured and the index fallback is the phase's only, collision-free, numbering). Both cases log a WARNING naming the phase and the trial id."""
        import phoenx.ray_tune as ray_tune_mod

        def _raise_runtime_error(name):
            raise RuntimeError("actor is gone")

        monkeypatch.setattr(ray_tune_mod.ray, "get_actor", _raise_runtime_error)
        sweep = {"_sweep_name": "ppo_sweep", "_phase_counter": "gone", "_phase_trial_total": 10}

        with caplog.at_level(logging.WARNING):
            wcfg_a = self._wandb_config_for(monkeypatch, "2c7b5_00006", sweep)
        assert wcfg_a["run_name"] == "ppo_sweep-optimizers-2c7b5_00006"
        assert "ray/trial_ordinal" not in wcfg_a["extra_config"]
        assert wcfg_a["extra_config"]["ray/trial_id"] == "2c7b5_00006"
        assert any(
            "optimizers" in rec.message and "2c7b5_00006" in rec.message
            for rec in caplog.records if rec.levelno == logging.WARNING
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            wcfg_b = self._wandb_config_for(monkeypatch, "c6a5bab2", sweep)
        assert wcfg_b["run_name"] == "ppo_sweep-optimizers-c6a5bab2"
        assert "ray/trial_ordinal" not in wcfg_b["extra_config"]
        assert wcfg_b["extra_config"]["ray/trial_id"] == "c6a5bab2"
        assert any(
            "optimizers" in rec.message and "c6a5bab2" in rec.message
            for rec in caplog.records if rec.levelno == logging.WARNING
        )

    def test_extra_config_survives_build_trainer_from_config_and_get_config_round_trip(
        self, monkeypatch, tmp_path, force_cpu
    ):
        """Integration check: the `extra_config` `_inject_trial_callbacks` builds
        must survive a real `build_trainer_from_config` -> `Trainer.get_config()`
        round trip (`_inject_trial_callbacks`'s own docstring names `rl_trainable`
        as its caller, and `rl_trainable` immediately hands the mutated config to
        `build_trainer_from_config`), not just a `WandbCallback.get_config()`
        round trip in isolation (already pinned by C9 in `test_rl_callbacks.py`).
        Also confirms the WHOLE resulting trial config -- `sweep_params` and
        `extra_config` together -- is still `json.dumps`-able, matching what
        `Trainer.save()`'s plain `json.dump` needs mid-run. This is a
        cross-module path (`ray_tune` -> `builder` -> `rl_callbacks`) invisible
        from any single function's contract text, so it needs its own
        integration-level test."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(ray_tune_mod.tune, "get_context", lambda: _FakeTrialContext("2c7b5_00006"))

        config = _tiny_cartpole_config(save_dir=str(tmp_path / "trial_00007") + "/")
        # A plain float: the numpy-scalar-in-`discount` concern is this
        # helper's own pre-existing purpose (see `TestWriteBestConfig`) and
        # is orthogonal to what this test pins.
        config["agent"]["config"]["discount"] = 0.99
        sweep = {"_sweep_name": "ppo_sweep", "_phase_trial_total": 10}
        phase = {"name": "optimizers", "wandb": {"project": "p"}}
        _inject_trial_callbacks(config, sweep, phase, sampled={"agent.config.discount": 0.97})

        trainer = build_trainer_from_config(config)
        try:
            trainer_config = trainer.get_config()
            wandb_cbs = [cb for cb in trainer_config["callbacks"] if cb["type"] == "WandbCallback"]
            assert len(wandb_cbs) == 1
            wcfg = wandb_cbs[0]["config"]
            assert wcfg["sweep_params"] == {"agent.config.discount": 0.97}
            extra = wcfg["extra_config"]
            assert extra["ray/trial_id"] == "2c7b5_00006"
            assert extra["ray/trial_ordinal"] == 7
            assert extra["ray/phase"] == "optimizers"
            json.dumps(trainer_config)  # must not raise
        finally:
            trainer.env.close()


# =============================================================================
# 5b. run_phase's counter-actor lifecycle wiring -- no Ray cluster.
# =============================================================================
class TestRunPhaseCounterLifecycle:
    """`run_phase` stashes the counter NAME (never its handle) plus a phase
    trial total on the trial sweep dict, and kills the counter in a
    `finally` that spans Tuner construction/restore through `tuner.fit()`.
    """

    def _patch(
        self, monkeypatch, *, fit_exc=None, tuner_init_exc=None, restore_exc=None, can_restore=True,
    ):
        """Install the shared no-Ray-cluster harness; returns (sentinel_handle, stop_calls, with_parameters_calls)."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(ray_tune_mod, "_ensure_ray_initialized", lambda sweep: None)
        monkeypatch.setattr(
            ray_tune_mod, "resolve_trial_config",
            lambda sweep, phase, sampled, base_config: {"resolved": True},
        )
        monkeypatch.setattr(ray_tune_mod, "write_best_config", lambda config, path: None)

        sentinel_handle = object()
        monkeypatch.setattr(
            ray_tune_mod, "_start_phase_counter",
            lambda sweep_name, phase_name: (sentinel_handle, "actor-name"),
        )
        stop_calls = []
        monkeypatch.setattr(ray_tune_mod, "_stop_phase_counter", lambda handle: stop_calls.append(handle))

        with_parameters_calls = []

        def _fake_with_parameters(fn, **kwargs):
            with_parameters_calls.append(kwargs)
            return SimpleNamespace(kwargs=kwargs)

        monkeypatch.setattr(ray_tune_mod.tune, "with_parameters", _fake_with_parameters)
        monkeypatch.setattr(ray_tune_mod.tune, "with_resources", lambda trainable, resources: trainable)

        fake_result = SimpleNamespace(metrics={"config": {"a": 1}, "avg_reward": 5.0}, error=None)

        class _FakeResultGrid:
            num_errors = 0
            errors = []

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return fake_result

            def get_best_result(self, metric, mode):
                return fake_result

        fake_result_grid = _FakeResultGrid()

        class _FakeTuner:
            def __init__(self, *args, **kwargs):
                if tuner_init_exc is not None:
                    raise tuner_init_exc

            def fit(self):
                if fit_exc is not None:
                    raise fit_exc
                return fake_result_grid

            @staticmethod
            def can_restore(path):
                return can_restore

            @classmethod
            def restore(cls, *args, **kwargs):
                if restore_exc is not None:
                    raise restore_exc
                return cls()

        monkeypatch.setattr(ray_tune_mod.tune, "Tuner", _FakeTuner)

        return sentinel_handle, stop_calls, with_parameters_calls

    def _grid_phase(self, search_alg_type=None, *, wandb=False):
        phase = {
            "name": "p1",
            "search_space": {
                "agent.config.discount": {"dist": "grid_search", "values": [0.9, 0.95, 0.99]},
            },
            "tune": {"num_samples": 10},
        }
        if search_alg_type is not None:
            phase["tune"]["search_alg"] = {"type": search_alg_type}
        if wandb:
            phase["wandb"] = {"project": "p"}
        return phase

    @staticmethod
    def _contains_value(obj, target, _seen=None):
        """Recursively search dict/list/tuple/set structures (by identity) for `target`."""
        if _seen is None:
            _seen = set()
        if id(obj) in _seen:
            return False
        _seen.add(id(obj))
        if obj is target:
            return True
        if isinstance(obj, dict):
            return any(
                TestRunPhaseCounterLifecycle._contains_value(k, target, _seen)
                or TestRunPhaseCounterLifecycle._contains_value(v, target, _seen)
                for k, v in obj.items()
            )
        if isinstance(obj, (list, tuple, set)):
            return any(TestRunPhaseCounterLifecycle._contains_value(v, target, _seen) for v in obj)
        return False

    @pytest.mark.parametrize(
        "search_alg_type,expected_total",
        [(None, 30), ("optuna", 10), ("variant_generator", 30)],
    )
    def test_c11a_stashes_counter_name_and_correctly_gated_phase_trial_total(
        self, monkeypatch, tmp_path, search_alg_type, expected_total
    ):
        """C11a: `run_phase` stashes the counter's NAME (never its handle, checked by walking the ENTIRE captured `with_parameters` kwargs recursively, since an actor handle can reach the object store from any depth, not just the top level), the sweep name, and a phase trial total that expands a 3-value grid only for `random`/`variant_generator` searchers -- not for e.g. `optuna`. All three phases now declare a `wandb:` block (C16: with no `wandb:` block, `run_phase` must not even call `_start_phase_counter`, so a wandb-less phase here would no longer exercise this contract once that gating lands)."""
        sentinel_handle, _stop_calls, calls = self._patch(monkeypatch)
        phase = self._grid_phase(search_alg_type, wandb=True)

        run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")

        assert len(calls) == 1
        captured_sweep = calls[0]["sweep"]
        assert captured_sweep["_sweep_name"] == "s"
        assert captured_sweep["_phase_counter"] == "actor-name"
        assert isinstance(captured_sweep["_phase_counter"], str)
        assert captured_sweep["_phase_trial_total"] == expected_total
        assert not self._contains_value(calls[0], sentinel_handle)

    def test_c11a_num_seed_points_gated_to_random_searcher_only(self, monkeypatch, tmp_path):
        """`num_seed_points` (the overshoot-bail-out input to `_phase_trial_total`) must be counted only for `search_alg.type == "random"` -- verified against Ray 2.52, only `BasicVariantGenerator` can run more trials than `num_samples`, and only when actually handed presets; `_POINTS_TO_EVALUATE_SUPPORTED` drops a `variant_generator` phase's seeds outright, and every other searcher's `SearchGenerator.is_finished` hard-caps at `num_samples`. An `optuna` phase seeded with 11 points and `num_samples=10` must therefore stash `_phase_trial_total == 10` (the seed count must NOT be forwarded and trigger the overshoot-guard's `None`)."""
        _sentinel_handle, _stop_calls, calls = self._patch(monkeypatch)
        phase = {
            "name": "p1",
            "tune": {"num_samples": 10, "search_alg": {"type": "optuna"}},
            "wandb": {"project": "p"},
        }
        points_to_evaluate = [{"agent.config.discount": 0.9 + 0.001 * i} for i in range(11)]

        run_phase(
            sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s",
            points_to_evaluate=points_to_evaluate,
        )

        assert calls[0]["sweep"]["_phase_trial_total"] == 10

    def test_c11b_counter_killed_exactly_once_when_fit_raises(self, monkeypatch, tmp_path):
        """C11b: `run_phase` kills the counter, exactly once, with the handle it was given, when `tuner.fit()` raises -- and the error still propagates."""
        sentinel_handle, stop_calls, _calls = self._patch(monkeypatch, fit_exc=RuntimeError("fit boom"))
        phase = self._grid_phase(wandb=True)

        with pytest.raises(RuntimeError, match="fit boom"):
            run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")

        assert stop_calls == [sentinel_handle]

    def test_c11c_counter_killed_when_tuner_construction_raises(self, monkeypatch, tmp_path):
        """C11c: `run_phase` kills the counter when Tuner CONSTRUCTION raises, not only when `fit()` does -- a `try/finally` that wraps only `fit()` would fail this."""
        sentinel_handle, stop_calls, _calls = self._patch(monkeypatch, tuner_init_exc=RuntimeError("construct boom"))
        phase = self._grid_phase(wandb=True)

        with pytest.raises(RuntimeError, match="construct boom"):
            run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")

        assert stop_calls == [sentinel_handle]

    def test_c11c_counter_killed_when_tuner_restore_raises_on_resume(self, monkeypatch, tmp_path):
        """C11c: `run_phase` kills the counter when `Tuner.restore` raises on the `--resume` path, not only on a fresh-Tuner construction failure."""
        sentinel_handle, stop_calls, _calls = self._patch(
            monkeypatch, restore_exc=RuntimeError("restore boom"), can_restore=True,
        )
        phase = self._grid_phase(wandb=True)

        with pytest.raises(RuntimeError, match="restore boom"):
            run_phase(
                sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s", resume=True,
            )

        assert stop_calls == [sentinel_handle]

    def test_c16_no_wandb_block_never_starts_a_counter(self, monkeypatch, tmp_path):
        """C16: `run_phase` must create NO counter actor for a phase with no `wandb:` block -- every phase of every sweep paying for a counter round trip (and blocking the driver on it) it will never read from would break the guarantee that a sweep with no `wandb:` block behaves exactly as it did before counters existed. Uses a call-COUNTING recorder in place of `_start_phase_counter` rather than the unconditional lambda `_patch` installs, so a call that happens despite the missing `wandb:` block is visible. Also asserts that the sweep dict handed to the trainable carries no live `_phase_counter`, so `_resolve_trial_ordinal` cannot mistake this phase for one that has a counter configured."""
        import phoenx.ray_tune as ray_tune_mod

        _sentinel_handle, _stop_calls, calls = self._patch(monkeypatch)
        start_calls = []

        def _counting_start(sweep_name, phase_name):
            start_calls.append((sweep_name, phase_name))
            return object(), "actor-name"

        monkeypatch.setattr(ray_tune_mod, "_start_phase_counter", _counting_start)

        phase = {"name": "p1", "tune": {"num_samples": 10}}  # deliberately no "wandb" key

        run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")

        assert start_calls == []
        assert len(calls) == 1
        captured_sweep = calls[0]["sweep"]
        assert captured_sweep.get("_phase_counter") is None

    def test_c16_wandb_block_present_starts_exactly_one_counter(self, monkeypatch, tmp_path):
        """C16 (contrast case): a phase that DOES declare `wandb:` still starts exactly one counter, and its name is still what gets stashed on the trial sweep -- the gate must not also disable the already-working path."""
        import phoenx.ray_tune as ray_tune_mod

        _sentinel_handle, _stop_calls, calls = self._patch(monkeypatch)
        start_calls = []

        def _counting_start(sweep_name, phase_name):
            start_calls.append((sweep_name, phase_name))
            return object(), "actor-name"

        monkeypatch.setattr(ray_tune_mod, "_start_phase_counter", _counting_start)

        phase = {"name": "p1", "tune": {"num_samples": 10}, "wandb": {"project": "p"}}

        run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")

        assert len(start_calls) == 1
        assert calls[0]["sweep"]["_phase_counter"] == "actor-name"

    def test_c17_resume_restore_stashes_phase_trial_total_none(self, monkeypatch, tmp_path):
        """C17: on the `--resume` restore branch, `run_phase` stashes `_phase_trial_total = None` -- `Tuner.restore` takes neither `param_space` nor `tune_config`, so Ray replays the persisted experiment spec and this invocation's `num_samples` may not be what actually runs (a restored `BasicVariantGenerator` also continues its own counter); printing a total we cannot bound is worse than printing none."""
        _sentinel_handle, _stop_calls, calls = self._patch(monkeypatch, can_restore=True)
        phase = self._grid_phase(wandb=True)  # num_samples=10, random searcher, 3-value grid -> would stash 30 if not on the restore branch

        run_phase(
            sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s", resume=True,
        )

        assert calls[0]["sweep"]["_phase_trial_total"] is None

    def test_c14_stashed_private_keys_do_not_change_trainable_identity(self):
        """C14: adding `_phase_counter`/`_phase_trial_total` to the trial sweep dict does not change `Experiment.get_trainable_name`, so Ray's `--resume` restore path (which compares the pickled trainable name) still accepts it."""
        phase = {"name": "p1"}
        name_without = Experiment.get_trainable_name(
            tune.with_parameters(rl_trainable, sweep={"_sweep_name": "s"}, phase=phase, base_config={})
        )
        name_with = Experiment.get_trainable_name(
            tune.with_parameters(
                rl_trainable,
                sweep={"_sweep_name": "s", "_phase_counter": "a", "_phase_trial_total": 10},
                phase=phase,
                base_config={},
            )
        )
        assert name_without == name_with == "rl_trainable"


# =============================================================================
# 5c. _start_phase_counter / _stop_phase_counter / _phase_counter_name failure
# paths and name shape -- no Ray cluster. C11a-C11c (above) only drove these
# through `run_phase`'s wrapper with `_start_phase_counter` itself stubbed
# out entirely; these test the two private helpers directly.
# =============================================================================
class TestPhaseCounterLifecycleHelpers:
    def test_phase_counter_name_shape_and_uniqueness_across_start_calls(self, monkeypatch):
        """`_phase_counter_name`'s exact f-string shape, and that two
        `_start_phase_counter` calls for the SAME sweep/phase never collide
        (the random token is what prevents two concurrent sweeps under one
        Ray namespace from colliding on the same actor name)."""
        from phoenx.ray_tune import _phase_counter_name
        import phoenx.ray_tune as ray_tune_mod

        assert (
            _phase_counter_name("ppo_sweep", "optimizers", "a1b2c3d4")
            == "phoenx-trial-counter-ppo_sweep-optimizers-a1b2c3d4"
        )

        class _StubHandle:
            """Stands in for the actor handle `_start_phase_counter` calls `.issued.remote()` on."""

            def __init__(self, name):
                self.name = name
                self.issued = SimpleNamespace(remote=lambda: "issued-future")

        class _StubOptions:
            def __init__(self, name):
                self._name = name

            def remote(self):
                return _StubHandle(self._name)

        class _StubCounterClass:
            def options(self, name):
                return _StubOptions(name)

        monkeypatch.setattr(ray_tune_mod, "_PhaseTrialCounter", _StubCounterClass())
        monkeypatch.setattr(ray_tune_mod.ray, "get", lambda obj, timeout=None: 0)
        monkeypatch.setattr(
            ray_tune_mod.ray, "kill",
            lambda h: (_ for _ in ()).throw(AssertionError("readiness succeeds; must not kill")),
        )

        handle1, name1 = ray_tune_mod._start_phase_counter("s", "p")
        handle2, name2 = ray_tune_mod._start_phase_counter("s", "p")
        assert name1 != name2
        assert name1.startswith("phoenx-trial-counter-s-p-")
        assert name2.startswith("phoenx-trial-counter-s-p-")
        assert handle1.name == name1
        assert handle2.name == name2

    def test_start_phase_counter_returns_none_none_when_actor_creation_itself_raises(
        self, monkeypatch, caplog
    ):
        """`_PhaseTrialCounter.options(name=...).remote()` raising (e.g. the cluster is out of resources) must degrade to `(None, None)`, log a WARNING naming the phase, and never call `ray.kill` -- no actor handle was ever created, so there is nothing to kill."""
        import phoenx.ray_tune as ray_tune_mod

        class _StubOptions:
            def remote(self):
                raise RuntimeError("cannot create actor")

        class _StubCounterClass:
            def options(self, name):
                return _StubOptions()

        monkeypatch.setattr(ray_tune_mod, "_PhaseTrialCounter", _StubCounterClass())
        kill_calls = []
        monkeypatch.setattr(ray_tune_mod.ray, "kill", lambda h: kill_calls.append(h))

        with caplog.at_level(logging.WARNING):
            result = ray_tune_mod._start_phase_counter("s", "optimizers")

        assert result == (None, None)
        assert kill_calls == []
        assert any(
            "optimizers" in rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        )

    def test_start_phase_counter_kills_half_created_actor_when_readiness_wait_fails(
        self, monkeypatch, caplog
    ):
        """When actor creation SUCCEEDS but the readiness `ray.get(..., timeout=...)` raises (e.g. it times out), `_start_phase_counter` must still return `(None, None)` AND kill the half-created actor via the real `_stop_phase_counter` -- otherwise a phase that "fails to start a counter" would silently leak a live actor for the rest of the Ray session."""
        import phoenx.ray_tune as ray_tune_mod

        sentinel_handle = object()

        class _StubOptions:
            def remote(self):
                return sentinel_handle

        class _StubCounterClass:
            def options(self, name):
                return _StubOptions()

        monkeypatch.setattr(ray_tune_mod, "_PhaseTrialCounter", _StubCounterClass())
        monkeypatch.setattr(
            ray_tune_mod.ray, "get",
            lambda obj, timeout=None: (_ for _ in ()).throw(TimeoutError("actor not ready")),
        )
        kill_calls = []
        monkeypatch.setattr(ray_tune_mod.ray, "kill", lambda h: kill_calls.append(h))

        with caplog.at_level(logging.WARNING):
            result = ray_tune_mod._start_phase_counter("s", "optimizers")

        assert result == (None, None)
        assert kill_calls == [sentinel_handle]

    def test_stop_phase_counter_swallows_a_raising_ray_kill(self, monkeypatch, caplog):
        """`ray.kill(handle)` raising (e.g. the actor already died on its own) must not propagate out of `_stop_phase_counter` -- it runs from `run_phase`'s `finally`, where a raised exception would mask (or replace) the trial failure already being handled."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(
            ray_tune_mod.ray, "kill",
            lambda h: (_ for _ in ()).throw(RuntimeError("kill failed")),
        )

        with caplog.at_level(logging.DEBUG):
            ray_tune_mod._stop_phase_counter(object())  # must not raise

    def test_stop_phase_counter_none_handle_is_a_pure_noop(self, monkeypatch):
        """`_stop_phase_counter(None)` (the shape `_start_phase_counter` returns on its own failure path) must not touch `ray.kill` at all."""
        import phoenx.ray_tune as ray_tune_mod

        monkeypatch.setattr(
            ray_tune_mod.ray, "kill",
            lambda h: (_ for _ in ()).throw(AssertionError("must not be called for a None handle")),
        )
        ray_tune_mod._stop_phase_counter(None)


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

        # Short trial_<Ray-trial-id-suffix> dir names (mandatory on Windows:
        # Ray's default embeds every sampled param and blows past MAX_PATH
        # mid-sweep). This phase's `search_alg: {type: random}` mints
        # zero-padded numeric ids, so the suffix is always digits here; a
        # searcher phase would instead give `trial_<8-hex-char uuid>`.
        explore_phase_dir = tmp_path / "storage" / "e2e_cartpole" / "explore"
        trial_dirs = [p.name for p in explore_phase_dir.iterdir() if p.is_dir()]
        assert trial_dirs, f"no trial directories found under {explore_phase_dir}"
        for name in trial_dirs:
            assert re.fullmatch(r"trial_\d+", name), f"unexpected trial dir name: {name}"
