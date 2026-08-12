"""Unit tests for ``WandbCallback._ensure_wandb_login()`` in
``src/phoenx/rl_callbacks.py``.

Covers the credential-resolution order (env var -> key file -> wandb's own
``wandb.login`` fallback) and, in particular, guards against the regression
where a machine already authenticated via a cached ``wandb login``
(``~/.netrc``) was incorrectly rejected with a ``ValueError`` even though
``wandb.init()`` would have worked fine.

``wandb.login``/``wandb.run`` are always monkeypatched; no test in this file
ever touches the network, a real wandb session, or writes anything outside
``tmp_path``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import ray.tune as ray_tune
import wandb
import wandb.errors

from phoenx import rl_callbacks
from phoenx.rl_callbacks import RayTuneCallback, WandbCallback


class _RecordingLogin:
    """Stand-in for ``wandb.login`` that records calls and returns/raises as configured."""

    def __init__(self, return_value=True, exc: Exception | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value = return_value
        self.exc = exc

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.return_value


@pytest.fixture()
def callback() -> WandbCallback:
    return WandbCallback(project_name="phoenx-test-project")


@pytest.fixture()
def no_key_file(monkeypatch, tmp_path):
    """Redirect the module's key-file lookup into an empty ``tmp_path``.

    ``_ensure_wandb_login`` computes ``Path(__file__).with_name("wandb_api_key")``;
    replacing the ``Path`` symbol the module resolves against makes
    ``.with_name(...)`` land inside ``tmp_path`` instead of the real package
    directory, so no ``wandb_api_key`` file is ever created next to
    ``src/phoenx/rl_callbacks.py``.
    """
    monkeypatch.setattr(rl_callbacks, "Path", lambda _: tmp_path / "module.py")
    return tmp_path


# -----------------------------------------------------------------------------
# Contract 1: an active run short-circuits everything.
# -----------------------------------------------------------------------------
def test_active_run_returns_without_calling_login(monkeypatch, callback):
    fake_login = _RecordingLogin()
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)
    monkeypatch.setattr(rl_callbacks.wandb, "run", object())

    callback._ensure_wandb_login()

    assert fake_login.calls == []


# -----------------------------------------------------------------------------
# Contract 2: WANDB_API_KEY env var takes priority and is used verbatim.
# -----------------------------------------------------------------------------
def test_env_var_key_is_used_and_fallback_never_called(monkeypatch, callback):
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    fake_login = _RecordingLogin()
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    callback._ensure_wandb_login()

    assert fake_login.calls == [((), {"key": "env-key-123", "relogin": False})]


# -----------------------------------------------------------------------------
# Contract 3: no env var, but a non-empty key file next to the module.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "file_contents, expected_key",
    [
        ("file-key-456", "file-key-456"),
        ("  file-key-456  \n", "file-key-456"),
        ("\nfile-key-456\n\n", "file-key-456"),
    ],
)
def test_key_file_is_used_when_no_env_var(
    monkeypatch, callback, no_key_file, file_contents, expected_key
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    (no_key_file / "wandb_api_key").write_text(file_contents, encoding="utf-8")
    fake_login = _RecordingLogin()
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    try:
        callback._ensure_wandb_login()

        assert os.environ["WANDB_API_KEY"] == expected_key
        assert fake_login.calls == [((), {"key": expected_key, "relogin": False})]
    finally:
        # The production code writes directly to os.environ (not through
        # monkeypatch), so it must be cleaned up explicitly or it leaks into
        # later tests.
        monkeypatch.delenv("WANDB_API_KEY", raising=False)


# -----------------------------------------------------------------------------
# Edge case: the key file EXISTS but is empty or whitespace-only, so
# ``.strip()`` yields "". That must NOT be forwarded to wandb.login() as a
# key, and must NOT be written to os.environ; instead it falls through to
# the same fallback as "no key file at all" (contract 4/5/6's branch).
#
# Under BASE_REF this input took the *other* branch entirely: ``api_key``
# stayed falsy, so the old code's ``if api_key: ... else: raise ValueError(...)``
# hit the unconditional ``else`` and raised immediately -- ``wandb.login()``
# was never called at all. This test's success-returns-without-raising
# assertion, plus the recorded fallback call, would both fail under that
# implementation.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("file_contents", ["", "   \n"])
def test_empty_key_file_falls_through_to_fallback(
    monkeypatch, callback, no_key_file, file_contents
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    (no_key_file / "wandb_api_key").write_text(file_contents, encoding="utf-8")
    fake_login = _RecordingLogin(return_value=True)
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    callback._ensure_wandb_login()  # must not raise

    assert fake_login.calls == [
        ((), {"relogin": False, "timeout": rl_callbacks._WANDB_LOGIN_TIMEOUT})
    ]
    assert "key" not in fake_login.calls[0][1]
    assert "WANDB_API_KEY" not in os.environ


# -----------------------------------------------------------------------------
# Contract 4 (REGRESSION GUARD): no env var, no key file, but wandb's own
# fallback resolves credentials (e.g. a cached `wandb login` in ~/.netrc).
#
# Before the fix, this path never consulted wandb.login() at all and raised
# ValueError unconditionally whenever WANDB_API_KEY and the key file were
# both absent -- even on a machine already authenticated via `wandb login`.
# This test fails against that old implementation because it would raise
# ValueError here instead of returning.
# -----------------------------------------------------------------------------
def test_fallback_login_success_returns_without_raising(monkeypatch, callback, no_key_file):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    fake_login = _RecordingLogin(return_value=True)
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    callback._ensure_wandb_login()  # must not raise

    assert fake_login.calls == [
        ((), {"relogin": False, "timeout": rl_callbacks._WANDB_LOGIN_TIMEOUT})
    ]
    # No explicit key was ever supplied on this path.
    assert "key" not in fake_login.calls[0][1]
    assert "WANDB_API_KEY" not in os.environ


# -----------------------------------------------------------------------------
# Contract 5: no env var, no key file, wandb.login() returns falsy -> ValueError
# naming all three remedies.
# -----------------------------------------------------------------------------
def test_fallback_login_falsy_raises_value_error_with_all_remedies(
    monkeypatch, callback, no_key_file
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    fake_login = _RecordingLogin(return_value=False)
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    with pytest.raises(ValueError) as excinfo:
        callback._ensure_wandb_login()

    message = str(excinfo.value)
    assert "WANDB_API_KEY" in message
    assert "wandb login" in message
    assert str(no_key_file / "wandb_api_key") in message


# -----------------------------------------------------------------------------
# Contract 6: no env var, no key file, wandb.login() raises -> ValueError with
# the same message, chained from the original exception.
# -----------------------------------------------------------------------------
def test_fallback_login_exception_raises_chained_value_error(
    monkeypatch, callback, no_key_file
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(rl_callbacks.wandb, "run", None)
    original_exc = wandb.errors.UsageError("api_key not configured (no tty present)")
    fake_login = _RecordingLogin(exc=original_exc)
    monkeypatch.setattr(rl_callbacks.wandb, "login", fake_login)

    with pytest.raises(ValueError) as excinfo:
        callback._ensure_wandb_login()

    assert excinfo.value.__cause__ is original_exc
    message = str(excinfo.value)
    assert "WANDB_API_KEY" in message
    assert "wandb login" in message
    assert str(no_key_file / "wandb_api_key") in message


# -----------------------------------------------------------------------------
# Contract 7: ``WandbCallback.on_train_epoch_end``'s best-checkpoint artifact
# save, gated on ``logs.get("best", False)``.
#
# ``src/phoenx/trainer.py`` is the only producer of that ``"best"`` key
# (``episode_logs[-1]['best'] = True`` when a completed training episode
# beats the running-average best). These tests pin the consumer side: given
# the flag, the artifact save fires with ``model_is_best=True``; without it,
# ``save_model_artifact`` is never called at all.
# -----------------------------------------------------------------------------
def test_on_train_epoch_end_saves_best_artifact_when_logs_flag_best(
    monkeypatch, callback, tmp_path
):
    """A `best: True` log entry makes `on_train_epoch_end` save a
    `model_is_best=True` W&B artifact to `save_dir`."""
    monkeypatch.setattr(rl_callbacks.wandb, "log", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        rl_callbacks.wandb_support,
        "save_model_artifact",
        lambda *a, **k: calls.append((a, k)),
    )
    save_dir = str(tmp_path / "run")
    callback.save_dir = save_dir

    callback.on_train_epoch_end(epoch=3, logs={"avg_reward": 5.0, "best": True})

    assert calls == [((save_dir, "phoenx-test-project"), {"model_is_best": True})]
    assert os.path.isdir(save_dir)


def test_on_train_epoch_end_skips_artifact_without_best_key(
    monkeypatch, callback, tmp_path
):
    """No `best` key in the log dict (the pre-fix behavior, always) means the
    artifact save is never attempted."""
    monkeypatch.setattr(rl_callbacks.wandb, "log", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        rl_callbacks.wandb_support,
        "save_model_artifact",
        lambda *a, **k: calls.append((a, k)),
    )
    callback.save_dir = str(tmp_path / "run")

    callback.on_train_epoch_end(epoch=3, logs={"avg_reward": 5.0})

    assert calls == []


def test_on_train_epoch_end_treats_best_false_same_as_absent(
    monkeypatch, callback, tmp_path
):
    """An explicit `best: False` behaves identically to omitting the key —
    `.get("best", False)` treats both as falsy."""
    monkeypatch.setattr(rl_callbacks.wandb, "log", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        rl_callbacks.wandb_support,
        "save_model_artifact",
        lambda *a, **k: calls.append((a, k)),
    )
    callback.save_dir = str(tmp_path / "run")

    callback.on_train_epoch_end(epoch=3, logs={"avg_reward": 5.0, "best": False})

    assert calls == []


# =============================================================================
# WandbCallback (changed): new constructor kwargs, run-number call-count
# discipline, tag-list hygiene, sweep_params flattening, idempotent finish,
# and get_config() round-trip.
# =============================================================================


def _make_wandb_init_recorder():
    """Return a ``wandb.init`` stand-in recording kwargs and returning a stub run.

    The stub run exposes ``.name`` (the only attribute
    ``WandbCallback.initialize_run`` reads back), mirroring what a real
    ``wandb.init(...)`` call returns.
    """
    calls: list[dict] = []

    def _init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name=kwargs.get("name"))

    _init.calls = calls
    return _init


def _make_run_number_recorder(return_value: int):
    """Return a ``wandb_support.get_next_run_number`` stand-in that records calls."""
    calls: list[tuple[tuple, dict]] = []

    def _get_next_run_number(*args, **kwargs):
        calls.append((args, kwargs))
        return return_value

    _get_next_run_number.calls = calls
    return _get_next_run_number


# -----------------------------------------------------------------------------
# Backward compatibility: old positional construction and old YAML both work.
# -----------------------------------------------------------------------------
def test_backward_compat_positional_project_and_run_name():
    cb = WandbCallback("phoenx-test-project", "explicit-run")

    assert cb.project_name == "phoenx-test-project"
    assert cb.run_name == "explicit-run"
    assert cb.group is None
    assert cb.tags is None
    assert cb.run_id is None
    assert cb.resume is None
    assert cb.sweep_params is None


def test_backward_compat_project_name_only():
    cb = WandbCallback("phoenx-test-project")

    assert cb.project_name == "phoenx-test-project"
    assert cb.run_name is None


def test_legacy_yaml_config_loads_via_registry():
    cb = rl_callbacks.load(
        {"type": "WandbCallback", "config": {"project_name": "legacy-project"}}
    )

    assert isinstance(cb, WandbCallback)
    assert cb.project_name == "legacy-project"
    assert cb.run_name is None
    assert cb.group is None


# -----------------------------------------------------------------------------
# get_next_run_number call-count discipline: a 40-trial sweep with explicit
# run_name + group must make ZERO extra network calls; anything less
# fully-named still resolves the run number exactly once (shared between
# naming and grouping via the `nonlocal` cache in `_resolve_run_number`).
# -----------------------------------------------------------------------------
def test_run_number_zero_calls_when_run_name_and_group_both_explicit(
    monkeypatch, tmp_path
):
    cb = WandbCallback("proj", run_name="my-run", group="my-group")
    run_number_recorder = _make_run_number_recorder(999)
    monkeypatch.setattr(
        rl_callbacks.wandb_support, "get_next_run_number", run_number_recorder
    )
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {"type": "PPO"}},
        models=None,
        run_number=None,
        run_name_prefix="train",
        tags=["train"],
        job_type="train",
    )

    assert run_number_recorder.calls == []
    assert init_recorder.calls[0]["name"] == "my-run"
    assert init_recorder.calls[0]["group"] == "my-group"


def test_run_number_called_once_when_only_run_name_explicit(monkeypatch, tmp_path):
    """No explicit `group` still needs one lookup, because grouping falls
    back to `"group-{run_number}"`."""
    cb = WandbCallback("proj", run_name="my-run")
    run_number_recorder = _make_run_number_recorder(7)
    monkeypatch.setattr(
        rl_callbacks.wandb_support, "get_next_run_number", run_number_recorder
    )
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {}},
        models=None,
        run_number=None,
        run_name_prefix="train",
        tags=None,
        job_type="train",
    )

    assert len(run_number_recorder.calls) == 1
    assert init_recorder.calls[0]["name"] == "my-run"
    assert init_recorder.calls[0]["group"] == "group-7"


def test_run_number_called_once_when_nothing_named_and_shared_between_name_and_group(
    monkeypatch, tmp_path
):
    """With nothing named, the run number is resolved exactly ONCE and the
    same value backs both the legacy `"{prefix}-{n}"` name and
    `"group-{n}"` group (the `nonlocal` cache in `_resolve_run_number`)."""
    cb = WandbCallback("proj")
    run_number_recorder = _make_run_number_recorder(42)
    monkeypatch.setattr(
        rl_callbacks.wandb_support, "get_next_run_number", run_number_recorder
    )
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {}},
        models=None,
        run_number=None,
        run_name_prefix="train",
        tags=None,
        job_type="train",
    )

    assert len(run_number_recorder.calls) == 1
    assert init_recorder.calls[0]["name"] == "train-42"
    assert init_recorder.calls[0]["group"] == "group-42"


def test_explicit_run_number_argument_skips_network_call_entirely(
    monkeypatch, tmp_path
):
    """An explicit `run_number` argument (e.g. reused across train/test)
    short-circuits `_resolve_run_number` before it ever calls the network."""
    cb = WandbCallback("proj")
    run_number_recorder = _make_run_number_recorder(999)
    monkeypatch.setattr(
        rl_callbacks.wandb_support, "get_next_run_number", run_number_recorder
    )
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {}},
        models=None,
        run_number=5,
        run_name_prefix="train",
        tags=None,
        job_type="train",
    )

    assert run_number_recorder.calls == []
    assert init_recorder.calls[0]["name"] == "train-5"
    assert init_recorder.calls[0]["group"] == "group-5"


# -----------------------------------------------------------------------------
# Tags: fixed regression for `tags=tags.append(...)` (mutates caller's list
# and passes None to wandb.init). The fix must send a real list containing
# both the caller's tags and the agent type, and must not mutate either the
# constructor's or the per-call tags list.
# -----------------------------------------------------------------------------
def test_tags_combined_into_fresh_list_and_caller_lists_not_mutated(
    monkeypatch, tmp_path
):
    ctor_tags = ["custom"]
    cb = WandbCallback("proj", tags=ctor_tags)
    # Mutating the caller's original list after construction must not affect
    # the callback (it copies at __init__ time).
    ctor_tags.append("mutated-after-construction")

    per_call_tags = ["train"]
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {"type": "SAC"}},
        models=None,
        run_number=1,
        run_name_prefix="train",
        tags=per_call_tags,
        job_type="train",
    )

    sent_tags = init_recorder.calls[0]["tags"]
    assert sent_tags == ["custom", "train", "SAC"]
    assert isinstance(sent_tags, list)
    # Neither the constructor list nor the per-call list was mutated by the
    # combination logic itself.
    assert cb.tags == ["custom"]
    assert per_call_tags == ["train"]


def test_tags_omitted_at_construction_and_per_call_still_yields_agent_type_only(
    monkeypatch, tmp_path
):
    cb = WandbCallback("proj")
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {"type": "TD3"}},
        models=None,
        run_number=1,
        run_name_prefix="train",
        tags=None,
        job_type="train",
    )

    assert init_recorder.calls[0]["tags"] == ["TD3"]


# -----------------------------------------------------------------------------
# sweep_params: flattened under a "sweep/" prefix into wandb.init(config=...).
# -----------------------------------------------------------------------------
def test_sweep_params_flattened_under_sweep_prefix(monkeypatch, tmp_path):
    cb = WandbCallback("proj", sweep_params={"lr": 0.001, "nested": {"gamma": 0.99}})
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)

    cb.initialize_run(
        logs={"save_dir": str(tmp_path), "agent": {}},
        models=None,
        run_number=1,
        run_name_prefix="train",
        tags=None,
        job_type="train",
    )

    config = init_recorder.calls[0]["config"]
    assert config["sweep/lr"] == 0.001
    assert config["sweep/nested/gamma"] == 0.99
    # The original trainer-config keys are still present alongside the
    # flattened sweep params.
    assert config["save_dir"] == str(tmp_path)


def test_no_sweep_params_leaves_config_as_plain_logs(monkeypatch, tmp_path):
    cb = WandbCallback("proj")
    init_recorder = _make_wandb_init_recorder()
    monkeypatch.setattr(rl_callbacks.wandb, "init", init_recorder)
    logs = {"save_dir": str(tmp_path), "agent": {}}

    cb.initialize_run(
        logs=logs, models=None, run_number=1, run_name_prefix="train", tags=None,
        job_type="train",
    )

    assert init_recorder.calls[0]["config"] == logs
    assert not any(k.startswith("sweep/") for k in init_recorder.calls[0]["config"])


# -----------------------------------------------------------------------------
# wandb.finish() idempotency: the Trainer can call on_train_end/on_test_end
# once per episode finished on the final step, so more than one call must
# still finish the run exactly once.
# -----------------------------------------------------------------------------
def test_finish_run_idempotent_across_multiple_on_train_end_calls(
    monkeypatch, callback
):
    finish_calls = []
    monkeypatch.setattr(rl_callbacks.wandb, "finish", lambda: finish_calls.append(1))
    callback.initialized = True

    callback.on_train_end()
    callback.on_train_end()
    callback.on_train_end(logs={"episode_reward": 1.0})

    assert len(finish_calls) == 1


def test_finish_run_idempotent_across_multiple_on_test_end_calls(monkeypatch, callback):
    finish_calls = []
    monkeypatch.setattr(rl_callbacks.wandb, "finish", lambda: finish_calls.append(1))
    callback.initialized = True

    callback.on_test_end()
    callback.on_test_end(logs={"episode_reward": 1.0})

    assert len(finish_calls) == 1


# -----------------------------------------------------------------------------
# get_config() round-trips all seven constructor kwargs.
# -----------------------------------------------------------------------------
def test_get_config_round_trips_all_seven_kwargs():
    cb = WandbCallback(
        project_name="proj",
        run_name="run-1",
        group="grp-1",
        tags=["a", "b"],
        run_id="run-id-1",
        resume="allow",
        sweep_params={"lr": 0.1},
    )

    config = cb.get_config()

    assert config == {
        "type": "WandbCallback",
        "config": {
            "project_name": "proj",
            "run_name": "run-1",
            "group": "grp-1",
            "tags": ["a", "b"],
            "run_id": "run-id-1",
            "resume": "allow",
            "sweep_params": {"lr": 0.1},
        },
    }

    rebuilt = WandbCallback.load(config)
    assert rebuilt.project_name == "proj"
    assert rebuilt.run_name == "run-1"
    assert rebuilt.group == "grp-1"
    assert rebuilt.tags == ["a", "b"]
    assert rebuilt.run_id == "run-id-1"
    assert rebuilt.resume == "allow"
    assert rebuilt.sweep_params == {"lr": 0.1}


# =============================================================================
# RayTuneCallback (new): lazy `ray` import, cached session detection, a
# fixed report key set, suppression before the first episode, cadence
# throttling / aggregation-buffer reset, and the get_config()/load()/registry
# round-trip.
# =============================================================================


class _FakeTrialContext:
    """Stand-in for ``ray.tune.get_context()``'s return value."""

    def __init__(self, trial_id):
        self._trial_id = trial_id

    def get_trial_id(self):
        return self._trial_id


@pytest.fixture()
def ray_tune_callback() -> RayTuneCallback:
    return RayTuneCallback(every=100, unit="timestep")


def _patch_session(monkeypatch, trial_id):
    """Patch `ray.tune.get_context` to report an active/inactive session."""
    monkeypatch.setattr(ray_tune, "get_context", lambda: _FakeTrialContext(trial_id))


def _patch_report(monkeypatch):
    """Patch `ray.tune.report`; returns the list it appends call dicts to."""
    report_calls: list[dict] = []
    monkeypatch.setattr(ray_tune, "report", lambda metrics: report_calls.append(metrics))
    return report_calls


# -----------------------------------------------------------------------------
# __init__ validation.
# -----------------------------------------------------------------------------
def test_raytune_init_invalid_unit_raises_value_error():
    with pytest.raises(ValueError):
        RayTuneCallback(unit="bad_unit")


def test_raytune_init_defaults():
    cb = RayTuneCallback()
    assert cb.every == 50000
    assert cb.unit == "timestep"


# -----------------------------------------------------------------------------
# No module-level `ray` import. Two complementary, order-independent checks:
# (1) the production module's own namespace never binds a `ray`/`tune` name
#     at all, regardless of whether some OTHER test already imported ray
#     into sys.modules; (2) a fresh subprocess -- which starts with an empty
#     sys.modules and cannot be polluted by any other test in this session --
#     imports phoenx.rl_callbacks and constructs the callback, then asserts
#     `ray` never entered sys.modules.
# -----------------------------------------------------------------------------
def test_rl_callbacks_module_namespace_has_no_ray_or_tune_name():
    assert "ray" not in vars(rl_callbacks)
    assert "tune" not in vars(rl_callbacks)


def test_importing_rl_callbacks_does_not_import_ray_in_fresh_subprocess():
    script = (
        "import sys\n"
        "import phoenx.rl_callbacks as rl_callbacks\n"
        "rl_callbacks.RayTuneCallback()\n"
        "leaked = sorted(m for m in sys.modules if m == 'ray' or m.startswith('ray.'))\n"
        "assert not leaked, leaked\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# -----------------------------------------------------------------------------
# bind(trainer): stores the trainer reference and does nothing else.
# -----------------------------------------------------------------------------
def test_bind_stores_trainer_and_nothing_else(ray_tune_callback):
    sentinel_trainer = object()
    before = {k: v for k, v in vars(ray_tune_callback).items() if k != "_trainer"}

    ray_tune_callback.bind(sentinel_trainer)

    assert ray_tune_callback._trainer is sentinel_trainer
    after = {k: v for k, v in vars(ray_tune_callback).items() if k != "_trainer"}
    assert before == after


# -----------------------------------------------------------------------------
# Outside a Tune session: harmless no-op, session detection cached.
# -----------------------------------------------------------------------------
def test_outside_session_never_reports(monkeypatch, ray_tune_callback):
    _patch_session(monkeypatch, trial_id=None)
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=10,
        logs={
            "avg_reward": 1.0,
            "episode_reward": 1.0,
            "episode_steps": 5,
            "episode": 1,
        },
    )
    ray_tune_callback.on_train_step_end(step=10_000_000, logs={"step_reward": 1.0})

    assert report_calls == []


def test_session_detection_is_cached_after_first_check(monkeypatch, ray_tune_callback):
    get_context_calls: list[int] = []

    def fake_get_context():
        get_context_calls.append(1)
        return _FakeTrialContext("trial-1")

    monkeypatch.setattr(ray_tune, "get_context", fake_get_context)
    _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_step_end(step=1, logs={})
    ray_tune_callback.on_train_step_end(step=2, logs={})

    assert len(get_context_calls) == 1


# -----------------------------------------------------------------------------
# Suppression until the first episode completes.
# -----------------------------------------------------------------------------
def test_reporting_suppressed_until_first_episode_completes(monkeypatch, ray_tune_callback):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    # Cadence has long since elapsed, but no episode has completed yet.
    ray_tune_callback.on_train_step_end(step=1_000_000, logs={"step_reward": 1.0})
    assert report_calls == []

    ray_tune_callback.on_train_epoch_end(
        epoch=1_000_000,
        logs={
            "avg_reward": 3.0,
            "episode_reward": 3.0,
            "episode_steps": 20,
            "episode": 1,
        },
    )
    ray_tune_callback.on_train_step_end(step=1_000_001, logs={"step_reward": 1.0})

    assert len(report_calls) == 1


# -----------------------------------------------------------------------------
# Fixed report key set, including the success_rate=0.0 default.
# -----------------------------------------------------------------------------
def test_report_has_fixed_key_set_with_empty_step_logs(monkeypatch, ray_tune_callback):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=0,
        logs={
            "avg_reward": 2.0,
            "episode_reward": 2.0,
            "episode_steps": 10,
            "episode": 1,
        },
    )
    ray_tune_callback.on_train_step_end(step=100, logs=None)

    assert len(report_calls) == 1
    metrics = report_calls[0]
    assert set(metrics.keys()) == {
        "timestep",
        "episodes",
        "avg_reward",
        "success_rate",
        "episode_reward",
        "episode_steps",
    }
    assert metrics["timestep"] == 100
    assert metrics["episodes"] == 1
    assert metrics["avg_reward"] == 2.0
    assert metrics["success_rate"] == rl_callbacks._DEFAULT_SUCCESS_RATE
    assert metrics["episode_reward"] == 2.0
    assert metrics["episode_steps"] == 10


def test_report_uses_success_rate_from_epoch_log_when_present(
    monkeypatch, ray_tune_callback
):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=0,
        logs={
            "avg_reward": 2.0,
            "episode_reward": 2.0,
            "episode_steps": 10,
            "episode": 1,
            "success_rate": 0.75,
        },
    )
    ray_tune_callback.on_train_step_end(step=100, logs=None)

    assert report_calls[0]["success_rate"] == 0.75


# -----------------------------------------------------------------------------
# Aggregation: mean over the interval, skipping bools / non-numeric /
# non-finite values so they cannot poison the aggregate.
# -----------------------------------------------------------------------------
def test_step_log_aggregation_means_numeric_and_skips_bad_values(
    monkeypatch, ray_tune_callback
):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=0,
        logs={
            "avg_reward": 1.0,
            "episode_reward": 1.0,
            "episode_steps": 5,
            "episode": 1,
        },
    )
    ray_tune_callback.on_train_step_end(
        step=10,
        logs={
            "policy_loss": 1.0,
            "flag": True,
            "label": "not-a-number",
            "bad_nan": float("nan"),
            "bad_inf": float("inf"),
        },
    )
    ray_tune_callback.on_train_step_end(step=100, logs={"policy_loss": 3.0})

    assert len(report_calls) == 1
    metrics = report_calls[0]
    assert metrics["policy_loss"] == pytest.approx(2.0)
    assert "flag" not in metrics
    assert "label" not in metrics
    assert "bad_nan" not in metrics
    assert "bad_inf" not in metrics


def test_aggregation_buffers_reset_after_each_report(monkeypatch, ray_tune_callback):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=0,
        logs={
            "avg_reward": 1.0,
            "episode_reward": 1.0,
            "episode_steps": 5,
            "episode": 1,
        },
    )

    ray_tune_callback.on_train_step_end(step=50, logs={"x": 10.0})
    ray_tune_callback.on_train_step_end(step=100, logs={"x": 20.0})  # reports: mean 15
    ray_tune_callback.on_train_step_end(step=150, logs={"x": 5.0})
    ray_tune_callback.on_train_step_end(step=200, logs={"x": 7.0})  # reports: mean 6

    assert len(report_calls) == 2
    assert report_calls[0]["x"] == pytest.approx(15.0)
    assert report_calls[1]["x"] == pytest.approx(6.0)


# -----------------------------------------------------------------------------
# Cadence throttling: timestep and episode units.
# -----------------------------------------------------------------------------
def test_cadence_timestep_throttles_reports(monkeypatch, ray_tune_callback):
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)

    ray_tune_callback.on_train_begin(logs={})
    ray_tune_callback.on_train_epoch_end(
        epoch=0,
        logs={
            "avg_reward": 1.0,
            "episode_reward": 1.0,
            "episode_steps": 5,
            "episode": 1,
        },
    )

    ray_tune_callback.on_train_step_end(step=50, logs={})
    assert len(report_calls) == 0
    ray_tune_callback.on_train_step_end(step=100, logs={})
    assert len(report_calls) == 1
    ray_tune_callback.on_train_step_end(step=150, logs={})
    assert len(report_calls) == 1
    ray_tune_callback.on_train_step_end(step=200, logs={})
    assert len(report_calls) == 2


def test_cadence_episode_throttles_reports(monkeypatch):
    cb = RayTuneCallback(every=2, unit="episode")
    _patch_session(monkeypatch, trial_id="trial-1")
    report_calls = _patch_report(monkeypatch)
    cb.on_train_begin(logs={})

    def _episode(n):
        cb.on_train_epoch_end(
            epoch=n,
            logs={
                "avg_reward": 1.0,
                "episode_reward": 1.0,
                "episode_steps": 5,
                "episode": n,
            },
        )

    _episode(1)
    cb.on_train_step_end(step=1, logs={})
    assert len(report_calls) == 0

    _episode(2)
    cb.on_train_step_end(step=2, logs={})
    assert len(report_calls) == 1

    _episode(3)
    cb.on_train_step_end(step=3, logs={})
    assert len(report_calls) == 1

    _episode(4)
    cb.on_train_step_end(step=4, logs={})
    assert len(report_calls) == 2


# -----------------------------------------------------------------------------
# on_train_epoch_end caching semantics: `logs=None` does not mark the
# first-episode gate open, and missing keys retain their previous cached
# values instead of being reset.
# -----------------------------------------------------------------------------
def test_epoch_end_with_none_logs_does_not_mark_episode_completed(ray_tune_callback):
    ray_tune_callback.on_train_epoch_end(epoch=5, logs=None)

    assert ray_tune_callback._episode_completed is False


def test_epoch_end_caches_partial_fields_and_keeps_previous_for_missing_keys(
    ray_tune_callback,
):
    ray_tune_callback.on_train_epoch_end(
        epoch=1,
        logs={
            "avg_reward": 1.0,
            "episode_reward": 1.0,
            "episode_steps": 10,
            "episode": 1,
            "success_rate": 0.5,
        },
    )
    ray_tune_callback.on_train_epoch_end(epoch=2, logs={"episode": 2})

    assert ray_tune_callback._episodes == 2
    assert ray_tune_callback._avg_reward == 1.0
    assert ray_tune_callback._success_rate == 0.5
    assert ray_tune_callback._episode_reward == 1.0
    assert ray_tune_callback._episode_steps == 10


# -----------------------------------------------------------------------------
# get_config() / load() / registry round-trip.
# -----------------------------------------------------------------------------
def test_raytune_get_config_and_load_round_trip():
    cb = RayTuneCallback(every=777, unit="episode")

    config = cb.get_config()
    assert config == {"type": "RayTuneCallback", "config": {"every": 777, "unit": "episode"}}

    rebuilt = RayTuneCallback.load(config)
    assert rebuilt.every == 777
    assert rebuilt.unit == "episode"


def test_raytune_registry_load_constructs_instance():
    cb = rl_callbacks.load(
        {"type": "RayTuneCallback", "config": {"every": 123, "unit": "timestep"}}
    )

    assert isinstance(cb, RayTuneCallback)
    assert cb.every == 123
    assert cb.unit == "timestep"
