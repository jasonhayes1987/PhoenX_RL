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

import pytest
import wandb
import wandb.errors

from phoenx import rl_callbacks
from phoenx.rl_callbacks import WandbCallback


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
