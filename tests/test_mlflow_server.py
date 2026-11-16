"""Tests for `phoenx.mlflow_server`: starting a local MLflow server on demand.

The integration tests start a real ``mlflow server`` on a free port with its
data in ``tmp_path`` (``PHOENX_MLFLOW_HOME``) and stop it afterwards.
"""
from __future__ import annotations

import socket
import threading

import pytest

pytest.importorskip("mlflow")

from phoenx import mlflow_server
from phoenx.rl_callbacks import MLflowCallback


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOENX_MLFLOW_HOME", str(tmp_path / "mlflow_home"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield tmp_path / "mlflow_home"
    mlflow_server.stop_server()


@pytest.mark.parametrize("uri, expected", [
    ("http://127.0.0.1:5000", True),
    ("http://localhost:8080", True),
    ("http://10.0.0.5:5000", False),
    ("https://mlflow.example.com", False),
    ("sqlite:///x.db", False),
])
def test_is_local(uri, expected):
    assert mlflow_server.is_local(uri) is expected


def test_remote_unreachable_raises_without_starting(home):
    with pytest.raises(ValueError, match="only starts servers on this machine"):
        mlflow_server.ensure_server("http://10.255.255.1:5999", timeout=1)
    assert not (home / "server.pid").exists()


@pytest.mark.slow
def test_starts_server_once_and_reuses_it(home):
    uri = f"http://127.0.0.1:{_free_port()}"
    assert mlflow_server.ensure_server(uri) is True
    assert mlflow_server.is_up(uri)
    assert (home / "mlflow.db").exists()
    assert (home / "server.pid").exists()
    assert not (home / "server.lock").exists()
    assert mlflow_server.ensure_server(uri) is False  # already up: nothing started


@pytest.mark.slow
def test_concurrent_callers_start_one_server(home):
    uri = f"http://127.0.0.1:{_free_port()}"
    results, errors = [], []

    def call():
        try:
            results.append(mlflow_server.ensure_server(uri))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(results) == [False, False, False, True]


@pytest.mark.slow
def test_port_taken_by_other_program_reports_log(home):
    with socket.socket() as blocker:  # a listener that is not MLflow
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        uri = f"http://127.0.0.1:{blocker.getsockname()[1]}"
        with pytest.raises(ValueError, match="server.log"):
            mlflow_server.ensure_server(uri, timeout=60)


@pytest.mark.slow
def test_stop_server(home):
    uri = f"http://127.0.0.1:{_free_port()}"
    mlflow_server.ensure_server(uri)
    assert mlflow_server.stop_server() is True
    import time
    deadline = time.monotonic() + 15
    while mlflow_server.is_up(uri) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not mlflow_server.is_up(uri)
    assert mlflow_server.stop_server() is False  # nothing left to stop


@pytest.mark.slow
def test_callback_auto_starts_server_and_logs(home, tmp_path):
    uri = f"http://127.0.0.1:{_free_port()}"
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    cb = MLflowCallback("auto", tracking_uri=uri, async_logging=False)
    cb.on_train_begin({"save_dir": str(save_dir), "agent": {"type": "PPO"}})
    cb.on_train_step_end(1, {"loss": 0.25})
    cb.on_train_end()
    run = cb._client.get_run(cb.active_run_id)
    assert run.info.status == "FINISHED"
    assert [m.value for m in cb._client.get_metric_history(cb.active_run_id, "loss")] == [0.25]
    assert cb.run_url.startswith(uri)


def test_callback_default_uri_is_local_server(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    calls = []
    monkeypatch.setattr(mlflow_server, "ensure_server", lambda uri, **k: calls.append(uri))
    cb = MLflowCallback("p")
    try:
        cb._ensure_mlflow_tracking()
    except Exception:
        pass  # no server behind the stub; only the resolved address matters
    assert calls == [mlflow_server.DEFAULT_TRACKING_URI]
    assert cb.get_config()["config"]["auto_start_server"] is True
