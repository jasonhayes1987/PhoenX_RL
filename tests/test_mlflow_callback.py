"""Tests for `MLflowCallback`.

Every test runs against a real, throwaway MLflow store in ``tmp_path``
(SQLite backend, local artifacts), so nothing touches a tracking server or
the network. The one server test points at a closed localhost port to check
the fail-fast health check.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import torch as T

mlflow = pytest.importorskip("mlflow")

from phoenx import rl_callbacks
from phoenx.rl_callbacks import MLflowCallback, callback_load, load


@pytest.fixture()
def tracking_uri(tmp_path, monkeypatch):
    """A fresh SQLite tracking store with artifacts under ``tmp_path``."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)  # default artifact root (./mlruns) lands here
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture()
def config(tmp_path):
    """Minimal trainer config tree, as `Trainer.get_config()` would pass it."""
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    return {
        "save_dir": str(save_dir),
        "agent": {"type": "PPO", "learning_rate": 3e-4, "layers": [64, 64]},
        "env": {"id": "LunarLander-v3", "num_envs": 4},
    }


def _client(uri):
    return mlflow.MlflowClient(tracking_uri=uri)


def _run(cb):
    return _client(cb.resolved_tracking_uri).get_run(cb.active_run_id)


def _history(cb, key):
    cb._flush()  # metrics are buffered; push them before reading back
    hist = _client(cb.resolved_tracking_uri).get_metric_history(cb.active_run_id, key)
    return [(m.step, m.value) for m in sorted(hist, key=lambda m: m.step)]


# --------------------------------------------------------------- registration
def test_registered_and_round_trips_through_load(tracking_uri):
    cb = MLflowCallback("proj", run_name="r", tags=["a"], artifact_every=10, tracking_uri=tracking_uri)
    assert callback_load("MLflowCallback") is MLflowCallback
    rebuilt = load(cb.get_config())
    assert isinstance(rebuilt, MLflowCallback)
    assert rebuilt.get_config() == cb.get_config()


def test_constructor_matches_wandb_callback_arguments():
    """Every WandbCallback kwarg is accepted, so sweep injection can reuse its dict."""
    import inspect
    wandb_args = set(inspect.signature(rl_callbacks.WandbCallback.__init__).parameters)
    mlflow_args = set(inspect.signature(MLflowCallback.__init__).parameters)
    assert wandb_args <= mlflow_args


def test_missing_mlflow_raises_install_hint(monkeypatch):
    def boom():
        raise ImportError("MLflowCallback needs MLflow. Install it with `pip install mlflow`")
    monkeypatch.setattr(rl_callbacks, "_import_mlflow", boom)
    with pytest.raises(ImportError, match="pip install mlflow"):
        MLflowCallback("proj")


# -------------------------------------------------------------- tracking URI
def test_unreachable_server_fails_fast(config, monkeypatch):
    monkeypatch.setattr(rl_callbacks, "_MLFLOW_CONNECT_TIMEOUT", 1)
    cb = MLflowCallback("proj", tracking_uri="http://127.0.0.1:9")
    with pytest.raises(ValueError, match="not reachable"):
        cb.on_train_begin(config)
    assert not cb.initialized


def test_env_var_used_when_no_tracking_uri(config, tracking_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    cb = MLflowCallback("proj", async_logging=False)
    cb.on_train_begin(config)
    assert cb.resolved_tracking_uri == tracking_uri


# ------------------------------------------------------------------- run init
def test_train_begin_creates_named_run_with_tags_and_params(config, tracking_uri):
    cb = MLflowCallback("proj", tags=["mine"], tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    run = _run(cb)
    assert run.info.run_name == "train-1" == cb.run_name
    assert run.data.tags["group"] == "group-1"
    assert run.data.tags["job_type"] == "train"
    assert run.data.tags["agent_type"] == "PPO"
    for tag in ("mine", "train", "PPO"):
        assert run.data.tags[tag] == "true"
    assert run.data.params["agent/learning_rate"] == "0.0003"
    assert run.data.params["env/num_envs"] == "4"
    assert run.data.params["agent/layers"] == "[64, 64]"
    assert cb.tags == ["mine"]  # never mutated


def test_run_number_increments_per_experiment(config, tracking_uri):
    first = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    first.on_train_begin(config)
    first.on_train_end()
    second = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    second.on_train_begin(config)
    assert second.run_name == "train-2"


def test_explicit_name_and_group_skip_run_number_lookup(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", run_name="sweep-p1-0001of0008", group="sweep-p1",
                        tracking_uri=tracking_uri, async_logging=False)
    monkeypatch.setattr(MLflowCallback, "_next_run_number",
                        lambda self: pytest.fail("should not look up a run number"))
    cb.on_train_begin(config)
    run = _run(cb)
    assert run.info.run_name == "sweep-p1-0001of0008"
    assert run.data.tags["group"] == "sweep-p1"


def test_sweep_params_and_extra_config_logged(config, tracking_uri):
    cb = MLflowCallback("proj", sweep_params={"agent": {"lr": 0.01}},
                        extra_config={"ray/trial_id": "abc", "env": "override"},
                        tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    params = _run(cb).data.params
    assert params["sweep/agent/lr"] == "0.01"
    assert params["ray/trial_id"] == "abc"
    assert params["env"] == "override"  # extra_config wins collisions


def test_full_config_stored_as_json(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    path = mlflow.artifacts.download_artifacts(run_id=cb.active_run_id, artifact_path="config.json",
                                               tracking_uri=tracking_uri)
    assert json.load(open(path))["agent"]["type"] == "PPO"


def test_models_log_architecture(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config, models=(T.nn.Linear(3, 2),))
    params = _run(cb).data.params
    assert params["model_0/num_params"] == "8"
    files = [f.path for f in _client(tracking_uri).list_artifacts(cb.active_run_id, "model_architecture")]
    assert files == ["model_architecture/model_0.txt"]


def test_second_train_begin_does_not_open_second_run(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    run_id = cb.active_run_id
    cb.on_train_begin(config)
    assert cb.active_run_id == run_id


def test_test_begin_skips_init_after_training(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    run_id = cb.active_run_id
    cb.on_test_begin(config)
    assert cb.active_run_id == run_id


def test_test_begin_alone_creates_test_run(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_test_begin(config)
    run = _run(cb)
    assert run.info.run_name == "test-1"
    assert run.data.tags["job_type"] == "test"


# ---------------------------------------------------------------------- resume
def test_run_id_reattaches_to_existing_run(config, tracking_uri):
    first = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    first.on_train_begin(config)
    first.on_train_end()
    again = MLflowCallback("proj", run_id=first.active_run_id, tracking_uri=tracking_uri, async_logging=False)
    again.on_train_begin(config)
    assert again.active_run_id == first.active_run_id
    assert _run(again).info.status == "RUNNING"


def test_missing_run_id_with_resume_must_raises(config, tracking_uri):
    cb = MLflowCallback("proj", run_id="0" * 32, resume="must", tracking_uri=tracking_uri)
    with pytest.raises(ValueError, match="resume='must'"):
        cb.on_train_begin(config)


def test_missing_run_id_without_must_starts_new_run(config, tracking_uri):
    cb = MLflowCallback("proj", run_id="0" * 32, resume="allow", tracking_uri=tracking_uri,
                        async_logging=False)
    with pytest.warns(UserWarning, match="starting a new run"):
        cb.on_train_begin(config)
    assert cb.active_run_id != "0" * 32


# -------------------------------------------------------------------- metrics
def test_step_and_episode_metrics_logged_at_trainer_step(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    cb.on_train_step_end(4, {"step_reward": 1.5, "rollout/truncation_rate": T.tensor(0.25)})
    cb.on_train_step_end(8, {"step_reward": np.float32(2.0)})
    cb.on_train_epoch_end(8, {"episode_reward": 10.0, "best": True, "env": 0})
    assert _history(cb, "step_reward") == [(4, 1.5), (8, 2.0)]
    assert _history(cb, "rollout/truncation_rate") == [(4, 0.25)]
    assert _history(cb, "episode_reward") == [(8, 10.0)]
    assert _history(cb, "best") == [(8, 1.0)]


def test_non_numeric_values_skipped_with_one_warning(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    with pytest.warns(UserWarning, match="'note'"):
        cb.on_train_step_end(1, {"note": "hi", "x": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cb.on_train_step_end(2, {"note": "again", "x": 2})
    assert _history(cb, "x") == [(1, 1.0), (2, 2.0)]


def test_none_logs_are_fine(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    cb.on_train_step_end(1, None)
    cb.on_train_epoch_end(1, None)
    cb.on_test_step_end(1, None)
    cb.on_test_epoch_end(1, None)


def test_async_metrics_flushed_on_end(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=True)
    cb.on_train_begin(config)
    for step in range(1, 51):
        cb.on_train_step_end(step, {"loss": float(step)})
    cb.on_train_end()
    assert len(_history(cb, "loss")) == 50


def test_metrics_are_batched_not_sent_per_step(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    monkeypatch.setattr(rl_callbacks, "_MLFLOW_FLUSH_SECONDS", 3600)
    monkeypatch.setattr(rl_callbacks, "_MLFLOW_BATCH_SIZE", 10)
    sent = []
    real = cb._client.log_batch
    monkeypatch.setattr(cb._client, "log_batch",
                        lambda run_id, metrics=(), **k: (sent.append(len(metrics)), real(run_id, metrics=metrics, **k)))
    for step in range(1, 10):
        cb.on_train_step_end(step, {"loss": 1.0})
    assert sent == []  # 9 buffered values, below the batch size
    cb.on_train_step_end(10, {"loss": 1.0})
    assert sent == [10]
    cb.on_train_step_end(11, {"loss": 1.0})
    cb.on_train_end()
    assert sent == [10, 1]  # remainder flushed at run end


def test_metrics_flushed_after_time_limit(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    monkeypatch.setattr(rl_callbacks, "_MLFLOW_FLUSH_SECONDS", 0.0)
    cb.on_train_step_end(1, {"loss": 1.0})
    assert cb._pending == []


# ------------------------------------------------------------------ artifacts
def _artifact_paths(cb):
    return {f.path for f in _client(cb.resolved_tracking_uri).list_artifacts(cb.active_run_id, "model")}


def test_artifact_uploaded_only_when_saved(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    (cb_dir := __import__("pathlib").Path(cb.save_dir) / "config.json").write_text("{}")
    cb.on_train_epoch_end(10, {"episode_reward": 1.0, "best": True})
    assert _artifact_paths(cb) == set()
    cb.on_train_epoch_end(20, {"episode_reward": 2.0, "saved": True})
    assert _artifact_paths(cb) == {"model/config.json"}
    assert _run(cb).data.tags["best_checkpoint_step"] == "20"


def test_artifact_skips_renders_but_keeps_subfolders(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    root = __import__("pathlib").Path(cb.save_dir)
    (root / "agent").mkdir()
    (root / "agent" / "model.pt").write_bytes(b"w")
    (root / "renders" / "train").mkdir(parents=True)
    (root / "renders" / "train" / "episode_1.mp4").write_bytes(b"v")
    cb.on_train_epoch_end(5, {"saved": True})
    client = _client(tracking_uri)
    assert _artifact_paths(cb) == {"model/agent"}
    assert [f.path for f in client.list_artifacts(cb.active_run_id, "model/agent")] == ["model/agent/model.pt"]


def test_artifact_every_throttles_uploads(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", artifact_every=100, tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    uploads = []
    monkeypatch.setattr(cb, "_save_model_artifact", lambda step: uploads.append(step))
    for step in (10, 50, 110, 150, 210):
        cb.on_train_epoch_end(step, {"saved": True})
    assert uploads == [10, 110, 210]


# --------------------------------------------------------------------- finish
def test_finish_is_idempotent(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    calls = []
    real = cb._client.set_terminated
    monkeypatch.setattr(cb._client, "set_terminated", lambda *a, **k: (calls.append(1), real(*a, **k)))
    cb.on_train_end()
    cb.on_train_end()
    cb.on_test_end()
    assert len(calls) == 1
    assert _run(cb).info.status == "FINISHED"


def test_unfinished_run_closed_as_killed_at_exit(config, tracking_uri, monkeypatch):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    monkeypatch.setattr(rl_callbacks, "_MLFLOW_FLUSH_SECONDS", 3600)
    cb.on_train_step_end(1, {"loss": 0.5})  # still buffered
    cb._close_at_exit()  # what atexit runs if the trainer never called on_train_end
    assert _run(cb).info.status == "KILLED"
    assert [m.value for m in _client(tracking_uri).get_metric_history(cb.active_run_id, "loss")] == [0.5]


def test_finished_run_not_touched_at_exit(config, tracking_uri):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    cb.on_train_end()
    cb._close_at_exit()
    assert _run(cb).info.status == "FINISHED"


def test_end_before_begin_is_noop():
    cb = MLflowCallback("proj")
    cb.on_train_end()
    cb.on_test_end()


# ---------------------------------------------------------------- persistence
def test_save_writes_config_and_run_section(config, tracking_uri, tmp_path):
    cb = MLflowCallback("proj", tracking_uri=tracking_uri, async_logging=False)
    cb.on_train_begin(config)
    cb.save()
    data = json.load(open(tmp_path / "run" / "mlflow_config.json"))
    assert data["type"] == "MLflowCallback"
    assert data["run"]["run_id"] == cb.active_run_id
    assert data["run"]["run_url"] is None  # local store has no web UI link
    assert isinstance(load(data), MLflowCallback)


def test_urls_for_http_tracking_server():
    cb = MLflowCallback("proj")
    cb.resolved_tracking_uri = "http://127.0.0.1:5000/"
    cb.experiment_id, cb.active_run_id = "3", "abc"
    assert cb.run_url == "http://127.0.0.1:5000/#/experiments/3/runs/abc"
    assert cb.experiment_url == "http://127.0.0.1:5000/#/experiments/3"
