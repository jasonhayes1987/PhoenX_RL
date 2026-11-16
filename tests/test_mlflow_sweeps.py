"""MLflow in sweeps, renders and trainer wiring.

Mirrors the W&B cases in ``test_ray_tune_driver.py``: a phase's ``mlflow:``
block injects exactly one ``MLflowCallback`` per trial with the same run
name / group / tags / ``sweep_params`` / ``extra_config`` W&B gets, and the
phase-counter actor starts for an MLflow-only phase. No Ray cluster is used.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as T

mlflow = pytest.importorskip("mlflow")

import phoenx.ray_tune as ray_tune_mod
from phoenx import rl_callbacks
from phoenx.ray_tune import _inject_trial_callbacks, run_phase
from phoenx.rl_callbacks import MLflowCallback, load
from phoenx.sweep_schema import PHASE_KEYS, build_sweep_schema


class _FakeTrialContext:
    def __init__(self, trial_id):
        self._trial_id = trial_id

    def get_trial_id(self):
        return self._trial_id


def _of_type(config, cb_type):
    return [cb for cb in config["callbacks"] if cb.get("type") == cb_type]


@pytest.fixture()
def trial(monkeypatch):
    monkeypatch.setattr(ray_tune_mod.tune, "get_context", lambda: _FakeTrialContext("abcd_00003"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


# ------------------------------------------------------- _inject_trial_callbacks
def test_mlflow_block_injects_one_callback_with_explicit_fields(trial):
    config = {"callbacks": []}
    sampled = {"agent.config.discount": 0.97}
    phase = {"name": "architecture", "mlflow": {"project": "franka-sweeps"}}
    _inject_trial_callbacks(config, {"_sweep_name": "my_sweep"}, phase, sampled)

    (cb,) = _of_type(config, "MLflowCallback")
    cfg = cb["config"]
    assert cfg["project_name"] == "franka-sweeps"
    assert cfg["run_name"] == "my_sweep-architecture-0004"
    assert cfg["group"] == "my_sweep-architecture"
    assert set(cfg["tags"]) >= {"my_sweep", "architecture"}
    assert cfg["sweep_params"] == sampled
    assert cfg["extra_config"]["ray/trial_id"] == "abcd_00003"
    assert cfg["extra_config"]["ray/trial_ordinal"] == 4
    assert "tracking_uri" not in cfg
    assert _of_type(config, "WandbCallback") == []
    assert len(_of_type(config, "RayTuneCallback")) == 1
    assert isinstance(load(cb), MLflowCallback)  # the injected dict builds a real callback


def test_tracking_uri_from_phase_block(trial):
    config = {}
    phase = {"name": "p", "mlflow": {"project": "x", "tracking_uri": "http://127.0.0.1:5000"}}
    _inject_trial_callbacks(config, {}, phase, sampled={})
    assert _of_type(config, "MLflowCallback")[0]["config"]["tracking_uri"] == "http://127.0.0.1:5000"


def test_tracking_uri_falls_back_to_env_var(trial, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://10.0.0.5:5000")
    config = {}
    _inject_trial_callbacks(config, {}, {"name": "p", "mlflow": {"project": "x"}}, sampled={})
    assert _of_type(config, "MLflowCallback")[0]["config"]["tracking_uri"] == "http://10.0.0.5:5000"


def test_wandb_and_mlflow_together_share_run_identity(trial):
    config = {}
    phase = {"name": "lr", "wandb": {"project": "w"}, "mlflow": {"project": "m"}}
    _inject_trial_callbacks(config, {"_sweep_name": "s"}, phase, sampled={"a": 1})
    (w,) = _of_type(config, "WandbCallback")
    (m,) = _of_type(config, "MLflowCallback")
    for key in ("run_name", "group", "tags", "sweep_params", "extra_config"):
        assert w["config"][key] == m["config"][key]
    assert w["config"]["project_name"] == "w" and m["config"]["project_name"] == "m"


def test_mlflow_absent_strips_base_config_callback(trial):
    config = {"callbacks": [{"type": "MLflowCallback", "config": {"project_name": "base"}}]}
    _inject_trial_callbacks(config, {}, {"name": "p"}, sampled={})
    assert _of_type(config, "MLflowCallback") == []


def test_mlflow_block_replaces_base_config_callback(trial):
    config = {"callbacks": [{"type": "MLflowCallback", "config": {"project_name": "base"}}]}
    _inject_trial_callbacks(config, {}, {"name": "p", "mlflow": {"project": "new"}}, sampled={})
    (cb,) = _of_type(config, "MLflowCallback")
    assert cb["config"]["project_name"] == "new"


def test_mlflow_missing_project_raises(trial):
    with pytest.raises(ValueError, match=r"phase\['mlflow'\].*'project'"):
        _inject_trial_callbacks({}, {}, {"name": "p", "mlflow": {"tracking_uri": "x"}}, sampled={})


def test_wandb_only_phase_unchanged(trial):
    config = {}
    _inject_trial_callbacks(config, {"_sweep_name": "s"}, {"name": "p", "wandb": {"project": "w"}}, sampled={})
    assert len(_of_type(config, "WandbCallback")) == 1
    assert _of_type(config, "MLflowCallback") == []


def test_write_best_config_accepts_null_callbacks(tmp_path):
    """A base config with `callbacks: null` (what Trainer.get_config writes) must not crash."""
    import yaml
    from phoenx.ray_tune import write_best_config
    write_best_config({"agent": {}, "callbacks": None}, tmp_path / "best.yml")
    assert "callbacks" not in yaml.safe_load((tmp_path / "best.yml").read_text())


def test_write_best_config_keeps_mlflow_callback(tmp_path):
    """Like WandbCallback, the MLflow callback survives into best_config.yml (only RayTuneCallback is stripped)."""
    import yaml
    from phoenx.ray_tune import write_best_config
    cfg = {"callbacks": [{"type": "RayTuneCallback", "config": {}},
                         {"type": "MLflowCallback", "config": {"project_name": "m"}}]}
    write_best_config(cfg, tmp_path / "best.yml")
    assert [cb["type"] for cb in yaml.safe_load((tmp_path / "best.yml").read_text())["callbacks"]] == ["MLflowCallback"]


# ------------------------------------------------------------- phase counter
def _patch_run_phase(monkeypatch):
    """No-Ray harness for run_phase; returns (start_calls, with_parameters_calls)."""
    monkeypatch.setattr(ray_tune_mod, "_ensure_ray_initialized", lambda sweep: None)
    monkeypatch.setattr(ray_tune_mod, "resolve_trial_config",
                        lambda sweep, phase, sampled, base_config: {"resolved": True})
    monkeypatch.setattr(ray_tune_mod, "write_best_config", lambda config, path: None)
    from phoenx import mlflow_server
    server_calls = []
    monkeypatch.setattr(mlflow_server, "ensure_server", lambda uri, **k: server_calls.append(uri))
    monkeypatch.setattr(ray_tune_mod, "_server_calls", server_calls, raising=False)
    start_calls = []

    def _start(sweep_name, phase_name):
        start_calls.append((sweep_name, phase_name))
        return object(), "actor-name"

    monkeypatch.setattr(ray_tune_mod, "_start_phase_counter", _start)
    monkeypatch.setattr(ray_tune_mod, "_stop_phase_counter", lambda handle: None)
    wp_calls = []

    def _fake_with_parameters(fn, **kwargs):
        wp_calls.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(ray_tune_mod.tune, "with_parameters", _fake_with_parameters)
    monkeypatch.setattr(ray_tune_mod.tune, "with_resources", lambda trainable, resources: trainable)
    fake_result = SimpleNamespace(metrics={"config": {"a": 1}, "avg_reward": 5.0}, error=None)

    class _Grid:
        num_errors = 0
        errors = []

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return fake_result

        def get_best_result(self, metric, mode):
            return fake_result

    class _Tuner:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def can_restore(path):
            return False

        def fit(self):
            return _Grid()

    monkeypatch.setattr(ray_tune_mod.tune, "Tuner", _Tuner)
    return start_calls, wp_calls


def test_mlflow_only_phase_starts_counter(monkeypatch, tmp_path):
    start_calls, wp_calls = _patch_run_phase(monkeypatch)
    phase = {"name": "p1", "tune": {"num_samples": 10}, "mlflow": {"project": "m"}}
    run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")
    assert start_calls == [("s", "p1")]
    assert wp_calls[0]["sweep"]["_phase_counter"] == "actor-name"


def test_driver_starts_local_server_once_before_trials(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    _patch_run_phase(monkeypatch)
    phase = {"name": "p1", "tune": {"num_samples": 4}, "mlflow": {"project": "m"}}
    run_phase(sweep={}, phase=phase, base_config={}, storage_path=tmp_path, sweep_name="s")
    assert ray_tune_mod._server_calls == ["http://127.0.0.1:5000"]


def test_driver_skips_server_for_remote_or_disabled(monkeypatch, tmp_path):
    _patch_run_phase(monkeypatch)
    for block in ({"project": "m", "tracking_uri": "http://10.0.0.5:5000"},
                  {"project": "m", "auto_start_server": False}):
        run_phase(sweep={}, phase={"name": "p1", "tune": {"num_samples": 1}, "mlflow": block},
                  base_config={}, storage_path=tmp_path, sweep_name="s")
    assert ray_tune_mod._server_calls == []


def test_auto_start_flag_passed_to_trials(trial):
    config = {}
    phase = {"name": "p", "mlflow": {"project": "x", "auto_start_server": False}}
    _inject_trial_callbacks(config, {}, phase, sampled={})
    assert _of_type(config, "MLflowCallback")[0]["config"]["auto_start_server"] is False


def test_no_tracker_phase_starts_no_counter(monkeypatch, tmp_path):
    start_calls, _ = _patch_run_phase(monkeypatch)
    run_phase(sweep={}, phase={"name": "p1", "tune": {"num_samples": 10}},
              base_config={}, storage_path=tmp_path, sweep_name="s")
    assert start_calls == []


# -------------------------------------------------------------------- schema
def test_sweep_schema_accepts_mlflow_key():
    assert "mlflow" in PHASE_KEYS
    schema = build_sweep_schema()
    assert "mlflow" in str(schema)


# ---------------------------------------------------- concurrent experiment
def test_experiment_create_race_recovers(tmp_path, monkeypatch):
    """Two trials creating the same experiment: the loser re-reads it instead of crashing."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    cb = MLflowCallback("race", tracking_uri=uri)
    cb._client = mlflow.MlflowClient(uri)
    real_id = cb._client.create_experiment("race")
    calls = {"n": 0}
    real_get = cb._client.get_experiment_by_name

    def _stale_then_real(name):  # first lookup misses, as if the other trial hadn't committed yet
        calls["n"] += 1
        return None if calls["n"] == 1 else real_get(name)

    monkeypatch.setattr(cb._client, "get_experiment_by_name", _stale_then_real)
    assert cb._get_experiment_id() == real_id


# ------------------------------------------------------------------- renders
def _started(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    cb = MLflowCallback("proj", tracking_uri=uri, async_logging=False)
    return cb, {"save_dir": str(save_dir), "agent": {"type": "PPO"}}


def test_log_render_uploads_video_and_metrics(tmp_path, monkeypatch):
    cb, config = _started(tmp_path, monkeypatch)
    cb.on_train_begin(config)
    video = tmp_path / "episode_3.mp4"
    video.write_bytes(b"\x00fake-mp4")
    cb.log_render(video, "train", 400, {"render_episode_reward": 12.5, "render_episode_length": 80})
    cb._flush()
    files = [f.path for f in cb._client.list_artifacts(cb.active_run_id, "renders/train")]
    assert files == ["renders/train/episode_3.mp4"]
    hist = cb._client.get_metric_history(cb.active_run_id, "render_episode_reward")
    assert [(m.step, m.value) for m in hist] == [(400, 12.5)]


def test_log_render_without_open_run_is_noop(tmp_path, monkeypatch):
    cb, _ = _started(tmp_path, monkeypatch)
    cb.log_render(tmp_path / "missing.mp4", "train", 1, {"x": 1.0})  # must not raise


def test_renderer_sends_video_to_mlflow_callback(tmp_path, monkeypatch):
    """Renderer.render_episode calls log_render on every MLflowCallback after writing the video."""
    from phoenx import renderer as renderer_mod

    seen = []

    class _Recorder(MLflowCallback, register=False):
        def __init__(self):
            pass

        def log_render(self, video_path, context, step, metrics=None):
            seen.append((str(video_path), context, step, metrics))

    class _Obs:
        terminations = T.tensor([True])
        truncations = T.tensor([False])
        rewards = T.tensor([1.5])

    class _RenderEnv:
        def reset(self):
            return _Obs()

        def _find_nstep_wrapper(self):
            return None

        def step(self, action):
            return _Obs()

        def render_frame(self):
            import numpy as np
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    class _Env:
        def clone(self, **kwargs):
            return _RenderEnv()

    r = renderer_mod.Renderer(render_freq=1, save_dir=str(tmp_path))
    monkeypatch.setattr(r, "_save_video", lambda frames, episode, context: None)
    trainer = SimpleNamespace(
        env=_Env(),
        callbacks=[_Recorder()],
        normalize_observation=lambda obs: SimpleNamespace(states=None, goals=None),
        get_action=lambda states, goals, context: SimpleNamespace(actions=T.zeros(1, 1)),
    )
    r.render_episode(trainer, episode=3, step=400, context="train")
    assert len(seen) == 1
    path, context, step, metrics = seen[0]
    assert path.endswith("episode_3.mp4") and context == "train" and step == 400
    assert metrics == {"render_episode_reward": 1.5, "render_episode_length": 1}


# ------------------------------------------------------------- trainer wiring
def test_trainer_passes_models_and_run_number_to_mlflow():
    """`_initialize_run` gives MLflowCallback the same (config, run_number, models) W&B gets."""
    import inspect
    from phoenx import trainer as trainer_mod
    src = inspect.getsource(trainer_mod.Trainer._initialize_run)
    assert "(WandbCallback, MLflowCallback)" in src


def test_models_logged_including_non_module(tmp_path, monkeypatch):
    cb, config = _started(tmp_path, monkeypatch)
    cb.on_train_begin(config, run_number=None, models=(T.nn.Linear(3, 2), object()))
    params = cb._client.get_run(cb.active_run_id).data.params
    assert params["model_0/class"] == "Linear"
    assert params["model_0/num_params"] == "8"
    assert params["model_1/class"] == "object"
    assert "model_1/num_params" not in params
    files = {f.path for f in cb._client.list_artifacts(cb.active_run_id, "model_architecture")}
    assert files == {"model_architecture/model_0.txt", "model_architecture/model_1.txt"}
