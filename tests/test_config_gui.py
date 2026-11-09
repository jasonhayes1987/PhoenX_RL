"""Smoke tests for the Tkinter config editor (``phoenx.config_gui``).

These drive the real window through its widgets: open a bundled config,
change the agent type with the dropdown, add a layer with the + Add button,
type into fields, and save through the same path the Save button uses. They
skip when tkinter or a display is unavailable (headless CI without Xvfb).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PPO = ROOT / "src/phoenx/examples/configs/LunarLanderContinuous-v3/ppo.yml"


def _display_available() -> bool:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _display_available(), reason="no display for tkinter")


@pytest.fixture()
def editor(monkeypatch):
    from phoenx import config_gui

    # Never block on a dialog during tests.
    monkeypatch.setattr(config_gui.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(config_gui.messagebox, "askyesnocancel", lambda *a, **k: False)
    monkeypatch.setattr(config_gui.messagebox, "showerror", lambda *a, **k: None)
    root = tk.Tk()
    root.withdraw()
    ed = config_gui.ConfigEditor(root)
    yield ed
    ed.dirty = False
    root.destroy()


def _pump(ed):
    ed.master.update()
    ed.validate_now()


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def test_blank_editor_reports_missing_env(editor):
    _pump(editor)
    assert any("env.config" in p and "cfg" in p for p in editor.problems)


def test_open_example_has_no_problems_and_previews_yaml(editor):
    editor.open(str(PPO))
    _pump(editor)
    assert editor.problems == []
    preview = editor.preview.get("1.0", "end")
    assert "type: PPO" in preview
    assert yaml.safe_load(preview)["agent"]["type"] == "PPO"


def test_widgets_edit_the_config_and_save_writes_it(editor, tmp_path):
    editor.open(str(PPO))
    _pump(editor)

    # Change the agent type with its dropdown.
    agent_box = editor.view.widgets["agent.type"]
    agent_box._var.set("SAC")
    _pump(editor)
    assert editor.current_config()["agent"]["type"] == "SAC"

    # Type into a field.
    tau = editor.view.widgets["agent.config.tau"]
    tau._var.set("0.02")
    _pump(editor)
    assert editor.current_config()["agent"]["config"]["tau"] == 0.02

    # Bad input shows up as a problem.
    tau._var.set("fast")
    _pump(editor)
    assert any("agent.config.tau" in p for p in editor.problems)
    tau._var.set("0.02")

    # Add a branch and a layer through the buttons.
    editor.view.expand_to("agent.config.model.branches")
    editor.view.render(editor.form)
    _pump(editor)
    branches = editor.form.children["agent"].variant.children["model"].children["branches"]
    branches.add("policy").set_type("StochasticContinuousHead")
    editor.rerender()
    editor.view.expand_to("agent.config.model.branches.policy.layer_config")
    editor.view.render(editor.form)
    _pump(editor)
    add_buttons = [w for w in _descendants(editor.scroll.inner)
                   if isinstance(w, ttk.Button) and w.cget("text") == "+ Add"]
    assert add_buttons
    layers = branches.entries["policy"].variant.children["layer_config"]
    header = editor.view.widgets["agent.config.model.branches.policy.layer_config"]
    add = next(w for w in header.master.winfo_children()[-1].winfo_children()
               if isinstance(w, ttk.Button))
    add.invoke()
    _pump(editor)
    assert len(layers.items) == 1

    out = tmp_path / "edited.yml"
    assert editor._write(out)
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert saved["agent"]["type"] == "SAC"
    assert saved["agent"]["config"]["tau"] == 0.02
    assert editor.dirty is False


def test_jump_to_problem_focuses_the_field(editor):
    editor.open(str(PPO))
    editor.form.children["agent"].variant.children["context_length"].text = "x"
    editor.rerender()
    _pump(editor)
    index = next(i for i, p in enumerate(editor.problems) if "context_length" in p)
    editor.problem_list.selection_set(index)
    editor._jump_to_problem()
    editor.master.update()
    assert "agent.config.context_length" in editor.view.widgets


def test_show_only_set_fields_hides_defaults(editor):
    editor.open(str(PPO))
    _pump(editor)
    assert "agent.config.context_length" in editor.view.widgets  # not in the file
    editor.only_set.set(True)
    editor.rerender()
    assert "agent.config.context_length" not in editor.view.widgets
    assert "agent.config.discount" in editor.view.widgets  # in the file, even at its default
    assert "agent.config.entropy_coefficient" in editor.view.widgets  # changed in the file


# =============================================================================
# Run tab (PhoenxApp)
# =============================================================================

FIXTURE = ROOT / "tests/fixtures/smoke_ppo_cartpole.yml"


@pytest.fixture()
def app(monkeypatch):
    from phoenx import config_gui, run_panel

    for mod in (config_gui, run_panel):
        monkeypatch.setattr(mod.messagebox, "askyesno", lambda *a, **k: True)
        monkeypatch.setattr(mod.messagebox, "showerror",
                            lambda *a, **k: pytest.fail(f"unexpected error dialog: {a}"))
    monkeypatch.setattr(config_gui.messagebox, "askyesnocancel", lambda *a, **k: False)
    root = tk.Tk()
    root.withdraw()
    a = config_gui.PhoenxApp(root)
    yield a
    a.run_panel.runner.kill()
    a.editor.dirty = False
    root.destroy()


def _wait(app, predicate, timeout=240):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.master.update()
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out\n{_why(app.run_panel)}")


def _finished(rp):
    return not rp.runner.running and rp._current_mode is None


def _why(rp) -> str:
    """Status and log tail, so an intermittent failure explains itself."""
    tail = "\n".join(rp.log.get("1.0", "end").strip().splitlines()[-25:])
    return f"exit={rp.runner.returncode} status={rp.status.cget('text')!r}\n{tail}"


def test_run_tab_train_stop_resume_test(app, tmp_path):
    """Train from the Run tab, Stop mid-run (checkpoint flushed), resume past
    the checkpoint, then test the saved agent."""
    cfg = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    cfg["save_dir"] = "run"
    cfg["schedule"]["stop_units"] = 5_000_000
    cfg["schedule"]["save_every"] = 2_000
    cfg["env"]["config"]["num_envs"] = 4
    (tmp_path / "c.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rp = app.run_panel
    app.show_tab("run")
    rp.vars["cwd"].set(str(tmp_path))
    rp.vars["config"].set("c.yml")
    assert rp.command()[-2:] == ["--config", "c.yml"]
    rp.start()
    _wait(app, lambda: "timestep" in rp.status.cget("text"))   # [progress] reached the GUI
    _wait(app, lambda: (tmp_path / "run" / "trainer_state.pt").exists())
    rp.stop()
    _wait(app, lambda: _finished(rp))
    assert rp.runner.returncode == 130, _why(rp)
    assert rp.status.cget("text").startswith("Stopped"), _why(rp)
    log = rp.log.get("1.0", "end")
    assert "[interrupted]" in log and "[checkpoint]" in log, _why(rp)

    rp.mode.set("resume")
    rp._show_mode()
    rp.vars["agent_dir"].set(str(tmp_path / "run"))
    _wait(app, lambda: str(tmp_path / "run") in rp.run_infos, timeout=120)
    reached = rp.run_infos[str(tmp_path / "run")]["progress"]
    rp.vars["stop_units"].set(str(reached + 2048))
    rp.start()
    _wait(app, lambda: _finished(rp))
    assert rp.runner.returncode == 0, _why(rp)

    rp.mode.set("test")
    rp._show_mode()
    rp.vars["test_agent_dir"].set(str(tmp_path / "run"))
    rp.vars["render_mode"].set("rgb_array")
    rp.vars["num_episodes"].set("2")
    rp.start()
    _wait(app, lambda: _finished(rp))
    assert rp.runner.returncode == 0, _why(rp)
    assert "[test] episode" in rp.log.get("1.0", "end"), _why(rp)


def test_resume_refuses_target_at_checkpoint(app, tmp_path, monkeypatch):
    from phoenx import run_panel

    errors = []
    monkeypatch.setattr(run_panel.messagebox, "showerror", lambda *a, **k: errors.append(a[1]))
    rp = app.run_panel
    run = tmp_path / "r"
    (run / "agent").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    rp.mode.set("resume")
    rp._show_mode()
    rp.vars["cwd"].set(str(tmp_path))
    rp.vars["agent_dir"].set("r")
    rp.run_infos["r"] = {"progress": 1000}
    rp.vars["stop_units"].set("1000")
    rp.start()
    assert errors and "must be past the checkpoint" in errors[0]
    assert not rp.runner.running


# =============================================================================
# Sweep tab
# =============================================================================

SWEEP = ROOT / "src/phoenx/examples/sweeps/lunarlander_ppo.yml"


def test_sweep_tab_opens_bundled_sweep_with_no_problems(app):
    app.sweep.open(str(SWEEP))
    app.show_tab("sweep")
    _pump(app.sweep)
    assert app.sweep.problems == []
    assert "bundled example" in app.sweep.base_label_var.get()
    assert len(app.sweep._tree_fields) > 50


def test_sweep_this_from_config_tab(app, tmp_path, monkeypatch):
    """Right-click 'Sweep this' in the Config tab: sets base_config, adds a spec;
    a field at its default is first written into the base config file."""
    from phoenx import sweep_editor
    from phoenx.config_form import find_field

    monkeypatch.setattr(sweep_editor.messagebox, "askyesno", lambda *a, **k: True)
    base = tmp_path / "base.yml"
    base.write_text(PPO.read_text(encoding="utf-8"), encoding="utf-8")
    app.run_panel.vars["cwd"].set(str(tmp_path))
    app.editor.open(str(base))
    sw = app.sweep
    sw.add_from_config("agent.config.policy_clip",
                       find_field(app.editor.form, "agent.config.policy_clip"), "search")
    sw.add_from_config("agent.config.context_length",
                       find_field(app.editor.form, "agent.config.context_length"), "search")
    sw.add_from_config("schedule.stop_units",
                       find_field(app.editor.form, "schedule.stop_units"), "override")
    _pump(sw)
    cfg = sw.current_config()
    assert cfg["base_config"] == "base.yml"
    space = cfg["phases"][0]["search_space"]
    assert space["agent.config.policy_clip"] == {"dist": "uniform", "low": 0.1, "high": 0.4}
    assert space["agent.config.context_length"]["dist"] == "randint"
    assert cfg["overrides"] == {"schedule.stop_units": 2000000}
    # context_length was at its default, so it was written into the base file.
    assert yaml.safe_load(base.read_text(encoding="utf-8"))["agent"]["config"]["context_length"] == 16
    assert sw.problems == []


def test_sweep_tab_tree_add_and_double_add(app, monkeypatch):
    from phoenx import sweep_editor

    infos = []
    monkeypatch.setattr(sweep_editor.messagebox, "showinfo", lambda *a, **k: infos.append(a[0]))
    sw = app.sweep
    sw.open(str(SWEEP))
    _pump(sw)
    path = "agent.config.model.branches.policy.layer_config.0.params.units"
    leaf = next(i for i, (p, _) in sw._tree_fields.items() if p == path)
    sw.phase_var.set("refine")
    sw.tree.selection_set(leaf)
    sw._add_selected("search")
    refine = next(p for p in sw.current_config()["phases"] if p["name"] == "refine")
    assert refine["search_space"][path]["dist"] == "choice"
    sw._add_selected("search")
    assert infos == ["Already searched"]


# =============================================================================
# Guards against mixing up training configs and sweeps
# =============================================================================

def test_saving_a_sweep_over_a_training_config_warns(app, tmp_path, monkeypatch):
    from phoenx import config_gui

    asked = []
    monkeypatch.setattr(config_gui.messagebox, "askyesno",
                        lambda title, *a, **k: (asked.append(title), False)[1])
    target = tmp_path / "sac.yml"
    target.write_text(PPO.read_text(encoding="utf-8"), encoding="utf-8")
    app.sweep.open(str(SWEEP))
    assert app.sweep._write(target) is False
    assert asked == ["Replace a training config?"]
    assert "agent:" in target.read_text(encoding="utf-8")  # untouched


def test_run_tab_rejects_the_wrong_kind_of_file(app, tmp_path, monkeypatch):
    from phoenx import run_panel

    errors = []
    monkeypatch.setattr(run_panel.messagebox, "showerror", lambda *a, **k: errors.append(a[1]))
    rp = app.run_panel
    rp.vars["cwd"].set(str(tmp_path))
    (tmp_path / "train.yml").write_text(PPO.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "sweep.yml").write_text(SWEEP.read_text(encoding="utf-8"), encoding="utf-8")

    rp.mode.set("sweep")
    rp._show_mode()
    rp.vars["sweep"].set("train.yml")
    rp.start()
    rp.mode.set("train")
    rp._show_mode()
    rp.vars["config"].set("sweep.yml")
    rp.start()
    assert "train.yml is a training config, not a sweep" in errors[0]
    assert "sweep.yml is a sweep, not a training config" in errors[1]
    assert not rp.runner.running
    # Bundled example names are resolved by the command itself, so they pass.
    assert rp._precheck(str(tmp_path)) is not None
    rp.vars["config"].set("LunarLanderContinuous-v3/ppo.yml")
    assert rp._precheck(str(tmp_path)) is None


def test_opening_the_wrong_kind_of_file(app, tmp_path, monkeypatch):
    """A training config opened in the Sweep tab becomes a new sweep's base_config
    (it is never edited as a sweep); a sweep opened in the Config tab goes to the
    Sweep tab."""
    from phoenx import config_gui, sweep_editor

    monkeypatch.setattr(sweep_editor.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(config_gui.messagebox, "askyesno", lambda *a, **k: True)
    train = tmp_path / "sac.yml"
    train.write_text(PPO.read_text(encoding="utf-8"), encoding="utf-8")
    before = train.read_text(encoding="utf-8")
    app.run_panel.vars["cwd"].set(str(tmp_path))

    app.sweep.open(str(train))
    cfg = app.sweep.current_config()
    assert cfg["base_config"] == "sac.yml"
    assert "agent" not in cfg and app.sweep.path is None
    assert train.read_text(encoding="utf-8") == before

    app.editor.open(str(SWEEP))
    assert app.sweep.path == SWEEP
    assert app.editor.path is None


def test_intrinsic_motivation_networks_and_composite_are_form_fields(editor):
    """model_configs, optimizer_params and a Composite's components render as
    fields (not raw YAML boxes), and what the GUI writes validates."""
    from phoenx.config_schema import validate_config

    editor.open(str(PPO))
    _pump(editor)
    agent = editor.form.children["agent"].variant
    im = agent.children["intrinsic_motivation"]
    im.enabled = True
    im.inner.set_type("Composite")
    comps = im.inner.variant.children["components"]
    for kind, nets in (("EpisodicNovelty", ("encoder", "inverse_model")),
                       ("RND", ("target", "predictor"))):
        comp = comps.add()
        comp.set_type(kind)
        cfg = comp.variant.children
        for net in nets:
            layer = cfg["model_configs"].children[net].children["layer_config"].add()
            layer.set_type("dense")
            layer.variant.children["units"].text = "16"
        cfg["optimizer_params"].children["type"].text = "Adam"
    im.inner.variant.children["combination_rule"].text = "ngu"
    editor.rerender()
    base = "agent.config.intrinsic_motivation.config.components"
    for path in (f"{base}.0.config.model_configs.encoder.layer_config",
                 f"{base}.1.config.model_configs.target.output_layer",
                 f"{base}.0.config.optimizer_params.type"):
        editor.view.expand_to(path)
    editor.view.render(editor.form)
    _pump(editor)

    keys = set(editor.view.widgets)
    assert "agent.config.intrinsic_motivation.config.combination_rule" in keys
    assert f"{base}.0.config.model_configs.encoder.layer_config" in keys
    assert f"{base}.1.config.model_configs.target.output_layer" in keys
    assert f"{base}.0.config.optimizer_params.type" in keys
    assert editor.problems == []
    cfg = editor.current_config()
    assert validate_config(cfg) == []
    written = cfg["agent"]["config"]["intrinsic_motivation"]
    assert written["type"] == "Composite"
    assert [c["type"] for c in written["config"]["components"]] == ["EpisodicNovelty", "RND"]
    # The output layer is pre-filled with the one dense layer the networks need.
    assert written["config"]["components"][1]["config"]["model_configs"]["target"]["output_layer"] \
        == [{"type": "dense"}]
