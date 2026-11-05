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
    assert "agent.config.discount" in editor.view.widgets  # at its default
    editor.only_set.set(True)
    editor.rerender()
    assert "agent.config.discount" not in editor.view.widgets
    assert "agent.config.entropy_coefficient" in editor.view.widgets  # changed in the file
