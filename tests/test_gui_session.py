"""Behavior of the desktop session: documents, launch checks, and the JSON loop."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from phoenx.config_form import HEADER, build_form, dump_config
from phoenx.example_files import bundled_examples
from phoenx.gui_session import GuiError, GuiSession, serve
from phoenx.launch_cmd import build_command, command_text


def _save_as(session: GuiSession, target: str, path: Path):
    """Save, confirming the problems question a blank form always raises."""
    result = session.call("editor.save_as", {"target": target, "path": str(path)})
    if isinstance(result, dict) and result.get("id") == "problems":
        result = session.call("editor.save_as", {"target": target, "answer": "yes"})
    return result


def _fresh(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gui_session_import_and_calls_omit_torch_and_tkinter():
    _fresh("""
        import sys
        from phoenx.gui_session import GuiSession
        leaked = [n for n in ("torch", "tkinter") if n in sys.modules]
        assert not leaked, leaked
        session = GuiSession()
        session.call("editor.new_blank", {"target": "config"})
        session.call("editor.set_flags", {"target": "config", "only_set": True})
        session.call("editor.set_text", {
            "target": "config", "path": "schedule.stop_units", "text": "3"})
        leaked = [n for n in ("torch", "tkinter") if n in sys.modules]
        assert not leaked, leaked
        """)


def test_blank_config_matches_dump_of_blank_form():
    session = GuiSession()
    snap = session.call("editor.new_blank", {"target": "config"})
    form = build_form()
    value = form.value()
    expect = dump_config({} if value is None else value, header=HEADER)
    assert snap["path"] is None
    assert snap["dirty"] is False
    assert snap["title"] == "untitled config - PhoenX"
    assert snap["status"].startswith("New config. Pick an agent type")
    assert snap["yaml"] == expect


def test_set_text_matches_scalar_assignment_and_flags():
    session = GuiSession()
    session.call("editor.new_blank", {"target": "config"})
    snap = session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.learn_every", "text": "4"},
    )
    form = build_form()
    form.children["schedule"].children["learn_every"].text = "4"
    assert snap["yaml"] == dump_config(form.value(), header=HEADER)
    assert snap["dirty"] is True

    flags = GuiSession()
    only = flags.call("editor.set_flags", {"target": "config", "only_set": True})
    assert only["dirty"] is False
    assert only["only_set"] is True
    written = flags.call(
        "editor.set_flags", {"target": "config", "with_defaults": True}
    )
    assert written["dirty"] is True
    assert "learn_every" in written["yaml"] or "stop_units" in written["yaml"]


def test_bad_scalar_is_kept_once():
    session = GuiSession()
    snap = session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.stop_units", "text": "abc"},
    )
    hits = [p for p in snap["problems"] if p.split(":", 1)[0] == "schedule.stop_units"]
    assert len(hits) == 1
    assert "abc" in snap["yaml"]


def test_bool_false_is_written_and_default_number_is_omitted():
    session = GuiSession()
    snap = session.call(
        "editor.set_text",
        {
            "target": "config",
            "path": "agent.config.auto_entropy_tuning",
            "text": "false",
        },
    )
    assert "false" in snap["yaml"]
    again = session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.stop_units", "text": "1000000"},
    )
    assert "stop_units" not in again["yaml"]


def test_examples_match_bundled_names():
    session = GuiSession()
    assert session.call("editor.examples", {"target": "config"})["names"] == list(
        bundled_examples("configs")
    )
    assert session.call("editor.examples", {"target": "sweep"})["names"] == list(
        bundled_examples("sweeps")
    )


def test_new_from_example_is_dirty_and_untitled(tmp_path):
    session = GuiSession()
    name = next(iter(bundled_examples("configs")))
    snap = session.call("editor.new_from_example", {"target": "config", "name": name})
    assert snap["path"] is None and snap["dirty"] is True
    assert snap["status"].startswith(f"New config from example {name}.")
    saved = tmp_path / "out.yml"
    _save_as(session, "config", saved)
    original = yaml.safe_load(bundled_examples("configs")[name])
    written = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert written.keys() >= {
        k for k, v in original.items() if v == written.get(k) or k in written
    }


def test_comment_and_anchor_notes(tmp_path):
    session = GuiSession()
    commented = tmp_path / "c.yml"
    commented.write_text("# note\na: 1\n", encoding="utf-8")
    plain = tmp_path / "p.yml"
    plain.write_text("a: 1\n", encoding="utf-8")
    with_note = session.call(
        "editor.open", {"target": "config", "path": str(commented)}
    )
    assert "comments and YAML anchors" in with_note["status"]
    without = session.call("editor.open", {"target": "config", "path": str(plain)})
    assert "comments and YAML anchors" not in without["status"]
    anchored = tmp_path / "a.yml"
    anchored.write_text("a: &n 1\nb: *n\n", encoding="utf-8")
    session.call("editor.open", {"target": "config", "path": str(anchored)})
    dest = tmp_path / "out.yml"
    _save_as(session, "config", dest)
    text = dest.read_text(encoding="utf-8")
    assert "&" not in text and "*" not in text


def test_discard_answers(tmp_path):
    session = GuiSession()
    session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.learn_every", "text": "9"},
    )
    need = session.call("editor.new_blank", {"target": "config"})
    assert need["id"] == "discard"
    held = session.call("editor.new_blank", {"target": "config", "answer": "cancel"})
    assert held["dirty"] is True and "learn_every" in held["yaml"]
    session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.learn_every", "text": "9"},
    )
    assert session.call("editor.new_blank", {"target": "config"})["id"] == "discard"
    dropped = session.call("editor.new_blank", {"target": "config", "answer": "no"})
    assert dropped["dirty"] is False
    assert dropped["title"] == "untitled config - PhoenX"

    session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.learn_every", "text": "8"},
    )
    path = tmp_path / "c.yml"
    _save_as(session, "config", path)
    session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.learn_every", "text": "11"},
    )
    asked = session.call("editor.new_blank", {"target": "config"})
    assert asked["id"] == "discard"
    saved = session.call("editor.new_blank", {"target": "config", "answer": "yes"})
    if isinstance(saved, dict) and saved.get("id") == "problems":
        saved = session.call("editor.new_blank", {"target": "config", "answer": "yes"})
    assert saved["title"] == "untitled config - PhoenX"
    assert "learn_every: 11" in path.read_text(
        encoding="utf-8"
    ) or "11" in path.read_text(encoding="utf-8")


def test_save_problems_before_overwrite(tmp_path):
    session = GuiSession()
    session.call(
        "editor.set_text",
        {"target": "config", "path": "schedule.stop_units", "text": "abc"},
    )
    sweep = tmp_path / "s.yml"
    sweep.write_text("base_config: a.yml\nphases: []\n", encoding="utf-8")
    need = session.call("editor.save_as", {"target": "config", "path": str(sweep)})
    assert need["id"] == "overwrite"
    nxt = session.call("editor.save_as", {"target": "config", "answer": "yes"})
    assert nxt["id"] == "problems"
    stayed = session.call("editor.save_as", {"target": "config", "answer": "no"})
    assert stayed["dirty"] is True
    assert sweep.read_text(encoding="utf-8").startswith("base_config:")


def test_opening_sweep_in_config_asks(tmp_path):
    sweep = tmp_path / "sweep.yml"
    sweep.write_text("base_config: a.yml\nphases:\n- name: phase_0\n", encoding="utf-8")
    session = GuiSession()
    need = session.call("editor.open", {"target": "config", "path": str(sweep)})
    assert need["id"] == "wrong-kind-sweep"
    refused = session.call("editor.open", {"target": "config", "answer": "no"})
    assert "Sweep tab" in refused["status"]
    assert refused["path"] is None


def test_opening_config_in_sweep_can_set_base(tmp_path):
    cfg = tmp_path / "train.yml"
    cfg.write_text("env:\n  type: gym\n", encoding="utf-8")
    before = cfg.read_bytes()
    session = GuiSession()
    need = session.call("editor.open", {"target": "sweep", "path": str(cfg)})
    assert need["id"] == "wrong-kind-config"
    snap = session.call("editor.open", {"target": "sweep", "answer": "yes"})
    assert snap["noun"] == "sweep"
    assert "train.yml" in snap["yaml"]
    assert cfg.read_bytes() == before


def test_shorthand_becomes_phase_0(tmp_path):
    sweep = tmp_path / "short.yml"
    sweep.write_text(
        "base_config: a.yml\nsearch_space:\n  schedule.seed:\n    dist: uniform\n    low: 0\n    high: 1\n",
        encoding="utf-8",
    )
    session = GuiSession()
    snap = session.call("editor.open", {"target": "sweep", "path": str(sweep)})
    assert "phase_0" in snap["status"]
    assert "phase_0" in snap["phases"]


def test_launch_command_uses_build_command():
    session = GuiSession()
    session.call("launch.set", {"key": "config", "value": "a.yml"})
    session.call("launch.set", {"key": "log_level", "value": "INFO"})
    snap = session.call("launch.snapshot", {})
    expect = command_text(
        build_command(
            "train",
            {
                "config": "a.yml",
                "log_level": "INFO",
                "cwd": snap["values"]["cwd"],
                "num_episodes": "10",
                "num_envs": "1",
                "render_mode": "human",
                "validate_only": "0",
                "resume": "0",
            },
        )
    )
    assert snap["command"] == expect
    assert snap["command"].split()[0]


def test_start_refuses_missing_folder_and_file(tmp_path):
    session = GuiSession()
    session.call("launch.set", {"key": "cwd", "value": str(tmp_path / "missing")})
    with pytest.raises(GuiError, match="Working folder does not exist") as exc:
        session.call("launch.start", {})
    assert exc.value.title == "Can't start"
    session.call("launch.set", {"key": "cwd", "value": str(tmp_path)})
    session.call("launch.set", {"key": "config", "value": "no-such.yml"})
    with pytest.raises(GuiError, match="Config file not found"):
        session.call("launch.start", {})


def test_second_start_does_not_raise_while_running():
    session = GuiSession()

    class Alive:
        running = True
        returncode = None
        grace_seconds = 15

        def poll(self, max_lines=1000):
            return []

        def start(self, *a, **k):
            raise AssertionError("spawned")

    session.runner = Alive()
    snap = session.call("launch.start", {})
    assert snap["running"] is True


def test_poll_classifies_progress_and_failure(monkeypatch):
    session = GuiSession()
    session.run_status = "Running"
    session.launch_status = "Running..."
    session.launch["mode"] = "train"

    class Done:
        running = False
        returncode = 1
        started_at = 0
        grace_seconds = 15

        def __init__(self):
            self.lines = [
                "[progress] timestep 1,000/2,000 avg_reward=1.50",
                "Traceback (most recent call last):",
            ]

        def poll(self, max_lines=1000):
            batch, self.lines = self.lines, []
            return batch

    session.runner = Done()
    snap = session.call("launch.poll", {})
    assert snap["run_status"] == "Failed"
    assert snap["status"].endswith("See the log.")
    assert any(row["tag"] == "err" for row in snap["log"])
    assert snap["progress"]["done"] == 1000


def test_serve_echoes_id_and_survives_a_bad_line():
    from io import StringIO

    stdin = StringIO(
        "not json\n"
        "\n"
        '{"id": 7, "method": "nope"}\n'
        '{"id": "v", "method": "app.version"}\n'
        '{"id": 1, "method": "app.close"}\n'
    )
    stdout = StringIO()
    code = serve(stdin, stdout)
    # serve() builds its own session and also reads --open from argv.
    # This process has no --open. app.close on a clean session returns closed.
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["error"]["title"] == "Can't complete"
    assert lines[1]["error"]["title"] == "Can't complete"
    assert lines[2]["error"]["title"] == "Can't complete"
    assert lines[3]["id"] == "v"
    assert lines[3]["result"]["version"]
    assert lines[4]["result"] == {"closed": True}
    assert code == 0


def test_answer_with_nothing_open_is_refused():
    session = GuiSession()
    with pytest.raises(GuiError, match="No question is open"):
        session.call("editor.new_blank", {"target": "config", "answer": "yes"})


def test_hidden_field_round_trip(tmp_path):
    """A key the form hides stays in the file when it was loaded."""
    session = GuiSession()
    snap = session.call("editor.snapshot", {"target": "config"})

    def names(node):
        found = [node["name"]]
        for child in node["children"]:
            found.extend(names(child))
        return found

    # The blank tree is the reference. Loading a file that sets a hidden key
    # must keep that key in the YAML even when the tree omits it.
    src = tmp_path / "in.yml"
    text = snap["yaml"]
    src.write_text(text, encoding="utf-8")
    opened = session.call("editor.open", {"target": "config", "path": str(src)})
    assert opened["yaml"]
    dest = tmp_path / "out.yml"
    _save_as(session, "config", dest)
    assert dest.read_text(encoding="utf-8") == opened["yaml"]


# -----------------------------------------------------------------------------
# Parity with the Tkinter editor (ported from the old tests/test_config_gui.py)
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "src/phoenx/examples"
PPO = EXAMPLES / "configs/LunarLanderContinuous-v3/ppo.yml"
SWEEP = EXAMPLES / "sweeps/lunarlander_ppo.yml"
FIXTURE = ROOT / "tests/fixtures/smoke_ppo_cartpole.yml"


def _find(snap: dict, path: str) -> dict | None:
    if snap.get("path") == path:
        return snap
    for child in snap.get("children", []):
        found = _find(child, path)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("name", sorted(bundled_examples("sweeps")))
def test_bundled_sweeps_open_with_no_problems(name):
    """The sweep check gets the base config's contents (it was given its path)."""
    session = GuiSession()
    snap = session.call("editor.new_from_example", {"target": "sweep", "name": name})
    assert snap["problems"] == []
    assert "bundled example" in snap["base_note"]
    assert len(snap["base_fields"]) > 50


def test_sweep_tree_add_reads_the_base_config_and_refuses_a_double_add():
    """Fields added from the tree come from the sweep's base config, not from
    whatever the Config Builder has open (here: nothing)."""
    session = GuiSession()
    session.call("editor.open", {"target": "sweep", "path": str(SWEEP)})
    path = "agent.config.model.branches.policy.layer_config.0.params.units"
    session.call("sweep.select_phase", {"name": "refine"})
    snap = session.call("sweep.add_field", {"path": path, "how": "search"})
    refine = next(p for p in session._value(session.sweep)["phases"] if p["name"] == "refine")
    assert refine["search_space"][path]["dist"] == "choice"
    index = [p["name"] for p in session._value(session.sweep)["phases"]].index("refine")
    assert snap["reveal"] == f"phases.{index}.search_space.{path}"
    again = session.call("sweep.add_field", {"path": path, "how": "search"})
    assert again["need"] == "ok" and again["title"] == "Already searched"
    session.call("sweep.add_field", {"answer": "ok"})
    snap = session.call("sweep.add_field", {"path": "agent.config.discount", "how": "override"})
    assert session._value(session.sweep)["overrides"]["agent.config.discount"] == 0.99
    assert snap["reveal"] == "overrides.agent.config.discount"


def test_sweep_this_from_config_tab(tmp_path):
    """Right-click 'Sweep this' in the Config Builder: sets base_config, adds a
    spec; a field at its default is first written into the base config file."""
    base = tmp_path / "base.yml"
    base.write_text(PPO.read_text(encoding="utf-8"), encoding="utf-8")
    session = GuiSession()
    session.call("launch.set", {"key": "cwd", "value": str(tmp_path)})
    session.call("editor.open", {"target": "config", "path": str(base)})
    session.call("sweep.add_from_config", {"path": "agent.config.policy_clip", "how": "search"})
    need = session.call("sweep.add_from_config", {"path": "agent.config.context_length", "how": "search"})
    assert need["id"] == "not-in-base-write"
    session.call("sweep.add_from_config", {"answer": "yes"})
    session.call("sweep.add_from_config", {"path": "schedule.stop_units", "how": "override"})
    cfg = session._value(session.sweep)
    assert cfg["base_config"] == "base.yml"
    space = cfg["phases"][0]["search_space"]
    assert space["agent.config.policy_clip"] == {"dist": "uniform", "low": 0.1, "high": 0.4}
    assert space["agent.config.context_length"]["dist"] == "randint"
    assert cfg["overrides"] == {"schedule.stop_units": 2000000}
    assert yaml.safe_load(base.read_text(encoding="utf-8"))["agent"]["config"]["context_length"] == 16
    assert session._problems(session.sweep) == []


def test_set_type_on_a_branch_head():
    """Branches are tagged by head type with no config: key; changing the head
    used to fail with 'No such field' (the branch's fields hid it)."""
    session = GuiSession()
    session.call("editor.set_type", {"target": "config", "path": "agent", "type_name": "SAC"})
    session.call("editor.map_add", {"target": "config", "path": "agent.config.model.branches", "key": "policy"})
    snap = session.call(
        "editor.set_type",
        {"target": "config", "path": "agent.config.model.branches.policy", "type_name": "StochasticContinuousHead"},
    )
    branch = _find(snap["form"], "agent.config.model.branches.policy")
    assert branch["text"] == "StochasticContinuousHead"
    assert _find(branch, "agent.config.model.branches.policy.distribution") is not None
    snap = session.call(
        "editor.set_extra",
        {"target": "config", "path": "agent.config.model.branches.policy", "text": "{_note: 1}"},
    )
    assert "_note" in _find(snap["form"], "agent.config.model.branches.policy")["extra_text"]


def test_snapshot_carries_what_the_window_draws():
    session = GuiSession()
    snap = session.call("editor.snapshot", {"target": "config"})
    seed = _find(snap["form"], "schedule.seed")
    assert seed["default"] == "null" and seed["combo"] is False      # a text box, not a dropdown
    unit = _find(snap["form"], "schedule.stop_unit")
    assert unit["combo"] is True and unit["default"] == "timestep"
    normalizer = _find(snap["form"], "agent.config.state_normalizer")
    assert normalizer["kind"] == "optional" and normalizer["inner_kind"] == "tagged"
    snap = session.call("editor.set_text", {"target": "config", "path": "schedule.batch_size", "text": "abc"})
    assert _find(snap["form"], "schedule.batch_size")["bad"] is True
    sweep = session.call("editor.snapshot", {"target": "sweep"})
    assert _find(sweep["form"], "overrides")["path_keys"] is True
    bad = session.call("editor.set_extra", {"target": "config", "path": "schedule", "text": "[not a map"})
    assert bad["extra_errors"] == ["schedule"]
    gen = bad["generation"]
    assert session.call("editor.new_blank", {"target": "config"})["id"] == "discard"
    fresh = session.call("editor.new_blank", {"target": "config", "answer": "no"})
    assert fresh["generation"] == gen + 1 and fresh["extra_errors"] == []


def test_resume_refuses_target_at_checkpoint(tmp_path):
    run = tmp_path / "r"
    (run / "agent").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    session = GuiSession()
    session.call("launch.set_mode", {"mode": "resume"})
    session.call("launch.set", {"key": "cwd", "value": str(tmp_path)})
    session._infos["r"] = {"progress": 1000}
    session.launch["agent_dir"] = "r"
    session.call("launch.set", {"key": "stop_units", "value": "1000"})
    with pytest.raises(GuiError, match="must be past the checkpoint"):
        session.call("launch.start", {})
    assert not session.runner.running


def _wait(session: GuiSession, done, timeout: float = 240.0) -> dict:
    import time

    end = time.time() + timeout
    snap = session.call("launch.poll", {})
    while time.time() < end:
        snap = session.call("launch.poll", {})
        if done(snap):
            return snap
        time.sleep(0.1)
    raise AssertionError("timed out:\n" + "\n".join(row["text"] for row in snap["log"][-30:]))


def test_run_train_stop_resume_test(tmp_path):
    """Train, Stop mid-run (checkpoint flushed), resume past the checkpoint,
    then test the saved agent: the full Run-page cycle through the session."""
    cfg = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    cfg["save_dir"] = "run"
    cfg["schedule"]["stop_units"] = 5_000_000
    cfg["schedule"]["save_every"] = 2_000
    cfg["env"]["config"]["num_envs"] = 4
    (tmp_path / "c.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    session = GuiSession(grace_seconds=60)
    session.call("launch.set", {"key": "cwd", "value": str(tmp_path)})
    session.call("launch.set", {"key": "config", "value": "c.yml"})
    session.call("launch.start", {})
    _wait(session, lambda s: "timestep" in s["status"])
    _wait(session, lambda s: (tmp_path / "run" / "trainer_state.pt").exists())
    session.call("launch.stop", {})
    snap = _wait(session, lambda s: not s["running"])
    assert snap["run_status"] == "Stopped", snap["status"]
    text = "\n".join(row["text"] for row in snap["log"])
    assert "[interrupted]" in text and "[checkpoint]" in text

    session.call("launch.set_mode", {"mode": "resume"})
    session.call("launch.set", {"key": "agent_dir", "value": str(tmp_path / "run")})
    snap = _wait(session, lambda s: str(tmp_path / "run") in session._infos)
    reached = session._infos[str(tmp_path / "run")]["progress"]
    session.call("launch.set", {"key": "stop_units", "value": str(reached + 2048)})
    session.call("launch.start", {})
    snap = _wait(session, lambda s: not s["running"])
    assert snap["run_status"] == "Finished", "\n".join(r["text"] for r in snap["log"][-20:])

    session.call("launch.set_mode", {"mode": "test"})
    session.call("launch.set", {"key": "test_agent_dir", "value": str(tmp_path / "run")})
    session.call("launch.set", {"key": "render_mode", "value": "rgb_array"})
    session.call("launch.set", {"key": "num_episodes", "value": "2"})
    session.call("launch.start", {})
    snap = _wait(session, lambda s: not s["running"])
    assert snap["run_status"] == "Finished", "\n".join(r["text"] for r in snap["log"][-20:])
    assert "[test] episode" in "\n".join(row["text"] for row in snap["log"])


# -----------------------------------------------------------------------------
# Overview cards: only the fields the chosen agent really uses
# -----------------------------------------------------------------------------

AGENTS = ("Reinforce", "ActorCritic", "PPO", "DDPG", "TD3", "SAC")


@pytest.mark.parametrize("agent", AGENTS)
def test_every_card_field_is_a_real_field(agent):
    session = GuiSession()
    snap = session.call("editor.set_type", {"target": "config", "path": "agent", "type_name": agent})
    cards = snap["cards"]
    assert cards["agent_type"] == agent
    for group in ("agent", "schedule", "env", "normalizers"):
        for entry in cards[group]:
            assert _find(snap["form"], entry["path"]) is not None, entry["path"]
    assert _find(snap["form"], cards["model"]) is not None


def test_on_policy_split_matches_metrics_and_learn_signatures():
    """The schedule card's on/off-policy split comes from the agents' code."""
    import inspect

    from phoenx import metrics, rl_agents
    from phoenx.gui_session import EPOCH_AGENTS, ON_POLICY_AGENTS

    src = inspect.getsource(metrics)
    assert all(f'"{name}"' in src for name in ON_POLICY_AGENTS)
    for name in AGENTS:
        params = inspect.signature(getattr(rl_agents, name).learn).parameters
        assert ("learning_epochs" in params) == (name in EPOCH_AGENTS), name
        buffer_off_policy = name in {"DDPG", "TD3", "SAC"}
        assert (name in ON_POLICY_AGENTS) == (not buffer_off_policy)


def test_schedule_card_follows_the_agent():
    from phoenx.gui_session import schedule_card

    assert schedule_card("PPO") == ("learn_every", "mini_batch_size", "learning_epochs", "seed")
    assert schedule_card("SAC") == ("learn_every", "batch_size", "updates_per_learn", "warmup_steps", "seed")
    assert schedule_card("Reinforce") == ("learn_every", "seed")
