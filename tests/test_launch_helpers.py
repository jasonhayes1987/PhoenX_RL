"""Branch and error-path coverage for the moved example and launch helpers.

Expected command lines and run summaries are written out literally. Installed
example text is read from the package directory with ``pathlib``, not from
``bundled_examples``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import phoenx.example_files as example_files
import phoenx.examples as examples
import phoenx.launch_cmd as launch_cmd


def _argv(module: str, *flags: str) -> list[str]:
    """``[this interpreter, -m, module, *flags]``, the closed form of an argv."""
    return [sys.executable, "-m", module, *flags]


def _examples_root() -> Path:
    return Path(examples.__file__).resolve().parent


def _yml_text_on_disk(folder: str) -> dict[str, str]:
    """Map each on-disk ``*.yml`` to its text, keyed by a forward-slash path."""
    root = _examples_root() / folder
    if not root.is_dir():
        return {}
    text: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name.endswith(".yml"):
            text[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return text


def test_bundled_examples_returns_installed_yml_text():
    """Default folder is configs; configs and sweeps match the installed files."""
    configs = _yml_text_on_disk("configs")
    sweeps = _yml_text_on_disk("sweeps")
    assert configs, "shipped configs should contain yml examples"
    assert any("/" in name for name in configs)
    assert sweeps, "shipped sweeps should contain yml examples"

    assert example_files.bundled_examples() == configs
    assert example_files.bundled_examples("configs") == configs
    assert example_files.bundled_examples("sweeps") == sweeps


def test_bundled_examples_ignores_directory_without_yml():
    """A real package directory that only contains Python files yields nothing."""
    isaac = _examples_root() / "isaac"
    assert isaac.is_dir()
    assert any(path.suffix == ".py" for path in isaac.iterdir())
    assert _yml_text_on_disk("isaac") == {}
    assert example_files.bundled_examples("isaac") == {}


def test_bundled_examples_non_directory_folder_is_empty(tmp_path, monkeypatch):
    """A missing folder and a file used as a folder are both empty mappings."""
    root = tmp_path / "examples"
    root.mkdir()
    (root / "configs").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(example_files.resources, "files", lambda _name: root)

    assert example_files.bundled_examples("configs") == {}
    assert example_files.bundled_examples("missing") == {}


def test_bundled_examples_walks_yml_in_case_sensitive_name_order(tmp_path, monkeypatch):
    """Sibling names are sorted case-sensitively; only ``.yml`` files are read."""
    root = tmp_path / "examples"
    configs = root / "configs"
    nested = configs / "a"
    nested.mkdir(parents=True)
    (configs / "B.yml").write_text("B: 1\n", encoding="utf-8")
    (configs / "Z.YML").write_text("nope\n", encoding="utf-8")
    (configs / "a.yml").write_text("a: 1\n", encoding="utf-8")
    (configs / "c.yml").write_text("c: 2\n", encoding="utf-8")
    (configs / "skip.yaml").write_text("no: 1\n", encoding="utf-8")
    (nested / "z.yml").write_text("z: café\n", encoding="utf-8")
    (nested / "note.txt").write_text("skip", encoding="utf-8")
    (nested / "m.py").write_text("pass\n", encoding="utf-8")
    notes = configs / "notes"
    notes.mkdir()
    (notes / "readme.txt").write_text("skip", encoding="utf-8")
    (root / "empty").mkdir()

    seen: list[str] = []

    def files(name: str):
        seen.append(name)
        return root

    monkeypatch.setattr(example_files.resources, "files", files)

    found = example_files.bundled_examples("configs")
    assert seen == ["phoenx.examples"]
    assert list(found) == ["B.yml", "a/z.yml", "a.yml", "c.yml"]
    assert found == {
        "B.yml": "B: 1\n",
        "a/z.yml": "z: café\n",
        "a.yml": "a: 1\n",
        "c.yml": "c: 2\n",
    }
    assert example_files.bundled_examples("empty") == {}


def test_bundled_examples_missing_package_returns_empty(monkeypatch):
    """A missing examples package is an empty mapping, not an exception."""

    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(example_files.resources, "files", missing)
    assert example_files.bundled_examples() == {}
    assert example_files.bundled_examples("sweeps") == {}


def test_bundled_examples_propagates_unrelated_lookup_errors(monkeypatch):
    """Only ``ModuleNotFoundError`` is turned into an empty mapping."""

    def boom(_name: str):
        raise FileNotFoundError("examples package unreadable")

    monkeypatch.setattr(example_files.resources, "files", boom)
    with pytest.raises(FileNotFoundError, match="^examples package unreadable$"):
        example_files.bundled_examples("configs")


def test_build_command_train_strips_values_and_optional_flags():
    """Train strips field text, drops a blank log level, and ignores other keys."""
    values = {
        "config": "  my file.yml  ",
        "stop_units": " 100 ",
        "log_level": " \t ",
        "agent_dir": "runs/ignored",
    }
    original = dict(values)
    assert launch_cmd.build_command("train", values) == _argv(
        "phoenx.cli.train",
        "--config",
        "my file.yml",
        "--stop_units",
        "100",
    )
    assert values == original

    assert launch_cmd.build_command(
        "train",
        {"config": "a.yml", "stop_units": "10", "log_level": "DEBUG"},
    ) == _argv(
        "phoenx.cli.train",
        "--config",
        "a.yml",
        "--stop_units",
        "10",
        "--log_level",
        "DEBUG",
    )


@pytest.mark.parametrize("config", [None, "", "\t"])
def test_build_command_train_rejects_missing_config(config):
    """Train rejects a missing config before it reads any optional field."""
    with pytest.raises(ValueError, match="^Config file is required$"):
        launch_cmd.build_command("train", {"config": config, "log_level": "INFO"})


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"agent_dir": None, "stop_units": "5"},
        {"agent_dir": " \t ", "log_level": "DEBUG"},
    ],
)
def test_build_command_resume_requires_agent_folder(values):
    """Resume's required field is the agent folder, including when it is blank."""
    with pytest.raises(ValueError, match="^Agent folder is required$"):
        launch_cmd.build_command("resume", values)


def test_build_command_resume_keeps_zero_stop_and_log_level():
    """Resume keeps a stop of ``0`` and places the log level after it."""
    assert launch_cmd.build_command(
        "resume",
        {"agent_dir": " runs/x ", "stop_units": "0", "log_level": " DEBUG "},
    ) == _argv(
        "phoenx.cli.train",
        "--agent_dir",
        "runs/x",
        "--stop_units",
        "0",
        "--log_level",
        "DEBUG",
    )


def test_build_command_test_emits_every_optional_in_order():
    """Test emits each optional flag after the agent folder, in field order."""
    assert launch_cmd.build_command(
        "test",
        {
            "agent_dir": "runs/x",
            "num_episodes": "3",
            "num_envs": "2",
            "render_mode": "human",
            "env": "CartPole-v1",
            "seed": "7",
            "log_level": "WARNING",
        },
    ) == _argv(
        "phoenx.cli.test",
        "--agent_dir",
        "runs/x",
        "--num_episodes",
        "3",
        "--num_envs",
        "2",
        "--render_mode",
        "human",
        "--env",
        "CartPole-v1",
        "--seed",
        "7",
        "--log_level",
        "WARNING",
    )


def test_build_command_test_drops_blank_options_and_keeps_zero():
    """Blank test options are omitted; the string ``0`` is still an option value."""
    assert launch_cmd.build_command(
        "test",
        {
            "agent_dir": "runs/x",
            "num_episodes": None,
            "num_envs": "0",
            "render_mode": "  ",
            "env": "",
            "seed": "\t",
            "log_level": None,
        },
    ) == _argv("phoenx.cli.test", "--agent_dir", "runs/x", "--num_envs", "0")


@pytest.mark.parametrize(
    "values",
    [{}, {"sweep": None}, {"sweep": " \t "}, {"base_config": "b.yml"}],
)
def test_build_command_sweep_requires_file(values):
    """Sweep requires its own file, even when a base config was filled in."""
    with pytest.raises(ValueError, match="^Sweep file is required$"):
        launch_cmd.build_command("sweep", values)


def test_build_command_sweep_emits_hyphen_flags_without_bare_switches():
    """Sweep options use hyphen flags, and only the exact text ``1`` adds a switch."""
    assert launch_cmd.build_command(
        "sweep",
        {
            "sweep": " s.yml ",
            "base_config": "base.yml",
            "num_samples": "   ",
            "max_concurrent": "2",
            "name": "trial",
            "storage_path": "out/dir",
            "from_phase": "3",
            "log_level": "INFO",
            "resume": " 0 ",
            "validate_only": "true",
        },
    ) == _argv(
        "phoenx.cli.tune",
        "--config",
        "s.yml",
        "--base-config",
        "base.yml",
        "--max-concurrent",
        "2",
        "--name",
        "trial",
        "--storage-path",
        "out/dir",
        "--from-phase",
        "3",
        "--log-level",
        "INFO",
    )


def test_build_command_sweep_resume_and_validate_only_when_value_is_one():
    """``resume`` and ``validate_only`` append bare switches after the options."""
    assert launch_cmd.build_command(
        "sweep",
        {
            "sweep": "s.yml",
            "name": "trial",
            "resume": " 1 ",
            "validate_only": " 1 ",
        },
    ) == _argv(
        "phoenx.cli.tune",
        "--config",
        "s.yml",
        "--name",
        "trial",
        "--resume",
        "--validate-only",
    )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bogus", "unknown mode 'bogus'"),
        ("Train", "unknown mode 'Train'"),
        ("", "unknown mode ''"),
        (" test ", "unknown mode ' test '"),
    ],
)
def test_build_command_unknown_mode_raises(mode, message):
    """An unrecognized mode is rejected without building a command."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        launch_cmd.build_command(mode, {"config": "a.yml"})


def test_describe_run_returns_error_text_only():
    """A reported error is the whole summary, with no checkpoint sentence added."""
    assert (
        launch_cmd.describe_run(
            {
                "error": "folder missing",
                "agent_type": "SAC",
                "env": "Pendulum-v1",
                "progress": 5,
                "stop_units": 10,
                "best_reward": 1.0,
            }
        )
        == "folder missing"
    )


def test_describe_run_blank_error_uses_checkpoint_text():
    """A blank or null error is ignored, and missing names are question marks."""
    assert launch_cmd.describe_run({"error": "", "progress": 4}) == (
        "? on ?. Best checkpoint at timestep 4."
    )
    assert launch_cmd.describe_run(
        {"error": None, "agent_type": "PPO", "env": "E", "progress": 4}
    ) == ("PPO on E. Best checkpoint at timestep 4.")


def test_describe_run_unknown_progress_with_int_limit():
    """No saved step, with an integer limit, names the limit and the unit."""
    text = launch_cmd.describe_run(
        {
            "agent_type": "PPO",
            "env": "CartPole-v1",
            "stop_unit": "episode",
            "stop_units": 1500,
            "progress": None,
        }
    )
    assert text == (
        "PPO on CartPole-v1. No trainer_state.pt, so the step reached is "
        "unknown (limit 1,500 episodes)."
    )


def test_describe_run_unknown_progress_without_int_limit():
    """No saved step and a non-integer limit produce the short sentence."""
    text = launch_cmd.describe_run(
        {
            "agent_type": "PPO",
            "env": "CartPole-v1",
            "stop_unit": "episode",
            "stop_units": "1500",
        }
    )
    assert text == "PPO on CartPole-v1. No trainer_state.pt."


def test_describe_run_zero_progress_uses_defaults():
    """Progress ``0`` is a real checkpoint; blank agent, env, and unit use defaults."""
    assert launch_cmd.describe_run(
        {
            "agent_type": "",
            "env": None,
            "stop_unit": "",
            "stop_units": None,
            "progress": 0,
        }
    ) == ("? on ?. Best checkpoint at timestep 0.")


def test_describe_run_below_limit_shows_zero_reward_and_saved_buffer():
    """Under the limit, a zero reward is shown and a saved buffer adds no warning."""
    assert launch_cmd.describe_run(
        {
            "agent_type": "SAC",
            "env": "Pendulum-v1",
            "stop_unit": "episode",
            "stop_units": 10000,
            "progress": 2500,
            "best_reward": 0,
            "buffer_type": "ReplayBuffer",
            "has_buffer": True,
        }
    ) == (
        "SAC on Pendulum-v1. Best checkpoint at episode 2,500 of 10,000, "
        "best avg reward 0.00."
    )


def test_describe_run_negative_best_reward_stays_negative():
    """A negative best reward keeps its minus sign and two decimal places."""
    text = launch_cmd.describe_run(
        {
            "agent_type": "SAC",
            "env": "Pendulum-v1",
            "progress": 1,
            "best_reward": -150.5,
        }
    )
    assert text == (
        "SAC on Pendulum-v1. Best checkpoint at timestep 1, "
        "best avg reward -150.50."
    )
    assert "-150.50" in text


def test_describe_run_past_limit_warns_and_notes_missing_prioritized_buffer():
    """Past the limit, a missing prioritized buffer adds both trailing sentences."""
    assert launch_cmd.describe_run(
        {
            "agent_type": "TD3",
            "env": "Hopper-v4",
            "stop_units": 100,
            "progress": 101,
            "buffer_type": "PrioritizedReplayBuffer",
        }
    ) == (
        "TD3 on Hopper-v4. Best checkpoint at timestep 101 of 100. "
        "It already reached its limit: set 'Train until' higher to train further. "
        "No replay buffer was saved, so resuming refills it from scratch."
    )


def test_describe_run_ignores_float_limit_and_inexact_buffer_name():
    """A float limit and a buffer name outside the replay pair add no extra sentences."""
    assert launch_cmd.describe_run(
        {
            "agent_type": "A2C",
            "env": "CartPole-v1",
            "stop_units": 100.0,
            "progress": 500,
            "best_reward": 1.234,
            "buffer_type": "ReplayBufferExtra",
            "has_buffer": False,
        }
    ) == ("A2C on CartPole-v1. Best checkpoint at timestep 500, best avg reward 1.23.")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ""),
        (
            ["python", "-m", "phoenx.cli.train", "--config", "a.yml"],
            "python -m phoenx.cli.train --config a.yml",
        ),
        (
            ["python", "--config", "my config.yml"],
            'python --config "my config.yml"',
        ),
        ([""], '""'),
        (["a&b"], "a&b"),
        (['say "hi"'], '"say \\"hi\\""'),
        (["foo\\"], "foo\\"),
        (["a \\"], '"a \\\\"'),
        (["C:\\Program Files\\Python"], '"C:\\Program Files\\Python"'),
    ],
)
def test_command_text_windows_quoting(monkeypatch, argv, expected):
    """On Windows, spaces and empty args are double-quoted; other characters stay literal."""
    monkeypatch.setattr(launch_cmd.sys, "platform", "win32")
    assert launch_cmd.command_text(argv) == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ""),
        (
            ["python", "-m", "phoenx.cli.train", "--config", "a.yml"],
            "python -m phoenx.cli.train --config a.yml",
        ),
        (
            ["python", "--config", "my config.yml"],
            "python --config 'my config.yml'",
        ),
        ([""], "''"),
        (["a&b"], "'a&b'"),
        (['say "hi"'], "'say \"hi\"'"),
        (["foo\\"], "'foo\\'"),
        (["a \\"], "'a \\'"),
        (["C:\\Program Files\\Python"], "'C:\\Program Files\\Python'"),
    ],
)
def test_command_text_posix_quoting(monkeypatch, argv, expected):
    """On other platforms, unsafe arguments are wrapped in single quotes."""
    monkeypatch.setattr(launch_cmd.sys, "platform", "linux")
    assert launch_cmd.command_text(argv) == expected
