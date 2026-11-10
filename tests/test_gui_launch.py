"""Launcher contracts for the PhoenX window.

The command starts the resolved executable and does not import tkinter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import phoenx.gui_launch as gui_launch
from phoenx import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_gui_command_starts_the_window_without_tkinter(tmp_path: Path) -> None:
    """C30: phoenx-config gui FILE runs the stub with --open and this Python."""
    record = tmp_path / "record.txt"
    stub = tmp_path / "stub.cmd"
    stub.write_text(
        "\r\n".join(
            [
                "@echo off",
                f'echo %* > "{record}"',
                f'echo %PHOENX_PYTHON%>> "{record}"',
                "exit /b 4",
                "",
            ]
        ),
        encoding="ascii",
    )
    script = "\n".join(
        [
            "import sys",
            "import os",
            f"os.environ['PHOENX_GUI'] = r'{stub}'",
            "from phoenx.cli.config import main",
            "print('TK', 'tkinter' in sys.modules)",
            "raise SystemExit(main(['gui', r'my run.yml']))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "TK False" in result.stdout
    text = record.read_text(encoding="utf-8")
    assert "--open" in text
    assert "my run.yml" in text
    assert sys.executable in text


def test_search_prefers_override_then_repo_then_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C31: each root wins only when the earlier ones are missing."""
    repo = tmp_path / "repo.exe"
    installed = tmp_path / "installed.exe"
    override = tmp_path / "override.exe"
    for path in (repo, installed, override):
        path.write_bytes(b"")
    monkeypatch.setattr(gui_launch, "_repo_release", lambda: repo)
    monkeypatch.setattr(gui_launch, "_installed_copy", lambda: installed)

    monkeypatch.setenv("PHOENX_GUI", str(override))
    assert gui_launch.resolve_window_executable() == override

    missing = tmp_path / "missing.exe"
    monkeypatch.setenv("PHOENX_GUI", str(missing))
    assert gui_launch.resolve_window_executable() is None

    monkeypatch.delenv("PHOENX_GUI")
    assert gui_launch.resolve_window_executable() == repo

    monkeypatch.setattr(gui_launch, "_repo_release", lambda: tmp_path / "absent.exe")
    assert gui_launch.resolve_window_executable() == installed


def test_doctor_prints_paths_and_does_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C31: doctor prints two lines and never starts a process."""
    found = tmp_path / "window.exe"
    found.write_bytes(b"")
    monkeypatch.delenv("PHOENX_GUI", raising=False)
    monkeypatch.setattr(gui_launch, "_repo_release", lambda: found)
    monkeypatch.setattr(gui_launch, "_installed_copy", lambda: None)

    def explode(*_args, **_kwargs):
        raise AssertionError("doctor spawned a process")

    monkeypatch.setattr(gui_launch.subprocess, "run", explode)
    assert gui_launch.doctor() == 0
    out = capsys.readouterr().out
    assert f"python: {sys.executable}" in out
    assert f"window: {found}" in out

    monkeypatch.setenv("PHOENX_GUI", str(tmp_path / "nope.exe"))
    assert gui_launch.doctor() == 1
    missed = capsys.readouterr().out
    assert "window: MISSING" in missed


def test_launch_without_a_window_prints_the_not_found_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C31: a normal launch with nothing to run returns 1."""
    monkeypatch.delenv("PHOENX_GUI", raising=False)
    monkeypatch.setattr(gui_launch, "_repo_release", lambda: Path("missing-release"))
    monkeypatch.setattr(gui_launch, "_installed_copy", lambda: None)
    assert gui_launch.launch_window() == 1
    assert (
        "PhoenX window not found. Install it, or set PHOENX_GUI to the executable."
        in capsys.readouterr().err
    )


def test_version_label_matches_the_package() -> None:
    """C40: the brand label is the installed PhoenX version."""
    text = (ROOT / "desktop" / "src" / "brand" / "index.ts").read_text(encoding="utf-8")
    assert f'versionLabel = "{__version__}"' in text


def test_installer_config_is_per_user_and_has_no_python() -> None:
    """C32: the bundle names, size, and install mode match the window spec."""
    path = ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    window = data["app"]["windows"][0]
    blob = json.dumps(data).lower()
    assert data["productName"] == "PhoenX RL"
    assert data["identifier"] == "com.phoenx.rl"
    assert data["bundle"]["windows"]["nsis"]["installMode"] == "currentUser"
    assert window["width"] == 1360
    assert window["height"] == 880
    assert window["minWidth"] == 900
    assert window["minHeight"] == 560
    assert "python" not in blob


def test_tkinter_modules_are_gone() -> None:
    """C33: the old window modules can no longer be imported."""
    for name in ("phoenx.config_gui", "phoenx.sweep_editor", "phoenx.run_panel"):
        with pytest.raises(ModuleNotFoundError):
            __import__(name)
