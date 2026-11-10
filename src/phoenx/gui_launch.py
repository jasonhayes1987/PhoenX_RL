"""Find the PhoenX window and wait until it closes.

``phoenx-config gui`` uses this module. The window is a separate program.
This module does not import torch or tkinter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_NOT_FOUND = "PhoenX window not found. Install it, or set PHOENX_GUI to the executable."


def _binary_name() -> str:
    """Return the window file name for this operating system."""
    return "phoenx-rl.exe" if sys.platform == "win32" else "phoenx-rl"


def _repo_release() -> Path:
    """Return the release binary built inside this checkout."""
    import phoenx

    repo = Path(phoenx.__file__).resolve().parents[2]
    return repo / "desktop" / "src-tauri" / "target" / "release" / _binary_name()


def _installed_copy() -> Path | None:
    """Return the per-user install path on Windows, or None elsewhere."""
    if sys.platform != "win32":
        return None
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    return Path(local) / "PhoenX RL" / "phoenx-rl.exe"


def resolve_window_executable() -> Path | None:
    """Return the window executable, or None when none is available.

    A set ``PHOENX_GUI`` is used alone. A missing file is a failed search.
    When the variable is unset, a release binary in this checkout wins over
    the per-user install under ``%LOCALAPPDATA%/PhoenX RL``.

    Returns:
        The executable path, or None.
    """
    override = os.environ.get("PHOENX_GUI")
    if override is not None:
        path = Path(override)
        return path if path.is_file() else None
    release = _repo_release()
    if release.is_file():
        return release
    installed = _installed_copy()
    if installed is not None and installed.is_file():
        return installed
    return None


def launch_window(open_path: str | None = None) -> int:
    """Start the window and block until it exits.

    Args:
        open_path: Config or sweep file to open. None starts a blank window.

    Returns:
        The window's exit code, or 1 when no executable was found.
    """
    exe = resolve_window_executable()
    if exe is None:
        print(_NOT_FOUND, file=sys.stderr)
        return 1
    argv = [str(exe)]
    if open_path:
        argv.extend(["--open", open_path])
    env = os.environ.copy()
    env["PHOENX_PYTHON"] = sys.executable
    completed = subprocess.run(argv, env=env)
    return int(completed.returncode)


def doctor() -> int:
    """Print the interpreter and the window path. Do not start the window.

    Returns:
        0 when an executable was found, otherwise 1.
    """
    exe = resolve_window_executable()
    print(f"python: {sys.executable}")
    print(f"window: {exe if exe is not None else 'MISSING'}")
    return 0 if exe is not None else 1
