"""Contracts for the moved example-file and launch-command helpers.

C5: importing ``phoenx.example_files`` or ``phoenx.launch_cmd``, and calling the
helpers below, must not import ``torch`` or ``tkinter``. Each check is a real
second process. The child does not catch exceptions: a stub that raises
``NotImplementedError`` exits non-zero and fails the test by assertion.

"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_FRESH_TIMEOUT_S = 60


def _fresh(script: str) -> None:
    """Run ``script`` in a new interpreter. Non-zero exit fails the test."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=_FRESH_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bundled_examples_fresh_interpreter_omits_torch_and_tkinter():
    """C5: importing example_files and both bundled_examples calls stay clean.

    ``bundled_examples()`` and ``bundled_examples("sweeps")`` share one
    interpreter. Return values are not checked.
    """
    _fresh("""
        import sys
        import phoenx.example_files as example_files

        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        example_files.bundled_examples()
        example_files.bundled_examples("sweeps")
        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        """)


def test_build_command_fresh_interpreter_omits_torch_and_tkinter():
    """C5: importing launch_cmd and calling build_command stay clean."""
    _fresh("""
        import sys
        import phoenx.launch_cmd as launch_cmd

        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        launch_cmd.build_command("train", {"config": "a.yml", "log_level": "INFO"})
        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        """)


def test_describe_run_fresh_interpreter_omits_torch_and_tkinter():
    """C5: importing launch_cmd and calling describe_run stay clean."""
    _fresh("""
        import sys
        import phoenx.launch_cmd as launch_cmd

        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        launch_cmd.describe_run({"error": "missing"})
        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        """)


def test_command_text_fresh_interpreter_omits_torch_and_tkinter():
    """C5: importing launch_cmd and calling command_text stay clean."""
    _fresh("""
        import sys
        import phoenx.launch_cmd as launch_cmd

        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        launch_cmd.command_text(
            ["python", "-m", "phoenx.cli.train", "--config", "a.yml"]
        )
        leaked = [name for name in ("torch", "tkinter") if name in sys.modules]
        assert not leaked, leaked
        """)
