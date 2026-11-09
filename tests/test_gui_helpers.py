"""Contracts for the moved example-file and launch-command helpers.

C5: importing ``phoenx.example_files`` or ``phoenx.launch_cmd``, and calling the
helpers below, must not import ``torch`` or ``tkinter``. Each check is a real
second process. The child does not catch exceptions: a stub that raises
``NotImplementedError`` exits non-zero and fails the test by assertion.

REEXPORT: the names still published from ``config_gui`` and ``run_panel`` are
the same objects as the moved helpers.
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


def test_config_gui_bundled_examples_is_example_files_bundled_examples():
    """REEXPORT: config_gui.bundled_examples is example_files.bundled_examples."""
    import phoenx.config_gui as config_gui
    import phoenx.example_files as example_files

    assert config_gui.bundled_examples is example_files.bundled_examples


def test_run_panel_build_command_is_launch_cmd_build_command():
    """REEXPORT: run_panel.build_command is launch_cmd.build_command."""
    import phoenx.launch_cmd as launch_cmd
    import phoenx.run_panel as run_panel

    assert run_panel.build_command is launch_cmd.build_command


def test_run_panel_describe_run_is_launch_cmd_describe_run():
    """REEXPORT: run_panel.describe_run is launch_cmd.describe_run."""
    import phoenx.launch_cmd as launch_cmd
    import phoenx.run_panel as run_panel

    assert run_panel.describe_run is launch_cmd.describe_run


def test_run_panel_join_is_launch_cmd_command_text():
    """REEXPORT: run_panel._join is launch_cmd.command_text."""
    import phoenx.launch_cmd as launch_cmd
    import phoenx.run_panel as run_panel

    assert run_panel._join is launch_cmd.command_text
