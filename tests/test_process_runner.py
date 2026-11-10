"""Tests for the Run tab's process control (``phoenx.process_runner``) and
command building (``phoenx.launch_cmd.build_command``), without a display.

The runner is exercised with real child processes: output streaming,
graceful stop (the child sees KeyboardInterrupt and can clean up), and the
kill escalation for a child that ignores the interrupt.
"""

from __future__ import annotations

import sys
import time
import textwrap

import pytest

from phoenx.process_runner import ProcessRunner, parse_progress, python_command


def _drain(runner: ProcessRunner, timeout: float = 30.0) -> list[str]:
    lines: list[str] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        lines += runner.poll()
        if not runner.running:
            runner.wait(timeout=5)
            lines += runner.poll()
            return lines
        time.sleep(0.05)
    raise AssertionError(f"process still running; got {lines!r}")


def _script(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


def test_streams_output_and_exit_code():
    r = ProcessRunner()
    r.start(_script("""
        import sys
        print("hello")
        print("caf\\u00e9")  # non-ASCII must survive the pipe
        print("oops", file=sys.stderr)
        sys.exit(3)
    """))
    lines = _drain(r)
    assert lines[:3] == ["hello", "café", "oops"]
    assert r.returncode == 3


def test_refuses_second_start_while_running():
    r = ProcessRunner()
    r.start(_script("import time; time.sleep(5)"))
    try:
        with pytest.raises(RuntimeError):
            r.start(_script("pass"))
    finally:
        r.kill()
        r.wait(timeout=5)


def test_stop_is_graceful():
    """The child gets KeyboardInterrupt and can run its cleanup."""
    r = ProcessRunner(grace_seconds=20)
    r.start(_script("""
        import signal, sys, time
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.default_int_handler)
        print("ready", flush=True)
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("cleaned up", flush=True)
            sys.exit(130)
    """))
    deadline = time.time() + 20
    seen: list[str] = []
    while "ready" not in seen and time.time() < deadline:
        seen += r.poll()
        time.sleep(0.05)
    r.stop()
    seen += _drain(r)
    assert "cleaned up" in seen
    assert r.returncode == 130


def test_stop_escalates_to_kill():
    r = ProcessRunner(grace_seconds=0.5)
    r.start(_script("""
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(60)
    """))
    deadline = time.time() + 20
    seen: list[str] = []
    while "ready" not in seen and time.time() < deadline:
        seen += r.poll()
        time.sleep(0.05)
    r.stop()
    _drain(r, timeout=20)  # poll() escalates once the grace period passes
    assert not r.running


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "[progress] timestep 5,000/20,000 episodes=12 avg_reward=3.50 elapsed=0:00:07",
            ("train", "timestep", 5000, 20000, 12, 3.5),
        ),
        (
            "[progress] episode 40/100 avg_reward=-1.25",
            ("train", "episode", 40, 100, None, -1.25),
        ),
        (
            "[test] episode 3/10 avg_reward=200.00",
            ("test", "episode", 3, 10, None, 200.0),
        ),
    ],
)
def test_parse_progress(line, expected):
    p = parse_progress(line)
    assert (p.mode, p.unit, p.done, p.total, p.episodes, p.avg_reward) == expected


def test_parse_progress_ignores_other_lines():
    assert parse_progress("Traceback (most recent call last):") is None
    assert parse_progress("[checkpoint] saved best at timestep 100") is None


def test_python_command_uses_this_interpreter():
    assert python_command("phoenx.cli.train", ["--config", "a.yml"]) == [
        sys.executable,
        "-m",
        "phoenx.cli.train",
        "--config",
        "a.yml",
    ]


# =============================================================================
# Run tab command building (no Tk needed)
# =============================================================================


def test_build_command_modes():
    from phoenx.launch_cmd import build_command

    assert build_command("train", {"config": "a.yml", "log_level": "INFO"})[2:] == [
        "phoenx.cli.train",
        "--config",
        "a.yml",
        "--log_level",
        "INFO",
    ]
    assert build_command("resume", {"agent_dir": "runs/x", "stop_units": "5000"})[
        2:
    ] == ["phoenx.cli.train", "--agent_dir", "runs/x", "--stop_units", "5000"]
    assert build_command(
        "test",
        {
            "agent_dir": "runs/x",
            "num_episodes": "3",
            "render_mode": "headless",
            "seed": "",
        },
    )[2:] == [
        "phoenx.cli.test",
        "--agent_dir",
        "runs/x",
        "--num_episodes",
        "3",
        "--render_mode",
        "headless",
    ]
    assert build_command(
        "sweep",
        {"sweep": "s.yml", "validate_only": "1", "resume": "0", "num_samples": "4"},
    )[2:] == [
        "phoenx.cli.tune",
        "--config",
        "s.yml",
        "--num-samples",
        "4",
        "--validate-only",
    ]


def test_build_command_requires_inputs():
    from phoenx.launch_cmd import build_command

    with pytest.raises(ValueError, match="Config file is required"):
        build_command("train", {"config": "  "})
    with pytest.raises(ValueError, match="Agent folder is required"):
        build_command("test", {})


def test_describe_run():
    from phoenx.launch_cmd import describe_run

    text = describe_run(
        {
            "agent_type": "SAC",
            "env": "Pendulum-v1",
            "stop_unit": "timestep",
            "stop_units": 1000,
            "progress": 1000,
            "best_reward": -150.5,
            "has_buffer": False,
            "buffer_type": "ReplayBuffer",
        }
    )
    assert "SAC on Pendulum-v1" in text and "1,000 of 1,000" in text
    assert "already reached its limit" in text and "replay buffer" in text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal inheritance")
def test_stop_works_when_started_with_sigint_ignored(tmp_path):
    """A child that inherits SIGINT=ignored (nohup, a background job) still
    turns Stop into KeyboardInterrupt, so training can flush its checkpoint."""
    import signal as _signal

    script = tmp_path / "child.py"
    script.write_text(
        "import time\n"
        "from phoenx.process_runner import run_interruptible\n"
        "def main():\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(60)\n"
        "run_interruptible(main)\n",
        encoding="utf-8",
    )
    previous = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    try:
        import os
        from pathlib import Path

        import phoenx

        src = str(Path(phoenx.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
        runner = ProcessRunner(grace_seconds=20)
        runner.start([sys.executable, str(script)], cwd=str(tmp_path), env=env)
    finally:
        _signal.signal(_signal.SIGINT, previous)
    deadline = time.time() + 30
    lines: list[str] = []
    while "ready" not in lines and time.time() < deadline:
        lines += runner.poll()
        time.sleep(0.05)
    runner.stop()
    while runner.running and time.time() < deadline:
        lines += runner.poll()
        time.sleep(0.05)
    lines += runner.poll()
    assert "[interrupted] stopped by user" in lines
    assert runner.returncode == 130
