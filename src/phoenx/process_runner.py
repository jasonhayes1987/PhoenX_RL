"""Run a PhoenX command in a child process and stream its output.

The config GUI's Run tab uses [ProcessRunner][phoenx.process_runner.ProcessRunner]
to start ``phoenx-train`` / ``phoenx-test`` / ``phoenx-tune`` exactly as the
terminal would (``python -m phoenx.cli.<name> ...`` with the same
interpreter), so training never runs inside the GUI process and a crash or
a hung CUDA context cannot take the window with it.

Output lines are collected by a reader thread and handed to the GUI through
``poll()``, which is safe to call from the Tk event loop.

Stopping is two-stage. ``stop()`` first sends an interrupt (SIGINT on
Linux/macOS; Ctrl+Break on Windows, which the PhoenX CLIs turn into
``KeyboardInterrupt``) so training can flush a pending best checkpoint and
Ray can shut trials down. If the process is still alive after
``grace_seconds``, the whole process tree is killed.

[parse_progress][phoenx.process_runner.parse_progress] reads the
``[progress]`` lines the Trainer prints when its output is not a terminal.
"""

from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

IS_WINDOWS = sys.platform.startswith("win")


@dataclass
class Progress:
    """One parsed ``[progress]`` line."""

    mode: str            # "train" or "test"
    unit: str            # "timestep" or "episode"
    done: int
    total: int | None
    episodes: int | None = None
    avg_reward: float | None = None
    elapsed: str | None = None

    @property
    def fraction(self) -> float | None:
        """Completed fraction in [0, 1], or None when the total is unknown."""
        if not self.total:
            return None
        return max(0.0, min(1.0, self.done / self.total))


_PROGRESS_RE = re.compile(
    r"^\[(?P<mode>progress|test)\]\s+(?P<unit>timestep|episode)\s+"
    r"(?P<done>[\d,]+)(?:/(?P<total>[\d,]+))?(?P<rest>.*)$")
_FIELD_RE = re.compile(r"(\w+)=(\S+)")


def parse_progress(line: str) -> Progress | None:
    """Parse a Trainer ``[progress]`` / ``[test]`` line, or return None.

    Example:
        >>> p = parse_progress("[progress] timestep 5,000/20,000 episodes=12 avg_reward=3.50 elapsed=0:00:07")
        >>> (p.done, p.total, p.episodes, p.avg_reward, round(p.fraction, 2))
        (5000, 20000, 12, 3.5, 0.25)
    """
    m = _PROGRESS_RE.match(line.strip())
    if not m:
        return None
    fields = dict(_FIELD_RE.findall(m.group("rest")))

    def num(key, cast):
        try:
            return cast(fields[key].replace(",", ""))
        except (KeyError, ValueError):
            return None

    total = m.group("total")
    return Progress(
        mode="train" if m.group("mode") == "progress" else "test",
        unit=m.group("unit"),
        done=int(m.group("done").replace(",", "")),
        total=int(total.replace(",", "")) if total else None,
        episodes=num("episodes", int),
        avg_reward=num("avg_reward", float),
        elapsed=fields.get("elapsed"),
    )


def install_interrupt_handlers() -> None:
    """Make the GUI's Stop raise ``KeyboardInterrupt`` in a PhoenX CLI.

    Windows can only interrupt a child process in its own process group with
    Ctrl+Break, whose default action kills Python outright, so Ctrl+Break is
    mapped to ``KeyboardInterrupt`` like Ctrl+C.

    Elsewhere, a process started in the background (``nohup``, ``&`` in a
    script, some launchers) inherits SIGINT set to *ignored*, and Python then
    never raises ``KeyboardInterrupt``: Stop would always end in a kill after
    the grace period, without flushing the best checkpoint. SIGINT is put
    back to the normal handler in that case.
    """
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def run_interruptible(main) -> None:
    """Run a CLI ``main`` and turn an interrupt into exit code 130 without a traceback."""
    install_interrupt_handlers()
    try:
        main()
    except KeyboardInterrupt:
        print("[interrupted] stopped by user", flush=True)
        sys.exit(130)


def python_command(module: str, args: Sequence[str]) -> list[str]:
    """``[python, -m, module, *args]`` using the interpreter running PhoenX."""
    return [sys.executable, "-m", module, *[str(a) for a in args]]


class ProcessRunner:
    """Start one child process, stream its output, stop it on request."""

    def __init__(self, grace_seconds: float = 15.0):
        """Create an idle runner; ``grace_seconds`` is how long ``stop`` waits before killing."""
        self.grace_seconds = grace_seconds
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop_requested_at: float | None = None
        self.started_at: float | None = None

    # -- state -----------------------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the child process is alive."""
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        """Exit code once finished, else None."""
        return None if self.proc is None else self.proc.poll()

    # -- control ---------------------------------------------------------------

    def start(self, argv: Sequence[str], cwd: str | Path | None = None,
              env: dict | None = None) -> None:
        """Launch ``argv``; raises RuntimeError if a process is already running."""
        if self.running:
            raise RuntimeError("a process is already running")
        child_env = dict(os.environ if env is None else env)
        # Line-by-line output, and UTF-8 so Rich / tracebacks never fail to
        # encode on a Windows console code page.
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env["PYTHONIOENCODING"] = "utf-8"
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # own process group for killpg
        self._lines = queue.Queue()
        self._stop_requested_at = None
        self.proc = subprocess.Popen(
            list(argv), cwd=str(cwd) if cwd else None, env=child_env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **kwargs)
        self.started_at = time.time()
        self._reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self._reader.start()

    def _read(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            self._lines.put(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        proc.stdout.close()

    def poll(self, max_lines: int = 2000) -> list[str]:
        """Return output lines received since the last call (non-blocking).

        Also escalates a pending ``stop()`` to a kill once the grace period
        has passed, so the GUI only needs to call this periodically.
        """
        out: list[str] = []
        try:
            while len(out) < max_lines:
                out.append(self._lines.get_nowait())
        except queue.Empty:
            pass
        if (self._stop_requested_at is not None and self.running
                and time.time() - self._stop_requested_at > self.grace_seconds):
            self.kill()
        return out

    def stop(self) -> None:
        """Ask the process to stop gracefully (escalates to ``kill`` later)."""
        if not self.running:
            return
        self._stop_requested_at = time.time()
        try:
            if IS_WINDOWS:
                os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.proc.pid, signal.SIGINT)
        except (OSError, ProcessLookupError):
            self.kill()

    def kill(self) -> None:
        """Kill the process and everything it started (Ray workers, Isaac Kit)."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                os.killpg(self.proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        self._stop_requested_at = None

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the process exits (tests / shutdown); returns the exit code."""
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        finally:
            if self._reader is not None:
                self._reader.join(timeout=1)
