"""Start a local MLflow tracking server on demand.

`MLflowCallback` (and the sweep driver) call `ensure_server` before a run
talks to MLflow. When the tracking URI points at this machine
(``127.0.0.1`` / ``localhost``) and nothing answers there, a background
``mlflow server`` is started with the same Python interpreter, and the call
waits until its ``/health`` endpoint responds. The server keeps running after
the run ends so the dashboard stays available; stop it with
``python -m phoenx.mlflow_server stop``.

All runs are stored in one fixed folder, ``~/.phoenx/mlflow`` (override with
the ``PHOENX_MLFLOW_HOME`` environment variable):

    mlflow.db        run / metric / param database (SQLite)
    artifacts/       checkpoints, videos, config files
    server.log       the server's output (read this if it fails to start)
    server.pid       process id of the server PhoenX started

Command line:

    python -m phoenx.mlflow_server start    # start (or confirm) the server
    python -m phoenx.mlflow_server status   # is it up? where is the data?
    python -m phoenx.mlflow_server stop     # stop the server PhoenX started
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

#: Address used when neither the config nor ``MLFLOW_TRACKING_URI`` sets one.
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

#: Seconds to wait for a newly started server (the first start creates the
#: database, which takes a while on Windows).
STARTUP_TIMEOUT = 90.0

#: A lock file younger than this means another process is already starting
#: the server; older ones are treated as left over from a crash.
_STALE_LOCK_SECONDS = STARTUP_TIMEOUT + 30.0

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def server_home() -> Path:
    """Folder holding the database, artifacts, log and pid file."""
    return Path(os.environ.get("PHOENX_MLFLOW_HOME") or Path.home() / ".phoenx" / "mlflow")


def is_local(uri: str) -> bool:
    """True for an ``http(s)`` URI that points at this machine."""
    parsed = urlparse(uri)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "") in _LOCAL_HOSTS


def is_up(uri: str, timeout: float = 2.0) -> bool:
    """True when an MLflow server answers ``/health`` at ``uri``."""
    try:
        return requests.get(f"{uri.rstrip('/')}/health", timeout=timeout).ok
    except requests.RequestException:
        return False


def _server_command(uri: str, home: Path) -> list[str]:
    parsed = urlparse(uri)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return [
        sys.executable, "-m", "mlflow", "server",
        "--host", parsed.hostname,
        "--port", str(port),
        "--backend-store-uri", f"sqlite:///{(home / 'mlflow.db').as_posix()}",
        "--artifacts-destination", (home / "artifacts").as_posix(),
        "--workers", "1",
    ]


def _spawn(uri: str, home: Path) -> subprocess.Popen:
    """Start the server detached from this process, so it outlives the run."""
    home.mkdir(parents=True, exist_ok=True)
    log = open(home / "server.log", "ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # No console window, own process group (Ctrl+C on the run doesn't
        # kill it), not tied to the parent's console.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    process = subprocess.Popen(_server_command(uri, home), env=env, **kwargs)
    log.close()
    (home / "server.pid").write_text(str(process.pid))
    return process


def _take_lock(home: Path) -> bool:
    """Create ``server.lock`` atomically; False if another process holds it."""
    home.mkdir(parents=True, exist_ok=True)
    lock = home / "server.lock"
    try:
        if time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS:
            lock.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _log_tail(home: Path, lines: int = 15) -> str:
    try:
        text = (home / "server.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no server.log)"
    return "\n".join(text.splitlines()[-lines:])


def ensure_server(uri: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Make sure an MLflow server answers at ``uri``, starting one if it is local.

    Safe to call from many processes at once (sweep trials): one process
    starts the server while the others wait for it.

    Args:
        uri: Tracking URI, e.g. ``http://127.0.0.1:5000``.
        timeout: Seconds to wait for a new server to come up.

    Returns:
        True if this call started the server, False if it was already up.

    Raises:
        ValueError: If ``uri`` is a remote server that is not reachable, or a
            local server could not be started (the message includes the end
            of ``server.log``).
    """
    if is_up(uri):
        return False
    if not is_local(uri):
        raise ValueError(
            f"MLflow tracking server at {uri} is not reachable, and PhoenX only "
            "starts servers on this machine (127.0.0.1 / localhost). Start it "
            "there, or point tracking_uri / MLFLOW_TRACKING_URI at a running server."
        )

    home = server_home()
    started = _take_lock(home)
    process = _spawn(uri, home) if started else None
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if is_up(uri):
                return started
            if process is not None and process.poll() is not None:
                raise ValueError(
                    f"The MLflow server PhoenX started for {uri} exited "
                    f"(code {process.returncode}). Is the port in use by another "
                    f"program? Last lines of {home / 'server.log'}:\n{_log_tail(home)}"
                )
            time.sleep(0.5)
    finally:
        if started:
            (home / "server.lock").unlink(missing_ok=True)
    raise ValueError(
        f"The MLflow server at {uri} did not come up within {timeout:.0f}s. "
        f"Last lines of {home / 'server.log'}:\n{_log_tail(home)}"
    )


def stop_server() -> bool:
    """Stop the server PhoenX started (from ``server.pid``). True if one was stopped."""
    pid_file = server_home() / "server.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        if os.name == "nt":
            # /T also ends the server's worker processes.
            result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            stopped = result.returncode == 0
        else:
            import signal
            os.killpg(pid, signal.SIGTERM)
            stopped = True
    except (OSError, ProcessLookupError):
        stopped = False
    pid_file.unlink(missing_ok=True)
    return stopped


def _main(argv: list[str]) -> int:
    command = argv[0] if argv else "status"
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if command == "start":
        try:
            started = ensure_server(uri)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"{'Started' if started else 'Already running'}: {uri}")
    elif command == "stop":
        print("Stopped." if stop_server() else "No PhoenX-started MLflow server found.")
    elif command == "status":
        print(f"{uri}: {'up' if is_up(uri) else 'down'}")
        print(f"data folder: {server_home()}")
    else:
        print("usage: python -m phoenx.mlflow_server [start|stop|status]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
