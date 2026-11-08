"""The config GUI's Run tab: start training, resuming, testing and sweeps.

Each action runs the matching PhoenX command in a child process (see
[phoenx.process_runner][]), so the GUI behaves exactly like the terminal:

* **Train**: ``phoenx-train --config FILE``
* **Resume**: ``phoenx-train --agent_dir DIR [--stop_units N]``
* **Test**: ``phoenx-test --agent_dir DIR ...``
* **Sweep**: ``phoenx-tune --config SWEEP ...``

The exact command is shown (and can be copied) before it runs. Output
streams into the log pane; the Trainer's ``[progress]`` lines drive the
progress bar. Choosing an agent folder reads its saved checkpoint
(``python -m phoenx.run_info``, also in a child process so the GUI never
imports torch) to show where training stopped.

Relative paths in configs (``save_dir: ./Trained_Models/...``) resolve against
the **working folder**, the same as running the command from that folder in a
terminal.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Callable

import yaml

from phoenx.config_form import file_kind
from phoenx.process_runner import ProcessRunner, parse_progress, python_command

if TYPE_CHECKING:  # pragma: no cover
    from phoenx.config_gui import PhoenxApp

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
RENDER_MODES = ("human", "rgb_array", "headless")
MAX_LOG_LINES = 5000
REPLAY_BUFFERS = ("ReplayBuffer", "PrioritizedReplayBuffer")
ERROR_WORDS = ("Traceback", "Error", "ERROR", "Exception", "error:")

MODES = {
    "train": ("Train", "Train a new agent from a YAML config."),
    "resume": ("Resume", "Continue training a saved agent folder, optionally past its old limit."),
    "test": ("Test", "Run a saved agent without training it."),
    "sweep": ("Sweep", "Run a Ray Tune hyperparameter sweep."),
}


def _join(argv: list[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(argv)
    import shlex
    return shlex.join(argv)


def build_command(mode: str, values: dict[str, str]) -> list[str]:
    """Turn the Run tab's field values into an argv list.

    Raises ``ValueError`` naming the first missing required field. Empty
    optional fields are left out so the command's own defaults apply.

    Example:
        >>> build_command("resume", {"agent_dir": "runs/a", "stop_units": "3000000",
        ...                          "log_level": ""})[-4:]
        ['--agent_dir', 'runs/a', '--stop_units', '3000000']
    """
    v = {k: (val or "").strip() for k, val in values.items()}

    def need(key: str, label: str) -> str:
        if not v.get(key):
            raise ValueError(f"{label} is required")
        return v[key]

    args: list[str] = []

    def opt(flag: str, key: str) -> None:
        if v.get(key):
            args.extend([flag, v[key]])

    if mode == "train":
        args += ["--config", need("config", "Config file")]
        opt("--stop_units", "stop_units")
        opt("--log_level", "log_level")
        return python_command("phoenx.cli.train", args)
    if mode == "resume":
        args += ["--agent_dir", need("agent_dir", "Agent folder")]
        opt("--stop_units", "stop_units")
        opt("--log_level", "log_level")
        return python_command("phoenx.cli.train", args)
    if mode == "test":
        args += ["--agent_dir", need("agent_dir", "Agent folder")]
        for flag, key in (("--num_episodes", "num_episodes"), ("--num_envs", "num_envs"),
                          ("--render_mode", "render_mode"), ("--env", "env"),
                          ("--seed", "seed"), ("--log_level", "log_level")):
            opt(flag, key)
        return python_command("phoenx.cli.test", args)
    if mode == "sweep":
        args += ["--config", need("sweep", "Sweep file")]
        for flag, key in (("--base-config", "base_config"), ("--num-samples", "num_samples"),
                          ("--max-concurrent", "max_concurrent"), ("--name", "name"),
                          ("--storage-path", "storage_path"), ("--from-phase", "from_phase"),
                          ("--log-level", "log_level")):
            opt(flag, key)
        if v.get("resume") == "1":
            args.append("--resume")
        if v.get("validate_only") == "1":
            args.append("--validate-only")
        return python_command("phoenx.cli.tune", args)
    raise ValueError(f"unknown mode {mode!r}")


def describe_run(info: dict) -> str:
    """One-paragraph summary of ``read_run_info`` output for the Resume / Test forms."""
    if info.get("error"):
        return info["error"]
    what = f"{info.get('agent_type') or '?'} on {info.get('env') or '?'}"
    unit = info.get("stop_unit") or "timestep"
    limit = info.get("stop_units")
    if info.get("progress") is None:
        return f"{what}. No trainer_state.pt, so the step reached is unknown (limit {limit:,} {unit}s)." \
            if isinstance(limit, int) else f"{what}. No trainer_state.pt."
    text = f"{what}. Best checkpoint at {unit} {info['progress']:,}"
    if isinstance(limit, int):
        text += f" of {limit:,}"
    if info.get("best_reward") is not None:
        text += f", best avg reward {info['best_reward']:.2f}"
    text += "."
    if isinstance(limit, int) and info["progress"] >= limit:
        text += " It already reached its limit: set 'Train until' higher to train further."
    if info.get("buffer_type") in REPLAY_BUFFERS and not info.get("has_buffer"):
        text += " No replay buffer was saved, so resuming refills it from scratch."
    return text


class RunPanel:
    """Widgets and process control for the Run tab."""

    def __init__(self, app: "PhoenxApp", parent: ttk.Frame):
        """Build the tab inside ``parent``."""
        self.app = app
        self.parent = parent
        self.master = parent.winfo_toplevel()
        self.runner = ProcessRunner()
        self.mode = tk.StringVar(value="train")
        self.vars: dict[str, tk.StringVar] = {}
        self.run_infos: dict[str, dict] = {}
        self._info_queue: queue.Queue = queue.Queue()
        self._last_line_was_progress = False
        self._current_mode: str | None = None
        self._output_dir: Path | None = None
        self._build()
        self._tick()

    # -- layout ----------------------------------------------------------------

    def _var(self, key: str, value: str = "") -> tk.StringVar:
        var = self.vars.setdefault(key, tk.StringVar(value=value))
        var.trace_add("write", lambda *_: self._refresh_command())
        return var

    def _build(self) -> None:
        top = ttk.Frame(self.parent, padding=(10, 8))
        top.pack(fill="x")
        for key, (label, tip) in MODES.items():
            b = ttk.Radiobutton(top, text=label, value=key, variable=self.mode,
                                command=self._show_mode)
            b.pack(side="left", padx=(0, 14))
        self.mode_hint = ttk.Label(top, foreground="#6b6b6b")
        self.mode_hint.pack(side="left", padx=10)

        self.forms = ttk.Frame(self.parent, padding=(10, 0))
        self.forms.pack(fill="x")
        self.form_frames = {
            "train": self._train_form(),
            "resume": self._resume_form(),
            "test": self._test_form(),
            "sweep": self._sweep_form(),
        }

        common = ttk.Frame(self.parent, padding=(10, 6))
        common.pack(fill="x")
        self._path_row(common, "Working folder", "cwd", directory=True, value=os.getcwd(),
                       tip="Folder the command runs in. Relative paths in configs (save_dir, "
                           "base_config) resolve against it, just as in a terminal.")
        row = ttk.Frame(common)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Log level", width=16).pack(side="left")
        ttk.Combobox(row, textvariable=self._var("log_level"), values=("",) + LOG_LEVELS,
                     state="readonly", width=10).pack(side="left")
        ttk.Label(row, text="(empty = the config's own log_level)", foreground="#6b6b6b").pack(
            side="left", padx=8)

        cmd = ttk.Frame(self.parent, padding=(10, 4))
        cmd.pack(fill="x")
        ttk.Label(cmd, text="Command", width=16).pack(side="left")
        self.command_var = tk.StringVar()
        entry = ttk.Entry(cmd, textvariable=self.command_var, state="readonly")
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(cmd, text="Copy", command=self._copy_command).pack(side="left", padx=4)

        ctl = ttk.Frame(self.parent, padding=(10, 4))
        ctl.pack(fill="x")
        self.start_btn = ttk.Button(ctl, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.open_btn = ttk.Button(ctl, text="Open output folder", command=self._open_output,
                                   state="disabled")
        self.open_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(ctl, length=260, mode="determinate", maximum=1.0)
        self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(ctl, text="Idle.")
        self.status.pack(side="left")

        logf = ttk.Frame(self.parent, padding=(10, 4, 10, 10))
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, wrap="none", font=("TkFixedFont", 9), state="disabled",
                           background="#111418", foreground="#d8dee9", insertbackground="white")
        ys = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(logf, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("err", foreground="#ff7b7b")
        self.log.tag_configure("good", foreground="#8fdc8f")
        self.log.tag_configure("meta", foreground="#8aa4c8")

        self._show_mode()

    def _path_row(self, parent, label: str, key: str, *, directory: bool = False,
                  value: str = "", tip: str = "", filetypes=None,
                  on_change: Callable[[], None] | None = None, extra=None) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        lab = ttk.Label(row, text=label, width=16)
        lab.pack(side="left")
        var = self._var(key, value)
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        def browse():
            start = var.get() or os.getcwd()
            start_dir = start if os.path.isdir(start) else os.path.dirname(start) or os.getcwd()
            if directory:
                chosen = filedialog.askdirectory(parent=self.master, initialdir=start_dir)
            else:
                chosen = filedialog.askopenfilename(
                    parent=self.master, initialdir=start_dir,
                    filetypes=filetypes or [("YAML", "*.yml *.yaml"), ("All files", "*.*")])
            if chosen:
                var.set(chosen)
        ttk.Button(row, text="Browse...", command=browse).pack(side="left", padx=4)
        if extra:
            extra(row)
        if on_change:
            var.trace_add("write", lambda *_: on_change())
        if tip:
            from phoenx.config_gui import Tooltip
            Tooltip(lab, tip)
        return row

    def _field(self, parent, label: str, key: str, value: str = "", width: int = 12,
               choices=None, hint: str = "") -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        var = self._var(key, value)
        if choices:
            ttk.Combobox(row, textvariable=var, values=choices, width=width).pack(side="left")
        else:
            ttk.Entry(row, textvariable=var, width=width).pack(side="left")
        if hint:
            ttk.Label(row, text=hint, foreground="#6b6b6b").pack(side="left", padx=8)

    def _check(self, parent, label: str, key: str, hint: str = "") -> None:
        var = self._var(key, "0")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="", width=16).pack(side="left")
        ttk.Checkbutton(row, text=label, variable=var, onvalue="1", offvalue="0").pack(side="left")
        if hint:
            ttk.Label(row, text=hint, foreground="#6b6b6b").pack(side="left", padx=8)

    def _train_form(self) -> ttk.Frame:
        f = ttk.Frame(self.forms)
        self._path_row(f, "Config file", "config",
                       extra=lambda row: ttk.Button(row, text="Use editor's config",
                                                    command=self._use_editor_config).pack(side="left"))
        self._field(f, "Train until", "stop_units",
                    hint="optional: overrides schedule.stop_units (in the config's stop_unit)")
        return f

    def _resume_form(self) -> ttk.Frame:
        f = ttk.Frame(self.forms)
        self._path_row(f, "Agent folder", "agent_dir", directory=True,
                       on_change=lambda: self._inspect("agent_dir", self.resume_info),
                       tip="The run's save_dir: the folder holding config.json, agent/ and "
                           "trainer_state.pt.")
        self.resume_info = ttk.Label(f, foreground="#3b5b8a", wraplength=900, justify="left")
        self.resume_info.pack(fill="x", padx=(120, 0), pady=2)
        self._field(f, "Train until", "stop_units",
                    hint="total, in the run's stop_unit; must be past the checkpoint")
        return f

    def _test_form(self) -> ttk.Frame:
        f = ttk.Frame(self.forms)
        self._path_row(f, "Agent folder", "test_agent_dir", directory=True,
                       on_change=lambda: self._inspect("test_agent_dir", self.test_info))
        self.test_info = ttk.Label(f, foreground="#3b5b8a", wraplength=900, justify="left")
        self.test_info.pack(fill="x", padx=(120, 0), pady=2)
        self._field(f, "Episodes", "num_episodes", "10", width=8)
        self._field(f, "Parallel envs", "num_envs", "1", width=8)
        self._field(f, "Render mode", "render_mode", "human", choices=RENDER_MODES,
                    hint="human opens a window (Isaac: anything but headless opens the GUI)")
        self._field(f, "Env override", "env", width=40,
                    hint="optional: env id / Isaac cfg to evaluate on instead")
        self._field(f, "Seed", "seed", width=8, hint="optional")
        return f

    def _sweep_form(self) -> ttk.Frame:
        f = ttk.Frame(self.forms)
        self._path_row(f, "Sweep file", "sweep", on_change=self._load_phases,
                       extra=lambda row: ttk.Button(row, text="Use editor's sweep",
                                                    command=self._use_editor_sweep).pack(side="left"))
        self._check(f, "Validate only", "validate_only",
                    hint="sample a few trials per phase and print them; no Ray, no training")
        self._field(f, "Samples", "num_samples", width=8, hint="optional: overrides every phase")
        self._field(f, "Concurrent trials", "max_concurrent", width=8, hint="optional")
        self._field(f, "Sweep name", "name", width=24, hint="optional: results folder name")
        self._path_row(f, "Results folder", "storage_path", directory=True,
                       tip="Ray Tune storage root (default ./ray_results).")
        self.phase_box_row = ttk.Frame(f)
        self.phase_box_row.pack(fill="x", pady=2)
        ttk.Label(self.phase_box_row, text="Start at phase", width=16).pack(side="left")
        self.phase_box = ttk.Combobox(self.phase_box_row, textvariable=self._var("from_phase"),
                                      width=24)
        self.phase_box.pack(side="left")
        ttk.Label(self.phase_box_row, text="optional: skip earlier phases (they must have "
                  "finished)", foreground="#6b6b6b").pack(side="left", padx=8)
        self._check(f, "Resume interrupted phase", "resume")
        return f

    def _show_mode(self) -> None:
        for key, frame in self.form_frames.items():
            frame.pack_forget()
        self.form_frames[self.mode.get()].pack(fill="x")
        self.mode_hint.configure(text=MODES[self.mode.get()][1])
        self._refresh_command()

    # -- values / command ------------------------------------------------------

    def values(self) -> dict[str, str]:
        """Current field values for the selected mode, keyed as ``build_command`` expects."""
        v = {k: var.get() for k, var in self.vars.items()}
        if self.mode.get() == "test":
            v["agent_dir"] = v.get("test_agent_dir", "")
        return v

    def command(self) -> list[str]:
        """The argv the Start button would run (raises ValueError if incomplete)."""
        return build_command(self.mode.get(), self.values())

    def _refresh_command(self) -> None:
        if not hasattr(self, "command_var"):
            return
        try:
            self.command_var.set(_join(self.command()))
        except ValueError as exc:
            self.command_var.set(f"({exc})")

    def _copy_command(self) -> None:
        self.master.clipboard_clear()
        self.master.clipboard_append(self.command_var.get())

    def _use_editor_config(self) -> None:
        path = self.app.editor_path(save_first=True)
        if path:
            self.vars["config"].set(str(path))

    def _use_editor_sweep(self) -> None:
        path = self.app.sweep_path(save_first=True) if hasattr(self.app, "sweep_path") else None
        if path:
            self.vars["sweep"].set(str(path))

    def _load_phases(self) -> None:
        path = Path(self.vars["sweep"].get())
        names: list[str] = []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
            names = [p.get("name") for p in (data or {}).get("phases", []) if isinstance(p, dict)]
        except (OSError, yaml.YAMLError, AttributeError):
            pass
        self.phase_box.configure(values=[""] + [n for n in names if n])

    # -- agent folder inspection -------------------------------------------

    def _inspect(self, key: str, label: ttk.Label) -> None:
        path = self.vars[key].get().strip()
        if not path or not (Path(path) / "config.json").is_file():
            label.configure(text="(choose a run's save_dir: it must contain config.json)"
                            if path else "")
            return
        label.configure(text="Reading checkpoint...")

        def work():
            try:
                out = subprocess.run(python_command("phoenx.run_info", [path]),
                                     capture_output=True, text=True, timeout=120)
                line = (out.stdout.strip().splitlines() or ["{}"])[-1]
                info = json.loads(line)
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
                info = {"error": f"Could not read the run: {exc}"}
            self._info_queue.put((key, label, path, info))

        threading.Thread(target=work, daemon=True).start()

    def _apply_info(self, key: str, label: ttk.Label, path: str, info: dict) -> None:
        if self.vars[key].get().strip() != path:
            return  # the user picked another folder meanwhile
        self.run_infos[path] = info
        label.configure(text=describe_run(info))
        if key == "agent_dir" and not self.vars["stop_units"].get().strip():
            limit, done = info.get("stop_units"), info.get("progress")
            if isinstance(limit, int) and isinstance(done, (int, float)) and done >= limit:
                self.vars["stop_units"].set(str(int(done) + limit))
            elif isinstance(limit, int):
                self.vars["stop_units"].set(str(limit))

    # -- run control -----------------------------------------------------------

    def start(self) -> None:
        """Start the selected command (after checking inputs)."""
        if self.runner.running:
            return
        try:
            argv = self.command()
        except ValueError as exc:
            messagebox.showerror("Can't start", str(exc), parent=self.master)
            return
        cwd = self.vars["cwd"].get().strip() or os.getcwd()
        if not os.path.isdir(cwd):
            messagebox.showerror("Can't start", f"Working folder does not exist:\n{cwd}",
                                 parent=self.master)
            return
        problem = self._precheck(cwd)
        if problem:
            messagebox.showerror("Can't start", problem, parent=self.master)
            return
        self._clear_log()
        self._append(f"$ {_join(argv)}", "meta")
        self._append(f"  (in {cwd})", "meta")
        try:
            self.runner.start(argv, cwd=cwd)
        except OSError as exc:
            self._append(f"Failed to start: {exc}", "err")
            return
        self._current_mode = self.mode.get()
        self._output_dir = self._guess_output_dir(cwd)
        self.progress.configure(value=0, mode="indeterminate")
        self.progress.start(15)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.open_btn.configure(state="normal" if self._output_dir else "disabled")
        self.status.configure(text="Running...")

    def _precheck(self, cwd: str) -> str | None:
        mode, v = self.mode.get(), self.values()

        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else Path(cwd) / path

        expected = {"train": ("config", "Config file", "a training config"),
                    "sweep": ("sweep", "Sweep file", "a sweep")}.get(mode)
        if expected:
            key = "config" if mode == "train" else "sweep"
            path = resolve(v[key])
            if not path.is_file():
                from phoenx.config_gui import bundled_examples
                bundled = bundled_examples("configs" if mode == "train" else "sweeps")
                if v[key].strip().replace("\\", "/") in bundled:
                    return None  # a bundled example name; the command resolves it
                return f"{expected[1]} not found: {path}"
            kind = file_kind(path)
            if kind is not None and kind != expected[0]:
                actual = "a sweep" if kind == "sweep" else "a training config"
                return (f"{path.name} is {actual}, not {expected[2]}. "
                        + ("Sweep files have base_config and phases."
                           if mode == "sweep" else "Use Sweep mode to run a sweep."))
        if mode in ("resume", "test"):
            d = resolve(v["agent_dir"])
            if not (d / "config.json").is_file():
                return f"{d} is not a saved run (no config.json)."
        if mode == "resume" and v.get("stop_units", "").strip():
            info = self.run_infos.get(v["agent_dir"].strip())
            try:
                target = int(v["stop_units"])
            except ValueError:
                return "Train until must be a whole number."
            if info and isinstance(info.get("progress"), (int, float)) and target <= info["progress"]:
                return (f"Train until ({target:,}) must be past the checkpoint "
                        f"({int(info['progress']):,}), or training stops immediately.")
        return None

    def _guess_output_dir(self, cwd: str) -> Path | None:
        """Where the run writes: config save_dir (train), the agent folder, or results root."""
        mode, v = self.mode.get(), self.values()
        base = Path(cwd)
        try:
            if mode == "train":
                cfg_path = Path(v["config"])
                cfg_path = cfg_path if cfg_path.is_absolute() else base / cfg_path
                save_dir = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("save_dir")
                return (base / save_dir) if save_dir else None
            if mode in ("resume", "test"):
                p = Path(v["agent_dir"])
                return p if p.is_absolute() else base / p
            if mode == "sweep":
                root = v.get("storage_path") or "ray_results"
                p = Path(root)
                return p if p.is_absolute() else base / p
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            return None
        return None

    def stop(self) -> None:
        """Interrupt the run (it saves a pending best checkpoint); kill if it hangs."""
        if not self.runner.running:
            return
        self.runner.stop()
        self._append(f"[gui] stop requested: interrupting, killing after "
                     f"{int(self.runner.grace_seconds)} s if it has not exited", "meta")
        self.status.configure(text="Stopping...")
        self.stop_btn.configure(text="Kill", command=self.kill)

    def kill(self) -> None:
        """Kill the process tree immediately."""
        self.runner.kill()

    def _open_output(self) -> None:
        if not self._output_dir:
            return
        target = self._output_dir if self._output_dir.exists() else self._output_dir.parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            messagebox.showerror("Can't open folder", str(exc), parent=self.master)

    # -- log -------------------------------------------------------------------

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._last_line_was_progress = False

    def _append(self, line: str, tag: str | None = None, replace_last: bool = False) -> None:
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state="normal")
        if replace_last:
            self.log.delete("end-2l linestart", "end-1c")
        self.log.insert("end", line + "\n", (tag,) if tag else ())
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.log.delete("1.0", f"{lines - MAX_LOG_LINES}.0")
        self.log.configure(state="disabled")
        if at_bottom:
            self.log.see("end")

    def _handle_line(self, line: str) -> None:
        prog = parse_progress(line)
        if prog is not None:
            # Progress goes to the bar and status line, not the log, except a
            # finished evaluation's line, which is the result being asked for.
            if prog.mode == "test" and prog.total and prog.done >= prog.total:
                self._append(line, "good")
                self._last_line_was_progress = False
            if prog.fraction is not None:
                self.progress.stop()
                self.progress.configure(mode="determinate", value=prog.fraction)
            reward = f", avg reward {prog.avg_reward:.2f}" if prog.avg_reward is not None else ""
            total = f"/{prog.total:,}" if prog.total else ""
            self.status.configure(text=f"{prog.unit} {prog.done:,}{total}{reward}")
            return
        if line.startswith("[checkpoint]"):
            # Consecutive saves update one line instead of flooding the log.
            self._append(line, "good", replace_last=self._last_line_was_progress)
            self._last_line_was_progress = True
            return
        self._last_line_was_progress = False
        self._append(line, "err" if any(w in line for w in ERROR_WORDS) else None)

    def _tick(self) -> None:
        """Pump child output and finished-process state into the widgets (every 100 ms)."""
        try:
            for line in self.runner.poll():
                self._handle_line(line)
            while True:
                try:
                    self._apply_info(*self._info_queue.get_nowait())
                except queue.Empty:
                    break
            if self._current_mode is not None and not self.runner.running \
                    and self.runner.returncode is not None:
                for line in self.runner.poll():
                    self._handle_line(line)
                self._finished(self.runner.returncode)
        except tk.TclError:
            return  # window closed
        self.master.after(100, self._tick)

    def _finished(self, code: int) -> None:
        mode, self._current_mode = self._current_mode, None
        self.progress.stop()
        elapsed = timedelta(seconds=int(time.time() - (self.runner.started_at or time.time())))
        if code == 0:
            self.progress.configure(mode="determinate", value=1.0)
            text = f"Finished in {elapsed}."
            self._append(f"[gui] finished (exit 0) after {elapsed}", "good")
        elif code in (130, -2, 3221225786) or self.stop_btn.cget("text") == "Kill":
            text = f"Stopped after {elapsed}."
            self._append(f"[gui] stopped (exit {code}) after {elapsed}", "meta")
        else:
            text = f"Failed (exit {code}) after {elapsed}. See the log."
            self._append(f"[gui] failed with exit code {code}", "err")
        self.status.configure(text=text)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(text="Stop", command=self.stop, state="disabled")
        if code == 0 and mode == "train" and self._output_dir is not None:
            # Make the next step one click: test what was just trained.
            self.vars["test_agent_dir"].set(str(self._output_dir))
            self.vars["agent_dir"].set(str(self._output_dir))

    def confirm_close(self) -> bool:
        """Ask before closing the window while a run is active; stop it if confirmed."""
        if not self.runner.running:
            return True
        if not messagebox.askyesno(
                "A run is in progress",
                "Closing the window stops the running command (a pending best checkpoint "
                "is saved first). Stop it and close?", parent=self.master):
            return False
        self.runner.stop()
        self.runner.wait(timeout=self.runner.grace_seconds)
        self.runner.kill()
        return True
