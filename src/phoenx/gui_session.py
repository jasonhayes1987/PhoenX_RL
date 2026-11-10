"""Session the desktop window talks to.

The window sends one named method at a time through ``call`` and gets back
either a snapshot or a question (a *need*). Training runs in a child
process. Importing this module does not import torch or tkinter.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import re
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TextIO

import yaml

from phoenx import __version__
from phoenx.config_form import (
    HEADER,
    UNSET,
    ListNode,
    MapNode,
    ObjectNode,
    OptionalNode,
    ScalarNode,
    TaggedNode,
    _fmt,
    build_form,
    dump_config,
    file_kind,
)
from phoenx.config_schema import build_schema, validate_config
from phoenx.config_utils import resolve_n_step
from phoenx.example_files import bundled_examples
from phoenx.launch_cmd import build_command, command_text, describe_run
from phoenx.process_runner import ProcessRunner, parse_progress
from phoenx.sweep_schema import (
    build_sweep_schema,
    suggest_spec,
    to_phases_form,
    validate_sweep,
)

SWEEP_HEADER = (
    "# PhoenX sweep config, written by `phoenx-config gui`.\n"
    "# Run with: phoenx-tune --config <this file>\n"
)
BLANK_CONFIG = (
    "New config. Pick an agent type and add model branches, or start from an "
    "example (New menu)."
)
BLANK_SWEEP = (
    "New sweep. Set base_config and add a phase, then add fields from the Base "
    "config pane (double-click), or right-click a field in the Config tab."
)
COMMENT_NOTE = (
    " Note: comments and YAML anchors (&name / *name) are not kept when "
    "saving; anchored values are written out in full."
)
SHORTHAND_NOTE = (
    "Its single-phase shorthand is shown as one phase named phase_0 "
    "(it runs the same)."
)
_MAP_NAMES = ("policy", "value", "critic", "critic_b", "actor")
_ERROR_BITS = ("Traceback", "Error", "ERROR", "Exception", "error:")
_LAUNCH_KEYS = (
    "config",
    "stop_units",
    "agent_dir",
    "test_agent_dir",
    "num_episodes",
    "num_envs",
    "render_mode",
    "env",
    "seed",
    "sweep",
    "validate_only",
    "num_samples",
    "max_concurrent",
    "name",
    "storage_path",
    "from_phase",
    "resume",
    "cwd",
    "log_level",
)


class GuiError(Exception):
    """A refusal the window shows as a dialog, not a question.

    Args:
        title: Short heading, such as ``Can't edit``.
        message: The text under that heading.
    """

    def __init__(self, title: str, message: str) -> None:
        """Store the heading and the message.

        Args:
            title: Short heading the window shows.
            message: Text under that heading.
        """
        super().__init__(message)
        self.title = title
        self.message = message


@dataclass
class _Need:
    """One open question and the function that continues after the answer."""

    method: str
    need_id: str
    resume: Callable[[str], Any]


class _Doc:
    """One config or sweep document."""

    def __init__(self, kind: str, schema: dict, header: str, blank: str) -> None:
        self.kind = kind
        self.noun = kind
        self.schema = schema
        self.header = header
        self.blank = blank
        self.form = build_form(schema)
        self.path: Path | None = None
        self.dirty = False
        self.status = blank
        self.only_set = False
        self.with_defaults = False
        self.selected_phase = ""
        #: Bumped whenever the form is replaced, so the window can reset
        #: which sections are expanded (as the Tkinter editor did on load).
        self.generation = 0
        #: Path the window should expand and scroll to (one snapshot only).
        self.reveal = ""
        #: Paths of "other keys" boxes whose text is not a YAML mapping.
        self.extra_errors: set[str] = set()

    def rebuild(self) -> None:
        """Replace the form with an empty one. Flags stay as the user set them."""
        self.generation += 1
        self.extra_errors = set()
        self.form = build_form(self.schema)
        self.path = None
        self.dirty = False
        self.status = self.blank
        self.selected_phase = ""


class GuiSession:
    """Hold the config, the sweep, and the launch process.

    ``call`` is the only way to run a method. A method that needs a decision
    returns a need and waits. The next call must be that same method with
    an ``answer``, or ``editor.save_as`` while the open need is ``save-as``.
    """

    def __init__(self, grace_seconds: float = 15.0) -> None:
        """Start with a blank config, a blank sweep, and no running command.

        Args:
            grace_seconds: How long a stop waits before the process is killed.
        """
        self.grace_seconds = grace_seconds
        config_schema = build_schema()
        self.config = _Doc("config", config_schema, HEADER, BLANK_CONFIG)
        self.sweep = _Doc("sweep", build_sweep_schema(), SWEEP_HEADER, BLANK_SWEEP)
        self.initial_page = "config"
        self.runner = ProcessRunner(grace_seconds)
        self.launch = {key: "" for key in _LAUNCH_KEYS}
        self.launch.update(
            {
                "mode": "train",
                "cwd": os.getcwd(),
                "num_episodes": "10",
                "num_envs": "1",
                "render_mode": "human",
                "validate_only": "0",
                "resume": "0",
            }
        )
        self.run_status = "Idle"
        self.launch_status = "Idle."
        self.progress: dict | None = None
        self.log: list[dict[str, str]] = []
        self._last_checkpoint = False
        self._output_dir: str | None = None
        self._stop_asked = False
        self._finished_code: int | None = None
        self._phase_choices = [""]
        self._infos: dict[str, dict] = {}
        self._info_jobs: list[dict] = []
        self._pending: _Need | None = None
        self._caller = ""
        self._after_save: Callable[[], Any] | None = None
        self._methods = {
            "app.version": self._app_version,
            "app.snapshot": lambda _p: self._app_snapshot(),
            "app.close": self._app_close,
            "editor.snapshot": self._editor_snapshot_call,
            "editor.examples": self._examples,
            "editor.new_blank": self._new_blank,
            "editor.new_from_example": self._new_from_example,
            "editor.open": self._open,
            "editor.set_flags": self._set_flags,
            "editor.set_text": self._set_text,
            "editor.set_extra": self._set_extra,
            "editor.set_type": self._set_type,
            "editor.set_enabled": self._set_enabled,
            "editor.list_add": self._list_add,
            "editor.list_move": self._list_move,
            "editor.list_remove": self._list_remove,
            "editor.map_add": self._map_add,
            "editor.map_remove": self._map_remove,
            "editor.save": self._save,
            "editor.save_as": self._save_as,
            "sweep.select_phase": self._select_phase,
            "sweep.add_field": self._add_field,
            "sweep.add_from_config": self._add_from_config,
            "sweep.handoff": self._handoff,
            "launch.snapshot": lambda _p: self._launch_snapshot(),
            "launch.set_mode": self._set_mode,
            "launch.set": self._launch_set,
            "launch.use_editor": self._use_editor,
            "launch.start": lambda _p: self._start(),
            "launch.stop": lambda _p: self._stop(),
            "launch.kill": lambda _p: self._kill(),
            "launch.poll": lambda _p: self._poll(),
        }

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Run one method.

        Args:
            method: Dotted name, such as ``editor.new_blank``.
            params: Arguments for that method. ``answer`` replies to the
                open question.

        Returns:
            A snapshot, a need, or ``{"closed": true}``.

        Raises:
            GuiError: The method cannot run. ``title`` and ``message`` are
                the dialog text.
        """
        params = dict(params or {})
        answer = params.pop("answer", None)
        if self._pending is not None:
            same = method == self._pending.method and answer is not None
            save_as = self._pending.need_id == "save-as" and method == "editor.save_as"
            if not same and not save_as:
                raise GuiError("Can't answer", "Answer the open question first.")
            pending = self._pending
            self._pending = None
            self._caller = method
            if save_as:
                return self._save_as(params)
            return pending.resume(str(answer))
        if answer is not None:
            raise GuiError("Can't answer", "No question is open.")
        self._caller = method
        handler = self._methods.get(method)
        if handler is None:
            raise GuiError("Can't complete", f"Unknown method {method}.")
        return handler(params)

    # -- questions -------------------------------------------------------------

    def _ask(
        self,
        kind: str,
        need_id: str,
        title: str,
        message: str,
        resume: Callable[[str], Any],
    ) -> dict:
        self._pending = _Need(self._caller, need_id, resume)
        return {"need": kind, "id": need_id, "title": title, "message": message}

    def _doc(self, params: dict) -> _Doc:
        target = params.get("target", "config")
        if target == "config":
            return self.config
        if target == "sweep":
            return self.sweep
        raise GuiError("Can't edit", f"Unknown document {target}.")

    # -- app -------------------------------------------------------------------

    def _app_version(self, _params: dict) -> dict:
        return {"version": __version__}

    def _app_snapshot(self) -> dict:
        def one(doc: _Doc) -> dict:
            return {
                "path": None if doc.path is None else str(doc.path),
                "dirty": doc.dirty,
                "noun": doc.noun,
            }

        return {
            "version": __version__,
            "initial_page": self.initial_page,
            "run_status": self.run_status,
            "config": one(self.config),
            "sweep": one(self.sweep),
        }

    def _app_close(self, _params: dict) -> Any:
        if self.runner.running:
            return self._ask(
                "yes_no",
                "close-run",
                "A run is in progress",
                "Closing the window stops the running command (a pending best "
                "checkpoint is saved first). Stop it and close?",
                self._close_run_answer,
            )
        return self._close_config()

    def _close_run_answer(self, answer: str) -> Any:
        if answer == "no":
            return self._app_snapshot()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        self.runner.stop()
        self._stop_asked = True
        self.runner.wait(timeout=self.grace_seconds)
        self.runner.kill()
        self._drain()
        if self.runner.returncode is not None:
            self._finish(self.runner.returncode)
        return self._close_config()

    def _close_config(self) -> Any:
        if self.config.dirty:
            return self._discard(self.config, self._close_config_answer)
        return self._close_sweep()

    def _close_config_answer(self, answer: str) -> Any:
        if answer == "cancel":
            return self._app_snapshot()
        if answer == "no":
            return self._close_sweep()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes, no, or cancel.")
        return self._save_doc(self.config, then=self._close_sweep, chain=True)

    def _close_sweep(self) -> Any:
        if self.sweep.dirty:
            return self._discard(self.sweep, self._close_sweep_answer)
        return {"closed": True}

    def _close_sweep_answer(self, answer: str) -> Any:
        if answer == "cancel":
            return self._app_snapshot()
        if answer == "no":
            return {"closed": True}
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes, no, or cancel.")
        return self._save_doc(self.sweep, then=lambda: {"closed": True}, chain=True)

    # -- editor snapshots ------------------------------------------------------

    def _editor_snapshot_call(self, params: dict) -> dict:
        return self._editor_snapshot(self._doc(params))

    def _editor_snapshot(self, doc: _Doc) -> dict:
        name = doc.path.name if doc.path is not None else f"untitled {doc.noun}"
        title = f"{name} - PhoenX"
        if doc.dirty:
            title = f"*{title}"
        problems = self._problems(doc)
        value = self._value(doc)
        phases = _phase_names(doc.form) if doc.kind == "sweep" else []
        base_fields, base_note = ([], "")
        if doc.kind == "sweep":
            base_fields, base_note = self._base_fields(value)
        return {
            "noun": doc.noun,
            "path": None if doc.path is None else str(doc.path),
            "dirty": doc.dirty,
            "generation": doc.generation,
            "extra_errors": sorted(doc.extra_errors),
            "reveal": doc.reveal,
            "title": title,
            "status": doc.status,
            "only_set": doc.only_set,
            "with_defaults": doc.with_defaults,
            "problems": problems,
            "problem_summary": (
                f"Problems ({len(problems)})"
                if problems
                else "Problems: none, ready to save"
            ),
            "yaml": dump_config(value, header=doc.header),
            "form": _snap(doc.form, ""),
            "base_note": base_note,
            "base_fields": base_fields,
            "phases": phases,
            "selected_phase": doc.selected_phase if doc.kind == "sweep" else "",
            "cards": _cards(doc.form) if doc.kind == "config" else None,
        }

    def _value(self, doc: _Doc) -> dict:
        value = doc.form.value(include_defaults=doc.with_defaults)
        if value is UNSET or value is None:
            return {}
        return value

    def _problems(self, doc: _Doc) -> list[str]:
        problems = list(doc.form.problems())
        seen = {item.split(":", 1)[0] for item in problems}
        value = self._value(doc)
        if doc.kind == "config":
            extra = list(validate_config(value))
            try:
                resolve_n_step(copy.deepcopy(value))
            except ValueError as exc:
                extra.append(f"agent.config.N: {exc}")
        else:
            base_path, note = self._resolve_base(str(value.get("base_config") or ""))
            # validate_sweep needs the base config's contents, not its path.
            base = self._load_base(base_path)
            extra = list(validate_sweep(value, base))
            if base_path is None and value.get("base_config"):
                extra.append(f"base_config: {note}")
        for err in extra:
            key = err.split(":", 1)[0]
            if key not in seen:
                problems.append(err)
                seen.add(key)
        return problems

    def _examples(self, params: dict) -> dict:
        doc = self._doc(params)
        folder = "configs" if doc.kind == "config" else "sweeps"
        return {"names": list(bundled_examples(folder))}

    def _set_flags(self, params: dict) -> dict:
        doc = self._doc(params)
        if "only_set" in params and params["only_set"] is not None:
            doc.only_set = bool(params["only_set"])
        if "with_defaults" in params and params["with_defaults"] is not None:
            doc.with_defaults = bool(params["with_defaults"])
            if doc.with_defaults:
                doc.dirty = True
        return self._editor_snapshot(doc)

    def _touch(self, doc: _Doc) -> None:
        doc.dirty = True

    # -- structure edits -------------------------------------------------------

    def _set_text(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _must_scalar(_index(doc.form), str(params.get("path", "")))
        node.text = "" if params.get("text") is None else str(params["text"])
        if node.kind == "bool":
            node.pinned = True
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _set_extra(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _index(doc.form).get(str(params.get("path", "")))
        node = _object(node)
        if node is None:
            raise GuiError("Can't edit", "No such field.")
        path = str(params.get("path", ""))
        try:
            node.set_extra_text(
                "" if params.get("text") is None else str(params["text"])
            )
        except (ValueError, yaml.YAMLError):
            doc.extra_errors.add(path)
            return self._editor_snapshot(doc)
        doc.extra_errors.discard(path)
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _set_type(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _index(doc.form).get(str(params.get("path", "")))
        if isinstance(node, OptionalNode):
            node = node.inner
        if not isinstance(node, TaggedNode):
            raise GuiError("Can't edit", "No such field.")
        node.set_type(str(params.get("type_name", "")))
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _set_enabled(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _index(doc.form).get(str(params.get("path", "")))
        if not isinstance(node, OptionalNode):
            raise GuiError("Can't edit", "No such field.")
        node.enabled = bool(params.get("enabled"))
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _list_add(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _as(doc, params, ListNode)
        node.add()
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _list_move(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _as(doc, params, ListNode)
        index = _index_of(params, len(node.items))
        delta = params.get("delta")
        if delta not in (-1, 1):
            raise GuiError("Can't edit", "No such field.")
        node.move(index, int(delta))
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _list_remove(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _as(doc, params, ListNode)
        node.remove(_index_of(params, len(node.items)))
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _map_add(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _as(doc, params, MapNode)
        key = str(params.get("key") or "")
        try:
            if not key.strip():
                if node.schema.get("x-path-keys"):
                    raise ValueError("type a dotted config path first")
                key = _next_map_name(node)
            node.add(key)
        except ValueError as exc:
            raise GuiError("Can't add", str(exc)) from None
        self._touch(doc)
        return self._editor_snapshot(doc)

    def _map_remove(self, params: dict) -> dict:
        doc = self._doc(params)
        node = _as(doc, params, MapNode)
        key = str(params.get("key") or "")
        if key not in node.entries:
            raise GuiError("Can't edit", "No such field.")
        node.remove(key)
        self._touch(doc)
        return self._editor_snapshot(doc)

    # -- new / open / save -----------------------------------------------------

    def _new_blank(self, params: dict) -> Any:
        doc = self._doc(params)
        if doc.dirty:
            return self._discard(
                doc,
                lambda answer: self._after_discard(
                    doc, answer, lambda: self._do_blank(doc)
                ),
            )
        return self._do_blank(doc)

    def _do_blank(self, doc: _Doc) -> dict:
        doc.rebuild()
        return self._editor_snapshot(doc)

    def _new_from_example(self, params: dict) -> Any:
        doc = self._doc(params)
        name = str(params.get("name", ""))
        folder = "configs" if doc.kind == "config" else "sweeps"
        text = bundled_examples(folder).get(name.replace("\\", "/"))
        if text is None:
            raise GuiError("Can't open", f"{name} is not a bundled example.")

        def go() -> dict:
            return self._load_text(
                doc, text, None, f"New {doc.noun} from example {name}.", dirty=True
            )

        if doc.dirty:
            return self._discard(
                doc, lambda answer: self._after_discard(doc, answer, go)
            )
        return go()

    def _open(self, params: dict) -> Any:
        doc = self._doc(params)
        path = Path(str(params.get("path", "")))
        if doc.dirty:
            return self._discard(
                doc,
                lambda answer: self._after_discard(
                    doc, answer, lambda: self._open_path(doc, path)
                ),
            )
        return self._open_path(doc, path)

    def _after_discard(
        self, doc: _Doc, answer: str, continue_with: Callable[[], Any]
    ) -> Any:
        if answer == "cancel":
            return self._editor_snapshot(doc)
        if answer == "no":
            doc.dirty = False
            return continue_with()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes, no, or cancel.")
        return self._save_doc(doc, then=continue_with, chain=False)

    def _discard(self, doc: _Doc, resume: Callable[[str], Any]) -> dict:
        return self._ask(
            "yes_no_cancel",
            "discard",
            "Unsaved changes",
            f"Save changes to the current {doc.noun} first?",
            resume,
        )

    def _open_path(self, doc: _Doc, path: Path) -> Any:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise GuiError("Can't open", f"{path}\n\n{exc}") from None
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise GuiError("Can't open", f"{path}\n\n{exc}") from None
        if not isinstance(loaded, dict):
            raise GuiError("Can't open", f"{path} is not a YAML mapping.")
        kind = file_kind(path)
        if kind is not None and kind != doc.kind:
            return self._wrong_kind(doc, path, kind, loaded, text)
        return self._accept_open(doc, path, loaded, text)

    def _accept_open(
        self, doc: _Doc, path: Path | None, loaded: dict, text: str
    ) -> dict:
        note = ""
        if doc.kind == "sweep":
            loaded, changed = to_phases_form(loaded)
            if changed:
                note = SHORTHAND_NOTE
        if path is None:
            status = f"New {doc.noun} from example."
        else:
            status = f"Opened {path}. {note}".strip()
            if "&" in text or any(
                line.lstrip().startswith("#") for line in text.splitlines()
            ):
                status += COMMENT_NOTE
        self._replace(doc, loaded, path, status, dirty=path is None)
        return self._editor_snapshot(doc)

    def _load_text(
        self, doc: _Doc, text: str, path: Path | None, status: str, dirty: bool
    ) -> dict:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise GuiError("Can't open", "Example is not a YAML mapping.")
        note = ""
        if doc.kind == "sweep":
            loaded, changed = to_phases_form(loaded)
            if changed:
                note = SHORTHAND_NOTE
        if note:
            status = f"{status} {note}".strip()
        self._replace(doc, loaded, path, status, dirty)
        return self._editor_snapshot(doc)

    def _replace(
        self, doc: _Doc, loaded: dict, path: Path | None, status: str, dirty: bool
    ) -> None:
        doc.generation += 1
        doc.extra_errors = set()
        doc.form = build_form(doc.schema)
        doc.form.load(loaded)
        doc.path = path
        doc.dirty = dirty
        doc.status = status
        names = _phase_names(doc.form) if doc.kind == "sweep" else []
        doc.selected_phase = names[0] if names else ""

    def _wrong_kind(
        self, doc: _Doc, path: Path, kind: str, loaded: dict, text: str
    ) -> dict:
        if kind == "sweep":
            return self._ask(
                "yes_no",
                "wrong-kind-sweep",
                "This is a sweep",
                f"{path.name} is a sweep file, not a training config. "
                "Open it in the Sweep tab instead?",
                lambda answer: self._wrong_sweep_answer(
                    doc, path, loaded, text, answer
                ),
            )
        return self._ask(
            "yes_no",
            "wrong-kind-config",
            "This is a training config",
            f"{path.name} is a training config, not a sweep. Start a new sweep "
            "that uses it as its base_config?\n\n(The training config itself is not changed.)",
            lambda answer: self._wrong_config_answer(doc, path, answer),
        )

    def _wrong_sweep_answer(
        self, doc: _Doc, path: Path, loaded: dict, text: str, answer: str
    ) -> Any:
        if answer == "no":
            doc.status = f"{path.name} is a sweep file; open it in the Sweep tab."
            return self._editor_snapshot(doc)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        self._accept_open(self.sweep, path, loaded, text)
        self.initial_page = "sweep"
        return self._editor_snapshot(doc)

    def _wrong_config_answer(self, doc: _Doc, path: Path, answer: str) -> Any:
        if answer == "no":
            doc.status = f"{path.name} is a training config; open it in the Config tab."
            return self._editor_snapshot(doc)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        before = self._base_text()

        def after_blank() -> Any:
            self._set_base_text(self._cwd_relative(path))
            self._touch(self.sweep)
            return self._editor_snapshot(self.sweep)

        if self.sweep.dirty:
            return self._discard(
                self.sweep, lambda ans: self._blank_then(ans, before, after_blank)
            )
        self.sweep.rebuild()
        return after_blank()

    def _blank_then(self, answer: str, before: str, after: Callable[[], Any]) -> Any:
        if answer == "cancel":
            return self._editor_snapshot(self.sweep)
        if answer == "no":
            self.sweep.dirty = False
            self.sweep.rebuild()
            return after()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes, no, or cancel.")
        return self._save_doc(
            self.sweep, then=lambda: (self.sweep.rebuild(), after())[1], chain=False
        )

    def _save(self, params: dict) -> Any:
        doc = self._doc(params)
        return self._save_doc(doc, then=None, chain=False)

    def _save_as(self, params: dict) -> Any:
        doc = self._doc(params)
        raw = params.get("path")
        if not raw:
            raise GuiError("Can't save", "Choose a file.")
        then = self._after_save
        self._after_save = None
        return self._confirm_write(doc, Path(str(raw)), then)

    def _save_doc(self, doc: _Doc, then: Callable[[], Any] | None, chain: bool) -> Any:
        if doc.path is None:
            self._after_save = then if chain else None
            return self._ask(
                "save_as",
                "save-as",
                "Save As",
                "",
                lambda _a: self._editor_snapshot(doc),
            )
        return self._confirm_write(doc, doc.path, then)

    def _confirm_write(
        self, doc: _Doc, path: Path, then: Callable[[], Any] | None
    ) -> Any:
        existing = file_kind(path) if path.exists() else None
        if existing is not None and existing != doc.kind:
            other = "training config" if existing == "config" else "sweep"
            return self._ask(
                "yes_no",
                "overwrite",
                f"Replace a {other}?",
                f"{path.name} is a {other}, and you are saving a {doc.noun} over it. "
                f"The {other} will be lost (unless it is committed in git).\n\n"
                "Replace it anyway?",
                lambda answer: self._overwrite_answer(doc, path, then, answer),
            )
        return self._confirm_problems(doc, path, then)

    def _overwrite_answer(
        self, doc: _Doc, path: Path, then: Callable[[], Any] | None, answer: str
    ) -> Any:
        if answer == "no":
            return self._editor_snapshot(doc)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        return self._confirm_problems(doc, path, then)

    def _confirm_problems(
        self, doc: _Doc, path: Path, then: Callable[[], Any] | None
    ) -> Any:
        problems = self._problems(doc)
        if problems:
            return self._ask(
                "yes_no",
                "problems",
                "Save with problems?",
                f"This {doc.noun} has {len(problems)} problem(s), listed at the bottom "
                "of the window. PhoenX will refuse to run it until they are fixed.\n\n"
                "Save anyway?",
                lambda answer: self._problems_answer(doc, path, then, answer),
            )
        return self._finish_write(doc, path, then)

    def _problems_answer(
        self, doc: _Doc, path: Path, then: Callable[[], Any] | None, answer: str
    ) -> Any:
        if answer == "no":
            return self._editor_snapshot(doc)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        return self._finish_write(doc, path, then)

    def _finish_write(
        self, doc: _Doc, path: Path, then: Callable[[], Any] | None
    ) -> Any:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._editor_snapshot(doc)["yaml"], encoding="utf-8")
        except OSError as exc:
            raise GuiError("Can't save", f"{path}\n\n{exc}") from None
        doc.path = path
        doc.dirty = False
        doc.status = f"Saved {path}."
        if then is not None:
            return then()
        return self._editor_snapshot(doc)

    # -- sweep -----------------------------------------------------------------

    def _select_phase(self, params: dict) -> dict:
        name = str(params.get("name", ""))
        names = _phase_names(self.sweep.form)
        if name not in names:
            raise GuiError("Can't edit", "No such field.")
        self.sweep.selected_phase = name
        return self._editor_snapshot(self.sweep)

    def _add_from_config(self, params: dict) -> Any:
        path = str(params.get("path", ""))
        how = str(params.get("how", "search"))
        if self.config.path is None or self.config.dirty:
            return self._save_first(
                self.config, lambda answer: self._add_from_after_save(path, how, answer)
            )
        return self._add_from_ready(path, how)

    def _add_from_after_save(self, path: str, how: str, answer: str) -> Any:
        if answer == "no":
            return self._editor_snapshot(self.sweep)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        return self._save_doc(
            self.config, then=lambda: self._add_from_ready(path, how), chain=True
        )

    def _add_from_ready(self, path: str, how: str) -> Any:
        base_text = self._base_text()
        config_path = self.config.path
        if config_path is None:
            return self._editor_snapshot(self.sweep)
        if not base_text:
            self._set_base_text(self._cwd_relative(config_path))
            self._touch(self.sweep)
            return self._add_field({"path": path, "how": how})
        current, _note = self._resolve_base(base_text)
        if current is None or current.resolve() != Path(config_path).resolve():
            return self._ask(
                "yes_no",
                "different-base",
                "Different base config",
                f"This sweep's base_config is {base_text!r}, not the file open in the "
                f"Config tab ({config_path}). Switch the sweep to that file?",
                lambda answer: self._different_base(path, how, config_path, answer),
            )
        return self._add_field({"path": path, "how": how})

    def _different_base(
        self, path: str, how: str, config_path: Path, answer: str
    ) -> Any:
        if answer == "no":
            return self._editor_snapshot(self.sweep)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        self._set_base_text(self._cwd_relative(config_path))
        self._touch(self.sweep)
        return self._add_field({"path": path, "how": how})

    def _add_field(self, params: dict) -> Any:
        path = str(params.get("path", ""))
        how = str(params.get("how", "search"))
        # The field comes from the sweep's base config file (the tree shows
        # that file), not from whatever the Config Builder has open.
        node = self._base_field(path)
        if node is None:
            raise GuiError("Can't edit", "No such field.")
        if not node.pinned:
            return self._not_in_base(path, node, how)
        return self._write_sweep_field(path, node, how)

    @staticmethod
    def _load_base(base_path: Path | None) -> dict | None:
        """The base config file's contents, or None when it can't be read."""
        if base_path is None:
            return None
        try:
            loaded = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def _base_field(self, path: str) -> ScalarNode | None:
        base_path, _note = self._resolve_base(self._base_text())
        loaded = self._load_base(base_path)
        if loaded is None:
            return None
        form = build_form()
        form.load(loaded)
        for found_path, found in _iter_scalars(form):
            if found_path == path:
                return found
        return None

    def _not_in_base(self, path: str, node: ScalarNode, how: str) -> Any:
        first = (
            f"{path} is not in the base config file (it is at its default, "
            f"{node.text or 'unset'}). A sweep can only change values its base "
            "config sets explicitly."
        )
        base_path, _note = self._resolve_base(self._base_text())
        open_here = (
            self.config.path is not None
            and base_path is not None
            and Path(self.config.path).resolve() == base_path.resolve()
        )
        if not open_here:
            return self._ask(
                "ok",
                "not-in-base-info",
                "Not in the base config",
                first + "\n\nOpen the base config in the Config tab and type the value "
                "there (then save), or add it to the file by hand.",
                lambda _a: self._editor_snapshot(self.sweep),
            )
        # Pin the Config Builder's copy of the field, then save that file.
        target = None
        for found_path, found in _iter_scalars(self.config.form):
            if found_path == path:
                target = found
                break
        if target is None:
            raise GuiError("Can't edit", "No such field.")
        return self._ask(
            "yes_no",
            "not-in-base-write",
            "Not in the base config",
            first
            + f"\n\nWrite it into {base_path.name} (open in the Config tab) and save?",
            lambda answer: self._pin_then_add(path, target, how, answer),
        )

    def _pin_then_add(self, path: str, node: ScalarNode, how: str, answer: str) -> Any:
        if answer == "no":
            return self._editor_snapshot(self.sweep)
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        node.pinned = True
        self._touch(self.config)
        return self._save_doc(
            self.config,
            then=lambda: self._write_sweep_field(path, node, how),
            chain=True,
        )

    def _write_sweep_field(self, path: str, node: ScalarNode, how: str) -> Any:
        try:
            value = node.parse()
        except ValueError:
            value = node.text
        if how == "override":
            overrides = _child_map(self.sweep.form, "overrides")
            target = overrides.entries.get(path) or overrides.add(path)
            target.text = _fmt(value)
            where = f"overrides.{path}"
        elif how == "search":
            phase = self._phase_object()
            space = _child_map(phase, "search_space")
            if path in space.entries:
                return self._ask(
                    "ok",
                    "already-searched",
                    "Already searched",
                    f"{path} is already in this phase's search space.",
                    lambda _a: self._editor_snapshot(self.sweep),
                )
            spec = suggest_spec(node.kind, value, node.name, list(node.choices))
            entry = space.add(path)
            if isinstance(entry, TaggedNode):
                entry.set_type(str(spec.get("dist", "")))
                variant = entry.variant
                if variant is not None:
                    for key, val in spec.items():
                        if key != "dist" and key in variant.children:
                            variant.children[key].text = _fmt(val)
            phases = self.sweep.form.children["phases"]
            index = next(i for i, item in enumerate(phases.items)
                         if (item.inner if isinstance(item, OptionalNode) else item) is phase)
            where = f"phases.{index}.search_space.{path}"
        else:
            raise GuiError("Can't edit", f"Unknown add mode {how}.")
        self._touch(self.sweep)
        label = "override" if how == "override" else "search"
        self.sweep.status = f"Added {path} ({label}). Adjust the range in the form."
        # The window expands the form to the new entry and scrolls to it.
        self.sweep.reveal = where
        snap = self._editor_snapshot(self.sweep)
        self.sweep.reveal = ""
        return snap

    def _phase_object(self) -> ObjectNode:
        phases = self.sweep.form.children.get("phases")
        if not isinstance(phases, ListNode):
            raise GuiError("Can't edit", "No such field.")
        for item in phases.items:
            obj = item.inner if isinstance(item, OptionalNode) else item
            if (
                isinstance(obj, ObjectNode)
                and obj.children["name"].text == self.sweep.selected_phase
            ):
                return obj
        if phases.items:
            obj = phases.items[0]
            return obj.inner if isinstance(obj, OptionalNode) else obj
        item = phases.add()
        obj = item.inner if isinstance(item, OptionalNode) else item
        obj.children["name"].text = "phase_1"
        self.sweep.selected_phase = "phase_1"
        return obj

    def _handoff(self, params: dict) -> Any:
        validate_only = _is_true(params.get("validate_only"))
        if self.sweep.path is None or self.sweep.dirty:
            return self._save_first(
                self.sweep, lambda answer: self._handoff_after(validate_only, answer)
            )
        return self._handoff_go(validate_only)

    def _handoff_after(self, validate_only: bool, answer: str) -> Any:
        if answer == "no":
            return self._launch_snapshot()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        return self._save_doc(
            self.sweep, then=lambda: self._handoff_go(validate_only), chain=True
        )

    def _handoff_go(self, validate_only: bool) -> Any:
        self.launch["mode"] = "sweep"
        self.launch["sweep"] = "" if self.sweep.path is None else str(self.sweep.path)
        self.launch["validate_only"] = "1" if validate_only else "0"
        self._refresh_phase_choices()
        if validate_only:
            try:
                self._start()
            except GuiError:
                pass
        snap = self._launch_snapshot()
        return {"goto": "launch", "snapshot": snap}

    def _save_first(self, doc: _Doc, resume: Callable[[str], Any]) -> dict:
        return self._ask(
            "yes_no",
            "save-first",
            f"Save the {doc.noun} first?",
            f"The {doc.noun} has unsaved changes (or was never saved). "
            "Save it now so the run uses what you see?",
            resume,
        )

    def _base_text(self) -> str:
        node = self.sweep.form.children.get("base_config")
        if isinstance(node, ScalarNode):
            return node.text.strip()
        return ""

    def _set_base_text(self, value: str) -> None:
        node = self.sweep.form.children.get("base_config")
        if isinstance(node, ScalarNode):
            node.text = value

    def _resolve_base(self, name: str) -> tuple[Path | None, str]:
        name = (name or "").strip()
        if not name:
            return None, "No base_config set."
        raw = Path(name)
        candidate = raw if raw.is_absolute() else Path(self._cwd()) / raw
        if candidate.is_file():
            return candidate, str(candidate)
        key = name.replace("\\", "/")
        if key in bundled_examples("configs"):
            from importlib import resources

            located = resources.files("phoenx.examples").joinpath(
                "configs", *key.split("/")
            )
            return Path(str(located)), f"bundled example {name}"
        return None, (
            f"base_config {name!r} not found (looked in {Path(self._cwd()).resolve()} "
            "and the bundled examples). Paths are relative to the Run tab's working folder."
        )

    def _base_fields(self, sweep_value: dict) -> tuple[list[dict], str]:
        path, note = self._resolve_base(str(sweep_value.get("base_config") or ""))
        if path is None:
            return [], note if sweep_value.get("base_config") else ""
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return [], note
        if not isinstance(loaded, dict):
            return [], note
        form = build_form()
        form.load(loaded)
        fields = []
        for field_path, node in _iter_scalars(form):
            fields.append(
                {"path": field_path, "text": node.text, "in_file": bool(node.pinned)}
            )
        return fields, note

    def _cwd(self) -> str:
        return self.launch.get("cwd") or os.getcwd()

    def _cwd_relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(Path(self._cwd()).resolve()).as_posix()
        except ValueError:
            return str(path)

    # -- launch ----------------------------------------------------------------

    def _launch_snapshot(self) -> dict:
        self._collect_info()
        try:
            argv = self._argv()
            command = command_text(argv)
        except ValueError as exc:
            command = f"({exc})"
        running = self.runner.running
        return {
            "mode": self.launch["mode"],
            "values": dict(self.launch),
            "command": command,
            "running": running,
            "returncode": self.runner.returncode,
            "status": self.launch_status,
            "progress": self.progress,
            "log": list(self.log),
            "start_enabled": not running,
            "stop_enabled": running,
            "stop_label": "Kill" if self._stop_asked and running else "Stop",
            "open_output_enabled": self._output_dir is not None,
            "output_dir": self._output_dir,
            "resume_info": self._info_text("agent_dir"),
            "test_info": self._info_text("test_agent_dir"),
            "phase_choices": list(self._phase_choices),
            "run_status": self.run_status,
        }

    def _info_text(self, key: str) -> str:
        for job in self._info_jobs:
            if job["key"] == key and job.get("text") == "Reading checkpoint...":
                return "Reading checkpoint..."
        info = self._infos.get(self.launch.get(key, "").strip())
        if info is None:
            path = self.launch.get(key, "").strip()
            if not path:
                return ""
            return ""
        return describe_run(info)

    def _set_mode(self, params: dict) -> dict:
        mode = str(params.get("mode", ""))
        if mode not in ("train", "resume", "test", "sweep"):
            raise GuiError("Can't edit", f"Unknown mode {mode}.")
        self.launch["mode"] = mode
        return self._launch_snapshot()

    def _launch_set(self, params: dict) -> dict:
        key = str(params.get("key", ""))
        if key not in self.launch or key == "mode":
            raise GuiError("Can't edit", "No such field.")
        self.launch[key] = "" if params.get("value") is None else str(params["value"])
        if key in ("agent_dir", "test_agent_dir"):
            self._queue_info(key)
        if key == "sweep":
            self._refresh_phase_choices()
        return self._launch_snapshot()

    def _use_editor(self, params: dict) -> Any:
        which = str(params.get("which", ""))
        if which not in ("config", "sweep"):
            raise GuiError("Can't edit", "No such field.")
        doc = self.config if which == "config" else self.sweep
        if doc.path is None or doc.dirty:
            return self._save_first(
                doc, lambda answer: self._use_after(which, doc, answer)
            )
        self.launch[which] = str(doc.path)
        if which == "sweep":
            self._refresh_phase_choices()
        return self._launch_snapshot()

    def _use_after(self, which: str, doc: _Doc, answer: str) -> Any:
        if answer == "no":
            return self._launch_snapshot()
        if answer != "yes":
            raise GuiError("Can't answer", "Answer yes or no.")
        return self._save_doc(doc, then=lambda: self._use_apply(which, doc), chain=True)

    def _use_apply(self, which: str, doc: _Doc) -> dict:
        if doc.path is not None:
            self.launch[which] = str(doc.path)
        if which == "sweep":
            self._refresh_phase_choices()
        return self._launch_snapshot()

    def _argv(self) -> list[str]:
        values = dict(self.launch)
        if values.get("mode") == "test":
            values["agent_dir"] = values.get("test_agent_dir", "")
        return build_command(str(values.get("mode")), values)

    def _start(self) -> dict:
        if self.runner.running:
            return self._launch_snapshot()
        problem = self._precheck()
        if problem:
            raise GuiError("Can't start", problem)
        argv = self._argv()
        cwd = self._cwd()
        self.log = []
        self._last_checkpoint = False
        self.progress = None
        self._stop_asked = False
        self._finished_code = None
        self._output_dir = _guess_output(self.launch, cwd)
        try:
            self.runner.start(argv, cwd=cwd)
        except OSError as exc:
            self._append(f"Failed to start: {exc}", "err")
            self.run_status = "Idle"
            self.launch_status = "Idle."
            return self._launch_snapshot()
        self.run_status = "Running"
        self.launch_status = "Running..."
        return self._launch_snapshot()

    def _precheck(self) -> str | None:
        cwd = self._cwd()
        if not os.path.isdir(cwd):
            return f"Working folder does not exist:\n{cwd}"
        mode = self.launch["mode"]
        if mode in ("train", "sweep"):
            key = "config" if mode == "train" else "sweep"
            label = "Config file" if mode == "train" else "Sweep file"
            raw = self.launch.get(key, "").strip()
            if not raw:
                return f"{label} is required"
            flipped = raw.replace("\\", "/")
            folder = "configs" if mode == "train" else "sweeps"
            path = Path(raw)
            path = path if path.is_absolute() else Path(cwd) / path
            if not path.is_file():
                if flipped not in bundled_examples(folder):
                    return f"{label} not found: {path}"
            else:
                kind = file_kind(path)
                expected = "config" if mode == "train" else "sweep"
                if kind is not None and kind != expected:
                    actual = "a sweep" if kind == "sweep" else "a training config"
                    wanted = "a training config" if mode == "train" else "a sweep"
                    tail = (
                        "Use Sweep mode to run a sweep."
                        if mode == "train"
                        else "Sweep files have base_config and phases."
                    )
                    return f"{path.name} is {actual}, not {wanted}. {tail}"
        if mode in ("resume", "test"):
            raw = self.launch.get(
                "agent_dir" if mode == "resume" else "test_agent_dir", ""
            ).strip()
            if not raw:
                return "Agent folder is required"
            folder = Path(raw)
            folder = folder if folder.is_absolute() else Path(cwd) / folder
            if not (folder / "config.json").is_file():
                return f"{folder} is not a saved run (no config.json)."
        if mode == "resume" and self.launch.get("stop_units", "").strip():
            info = self._infos.get(self.launch["agent_dir"].strip())
            try:
                target = int(self.launch["stop_units"].strip())
            except ValueError:
                return "Train until must be a whole number."
            if (
                info
                and isinstance(info.get("progress"), (int, float))
                and not isinstance(info.get("progress"), bool)
            ):
                if target <= info["progress"]:
                    return (
                        f"Train until ({target:,}) must be past the checkpoint "
                        f"({int(info['progress']):,}), or training stops immediately."
                    )
        return None

    def _stop(self) -> dict:
        if not self.runner.running:
            return self._launch_snapshot()
        self.runner.stop()
        self._stop_asked = True
        self.run_status = "Stopping"
        self.launch_status = "Stopping..."
        self._append(
            f"[gui] stop requested: interrupting, killing after "
            f"{int(self.grace_seconds)} s if it has not exited",
            "meta",
        )
        return self._launch_snapshot()

    def _kill(self) -> dict:
        self._stop_asked = True
        self.runner.kill()
        self._drain()
        if (
            self.runner.returncode is not None
            and self._finished_code is None
            and self.run_status != "Idle"
        ):
            self._finish(self.runner.returncode)
        return self._launch_snapshot()

    def _poll(self) -> dict:
        self._drain()
        if (
            not self.runner.running
            and self.runner.returncode is not None
            and self._finished_code is None
            and self.run_status in ("Running", "Stopping")
        ):
            self._finish(self.runner.returncode)
        self._collect_info()
        return self._launch_snapshot()

    def _drain(self) -> None:
        while True:
            batch = self.runner.poll(max_lines=100000)
            if not batch:
                break
            for line in batch:
                self._take_line(line)

    def _take_line(self, line: str) -> None:
        prog = parse_progress(line)
        if prog is not None:
            if prog.mode == "test" and prog.total and prog.done >= prog.total:
                self._append(line, "good")
                self._last_checkpoint = False
            best = (self.progress or {}).get("best")
            self.progress = {
                "mode": prog.mode,
                "unit": prog.unit,
                "done": prog.done,
                "total": prog.total,
                "fraction": prog.fraction,
                # For the Launch page's live tiles.
                "episodes": prog.episodes,
                "avg_reward": prog.avg_reward,
                "elapsed": prog.elapsed,
                "best": best,
            }
            reward = (
                f", avg reward {prog.avg_reward:.2f}"
                if prog.avg_reward is not None
                else ""
            )
            total = f"/{prog.total:,}" if prog.total else ""
            self.launch_status = f"{prog.unit} {prog.done:,}{total}{reward}"
            return
        if line.startswith("[checkpoint]"):
            best = _BEST_RE.search(line)
            if best is not None:
                self.progress = {**(self.progress or {}), "best": float(best.group(1).replace(",", ""))}
            if self._last_checkpoint and self.log:
                self.log[-1] = {"text": line, "tag": "good"}
            else:
                self._append(line, "good")
            self._last_checkpoint = True
            return
        self._last_checkpoint = False
        tag = "err" if any(bit in line for bit in _ERROR_BITS) else ""
        self._append(line, tag)

    def _append(self, text: str, tag: str) -> None:
        self.log.append({"text": text, "tag": tag})
        if len(self.log) > 5000:
            self.log = self.log[-5000:]

    def _finish(self, code: int) -> None:
        self._finished_code = code
        started = self.runner.started_at or time.time()
        elapsed = str(timedelta(seconds=int(time.time() - started)))
        if code == 0:
            self.run_status = "Finished"
            self.launch_status = f"Finished in {elapsed}."
            self._append(f"[gui] finished (exit 0) after {elapsed}", "good")
            if self.launch["mode"] == "train" and self._output_dir:
                self.launch["agent_dir"] = self._output_dir
                self.launch["test_agent_dir"] = self._output_dir
        elif code in (130, -2, 3221225786) or self._stop_asked:
            self.run_status = "Stopped"
            self.launch_status = f"Stopped after {elapsed}."
            self._append(f"[gui] stopped (exit {code}) after {elapsed}", "meta")
        else:
            self.run_status = "Failed"
            self.launch_status = f"Failed (exit {code}) after {elapsed}. See the log."
            self._append(f"[gui] failed with exit code {code}", "err")
        self._stop_asked = False

    def _queue_info(self, key: str) -> None:
        raw = self.launch.get(key, "").strip()
        if not raw:
            self._infos.pop(raw, None)
            return
        folder = Path(raw)
        folder = folder if folder.is_absolute() else Path(self._cwd()) / folder
        if not (folder / "config.json").is_file():
            self._infos[raw] = {
                "error": "(choose a run's save_dir: it must contain config.json)"
            }
            return
        job = {"key": key, "path": raw, "text": "Reading checkpoint...", "proc": None}
        proc = subprocess.Popen(
            [sys.executable, "-m", "phoenx.run_info", str(folder)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        job["proc"] = proc
        self._info_jobs.append(job)

    def _collect_info(self) -> None:
        still = []
        for job in self._info_jobs:
            proc = job["proc"]
            if proc.poll() is None:
                still.append(job)
                continue
            line = ""
            if proc.stdout is not None:
                line = (proc.stdout.read() or "").strip().splitlines()
                line = line[-1] if line else ""
            try:
                info = json.loads(line or "{}")
            except json.JSONDecodeError:
                info = {"error": "Could not read the run."}
            if self.launch.get(job["key"], "").strip() != job["path"]:
                continue
            self._infos[job["path"]] = info
            if (
                job["key"] == "agent_dir"
                and not self.launch.get("stop_units", "").strip()
            ):
                limit, done = info.get("stop_units"), info.get("progress")
                if (
                    isinstance(limit, int)
                    and not isinstance(limit, bool)
                    and isinstance(done, (int, float))
                    and not isinstance(done, bool)
                    and done >= limit
                ):
                    self.launch["stop_units"] = str(int(done) + int(limit))
                elif isinstance(limit, int) and not isinstance(limit, bool):
                    self.launch["stop_units"] = str(limit)
        self._info_jobs = still

    def _refresh_phase_choices(self) -> None:
        names = [""]
        raw = self.launch.get("sweep", "").strip()
        if raw:
            path = Path(raw)
            path = path if path.is_absolute() else Path(self._cwd()) / path
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                loaded = None
            if isinstance(loaded, dict):
                loaded, _changed = to_phases_form(loaded)
                for phase in loaded.get("phases") or []:
                    if isinstance(phase, dict) and phase.get("name"):
                        names.append(str(phase["name"]))
        self._phase_choices = names

    def load_startup(self, path: str) -> None:
        """Open ``path`` before the window asks for anything.

        A sweep file selects the sweep page. Anything else is opened as a
        training config when it is one.

        Args:
            path: Config or sweep file to open.
        """
        kind = file_kind(path)
        target = self.sweep if kind == "sweep" else self.config
        if kind == "sweep":
            self.initial_page = "sweep"
        if kind in ("config", "sweep"):
            self._open_path(target, Path(path))


def _is_true(value: object) -> bool:
    return value in (True, 1, "1", "true", "True")


def _phase_names(form: ObjectNode) -> list[str]:
    phases = form.children.get("phases")
    if not isinstance(phases, ListNode):
        return []
    names = []
    for item in phases.items:
        obj = item.inner if isinstance(item, OptionalNode) else item
        if isinstance(obj, ObjectNode) and "name" in obj.children:
            names.append(obj.children["name"].text)
    return names


def _child_map(node: ObjectNode, name: str) -> MapNode:
    child = node.children[name]
    if isinstance(child, OptionalNode):
        child = child.inner
    if not isinstance(child, MapNode):
        raise GuiError("Can't edit", "No such field.")
    return child


def _next_map_name(node: MapNode) -> str:
    for candidate in _MAP_NAMES:
        if candidate not in node.entries:
            return candidate
    return f"entry{len(node.entries) + 1}"


def _index(form: ObjectNode) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def walk(node: Any, path: str, bind: bool) -> None:
        if bind:
            found[path] = node
        if isinstance(node, OptionalNode):
            walk(node.inner, path, False)
            return
        if isinstance(node, ObjectNode):
            for child in node.children.values():
                child_path = f"{path}.{child.name}" if path else child.name
                walk(child, child_path, True)
        elif isinstance(node, TaggedNode):
            tag_path = f"{path}.{node.tag}" if path else node.tag
            walk(node.type_node, tag_path, True)
            if node.variant is not None:
                # A flat variant (no config_key) shares the tagged node's path;
                # binding it there would hide the tagged node from set_type.
                if node.config_key:
                    walk(node.variant, f"{path}.{node.config_key}", True)
                else:
                    walk(node.variant, path, False)
        elif isinstance(node, ListNode):
            for i, item in enumerate(node.items):
                walk(item, f"{path}.{i}", True)
        elif isinstance(node, MapNode):
            for key, entry in node.entries.items():
                walk(entry, f"{path}.{key}" if path else key, True)

    walk(form, "", True)
    return found


def _iter_scalars(form: ObjectNode):
    from phoenx.config_form import iter_fields

    return iter_fields(form)


def _must_scalar(found: dict[str, Any], path: str) -> ScalarNode:
    node = found.get(path)
    if isinstance(node, OptionalNode):
        node = node.inner
    if not isinstance(node, ScalarNode):
        raise GuiError("Can't edit", "No such field.")
    return node


def _as(doc: _Doc, params: dict, kind: type):
    node = _index(doc.form).get(str(params.get("path", "")))
    if isinstance(node, OptionalNode):
        node = node.inner
    if not isinstance(node, kind):
        raise GuiError("Can't edit", "No such field.")
    return node


def _object(node: Any) -> ObjectNode | None:
    if isinstance(node, OptionalNode):
        node = node.inner
    if isinstance(node, TaggedNode):
        node = node.variant
    return node if isinstance(node, ObjectNode) else None


def _index_of(params: dict, length: int) -> int:
    try:
        index = int(params["index"])
    except (KeyError, TypeError, ValueError):
        raise GuiError("Can't edit", "No such field.") from None
    if index < 0 or index >= length:
        raise GuiError("Can't edit", "No such field.")
    return index


def _choices(node: ScalarNode) -> tuple[list[str], bool]:
    values = [_fmt(choice) for choice in node.choices]
    if node.nullable and "null" not in values:
        values.append("null")
    if node.kind == "enum" and not node.required:
        values = [""] + values
    editable = node.kind != "enum" and bool(node.choices)
    return values, editable


# -----------------------------------------------------------------------------
# Config Builder overview cards
# -----------------------------------------------------------------------------

#: Agent card fields per agent type: the settings people tune first. The full
#: form ("All fields") still has every field; these are shortcuts into it.
AGENT_CARD: dict[str, tuple[str, ...]] = {
    "Reinforce": ("discount", "entropy_coefficient"),
    "ActorCritic": ("discount", "gae_coefficient", "entropy_coefficient", "value_coef", "policy_grad_clip"),
    "PPO": ("discount", "gae_coefficient", "policy_clip", "value_clip", "entropy_coefficient", "value_coef"),
    "DDPG": ("discount", "tau", "policy_grad_clip", "critic_grad_clip"),
    "TD3": ("discount", "tau", "policy_update_delay", "noise_clip", "critic_grad_clip"),
    "SAC": ("discount", "tau", "entropy_coefficient", "auto_entropy_tuning", "policy_grad_clip"),
}

#: Agents that learn from a rollout buffer. Rollout buffers ignore
#: ``batch_size`` / ``updates_per_learn``, and warmup only matters for replay
#: learners. Same split as ``metrics.py`` uses for its on/off-policy metrics.
ON_POLICY_AGENTS = frozenset({"Reinforce", "ActorCritic", "PPO"})

#: Agents whose ``learn()`` takes ``learning_epochs`` and ``mini_batch_size``.
EPOCH_AGENTS = frozenset({"PPO"})

#: Environment card fields, when the chosen backend has them.
ENV_CARD = ("cfg", "num_envs", "seed", "render_mode")

CARD_LABELS: dict[str, str] = {
    "discount": "Discount",
    "entropy_coefficient": "Entropy coefficient",
    "gae_coefficient": "GAE λ",
    "value_coef": "Value loss weight",
    "policy_grad_clip": "Policy grad clip",
    "critic_grad_clip": "Critic grad clip",
    "policy_clip": "Policy clip",
    "value_clip": "Value clip",
    "tau": "Target smoothing (τ)",
    "policy_update_delay": "Policy update delay",
    "noise_clip": "Noise clip",
    "auto_entropy_tuning": "Auto entropy tuning",
    "learn_every": "Learn every",
    "batch_size": "Batch size",
    "updates_per_learn": "Updates per learn",
    "warmup_steps": "Warmup steps",
    "mini_batch_size": "Mini-batch size",
    "learning_epochs": "Learning epochs",
    "seed": "Seed",
    "cfg": "Environment id",
    "num_envs": "Parallel envs",
    "render_mode": "Render mode",
}


def schedule_card(agent_type: str) -> tuple[str, ...]:
    """Schedule fields that matter for ``agent_type``, in card order."""
    keys: list[str] = ["learn_every"]
    if agent_type not in ON_POLICY_AGENTS:
        keys += ["batch_size", "updates_per_learn", "warmup_steps"]
    if agent_type in EPOCH_AGENTS:
        keys += ["mini_batch_size", "learning_epochs"]
    keys.append("seed")
    return tuple(keys)


def _cards(form: ObjectNode) -> dict:
    """Which fields the overview cards show, as paths into the form snapshot."""
    agent = form.children.get("agent")
    env = form.children.get("env")
    agent_type = agent.type_node.text if isinstance(agent, TaggedNode) else ""
    agent_fields = agent.variant.children if isinstance(agent, TaggedNode) and agent.variant else {}
    env_fields = env.variant.children if isinstance(env, TaggedNode) and env.variant else {}

    def entry(path: str, key: str) -> dict:
        return {"path": path, "label": CARD_LABELS.get(key, key)}

    agent_sub = f"agent.{agent.config_key}" if isinstance(agent, TaggedNode) and agent.config_key else "agent"
    env_sub = f"env.{env.config_key}" if isinstance(env, TaggedNode) and env.config_key else "env"
    return {
        "agent_type": agent_type,
        "on_policy": agent_type in ON_POLICY_AGENTS,
        "agent": [entry(f"{agent_sub}.{key}", key) for key in AGENT_CARD.get(agent_type, ()) if key in agent_fields],
        "schedule": [entry(f"schedule.{key}", key) for key in schedule_card(agent_type)],
        "env": [entry(f"{env_sub}.{key}", key) for key in ENV_CARD if key in env_fields],
        "normalizers": [
            {"path": f"{agent_sub}.{key}", "label": key.removesuffix("_normalizer")}
            for key in agent_fields
            if key.endswith("_normalizer")
        ],
        "model": f"{agent_sub}.model" if "model" in agent_fields else "",
    }


def _snap(node: Any, path: str) -> dict:
    if isinstance(node, OptionalNode):
        body = _snap(node.inner, path)
        body["inner_kind"] = body["kind"]
        body["enabled"] = node.enabled
        body["kind"] = "optional"
        body["path"] = path
        body["name"] = node.name
        if not node.enabled:
            body["children"] = []
        body["set"] = node.is_set()
        body["pinned"] = node.pinned
        return body
    text = ""
    choices: list[str] = []
    editable = False
    tag = ""
    children: list[dict] = []
    extra = ""
    # Extra fields the screen needs to draw what the Tkinter editor drew:
    # the "default X" hint and tooltip, a red entry for text that does not
    # parse, a dropdown only for enums and fields with choices (a nullable
    # number is still a text box), and the "other keys" row only for
    # objects that accept unknown keys.
    extras: dict[str, Any] = {}
    if isinstance(node, ScalarNode):
        text = node.text
        choices, editable = _choices(node)
        extras = {
            "default": None if node.default is UNSET else _fmt(node.default),
            "bad": bool(node.problems()),
            "combo": node.kind == "enum" or bool(node.choices),
        }
    elif isinstance(node, ObjectNode):
        extra = _fmt(node.extra) if node.extra else ""
        extras = {"open": bool(node.open)}
        for child in node.visible_children():
            child_path = f"{path}.{child.name}" if path else child.name
            children.append(_snap(child, child_path))
    elif isinstance(node, TaggedNode):
        text = node.type_node.text
        tag = node.tag
        choices, _editable = _choices(node.type_node)
        choices = [c for c in choices if c]
        editable = False
        sub = path
        if node.variant is not None:
            sub = f"{path}.{node.config_key}" if node.config_key else path
            extra = _fmt(node.variant.extra) if node.variant.extra else ""
            for child in node.variant.visible_children():
                children.append(
                    _snap(child, f"{sub}.{child.name}" if sub else child.name)
                )
        extras = {
            "open": bool(node.variant is not None and node.variant.open),
            "variant_path": sub,
        }
    elif isinstance(node, ListNode):
        for i, item in enumerate(node.items):
            children.append(_snap(item, f"{path}.{i}"))
    elif isinstance(node, MapNode):
        extras = {"path_keys": bool(node.schema.get("x-path-keys"))}
        for key, entry in node.entries.items():
            children.append(_snap(entry, f"{path}.{key}" if path else key))
    return {
        "path": path,
        "name": getattr(node, "name", ""),
        "kind": getattr(node, "kind", ""),
        "description": getattr(node, "description", ""),
        "required": bool(getattr(node, "required", False)),
        "nullable": bool(getattr(node, "nullable", False)),
        "hidden": bool(getattr(node, "hidden", False)),
        "pinned": bool(getattr(node, "pinned", False)),
        "set": bool(node.is_set()) if hasattr(node, "is_set") else False,
        "text": text,
        "choices": choices,
        "editable_choices": editable,
        "extra_text": extra,
        "enabled": True,
        "tag": tag,
        "children": children,
        **extras,
    }


#: ``[checkpoint] saved best (avg_reward=201.70) at ...``
_BEST_RE = re.compile(r"avg_reward=(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def _guess_output(values: dict, cwd: str) -> str | None:
    mode = values.get("mode")
    base = Path(cwd)
    try:
        if mode == "train":
            cfg_path = Path(values.get("config", ""))
            cfg_path = cfg_path if cfg_path.is_absolute() else base / cfg_path
            save_dir = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get(
                "save_dir"
            )
            if not save_dir:
                return None
            found = base / save_dir
            return str(found)
        if mode in ("resume", "test"):
            raw = values.get("agent_dir" if mode == "resume" else "test_agent_dir", "")
            found = Path(raw)
            return str(found if found.is_absolute() else base / found)
        if mode == "sweep":
            root = values.get("storage_path") or "ray_results"
            found = Path(root)
            return str(found if found.is_absolute() else base / found)
    except (OSError, yaml.YAMLError, TypeError):
        return None
    return None


def _write(stdout, payload: dict) -> None:
    stdout.write(json.dumps(payload) + "\n")
    stdout.flush()


def _error(stdout, ident: object, title: str, message: str) -> None:
    _write(stdout, {"id": ident, "error": {"title": title, "message": message}})


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Read one JSON request per line and write one JSON response per line.

    Returns 0 on end of input, and after a response whose result is
    ``{"closed": true}``. A bad line becomes an error response and the loop
    keeps going. ``--open PATH`` on the command line loads that file first.

    Args:
        stdin: Line source. Defaults to ``sys.stdin``.
        stdout: Line sink. Defaults to ``sys.stdout``.

    Returns:
        Process exit code. ``0`` means the window closed or the input ended.
    """
    session = GuiSession()
    if "--open" in sys.argv:
        slot = sys.argv.index("--open")
        if slot + 1 < len(sys.argv):
            session.load_startup(sys.argv[slot + 1])
    return _serve_session(session, stdin or sys.stdin, stdout or sys.stdout)


def _serve_session(session: GuiSession, stdin, stdout) -> int:
    while True:
        line = stdin.readline()
        if line == "":
            return 0
        if not line.strip():
            _error(stdout, None, "Can't complete", "The request was empty.")
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _error(stdout, None, "Can't complete", "That line is not JSON.")
            continue
        if not isinstance(req, dict) or "method" not in req:
            _error(
                stdout,
                req.get("id") if isinstance(req, dict) else None,
                "Can't complete",
                "The request needs a method.",
            )
            continue
        try:
            result = session.call(str(req["method"]), req.get("params") or {})
        except GuiError as exc:
            _error(stdout, req.get("id"), exc.title, exc.message)
            continue
        except Exception as exc:
            _error(stdout, req.get("id"), "Can't complete", str(exc))
            continue
        _write(stdout, {"id": req.get("id"), "result": result})
        if isinstance(result, dict) and result.get("closed") is True:
            return 0


def main(argv: list[str] | None = None) -> int:
    """Run the session as a process. ``--open PATH`` loads that file first.

    Args:
        argv: Arguments after the program name. None uses ``sys.argv``.

    Returns:
        The process exit code from ``serve``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    # ``serve`` reads ``--open`` from sys.argv so a direct call and ``-m`` agree.
    if argv is not None:
        sys.argv = [sys.argv[0], *args]
    return serve()


if __name__ == "__main__":
    sys.exit(main())
