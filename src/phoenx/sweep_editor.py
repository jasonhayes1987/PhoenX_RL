"""The config GUI's Sweep tab: build Ray Tune sweep files without typing paths.

It is the config editor ([phoenx.config_gui.ConfigEditor][]) pointed at the
sweep schema ([phoenx.sweep_schema][]), plus a **Base config fields** pane:
every field of the sweep's ``base_config`` with its dotted path and current
value. Double-click one (or use the buttons) to add it to the selected
phase's ``search_space`` with a starting search spec chosen from its type
and value, or to ``overrides`` as a constant. In the Config tab, right-click a
field label for the same actions.

A sweep can only change values its base config sets explicitly (the sweep
runner refuses a path that is not in the file, to catch typos). Fields still
at their default are listed as *default*; adding one offers to write it into
the base config when that file is open in the Config tab.

Problems are checked without Ray: structure, search specs, that every path
exists in the base config, and a sample trial per phase validated against the
training-config schema (every option of each ``choice`` included).
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import yaml

from phoenx.config_form import (
    ListNode,
    MapNode,
    ObjectNode,
    ScalarNode,
    TaggedNode,
    _fmt,
    build_form,
    dump_config,
    find_field,
    iter_fields,
)
from phoenx.config_gui import ConfigEditor, Tooltip, bundled_examples
from phoenx.config_schema import build_schema
from phoenx.sweep_schema import build_sweep_schema, suggest_spec, to_phases_form, validate_sweep

if TYPE_CHECKING:  # pragma: no cover
    from phoenx.config_gui import PhoenxApp

SWEEP_HEADER = ("# PhoenX sweep config, written by `phoenx-config gui`.\n"
                "# Run with: phoenx-tune --config <this file>\n")


def resolve_base_config(name: str, cwd: str | Path) -> tuple[Path | None, str]:
    """Find a sweep's ``base_config`` the way ``phoenx-tune`` will.

    An existing path (absolute, or relative to the working folder the sweep
    runs in) wins; otherwise a bundled example name.

    Returns:
        ``(path or None, description)``.
    """
    name = (name or "").strip()
    if not name:
        return None, "No base_config set."
    p = Path(name)
    candidate = p if p.is_absolute() else Path(cwd) / p
    if candidate.is_file():
        return candidate, f"{candidate}"
    bundled = bundled_examples().get(name.replace("\\", "/"))
    if bundled is not None:
        from importlib import resources
        path = resources.files("phoenx.examples").joinpath("configs", *name.replace("\\", "/").split("/"))
        return Path(str(path)), f"bundled example {name}"
    return None, (f"base_config {name!r} not found (looked in {Path(cwd).resolve()} and the "
                  "bundled examples). Paths are relative to the Run tab's working folder.")


class SweepEditor(ConfigEditor):
    """Editor for sweep files (the Sweep tab)."""

    NOUN = "sweep"
    BLANK_NOTE = ("New sweep. Set base_config and add a phase, then add fields from the Base "
                  "config pane (double-click), or right-click a field in the Config tab.")

    def __init__(self, master: tk.Tk, path: str | None = None, parent: tk.Widget | None = None,
                 app: "PhoenxApp | None" = None):
        """Build the Sweep tab inside ``parent``."""
        self._base_cache: dict = {}
        self._tree_for: tuple | None = None
        self.base_label_var = tk.StringVar(master)
        super().__init__(master, path, parent=parent, app=app)

    # -- hooks -----------------------------------------------------------------

    def make_schema(self) -> dict:
        """The sweep schema."""
        return build_sweep_schema()

    def examples(self) -> dict[str, str]:
        """Bundled example sweeps."""
        return bundled_examples("sweeps")

    def dump(self, config: dict) -> str:
        """Sweep YAML with its own header."""
        return dump_config(config, header=SWEEP_HEADER)

    def prepare_loaded(self, config: dict) -> tuple[dict, str]:
        """Rewrite single-phase shorthand as a one-phase ``phases`` list."""
        config, changed = to_phases_form(config)
        return config, ("Its single-phase shorthand is shown as one phase named phase_0 "
                        "(it runs the same).") if changed else ""

    def check(self, config: dict) -> list[str]:
        """Sweep problems, checked against the base config when it can be found."""
        base, note = self.base_config(config)
        problems = validate_sweep(config, base)
        if base is None and config.get("base_config"):
            problems.append(f"base_config: {note}")
        self._refresh_tree(config)
        return problems

    def extra_toolbar(self, bar: ttk.Frame) -> None:
        """Dry-run and Run buttons."""
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        b = ttk.Button(bar, text="Dry run", command=lambda: self._to_run_tab(validate_only=True))
        b.pack(side="left")
        Tooltip(b, "Save, then run phoenx-tune --validate-only in the Run tab: samples a few "
                   "trials per phase and prints them, without Ray or training.")
        b = ttk.Button(bar, text="Open in Run tab", command=lambda: self._to_run_tab(False))
        b.pack(side="left", padx=4)

    def build_side_pane(self, parent: ttk.Frame) -> None:
        """Tabs: the base config's fields, and the YAML preview."""
        tabs = ttk.Notebook(parent)
        tabs.pack(fill="both", expand=True)
        fields = ttk.Frame(tabs, padding=4)
        tabs.add(fields, text="Base config fields")
        preview = ttk.Frame(tabs, padding=4)
        tabs.add(preview, text="YAML preview")
        super().build_side_pane(preview)

        ttk.Label(fields, textvariable=self.base_label_var, wraplength=480,
                  foreground="#3b5b8a", justify="left").pack(fill="x", anchor="w")
        row = ttk.Frame(fields)
        row.pack(fill="x", pady=(4, 2))
        ttk.Label(row, text="Filter").pack(side="left")
        self.filter_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.filter_var).pack(side="left", fill="x", expand=True, padx=4)
        self.filter_var.trace_add("write", lambda *_: self._fill_tree())

        tree_frame = ttk.Frame(fields)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("value", "where"), selectmode="browse")
        self.tree.heading("#0", text="Field")
        self.tree.heading("value", text="Value")
        self.tree.heading("where", text="In file")
        self.tree.column("#0", width=260)
        self.tree.column("value", width=150)
        self.tree.column("where", width=60, anchor="center")
        ys = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("default", foreground="#8a8a8a")
        self.tree.bind("<Double-1>", lambda e: self._add_selected("search"))

        actions = ttk.Frame(fields)
        actions.pack(fill="x", pady=4)
        ttk.Label(actions, text="Phase").pack(side="left")
        self.phase_var = tk.StringVar()
        self.phase_box = ttk.Combobox(actions, textvariable=self.phase_var, width=16,
                                      state="readonly")
        self.phase_box.pack(side="left", padx=4)
        ttk.Button(actions, text="Add to search space",
                   command=lambda: self._add_selected("search")).pack(side="left", padx=2)
        ttk.Button(actions, text="Add as override",
                   command=lambda: self._add_selected("override")).pack(side="left", padx=2)
        self._tree_fields: dict[str, tuple[str, ScalarNode]] = {}

    # -- base config -------------------------------------------------------

    def working_folder(self) -> str:
        """The Run tab's working folder (where phoenx-tune will resolve base_config)."""
        panel = getattr(self.app, "run_panel", None) if self.app else None
        if panel is not None and panel.vars.get("cwd") is not None:
            return panel.vars["cwd"].get().strip() or os.getcwd()
        return os.getcwd()

    def base_config(self, config: dict | None = None) -> tuple[dict | None, str]:
        """The parsed base config (cached by file and mtime) and a description."""
        config = self.current_config() if config is None else config
        path, note = resolve_base_config(str(config.get("base_config") or ""),
                                         self.working_folder())
        if path is None:
            return None, note
        try:
            key = (str(path), path.stat().st_mtime)
        except OSError as exc:
            return None, f"Can't read {path}: {exc}"
        if key not in self._base_cache:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                return None, f"Can't read {path}: {exc}"
            self._base_cache = {key: (data if isinstance(data, dict) else None)}
        return self._base_cache[key], note

    def base_path(self) -> Path | None:
        """Resolved path of the base config file, if found."""
        path, _ = resolve_base_config(str(self.current_config().get("base_config") or ""),
                                      self.working_folder())
        return path

    def _refresh_tree(self, config: dict) -> None:
        base, note = self.base_config(config)
        names = [str(p.get("name")) for p in (config.get("phases") or [])
                 if isinstance(p, dict) and p.get("name")]
        if hasattr(self, "phase_box"):
            self.phase_box.configure(values=names)
            if self.phase_var.get() not in names:
                self.phase_var.set(names[0] if names else "")
        key = (id(base), note)
        if key == self._tree_for:
            return
        self._tree_for = key
        self.base_label_var.set(f"Base config: {note}" if base is not None else note)
        self._base_form = None
        if base is not None:
            form = build_form(build_schema())
            form.load(base)
            self._base_form = form
        self._fill_tree()

    def _fill_tree(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        self._tree_fields = {}
        form = getattr(self, "_base_form", None)
        if form is None:
            return
        needle = self.filter_var.get().strip().lower()
        folders: dict[str, str] = {}

        def folder(parts: list[str]) -> str:
            iid = ""
            for i in range(len(parts)):
                key = ".".join(parts[: i + 1])
                if key not in folders:
                    folders[key] = self.tree.insert(iid, "end", text=parts[i],
                                                    open=bool(needle) or i < 1)
                iid = folders[key]
            return iid

        for path, node in iter_fields(form):
            if needle and needle not in path.lower():
                continue
            parts = path.split(".")
            parent = folder(parts[:-1]) if len(parts) > 1 else ""
            in_file = node.pinned
            iid = self.tree.insert(parent, "end", text=parts[-1],
                                   values=(node.text, "yes" if in_file else "default"),
                                   tags=() if in_file else ("default",))
            self._tree_fields[iid] = (path, node)

    # -- adding fields -----------------------------------------------------

    def _add_selected(self, how: str) -> None:
        sel = self.tree.selection()
        if not sel or sel[0] not in self._tree_fields:
            messagebox.showinfo("Pick a field", "Select a field (a leaf of the tree) first.",
                                parent=self.master)
            return
        path, node = self._tree_fields[sel[0]]
        self.add_field(path, node, how)

    def add_from_config(self, path: str, node: ScalarNode, how: str) -> None:
        """Called by the Config tab's right-click menu."""
        app = self.app
        editor = app.editor
        config_path = app.editor_path(save_first=True)
        if config_path is None:
            return
        base_text = str(self.current_config().get("base_config") or "").strip()
        current = self.base_path()
        if not base_text:
            self._set_base(str(config_path))
        elif current is None or current.resolve() != Path(config_path).resolve():
            if not messagebox.askyesno(
                    "Different base config",
                    f"This sweep's base_config is {base_text!r}, not the file open in the "
                    f"Config tab ({config_path}). Switch the sweep to that file?",
                    parent=self.master):
                return
            self._set_base(str(config_path))
        # Use the editor's node (it may have been pinned just now).
        field = find_field(editor.form, path) or node
        app.show_tab("sweep")
        self.add_field(path, field, how)

    def _set_base(self, value: str) -> None:
        cwd = Path(self.working_folder()).resolve()
        try:
            value = str(Path(value).resolve().relative_to(cwd)).replace("\\", "/")
        except ValueError:
            pass
        node = self.form.children["base_config"]
        node.text = value
        self.rerender()
        self.on_change()

    def _phase_node(self) -> ObjectNode:
        phases: ListNode = self.form.children["phases"]
        name = self.phase_var.get() if hasattr(self, "phase_var") else ""
        for item in phases.items:
            if isinstance(item, ObjectNode) and item.children["name"].text == name:
                return item
        if phases.items:
            return phases.items[0]
        item = phases.add()
        item.children["name"].text = "phase_1"
        return item

    def _ensure_in_base(self, path: str, node: ScalarNode) -> bool:
        """A sweep can only change values the base file sets; offer to set this one."""
        if node.pinned:
            return True
        app = self.app
        base = self.base_path()
        editor = app.editor if app is not None else None
        open_in_editor = (editor is not None and editor.path is not None and base is not None
                          and Path(editor.path).resolve() == base.resolve())
        msg = (f"{path} is not in the base config file (it is at its default, "
               f"{node.text or 'unset'}). A sweep can only change values its base config sets "
               "explicitly.")
        if not open_in_editor:
            messagebox.showinfo("Not in the base config", msg + "\n\nOpen the base config in the "
                                "Config tab and type the value there (then save), or add it to "
                                "the file by hand.", parent=self.master)
            return False
        if not messagebox.askyesno("Not in the base config", msg + "\n\nWrite it into "
                                   f"{base.name} (open in the Config tab) and save?",
                                   parent=self.master):
            return False
        target = find_field(editor.form, path)
        if target is None:
            return False
        target.pinned = True
        editor.on_change()
        if not editor.save():
            return False
        self._tree_for = None  # base changed on disk: rebuild the tree
        return True

    def add_field(self, path: str, node: ScalarNode, how: str) -> None:
        """Add ``path`` to the selected phase's search space, or to ``overrides``."""
        if not self._ensure_in_base(path, node):
            return
        try:
            value = node.parse()
        except ValueError:
            value = node.text
        if how == "override":
            overrides: MapNode = self.form.children["overrides"]
            target = overrides.entries.get(path) or overrides.add(path)
            target.text = _fmt(value)
            where = f"overrides.{path}"
        else:
            phase = self._phase_node()
            space: MapNode = phase.children["search_space"]
            if path in space.entries:
                messagebox.showinfo("Already searched", f"{path} is already in this phase's "
                                    "search space.", parent=self.master)
                return
            spec = suggest_spec(node.kind, value, node.name, list(node.choices))
            entry: TaggedNode = space.add(path)
            entry.set_type(spec["dist"])
            for key, val in spec.items():
                if key != "dist" and entry.variant is not None and key in entry.variant.children:
                    entry.variant.children[key].text = _fmt(val)
            index = self.form.children["phases"].items.index(phase)
            where = f"phases.{index}.search_space.{path}"
        self.view.expand_to(where)
        self.rerender()
        self.on_change()
        self.set_status(f"Added {path} ({'override' if how == 'override' else 'search'}). "
                        "Adjust the range in the form.")
        found = self.view.widget_for(where)
        if found:
            self.scroll.see(found[1])

    def _to_run_tab(self, validate_only: bool) -> None:
        app = self.app
        if app is None:
            return
        path = app.sweep_path(save_first=True)
        if path is None:
            return
        panel = app.run_panel
        panel.mode.set("sweep")
        panel._show_mode()
        panel.vars["sweep"].set(str(path))
        panel.vars["validate_only"].set("1" if validate_only else "0")
        app.show_tab("run")
        if validate_only:
            panel.start()
