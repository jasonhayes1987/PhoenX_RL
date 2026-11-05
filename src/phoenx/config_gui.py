"""Tkinter editor for PhoenX training configs.

Run it with ``phoenx-config gui`` (optionally followed by a file to open).

The window is generated from the config schema
([phoenx.config_schema][]) through the GUI-independent form model in
[phoenx.config_form][], so every agent, layer, head, normalizer and buffer
setting PhoenX understands appears as a field, dropdown or checkbox, with the
code's own docstring as its tooltip. Fields start at the value PhoenX would
use anyway, and only settings that differ from those defaults are written,
so a saved file contains exactly the choices that matter.

Layout:

* left: the form, one collapsible section per config block;
* right: a live preview of the YAML that **Save** will write;
* bottom: live problems (bad input, unknown keys, schema errors, n-step
  mismatches). Double-click a problem to jump to its field.

Only the Python standard library's ``tkinter`` is needed; torch is never
imported.
"""

from __future__ import annotations

import copy
import sys
import tkinter as tk
from importlib import resources
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import yaml

from phoenx.config_form import (
    UNSET,
    ListNode,
    MapNode,
    Node,
    ObjectNode,
    OptionalNode,
    ScalarNode,
    TaggedNode,
    ConstNode,
    _fmt,
    build_form,
    dump_config,
)
from phoenx.config_schema import build_schema, validate_config
from phoenx.config_utils import resolve_n_step

#: Sections at this depth (count of path segments) and deeper start collapsed.
COLLAPSE_DEPTH = 3
#: Tagged list items with at most this many fields are laid out on one row.
INLINE_MAX_FIELDS = 4

LABEL_WIDTH = 24
ENTRY_WIDTH = 26

ERROR_FG = "#b00020"
HINT_FG = "#6b6b6b"


# =============================================================================
# Small widgets
# =============================================================================

class Tooltip:
    """Hover text for a widget."""

    def __init__(self, widget: tk.Widget, text: str):
        """Create the node / widget; see the class docstring."""
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", wraplength=420, background="#ffffe8",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame; put content in ``.inner``."""

    def __init__(self, master, **kw):
        """Create the node / widget; see the class docstring."""
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))

    def scroll(self, event) -> str:
        """Mouse-wheel handler (Windows/macOS ``delta``, X11 buttons 4/5)."""
        if getattr(event, "num", None) == 4:
            step = -3
        elif getattr(event, "num", None) == 5:
            step = 3
        else:
            step = -int(event.delta / 120) * 3 if abs(event.delta) >= 120 else -int(event.delta)
        self.canvas.yview_scroll(step, "units")
        return "break"

    def see(self, widget: tk.Widget) -> None:
        """Scroll so ``widget`` is near the top of the view."""
        self.update_idletasks()
        y = widget.winfo_rooty() - self.inner.winfo_rooty()
        height = max(self.inner.winfo_height(), 1)
        self.canvas.yview_moveto(max(0.0, (y - 40) / height))


# =============================================================================
# Form rendering
# =============================================================================

class FormView:
    """Renders a form-node tree into Tk widgets and keeps the two in sync."""

    def __init__(self, editor: "ConfigEditor", parent: ttk.Frame):
        """Create the node / widget; see the class docstring."""
        self.editor = editor
        self.parent = parent
        self.collapsed: dict[str, bool] = {}
        #: None: collapse by depth; True/False: "Collapse all" / "Expand all".
        self.default_collapsed: bool | None = None
        self.widgets: dict[str, tk.Widget] = {}   # path -> focusable widget
        self.entry_nodes: list[tuple[ttk.Entry, ScalarNode, str]] = []

    # -- helpers -------------------------------------------------------------

    def changed(self) -> None:
        """Tell the editor the form changed (refresh preview / problems)."""
        self.editor.on_change()

    def _is_collapsed(self, path: str) -> bool:
        if path not in self.collapsed:
            if self.default_collapsed is not None:
                self.collapsed[path] = self.default_collapsed
            else:
                depth = len([p for p in path.split(".") if p])
                self.collapsed[path] = depth >= COLLAPSE_DEPTH
        return self.collapsed[path]

    def _show(self, node: Node) -> bool:
        """With "show only set fields", hide untouched optional fields."""
        if not self.editor.only_set.get():
            return True
        if node.required:
            return True
        if isinstance(node, OptionalNode):
            return node.enabled
        return node.is_set()

    @staticmethod
    def _label(node: Node) -> str:
        return node.name

    def _register(self, path: str, widget: tk.Widget) -> None:
        self.widgets.setdefault(path or "<root>", widget)

    # -- entry points ----------------------------------------------------------

    def render(self, root: ObjectNode) -> None:
        """(Re)build the whole form."""
        for child in self.parent.winfo_children():
            child.destroy()
        self.widgets.clear()
        self.entry_nodes.clear()
        self._children(root, self.parent, "")
        self.refresh_entry_styles()

    def _children(self, node: ObjectNode, frame: ttk.Frame, path: str) -> None:
        for child in node.visible_children():
            if not self._show(child):
                continue
            self.field(child, frame, f"{path}.{child.name}" if path else child.name)
        if node.open and (node.extra or not self.editor.only_set.get()):
            self._extra_row(node, frame, path)

    def field(self, node: Node, frame: ttk.Frame, path: str) -> None:
        """Render one node of any kind."""
        if isinstance(node, ScalarNode):
            self.scalar_row(node, frame, path)
        elif isinstance(node, ConstNode):
            return
        elif isinstance(node, OptionalNode):
            self.section(node, frame, path, optional=True)
        elif isinstance(node, (ObjectNode, TaggedNode, ListNode, MapNode)):
            self.section(node, frame, path)

    # -- scalars ---------------------------------------------------------------

    def scalar_row(self, node: ScalarNode, frame: ttk.Frame, path: str,
                   inline: bool = False) -> tk.Widget:
        """Render a label + input row for a scalar."""
        row = ttk.Frame(frame)
        row.pack(side="left" if inline else "top", fill="x" if not inline else None,
                 anchor="w", pady=1, padx=(0, 10) if inline else 0)
        label = ttk.Label(row, text=self._label(node) + (" *" if node.required else ""),
                          width=None if inline else LABEL_WIDTH, anchor="w")
        label.pack(side="left")
        tip = node.description
        if node.default is not UNSET:
            tip = f"{tip}\n\nDefault: {_fmt(node.default)}".strip()
        widget = self._scalar_widget(node, row, path, inline)
        widget.pack(side="left", padx=(4, 0))
        Tooltip(label, tip)
        Tooltip(widget, tip)
        if not inline and node.default is not UNSET and node.kind not in ("bool",):
            ttk.Label(row, text=f"default {_fmt(node.default)}", foreground=HINT_FG).pack(
                side="left", padx=8)
        self._register(path, widget)
        return widget

    def _scalar_widget(self, node: ScalarNode, row: ttk.Frame, path: str, inline: bool):
        if node.kind == "bool":
            var = tk.BooleanVar(value=node.text.strip() == "true")

            def on_bool(*_):
                node.text = "true" if var.get() else "false"
                self.changed()

            var.trace_add("write", on_bool)
            w = ttk.Checkbutton(row, variable=var)
            w._var = var  # keep a reference
            return w

        if node.kind == "enum" or node.choices:
            values = [_fmt(c) for c in node.choices]
            if node.nullable and "null" not in values:
                values.append("null")
            if not node.required and node.kind == "enum":
                values = [""] + values  # "" = use the default
            var = tk.StringVar(value=node.text)
            state = "readonly" if node.kind == "enum" else "normal"
            longest = max((len(v) for v in values), default=8) + 2
            w = ttk.Combobox(row, textvariable=var, values=values, state=state,
                             width=max(8, min(14 if inline else ENTRY_WIDTH, longest)))
        else:
            var = tk.StringVar(value=node.text)
            width = ENTRY_WIDTH + 14 if node.kind == "raw" else ENTRY_WIDTH
            w = ttk.Entry(row, textvariable=var, width=(14 if node.kind == "raw" else 8)
                          if inline else width)

        def on_text(*_):
            node.text = var.get()
            self._style_entry(w, node)
            self.changed()

        var.trace_add("write", on_text)
        w._var = var
        self.entry_nodes.append((w, node, path))
        return w

    def _style_entry(self, widget: tk.Widget, node: ScalarNode) -> None:
        bad = bool(node.problems())
        try:
            widget.configure(style="Bad.TEntry" if bad and isinstance(widget, ttk.Entry)
                             and not isinstance(widget, ttk.Combobox) else
                             ("Bad.TCombobox" if bad else
                              ("TCombobox" if isinstance(widget, ttk.Combobox) else "TEntry")))
        except tk.TclError:
            pass

    def refresh_entry_styles(self) -> None:
        """Re-color every input by whether its text parses."""
        for w, node, _ in self.entry_nodes:
            self._style_entry(w, node)

    def _extra_row(self, node: ObjectNode, frame: ttk.Frame, path: str) -> None:
        row = ttk.Frame(frame)
        row.pack(fill="x", anchor="w", pady=1)
        label = ttk.Label(row, text="other keys", width=LABEL_WIDTH, anchor="w",
                          foreground=HINT_FG)
        label.pack(side="left")
        var = tk.StringVar(value=_fmt(node.extra) if node.extra else "")
        entry = ttk.Entry(row, textvariable=var, width=ENTRY_WIDTH + 14)
        entry.pack(side="left", padx=(4, 0))
        Tooltip(label, "Keys this section has that the form has no field for, as a YAML "
                       "mapping, e.g. {_diag_freq: 1000}. Keys starting with _ are "
                       "diagnostic knobs; anything else is probably a typo.")

        def on_extra(*_):
            try:
                node.set_extra_text(var.get())
                entry.configure(style="TEntry")
            except (ValueError, yaml.YAMLError):
                entry.configure(style="Bad.TEntry")
            self.changed()

        var.trace_add("write", on_extra)
        entry._var = var

    # -- sections ----------------------------------------------------------

    def section(self, node: Node, frame: ttk.Frame, path: str, optional: bool = False) -> None:
        """Render a collapsible section (object, tagged, list, map, optional)."""
        outer = ttk.Frame(frame)
        outer.pack(fill="x", anchor="w", pady=(3, 1))
        header = ttk.Frame(outer)
        header.pack(fill="x", anchor="w")
        body = ttk.Frame(outer, padding=(18, 0, 0, 2))
        inner_node = node.inner if isinstance(node, OptionalNode) else node

        arrow = ttk.Label(header, width=2, cursor="hand2")
        arrow.pack(side="left")
        if optional:
            var = tk.BooleanVar(value=node.enabled)
            title = ttk.Checkbutton(header, text=node.name, variable=var)
        else:
            title = ttk.Label(header, text=node.name + (" *" if node.required else ""),
                              font=("TkDefaultFont", 10, "bold"), cursor="hand2")
        title.pack(side="left")
        Tooltip(title, node.description)
        self._register(path, title)
        summary = ttk.Label(header, foreground=HINT_FG)
        summary.pack(side="left", padx=8)

        def fill_body():
            for c in body.winfo_children():
                c.destroy()
            self._section_body(inner_node, body, path, header_extra)
            self.refresh_entry_styles()

        def update_visibility():
            enabled = node.enabled if optional else True
            open_ = enabled and not self._is_collapsed(path)
            arrow.configure(text=("-" if open_ else "+") if enabled else " ")
            summary.configure(text=self._summary(inner_node) if not open_ and enabled else "")
            if open_:
                if not body.winfo_children():
                    fill_body()
                body.pack(fill="x", anchor="w")
            else:
                body.pack_forget()

        def toggle(_e=None):
            self.collapsed[path] = not self._is_collapsed(path)
            update_visibility()

        arrow.bind("<Button-1>", toggle)
        if not optional:
            title.bind("<Button-1>", toggle)
        header_extra = ttk.Frame(header)
        header_extra.pack(side="left", padx=6)

        if optional:
            def on_enable(*_):
                node.enabled = var.get()
                if node.enabled:
                    self.collapsed[path] = False
                update_visibility()
                self.changed()
            var.trace_add("write", on_enable)
            title._var = var
        self._header_controls(inner_node, header_extra, body, path, fill_body, update_visibility)
        update_visibility()

    def _summary(self, node: Node) -> str:
        if isinstance(node, ListNode):
            return f"{len(node.items)} item(s)"
        if isinstance(node, MapNode):
            return ", ".join(node.entries) or "empty"
        return ""

    def _header_controls(self, node, header, body, path, fill_body, update_visibility):
        """Type dropdown (tagged) or add buttons (list/map) in the section header."""
        if isinstance(node, TaggedNode):
            var = tk.StringVar(value=node.type_node.text)
            box = ttk.Combobox(header, textvariable=var, values=node.choices, state="readonly",
                               width=max(14, max(len(c) for c in node.choices) + 2))
            box.pack(side="left")
            Tooltip(box, "Choose the type; the fields below change to match.")
            self._register(f"{path}.type", box)

            def on_type(*_):
                node.set_type(var.get())
                self.collapsed[path] = False
                fill_body()
                update_visibility()
                self.changed()

            var.trace_add("write", on_type)
            box._var = var
        elif isinstance(node, ListNode):
            def add():
                node.add()
                self.collapsed[path] = False
                fill_body()
                update_visibility()
                self.changed()
            ttk.Button(header, text="+ Add", width=7, command=add).pack(side="left")
        elif isinstance(node, MapNode):
            name_var = tk.StringVar()
            entry = ttk.Entry(header, textvariable=name_var, width=14)
            entry.pack(side="left")
            Tooltip(entry, "Name for the new entry (for branches: policy, value, critic, ...).")

            def add():
                try:
                    node.add(name_var.get() or self._next_name(node))
                except ValueError as exc:
                    messagebox.showerror("Can't add", str(exc), parent=self.parent)
                    return
                name_var.set("")
                self.collapsed[path] = False
                fill_body()
                update_visibility()
                self.changed()
            ttk.Button(header, text="+ Add", width=7, command=add).pack(side="left", padx=4)
            entry.bind("<Return>", lambda e: add())

    @staticmethod
    def _next_name(node: MapNode) -> str:
        for candidate in ("policy", "value", "critic", "critic_b", "actor"):
            if candidate not in node.entries:
                return candidate
        return f"entry{len(node.entries) + 1}"

    def _section_body(self, node: Node, body: ttk.Frame, path: str, header) -> None:
        if isinstance(node, ObjectNode):
            self._children(node, body, path)
        elif isinstance(node, TaggedNode):
            variant = node.variant
            if variant is not None:
                sub = f"{path}.{node.config_key}" if node.config_key else path
                self._children(variant, body, sub)
        elif isinstance(node, ListNode):
            self._list_items(node, body, path)
        elif isinstance(node, MapNode):
            self._map_entries(node, body, path)

    # -- lists and maps --------------------------------------------------------

    def _list_items(self, node: ListNode, body: ttk.Frame, path: str) -> None:
        if not node.items:
            ttk.Label(body, text="(empty: use + Add)", foreground=HINT_FG).pack(anchor="w")
            return

        def rebuild():
            for c in body.winfo_children():
                c.destroy()
            self._list_items(node, body, path)
            self.refresh_entry_styles()
            self.changed()

        for i, item in enumerate(node.items):
            item_path = f"{path}.{i}"
            box = ttk.Frame(body, relief="groove", borderwidth=1, padding=(4, 2))
            box.pack(fill="x", anchor="w", pady=1)
            bar = ttk.Frame(box)
            bar.pack(side="left", anchor="n")
            ttk.Label(bar, text=f"{i + 1}.", width=3).pack(side="left")
            for text, cmd, tip in (
                ("^", lambda i=i: (node.move(i, -1), rebuild()), "Move up"),
                ("v", lambda i=i: (node.move(i, +1), rebuild()), "Move down"),
                ("x", lambda i=i: (node.remove(i), rebuild()), "Remove"),
            ):
                b = ttk.Button(bar, text=text, width=2, command=cmd)
                b.pack(side="left")
                Tooltip(b, tip)
            content = ttk.Frame(box)
            content.pack(side="left", fill="x", expand=True, padx=(6, 0))
            self._list_item(item, content, item_path)

    def _list_item(self, item: Node, frame: ttk.Frame, path: str) -> None:
        """A list item: tagged items get their type dropdown on the same row."""
        if isinstance(item, TaggedNode):
            row = ttk.Frame(frame)
            row.pack(fill="x", anchor="w")
            var = tk.StringVar(value=item.type_node.text)
            box = ttk.Combobox(row, textvariable=var, values=item.choices, state="readonly",
                               width=max(10, min(16, max(len(c) for c in item.choices) + 2)))
            box.pack(side="left")
            self._register(f"{path}.type", box)
            holder = ttk.Frame(frame)
            holder.pack(fill="x", anchor="w")
            fields = ttk.Frame(frame)  # child of the common parent, so it may move

            def place_fields():
                # Few fields: same row as the type. More: a second row below.
                fields.pack_forget()
                if self._inline(item) and self._field_count(item) <= 2:
                    fields.pack(in_=row, side="left", anchor="w", padx=(8, 0))
                else:
                    fields.pack(in_=holder, side="top", fill="x", anchor="w", padx=(4, 0), pady=(2, 0))

            def fill():
                for c in fields.winfo_children():
                    c.destroy()
                variant = item.variant
                if variant is None:
                    return
                sub = f"{path}.{item.config_key}" if item.config_key else path
                inline = self._inline(item)
                for child in variant.visible_children():
                    child_path = f"{sub}.{child.name}"
                    if isinstance(child, ScalarNode) and inline:
                        self.scalar_row(child, fields, child_path, inline=True)
                    elif self._show(child):
                        self.field(child, fields, child_path)
                if variant.extra:
                    self._extra_row(variant, fields, sub)
                self.refresh_entry_styles()

            def on_type(*_):
                item.set_type(var.get())
                fill()
                place_fields()
                self.changed()

            var.trace_add("write", on_type)
            box._var = var
            fill()
            place_fields()
        elif isinstance(item, ObjectNode):
            self._children(item, frame, path)
        else:
            self.field(item, frame, path)

    @staticmethod
    def _field_count(item: TaggedNode) -> int:
        variant = item.variant
        return 0 if variant is None else len([c for c in variant.visible_children() if not c.hidden])

    def _inline(self, item: TaggedNode) -> bool:
        variant = item.variant
        if variant is None:
            return True
        kids = [c for c in variant.visible_children() if not c.hidden]
        return len(kids) <= INLINE_MAX_FIELDS and all(isinstance(c, ScalarNode) for c in kids)

    def _map_entries(self, node: MapNode, body: ttk.Frame, path: str) -> None:
        if not node.entries:
            ttk.Label(body, text="(empty: type a name and use + Add)",
                      foreground=HINT_FG).pack(anchor="w")
            return

        def rebuild():
            for c in body.winfo_children():
                c.destroy()
            self._map_entries(node, body, path)
            self.refresh_entry_styles()
            self.changed()

        for key, entry in list(node.entries.items()):
            box = ttk.Frame(body)
            box.pack(fill="x", anchor="w")
            remove = ttk.Button(box, text="x", width=2,
                                command=lambda k=key: (node.remove(k), rebuild()))
            remove.pack(side="left", anchor="n", pady=4)
            Tooltip(remove, f"Remove {key}")
            content = ttk.Frame(box)
            content.pack(side="left", fill="x", expand=True)
            self.section(entry, content, f"{path}.{key}")

    # -- navigation --------------------------------------------------------

    def widget_for(self, path: str) -> tuple[str, tk.Widget] | None:
        """Best registered widget for an error path (longest matching prefix)."""
        parts = path.split(".")
        while parts:
            p = ".".join(parts)
            if p in self.widgets:
                return p, self.widgets[p]
            parts.pop()
        return None

    def expand_to(self, path: str) -> None:
        """Mark every section on ``path`` as expanded (call before re-render)."""
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            self.collapsed[".".join(parts[:i])] = False


# =============================================================================
# Main window
# =============================================================================

def bundled_examples() -> dict[str, str]:
    """``{relative name: YAML text}`` for the configs shipped with PhoenX."""
    out: dict[str, str] = {}
    try:
        root = resources.files("phoenx.examples").joinpath("configs")
    except ModuleNotFoundError:
        return out

    def walk(node, prefix):
        for entry in sorted(node.iterdir(), key=lambda e: e.name):
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                walk(entry, rel)
            elif entry.name.endswith(".yml"):
                out[rel] = entry.read_text(encoding="utf-8")

    if root.is_dir():
        walk(root, "")
    return out


class ConfigEditor:
    """The editor window. ``ConfigEditor(tk.Tk()).run()`` or ``main()``."""

    def __init__(self, master: tk.Tk, path: str | None = None):
        """Create the node / widget; see the class docstring."""
        self.master = master
        self.schema = build_schema()
        self.form = build_form(self.schema)
        self.path: Path | None = None
        self.dirty = False
        self._pending = None
        self.problems: list[str] = []

        self.only_set = tk.BooleanVar(value=False)
        self.with_defaults = tk.BooleanVar(value=False)

        master.title("PhoenX Config Editor")
        master.geometry("1360x880")
        master.minsize(900, 560)
        self._styles()
        self._layout()
        master.protocol("WM_DELETE_WINDOW", self.quit)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            master.bind_all(seq, self._wheel, add="+")
            # Don't let the wheel silently change a dropdown while scrolling.
            master.bind_class("TCombobox", seq, self._wheel)
        master.bind_all("<Control-s>", lambda e: self.save())
        master.bind_all("<Control-o>", lambda e: self.open())

        if path:
            self.open(path)
        else:
            self.new_blank()

    # -- layout --------------------------------------------------------------

    def _styles(self) -> None:
        style = ttk.Style(self.master)
        if sys.platform.startswith("linux") and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Bad.TEntry", foreground=ERROR_FG, fieldbackground="#fde8ec")
        style.configure("Bad.TCombobox", foreground=ERROR_FG, fieldbackground="#fde8ec")
        style.configure("Toolbar.TFrame", padding=(6, 4))

    def _layout(self) -> None:
        bar = ttk.Frame(self.master, style="Toolbar.TFrame")
        bar.pack(fill="x")
        self.new_btn = ttk.Menubutton(bar, text="New")
        menu = tk.Menu(self.new_btn, tearoff=False)
        menu.add_command(label="Blank config", command=self.new_blank)
        examples = bundled_examples()
        if examples:
            menu.add_separator()
            for name, text in examples.items():
                menu.add_command(label=f"From example: {name}",
                                 command=lambda n=name, t=text: self.new_from_text(t, n))
        self.new_btn["menu"] = menu
        self.new_btn.pack(side="left")
        for text, cmd in (("Open...", self.open), ("Save", self.save), ("Save As...", self.save_as)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        c1 = ttk.Checkbutton(bar, text="Show only set fields", variable=self.only_set,
                             command=self.rerender)
        c1.pack(side="left")
        Tooltip(c1, "Hide fields still at their default, to see what this config changes.")
        c2 = ttk.Checkbutton(bar, text="Write default values", variable=self.with_defaults,
                             command=self.on_change)
        c2.pack(side="left", padx=8)
        Tooltip(c2, "Also write settings left at their default. Off keeps the file to the "
                    "choices that matter.")
        ttk.Button(bar, text="Expand all", command=lambda: self._set_all_collapsed(False)).pack(
            side="right", padx=2)
        ttk.Button(bar, text="Collapse all", command=lambda: self._set_all_collapsed(True)).pack(
            side="right", padx=2)

        vpane = ttk.PanedWindow(self.master, orient="vertical")
        vpane.pack(fill="both", expand=True)
        def place_sashes(_e=None):
            vpane.sashpos(0, max(200, vpane.winfo_height() - 130))
            hpane.sashpos(0, int(hpane.winfo_width() * 0.6))
            self.master.unbind("<Map>")
        self.master.bind("<Map>", lambda e: self.master.after(50, place_sashes))
        hpane = ttk.PanedWindow(vpane, orient="horizontal")
        vpane.add(hpane, weight=8)

        self.scroll = ScrollFrame(hpane, padding=(8, 4))
        hpane.add(self.scroll, weight=5)
        self.view = FormView(self, self.scroll.inner)

        right = ttk.Frame(hpane, padding=(6, 4))
        hpane.add(right, weight=3)
        ttk.Label(right, text="YAML preview (what Save writes)").pack(anchor="w")
        self.preview = tk.Text(right, wrap="none", font=("TkFixedFont", 9), state="disabled",
                               background="#fafafa", relief="flat")
        pv = ttk.Scrollbar(right, orient="vertical", command=self.preview.yview)
        ph = ttk.Scrollbar(right, orient="horizontal", command=self.preview.xview)
        self.preview.configure(yscrollcommand=pv.set, xscrollcommand=ph.set)
        pv.pack(side="right", fill="y")
        ph.pack(side="bottom", fill="x")
        self.preview.pack(fill="both", expand=True)

        bottom = ttk.Frame(vpane, padding=(8, 2))
        vpane.add(bottom, weight=1)
        self.problem_label = ttk.Label(bottom, text="Problems")
        self.problem_label.pack(anchor="w")
        self.problem_list = tk.Listbox(bottom, height=4, activestyle="none")
        self.problem_list.pack(fill="both", expand=True)
        self.problem_list.bind("<Double-Button-1>", self._jump_to_problem)
        self.status = ttk.Label(self.master, anchor="w", padding=(8, 2))
        self.status.pack(fill="x")

    def _wheel(self, event):
        widget = event.widget
        if isinstance(widget, (tk.Text, tk.Listbox)) or (
                isinstance(widget, str)):
            return None
        return self.scroll.scroll(event)

    # -- form lifecycle ----------------------------------------------------

    def _set_form(self, config: dict | None, path: Path | None, note: str = "") -> None:
        self.form = build_form(self.schema)
        if config:
            self.form.load(config)
        self.path = path
        self.view.collapsed.clear()
        self.view.default_collapsed = None
        self.rerender()
        self.dirty = False
        self._update_title()
        self.set_status(note)

    def rerender(self) -> None:
        """Rebuild the form widgets from the model."""
        self.view.render(self.form)
        self.on_change(mark_dirty=False)

    def _set_all_collapsed(self, value: bool) -> None:
        self.view.collapsed.clear()
        self.view.default_collapsed = value
        self.view.render(self.form)

    def new_blank(self) -> None:
        """Start an empty config."""
        if not self._confirm_discard():
            return
        self._set_form(None, None, "New config. Pick an agent type and add model branches, "
                                   "or start from an example (New menu).")

    def new_from_text(self, text: str, name: str) -> None:
        """Start a new config from a bundled example."""
        if not self._confirm_discard():
            return
        self._set_form(yaml.safe_load(text), None, f"New config from example {name}.")
        self.dirty = True
        self._update_title()

    def open(self, path: str | None = None) -> None:
        """Open a config file (asks for one when ``path`` is None)."""
        if not self._confirm_discard():
            return
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.master, title="Open PhoenX config",
                filetypes=[("YAML config", "*.yml *.yaml"), ("All files", "*.*")])
            if not path:
                return
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
            config = yaml.safe_load(text)
        except (OSError, yaml.YAMLError) as exc:
            messagebox.showerror("Can't open", f"{p}\n\n{exc}", parent=self.master)
            return
        if not isinstance(config, dict):
            messagebox.showerror("Can't open", f"{p} is not a YAML mapping.", parent=self.master)
            return
        note = f"Opened {p}."
        if "&" in text or any(line.lstrip().startswith("#") for line in text.splitlines()):
            note += (" Note: comments and YAML anchors (&name / *name) are not kept when "
                     "saving; anchored values are written out in full.")
        self._set_form(config, p, note)

    def current_config(self) -> dict:
        """The config the form currently describes."""
        value = self.form.value(include_defaults=self.with_defaults.get())
        return value if value is not UNSET else {}

    def save(self) -> bool:
        """Save to the current file, or ask for one."""
        if self.path is None:
            return self.save_as()
        return self._write(self.path)

    def save_as(self) -> bool:
        """Ask for a file name and save."""
        initial = self.path.name if self.path else "config.yml"
        path = filedialog.asksaveasfilename(
            parent=self.master, title="Save PhoenX config", defaultextension=".yml",
            initialfile=initial, initialdir=str(self.path.parent) if self.path else None,
            filetypes=[("YAML config", "*.yml *.yaml"), ("All files", "*.*")])
        if not path:
            return False
        return self._write(Path(path))

    def _write(self, path: Path) -> bool:
        self.validate_now()
        if self.problems:
            ok = messagebox.askyesno(
                "Save with problems?",
                f"This config has {len(self.problems)} problem(s), listed at the bottom of "
                "the window. PhoenX will refuse to train it until they are fixed.\n\n"
                "Save anyway?", parent=self.master)
            if not ok:
                return False
        try:
            path.write_text(dump_config(self.current_config()), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Can't save", f"{path}\n\n{exc}", parent=self.master)
            return False
        self.path = path
        self.dirty = False
        self._update_title()
        self.set_status(f"Saved {path}.")
        return True

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes", "Save changes to the current config first?", parent=self.master)
        if answer is None:
            return False
        if answer:
            return self.save()
        return True

    def quit(self) -> None:
        """Close the window, offering to save unsaved changes."""
        if self._confirm_discard():
            self.master.destroy()

    # -- change tracking, preview and validation ---------------------------

    def on_change(self, mark_dirty: bool = True) -> None:
        """Mark the form dirty and schedule a preview/validation refresh."""
        if mark_dirty and not self.dirty:
            self.dirty = True
            self._update_title()
        if self._pending is not None:
            self.master.after_cancel(self._pending)
        self._pending = self.master.after(250, self.validate_now)

    def validate_now(self) -> list[str]:
        """Refresh the YAML preview and the problems list; return the problems."""
        self._pending = None
        config = self.current_config()
        problems = self.form.problems()
        bad_paths = {p.split(":", 1)[0] for p in problems}
        for err in validate_config(config):
            # Bad input already reported; skip the schema's echo of the same field.
            if err.split(":", 1)[0] not in bad_paths:
                problems.append(err)
        try:
            resolve_n_step(copy.deepcopy(config))
        except ValueError as exc:
            problems.append(f"agent.config.N: {exc}")
        self.problems = problems

        self.problem_list.delete(0, "end")
        for p in problems:
            self.problem_list.insert("end", p)
        self.problem_label.configure(
            text=f"Problems ({len(problems)})" if problems else "Problems: none, ready to save",
            foreground=ERROR_FG if problems else "#1b7f3b")

        self.preview.configure(state="normal")
        top = self.preview.yview()[0]
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", dump_config(config))
        self.preview.yview_moveto(top)
        self.preview.configure(state="disabled")
        return problems

    def _jump_to_problem(self, _event=None) -> None:
        sel = self.problem_list.curselection()
        if not sel:
            return
        path = self.problem_list.get(sel[0]).split(":", 1)[0]
        self.view.expand_to(path)
        if self.only_set.get():
            self.only_set.set(False)
        self.view.render(self.form)
        found = self.view.widget_for(path)
        if found:
            _, widget = found
            self.scroll.see(widget)
            try:
                widget.focus_set()
            except tk.TclError:
                pass

    def _update_title(self) -> None:
        name = self.path.name if self.path else "untitled"
        self.master.title(f"{'*' if self.dirty else ''}{name} - PhoenX Config Editor")

    def set_status(self, text: str) -> None:
        """Show ``text`` in the status bar."""
        self.status.configure(text=text)

    def run(self) -> None:
        """Enter the Tk main loop."""
        self.master.mainloop()


def main(path: str | None = None) -> int:
    """Open the editor window (optionally on ``path``) and block until closed."""
    root = tk.Tk()
    ConfigEditor(root, path).run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
