"""A GUI-independent form model for PhoenX configs, built from the JSON Schema.

[build_form][phoenx.config_form.build_form] turns the schema from
[phoenx.config_schema][] into a tree of nodes that a GUI renders as widgets:

* [ScalarNode][phoenx.config_form.ScalarNode]: one value (text entry, number,
  checkbox, dropdown, or free YAML for values the form cannot structure).
* [ObjectNode][phoenx.config_form.ObjectNode]: a fixed set of named fields.
* [TaggedNode][phoenx.config_form.TaggedNode]: a ``type`` dropdown whose
  choice decides which fields appear (agents, layers, heads, normalizers).
* [ListNode][phoenx.config_form.ListNode]: an ordered list of items (layers,
  wrappers, callbacks).
* [MapNode][phoenx.config_form.MapNode]: named entries (``branches``,
  ``roots``).
* [OptionalNode][phoenx.config_form.OptionalNode]: an on/off switch around an
  optional section (a normalizer, a schedule, the renderer).

Each node can ``load`` part of a parsed YAML config and produce its
``value``. The value is *minimal* by default: a field you never set is
written only when it differs from the default the code would use anyway.
Anything that was in a loaded file stays written (it is *pinned*), even when
it equals the default, so opening and saving a file never drops a key; sweep
configs rely on that, since a sweep can only change values its base config
sets explicitly. Clearing a field un-sets it. Pass ``include_defaults=True``
to write every known default too.

Keys the form does not know (``_diag_freq``-style knobs, or typos) are kept
as each object's ``extra`` mapping, so loading and saving a file never drops
data silently; validation reports the typos.

Nothing here imports tkinter or torch, so the logic is unit-testable and
another front end could reuse it.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import yaml

#: Marker for "no value": the field is omitted from the saved config.
UNSET: Any = type("Unset", (), {"__repr__": lambda self: "UNSET", "__bool__": lambda self: False})()

_NULL = {"type": "null"}


# =============================================================================
# Schema helpers
# =============================================================================

def deref(schema: dict, root: dict) -> dict:
    """Follow ``$ref`` (local ``#/definitions/...`` refs only), keeping sibling keys."""
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema:
        target: Any = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            target = target[part]
        extra = {k: v for k, v in schema.items() if k != "$ref"}
        schema = {**target, **extra}
        seen += 1
        if seen > 50:
            raise ValueError("Circular $ref in schema")
    return schema


def split_nullable(schema: dict, root: dict) -> tuple[dict, bool]:
    """Return ``(inner_schema, nullable)`` for ``X | null`` style schemas.

    The outer ``description`` / ``default`` are carried onto the inner schema.
    """
    s = deref(schema, root)
    meta = {k: s[k] for k in ("description", "default", "x-hidden") if k in s}
    if "anyOf" in s:
        opts = [deref(o, root) for o in s["anyOf"]]
        non_null = [o for o in opts if o != _NULL]
        if len(non_null) == 1 and len(opts) == 2:
            inner = {**non_null[0], **meta}
            return inner, True
    t = s.get("type")
    if isinstance(t, list) and "null" in t:
        rest = [x for x in t if x != "null"]
        inner = {**s, "type": rest[0] if len(rest) == 1 else rest}
        return inner, True
    if "enum" in s and None in s["enum"]:
        return {**s, "enum": [v for v in s["enum"] if v is not None]}, True
    return s, False


def is_tagged(schema: dict) -> bool:
    """True for ``{type: <choice>, ...}`` unions whose fields depend on ``type``."""
    return "x-tagged" in schema or (
        "enum" in schema.get("properties", {}).get("type", {}) and "allOf" in schema)


def kind_of(schema: dict, root: dict) -> str:
    """Widget kind for an already de-nulled schema.

    One of ``enum``, ``bool``, ``int``, ``number``, ``string``, ``const``,
    ``tagged``, ``object``, ``map``, ``list`` or ``raw`` (free YAML).
    """
    s = deref(schema, root)
    if "const" in s:
        return "const"
    if "enum" in s:
        return "enum"
    if is_tagged(s):
        return "tagged"
    t = s.get("type")
    if t == "boolean":
        return "bool"
    if t == "integer":
        return "int"
    if t == "number":
        return "number"
    if t == "string":
        return "string"
    if t == "object" or "properties" in s:
        if s.get("properties"):
            return "object"
        add = s.get("additionalProperties")
        if isinstance(add, dict) and add:
            return "map"
        return "raw"
    if t == "array":
        items = s.get("items")
        if isinstance(items, dict) and items:
            item, _ = split_nullable(items, root)
            if kind_of(item, root) in ("object", "tagged"):
                return "list"
    return "raw"


def _fmt(value: Any) -> str:
    """Text shown in an entry for ``value`` (inverse of the scalar parsers)."""
    if value is UNSET:
        return ""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return ".nan"
        if value in (float("inf"), float("-inf")):
            return ".inf" if value > 0 else "-.inf"
        return repr(value)
    if isinstance(value, (int, str)):
        return str(value)
    return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()


# =============================================================================
# Nodes
# =============================================================================

class Node:
    """Base class: one field of the form.

    Attributes:
        name: Key under its parent (or a display label for list items).
        schema: De-nulled, de-referenced schema for this field.
        required: Whether the parent requires the key.
        nullable: Whether ``null`` is an accepted value.
        description: Help text from the code's docstrings.
        default: Default value, or ``UNSET`` when there is none.
    """

    kind = "node"

    def __init__(self, name: str, schema: dict, root: dict, required: bool = False,
                 nullable: bool = False):
        """Create the node / widget; see the class docstring."""
        self.name = name
        self.schema = schema
        self.root = root
        self.required = required
        self.nullable = nullable
        self.description = schema.get("description", "")
        self.default = schema.get("default", UNSET)
        self.hidden = bool(schema.get("x-hidden"))
        #: Written even when equal to the default (set by ``load``, or by
        #: the GUI when a field must be explicit, e.g. to sweep it).
        self.pinned = False
        #: A loaded explicit ``null``, written back as ``null``.
        self.null_loaded = False

    def load(self, value: Any) -> None:
        """Populate this node from part of a parsed config."""
        raise NotImplementedError

    def value(self, include_defaults: bool = False) -> Any:
        """The config value this node contributes, or ``UNSET`` to omit it."""
        raise NotImplementedError

    def problems(self, path: str = "") -> list[str]:
        """Local input errors (e.g. a number field holding text)."""
        return []

    def is_set(self) -> bool:
        """True when the node would be written with ``include_defaults=False``."""
        return self.value() is not UNSET

    def walk(self) -> Iterator["Node"]:
        """Yield this node and every descendant."""
        yield self


class ScalarNode(Node):
    """A single value, edited as text (or a bool / choice).

    ``text`` holds what the user typed. An empty text means "use the
    default". ``null`` means an explicit ``None``.
    """

    def __init__(self, name, schema, root, required=False, nullable=False, kind="string"):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        self.kind = kind
        self.choices: list = list(schema.get("enum", [])) or list(schema.get("examples", []))
        self.editable_choices = "enum" not in schema  # examples: suggestions only
        self.text = "" if self.default is UNSET or self.default is None else _fmt(self.default)
        if self.kind == "bool" and self.default is UNSET:
            self.text = "false"
        if self.kind == "enum" and self.default is UNSET and self.required and self.choices:
            self.text = _fmt(self.choices[0])

    def load(self, value):
        """Populate from part of a parsed config."""
        self.text = _fmt(value)
        self.pinned = True

    def parse(self) -> Any:
        """Parse ``text`` into a value; raises ``ValueError`` on bad input."""
        text = self.text.strip()
        if text == "":
            return UNSET
        if text in ("null", "~") and self.kind != "string":
            return None
        if self.kind == "string":
            return None if text == "null" and self.nullable else self.text
        if self.kind == "enum":
            for choice in self.choices:
                if _fmt(choice) == text:
                    return choice
            raise ValueError(f"{text!r} is not one of {', '.join(map(_fmt, self.choices))}")
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"not valid YAML: {exc.problem or exc}") from None
        if self.kind == "number" and isinstance(value, str):
            # YAML 1.1 reads ``1e-06`` (no dot) and ``inf`` as strings.
            try:
                value = float(text)
            except ValueError:
                pass
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError("must be true or false")
        elif self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("must be a whole number")
        elif self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("must be a number")
        return value

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        try:
            v = self.parse()
        except ValueError:
            return self.text  # keep the bad text; validation reports it
        if v is UNSET:
            if include_defaults and self.default is not UNSET:
                return self.default
            return UNSET
        if self.kind == "bool" and self.default is UNSET and not self.required \
                and not self.pinned:
            # A checkbox starts unchecked; that is not a choice until touched.
            return UNSET
        if v == self.default and type(v) is type(self.default) and not self.required \
                and not self.pinned and not include_defaults:
            return UNSET
        return v

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        try:
            self.parse()
        except ValueError as exc:
            return [f"{path or self.name}: {exc}"]
        return []


class ConstNode(Node):
    """A fixed value (``const``); written only when it was loaded."""

    kind = "const"

    def __init__(self, name, schema, root, required=False, nullable=False):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        self.present = required

    def load(self, value):
        """Populate from part of a parsed config."""
        self.present = True

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        return self.schema["const"] if self.present else UNSET


class ObjectNode(Node):
    """A fixed set of named fields, plus an ``extra`` mapping for unknown keys."""

    kind = "object"

    def __init__(self, name, schema, root, required=False, nullable=False):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        if schema.get("x-keep-order"):
            order = list(props)
        else:
            order = [k for k in props if k in req] + [k for k in props if k not in req]
        self.children: dict[str, Node] = {
            k: make_node(k, props[k], root, required=k in req) for k in order}
        self.extra: dict = {}
        self.open = schema.get("additionalProperties", True) is not False \
            or bool(schema.get("patternProperties"))

    def load(self, value):
        """Populate from part of a parsed config."""
        if not isinstance(value, dict):
            return
        self.pinned = True
        self.extra = {}
        for k, v in value.items():
            if k in self.children:
                self.children[k].load(v)
            else:
                self.extra[k] = v

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        out = {}
        for k, child in self.children.items():
            v = child.value(include_defaults)
            if v is not UNSET:
                out[k] = v
        out.update(self.extra)
        if not out and not self.required and not self.pinned:
            return UNSET
        return out

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        out = []
        for k, child in self.children.items():
            out += child.problems(f"{path}.{k}" if path else k)
        return out

    def walk(self):
        """Yield this node and its descendants."""
        yield self
        for child in self.children.values():
            yield from child.walk()

    def visible_children(self) -> list[Node]:
        """Children a GUI should show (hidden internals only when loaded)."""
        return [c for c in self.children.values() if not c.hidden or c.is_set()]

    def set_extra_text(self, text: str) -> None:
        """Replace ``extra`` from YAML mapping text (``{_diag_freq: 10}``)."""
        text = text.strip()
        if not text:
            self.extra = {}
            return
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("extra keys must be a mapping, e.g. {_diag_freq: 10}")
        self.extra = parsed


class OptionalNode(Node):
    """An on/off switch around an optional section."""

    kind = "optional"

    def __init__(self, inner: Node):
        """Create the node / widget; see the class docstring."""
        super().__init__(inner.name, inner.schema, inner.root, False, inner.nullable)
        self.inner = inner
        inner.required = True  # once switched on, the section is written
        self.enabled = False

    def load(self, value):
        """Populate from part of a parsed config."""
        if value is None:
            self.enabled = False
            self.null_loaded = True
            return
        self.enabled = True
        self.inner.load(value)

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        if not self.enabled:
            return None if self.null_loaded else UNSET
        return self.inner.value(include_defaults)

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        return self.inner.problems(path) if self.enabled else []

    def walk(self):
        """Yield this node and its descendants."""
        yield self
        yield from self.inner.walk()


class TaggedNode(Node):
    """A ``type`` choice plus the fields that type takes.

    In *config* mode (agents, normalizers, buffers) the type's fields live
    under one key such as ``config`` or ``params``; in *flat* mode (heads)
    they sit beside ``type``. Field values are cached per type, so switching
    the type and back does not lose what was typed.
    """

    kind = "tagged"

    def __init__(self, name, schema, root, required=False, nullable=False):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        self.config_key = schema.get("x-tagged", {}).get("config_key", "config")
        #: Key holding the variant's name: ``type`` (agents, layers, ...) or
        #: ``dist`` (sweep search specs).
        self.tag = schema.get("x-tagged", {}).get("tag", "type")
        type_schema = schema["properties"][self.tag]
        self.type_node = ScalarNode("type", type_schema, root, required=True, kind="enum")
        self.choices = list(type_schema["enum"])
        self.variant_schemas: dict[str, dict] = {}
        for branch in schema.get("allOf", []):
            const = branch.get("if", {}).get("properties", {}).get(self.tag, {}).get("const")
            if const is None:
                continue
            then = branch.get("then", {})
            if self.config_key is not None:
                self.variant_schemas[const] = deref(then["properties"][self.config_key], root)
            else:
                self.variant_schemas[const] = {
                    "type": "object",
                    "properties": then.get("properties", {}),
                    "required": then.get("required", []),
                    "additionalProperties": False,
                }
        self._variants: dict[str, ObjectNode] = {}

    @property
    def type_value(self) -> str | None:
        """The selected type, or ``None`` when the text is not a known type."""
        try:
            v = self.type_node.parse()
        except ValueError:
            return None
        return v if v in self.variant_schemas else None

    def set_type(self, type_name: str) -> None:
        """Select a type (the GUI's dropdown calls this)."""
        self.type_node.text = type_name

    @property
    def variant(self) -> ObjectNode | None:
        """Field node for the selected type (created on first use)."""
        t = self.type_value
        if t is None:
            return None
        if t not in self._variants:
            self._variants[t] = ObjectNode(
                self.config_key or self.name, self.variant_schemas[t], self.root, required=True)
        return self._variants[t]

    def load(self, value):
        """Populate from part of a parsed config."""
        if not isinstance(value, dict):
            return
        if self.tag in value:
            self.type_node.load(value[self.tag])
        variant = self.variant
        if variant is None:
            return
        if self.config_key is not None:
            variant.load(value.get(self.config_key) or {})
        else:
            variant.load({k: v for k, v in value.items() if k != self.tag})

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        t = self.type_node.value(include_defaults=True)
        if t is UNSET:
            return UNSET
        variant = self.variant
        out: dict = {self.tag: t}
        if variant is None:
            return out
        body = variant.value(include_defaults)
        if self.config_key is not None:
            if body is not UNSET and (body or self.config_key != "params"):
                out[self.config_key] = body
            elif self.config_key != "params":
                # Factories index ``spec["config"]`` directly; layers use
                # ``params or {}``, so an empty params block is just noise.
                out[self.config_key] = {}
        elif body is not UNSET:
            out.update(body)
        return out

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        out = self.type_node.problems(f"{path}.{self.tag}" if path else self.tag)
        if self.variant is not None:
            sub = f"{path}.{self.config_key}" if self.config_key else path
            out += self.variant.problems(sub)
        return out

    def walk(self):
        """Yield this node and its descendants."""
        yield self
        yield self.type_node
        if self.variant is not None:
            yield from self.variant.walk()


class ListNode(Node):
    """An ordered list of structured items."""

    kind = "list"

    def __init__(self, name, schema, root, required=False, nullable=False):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        self.item_schema = schema["items"]
        self.items: list[Node] = []
        if isinstance(self.default, list) and self.default:
            self.load(self.default)
            self.pinned = False  # a default is not something the user set

    def new_item(self) -> Node:
        """Create (but do not append) an empty item."""
        return make_node(f"{self.name}[]", self.item_schema, self.root, required=True)

    def add(self, index: int | None = None) -> Node:
        """Append (or insert at ``index``) a new item and return it."""
        item = self.new_item()
        self.items.insert(len(self.items) if index is None else index, item)
        return item

    def remove(self, index: int) -> None:
        """Delete the item at ``index``."""
        del self.items[index]

    def move(self, index: int, delta: int) -> None:
        """Move an item up (``-1``) or down (``+1``)."""
        j = index + delta
        if 0 <= j < len(self.items):
            self.items[index], self.items[j] = self.items[j], self.items[index]

    def load(self, value):
        """Populate from part of a parsed config."""
        self.items = []
        self.pinned = True
        self.null_loaded = value is None
        for v in value or []:
            self.add().load(v)

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        values = [item.value(include_defaults) for item in self.items]
        values = [{} if v is UNSET else v for v in values]
        if not values and self.null_loaded:
            return None
        if not values and not self.required and not self.pinned:
            return UNSET
        if values == self.default and not include_defaults and not self.required \
                and not self.pinned:
            return UNSET
        return values

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        out = []
        for i, item in enumerate(self.items):
            out += item.problems(f"{path}.{i}" if path else str(i))
        return out

    def walk(self):
        """Yield this node and its descendants."""
        yield self
        for item in self.items:
            yield from item.walk()


class MapNode(Node):
    """Named entries with a shared schema (``branches``, ``roots``)."""

    kind = "map"

    def __init__(self, name, schema, root, required=False, nullable=False):
        """Create the node / widget; see the class docstring."""
        super().__init__(name, schema, root, required, nullable)
        self.value_schema = schema["additionalProperties"]
        self.entries: dict[str, Node] = {}

    def add(self, key: str) -> Node:
        """Add an entry called ``key`` and return its node."""
        key = key.strip()
        if not key:
            raise ValueError("name must not be empty")
        if key in self.entries:
            raise ValueError(f"{key!r} already exists")
        node = make_node(key, self.value_schema, self.root, required=True)
        self.entries[key] = node
        return node

    def remove(self, key: str) -> None:
        """Delete entry ``key``."""
        del self.entries[key]

    def rename(self, old: str, new: str) -> None:
        """Rename an entry, keeping its position."""
        new = new.strip()
        if new == old:
            return
        if not new or new in self.entries:
            raise ValueError(f"cannot rename to {new!r}")
        self.entries = {(new if k == old else k): v for k, v in self.entries.items()}
        self.entries[new].name = new

    def load(self, value):
        """Populate from part of a parsed config."""
        self.entries = {}
        self.pinned = isinstance(value, dict)
        self.null_loaded = value is None
        for k, v in (value or {}).items():
            self.add(str(k)).load(v)

    def value(self, include_defaults=False):
        """Config value to write, or ``UNSET`` to omit."""
        out = {}
        for k, node in self.entries.items():
            v = node.value(include_defaults)
            out[k] = {} if v is UNSET else v
        if not out and self.null_loaded:
            return None
        if not out and not self.required and not self.pinned:
            return UNSET
        return out

    def problems(self, path=""):
        """Input errors under this node, as ``path: message`` strings."""
        out = []
        for k, node in self.entries.items():
            out += node.problems(f"{path}.{k}" if path else k)
        return out

    def walk(self):
        """Yield this node and its descendants."""
        yield self
        for node in self.entries.values():
            yield from node.walk()


def make_node(name: str, schema: dict, root: dict, required: bool = False) -> Node:
    """Build the right node type for one schema property."""
    inner, nullable = split_nullable(schema, root)
    inner = deref(inner, root)
    kind = kind_of(inner, root)
    if kind == "const":
        return ConstNode(name, inner, root, required, nullable)
    if kind in ("enum", "bool", "int", "number", "string", "raw"):
        return ScalarNode(name, inner, root, required, nullable, kind=kind)
    if kind == "list":
        return ListNode(name, inner, root, required, nullable)
    if kind == "map":
        return MapNode(name, inner, root, required, nullable)
    cls: Callable = TaggedNode if kind == "tagged" else ObjectNode
    node = cls(name, inner, root, required, nullable)
    if not required:
        return OptionalNode(node)
    return node


def iter_fields(node: Node, path: str = "") -> Iterator[tuple[str, ScalarNode]]:
    """Yield ``(dotted_path, node)`` for every value field under ``node``.

    Paths use the sweep syntax (``agent.config.model.branches.policy.
    layer_config.0.params.units``): list items by index, a tagged section's
    fields under its config key (``config`` / ``params``), and the choosing
    key itself (``...layer_config.1.type``). Disabled optional sections are
    skipped, since they are not in the config at all.
    """
    def join(a: str, b: str) -> str:
        return f"{a}.{b}" if a else b

    if isinstance(node, ScalarNode):
        yield path, node
    elif isinstance(node, OptionalNode):
        if node.enabled:
            yield from iter_fields(node.inner, path)
    elif isinstance(node, ObjectNode):
        for child in node.children.values():
            if child.hidden and not child.is_set():
                continue
            yield from iter_fields(child, join(path, child.name))
    elif isinstance(node, TaggedNode):
        yield join(path, node.tag), node.type_node
        if node.variant is not None:
            sub = join(path, node.config_key) if node.config_key else path
            yield from iter_fields(node.variant, sub)
    elif isinstance(node, ListNode):
        for i, item in enumerate(node.items):
            yield from iter_fields(item, join(path, str(i)))
    elif isinstance(node, MapNode):
        for key, entry in node.entries.items():
            yield from iter_fields(entry, join(path, key))


def find_field(root: Node, path: str) -> ScalarNode | None:
    """The value field at ``path`` (as named by ``iter_fields``), or None."""
    for p, node in iter_fields(root):
        if p == path:
            return node
    return None


def build_form(schema: dict | None = None) -> ObjectNode:
    """Build the root form node for a whole training config.

    Args:
        schema: Config schema; defaults to [build_schema][phoenx.config_schema.build_schema].

    Returns:
        Root [ObjectNode][phoenx.config_form.ObjectNode]. Call ``load(config)``
        to fill it and ``value()`` to read the config back.
    """
    if schema is None:
        from phoenx.config_schema import build_schema
        schema = build_schema()
    return ObjectNode("config", schema, schema, required=True)


# =============================================================================
# YAML output
# =============================================================================

class _ConfigDumper(yaml.SafeDumper):
    """Block style for sections; flow style for small leaf mappings and lists."""


class _Flow(dict):
    """A mapping written on one line (``{units: 64, kernel: orthogonal}``)."""


class _FlowList(list):
    """A list of scalars written on one line."""


def _is_leaf(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _mark_flow(value: Any, depth: int = 0) -> Any:
    """Copy ``value``, tagging small leaf containers below section level as flow."""
    if isinstance(value, dict):
        out = {k: _mark_flow(v, depth + 1) for k, v in value.items()}
        leafy = all(_is_leaf(v) or v == {} or isinstance(v, (_Flow, _FlowList))
                    for v in out.values())
        if depth >= 2 and 0 < len(out) <= 4 and leafy and len(repr(out)) <= 90:
            return _Flow(out)
        return out
    if isinstance(value, list):
        out = [_mark_flow(v, depth + 1) for v in value]
        return _FlowList(out) if all(_is_leaf(v) for v in out) else out
    return value


_ConfigDumper.add_representer(dict, lambda d, data: d.represent_mapping(
    "tag:yaml.org,2002:map", data, flow_style=False))
_ConfigDumper.add_representer(_Flow, lambda d, data: d.represent_mapping(
    "tag:yaml.org,2002:map", data, flow_style=True))
_ConfigDumper.add_representer(_FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))

HEADER = ("# PhoenX training config, written by `phoenx-config gui`.\n"
          "# Settings you have not set are omitted (PhoenX uses its defaults).\n")


def dump_config(config: dict, header: str = HEADER) -> str:
    """Serialize a config dict to readable YAML (keys keep their order)."""
    return header + yaml.dump(_mark_flow(config), Dumper=_ConfigDumper, sort_keys=False,
                              allow_unicode=True, width=100)
