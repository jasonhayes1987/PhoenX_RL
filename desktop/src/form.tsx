// The config form, drawn from the session's snapshot of the Python form
// model. Every layout rule here mirrors the Tkinter FormView it replaced
// (phoenx/config_gui.py at 549e5d0):
//
// * The root object is not a section: its fields are listed directly.
// * Objects, tagged unions, lists, maps and optional fields are collapsible
//   sections. A section starts collapsed when its path has 3 or more parts,
//   until the user toggles it or uses Expand all / Collapse all.
// * Choosing a type, adding a list item or map entry, or enabling an
//   optional section expands that section.
// * List items are not sections: each is a box with its number, move and
//   remove buttons, and (for tagged items such as layers) the type dropdown
//   with the type's fields right below it, or on the same row when there are
//   at most two plain fields.
// * Scalars are a label (with " *" when required), an input, and a grey
//   "default X" hint. Hovering shows the description and the default.
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react"

export type Field = {
  path: string
  name: string
  kind: string
  description: string
  required: boolean
  text: string
  choices: string[]
  editable_choices: boolean
  extra_text: string
  enabled: boolean
  tag: string
  set: boolean
  children: Field[]
  hidden?: boolean
  nullable?: boolean
  pinned?: boolean
  /** Scalars: the default as text, or null when there is none. */
  default?: string | null
  /** Scalars: the text does not parse as the field's type. */
  bad?: boolean
  /** Scalars: a dropdown (enum, or a field with suggested values). */
  combo?: boolean
  /** Objects and tagged unions: the section accepts keys the form has no field for. */
  open?: boolean
  /** Tagged unions: path of the chosen type's fields (`path.config` or `path`). */
  variant_path?: string
  /** Optional sections: the kind of the section inside. */
  inner_kind?: string
  /** Maps: keys are dotted config paths (sweep overrides and search spaces). */
  path_keys?: boolean
}

export type Editor = {
  noun: string
  path: string | null
  dirty: boolean
  generation?: number
  extra_errors?: string[]
  /** A path to expand to and scroll to once (e.g. a field just added to a sweep). */
  reveal?: string
  title: string
  status: string
  only_set: boolean
  with_defaults: boolean
  problems: string[]
  problem_summary: string
  yaml: string
  form: Field
  base_note: string
  base_fields: { path: string; text: string; in_file: boolean }[]
  phases: string[]
  selected_phase: string
}

/** Sections whose path has at least this many parts start collapsed. */
export const COLLAPSE_DEPTH = 3
/** Tagged list items with at most this many plain fields are laid out on one row. */
export const INLINE_MAX_FIELDS = 4

export type Handlers = {
  onText: (path: string, text: string) => void
  onExtra: (path: string, text: string) => void
  onType: (path: string, typeName: string) => void
  onEnabled: (path: string, enabled: boolean) => void
  onList: (path: string, op: "add" | "remove" | "up" | "down", index?: number) => void
  onMap: (path: string, op: "add" | "remove", key: string) => void
  /** Right-click on a field label (Config Builder only): send it to the sweep. */
  onMenu?: (path: string, how: "search" | "override") => void
}

export type Collapse = {
  /** Path -> collapsed, for sections the user (or an action) has toggled. */
  map: Record<string, boolean>
  /** null: collapse by depth; true / false: after Collapse all / Expand all. */
  all: boolean | null
}

export function isCollapsed(collapse: Collapse, path: string): boolean {
  if (path in collapse.map) return collapse.map[path]
  if (collapse.all !== null) return collapse.all
  return path.split(".").filter(Boolean).length >= COLLAPSE_DEPTH
}

/** Collapse state with every section on ``path`` (and its parents) expanded. */
export function expandTo(collapse: Collapse, path: string): Collapse {
  const parts = path.split(".")
  const map = { ...collapse.map }
  for (let i = 1; i <= parts.length; i += 1) map[parts.slice(0, i).join(".")] = false
  return { ...collapse, map }
}

type Ctx = Handlers & {
  onlySet: boolean
  collapse: Collapse
  setCollapsed: (path: string, value: boolean) => void
  extraErrors: Set<string>
  openMenu: (path: string, x: number, y: number) => void
}

const FormCtx = createContext<Ctx | null>(null)

function useForm(): Ctx {
  const ctx = useContext(FormCtx)
  if (!ctx) throw new Error("FormCtx missing")
  return ctx
}

/** "Show only set fields": hide optional fields still at their default. */
function shown(field: Field, onlySet: boolean): boolean {
  if (!onlySet || field.required) return true
  if (field.kind === "optional") return field.enabled
  return field.set
}

function tip(field: Field): string {
  const parts = [field.description || ""]
  if (field.default !== undefined && field.default !== null) parts.push(`Default: ${field.default}`)
  return parts.filter(Boolean).join("\n\n")
}

function sectionKind(field: Field): string {
  return field.kind === "optional" ? field.inner_kind || "object" : field.kind
}

const SCALAR_KINDS = new Set(["bool", "number", "int", "string", "enum", "raw"])

function isScalar(field: Field): boolean {
  return SCALAR_KINDS.has(field.kind)
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

type ScopeProps = {
  onlySet: boolean
  collapse: Collapse
  onCollapse: (next: Collapse) => void
  extraErrors?: string[]
  handlers: Handlers
  children: ReactNode
}

/**
 * The shared state every form control reads: handlers, the only-set flag,
 * which sections are open, and the right-click menu. The full form, the
 * overview cards and the model inspector each render inside one, so they all
 * edit the session's document the same way.
 */
export function FormScope({ onlySet, collapse, onCollapse, extraErrors, handlers, children }: ScopeProps) {
  const [menu, setMenu] = useState<{ path: string; x: number; y: number } | null>(null)
  const latest = useRef(collapse)
  latest.current = collapse
  const ctx: Ctx = {
    ...handlers,
    onlySet,
    collapse,
    setCollapsed: (path, value) => {
      const now = latest.current
      onCollapse({ ...now, map: { ...now.map, [path]: value } })
    },
    extraErrors: new Set(extraErrors || []),
    openMenu: (path, x, y) => setMenu({ path, x, y }),
  }
  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
    window.addEventListener("click", close)
    window.addEventListener("blur", close)
    return () => {
      window.removeEventListener("click", close)
      window.removeEventListener("blur", close)
    }
  }, [menu])
  return (
    <FormCtx.Provider value={ctx}>
      {children}
      {menu && handlers.onMenu && (
        <div className="menu" role="menu" style={{ left: menu.x, top: menu.y }}>
          <button type="button" role="menuitem" onClick={() => handlers.onMenu?.(menu.path, "search")}>
            Sweep {menu.path}
          </button>
          <button type="button" role="menuitem" onClick={() => handlers.onMenu?.(menu.path, "override")}>
            Fix this value for the whole sweep (override)
          </button>
        </div>
      )}
    </FormCtx.Provider>
  )
}

export function FormView({ form, ...scope }: Omit<ScopeProps, "children"> & { form: Field }) {
  return (
    <FormScope {...scope}>
      <div className="form">
        <ObjectBody fields={form.children} path={form.path} open={!!form.open} extraText={form.extra_text} />
      </div>
    </FormScope>
  )
}

/** A card row: friendly label, the key and default under it, the input on the right. */
export function CardField({ field, label }: { field: Field; label: string }) {
  const help = tip(field)
  return (
    <div className="card-field" data-path={field.path} title={help || undefined}>
      <span className="card-label">
        <span>{label}</span>
        <span className="card-key mono">
          {field.path.split(".").pop()}
          {field.default !== undefined && field.default !== null ? ` · default ${field.default}` : ""}
        </span>
      </span>
      <ScalarInput field={field} inline={false} help={help} />
    </div>
  )
}

// -----------------------------------------------------------------------------
// Objects and fields
// -----------------------------------------------------------------------------

function ObjectBody({
  fields,
  path,
  open,
  extraText,
}: {
  fields: Field[]
  path: string
  open: boolean
  extraText: string
}) {
  const { onlySet } = useForm()
  return (
    <>
      {fields.filter((child) => shown(child, onlySet)).map((child) => (
        <FieldNode key={child.path} field={child} />
      ))}
      {open && (extraText || !onlySet) && <ExtraRow path={path} text={extraText} />}
    </>
  )
}

export function FieldNode({ field }: { field: Field }) {
  if (field.kind === "const") return null
  if (isScalar(field)) return <ScalarRow field={field} />
  return <Section field={field} />
}

// -----------------------------------------------------------------------------
// Scalars
// -----------------------------------------------------------------------------

/** Text that follows the session but never jumps back while you type. */
function useLocalText(text: string, commit: (value: string) => void): [string, (value: string) => void] {
  const [local, setLocal] = useState(text)
  const sent = useRef(text)
  useEffect(() => {
    if (text !== sent.current) {
      sent.current = text
      setLocal(text)
    }
  }, [text])
  return [
    local,
    (value: string) => {
      sent.current = value
      setLocal(value)
      commit(value)
    },
  ]
}

export function ScalarRow({
  field,
  inline = false,
  naturalLabel = false,
}: {
  field: Field
  inline?: boolean
  naturalLabel?: boolean
}) {
  const { onMenu, openMenu } = useForm()
  const help = tip(field)
  return (
    <div className={inline ? "scalar inline" : "scalar"} data-path={field.path}>
      <span
        className={inline || naturalLabel ? "label natural" : "label"}
        title={help || undefined}
        onContextMenu={(event) => {
          if (!onMenu) return
          event.preventDefault()
          event.stopPropagation()
          openMenu(field.path, event.clientX, event.clientY)
        }}
      >
        {field.name}
        {field.required ? " *" : ""}
      </span>
      <ScalarInput field={field} inline={inline} help={help} />
      {!inline && field.default !== undefined && field.default !== null && field.kind !== "bool" && (
        <span className="hint">default {field.default}</span>
      )}
    </div>
  )
}

export function ScalarInput({ field, inline, help }: { field: Field; inline: boolean; help: string }) {
  const { onText } = useForm()
  const [local, setLocal] = useLocalText(field.text, (value) => onText(field.path, value))
  const bad = field.bad ? "bad" : ""
  if (field.kind === "bool") {
    return (
      <input
        type="checkbox"
        aria-label={field.path}
        data-focus={field.path}
        title={help || undefined}
        checked={local.trim() === "true"}
        onChange={(event) => setLocal(event.target.checked ? "true" : "false")}
      />
    )
  }
  const combo = field.combo ?? field.choices.length > 0
  if (combo && field.kind === "enum") {
    const choices = field.choices.includes(local) ? field.choices : [local, ...field.choices]
    return (
      <select
        aria-label={field.path}
        data-focus={field.path}
        title={help || undefined}
        className={`mono ${bad} ${inline ? "w-inline" : "w-combo"}`}
        value={local}
        onChange={(event) => setLocal(event.target.value)}
      >
        {choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    )
  }
  const listId = combo ? `choices-${field.path}` : undefined
  const width = inline ? (field.kind === "raw" ? "w-inline-raw" : "w-inline") : field.kind === "raw" ? "w-raw" : "w-entry"
  return (
    <>
      <input
        aria-label={field.path}
        data-focus={field.path}
        title={help || undefined}
        className={`mono ${bad} ${width}`}
        list={listId}
        value={local}
        spellCheck={false}
        onChange={(event) => setLocal(event.target.value)}
      />
      {combo && (
        <datalist id={listId}>
          {field.choices.map((choice) => (
            <option key={choice} value={choice} />
          ))}
        </datalist>
      )}
    </>
  )
}

function ExtraRow({ path, text }: { path: string; text: string }) {
  const { onExtra, extraErrors } = useForm()
  const [local, setLocal] = useLocalText(text, (value) => onExtra(path, value))
  return (
    <div className="scalar" data-path={path ? `${path}.<extra>` : "<extra>"}>
      <span
        className="label hint-label"
        title="Keys this section has that the form has no field for, as a YAML mapping, e.g. {_diag_freq: 1000}. Keys starting with _ are diagnostic knobs; anything else is probably a typo."
      >
        other keys
      </span>
      <input
        aria-label={`extra ${path || "root"}`}
        className={`mono w-raw ${extraErrors.has(path) && local.trim() ? "bad" : ""}`}
        value={local}
        spellCheck={false}
        onChange={(event) => setLocal(event.target.value)}
      />
    </div>
  )
}

// -----------------------------------------------------------------------------
// Sections
// -----------------------------------------------------------------------------

function summary(field: Field): string {
  const kind = sectionKind(field)
  if (kind === "list") return `${field.children.length} item(s)`
  if (kind === "map") return field.children.map((child) => child.name).join(", ") || "empty"
  return ""
}

export function Section({ field }: { field: Field }) {
  const ctx = useForm()
  const optional = field.kind === "optional"
  const enabled = optional ? field.enabled : true
  const open = enabled && !isCollapsed(ctx.collapse, field.path)
  const kind = sectionKind(field)
  const toggle = () => ctx.setCollapsed(field.path, !isCollapsed(ctx.collapse, field.path))
  return (
    <div className="section" data-path={field.path}>
      <div className="section-head">
        <button
          type="button"
          className="arrow"
          aria-label={`toggle ${field.name}`}
          aria-expanded={open}
          disabled={!enabled}
          onClick={toggle}
        >
          {enabled ? (open ? "−" : "+") : ""}
        </button>
        {optional ? (
          <label className="section-title" title={field.description || undefined}>
            <input
              type="checkbox"
              aria-label={`enable ${field.name}`}
              data-focus={field.path}
              checked={field.enabled}
              onChange={(event) => {
                if (event.target.checked) ctx.setCollapsed(field.path, false)
                ctx.onEnabled(field.path, event.target.checked)
              }}
            />
            {field.name}
          </label>
        ) : (
          <span
            className="section-title strong"
            title={field.description || undefined}
            data-focus={field.path}
            tabIndex={-1}
            onClick={toggle}
          >
            {field.name}
            {field.required ? " *" : ""}
          </span>
        )}
        {!open && enabled && summary(field) && <span className="hint">{summary(field)}</span>}
        <HeaderControls field={field} kind={kind} />
      </div>
      {open && (
        <div className="section-body">
          <SectionBody field={field} kind={kind} />
        </div>
      )}
    </div>
  )
}

function HeaderControls({ field, kind }: { field: Field; kind: string }) {
  const ctx = useForm()
  const [name, setName] = useState("")
  if (kind === "tagged") {
    return (
      <TypeSelect
        field={field}
        onChange={(value) => {
          ctx.setCollapsed(field.path, false)
          ctx.onType(field.path, value)
        }}
      />
    )
  }
  if (kind === "list") {
    return (
      <button
        type="button"
        className="add"
        onClick={() => {
          ctx.setCollapsed(field.path, false)
          ctx.onList(field.path, "add")
        }}
      >
        + Add
      </button>
    )
  }
  if (kind === "map") {
    const add = () => {
      ctx.setCollapsed(field.path, false)
      ctx.onMap(field.path, "add", name)
      setName("")
    }
    return (
      <span className="map-add">
        <input
          aria-label={`name ${field.path}`}
          className={`mono ${field.path_keys ? "w-path" : "w-name"}`}
          title={
            field.path_keys
              ? "Dotted path into the base config, e.g. agent.config.discount (list items by index: ...layer_config.0.params.units)."
              : "Name for the new entry (for branches: policy, value, critic, ...)."
          }
          value={name}
          spellCheck={false}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") add()
          }}
        />
        <button type="button" className="add" onClick={add}>
          + Add
        </button>
      </span>
    )
  }
  return null
}

function TypeSelect({ field, onChange }: { field: Field; onChange: (value: string) => void }) {
  const choices = field.choices.includes(field.text) ? field.choices : [field.text, ...field.choices]
  return (
    <select
      aria-label={`${field.path || field.name} ${field.tag}`}
      data-focus={field.path ? `${field.path}.${field.tag}` : field.tag}
      className="mono type"
      title="Choose the type; the fields below change to match."
      value={field.text}
      onChange={(event) => onChange(event.target.value)}
    >
      {choices.map((choice) => (
        <option key={choice} value={choice}>
          {choice}
        </option>
      ))}
    </select>
  )
}

function SectionBody({ field, kind }: { field: Field; kind: string }) {
  if (kind === "object") {
    return <ObjectBody fields={field.children} path={field.path} open={!!field.open} extraText={field.extra_text} />
  }
  if (kind === "tagged") {
    return (
      <ObjectBody
        fields={field.children}
        path={field.variant_path ?? field.path}
        open={!!field.open}
        extraText={field.extra_text}
      />
    )
  }
  if (kind === "list") return <ListItems field={field} />
  if (kind === "map") return <MapEntries field={field} />
  return null
}

// -----------------------------------------------------------------------------
// Lists and maps
// -----------------------------------------------------------------------------

function ListItems({ field }: { field: Field }) {
  const { onList } = useForm()
  if (field.children.length === 0) return <p className="hint">(empty: use + Add)</p>
  return (
    <>
      {field.children.map((item, index) => (
        <div className="list-item" key={item.path} data-path={item.path}>
          <div className="item-bar">
            <span className="item-no">{index + 1}.</span>
            <button type="button" title="Move up" aria-label={`move ${item.path} up`} onClick={() => onList(field.path, "up", index)}>
              ^
            </button>
            <button type="button" title="Move down" aria-label={`move ${item.path} down`} onClick={() => onList(field.path, "down", index)}>
              v
            </button>
            <button type="button" title="Remove" aria-label={`remove ${item.path}`} onClick={() => onList(field.path, "remove", index)}>
              x
            </button>
          </div>
          <div className="item-content">
            <ListItem item={item} />
          </div>
        </div>
      ))}
    </>
  )
}

function plainFields(item: Field): Field[] {
  return item.children.filter((child) => !child.hidden)
}

/** Few plain fields: drawn compactly beside / under the type dropdown. */
function inlineItem(item: Field): boolean {
  const kids = plainFields(item)
  return kids.length <= INLINE_MAX_FIELDS && kids.every(isScalar)
}

function ListItem({ item }: { item: Field }) {
  const ctx = useForm()
  if (item.kind === "tagged") {
    const inline = inlineItem(item)
    const sameRow = inline && plainFields(item).length <= 2
    const fields: ReactNode[] = []
    for (const child of item.children) {
      if (isScalar(child) && inline) fields.push(<ScalarRow key={child.path} field={child} inline />)
      else if (shown(child, ctx.onlySet)) fields.push(<FieldNode key={child.path} field={child} />)
    }
    if (item.extra_text) {
      fields.push(<ExtraRow key="<extra>" path={item.variant_path ?? item.path} text={item.extra_text} />)
    }
    const body = <div className={inline ? "item-fields inline" : "item-fields"}>{fields}</div>
    return (
      <>
        <div className="item-row">
          <TypeSelect field={item} onChange={(value) => ctx.onType(item.path, value)} />
          {sameRow && body}
        </div>
        {!sameRow && body}
      </>
    )
  }
  if (item.kind === "object") {
    return <ObjectBody fields={item.children} path={item.path} open={!!item.open} extraText={item.extra_text} />
  }
  return <FieldNode field={item} />
}

function MapEntries({ field }: { field: Field }) {
  const { onMap } = useForm()
  if (field.children.length === 0) return <p className="hint">(empty: type a name and use + Add)</p>
  return (
    <>
      {field.children.map((entry) => (
        <div className="map-entry" key={entry.path} data-path={entry.path}>
          <button
            type="button"
            className="remove"
            title={`Remove ${entry.name}`}
            aria-label={`remove ${entry.path}`}
            onClick={() => onMap(field.path, "remove", entry.name)}
          >
            x
          </button>
          <div className="item-content">
            {isScalar(entry) ? <ScalarRow field={entry} naturalLabel /> : <Section field={entry} />}
          </div>
        </div>
      ))}
    </>
  )
}

// -----------------------------------------------------------------------------
// Navigation
// -----------------------------------------------------------------------------

/** Focus the control for ``path`` (or its nearest drawn parent) and scroll to it. */
export function focusPath(root: HTMLElement | null, path: string): boolean {
  if (!root) return false
  const parts = path.split(".")
  while (parts.length) {
    const key = parts.join(".")
    const el = root.querySelector<HTMLElement>(`[data-focus="${CSS.escape(key)}"]`)
    if (el) {
      el.scrollIntoView({ block: "center" })
      el.focus()
      return true
    }
    parts.pop()
  }
  return false
}
