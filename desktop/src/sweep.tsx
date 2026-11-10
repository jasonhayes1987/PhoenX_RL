// Sweep Builder overview: phase cards, the selected phase's search table and
// settings, overrides and defaults, and the base config's field picker. Reads
// the session's sweep form; edits go through the same handlers as the form.
import { useState, type ReactElement } from "react"
import { FormView, type Collapse, type Editor, type Field, type Handlers } from "./form"
import { Icon, index } from "./overview"

const child = (node: Field | undefined, name: string) => node?.children.find((c) => c.name === name)
const oneLine = (text: string) => text.replace(/\s+/g, " ").trim()

/** "0.95 – 0.999", "[64, 128]", "1 – 4": a distribution's settings in one line. */
export function rangeOf(dist: Field): string {
  const get = (name: string) => child(dist, name)?.text ?? ""
  if (get("low") || get("high")) return `${get("low")} – ${get("high")}${get("q") ? ` · q ${get("q")}` : ""}`
  if (get("lower") || get("upper")) return `${get("lower")} – ${get("upper")}${get("q") ? ` · q ${get("q")}` : ""}`
  if (get("values")) return oneLine(get("values"))
  if (get("value")) return oneLine(get("value"))
  return dist.children.map((c) => `${c.name} ${oneLine(c.text)}`).join(" · ")
}

type Row = { label: string; path: string; kind: string; dist: string; range: string; at: string }

/** "agent.config.model.branches.critic.merged_config.0.params.units" -> "critic · merged_config · units". */
export function readable(path: string): string {
  const trimmed = path.replace(/^agent\.config\.model\.branches\./, "").replace(/^agent\.config\.model\./, "").replace(/^agent\.config\./, "").replace(/^branches\./, "")
  const parts = trimmed.split(".").filter((part) => !/^\d+$/.test(part) && part !== "params")
  return parts.join(" · ") || path
}

/** Knobs written inline, e.g. "{lr: {dist: loguniform, low: 1.0e-05, high: 0.001}}". */
function inlineKnobs(text: string): { knob: string; dist: string; range: string }[] {
  const out: { knob: string; dist: string; range: string }[] = []
  for (const match of text.matchAll(/(\w+):\s*\{\s*dist:\s*(\w+)\s*,?([^{}]*)\}/g)) {
    const rest = Object.fromEntries([...match[3].matchAll(/(\w+):\s*([^,]+)/g)].map((m) => [m[1], m[2].trim()]))
    const range =
      rest.low !== undefined ? `${rest.low} – ${rest.high}` : rest.lower !== undefined ? `${rest.lower} – ${rest.upper}` : oneLine(match[3])
    out.push({ knob: match[1], dist: match[2], range })
  }
  return out
}

function rows(phase: Field): Row[] {
  const out: Row[] = []
  for (const entry of child(phase, "search_space")?.children ?? []) {
    out.push({ label: readable(entry.name), path: entry.name, kind: "value", dist: entry.text, range: rangeOf(entry), at: entry.path })
  }
  for (const entry of child(phase, "architecture")?.children ?? []) {
    const depth = child(entry, "depth")
    const blocks = oneLine(child(entry, "blocks")?.text ?? "")
    out.push({
      label: `${readable(entry.name)} · depth`,
      path: entry.name,
      kind: "architecture",
      dist: depth?.text || "fixed",
      range: `${depth ? rangeOf(depth) : ""}${blocks ? ` × ${blocks}` : ""}`,
      at: entry.path,
    })
  }
  for (const entry of child(phase, "optimizers")?.children ?? []) {
    const knobs = entry.children.length
      ? entry.children
          .filter((c) => c.kind === "tagged" || (c.kind === "optional" && c.enabled))
          .map((c) => ({ knob: c.name, dist: c.text, range: rangeOf(c) }))
      : inlineKnobs(entry.text)
    for (const knob of knobs) {
      out.push({
        label: `${readable(entry.name)} · ${knob.knob}`,
        path: `${entry.name}.${knob.knob}`,
        kind: "optimizer",
        dist: knob.dist,
        range: knob.range,
        at: entry.path,
      })
    }
  }
  return out
}

function tuneSummary(phase: Field) {
  const tune = child(phase, "tune")
  const on = !!tune?.enabled
  const alg = on ? child(child(tune, "search_alg"), "type")?.text || "random" : "random"
  const trials = on ? child(tune, "num_samples")?.text || "1" : "1"
  const sched = on && child(tune, "scheduler")?.enabled ? child(child(tune, "scheduler"), "type")?.text || "—" : "FIFO"
  const promote = child(phase, "promote")
  return {
    alg,
    trials,
    sched: sched.toUpperCase() === "ASHA" ? "ASHA" : sched,
    promote: promote?.enabled ? child(promote, "mode")?.text || "best" : "off",
  }
}

export function SweepOverview({
  editor,
  handlers,
  onRun,
  onJump,
}: {
  editor: Editor
  handlers: Handlers
  onRun: (method: string, params?: Record<string, unknown>) => Promise<void>
  onJump: (path: string) => void
}) {
  const fields = index(editor.form)
  const [collapse, setCollapse] = useState<Collapse>({ map: {}, all: null })
  const [picker, setPicker] = useState(false)
  const phasesNode = fields.get("phases")
  const phases = phasesNode?.children ?? []
  const selectedIndex = Math.max(
    0,
    phases.findIndex((p) => child(p, "name")?.text === editor.selected_phase),
  )
  const phase = phases[selectedIndex]
  const overrides = fields.get("overrides")
  const defaults = fields.get("defaults")
  const base = fields.get("base_config")
  const total = phases.reduce((sum, p) => sum + (Number(tuneSummary(p).trials) || 0), 0)

  return (
    <div className="sweep-overview">
      <div className="sweep-cols">
        <section className="phase-col" aria-label="Phases">
          <div className="col-kicker">PHASES · RUN IN ORDER</div>
          {phases.map((p, i) => {
            const t = tuneSummary(p)
            const r = rows(p)
            const name = child(p, "name")?.text || `phase ${i + 1}`
            const on = i === selectedIndex
            return (
              <button
                key={p.path}
                type="button"
                className={on ? "phase on" : "phase"}
                data-tone={on ? "live" : undefined}
                onClick={() => void onRun("sweep.select_phase", { name })}
              >
                <span className="phase-badge">{i + 1}</span>
                <span className="phase-text">
                  <span className="phase-name">{name}</span>
                  <span className="phase-meta">
                    {t.alg} · {t.trials} trials · {t.sched}
                  </span>
                  <span className="phase-what">{r.length ? `${r.length} searched: ${r.slice(0, 2).map((x) => x.label).join(", ")}${r.length > 2 ? "…" : ""}` : "nothing searched yet"}</span>
                </span>
              </button>
            )
          })}
          <button type="button" className="add-phase" onClick={() => phasesNode && handlers.onList(phasesNode.path, "add")}>
            + Add phase
          </button>

          <div className="mini-card">
            <div className="mini-title">
              OVERRIDES · EVERY TRIAL
              <button type="button" className="link" onClick={() => onJump("overrides")}>
                edit
              </button>
            </div>
            {(overrides?.children.length ?? 0) === 0 && <div className="faint">none</div>}
            {overrides?.children.map((entry) => (
              <div key={entry.path} className="kv mono" title={entry.name}>
                <span className="kv-k">{entry.name.replace("agent.config.model.branches.", "…")}</span>
                <span className="kv-v">{oneLine(entry.text)}</span>
              </div>
            ))}
          </div>

          {defaults?.enabled && (
            <div className="mini-card">
              <div className="mini-title">
                DEFAULTS · EVERY PHASE
                <button type="button" className="link" onClick={() => onJump("defaults")}>
                  edit
                </button>
              </div>
              {["metric", "mode", "resources", "max_concurrent_trials"].map((key) => {
                const field = child(defaults, key)
                return field && field.text ? (
                  <div key={key} className="kv mono">
                    <span className="kv-k">{key}</span>
                    <span className="kv-v ink">{oneLine(field.text)}</span>
                  </div>
                ) : null
              })}
            </div>
          )}
        </section>

        <section className="card phase-panel" aria-label="Selected phase">
          {phase ? (
            <>
              <div className="phase-panel-head">
                <h2>
                  Phase {selectedIndex + 1} · {child(phase, "name")?.text}
                </h2>
                <span className="faint">
                  promote: <span className="ink">{tuneSummary(phase).promote}</span>
                </span>
                <button type="button" className="link danger" onClick={() => phasesNode && handlers.onList(phasesNode.path, "remove", selectedIndex)}>
                  Remove phase
                </button>
              </div>
              <div className="tiles">
                {Object.entries({ SEARCH: tuneSummary(phase).alg, TRIALS: tuneSummary(phase).trials, SCHEDULER: tuneSummary(phase).sched, PROMOTE: tuneSummary(phase).promote }).map(
                  ([k, v]) => (
                    <div key={k} className="tile small">
                      <div className="tile-k">{k}</div>
                      <div className="tile-v mono">{v}</div>
                    </div>
                  ),
                )}
              </div>
              <div className="table-box">
                <table>
                  <colgroup>
                    <col style={{ width: "36%" }} />
                    <col style={{ width: "13%" }} />
                    <col style={{ width: "15%" }} />
                    <col style={{ width: "36%" }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th scope="col">WHAT IS SEARCHED</th>
                      <th scope="col">KIND</th>
                      <th scope="col">DISTRIBUTION</th>
                      <th scope="col">RANGE / VALUES</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows(phase).map((row) => (
                      <tr key={row.at} onDoubleClick={() => onJump(row.at)} title="Double-click to edit">
                        <td>
                          <div>{row.label}</div>
                          <div className="mono faint small ellipsis" title={row.path}>
                            {row.path}
                          </div>
                        </td>
                        <td className="faint">{row.kind}</td>
                        <td>
                          <span className="pill mono">{row.dist}</span>
                        </td>
                        <td className="mono range" title={row.range}>
                          <span className="clamp">{row.range}</span>
                        </td>
                      </tr>
                    ))}
                    {rows(phase).length === 0 && (
                      <tr>
                        <td colSpan={4} className="faint">
                          Nothing searched yet. Add fields from the base config.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="phase-actions">
                <button type="button" onClick={() => setPicker(!picker)} aria-expanded={picker}>
                  {picker ? "Hide base config fields" : "+ Add from base config"}
                </button>
                <span className="faint">or right-click a field in Config Builder → “Sweep this”.</span>
              </div>
              {picker && <BaseFields editor={editor} onRun={onRun} />}
              <div className="phase-settings">
                <div className="col-kicker">PHASE SETTINGS</div>
                <FormView
                  form={phase}
                  onlySet={editor.only_set}
                  collapse={collapse}
                  onCollapse={setCollapse}
                  extraErrors={editor.extra_errors}
                  handlers={handlers}
                />
              </div>
            </>
          ) : (
            <p className="faint">No phases yet. Use + Add phase.</p>
          )}
        </section>
      </div>

      <div className="sweep-bar">
        <span className={editor.problems.length ? "valid bad" : "valid"}>
          <Icon d={editor.problems.length ? "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" : "M20 6 9 17l-5-5"} size={16} />
          {editor.problems.length
            ? `${editor.problems.length} problem${editor.problems.length === 1 ? "" : "s"}`
            : `Valid against ${base?.text || "the base config"} · ${total} trials across ${phases.length} phase${phases.length === 1 ? "" : "s"}`}
        </span>
        <span className="grow" />
        <button
          type="button"
          title="Save, then run phoenx-tune --validate-only on the Launch page: samples a few trials per phase and prints them, without Ray or training."
          onClick={() => void onRun("sweep.handoff", { validate_only: true })}
        >
          Dry run
        </button>
        <button type="button" className="primary" title="Save the sweep and set it up on the Launch page." onClick={() => void onRun("sweep.handoff", { validate_only: false })}>
          Set up launch →
        </button>
      </div>
    </div>
  )
}

// -----------------------------------------------------------------------------
// The base config's fields (what a sweep may change)
// -----------------------------------------------------------------------------

type BaseRow = { path: string; text: string; in_file: boolean }
type TreeNode = { name: string; key: string; row?: BaseRow; kids: TreeNode[] }

function buildTree(list: BaseRow[]): TreeNode[] {
  const top: TreeNode[] = []
  const folders = new Map<string, TreeNode>()
  for (const row of list) {
    const parts = row.path.split(".")
    let level = top
    for (let i = 0; i < parts.length - 1; i += 1) {
      const key = parts.slice(0, i + 1).join(".")
      let folder = folders.get(key)
      if (!folder) {
        folder = { name: parts[i], key, kids: [] }
        folders.set(key, folder)
        level.push(folder)
      }
      level = folder.kids
    }
    level.push({ name: parts[parts.length - 1], key: row.path, row, kids: [] })
  }
  return top
}

export function BaseFields({
  editor,
  onRun,
}: {
  editor: Editor
  onRun: (method: string, params?: Record<string, unknown>) => Promise<void>
}) {
  const [filter, setFilter] = useState("")
  const [picked, setPicked] = useState("")
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [notice, setNotice] = useState(false)
  const needle = filter.trim().toLowerCase()
  const list = editor.base_fields.filter((row) => !needle || row.path.toLowerCase().includes(needle))
  const tree = buildTree(list)

  function add(how: "search" | "override") {
    if (!picked) {
      setNotice(true)
      return
    }
    void onRun("sweep.add_field", { path: picked, how })
  }

  function draw(nodes: TreeNode[], depth: number): ReactElement[] {
    const out: ReactElement[] = []
    for (const node of nodes) {
      if (node.row) {
        const row = node.row
        out.push(
          <div
            key={node.key}
            role="treeitem"
            aria-selected={picked === row.path}
            aria-label={row.path}
            className={`tree-row leaf ${row.in_file ? "" : "default"}`}
            style={{ paddingLeft: 8 + depth * 14 }}
            onClick={() => setPicked(row.path)}
            onDoubleClick={() => {
              setPicked(row.path)
              void onRun("sweep.add_field", { path: row.path, how: "search" })
            }}
          >
            <span className="tree-name">{node.name}</span>
            <span className="tree-value mono">{row.text}</span>
            <span className="tree-where">{row.in_file ? "yes" : "default"}</span>
          </div>,
        )
        continue
      }
      const isOpen = node.key in open ? open[node.key] : !!needle || depth < 1
      out.push(
        <div
          key={node.key}
          role="treeitem"
          aria-expanded={isOpen}
          className="tree-row folder"
          style={{ paddingLeft: 8 + depth * 14 }}
          onClick={() => setOpen({ ...open, [node.key]: !isOpen })}
        >
          <span className="tree-name">
            {isOpen ? "▾" : "▸"} {node.name}
          </span>
        </div>,
      )
      if (isOpen) out.push(...draw(node.kids, depth + 1))
    }
    return out
  }

  return (
    <div className="base-fields">
      <p className="base-note">{editor.base_note}</p>
      <label className="filter">
        Filter
        <input aria-label="Filter base fields" value={filter} onChange={(event) => setFilter(event.target.value)} />
      </label>
      <div className="tree-head">
        <span className="tree-name">Field</span>
        <span className="tree-value">Value</span>
        <span className="tree-where">In file</span>
      </div>
      <div className="tree" role="tree">
        {draw(tree, 0)}
      </div>
      <div className="tree-actions">
        <label>
          Phase
          <select
            aria-label="Phase"
            value={editor.selected_phase}
            onChange={(event) => void onRun("sweep.select_phase", { name: event.target.value })}
          >
            {editor.phases.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={() => add("search")}>
          Add to search space
        </button>
        <button type="button" onClick={() => add("override")}>
          Add as override
        </button>
      </div>
      {notice && (
        <div className="need" role="alertdialog" aria-label="Pick a field">
          <article>
            <h2>Pick a field</h2>
            <p>Select a field (a leaf of the tree) first.</p>
            <div className="buttons">
              <button type="button" autoFocus onClick={() => setNotice(false)}>
                OK
              </button>
            </div>
          </article>
        </div>
      )}
    </div>
  )
}
