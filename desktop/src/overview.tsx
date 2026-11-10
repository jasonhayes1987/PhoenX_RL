// Config Builder overview: the Agent / Environment / Schedule cards and the
// model architecture diagram. Everything here reads the session's form
// snapshot and edits through the same handlers as the full form, so the
// cards are shortcuts into one document, never a second copy of it.
import { useEffect, useState } from "react"
import { CardField, FormScope, ScalarInput, Section, type Collapse, type Editor, type Field, type Handlers } from "./form"

export type Cards = {
  agent_type: string
  on_policy: boolean
  agent: { path: string; label: string }[]
  schedule: { path: string; label: string }[]
  env: { path: string; label: string }[]
  normalizers: { path: string; label: string }[]
  model: string
}

export function index(root: Field): Map<string, Field> {
  const out = new Map<string, Field>()
  const walk = (node: Field) => {
    out.set(node.path, node)
    node.children.forEach(walk)
  }
  walk(root)
  return out
}

const ICONS = {
  agent: "M12 2a4 4 0 0 1 4 4v1h1a3 3 0 0 1 3 3v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4v-8a3 3 0 0 1 3-3h1V6a4 4 0 0 1 4-4zM9 13h.01M15 13h.01M9 17h6",
  env: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20",
  schedule: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 6v6l4 2",
  model: "M6 3v12M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 9a9 9 0 0 1-9 9",
}

export function Icon({ d, size = 20 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  )
}

function CardHead({ icon, title, note }: { icon: string; title: string; note?: string }) {
  return (
    <div className="card-head">
      <span className="card-icon">
        <Icon d={icon} />
      </span>
      <h2>{title}</h2>
      {note && <span className="card-note mono">{note}</span>}
    </div>
  )
}

/** A row of type buttons (agent algorithm, env backend) that set a tagged field's type. */
function Segments({ field, label, onType }: { field: Field; label: string; onType: (path: string, t: string) => void }) {
  return (
    <div className="segments" role="radiogroup" aria-label={label}>
      {field.choices.map((choice) => (
        <button
          key={choice}
          type="button"
          role="radio"
          aria-checked={field.text === choice}
          className={field.text === choice ? "seg on" : "seg"}
          onClick={() => field.text !== choice && onType(field.path, choice)}
        >
          {choice}
        </button>
      ))}
    </div>
  )
}

/** "400 → 300" from a layer list: the dense widths in order. */
function widths(list: Field | undefined): string {
  if (!list) return ""
  const units = list.children
    .filter((item) => item.text === "dense")
    .map((item) => item.children.find((child) => child.name === "units")?.text || "?")
  return units.join(" → ")
}

function branchDims(branch: Field): string {
  const lists = new Map(branch.children.map((child) => [child.name, child]))
  const main = widths(lists.get("layer_config"))
  const merged = widths(lists.get("merged_config"))
  return [main, merged].filter(Boolean).join(" | ") || "no layers"
}

type Pick = { kind: "roots" | "trunk" | "branches" } | { kind: "branch" | "root"; name: string }

export function ConfigOverview({
  editor,
  handlers,
  onJump,
}: {
  editor: Editor & { cards?: Cards | null }
  handlers: Handlers
  onJump: (path: string) => void
}) {
  const cards = editor.cards
  const fields = index(editor.form)
  const [pick, setPick] = useState<Pick>({ kind: "branch", name: "policy" })
  const [inspect, setInspect] = useState<Collapse>({ map: {}, all: false })
  const [newBranch, setNewBranch] = useState("")
  const [cardCollapse, setCardCollapse] = useState<Collapse>({ map: {}, all: false })

  const model = cards ? fields.get(cards.model) : undefined
  const part = (name: string) => model?.children.find((child) => child.name === name)
  const roots = part("roots")
  const trunk = part("trunk")
  const branches = part("branches")

  // A newly loaded file starts on its first branch.
  useEffect(() => {
    const first = branches?.children[0]
    setPick(first ? { kind: "branch", name: first.name } : { kind: "branches" })
  }, [editor.generation])

  // Keep the inspector on something that exists (a removed branch, a new agent).
  useEffect(() => {
    if (pick.kind === "branch" && branches && !branches.children.some((b) => b.name === pick.name)) {
      const first = branches.children[0]
      setPick(first ? { kind: "branch", name: first.name } : { kind: "branches" })
    }
  }, [branches?.children.map((b) => b.name).join(","), pick])

  if (!cards) return null
  const agent = fields.get("agent")
  const env = fields.get("env")
  const stop = fields.get("schedule.stop_units")
  const stopUnit = fields.get("schedule.stop_unit")
  const learnUnit = fields.get("schedule.learn_every_unit")

  const card = (list: { path: string; label: string }[]) =>
    list.map((entry) => {
      const field = fields.get(entry.path)
      return field ? <CardField key={entry.path} field={field} label={entry.label} /> : null
    })

  let inspected: Field | undefined
  let inspectTitle = ""
  if (pick.kind === "branch") {
    inspected = branches?.children.find((b) => b.name === pick.name)
    inspectTitle = pick.name
  } else if (pick.kind === "root") {
    inspected = roots?.children.find((r) => r.name === pick.name)
    inspectTitle = pick.name
  } else {
    inspected = { roots, trunk, branches }[pick.kind]
    inspectTitle = pick.kind
  }

  const node = (on: boolean, extra = "") => `node ${on ? "on" : ""} ${extra}`
  const trunkOn = !!trunk?.enabled
  const trunkLayers = trunk?.children.find((child) => child.name === "layer_config")

  return (
    <FormScope
      onlySet={false}
      collapse={cardCollapse}
      onCollapse={setCardCollapse}
      extraErrors={editor.extra_errors}
      handlers={handlers}
    >
      <div className="overview">
        <div className="card-row">
          <section className="card" aria-label="Agent">
            <CardHead icon={ICONS.agent} title="Agent" note="agent.type" />
            {agent && <Segments field={agent} label="Algorithm" onType={handlers.onType} />}
            {card(cards.agent)}
            <button type="button" className="link" onClick={() => onJump("agent")}>
              All {cards.agent_type} settings →
            </button>
          </section>

          <section className="card" aria-label="Environment">
            <CardHead icon={ICONS.env} title="Environment" note="env.type" />
            {env && <Segments field={env} label="Environment backend" onType={handlers.onType} />}
            {card(cards.env)}
            <div className="chips">
              <span className="chips-label">Normalizers</span>
              {cards.normalizers.map((entry) => {
                const field = fields.get(entry.path)
                const on = !!field?.enabled
                return (
                  <button
                    key={entry.path}
                    type="button"
                    className={on ? "chip" : "chip off"}
                    title={field?.description || undefined}
                    onClick={() => onJump(entry.path)}
                  >
                    {entry.label}: {on ? field?.text || "on" : "off"}
                  </button>
                )
              })}
            </div>
          </section>

          <section className="card" aria-label="Schedule">
            <CardHead icon={ICONS.schedule} title="Schedule" note={cards.on_policy ? "on-policy" : "off-policy"} />
            {stop && (
              <div className="stop" data-path={stop.path} title={stop.description}>
                <ScalarInput field={stop} inline={false} help={stop.description} />
                {stopUnit && <ScalarInput field={stopUnit} inline help={stopUnit.description} />}
                <span className="stop-note">stop</span>
              </div>
            )}
            {card(cards.schedule)}
            {learnUnit && <CardField field={learnUnit} label="Learn every counts" />}
          </section>
        </div>

        <section className="card model" aria-label="Model architecture">
          <CardHead icon={ICONS.model} title="Model architecture" />
          <p className="card-sub">roots → trunk → branches · click a block to edit its layers</p>
          <div className="model-row">
            <div className="diagram">
              <div className="node io">
                <span className="node-kicker">INPUT</span>
                <span className="mono">observation</span>
              </div>
              <span className="wire" />
              <div className="stack">
                {(roots?.children.length ?? 0) > 0 ? (
                  roots!.children.map((root) => (
                    <button
                      key={root.path}
                      type="button"
                      className={node(pick.kind === "root" && pick.name === root.name)}
                      onClick={() => setPick({ kind: "root", name: root.name })}
                    >
                      <span className="node-kicker">ROOT</span>
                      <span className="node-name">{root.name}</span>
                    </button>
                  ))
                ) : (
                  <button type="button" className={node(pick.kind === "roots", "empty")} onClick={() => setPick({ kind: "roots" })}>
                    <span className="node-kicker">ROOTS</span>
                    <span>none</span>
                    <span className="node-hint">+ add a root</span>
                  </button>
                )}
              </div>
              <span className="wire" />
              <button
                type="button"
                className={node(pick.kind === "trunk", trunkOn ? "" : "empty")}
                onClick={() => setPick({ kind: "trunk" })}
              >
                <span className="node-kicker">TRUNK</span>
                <span>{trunkOn ? widths(trunkLayers) || "on" : "none"}</span>
                <span className="node-hint">{trunkOn ? "shared layers" : "turn on"}</span>
              </button>
              <span className="wire" />
              <div className="stack branches">
                {branches?.children.map((branch) => (
                  <button
                    key={branch.path}
                    type="button"
                    className={node(pick.kind === "branch" && pick.name === branch.name, "branch")}
                    onClick={() => setPick({ kind: "branch", name: branch.name })}
                  >
                    <span className="branch-top">
                      <span className="node-name">{branch.name}</span>
                      <span className="node-dims">{branchDims(branch)}</span>
                    </span>
                    <span className="mono node-head">{branch.text}</span>
                  </button>
                ))}
                {branches && (
                  <span className="add-branch">
                    <input
                      aria-label="New branch name"
                      placeholder="new branch"
                      value={newBranch}
                      onChange={(event) => setNewBranch(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          handlers.onMap(branches.path, "add", newBranch)
                          setPick({ kind: "branch", name: newBranch || "policy" })
                          setNewBranch("")
                        }
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        handlers.onMap(branches.path, "add", newBranch)
                        if (newBranch) setPick({ kind: "branch", name: newBranch })
                        setNewBranch("")
                      }}
                    >
                      + Branch
                    </button>
                  </span>
                )}
              </div>
            </div>

            <div className="inspector" aria-label="Inspector">
              <div className="inspector-head">
                <span className="inspector-title">{inspectTitle}</span>
                <span className="mono inspector-path">{inspected?.path}</span>
                {pick.kind === "branch" && branches && (
                  <button
                    type="button"
                    className="link danger"
                    onClick={() => handlers.onMap(branches.path, "remove", pick.name)}
                  >
                    Remove branch
                  </button>
                )}
              </div>
              {inspected ? (
                <FormScope onlySet={false} collapse={inspect} onCollapse={setInspect} extraErrors={editor.extra_errors} handlers={handlers}>
                  <div className="form inspector-form">
                    <Section field={inspected} />
                  </div>
                </FormScope>
              ) : (
                <p className="card-sub">This model has no {inspectTitle}.</p>
              )}
            </div>
          </div>
        </section>
      </div>
    </FormScope>
  )
}
