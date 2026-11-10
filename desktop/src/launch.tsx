// The Launch page: mode cards, the mode's fields, the command, Start / Stop,
// and live status built from the trainer's [progress] and [checkpoint] lines.
import { useEffect, useRef, useState, type ReactElement } from "react"
import { Icon } from "./overview"

export type Progress = {
  mode?: string
  unit?: string
  done?: number
  total?: number | null
  fraction: number | null
  episodes?: number | null
  avg_reward?: number | null
  elapsed?: string | null
  best?: number | null
}

export type Launch = {
  mode: string
  values: Record<string, string>
  command: string
  running: boolean
  status: string
  progress: Progress | null
  log: { text: string; tag: string }[]
  start_enabled: boolean
  stop_enabled: boolean
  stop_label: string
  open_output_enabled: boolean
  output_dir: string | null
  resume_info: string
  test_info: string
  phase_choices: string[]
  run_status: string
}

/** Native file dialogs (Tauri dialog plugin), with a typed-path fallback. */
export type Dialogs = {
  openFile: (title: string, start?: string) => Promise<string | null>
  saveFile: (title: string, start?: string) => Promise<string | null>
  openFolder: (title: string, start?: string) => Promise<string | null>
}

const MODE_ICONS = {
  train: "M5 19c4-1 6-3 7-7m0 0c3-4 6-7 9-8-1 3-4 6-8 9m-1-1-3-3m0 0-4 1-2 3 3 1m3-5 3 3m0 0-1 4-3 2-1-3",
  resume: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
  test: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
  sweep: "M3 12h4l3-8 4 16 3-8h4",
}

// The modes and fields, with the labels and hints of the Tkinter Run tab.
export const MODES: { key: keyof typeof MODE_ICONS; label: string; hint: string }[] = [
  { key: "train", label: "Train", hint: "Train a new agent from a YAML config." },
  { key: "resume", label: "Resume", hint: "Continue training a saved agent folder, optionally past its old limit." },
  { key: "test", label: "Test", hint: "Run a saved agent without training it." },
  { key: "sweep", label: "Sweep", hint: "Run a Ray Tune hyperparameter sweep." },
]

type LaunchField = {
  key: string
  label: string
  kind: "file" | "folder" | "text" | "check" | "choice" | "phase"
  hint?: string
  tip?: string
  width?: "short" | "mid" | "long"
  choices?: string[]
  use?: "config" | "sweep"
  info?: "resume_info" | "test_info"
}

const LAUNCH_FORMS: Record<string, LaunchField[]> = {
  train: [
    { key: "config", label: "Config file", kind: "file", use: "config" },
    { key: "stop_units", label: "Train until", kind: "text", width: "mid", hint: "optional: overrides schedule.stop_units (in the config's stop_unit)" },
  ],
  resume: [
    {
      key: "agent_dir",
      label: "Agent folder",
      kind: "folder",
      info: "resume_info",
      tip: "The run's save_dir: the folder holding config.json, agent/ and trainer_state.pt.",
    },
    { key: "stop_units", label: "Train until", kind: "text", width: "mid", hint: "total, in the run's stop_unit; must be past the checkpoint" },
  ],
  test: [
    { key: "test_agent_dir", label: "Agent folder", kind: "folder", info: "test_info" },
    { key: "num_episodes", label: "Episodes", kind: "text", width: "short" },
    { key: "num_envs", label: "Parallel envs", kind: "text", width: "short" },
    {
      key: "render_mode",
      label: "Render mode",
      kind: "choice",
      choices: ["human", "rgb_array", "headless"],
      hint: "human opens a window (Isaac: anything but headless opens the GUI)",
    },
    { key: "env", label: "Env override", kind: "text", width: "long", hint: "optional: env id / Isaac cfg to evaluate on instead" },
    { key: "seed", label: "Seed", kind: "text", width: "short", hint: "optional" },
  ],
  sweep: [
    { key: "sweep", label: "Sweep file", kind: "file", use: "sweep" },
    { key: "validate_only", label: "Validate only", kind: "check", hint: "sample a few trials per phase and print them; no Ray, no training" },
    { key: "num_samples", label: "Samples", kind: "text", width: "short", hint: "optional: overrides every phase" },
    { key: "max_concurrent", label: "Concurrent trials", kind: "text", width: "short", hint: "optional" },
    { key: "name", label: "Sweep name", kind: "text", width: "mid", hint: "optional: results folder name" },
    { key: "storage_path", label: "Results folder", kind: "folder", tip: "Ray Tune storage root (default ./ray_results)." },
    { key: "from_phase", label: "Start at phase", kind: "phase", hint: "optional: skip earlier phases (they must have finished)" },
    { key: "resume", label: "Resume interrupted phase", kind: "check" },
  ],
}

function LaunchText({ value, label, className, onCommit }: { value: string; label: string; className?: string; onCommit: (value: string) => void }) {
  const [local, setLocal] = useState(value)
  const sent = useRef(value)
  useEffect(() => {
    if (value !== sent.current) {
      sent.current = value
      setLocal(value)
    }
  }, [value])
  return (
    <input
      aria-label={label}
      className={`mono ${className || ""}`}
      value={local}
      spellCheck={false}
      onChange={(event) => {
        sent.current = event.target.value
        setLocal(event.target.value)
        onCommit(event.target.value)
      }}
    />
  )
}

const fmt = (n: number | null | undefined) => (n === null || n === undefined ? "—" : n.toLocaleString("en-US"))

function tiles(launch: Launch) {
  const p = launch.progress
  const unit = p?.unit || "timestep"
  const pct = p?.fraction !== null && p?.fraction !== undefined ? ` · ${Math.round(p.fraction * 100)}%` : ""
  return [
    { k: (unit === "episode" ? "EPISODE" : "TIMESTEP"), v: fmt(p?.done), sub: p?.total ? `of ${fmt(p.total)}${pct}` : launch.running ? "waiting for the first report" : "no run yet", tone: "" },
    { k: "EPISODES", v: fmt(p?.episodes), sub: p?.mode === "test" ? "evaluated" : "since start", tone: "" },
    {
      k: "AVG REWARD",
      v: p?.avg_reward === null || p?.avg_reward === undefined ? "—" : p.avg_reward.toFixed(1),
      sub: p?.best !== null && p?.best !== undefined ? `best ${p.best.toFixed(1)} · saved` : "best not saved yet",
      tone: p?.best !== null && p?.best !== undefined ? "good" : "",
    },
    { k: "ELAPSED", v: p?.elapsed || "—", sub: launch.run_status.toLowerCase(), tone: launch.running ? "live" : "" },
  ]
}

export function LaunchPage({
  launch,
  dialogs,
  onRun,
}: {
  launch: Launch
  dialogs: Dialogs
  onRun: (method: string, params?: Record<string, unknown>, opts?: { merge?: string }) => Promise<void>
}) {
  const logRef = useRef<HTMLDivElement>(null)
  const atBottom = useRef(true)
  const set = (key: string, value: string) => void onRun("launch.set", { key, value }, { merge: `launch:${key}` })

  useEffect(() => {
    const el = logRef.current
    if (el && atBottom.current) el.scrollTop = el.scrollHeight
  }, [launch.log.length, launch.log[launch.log.length - 1]?.text])

  async function browse(field: { key: string; label: string; kind: string }) {
    const start = launch.values[field.key] || launch.values.cwd || undefined
    const chosen = field.kind === "folder" ? await dialogs.openFolder(field.label, start) : await dialogs.openFile(field.label, start)
    if (chosen) set(field.key, chosen)
  }

  function row(field: LaunchField): ReactElement {
    const value = launch.values[field.key] || ""
    if (field.kind === "check") {
      return (
        <div className="run-row" key={field.key}>
          <span className="run-label" />
          <label className="check">
            <input type="checkbox" aria-label={field.label} checked={value === "1"} onChange={(event) => set(field.key, event.target.checked ? "1" : "0")} />
            {field.label}
          </label>
          {field.hint && <span className="hint">{field.hint}</span>}
        </div>
      )
    }
    let control: ReactElement
    if (field.kind === "choice") {
      control = (
        <select aria-label={field.label} className="mono w-combo" value={value} onChange={(event) => set(field.key, event.target.value)}>
          {(field.choices || []).concat(field.choices?.includes(value) ? [] : [value]).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      )
    } else if (field.kind === "phase") {
      const choices = launch.phase_choices.includes(value) ? launch.phase_choices : [...launch.phase_choices, value]
      control = (
        <select aria-label={field.label} className="mono w-mid" value={value} onChange={(event) => set(field.key, event.target.value)}>
          {choices.map((name) => (
            <option key={name} value={name}>
              {name || "(first phase)"}
            </option>
          ))}
        </select>
      )
    } else {
      const wide = field.kind === "file" || field.kind === "folder"
      control = (
        <LaunchText
          value={value}
          label={field.label}
          className={wide ? "grow-input" : field.width === "short" ? "w-short" : field.width === "long" ? "w-long" : "w-mid"}
          onCommit={(text) => set(field.key, text)}
        />
      )
    }
    return (
      <div key={field.key}>
        <div className="run-row">
          <span className="run-label" title={field.tip}>
            {field.label}
          </span>
          {control}
          {(field.kind === "file" || field.kind === "folder") && (
            <button type="button" onClick={() => void browse(field)}>
              Browse...
            </button>
          )}
          {field.use && (
            <button type="button" onClick={() => void onRun("launch.use_editor", { which: field.use })}>
              {field.use === "config" ? "Use editor's config" : "Use editor's sweep"}
            </button>
          )}
          {field.hint && <span className="hint">{field.hint}</span>}
        </div>
        {field.info && launch[field.info] && <div className="run-info">{launch[field.info]}</div>}
      </div>
    )
  }

  const fraction = launch.progress?.fraction
  return (
    <section className="launch">
      <div className="mode-cards" role="radiogroup" aria-label="Mode">
        {MODES.map((item) => (
          <label key={item.key} className={launch.mode === item.key ? "mode-card on" : "mode-card"}>
            <input
              type="radio"
              name="mode"
              aria-label={item.label}
              checked={launch.mode === item.key}
              onChange={() => void onRun("launch.set_mode", { mode: item.key })}
            />
            <Icon d={MODE_ICONS[item.key]} size={26} />
            <span className="mode-name">{item.label}</span>
            <span className="mode-hint">{item.hint}</span>
          </label>
        ))}
      </div>

      <section className="card run-card" aria-label="Command">
        <div className="run-form">{(LAUNCH_FORMS[launch.mode] || []).map(row)}</div>
        <div className="run-form common">
          {row({
            key: "cwd",
            label: "Working folder",
            kind: "folder",
            tip: "Folder the command runs in. Relative paths in configs (save_dir, base_config) resolve against it, just as in a terminal.",
          })}
          <div className="run-row">
            <span className="run-label">Log level</span>
            <select aria-label="Log level" className="mono w-short" value={launch.values.log_level || ""} onChange={(event) => set("log_level", event.target.value)}>
              {["", "DEBUG", "INFO", "WARNING", "ERROR"].map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
            <span className="hint">(empty = the config's own log_level)</span>
          </div>
        </div>
        <div className="run-row">
          <span className="run-label">Command</span>
          <input aria-label="Command" className="mono grow-input command" readOnly value={launch.command} />
          <button type="button" onClick={() => void navigator.clipboard.writeText(launch.command)}>
            Copy
          </button>
        </div>
        <div className="run-controls">
          <button
            type="button"
            className="primary big"
            data-tone={launch.start_enabled ? "live" : undefined}
            disabled={!launch.start_enabled}
            onClick={() => void onRun("launch.start")}
          >
            <Icon d="M7 4v16l13-8z" size={16} />
            Start
          </button>
          <button
            type="button"
            className="big"
            disabled={!launch.stop_enabled}
            onClick={() => void onRun(launch.stop_label === "Kill" ? "launch.kill" : "launch.stop")}
          >
            {launch.stop_label}
          </button>
          <button
            type="button"
            className="big"
            disabled={!launch.open_output_enabled}
            onClick={() => launch.output_dir && void onRun("open_folder", { path: launch.output_dir })}
          >
            Open output folder
          </button>
        </div>
      </section>

      <section className={launch.running ? "card live-card running" : "card live-card"} aria-label="Live status">
        <div className="card-head">
          <h2>Live status</h2>
          <span className="run-status">{launch.status}</span>
        </div>
        <div className="tiles">
          {tiles(launch).map((tile) => (
            <div key={tile.k} className="tile">
              <div className="tile-k">{tile.k}</div>
              <div className="tile-v mono">{tile.v}</div>
              <div className={`tile-sub ${tile.tone}`}>{tile.sub}</div>
            </div>
          ))}
        </div>
        <div className="bar" aria-label="Progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={fraction === null || fraction === undefined ? undefined : Math.round(fraction * 100)}>
          <div
            className={fraction === null && launch.running ? "bar-fill busy" : "bar-fill"}
            style={{ width: `${fraction === null || fraction === undefined ? (launch.running ? 30 : 0) : Math.round(fraction * 1000) / 10}%` }}
          />
        </div>
        <div
          className="mono log"
          ref={logRef}
          onScroll={(event) => {
            const el = event.currentTarget
            atBottom.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 4
          }}
        >
          {launch.log.length === 0 && <div className="faint">The command's output appears here.</div>}
          {launch.log.map((line, index) => (
            <div key={index} data-tag={line.tag || undefined}>
              {line.text}
            </div>
          ))}
        </div>
      </section>
    </section>
  )
}
