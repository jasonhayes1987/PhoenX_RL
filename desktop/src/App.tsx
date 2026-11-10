import { useEffect, useMemo, useRef, useState } from "react"
import { logoUrl, nav, versionLabel, wordmark, type PageId } from "./brand"
import { expandTo, focusPath, FormView, type Collapse, type Editor, type Handlers } from "./form"
import { LaunchPage, type Dialogs, type Launch } from "./launch"
import { ConfigOverview, type Cards } from "./overview"
import { createQueue, type Client, type Need } from "./session"
import { SweepOverview } from "./sweep"

export type { Dialogs } from "./launch"

type Pending = { method: string; params: Record<string, unknown> }
type Failure = { title: string; message: string }
type Focus = { target: string; path: string; n: number; show: boolean }

const YAML_FILTER = [{ name: "YAML", extensions: ["yml", "yaml"] }, { name: "All files", extensions: ["*"] }]

export const promptDialogs: Dialogs = {
  openFile: async (title) => window.prompt(title) || null,
  saveFile: async (title) => window.prompt(title) || null,
  openFolder: async (title) => window.prompt(title) || null,
}

export const tauriDialogs: Dialogs = {
  async openFile(title, start) {
    const { open } = await import("@tauri-apps/plugin-dialog")
    const picked = await open({ title, defaultPath: start || undefined, filters: YAML_FILTER, multiple: false, directory: false })
    return typeof picked === "string" ? picked : null
  },
  async saveFile(title, start) {
    const { save } = await import("@tauri-apps/plugin-dialog")
    return (await save({ title, defaultPath: start || undefined, filters: YAML_FILTER })) ?? null
  },
  async openFolder(title, start) {
    const { open } = await import("@tauri-apps/plugin-dialog")
    const picked = await open({ title, defaultPath: start || undefined, directory: true, multiple: false })
    return typeof picked === "string" ? picked : null
  },
}


function isNeed(result: unknown): result is Need {
  return !!result && typeof result === "object" && "need" in (result as object)
}

function failureOf(error: unknown): Failure {
  const text = error instanceof Error ? error.message : String(error)
  const cut = text.indexOf(": ")
  if (cut > 0 && cut < 40) return { title: text.slice(0, cut), message: text.slice(cut + 2) }
  return { title: "Error", message: text }
}

export function App({
  client,
  pythonReady = true,
  onPython,
  dialogs = promptDialogs,
  marks = { logoUrl, nav, wordmark },
}: {
  client: Client
  pythonReady?: boolean
  onPython?: (path: string) => Promise<boolean>
  dialogs?: Dialogs
  marks?: { logoUrl: string; wordmark: string; nav: readonly { id: PageId; label: string; iconUrl: string }[] }
}) {
  const [ready, setReady] = useState(pythonReady)
  const [attempt, setAttempt] = useState(0)
  const [pickerError, setPickerError] = useState("")
  const [page, setPage] = useState<PageId>("config")
  const [runStatus, setRunStatus] = useState("Idle")
  const [version, setVersion] = useState(versionLabel)
  const [config, setConfig] = useState<Editor | null>(null)
  const [sweep, setSweep] = useState<Editor | null>(null)
  const [launch, setLaunch] = useState<Launch | null>(null)
  const [examples, setExamples] = useState<{ config: string[]; sweep: string[] }>({ config: [], sweep: [] })
  const [need, setNeed] = useState<Need | null>(null)
  const [pending, setPending] = useState<Pending | null>(null)
  const [failure, setFailure] = useState<Failure | null>(null)
  const [focus, setFocus] = useState<Focus | null>(null)
  const queue = useMemo(() => createQueue(client), [client])
  const pageRef = useRef(page)
  pageRef.current = page

  function apply(method: string, params: Record<string, unknown>, result: unknown) {
    if (isNeed(result)) {
      setNeed(result)
      setPending({ method, params })
      return
    }
    setNeed(null)
    setPending(null)
    const value = result as Record<string, unknown> | null
    if (!value || typeof value !== "object") return
    if (value.closed === true) {
      void client.quit?.()
      return
    }
    if (value.goto === "launch") {
      const snap = value.snapshot as Launch
      setLaunch(snap)
      setRunStatus(snap.run_status)
      setPage("launch")
      return
    }
    if ("run_status" in value && "command" in value) {
      setLaunch(value as unknown as Launch)
      setRunStatus(String(value.run_status))
      return
    }
    if (value.noun === "config") setConfig(value as unknown as Editor)
    if (value.noun === "sweep") setSweep(value as unknown as Editor)
    if ((value.noun === "config" || value.noun === "sweep") && value.reveal) {
      const target = String(value.noun)
      setPage(target as PageId)
      setFocus((now) => ({ target, path: String(value.reveal), n: (now?.n ?? 0) + 1, show: false }))
    }
  }

  /** One session call, in order. Edits to the same field are merged while one is in flight. */
  async function run(method: string, params: Record<string, unknown> = {}, opts: { merge?: string } = {}) {
    try {
      const outcome = await queue.call(method, params, opts.merge)
      if (outcome.stale) return
      apply(method, params, outcome.result)
      if (isNeed(outcome.result) && outcome.result.need === "save_as") {
        await answerSaveAs(method, params)
      }
    } catch (error) {
      setFailure(failureOf(error))
      // Re-sync the page that failed, so the screen matches the session again.
      try {
        if (method.startsWith("editor.") || method.startsWith("sweep.")) {
          const target = (params.target as string) || (method.startsWith("sweep.") ? "sweep" : "config")
          apply("editor.snapshot", {}, await queue.call("editor.snapshot", { target }))
        }
      } catch {
        // the failure dialog is already showing
      }
    }
  }

  async function answerSaveAs(method: string, params: Record<string, unknown>) {
    const target =
      (params.target as string) ||
      (method === "launch.use_editor" ? (params.which as string) : method.startsWith("sweep.") ? "sweep" : "config")
    const doc = target === "sweep" ? sweep : config
    let path: string | null = null
    try {
      path = await dialogs.saveFile(target === "sweep" ? "Save sweep as" : "Save config as", doc?.path || undefined)
    } catch (error) {
      setFailure(failureOf(error))
    }
    // Always answer the session's open question, or it refuses every later call.
    if (path) await run("editor.save_as", { target, path })
    else await run(method, { ...params, answer: "cancel" })
  }

  useEffect(() => {
    if (!ready) return
    let stop = false
    ;(async () => {
      try {
        const versionResult = (await client.call("app.version")) as { version: string }
        const app = (await client.call("app.snapshot")) as { initial_page: PageId; run_status: string }
        if (stop) return
        setVersion(versionResult.version || versionLabel)
        setRunStatus(app.run_status)
        setPage(app.initial_page === "sweep" ? "sweep" : "config")
        setConfig((await client.call("editor.snapshot", { target: "config" })) as Editor)
        setSweep((await client.call("editor.snapshot", { target: "sweep" })) as Editor)
        setLaunch((await client.call("launch.snapshot")) as Launch)
        const configNames = (await client.call("editor.examples", { target: "config" })) as { names: string[] }
        const sweepNames = (await client.call("editor.examples", { target: "sweep" })) as { names: string[] }
        if (!stop) setExamples({ config: configNames.names, sweep: sweepNames.names })
      } catch (error) {
        if (!stop) setFailure(failureOf(error))
      }
    })()
    return () => {
      stop = true
    }
  }, [client, ready])

  // The Run tab pumps the child's output every 100 ms while it runs, and
  // while an agent folder is being read.
  const polling = !!launch && (launch.running || /Reading checkpoint/.test(`${launch.resume_info} ${launch.test_info}`))
  useEffect(() => {
    if (!polling) return
    const timer = window.setInterval(() => {
      if (queue.busy()) return
      void run("launch.poll")
    }, 100)
    return () => window.clearInterval(timer)
  }, [polling])

  // The window's close button: ask the session, like the header's Close.
  useEffect(() => {
    const onClose = () => void run("app.close")
    window.addEventListener("phoenx-close", onClose)
    return () => window.removeEventListener("phoenx-close", onClose)
  })

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const current = pageRef.current
      if (current !== "config" && current !== "sweep") return
      if (event.ctrlKey && event.key.toLowerCase() === "s") {
        event.preventDefault()
        void run("editor.save", { target: current })
      }
      if (event.ctrlKey && event.key.toLowerCase() === "o") {
        event.preventDefault()
        void openFile(current)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  async function openFile(target: "config" | "sweep") {
    const doc = target === "sweep" ? sweep : config
    const path = await dialogs.openFile(target === "sweep" ? "Open sweep" : "Open config", doc?.path || undefined)
    if (path) await run("editor.open", { target, path })
  }

  if (!ready) {
    return (
      <div className="picker">
        <article>
          <h1>PhoenX Python</h1>
          <p>Choose the python.exe from the environment where PhoenX is installed.</p>
          {pickerError && <p>{pickerError}</p>}
          <form
            onSubmit={async (event) => {
              event.preventDefault()
              const data = new FormData(event.currentTarget)
              const path = String(data.get("python") || "")
              const ok = (await onPython?.(path)) ?? false
              if (ok) {
                setReady(true)
                return
              }
              const next = attempt + 1
              setAttempt(next)
              if (next >= 2) {
                await client.quit?.()
                return
              }
              setPickerError("That Python cannot import phoenx. Choose the interpreter from the PhoenX environment.")
            }}
          >
            <input name="python" aria-label="python.exe" />
            <button type="submit">Use this Python</button>
            <button
              type="button"
              onClick={() => {
                void client.quit?.()
              }}
            >
              Cancel
            </button>
          </form>
        </article>
      </div>
    )
  }

  const editor = page === "sweep" ? sweep : config
  const status = page === "launch" ? launch?.status : page === "config" || page === "sweep" ? editor?.status : version
  const pageLabel = marks.nav.find((item) => item.id === page)?.label

  const [first, ...rest] = marks.wordmark.split(" ")
  const live = runStatus === "Running" || runStatus === "Stopping"
  const subtitle =
    page === "config" || page === "sweep"
      ? editor?.title.replace(/ - PhoenX$/, "")
      : page === "launch"
        ? "The same commands you would type in a terminal."
        : ""

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <img src={marks.logoUrl} alt="" width={36} height={36} />
          <strong>
            {first} <span className="brand-accent">{rest.join(" ")}</span>
          </strong>
        </div>
        {marks.nav.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-current={page === item.id ? "page" : undefined}
            onClick={() => setPage(item.id)}
          >
            <span className="nav-icon" aria-hidden="true" style={{ maskImage: `url("${item.iconUrl}")`, WebkitMaskImage: `url("${item.iconUrl}")` }} />
            {item.label}
          </button>
        ))}
        <div className="sidebar-foot">
          <span className="session-dot" aria-hidden="true" />
          Python session
          <div className="version">{version}</div>
        </div>
      </aside>
      <header>
        <div className="page-title">
          <h1>{pageLabel}</h1>
          {subtitle && (
            <div className="page-sub">
              <span className="mono">{subtitle.replace(/^\*/, "")}</span>
              {editor?.dirty && (page === "config" || page === "sweep") && <span className="badge">edited</span>}
            </div>
          )}
        </div>
        <span className={live ? "run-pill on" : "run-pill"} data-tone={live ? "live" : undefined}>
          {runStatus}
        </span>
        <button type="button" className="close" onClick={() => void run("app.close")}>
          Close
        </button>
      </header>
      <main>
        {page === "dashboard" && <p className="empty-page">Dashboard</p>}
        {page === "settings" && <p className="empty-page">Settings</p>}
        {(page === "config" || page === "sweep") && editor && (
          <EditorPage
            key={page}
            target={page}
            editor={editor}
            examples={examples[page]}
            focus={focus && focus.target === page ? focus : null}
            onRun={run}
            onOpen={() => openFile(page)}
            onSaveAs={async () => {
              const path = await dialogs.saveFile(page === "sweep" ? "Save sweep as" : "Save config as", editor.path || undefined)
              if (path) await run("editor.save_as", { target: page, path })
            }}
            onJump={(path) => setFocus({ target: page, path, n: (focus?.n ?? 0) + 1, show: true })}
          />
        )}
        {page === "launch" && launch && <LaunchPage launch={launch} dialogs={dialogs} onRun={run} />}
      </main>
      <footer>{status}</footer>
      {need && need.need !== "save_as" && (
        <div className="need" role="dialog" aria-label={need.title}>
          <article>
            <h2>{need.title}</h2>
            <p>{need.message}</p>
            <div className="buttons">
              {need.need !== "ok" && (
                <button type="button" onClick={() => pending && run(pending.method, { ...pending.params, answer: "yes" })}>
                  Yes
                </button>
              )}
              {need.need !== "ok" && (
                <button type="button" onClick={() => pending && run(pending.method, { ...pending.params, answer: "no" })}>
                  No
                </button>
              )}
              {need.need === "yes_no_cancel" && (
                <button type="button" onClick={() => pending && run(pending.method, { ...pending.params, answer: "cancel" })}>
                  Cancel
                </button>
              )}
              {need.need === "ok" && (
                <button type="button" onClick={() => pending && run(pending.method, { ...pending.params, answer: "ok" })}>
                  OK
                </button>
              )}
            </div>
          </article>
        </div>
      )}
      {failure && (
        <div className="need" role="alertdialog" aria-label={failure.title}>
          <article>
            <h2>{failure.title}</h2>
            <p>{failure.message}</p>
            <div className="buttons">
              <button type="button" autoFocus onClick={() => setFailure(null)}>
                OK
              </button>
            </div>
          </article>
        </div>
      )}
    </div>
  )
}

// -----------------------------------------------------------------------------
// Config / Sweep editor page
// -----------------------------------------------------------------------------

type Tab = "overview" | "all" | "yaml"

function EditorPage({
  target,
  editor,
  examples,
  focus,
  onRun,
  onOpen,
  onSaveAs,
  onJump,
}: {
  target: "config" | "sweep"
  editor: Editor & { cards?: Cards | null }
  examples: string[]
  focus: Focus | null
  onRun: (method: string, params?: Record<string, unknown>, opts?: { merge?: string }) => Promise<void>
  onOpen: () => void
  onSaveAs: () => void
  onJump: (path: string) => void
}) {
  const [collapse, setCollapse] = useState<Collapse>({ map: {}, all: null })
  const [tab, setTab] = useState<Tab>("overview")
  const [newOpen, setNewOpen] = useState(false)
  const [problemsOpen, setProblemsOpen] = useState(false)
  const [selected, setSelected] = useState(-1)
  const formRef = useRef<HTMLDivElement>(null)
  const generation = editor.generation ?? 0

  // A new or opened document starts with the default expansion, as in Tkinter.
  useEffect(() => {
    setCollapse({ map: {}, all: null })
  }, [generation])

  // Going to a field (a problem, a card's "All settings", a normalizer chip):
  // switch to All fields, expand its sections, then focus it once drawn.
  // A field the session just added to a sweep is only revealed if All fields is showing.
  useEffect(() => {
    if (!focus) return
    if (focus.show) setTab("all")
    setCollapse((now) => expandTo(now, focus.path))
  }, [focus?.n])
  useEffect(() => {
    if (!focus || tab !== "all") return
    const timer = window.setTimeout(() => focusPath(formRef.current, focus.path), 30)
    return () => window.clearTimeout(timer)
  }, [focus?.n, editor, collapse, tab])

  const handlers: Handlers = {
    onText: (path, text) => void onRun("editor.set_text", { target, path, text }, { merge: `text:${target}:${path}` }),
    onExtra: (path, text) => void onRun("editor.set_extra", { target, path, text }, { merge: `extra:${target}:${path}` }),
    onType: (path, typeName) => void onRun("editor.set_type", { target, path, type_name: typeName }),
    onEnabled: (path, enabled) => void onRun("editor.set_enabled", { target, path, enabled }),
    onList: (path, op, index) => {
      if (op === "add") void onRun("editor.list_add", { target, path })
      if (op === "remove") void onRun("editor.list_remove", { target, path, index })
      if (op === "up") void onRun("editor.list_move", { target, path, index, delta: -1 })
      if (op === "down") void onRun("editor.list_move", { target, path, index, delta: 1 })
    },
    onMap: (path, op, key) => void onRun(op === "add" ? "editor.map_add" : "editor.map_remove", { target, path, key }),
    onMenu: target === "config" ? (path, how) => void onRun("sweep.add_from_config", { path, how }) : undefined,
  }

  function jump(problem: string) {
    const path = problem.split(":")[0]
    if (editor.only_set) void onRun("editor.set_flags", { target, only_set: false })
    onJump(path)
  }

  const count = editor.problems.length
  const blank = target === "sweep" ? "Blank sweep" : "Blank config"
  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: target === "sweep" ? "Phases" : "Overview" },
    { id: "all", label: "All fields" },
    { id: "yaml", label: "YAML" },
  ]
  const problemsList = (
    <div className="problems">
      <div className={count ? "problems-title err" : "problems-title ok"}>{editor.problem_summary}</div>
      <ul role="listbox" aria-label="Problems">
        {editor.problems.map((problem, index) => (
          <li
            key={`${index}:${problem}`}
            role="option"
            aria-selected={selected === index}
            title="Double-click to go to the field."
            onClick={() => setSelected(index)}
            onDoubleClick={() => jump(problem)}
          >
            {problem}
          </li>
        ))}
      </ul>
    </div>
  )

  return (
    <section className="editor">
      <div className="toolbar">
        <div className="tabs" role="tablist" aria-label="View">
          {tabs.map((item) => (
            <button key={item.id} type="button" role="tab" aria-selected={tab === item.id} onClick={() => setTab(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
        <span className="grow" />
        <button
          type="button"
          className={count ? "valid-pill bad" : "valid-pill"}
          aria-expanded={problemsOpen}
          onClick={() => setProblemsOpen(!problemsOpen)}
          title={count ? "Show the problems" : "Nothing to fix"}
        >
          {count ? `${count} problem${count === 1 ? "" : "s"}` : "Valid · 0 problems"}
        </button>
        <div className="menu-button">
          <button type="button" aria-haspopup="menu" aria-expanded={newOpen} onClick={() => setNewOpen(!newOpen)}>
            New ▾
          </button>
          {newOpen && (
            <div className="menu dropdown" role="menu" onMouseLeave={() => setNewOpen(false)}>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setNewOpen(false)
                  void onRun("editor.new_blank", { target })
                }}
              >
                {blank}
              </button>
              {examples.length > 0 && <hr />}
              {examples.map((name) => (
                <button
                  key={name}
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setNewOpen(false)
                    void onRun("editor.new_from_example", { target, name })
                  }}
                >
                  From example: {name}
                </button>
              ))}
            </div>
          )}
        </div>
        <button type="button" onClick={onOpen}>
          Open...
        </button>
        <button type="button" onClick={() => void onRun("editor.save", { target })}>
          Save
        </button>
        <button type="button" onClick={onSaveAs}>
          Save As...
        </button>
      </div>

      {tab === "overview" && (
        <div className="page-scroll">
          {target === "config" ? (
            <ConfigOverview editor={editor} handlers={handlers} onJump={onJump} />
          ) : (
            <SweepOverview editor={editor} handlers={handlers} onRun={onRun} onJump={onJump} />
          )}
        </div>
      )}

      {tab === "all" && (
        <>
          <div className="toolbar sub">
            <label title="Hide fields still at their default, to see what this file changes.">
              <input
                type="checkbox"
                checked={editor.only_set}
                onChange={(event) => void onRun("editor.set_flags", { target, only_set: event.target.checked })}
              />
              Show only set fields
            </label>
            <label title="Also write settings left at their default. Off keeps the file to the choices that matter.">
              <input
                type="checkbox"
                checked={editor.with_defaults}
                onChange={(event) => void onRun("editor.set_flags", { target, with_defaults: event.target.checked })}
              />
              Write default values
            </label>
            <span className="grow" />
            <button type="button" onClick={() => setCollapse({ map: {}, all: true })}>
              Collapse all
            </button>
            <button type="button" onClick={() => setCollapse({ map: {}, all: false })}>
              Expand all
            </button>
          </div>
          <div className="editor-panes">
            <div className="form-pane" ref={formRef}>
              <FormView
                form={editor.form}
                onlySet={editor.only_set}
                collapse={collapse}
                onCollapse={setCollapse}
                extraErrors={editor.extra_errors}
                handlers={handlers}
              />
            </div>
            <div className="side-pane">
              <div className="pane-title">YAML preview · what Save writes</div>
              <Preview yaml={editor.yaml} />
            </div>
          </div>
          {problemsList}
        </>
      )}

      {tab === "yaml" && (
        <div className="page-scroll">
          <Preview yaml={editor.yaml} big />
        </div>
      )}

      {problemsOpen && tab !== "all" && <div className="problems-drawer">{problemsList}</div>}
    </section>
  )
}

function Preview({ yaml, big = false }: { yaml: string; big?: boolean }) {
  return <pre className={big ? "mono preview big" : "mono preview"}>{yaml}</pre>
}
