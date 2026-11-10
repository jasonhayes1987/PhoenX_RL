import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { App, type Dialogs } from "./App"
import type { Client } from "./session"
import type { Editor, Field } from "./form"

const calls: { method: string; params?: Record<string, unknown> }[] = []

function field(partial: Partial<Field> & Pick<Field, "path" | "name" | "kind">): Field {
  return {
    description: "",
    required: false,
    text: "",
    choices: [],
    editable_choices: false,
    extra_text: "",
    enabled: true,
    tag: "",
    set: true,
    children: [],
    ...partial,
  }
}

function editor(noun: "config" | "sweep"): Editor {
  return {
    noun,
    path: null,
    dirty: false,
    generation: 0,
    title: "untitled",
    status: "ready",
    only_set: false,
    with_defaults: false,
    problems: ["schedule.learn_every: not a number"],
    problem_summary: "Problems (1)",
    yaml: "a: 1\n",
    form: field({
      path: "",
      name: "",
      kind: "object",
      open: true,
      children: [
        field({
          path: "schedule",
          name: "schedule",
          kind: "object",
          required: true,
          children: [
            field({ path: "schedule.learn_every", name: "learn_every", kind: "int", text: "1", default: "1" }),
          ],
        }),
        field({ path: "layers", name: "layers", kind: "list" }),
        field({ path: "branches", name: "branches", kind: "map" }),
        ...(noun === "sweep"
          ? [
              field({
                path: "phases",
                name: "phases",
                kind: "list",
                children: [
                  field({
                    path: "phases.0",
                    name: "0",
                    kind: "object",
                    children: [field({ path: "phases.0.name", name: "name", kind: "string", text: "phase_0" })],
                  }),
                ],
              }),
            ]
          : []),
      ],
    }),
    base_note: "Base fields",
    base_fields: [{ path: "schedule.learn_every", text: "1", in_file: true }],
    phases: ["phase_0"],
    selected_phase: "phase_0",
  }
}

const launch = {
  mode: "train",
  values: { cwd: ".", log_level: "", config: "", sweep: "", from_phase: "" },
  command: "python -m phoenx.cli.train",
  running: false,
  status: "Idle.",
  progress: null,
  log: [],
  start_enabled: true,
  stop_enabled: true,
  stop_label: "Stop",
  open_output_enabled: true,
  output_dir: "runs/out",
  resume_info: "",
  test_info: "",
  phase_choices: ["", "phase_0"],
  run_status: "Idle",
}

function client(reply?: (method: string, params?: Record<string, unknown>) => unknown): Client {
  return {
    async call(method, params) {
      calls.push({ method, params })
      const answer = reply?.(method, params)
      if (answer !== undefined) return answer
      if (method === "app.version") return { version: "0.1.0" }
      if (method === "app.snapshot") return { initial_page: "config", run_status: "Idle" }
      if (method === "editor.snapshot") return editor(params?.target === "sweep" ? "sweep" : "config")
      if (method === "editor.examples") return { names: ["LunarLander-v3/reinforce.yml"] }
      if (method.startsWith("launch.")) return { ...launch, mode: (params?.mode as string) || launch.mode }
      return editor(params?.target === "sweep" ? "sweep" : "config")
    },
  }
}

const dialogs: Dialogs = {
  openFile: vi.fn(async () => "a.yml"),
  saveFile: vi.fn(async () => "b.yml"),
  openFolder: vi.fn(async () => "runs"),
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  calls.length = 0
  vi.clearAllMocks()
})

describe("controls", () => {
  it("sends each Config Builder action to the session", async () => {
    render(<App client={client()} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })

    fireEvent.click(screen.getByRole("button", { name: "New ▾" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "Blank config" }))
    fireEvent.click(screen.getByRole("button", { name: "New ▾" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "From example: LunarLander-v3/reinforce.yml" }))
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Open..." })))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Save As..." })))
    fireEvent.click(screen.getByRole("tab", { name: "All fields" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Show only set fields" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Write default values" }))
    fireEvent.change(screen.getByRole("textbox", { name: "schedule.learn_every" }), { target: { value: "4" } })
    fireEvent.change(screen.getByRole("textbox", { name: "extra root" }), { target: { value: "k: 1" } })
    fireEvent.click(within(document.querySelector('[data-path="layers"]') as HTMLElement).getByRole("button", { name: "+ Add" }))
    fireEvent.change(screen.getByRole("textbox", { name: "name branches" }), { target: { value: "actor" } })
    fireEvent.click(within(document.querySelector('[data-path="branches"]') as HTMLElement).getByRole("button", { name: "+ Add" }))
    fireEvent.contextMenu(screen.getByText("learn_every"))
    fireEvent.click(screen.getByRole("menuitem", { name: /override/ }))
    fireEvent.doubleClick(screen.getByRole("option", { name: /schedule.learn_every/ }))
    await settle()

    const methods = calls.map((call) => call.method)
    expect(methods).toEqual(
      expect.arrayContaining([
        "editor.new_blank",
        "editor.new_from_example",
        "editor.open",
        "editor.save",
        "editor.save_as",
        "editor.set_flags",
        "editor.set_text",
        "editor.set_extra",
        "editor.list_add",
        "editor.map_add",
        "sweep.add_from_config",
      ]),
    )
    expect(calls.find((call) => call.method === "editor.open")?.params).toMatchObject({ path: "a.yml" })
    expect(calls.find((call) => call.method === "editor.save_as")?.params).toMatchObject({ path: "b.yml" })
    expect(calls.find((call) => call.method === "editor.map_add")?.params).toMatchObject({ key: "actor" })
    expect(calls.find((call) => call.method === "sweep.add_from_config")?.params).toMatchObject({
      path: "schedule.learn_every",
      how: "override",
    })
  })

  it("shows a need and sends the answer", async () => {
    const api = client((method, params) => {
      if (method === "editor.new_blank" && !params?.answer) {
        return {
          need: "yes_no_cancel",
          id: "discard",
          title: "Save changes?",
          message: "Save changes to untitled config before closing it?",
        }
      }
      return undefined
    })
    render(<App client={api} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })
    fireEvent.click(screen.getByRole("button", { name: "New ▾" }))
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: "Blank config" }))
    })
    expect(screen.getByRole("dialog", { name: "Save changes?" })).toBeTruthy()
    expect(screen.getByText("Save changes to untitled config before closing it?")).toBeTruthy()
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "No" }))
    })
    expect(calls.at(-1)).toMatchObject({ method: "editor.new_blank", params: { answer: "no" } })
  })

  it("answers a Save As question with the file dialog, or cancels it", async () => {
    let asked = 0
    const api = client((method, params) => {
      if (method === "editor.save" && !params?.answer) {
        asked += 1
        return { need: "save_as", id: "save-as", title: "Save As", message: "" }
      }
      return undefined
    })
    render(<App client={api} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Save" })))
    expect(calls.at(-1)).toMatchObject({ method: "editor.save_as", params: { target: "config", path: "b.yml" } })

    vi.mocked(dialogs.saveFile).mockResolvedValueOnce(null)
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Save" })))
    expect(calls.at(-1)).toMatchObject({ method: "editor.save", params: { answer: "cancel" } })
    expect(asked).toBe(2)
  })

  it("shows session errors instead of failing silently", async () => {
    const api = client((method) => {
      if (method === "editor.set_type") throw new Error("Can't edit: No such field.")
      return undefined
    })
    const form = editor("config")
    form.form.children.push(
      field({ path: "agent", name: "agent", kind: "tagged", tag: "type", text: "PPO", choices: ["PPO", "SAC"] }),
    )
    const withAgent = client((method, params) => {
      if (method === "editor.snapshot" && params?.target === "config") return form
      return api.call(method, params) as unknown
    })
    render(<App client={withAgent} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })
    fireEvent.click(screen.getByRole("tab", { name: "All fields" }))
    await act(async () => {
      fireEvent.change(screen.getByRole("combobox", { name: "agent type" }), { target: { value: "SAC" } })
    })
    await settle()
    expect(screen.getByRole("alertdialog", { name: "Can't edit" })).toBeTruthy()
    expect(screen.getByText("No such field.")).toBeTruthy()
  })

  it("sends Sweep Builder and Launch actions", async () => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn() } })
    render(<App client={client()} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })
    fireEvent.click(screen.getByRole("button", { name: "Sweep Builder" }))
    fireEvent.click(screen.getByRole("button", { name: /PHASE_0|phase_0/ }))
    fireEvent.click(screen.getByRole("button", { name: "+ Add from base config" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Phase" }), { target: { value: "phase_0" } })
    fireEvent.click(screen.getByRole("button", { name: "Add to search space" }))
    expect(screen.getByRole("alertdialog", { name: "Pick a field" })).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "OK" }))
    fireEvent.click(screen.getByRole("treeitem", { name: "schedule.learn_every" }))
    fireEvent.click(screen.getByRole("button", { name: "Add as override" }))
    fireEvent.doubleClick(screen.getByRole("treeitem", { name: "schedule.learn_every" }))
    fireEvent.click(screen.getByRole("button", { name: "Dry run" }))
    fireEvent.click(screen.getByRole("button", { name: "Set up launch →" }))
    await settle()

    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }))
    fireEvent.click(screen.getByRole("button", { name: "Use editor's config" }))
    await act(async () => {
      fireEvent.click(screen.getByRole("radio", { name: "Sweep" }))
    })
    fireEvent.click(screen.getByRole("button", { name: "Use editor's sweep" }))
    fireEvent.change(screen.getByRole("textbox", { name: "Working folder" }), { target: { value: "runs" } })
    await act(async () => fireEvent.click(screen.getAllByRole("button", { name: "Browse..." })[0]))
    fireEvent.click(screen.getByRole("button", { name: "Copy" }))
    fireEvent.click(screen.getByRole("button", { name: "Start" }))
    fireEvent.click(screen.getByRole("button", { name: "Stop" }))
    fireEvent.click(screen.getByRole("button", { name: "Open output folder" }))
    await settle()

    expect(calls.map((call) => call.method)).toEqual(
      expect.arrayContaining([
        "sweep.select_phase",
        "sweep.add_field",
        "sweep.handoff",
        "launch.use_editor",
        "launch.set_mode",
        "launch.set",
        "launch.start",
        "launch.stop",
        "open_folder",
      ]),
    )
    expect(calls.find((call) => call.method === "sweep.add_field")?.params).toMatchObject({
      path: "schedule.learn_every",
      how: "override",
    })
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("python -m phoenx.cli.train")
  })

  it("polls a running command so the log and progress update", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let polls = 0
    const api = client((method) => {
      if (method === "launch.snapshot") return { ...launch, running: true, start_enabled: false }
      if (method === "launch.poll") {
        polls += 1
        return { ...launch, running: polls < 3, log: [{ text: `line ${polls}`, tag: "" }] }
      }
      return undefined
    })
    render(<App client={api} dialogs={dialogs} />)
    await screen.findByRole("button", { name: "Save" })
    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }))
    for (let tick = 0; tick < 10; tick += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(100)
      })
    }
    expect(polls).toBe(3)
    expect(screen.getByText("line 3")).toBeTruthy()
    vi.useRealTimers()
  })
})
