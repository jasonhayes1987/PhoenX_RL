import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { App } from "./App"
import { nav } from "./brand"
import type { Client } from "./session"

const form = {
  path: "",
  name: "config",
  kind: "object",
  description: "",
  required: true,
  text: "",
  choices: [] as string[],
  editable_choices: false,
  extra_text: "",
  enabled: true,
  tag: "",
  set: true,
  children: [] as unknown[],
}

function editor(noun: "config" | "sweep") {
  return {
    noun,
    path: null,
    dirty: false,
    title: noun === "config" ? "untitled config - PhoenX" : "untitled sweep - PhoenX",
    status: "ready",
    only_set: false,
    with_defaults: false,
    problems: [],
    problem_summary: "Problems: none, ready to save",
    yaml: "a: 1\n",
    form,
    base_note: "base",
    base_fields: [],
    phases: ["phase_0"],
    selected_phase: "phase_0",
  }
}

function launch(extra: Record<string, unknown> = {}) {
  return {
    mode: "train",
    values: { cwd: ".", log_level: "", config: "" },
    command: "python -m phoenx.cli.train",
    running: false,
    status: "Idle.",
    progress: null,
    log: [],
    start_enabled: false,
    stop_enabled: false,
    stop_label: "Stop",
    open_output_enabled: false,
    output_dir: null,
    resume_info: "",
    test_info: "",
    phase_choices: [""],
    run_status: "Idle",
    ...extra,
  }
}

function client(snapshot = { initial_page: "config", run_status: "Idle" }, launchState = launch()): Client {
  return {
    call: vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "app.version") return { version: "0.1.0" }
      if (method === "app.snapshot") return snapshot
      if (method === "editor.snapshot") return editor(params?.target === "sweep" ? "sweep" : "config")
      if (method === "launch.snapshot") return launchState
      if (method === "editor.examples") return { names: [] }
      if (method === "app.close") return { closed: true }
      return editor("config")
    }),
    quit: vi.fn(async () => undefined),
  }
}

describe("chrome", () => {
  it("shows the five pages in order and one current page", async () => {
    render(<App client={client()} />)
    const labels = await screen.findAllByRole("button", { name: /Dashboard|Config Builder|Sweep Builder|Launch|Settings/ })
    expect(labels.map((node) => node.textContent)).toEqual(nav.map((item) => item.label))
    expect(screen.getAllByRole("button", { current: "page" })).toHaveLength(1)
  })

  it("keeps Dashboard and Settings empty of controls", async () => {
    render(<App client={client()} />)
    await screen.findByText("0.1.0")
    for (const name of ["Dashboard", "Settings"]) {
      fireEvent.click(screen.getByRole("button", { name }))
      expect(screen.getByRole("heading", { name })).toBeTruthy()
      const main = document.querySelector("main")!
      expect(main.querySelectorAll("button, input, table")).toHaveLength(0)
    }
  })

  it("swaps artwork when the brand module changes", async () => {
    const first = render(<App client={client()} />)
    await screen.findByText("0.1.0")
    const firstSrc = document.querySelector(".sidebar img")!.getAttribute("src")
    first.unmount()
    render(
      <App
        client={client()}
        marks={{
          logoUrl: "/other.svg",
          wordmark: "PhoenX RL",
          nav: nav.map((item) => ({ ...item, iconUrl: "/other.svg" })),
        }}
      />,
    )
    await screen.findByText("0.1.0")
    expect(document.querySelector(".sidebar img")!.getAttribute("src")).not.toBe(firstSrc)
  })

  it("paints amber only on a live run, an enabled Start, and the selected phase", async () => {
    const idle = render(<App client={client()} />)
    await screen.findByText("0.1.0")
    expect(document.querySelectorAll('[data-tone="live"]')).toHaveLength(0)
    idle.unmount()

    const armed = render(<App client={client({ initial_page: "config", run_status: "Idle" }, launch({ start_enabled: true }))} />)
    await screen.findByText("0.1.0")
    fireEvent.click(screen.getByRole("button", { name: "Launch" }))
    const start = screen.getByRole("button", { name: "Start" })
    expect(start.getAttribute("data-tone")).toBe("live")

    armed.unmount()
    render(
      <App client={client({ initial_page: "sweep", run_status: "Running" }, launch({ start_enabled: false, run_status: "Running" }))} />,
    )
    await screen.findByText("Running")
    const live = [...document.querySelectorAll('[data-tone="live"]')].map((node) => node.textContent)
    expect(live.every((text) => text === "Running" || text === "phase_0" || text === "Start")).toBe(true)
  })

  it("closes through app.close", async () => {
    const api = client()
    render(<App client={api} />)
    await screen.findByText("0.1.0")
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Close" }))
    })
    expect(api.call).toHaveBeenCalledWith("app.close", expect.anything())
  })
})
