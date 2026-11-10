// Regression tests for the form behaviors the Tkinter editor had and the
// first desktop screen lost.
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { FormView, isCollapsed, type Collapse, type Field, type Handlers } from "./form"
import { createQueue, type Client } from "./session"

function f(partial: Partial<Field> & Pick<Field, "path" | "name" | "kind">): Field {
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

const layerPath = "agent.config.model.branches.policy.layer_config"

/** agent (tagged) > model > branches (map) > policy (tagged head) > layer_config (list) > dense layer. */
function tree(): Field {
  const dense = f({
    path: `${layerPath}.0`,
    name: "0",
    kind: "tagged",
    tag: "type",
    text: "dense",
    choices: ["dense", "relu", "conv2d"],
    variant_path: `${layerPath}.0.params`,
    children: [
      f({ path: `${layerPath}.0.params.units`, name: "units", kind: "int", required: true, text: "64" }),
      f({ path: `${layerPath}.0.params.kernel`, name: "kernel", kind: "enum", choices: ["", "default"], combo: true }),
    ],
  })
  const policy = f({
    path: "agent.config.model.branches.policy",
    name: "policy",
    kind: "tagged",
    tag: "type",
    text: "ValueHead",
    choices: ["ValueHead", "StochasticContinuousHead"],
    variant_path: "agent.config.model.branches.policy",
    children: [f({ path: layerPath, name: "layer_config", kind: "list", children: [dense] })],
  })
  return f({
    path: "",
    name: "",
    kind: "object",
    children: [
      f({
        path: "agent",
        name: "agent",
        kind: "tagged",
        tag: "type",
        text: "SAC",
        choices: ["PPO", "SAC"],
        required: true,
        variant_path: "agent.config",
        children: [
          f({
            path: "agent.config.model",
            name: "model",
            kind: "object",
            required: true,
            children: [f({ path: "agent.config.model.branches", name: "branches", kind: "map", children: [policy] })],
          }),
          f({
            path: "agent.config.seed",
            name: "seed",
            kind: "int",
            nullable: true,
            default: "null",
            choices: ["null"],
            combo: false,
          }),
        ],
      }),
    ],
  })
}

function handlers(): Handlers {
  return {
    onText: vi.fn(),
    onExtra: vi.fn(),
    onType: vi.fn(),
    onEnabled: vi.fn(),
    onList: vi.fn(),
    onMap: vi.fn(),
  }
}

function Harness({ form, h, start }: { form: Field; h: Handlers; start?: Collapse }) {
  const [collapse, setCollapse] = useState<Collapse>(start ?? { map: {}, all: null })
  return <FormView form={form} onlySet={false} collapse={collapse} onCollapse={setCollapse} handlers={h} />
}

describe("form layout", () => {
  it("collapses sections three levels deep, like the Tkinter editor", () => {
    const none: Collapse = { map: {}, all: null }
    expect(isCollapsed(none, "agent")).toBe(false)
    expect(isCollapsed(none, "agent.config")).toBe(false)
    expect(isCollapsed(none, "agent.config.model")).toBe(true)
    expect(isCollapsed({ map: {}, all: false }, "a.b.c.d")).toBe(false)
    expect(isCollapsed({ map: { "a.b.c": false }, all: null }, "a.b.c")).toBe(false)
  })

  it("shows a layer's parameters under its type dropdown once its sections are open", () => {
    const h = handlers()
    render(<Harness form={tree()} h={h} start={{ map: {}, all: false }} />)
    // The layer is a list item box: its type dropdown and its fields, always shown.
    expect(screen.getByRole("combobox", { name: `${layerPath}.0 type` })).toBeTruthy()
    expect(screen.getByRole("textbox", { name: `${layerPath}.0.params.units` })).toBeTruthy()
    fireEvent.change(screen.getByRole("combobox", { name: `${layerPath}.0 type` }), { target: { value: "conv2d" } })
    expect(h.onType).toHaveBeenCalledWith(`${layerPath}.0`, "conv2d")
  })

  it("lets a collapsed section be expanded down to the layers", () => {
    render(<Harness form={tree()} h={handlers()} />)
    expect(screen.queryByText("layer_config")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "toggle model" }))
    fireEvent.click(screen.getByRole("button", { name: "toggle branches" }))
    fireEvent.click(screen.getByRole("button", { name: "toggle policy" }))
    fireEvent.click(screen.getByRole("button", { name: "toggle layer_config" }))
    expect(screen.getByRole("textbox", { name: `${layerPath}.0.params.units` })).toBeTruthy()
  })

  it("changes a branch's head type through the session", () => {
    const h = handlers()
    render(<Harness form={tree()} h={h} start={{ map: {}, all: false }} />)
    fireEvent.change(screen.getByRole("combobox", { name: "agent.config.model.branches.policy type" }), {
      target: { value: "StochasticContinuousHead" },
    })
    expect(h.onType).toHaveBeenCalledWith("agent.config.model.branches.policy", "StochasticContinuousHead")
  })

  it("expands a list or map section when something is added to it", () => {
    const h = handlers()
    const form = tree()
    render(<Harness form={form} h={h} />)
    fireEvent.click(screen.getByRole("button", { name: "toggle model" }))
    expect(screen.getByRole("button", { name: "toggle branches" }).getAttribute("aria-expanded")).toBe("false")
    fireEvent.click(screen.getAllByRole("button", { name: "+ Add" })[0])
    expect(h.onMap).toHaveBeenCalledWith("agent.config.model.branches", "add", "")
    expect(screen.getByRole("button", { name: "toggle branches" }).getAttribute("aria-expanded")).toBe("true")
  })

  it("keeps a nullable number as a text box with its default hint", () => {
    render(<Harness form={tree()} h={handlers()} />)
    const seed = screen.getByRole("textbox", { name: "agent.config.seed" })
    expect(seed).toBeTruthy()
    expect(screen.getByText("default null")).toBeTruthy()
  })

  it("does not snap back while the session is still answering", () => {
    const h = handlers()
    render(<Harness form={tree()} h={h} start={{ map: {}, all: false }} />)
    const units = screen.getByRole("textbox", { name: `${layerPath}.0.params.units` }) as HTMLInputElement
    fireEvent.change(units, { target: { value: "2" } })
    fireEvent.change(units, { target: { value: "25" } })
    fireEvent.change(units, { target: { value: "256" } })
    expect(units.value).toBe("256")
    expect(h.onText).toHaveBeenLastCalledWith(`${layerPath}.0.params.units`, "256")
  })
})

describe("session queue", () => {
  it("runs calls in order and merges edits to one field while one is in flight", async () => {
    const seen: string[] = []
    let release: () => void = () => undefined
    const client: Client = {
      call: vi.fn(async (_method: string, params?: Record<string, unknown>) => {
        seen.push(String(params?.text))
        if (seen.length === 1) await new Promise<void>((resolve) => (release = resolve))
        return { noun: "config", text: params?.text }
      }),
    }
    const queue = createQueue(client)
    const first = queue.call("editor.set_text", { text: "2" }, "units")
    const second = queue.call("editor.set_text", { text: "25" }, "units")
    const third = queue.call("editor.set_text", { text: "256" }, "units")
    expect(await second).toEqual({ stale: true })
    await act(async () => release())
    expect(await first).toMatchObject({ stale: true })
    expect(await third).toMatchObject({ stale: false, result: { text: "256" } })
    expect(seen).toEqual(["2", "256"])
  })
})
