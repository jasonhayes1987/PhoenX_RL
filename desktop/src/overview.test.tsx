// The overview pages: cards, model diagram, phase cards and live tiles all
// read the session's snapshot and edit through the same handlers.
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { Editor, Field, Handlers } from "./form"
import { LaunchPage, type Launch } from "./launch"
import { ConfigOverview, type Cards } from "./overview"
import { readable, SweepOverview } from "./sweep"

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

function handlers(): Handlers {
  return { onText: vi.fn(), onExtra: vi.fn(), onType: vi.fn(), onEnabled: vi.fn(), onList: vi.fn(), onMap: vi.fn() }
}

const dense = (path: string, units: string) =>
  f({
    path,
    name: path.split(".").pop()!,
    kind: "tagged",
    tag: "type",
    text: "dense",
    choices: ["dense", "relu"],
    variant_path: `${path}.params`,
    children: [f({ path: `${path}.params.units`, name: "units", kind: "int", text: units })],
  })

function configEditor(): Editor & { cards: Cards } {
  const branch = (name: string, head: string, units: string[]) =>
    f({
      path: `agent.config.model.branches.${name}`,
      name,
      kind: "tagged",
      tag: "type",
      text: head,
      choices: ["ValueHead", "StochasticContinuousHead"],
      variant_path: `agent.config.model.branches.${name}`,
      children: [
        f({
          path: `agent.config.model.branches.${name}.layer_config`,
          name: "layer_config",
          kind: "list",
          children: units.map((u, i) => dense(`agent.config.model.branches.${name}.layer_config.${i}`, u)),
        }),
      ],
    })
  const form = f({
    path: "",
    name: "",
    kind: "object",
    children: [
      f({
        path: "schedule",
        name: "schedule",
        kind: "object",
        children: [
          f({ path: "schedule.stop_units", name: "stop_units", kind: "int", text: "2000000" }),
          f({ path: "schedule.stop_unit", name: "stop_unit", kind: "enum", text: "timestep", choices: ["timestep", "episode"], combo: true }),
          f({ path: "schedule.mini_batch_size", name: "mini_batch_size", kind: "int", text: "256", default: "1" }),
        ],
      }),
      f({
        path: "agent",
        name: "agent",
        kind: "tagged",
        tag: "type",
        text: "PPO",
        choices: ["Reinforce", "ActorCritic", "PPO", "DDPG", "TD3", "SAC"],
        variant_path: "agent.config",
        children: [
          f({ path: "agent.config.policy_clip", name: "policy_clip", kind: "number", text: "0.2", default: "0.2" }),
          f({ path: "agent.config.state_normalizer", name: "state_normalizer", kind: "optional", inner_kind: "tagged", enabled: true, text: "RunningNorm" }),
          f({
            path: "agent.config.model",
            name: "model",
            kind: "object",
            children: [
              f({ path: "agent.config.model.roots", name: "roots", kind: "map" }),
              f({ path: "agent.config.model.trunk", name: "trunk", kind: "optional", inner_kind: "object", enabled: false }),
              f({
                path: "agent.config.model.branches",
                name: "branches",
                kind: "map",
                children: [branch("policy", "StochasticContinuousHead", ["64", "64"]), branch("value", "ValueHead", ["128"])],
              }),
            ],
          }),
        ],
      }),
      f({ path: "env", name: "env", kind: "tagged", tag: "type", text: "gymnasium", choices: ["gymnasium", "envpool", "isaacsim"], variant_path: "env.config" }),
    ],
  })
  return {
    noun: "config",
    path: null,
    dirty: false,
    generation: 1,
    title: "untitled",
    status: "",
    only_set: false,
    with_defaults: false,
    problems: [],
    problem_summary: "",
    yaml: "",
    form,
    base_note: "",
    base_fields: [],
    phases: [],
    selected_phase: "",
    cards: {
      agent_type: "PPO",
      on_policy: true,
      agent: [{ path: "agent.config.policy_clip", label: "Policy clip" }],
      schedule: [{ path: "schedule.mini_batch_size", label: "Mini-batch size" }],
      env: [],
      normalizers: [{ path: "agent.config.state_normalizer", label: "state" }],
      model: "agent.config.model",
    },
  }
}

describe("Config Builder overview", () => {
  it("edits card fields and switches the agent through the session", () => {
    const h = handlers()
    const onJump = vi.fn()
    render(<ConfigOverview editor={configEditor()} handlers={h} onJump={onJump} />)
    fireEvent.change(screen.getByRole("textbox", { name: "agent.config.policy_clip" }), { target: { value: "0.3" } })
    expect(h.onText).toHaveBeenCalledWith("agent.config.policy_clip", "0.3")
    expect(screen.getByText("Mini-batch size")).toBeTruthy()
    fireEvent.click(screen.getByRole("radio", { name: "SAC" }))
    expect(h.onType).toHaveBeenCalledWith("agent", "SAC")
    fireEvent.click(screen.getByRole("radio", { name: "isaacsim" }))
    expect(h.onType).toHaveBeenCalledWith("env", "isaacsim")
    fireEvent.click(screen.getByRole("button", { name: "state: RunningNorm" }))
    expect(onJump).toHaveBeenCalledWith("agent.config.state_normalizer")
  })

  it("draws the branches with their widths and edits the clicked one", () => {
    const h = handlers()
    render(<ConfigOverview editor={configEditor()} handlers={h} onJump={vi.fn()} />)
    const policy = screen.getByRole("button", { name: /policy\s*64 → 64/ })
    expect(policy).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: /value\s*128/ }))
    const inspector = screen.getByLabelText("Inspector")
    expect(within(inspector).getByText("agent.config.model.branches.value")).toBeTruthy()
    fireEvent.change(within(inspector).getByRole("textbox", { name: "agent.config.model.branches.value.layer_config.0.params.units" }), {
      target: { value: "256" },
    })
    expect(h.onText).toHaveBeenCalledWith("agent.config.model.branches.value.layer_config.0.params.units", "256")
    fireEvent.click(screen.getByRole("button", { name: "Remove branch" }))
    expect(h.onMap).toHaveBeenCalledWith("agent.config.model.branches", "remove", "value")
    fireEvent.change(screen.getByRole("textbox", { name: "New branch name" }), { target: { value: "critic" } })
    fireEvent.click(screen.getByRole("button", { name: "+ Branch" }))
    expect(h.onMap).toHaveBeenCalledWith("agent.config.model.branches", "add", "critic")
  })
})

describe("Sweep Builder overview", () => {
  it("shows each phase's search and selects a phase through the session", () => {
    const phase = (i: number, name: string, extra: Field[]) =>
      f({ path: `phases.${i}`, name: String(i), kind: "object", children: [f({ path: `phases.${i}.name`, name: "name", kind: "string", text: name }), ...extra] })
    const space = f({
      path: "phases.1.search_space",
      name: "search_space",
      kind: "map",
      children: [
        f({
          path: "phases.1.search_space.agent.config.discount",
          name: "agent.config.discount",
          kind: "tagged",
          text: "uniform",
          children: [
            f({ path: "x.low", name: "low", kind: "number", text: "0.95" }),
            f({ path: "x.high", name: "high", kind: "number", text: "0.999" }),
          ],
        }),
      ],
    })
    const optimizers = f({
      path: "phases.0.optimizers",
      name: "optimizers",
      kind: "map",
      children: [f({ path: "phases.0.optimizers.branches.policy", name: "branches.policy", kind: "raw", text: "{lr: {dist: loguniform, low: 1.0e-05, high: 0.001}}" })],
    })
    const editor = {
      ...configEditor(),
      noun: "sweep",
      phases: ["optimizers", "refine"],
      selected_phase: "refine",
      form: f({ path: "", name: "", kind: "object", children: [f({ path: "phases", name: "phases", kind: "list", children: [phase(0, "optimizers", [optimizers]), phase(1, "refine", [space])] })] }),
    }
    const onRun = vi.fn(async () => undefined)
    render(<SweepOverview editor={editor} handlers={handlers()} onRun={onRun} onJump={vi.fn()} />)
    expect(screen.getByText("Phase 2 · refine")).toBeTruthy()
    expect(screen.getByText("0.95 – 0.999")).toBeTruthy()
    expect(screen.getByText("1 searched: policy · lr")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: /optimizers/ }))
    expect(onRun).toHaveBeenCalledWith("sweep.select_phase", { name: "optimizers" })
    expect(readable("agent.config.model.branches.critic.merged_config.0.params.units")).toBe("critic · merged_config · units")
  })
})

describe("Launch live status", () => {
  it("shows the trainer's progress and best checkpoint", () => {
    const launch: Launch = {
      mode: "train",
      values: { cwd: "." },
      command: "python -m phoenx.cli.train",
      running: true,
      status: "timestep 412,160/1,000,000",
      progress: { mode: "train", unit: "timestep", done: 412160, total: 1000000, fraction: 0.41216, episodes: 1284, avg_reward: 186.4, elapsed: "0:21:47", best: 201.7 },
      log: [{ text: "[checkpoint] saved best (avg_reward=201.70)", tag: "good" }],
      start_enabled: false,
      stop_enabled: true,
      stop_label: "Stop",
      open_output_enabled: true,
      output_dir: "run",
      resume_info: "",
      test_info: "",
      phase_choices: [],
      run_status: "Running",
    }
    const dialogs = { openFile: vi.fn(), saveFile: vi.fn(), openFolder: vi.fn() }
    render(<LaunchPage launch={launch} dialogs={dialogs} onRun={vi.fn(async () => undefined)} />)
    expect(screen.getByText("412,160")).toBeTruthy()
    expect(screen.getByText("of 1,000,000 · 41%")).toBeTruthy()
    expect(screen.getByText("1,284")).toBeTruthy()
    expect(screen.getByText("186.4")).toBeTruthy()
    expect(screen.getByText("best 201.7 · saved")).toBeTruthy()
    expect(screen.getByText("0:21:47")).toBeTruthy()
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("41")
  })
})
