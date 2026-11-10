import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { App } from "./App"
import type { Client } from "./session"

function quietClient(): Client {
  return {
    async call() {
      return { version: "0.1.0" }
    },
    quit: vi.fn(async () => undefined),
  }
}

describe("python picker", () => {
  it("asks again after a failed import and quits on the second failure", async () => {
    const quiet = quietClient()
    const onPython = vi.fn(async () => false)
    render(<App client={quiet} pythonReady={false} onPython={onPython} />)
    expect(screen.getByRole("heading", { name: "PhoenX Python" })).toBeTruthy()
    expect(screen.getByText(/Choose the python.exe/)).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Config Builder" })).toBeNull()

    fireEvent.change(screen.getByRole("textbox", { name: "python.exe" }), {
      target: { value: "C:\\missing\\python.exe" },
    })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Use this Python" }))
    })
    expect(screen.getByText(/That Python cannot import phoenx/)).toBeTruthy()

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Use this Python" }))
    })
    expect(quiet.quit).toHaveBeenCalledOnce()
    expect(screen.queryByRole("button", { name: "Config Builder" })).toBeNull()
  })

  it("quits on cancel without opening a page", async () => {
    const quiet = quietClient()
    render(<App client={quiet} pythonReady={false} onPython={async () => true} />)
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    })
    expect(quiet.quit).toHaveBeenCalledOnce()
    expect(screen.queryByRole("button", { name: "Dashboard" })).toBeNull()
  })
})
