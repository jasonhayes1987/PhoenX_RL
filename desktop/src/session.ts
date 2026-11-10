export type Need = {
  need: "yes_no" | "yes_no_cancel" | "ok" | "save_as"
  id: string
  title: string
  message: string
}

export type Client = {
  call: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  quit?: () => Promise<void>
}

export function parseLine(line: string): unknown {
  let message: { id?: unknown; result?: unknown; error?: { title: string; message: string } }
  try {
    message = JSON.parse(line)
  } catch {
    throw new Error("The session returned a line that is not JSON.")
  }
  if (message.error) {
    throw new Error(`${message.error.title}: ${message.error.message}`)
  }
  return message.result
}

export function createTauriClient(): Client {
  return {
    async call(method, params = {}) {
      const { invoke } = await import("@tauri-apps/api/core")
      if (method === "open_folder") {
        await invoke("open_folder", params)
        return null
      }
      const line = await invoke<string>("session_call", {
        request: JSON.stringify({ id: method, method, params }),
      })
      return parseLine(line)
    },
    async quit() {
      const { invoke } = await import("@tauri-apps/api/core")
      await invoke("quit_app")
    },
  }
}

export type Outcome = { result?: unknown; stale: boolean }

export type Queue = {
  /**
   * Run one session call after every earlier one. Calls with the same
   * ``merge`` key that are still waiting are replaced by the newest (typing
   * in one field sends one call per pause, not per key), and a finished
   * call whose key is already queued again comes back ``stale`` so the
   * screen skips that in-between snapshot.
   */
  call: (method: string, params: Record<string, unknown>, merge?: string) => Promise<Outcome>
  busy: () => boolean
}

type Job = {
  method: string
  params: Record<string, unknown>
  merge?: string
  resolve: (outcome: Outcome) => void
  reject: (error: unknown) => void
}

export function createQueue(client: Client): Queue {
  const waiting: Job[] = []
  let running = false

  async function pump() {
    if (running) return
    running = true
    while (waiting.length) {
      const job = waiting.shift()!
      try {
        const result = await client.call(job.method, job.params)
        const isNeed = !!result && typeof result === "object" && "need" in (result as object)
        const stale = !isNeed && !!job.merge && waiting.some((next) => next.merge === job.merge)
        job.resolve({ result, stale })
      } catch (error) {
        job.reject(error)
      }
    }
    running = false
  }

  return {
    call(method, params, merge) {
      return new Promise<Outcome>((resolve, reject) => {
        if (merge) {
          const index = waiting.findIndex((job) => job.merge === merge)
          if (index >= 0) {
            waiting[index].resolve({ stale: true })
            waiting.splice(index, 1)
          }
        }
        waiting.push({ method, params, merge, resolve, reject })
        void pump()
      })
    },
    busy: () => running || waiting.length > 0,
  }
}
