import { StrictMode, useEffect, useState } from "react"
import { createRoot } from "react-dom/client"
import { App, tauriDialogs } from "./App"
import { createTauriClient } from "./session"
import "@fontsource/inter/400.css"
import "@fontsource/inter/500.css"
import "@fontsource/inter/600.css"
import "@fontsource/chakra-petch/500.css"
import "@fontsource/chakra-petch/600.css"
import "@fontsource/chakra-petch/700.css"
import "@fontsource/jetbrains-mono/400.css"
import "@fontsource/jetbrains-mono/500.css"
import "./brand/theme.css"
import "./app.css"

const client = createTauriClient()

// Closing the window goes through the session (app.close), which asks about
// unsaved edits and a running command first, as the Tkinter window did.
// The session answers {closed: true} when it is safe, and the App quits then.
async function guardClose(onClose: () => void) {
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window")
    await getCurrentWindow().onCloseRequested((event) => {
      event.preventDefault()
      onClose()
    })
  } catch {
    // Not inside Tauri (tests, a browser preview): nothing to guard.
  }
}

void guardClose(() => window.dispatchEvent(new CustomEvent("phoenx-close")))

function Root() {
  const [ready, setReady] = useState<boolean | null>(null)
  useEffect(() => {
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke<{ ready: boolean }>("python_status"))
      .then((status) => setReady(status.ready))
      .catch(() => setReady(false))
  }, [])
  if (ready === null) return null
  return (
    <App
      client={client}
      dialogs={tauriDialogs}
      pythonReady={ready}
      onPython={async (path) => {
        const { invoke } = await import("@tauri-apps/api/core")
        try {
          await invoke("set_python", { path })
          return true
        } catch {
          return false
        }
      }}
    />
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
