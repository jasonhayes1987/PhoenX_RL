import mark from "./mark.svg"
import dashboard from "./dashboard.svg"
import config from "./config.svg"
import sweep from "./sweep.svg"
import launch from "./launch.svg"
import settings from "./settings.svg"

export const wordmark = "PhoenX RL"
export const versionLabel = "0.1.0"
export const logoUrl = mark

export const nav = [
  { id: "dashboard", label: "Dashboard", iconUrl: dashboard },
  { id: "config", label: "Config Builder", iconUrl: config },
  { id: "sweep", label: "Sweep Builder", iconUrl: sweep },
  { id: "launch", label: "Launch", iconUrl: launch },
  { id: "settings", label: "Settings", iconUrl: settings },
] as const

export type PageId = (typeof nav)[number]["id"]
