# 0017. The product name is PhoenX RL and the install is per-user

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The window needs a name, an installer id, a file name, and a folder.
An override must be able to force one copy, including a missing path,
so a bad setting does not silently launch a different install.

## Options considered

1. **PhoenX RL, `com.phoenx.rl`, `phoenx-rl`, per-user under
   `%LOCALAPPDATA%\PhoenX RL`** — no administrator prompt. Good: a normal
   account can install it. Bad: it is not on the system PATH.
2. **A machine-wide install** — one copy for every user. Bad: it needs
   an administrator, and two users may want different builds.

## Decision

The product is PhoenX RL, the identifier is `com.phoenx.rl`, and the
binary is `phoenx-rl` (`phoenx-rl.exe` on Windows). The NSIS install
mode is `currentUser`. If `PHOENX_GUI` is set, that path is the only
candidate. A missing file fails the search.

## Why

The window is a personal tool next to a personal Python environment.
A per-user folder matches that. An explicit override should not fall
through to a different program.

## Consequences

The command looks for the checkout's release binary first, then the
per-user install. `PHOENX_GUI` replaces both.
