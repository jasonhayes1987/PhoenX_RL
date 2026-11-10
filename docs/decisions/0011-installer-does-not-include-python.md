# 0011. The installer does not include Python or PhoenX

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

Training needs the PhoenX package, its dependencies, and sometimes a GPU
build of PyTorch. Those are installed in an environment the user already
chose. The window is a screen on top of that environment.

## Options considered

1. **Install only the window** — the user installs PhoenX, then launches
   the window. A Start Menu launch asks for `python.exe` and saves it
   only after `import phoenx` succeeds. Good: the GPU build stays the
   one the user installed. Bad: the installer alone cannot train.
2. **Embed a Python runtime** — one download trains. Bad: it would be a
   second PhoenX install, and it would not match the user's environment.

## Decision

The NSIS installer is per-user and contains the window only. It does not
embed Python and it does not install PhoenX.

## Why

The environment is where PhoenX, Isaac, and the GPU stack already live.
Pointing the window at that interpreter keeps one install.

## Consequences

`phoenx-config gui` from that environment sets `PHOENX_PYTHON` and does
not ask. A Start Menu launch asks once. A wheel of PhoenX is a separate
release; this installer is not that wheel.
