# 0016. phoenx-config gui waits for the window

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The old command blocked inside the Tkinter main loop. Scripts and tests
that run `phoenx-config gui` expect it to return when the window closes,
with that window's exit code.

## Options considered

1. **Wait for the child** — `launch_window` runs the executable and
   returns its code. Good: the command still blocks. Bad: a terminal
   stays busy for the whole session.
2. **Detach and return immediately** — the terminal is free. Bad: the
   command cannot report the window's exit code, and a script cannot
   tell when the edit finished.

## Decision

`phoenx-config gui` starts the window and waits until it exits. The
child's exit code is the command's exit code. `--doctor` does not start
the window.

## Why

That is the behavior the Tkinter command already had. Changing it would
surprise every caller.

## Consequences

Closing the terminal can close the window, because the window is a child
of the command. A Start Menu launch is not a child of that command.
