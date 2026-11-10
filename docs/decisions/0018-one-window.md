# 0018. There is one window implementation

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

[0010](0010-tkinter-until-the-window-existed.md) kept the Tkinter window
so the command still opened an editor. That overlap is over. Two
implementations mean two sets of dialogs and a choice the user should
not have to make.

## Options considered

1. **Delete the Tkinter window on every platform** — `phoenx-config gui`
   launches the desktop program. Where no binary has been built, the
   command prints `PhoenX window not found. Install it, or set PHOENX_GUI
   to the executable.` and returns 1. Good: one window. Bad: a machine
   with no build cannot edit until it builds or installs one.
2. **Keep Tkinter as a fallback** — the command still opens something.
   Bad: bugs get fixed in one window and not the other.

## Decision

The Tkinter modules are gone. There is no fallback.

## Why

One window means one place for a save prompt, a stop button, and a
shortcut. A missing binary is a clear error, not a surprise return to
the old screen.

## Consequences

`import phoenx.config_gui`, `import phoenx.sweep_editor`, and `RunPanel`
fail. `build_command` and `describe_run` live in `phoenx.launch_cmd`.
Building or installing the window is required before `phoenx-config gui`
can open a screen.
