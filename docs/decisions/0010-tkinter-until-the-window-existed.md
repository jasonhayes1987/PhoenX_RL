# 0010. The command kept the old window until the new one could open

- **Status:** Superseded by [0018](0018-one-window.md)
- **Date:** 2026-11-09

## Context

People were already using `phoenx-config gui`. Replacing it before a
window existed would have made the command do nothing.

## Options considered

1. **Keep the Tkinter window until the new program can be launched** —
   both can be compared. Good: the command keeps working. Bad: two
   windows exist for a while.
2. **Switch the command immediately** — the new screen is the only one.
   Bad: without a built program, the command fails.

## Decision

`phoenx-config gui` kept opening the Tkinter window until the desktop
window and its launcher were in place.

## Why

A command that people type should keep opening an editor while the
replacement is unfinished.

## Consequences

[0018](0018-one-window.md) removes the Tkinter window. This record stays
so the temporary overlap is explained. Do not bring the Tkinter window
back as a fallback.
