# 0013. The screen is tested with Vitest

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The Python session is tested with pytest. The screen is TypeScript. It
should not need a display or a built window to prove that a button calls
the right method.

## Options considered

1. **Vitest and Testing Library, with a fake session** — each control is
   clicked in jsdom. Good: fast, and the test names the method. Bad: it
   does not start the real window.
2. **Only a manual click-through** — Good: you see the real program.
   Bad: a renamed method can ship unnoticed.

## Decision

Python behavior is tested with pytest. Screen behavior is tested with
Vitest against a fake session. The tests do not import the desktop shell.

## Why

The contract of a button is the method name it sends. A fake session
records that without Rust or a window.

## Consequences

A real launch of `phoenx-rl.exe` is still a manual check. The tests do
not prove the installer, the icon files, or the native file dialog.
