# 0009. One Python process owns the running command

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

Launch starts a training or sweep process and later stops it. The stop
only works if the same object that started the child is still alive.

## Options considered

1. **One long-lived process, one JSON line per call** — the process holds
   the form and the child. Good: Stop talks to the same child Start
   created. Bad: the window must keep that process running.
2. **One process per click** — each button starts Python and exits.
   Good: a crash does not stick. Bad: the next click cannot see the child.

## Decision

The window starts one `python -m phoenx.gui_session` and keeps it for the
life of the window. Requests and responses are single JSON lines on stdin
and stdout.

## Why

The runner has to remember the child, the log, and the unsaved form.
A fresh process on every click would forget all three.

## Consequences

Training output is not written to that stdout. It is carried inside
`launch.poll` results. A non-JSON line on the session stdout is a failure.
