# 0019. Rust only forwards the session pipes

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The window has to start the Python session and carry JSON lines to it.
A socket would be another listener on the machine. The form logic does
not belong in Rust.

## Options considered

1. **Spawn the session and forward stdin and stdout** — one child, no
   port. Good: the session is reachable only by this window. Bad: Rust
   has to keep the read and write in order.
2. **Open a TCP port** — other tools could connect. Bad: a port is a
   service, and a second client could interleave lines.

## Decision

The Rust side spawns `python -m phoenx.gui_session`, writes one request
line, and reads one response line. It does not open a TCP port. It does
not interpret the form.

## Why

The session already owns the rules. The shell only has to keep the pipes
paired and remember which Python the user picked.

## Consequences

Calls are serialized. A hung Python call hangs the button that issued
it. There is no network endpoint to secure or to document.
