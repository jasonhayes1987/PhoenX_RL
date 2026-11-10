# 0012. Desktop code is part of the same tree

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The window is a program in `desktop/`, not a Python module. It still
ships with PhoenX and can break `phoenx-config gui`.

## Options considered

1. **Keep `desktop/` in this repository** — the screen, the session, and
   the launcher change together. Good: one checkout builds both. Bad:
   the tree contains Node and Rust files.
2. **A separate repository** — the Python package stays smaller. Bad:
   the command and the window can be released out of step.

## Decision

The window source lives in `desktop/` in this repository. `node_modules`,
Rust `target`, and lockfile noise are not part of a review of the screen.

## Why

The session protocol and the screen have to match. Putting them in one
tree makes that review possible.

## Consequences

A change to a button and a change to the Python method land together.
Building the installer still needs Node and Rust on the machine that
builds it.
