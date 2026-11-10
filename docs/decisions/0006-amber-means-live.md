# 0006. Amber marks only a live action

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The screen uses a cyan accent for navigation and a second color for things
that are in progress. If that second color is used for ordinary buttons, it
stops meaning "this is running".

## Options considered

1. **Amber only for a live action** — the enabled Start button, the words
   Running and Stopping, and the selected sweep phase. Good: one meaning.
   Bad: Save and Validate do not get a highlight.
2. **Amber on Save and Validate** — those buttons look urgent. Bad: they
   are not a running process.

## Decision

Amber is only the enabled Start button, the Running or Stopping status,
and the selected sweep phase.

## Why

The color answers "what is active right now". Start is the action that
makes a run active. The selected phase is the one the next add will change.

## Consequences

Other buttons stay in the normal text color. A future warning color needs
a new token so it is not confused with a running job.
