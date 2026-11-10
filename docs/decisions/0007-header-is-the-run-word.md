# 0007. The header's right side is the run status word

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

A mockup put a clock, a GPU name, and a connection mark in the header.
None of those are measured by the window.

## Options considered

1. **Show Idle, Running, Stopping, Stopped, or Failed** — the word the
   session already computes. Good: it matches the log. Bad: the header
   looks sparse.
2. **Add decorative meters** — Good: the header looks like a console.
   Bad: the meters would not be real.

## Decision

The right side of the header is the run status word and nothing else.

## Why

The status is the one fact the session updates for the whole window.
A decoration that does not change with the run is noise.

## Consequences

GPU and clock, if they are added later, need a real source and a place
that is not this slot.
