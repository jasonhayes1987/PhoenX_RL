# 0008. Confirmation questions stay in Python

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

Saving, opening, and closing ask the user before they throw work away.
Those rules used to live in the screen. Two different questions were easy
to mix up: throwing away edits, and saving before a hand-off.

## Options considered

1. **The session returns a need, and the window sends an answer** — the
   rule stays next to the file it protects. Good: the screen cannot skip
   a question. Bad: every dialog is a round trip.
2. **The screen decides** — faster dialogs. Bad: a screen bug can save or
   discard without the check the session expects.

## Decision

A method that needs a decision returns a need. The next call carries
`answer`. `discard` (yes, no, or cancel) is for a new file or an open.
`save-first` (yes or no) is for a hand-off. No on save-first keeps the
edits.

## Why

The dangerous action is the one that writes or drops a file. That code
already lives in the session. The screen only shows the title and the
message and sends the button back.

## Consequences

A new question is a new need id in the session, plus a button in the
dialog. The screen does not invent a third answer.
