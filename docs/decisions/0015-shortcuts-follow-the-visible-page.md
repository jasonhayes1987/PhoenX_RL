# 0015. Ctrl+S and Ctrl+O follow the visible page

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The old window bound those keys to both editors. The sweep editor was
created second, so the keys saved or opened the sweep even while the
config or the run was showing.

## Options considered

1. **The visible editor only** — Config Builder or Sweep Builder receives
   the key. Launch, Dashboard, and Settings ignore it. Good: the file
   on screen is the file that changes. Bad: you cannot save the other
   document without switching to it.
2. **Copy the old binding** — both documents listen. Bad: the hidden
   document wins, which is what the old window did by accident.

## Decision

Ctrl+S saves the visible editor page. Ctrl+O opens into that page.
Neither key fires for the other document, and neither fires off an
editor page.

## Why

The key should affect the page the user is looking at.

## Consequences

Saving the sweep while Config Builder is open requires a switch to
Sweep Builder first.
