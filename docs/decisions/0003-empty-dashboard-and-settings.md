# 0003. Dashboard and Settings stay empty

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

A mockup showed status cards on a dashboard and a form on a settings page.
Neither card nor form has a backing feature yet.

## Options considered

1. **Leave both pages blank** — the title is the whole page. Good: the
   window does not pretend to show live data. Bad: the pages look unfinished.
2. **Copy the mockup cards** — decorative text such as a GPU reading.
   Good: the page looks busy. Bad: the numbers would be fake.

## Decision

Dashboard and Settings show their titles and no controls.

## Why

A fake reading is worse than an empty page. The Python prompt on first
launch is not Settings. It is a separate question about which interpreter
can import PhoenX.

## Consequences

There is nowhere in the window to change the saved Python path. Delete the
saved `python-path` file, or set `PHOENX_PYTHON`, to choose again.
