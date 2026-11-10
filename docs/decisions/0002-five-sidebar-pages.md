# 0002. The sidebar has five pages

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The previous window had three tabs: a training config, a sweep file, and a
run. The new window also needs a place for a future summary and a future
settings page.

## Options considered

1. **Five pages** — Dashboard, Config Builder, Sweep Builder, Launch,
   Settings. Launch is the old run tab. Good: the names say what the page
   is for. Bad: two pages start empty.
2. **Keep three tabs** — less navigation. Bad: no place for a summary or
   settings without adding tabs later.
3. **A separate results page** — a mockup showed charts. Bad: those charts
   are not built, and a page with no data would look finished when it is not.

## Decision

The sidebar is Dashboard, Config Builder, Sweep Builder, Launch, Settings.
There is no results page.

## Why

Config, sweep, and launch are the work the window does today. The empty
pages keep a stable map so later work has a home.

## Consequences

Users land on an editor, not on an empty dashboard. Adding a real dashboard
later fills a page that already exists.
