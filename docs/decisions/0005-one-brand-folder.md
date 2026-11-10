# 0005. Artwork comes from one brand folder

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The wordmark, the logo, and the five page icons will be replaced. If those
files are imported from many components, a swap touches the whole screen.

## Options considered

1. **One module under `desktop/src/brand/`** — every image is imported
   there and re-exported. Colors live in one stylesheet. Good: a swap is
   a file replace plus that module. Bad: components cannot pick an icon
   ad hoc.
2. **Import icons where they are drawn** — fewer files. Bad: a new logo
   means hunting through the screen.

## Decision

`desktop/src/brand/index.ts` is the only module that imports artwork.
`desktop/src/brand/theme.css` is the only place that sets the color tokens.

## Why

The mark is a placeholder. Replacing it should not be a code change
outside that folder.

## Consequences

A new page icon is added in the brand folder and listed in `nav`. A
component that needs a color uses a token, not a hex value.
