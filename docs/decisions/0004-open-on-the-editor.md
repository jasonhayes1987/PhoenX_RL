# 0004. The window opens on an editor

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The first screen a person sees should be the work they opened the window
to do. A training file and a sweep file are different documents.

## Options considered

1. **Open on Config Builder**, or on Sweep Builder when the file is a sweep.
   Good: the file you named is on screen. Bad: Dashboard is not the landing
   page its name suggests.
2. **Open on Dashboard** — a home page. Bad: the user then has to find the
   file they just passed on the command line.

## Decision

A sweep file opens Sweep Builder. Every other start opens Config Builder.

## Why

`phoenx-config gui FILE` is a request to edit that file. The page should
match the file.

## Consequences

Dashboard is a destination, not a start. A file that is neither a config
nor a sweep is reported on Config Builder and is not loaded as a sweep.
