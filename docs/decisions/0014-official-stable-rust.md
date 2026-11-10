# 0014. The window is built with official stable Rust

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The window shell is a Tauri program. It needs a Rust compiler. A random
nightly toolchain would make the build unreproducible.

## Options considered

1. **The official stable toolchain from rustup** — the channel Tauri
   documents. Good: one known compiler. Bad: the machine that builds the
   installer must install it.
2. **Whatever `rustc` is already on the path** — no install step. Bad:
   an old or nightly compiler can fail the same sources.

## Decision

Build the window with the official stable Rust toolchain. Python work
does not need Rust. The installer build does.

## Why

Stable is the toolchain Tauri tests against. Pinning the channel avoids
a build that works on one machine and fails on another.

## Consequences

A checkout can run the Python session and the Vitest screen tests with
no Rust. `npm run tauri build` needs the stable toolchain installed.
