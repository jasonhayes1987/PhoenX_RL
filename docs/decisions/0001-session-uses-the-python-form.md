# 0001. The window calls the existing Python form

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

PhoenX already builds a config form from its schema, checks that form, and
writes YAML. A second copy of those rules in the window would drift.

## Options considered

1. **Call the Python form** — one implementation of fields, checks, and YAML.
   Good: a schema change shows up in the window. Bad: the window needs a
   running Python process.
2. **Rebuild the form in TypeScript** — the screen would own the rules.
   Good: no process hop for each keystroke. Bad: two copies of the schema.

## Decision

The window calls the existing Python form, sweep checks, and process runner.

## Why

The form is the product. The screen is a view of it. One copy of the rules
is easier to keep correct than a translated copy.

## Consequences

Every edit is a call into Python. A schema change does not require a window
change, unless the screen needs a new control for a new kind of field.
Revisit this if a call per keystroke becomes too slow.
