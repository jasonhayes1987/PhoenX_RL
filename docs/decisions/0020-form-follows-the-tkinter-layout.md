# 0020. The form follows the Tkinter editor's layout rules

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The first window drew the session's form tree with its own rules. Sections
deeper than two levels started closed and tagged items had no open button,
so a layer's parameters could never be shown. Adding a branch or a layer
did not open anything. Text boxes showed the session's last answer while you
typed, so fast typing lost characters. Errors from the session were dropped,
so a failed edit looked like nothing happened.

## Options considered

1. **Copy the Tkinter editor's rules** — they were tested and in use.
   Good: no new behavior to learn or test. Bad: some rules (the depth that
   starts closed) are arbitrary.
2. **Design new rules for the window** — Bad: every rule is a new chance
   to hide a field, as the first version did.

## Decision

`desktop/src/form.tsx` follows `FormView` in `phoenx/config_gui.py` as of
549e5d0. The root is not a section. A section starts closed when its path has
three or more parts. Choosing a type, adding an item, or turning an optional
field on opens that section. A list item is a box with its number, ^ v x,
and its type dropdown with the type's fields right under it, so a layer's
parameters are always visible once its list is open. A scalar is a label,
an input, and a grey `default X` hint; hovering shows the description.

Text boxes keep what you typed and send it through one ordered queue. Edits
to one field that pile up while a call is running are sent once.
Every session error is shown in a dialog.

## Why

The Tkinter rules are the behavior people already relied on. The window's
job is to look better, not to behave differently.

## Consequences

`form.test.tsx` checks these rules. A change to them is a change to this
record. The session sends what the rules need: `default`, `bad`, `combo`,
`inner_kind`, `open`, `variant_path`, `path_keys`, and `generation`, which
resets which sections are open when a document is replaced.
