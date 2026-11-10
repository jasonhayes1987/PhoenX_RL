# 0021. Overview cards are shortcuts into the one form

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

Config Builder and Sweep Builder open on an overview: Agent, Environment
and Schedule cards, a roots → trunk → branches diagram, and phase cards
with a table of what each phase searches. A hand-made screen like that can
drift from the schema, or show settings an agent never reads.

## Options considered

1. **Cards that read the session's form** — each card field is a path into
   the same snapshot the full form uses, and edits go through the same
   calls. Good: no second copy of any value. Bad: the card lists still have
   to be chosen.
2. **Hand-written fields per screen** — looks the same at first. Bad: every
   schema change becomes a screen change, and values can disagree.

## Decision

The session sends `cards` with each Config Builder snapshot: the Agent card's
fields for the chosen agent (`AGENT_CARD`), the Schedule fields that agent
uses (`schedule_card`), the Environment fields the backend has, and the
normalizers. On-policy agents (`ON_POLICY_AGENTS`, the same split as
`metrics.py`) do not show `batch_size`, `updates_per_learn` or
`warmup_steps`, which rollout buffers ignore. Only agents whose `learn()`
takes them (`EPOCH_AGENTS`) show `learning_epochs` and `mini_batch_size`.
The diagram and the inspector draw the model section of the same form; the
inspector edits the selected block with the normal form controls.

## Why

The overview should make a run quick to set up without becoming a second
config editor that can disagree with the first.

## Consequences

`tests/test_gui_session.py` checks that every card path is a real field for
every agent, and that the on/off-policy and epoch splits match the agents'
code. "All fields" still shows everything, with the Tkinter layout rules of
[0020](0020-form-follows-the-tkinter-layout.md).
