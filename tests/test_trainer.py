"""Unit tests for ``Trainer.step``'s "best episode" bookkeeping and the
``TrainingSchedule.save_every`` checkpoint cooldown in ``src/phoenx/trainer.py``.

Pins the one-line fix that sets ``episode_logs[-1]['best'] = True`` whenever a
completed training episode pushes the running average reward above the
previous best, plus the follow-up fix that gates the resulting checkpoint
``save()`` behind ``TrainingSchedule.should_save`` (a minimum-timesteps
cooldown) instead of firing on every new best. A rolling 100-episode average
climbs almost monotonically early in training with many parallel envs, so
"new best" was true on nearly every step and the ungated version wrote a full
checkpoint (and uploaded a full W&B artifact) almost continuously.

``Trainer.step`` is driven directly through minimal stand-in ``agent``/``env``/
``buffer`` objects that script a fixed reward/termination per call, rather
than a real Gymnasium env or agent — fast, deterministic, no I/O.

Also pins two further fixes to the ``save_every`` cooldown: (1) the
checkpoint ``self.save()`` writes inside ``step`` must be persisted AFTER
``_last_save``/``_best_pending`` are updated to their new values, not
before, so ``trainer_state.pt`` never records a stale (previous-window)
cooldown state; and (2) ``Trainer.train()`` flushes any best still awaiting
its cooldown once the training loop exits, bypassing the cooldown, so a
peak is never silently dropped just because the run ended before the next
episode boundary.
"""

from __future__ import annotations

import os
from collections import deque

import torch as T

from phoenx.env_wrapper import Action, Observation
from phoenx.trainer import Trainer, TrainingSchedule


class _FakeAgent:
    """Minimal agent stand-in: no normalizers, no intrinsic motivation, and an
    ``act`` that always returns a fixed action regardless of the observation.
    """

    def __init__(self):
        self.device = "cpu"
        self.intrinsic_motivation = None

    def act(self, states, *_args, **_kwargs):
        n = states.shape[0]
        return Action(actions=T.zeros(n, 1))

    def get_config(self):
        """Minimal stub so a real (non-monkeypatched) ``Trainer.save()`` can
        serialize a config for the explicit-save test below."""
        return {"type": "FakeAgent"}

    def save_state(self, path):
        """Minimal stub so a real ``Trainer.save()`` succeeds without a real
        agent's weights/optimizers/normalizers to persist."""
        os.makedirs(path, exist_ok=True)


class _FakeEnv:
    """Single-env stand-in whose ``step`` always returns a scripted reward and
    termination, independent of the action it receives.
    """

    def __init__(self, reward: float, done: bool = True):
        self.num_envs = 1
        self.goal_key = None
        self.config = {"type": "FakeEnv"}
        self._reward = reward
        self._done = done

    def _find_nstep_wrapper(self):
        return None

    def step(self, _actions):
        return Observation(
            states=T.zeros(1, 4),
            rewards=T.tensor([self._reward]),
            terminations=T.tensor([self._done]),
            truncations=T.zeros(1, dtype=T.bool),
        )

    def close(self):
        """Minimal stub so a real (non-monkeypatched) ``Trainer.train()``
        can exit without a real env to tear down."""
        pass


class _FakeBuffer:
    def record(self, *args, **kwargs):
        pass

    def get_config(self):
        """Minimal stub so a real (non-monkeypatched) ``Trainer.save()`` can
        serialize a config for the explicit-save test below."""
        return None


def _make_trainer(reward: float, best_reward: float, tmp_path, score_history=None) -> Trainer:
    """Build a real ``Trainer`` wired to the fakes above, with the internal
    episode-tracking state ``_initialize_run`` would normally set pre-seeded
    by hand so the test can drive ``step`` without booting a real env/agent.
    """
    trainer = Trainer(
        agent=_FakeAgent(),
        env=_FakeEnv(reward=reward),
        schedule=TrainingSchedule(),
        success_criterion=None,
        buffer=_FakeBuffer(),
        renderer=None,
        callbacks=None,
        save_dir=str(tmp_path),
    )
    trainer._step = 0
    trainer._prev_obs = Observation(states=T.zeros(1, 4))
    trainer._prev_done = T.zeros(1, dtype=T.bool)
    trainer._best_reward = best_reward
    trainer._episode_steps = T.zeros(1, dtype=T.int32)
    trainer._completed_episodes = T.zeros(1, dtype=T.int32)
    trainer._episode_scores = T.zeros(1, dtype=T.float32)
    trainer._score_history = deque(score_history or [], maxlen=100)
    return trainer


def test_step_marks_new_best_episode_and_saves(tmp_path, monkeypatch):
    """A completed training episode whose avg reward beats the running best
    carries `best: True` on its episode log and triggers `Trainer.save()`."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert episode_log.get("best") is True
    assert saved == [True]
    assert trainer._best_reward == 5.0


def test_step_omits_best_key_when_average_not_improved(tmp_path, monkeypatch):
    """A completed episode that does NOT beat the running best omits the
    `best` key entirely (absence, not `False`) and never calls `save()`."""
    trainer = _make_trainer(reward=1.0, best_reward=100.0, tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert "best" not in episode_log
    assert saved == []
    assert trainer._best_reward == 100.0


def test_step_testing_mode_never_marks_best_or_saves(tmp_path, monkeypatch):
    """`training=False` suppresses both the `best` flag and the save, even
    though the average would otherwise beat the (still `-inf`) best."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=False)

    episode_log = result["episode_logs"][-1]
    assert "best" not in episode_log
    assert saved == []
    assert trainer._best_reward == float("-inf")


def test_step_no_completed_episode_leaves_logs_and_best_untouched(tmp_path, monkeypatch):
    """When no env reaches a terminal/truncated state, `episode_logs` stays
    empty and neither the best-tracking nor `save()` fires."""
    trainer = _make_trainer(reward=5.0, best_reward=float("-inf"), tmp_path=tmp_path)
    trainer.env._done = False  # scripted env: no termination this step
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    assert result["episode_logs"] == []
    assert saved == []
    assert trainer._best_reward == float("-inf")


# =============================================================================
# TrainingSchedule.should_save: the standalone cooldown predicate.
# =============================================================================


def test_should_save_true_when_no_automatic_save_has_happened_yet():
    """`last_save_at=None` (no automatic save yet this run) always allows the
    first save, regardless of `step` or `save_every`."""
    schedule = TrainingSchedule(save_every=50_000)
    assert schedule.should_save(step=0, last_save_at=None) is True
    assert schedule.should_save(step=10**9, last_save_at=None) is True


def test_should_save_refuses_inside_the_window():
    schedule = TrainingSchedule(save_every=1_000)
    assert schedule.should_save(step=999, last_save_at=0) is False


def test_should_save_allows_exactly_at_the_boundary():
    """`step == last_save_at + save_every` must be allowed, not refused — an
    off-by-one here would silently double (or halve) the real save cadence."""
    schedule = TrainingSchedule(save_every=1_000)
    assert schedule.should_save(step=1_000, last_save_at=0) is True


def test_should_save_allows_past_the_window():
    schedule = TrainingSchedule(save_every=1_000)
    assert schedule.should_save(step=1_001, last_save_at=0) is True


def test_should_save_zero_always_allows():
    """`save_every=0` disables the cooldown entirely (the old
    save-on-every-best behavior), independent of how close `step` is to
    `last_save_at`."""
    schedule = TrainingSchedule(save_every=0)
    assert schedule.should_save(step=0, last_save_at=0) is True
    assert schedule.should_save(step=5, last_save_at=5) is True


# =============================================================================
# TrainingSchedule.get_config(): save_every is included and round-trips.
# =============================================================================


def test_schedule_get_config_includes_save_every():
    schedule = TrainingSchedule(save_every=12_345)
    assert schedule.get_config()["save_every"] == 12_345


def test_schedule_get_config_round_trip_preserves_save_every():
    """A schedule that failed to round-trip `save_every` through
    `get_config()`/reconstruction would silently revert to the default
    (50,000) on reload — e.g. every `Trainer.load()` call."""
    schedule = TrainingSchedule(save_every=7_777)
    rebuilt = TrainingSchedule(**schedule.get_config())
    assert rebuilt.save_every == 7_777


# =============================================================================
# Trainer.step: the save_every cooldown gating the automatic best-model save.
#
# `_make_trainer` never sets `_last_save` (it defaults to `None` from
# `Trainer.__init__`), so the tests above this section only ever exercise the
# unconditional "first save" path (`last_save_at is None`). The tests below
# explicitly set `_last_save` (and, for some, `_best_pending`) to actually
# drive the gated path.
# =============================================================================


def test_step_new_best_inside_cooldown_window_defers_save(tmp_path, monkeypatch):
    """A new best that lands inside the cooldown window still sets `best` on
    the episode log (it IS a new rolling-average peak), but does NOT call
    `save()` and does NOT set `saved`."""
    trainer = _make_trainer(reward=5.0, best_reward=1.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 1_000
    trainer._last_save = 0
    trainer._step = 500  # inside the window: 500 < 0 + 1000
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert episode_log.get("best") is True
    assert "saved" not in episode_log
    assert saved == []
    assert trainer._best_pending is True
    assert trainer._last_save == 0  # unchanged


def test_step_pending_best_saved_once_cooldown_window_elapses(tmp_path, monkeypatch):
    """Once `step` has advanced by at least `save_every` timesteps since
    `_last_save`, a still-pending best IS written: `save()` fires, `saved` is
    set on the episode log, and `_last_save` advances to the current step."""
    trainer = _make_trainer(reward=1.0, best_reward=100.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 1_000
    trainer._last_save = 0
    trainer._step = 1_000  # exactly at the boundary: last_save_at + save_every
    trainer._best_pending = True  # a best from an earlier step is still pending
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    # This step's own average (1.0) does not beat the recorded best (100.0);
    # the save fires purely because a *prior* best is still pending.
    assert "best" not in episode_log
    assert episode_log.get("saved") is True
    assert saved == [True]
    assert trainer._last_save == 1_000
    assert trainer._best_pending is False


def test_step_best_inside_window_is_not_lost_once_cooldown_elapses(tmp_path, monkeypatch):
    """The whole reason `_best_pending` exists: a new best that lands inside
    the cooldown window must NOT be lost even if no further best occurs
    afterwards. `_best_reward` advances unconditionally, so without the
    pending flag this peak would never be checkpointed once performance later
    degraded (the next completed episode would see `avg_reward <=
    _best_reward` and skip the save entirely)."""
    trainer = _make_trainer(reward=5.0, best_reward=1.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 1_000
    trainer._last_save = 0
    trainer._step = 500  # inside the window
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    # Step 1: a new best lands inside the cooldown window -> deferred.
    result1 = trainer.step(training=True)
    assert result1["episode_logs"][-1].get("best") is True
    assert "saved" not in result1["episode_logs"][-1]
    assert saved == []  # not written yet

    # Step 2: performance drops back down (no further best), but the
    # cooldown window has now elapsed.
    trainer.env._reward = 0.0
    trainer._step = 1_000
    result2 = trainer.step(training=True)

    assert "best" not in result2["episode_logs"][-1]
    assert result2["episode_logs"][-1].get("saved") is True
    # The peak from step 1 was still written to disk -- it was never lost.
    assert saved == [True]


def test_step_save_every_zero_saves_on_every_best(tmp_path, monkeypatch):
    """`save_every=0` reproduces the pre-fix save-on-every-best behavior,
    even when a previous automatic save happened on the immediately
    preceding step."""
    trainer = _make_trainer(reward=5.0, best_reward=1.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 0
    trainer._last_save = 0  # a previous save already happened, one step ago
    trainer._step = 1
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    result = trainer.step(training=True)

    episode_log = result["episode_logs"][-1]
    assert episode_log.get("best") is True
    assert episode_log.get("saved") is True
    assert saved == [True]
    assert trainer._last_save == 1





# =============================================================================
# Trainer.step + Trainer.save: `trainer_state.pt` must record the cooldown
# state AFTER it is updated for this call, not the previous window's values.
#
# `self.save()` inside the automatic best-model path used to run BEFORE
# `_last_save`/`_best_pending` were reassigned, so a checkpoint written at a
# window boundary persisted the *previous* window's cooldown state while
# memory already held the new one. A run resumed from that checkpoint
# inherited the stale state and could immediately re-save on its very next
# completed episode regardless of whether a new best had occurred.
# =============================================================================


def test_step_persists_updated_cooldown_state_not_the_stale_values(tmp_path, monkeypatch):
    """The `trainer_state.pt` written by the automatic best-model save must
    match the in-memory `_last_save`/`_best_pending` computed by THIS call to
    `step`, not the pre-call (previous window's) values. Then: resuming from
    that checkpoint and completing one more episode must NOT immediately save
    again just because the persisted cooldown state was stale.
    """
    trainer = _make_trainer(reward=1.0, best_reward=100.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 50_000
    trainer._last_save = 10_000
    trainer._step = 60_000  # exactly at the boundary: last_save_at + save_every
    trainer._best_pending = True  # a best from an earlier window is still pending

    trainer.step(training=True)  # real save() call -- NOT monkeypatched

    assert trainer._last_save == 60_000
    assert trainer._best_pending is False

    persisted = T.load(tmp_path / "trainer_state.pt", weights_only=False)
    # The crux of the fix: the file on disk must match what's in memory
    # right now (60_000 / False), not the pre-save values (10_000 / True).
    assert persisted["_last_save"] == 60_000
    assert persisted["_best_pending"] is False

    # Overlay the persisted state onto a freshly constructed trainer via the
    # real resume path (`Trainer.load` stages `_resume_state`; `train`'s
    # `_initialize_run` calls `_apply_resume_state`).
    resumed = _make_trainer(reward=0.0, best_reward=float("-inf"), tmp_path=tmp_path)
    resumed._resume_state = persisted
    resumed._apply_resume_state(context="train")
    assert resumed._last_save == 60_000
    assert resumed._best_pending is False

    saved = []
    monkeypatch.setattr(resumed, "save", lambda *a, **k: saved.append(True))
    result = resumed.step(training=True)  # the "next completed episode" after resume

    assert saved == []
    assert "saved" not in result["episode_logs"][-1]


def test_explicit_save_call_is_never_gated_by_the_cooldown(tmp_path):
    """`Trainer.save()` called directly (not through `step`'s automatic
    best-model path) always writes to disk, regardless of what
    `schedule.should_save` would say for the current `_step`/`_last_save` --
    only the automatic save inside `step()` consults the cooldown."""
    trainer = _make_trainer(reward=5.0, best_reward=100.0, tmp_path=tmp_path)
    trainer.schedule.save_every = 1_000_000  # a huge cooldown window
    trainer._last_save = trainer._step  # would refuse an automatic save right now
    assert trainer.schedule.should_save(step=trainer._step, last_save_at=trainer._last_save) is False

    trainer.save()  # real call, not monkeypatched

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "trainer_state.pt").exists()


# =============================================================================
# Trainer.train(): the end-of-run pending-best flush.
#
# The `_best_pending` gate inside `step()` only fires at an episode boundary
# once the cooldown window has elapsed, so a best still pending when the
# training loop exits was previously never written at all -- there is no
# further episode boundary to retry the save at. `train()` now flushes any
# still-pending best once the loop exits, bypassing `schedule.should_save`
# entirely (the run is over; there is no "next window" to wait for).
#
# These tests drive the real `Trainer.train()` loop rather than `step()`
# directly. Setting `_initialized = True` (never true for a freshly
# constructed `Trainer`) makes `_initialize_run` return immediately without
# resetting `_best_pending`/`_last_save`/etc, so the pre-seeded state below
# survives into the loop untouched -- the same pre-seeding trick
# `_make_trainer` already uses for the `step()` tests above, extended to
# `train()`. `schedule.stop_units = 0` makes `is_done` true on the very
# first check, so the loop body runs zero times: `step()` is never called,
# isolating the end-of-run flush from `step()`'s own (already-covered)
# mid-run gating.
# =============================================================================


def _make_ended_trainer(tmp_path, *, best_pending: bool) -> Trainer:
    """A trainer configured so `train()`'s loop runs zero iterations (the
    run ends immediately), with `_best_pending` pre-seeded for the
    end-of-run flush check below.
    """
    trainer = _make_trainer(reward=0.0, best_reward=float("-inf"), tmp_path=tmp_path)
    trainer._initialized = True
    trainer.schedule.stop_unit = "timestep"
    trainer.schedule.stop_units = 0  # is_done(step=..., ...) is True immediately
    trainer.schedule.save_every = 50_000  # large cooldown the flush must bypass
    trainer._last_save = 0
    trainer._step = 12_345
    trainer._best_pending = best_pending
    return trainer


def test_train_flushes_pending_best_at_run_end_bypassing_cooldown(tmp_path, monkeypatch):
    """A run that ends with `_best_pending=True` writes exactly one final
    checkpoint and clears the flag -- even though `schedule.should_save`
    would refuse a save at this `_step`/`_last_save` (the flush deliberately
    bypasses the cooldown: there is no further episode boundary to retry
    at)."""
    trainer = _make_ended_trainer(tmp_path, best_pending=True)
    assert trainer.schedule.should_save(step=trainer._step, last_save_at=trainer._last_save) is False
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    trainer.train()

    assert saved == [True]
    assert trainer._best_pending is False
    assert trainer._last_save == 12_345


def test_train_does_not_flush_when_nothing_pending_at_run_end(tmp_path, monkeypatch):
    """The negative case that stops the flush from firing unconditionally:
    a run that ends with `_best_pending=False` writes NO extra checkpoint."""
    trainer = _make_ended_trainer(tmp_path, best_pending=False)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    trainer.train()

    assert saved == []
    assert trainer._last_save == 0  # unchanged


def test_train_run_that_never_steps_does_not_raise(tmp_path, monkeypatch):
    """A run whose loop never executes even once -- `step()` is never
    called, so `_best_pending` is never flipped to `True` -- must exit
    `train()` cleanly (in particular, the flush check must not blow up on a
    trainer that never stepped)."""
    trainer = _make_ended_trainer(tmp_path, best_pending=False)
    trainer._step = 0
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *a, **k: saved.append(True))

    trainer.train()  # must not raise

    assert saved == []
