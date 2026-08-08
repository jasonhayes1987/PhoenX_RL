"""Unit tests for ``src/phoenx/schedulers.py``.

These tests exercise the real ``ScheduleWrapper`` API directly so that any
refactor of the schedule/progress split is automatically validated by
re-running this file.

``ScheduleWrapper`` deliberately separates configuration from progress:
``get_config()`` returns only the constructor arguments needed to rebuild
the schedule (``schedule_type``, ``steps``, ``start_value``, ``end_value``,
plus extra kwargs), while ``get_state()`` / ``set_state()`` carry the
underlying PyTorch scheduler's mutable progress (``last_epoch`` and
friends). Rebuilding from ``get_config()`` alone therefore always restarts
the schedule at step 0 — the negative-case test below pins exactly that,
since it is the whole reason ``get_state`` / ``set_state`` exist.
"""

from __future__ import annotations

import pytest

from phoenx.schedulers import ScheduleWrapper

# -----------------------------------------------------------------------------
# A single (start, end, steps, stepped-iters) shape whose factor lands clearly
# between the start and end value for every schedule kind: linear interpolates
# directly, cosine anneals from the dummy optimizer's base LR (1.0) down to
# ``end_value``, and exponential decays geometrically from the same base LR.
# -----------------------------------------------------------------------------
SCHEDULE_TYPES = ["linear", "cosine", "exponential"]
TOTAL_STEPS = 100
STEPPED_ITERS = 30
START_VALUE = 1.0
END_VALUE = 0.1


def _build_wrapper(schedule_type: str) -> ScheduleWrapper:
    """Construct a ``ScheduleWrapper`` of ``schedule_type`` with no optimizer.

    Args:
        schedule_type: One of ``"linear"``, ``"cosine"``, ``"exponential"``.

    Returns:
        A fresh wrapper over an internal dummy SGD optimizer.
    """
    return ScheduleWrapper(
        schedule_type=schedule_type,
        steps=TOTAL_STEPS,
        start_value=START_VALUE,
        end_value=END_VALUE,
    )


@pytest.mark.parametrize("schedule_type", SCHEDULE_TYPES)
class TestScheduleWrapperResume:
    """Pin the ``get_state`` / ``set_state`` resume path for every schedule kind."""

    def test_state_roundtrip_matches_stepped_original(self, schedule_type):
        """A wrapper rebuilt from config + state matches the stepped original.

        Steps an original wrapper ``STEPPED_ITERS`` times, rebuilds a fresh
        wrapper from ``get_config()`` alone, then restores progress via
        ``set_state(get_state())``. The restored wrapper's factor and its
        underlying scheduler's ``last_epoch`` must both match the original.

        Args:
            schedule_type: Parametrized schedule kind under test.
        """
        original = _build_wrapper(schedule_type)
        original.step(STEPPED_ITERS)

        restored = ScheduleWrapper.from_config(original.get_config())
        restored.set_state(original.get_state())

        assert restored.get_factor() == pytest.approx(original.get_factor())
        assert restored.scheduler.last_epoch == original.scheduler.last_epoch

    def test_config_alone_does_not_resume_progress(self, schedule_type):
        """Rebuilding from ``get_config()`` alone restarts at step 0.

        This is the negative case that motivates the whole ``get_state`` /
        ``set_state`` mechanism: a wrapper rebuilt from config only, without
        restoring state, must sit back at the start of the schedule rather
        than matching the stepped original.

        Args:
            schedule_type: Parametrized schedule kind under test.
        """
        original = _build_wrapper(schedule_type)
        original.step(STEPPED_ITERS)

        restarted = ScheduleWrapper.from_config(original.get_config())

        assert restarted.scheduler.last_epoch == 0
        assert abs(restarted.get_factor() - original.get_factor()) > 1e-6, (
            "Config-only rebuild unexpectedly matched the stepped original's "
            "factor; get_state/set_state would be pointless"
        )


class TestScheduleWrapperStateGuards:
    """Pin edge-case behavior of ``get_state`` / ``set_state`` directly."""

    def test_set_state_none_leaves_scheduler_untouched(self):
        """``set_state(None)`` is a no-op per the guard in ``ScheduleWrapper.set_state``."""
        wrapper = _build_wrapper("linear")
        wrapper.step(STEPPED_ITERS)
        factor_before = wrapper.get_factor()
        epoch_before = wrapper.scheduler.last_epoch

        wrapper.set_state(None)

        assert wrapper.get_factor() == pytest.approx(factor_before)
        assert wrapper.scheduler.last_epoch == epoch_before

    def test_get_state_contains_last_epoch(self):
        """``get_state()`` returns the scheduler's ``state_dict`` including ``last_epoch``."""
        wrapper = _build_wrapper("linear")

        state = wrapper.get_state()

        assert isinstance(state, dict)
        assert "last_epoch" in state
        assert state["last_epoch"] == 0
