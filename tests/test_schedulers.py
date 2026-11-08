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
import torch as T

from phoenx.schedulers import ScheduleWrapper

# -----------------------------------------------------------------------------
# Resume tests use start_value 1.0 so every kind's factor moves from 1.0
# toward 0.1. end/start equals end_value in that special case, which hides
# the cosine coefficient bug. Endpoint coverage for start_value != 1 is in
# TestScheduleWrapperEndpoints.
# -----------------------------------------------------------------------------
SCHEDULE_TYPES = ["linear", "cosine", "exponential"]
TOTAL_STEPS = 100
STEPPED_ITERS = 30
START_VALUE = 1.0
END_VALUE = 0.1
COEFF_STEPS = 1000
COEFF_START = 0.02
COEFF_END = 0.001
LR_STEPS = 1000
LR_START = 3.0e-4
LR_END = 1.0e-5

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

def _coefficient_wrapper(schedule_type: str) -> ScheduleWrapper:
    """Build a no-optimizer wrapper, the coefficient-schedule path.
    Args:
        schedule_type: One of ``"linear"``, ``"cosine"``, ``"exponential"``.
    Returns:
        A wrapper over the internal dummy SGD.
    """
    return ScheduleWrapper(
        schedule_type=schedule_type,
        steps=COEFF_STEPS,
        start_value=COEFF_START,
        end_value=COEFF_END,
    )

def _learning_rate_wrapper(schedule_type: str):
    """Build a wrapper and attach a real Adam, the learning-rate path.
    The wrapper is constructed with no optimizer first. ``attach_optimizer``
    then rebinds it, which is the order ``ModularModel`` uses. That second
    construction is what a ``self._param is not None`` check gets wrong.
    Args:
        schedule_type: One of ``"linear"``, ``"cosine"``, ``"exponential"``.
    Returns:
        The attached wrapper and the Adam optimizer it now drives.
    """
    param = T.nn.Parameter(T.zeros(1))
    optimizer = T.optim.Adam([param], lr=LR_START)
    wrapper = ScheduleWrapper(
        schedule_type=schedule_type,
        steps=LR_STEPS,
        start_value=LR_START,
        end_value=LR_END,
    )
    wrapper.attach_optimizer(optimizer)
    return wrapper, optimizer

def test_cosine_coefficient_midpoint_is_halfway_between_endpoints():
    """A cosine coefficient is halfway in value at half the horizon.
    With the dummy optimizer, half of 1,000 steps from 0.02 toward 0.001 is
    0.0105. That is the closed cosine midpoint, so the whole curve is scaled
    by ``end / start``, not only the final sample.
    """
    wrapper = _coefficient_wrapper("cosine")
    wrapper.step(COEFF_STEPS // 2)
    midpoint = 0.5 * (COEFF_START + COEFF_END)
    assert COEFF_START * wrapper.get_factor() == pytest.approx(midpoint)


@pytest.mark.parametrize("schedule_type", SCHEDULE_TYPES)
class TestScheduleWrapperEndpoints:
    """Pin the finished value of every schedule kind.
    Coefficient schedules scale ``start_value`` by ``get_factor()`` and must
    finish at ``end_value``. Learning-rate schedules are attached to a real
    optimizer whose base lr is ``start_value``, and that lr must finish at
    ``end_value`` too.
    """
    def test_coefficient_starts_at_start_value_and_ends_at_end_value(
        self, schedule_type
    ):
        """``start_value * get_factor()`` runs from the start value to the end value.
        Args:
            schedule_type: Parametrized schedule kind under test.
        """
        wrapper = _coefficient_wrapper(schedule_type)
        assert COEFF_START * wrapper.get_factor() == pytest.approx(COEFF_START)
        wrapper.step(COEFF_STEPS)
        assert wrapper.scheduler.last_epoch == COEFF_STEPS
        assert COEFF_START * wrapper.get_factor() == pytest.approx(COEFF_END)
    def test_attached_optimizer_learning_rate_ends_at_end_value(self, schedule_type):
        """An attached optimizer's lr runs from the start value to the end value.
        Args:
            schedule_type: Parametrized schedule kind under test.
        """
        wrapper, optimizer = _learning_rate_wrapper(schedule_type)
        learning_rate = optimizer.param_groups[0]["lr"]
        assert learning_rate == pytest.approx(LR_START)
        assert wrapper.get_factor() == pytest.approx(LR_START)
        wrapper.step(LR_STEPS)
        learning_rate = optimizer.param_groups[0]["lr"]
        assert wrapper.scheduler.last_epoch == LR_STEPS
        assert learning_rate == pytest.approx(LR_END)
        assert wrapper.get_factor() == pytest.approx(learning_rate)


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
