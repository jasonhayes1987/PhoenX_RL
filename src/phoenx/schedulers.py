"""Learning-rate and scalar schedules wrapped around PyTorch schedulers.

``ScheduleWrapper`` owns a linear, cosine, or exponential schedule that can
drive either an optimizer's learning rate or a free-floating scalar (via an
internal dummy optimizer). Callers advance the schedule with ``step`` and
read the current factor through ``get_factor``. Configuration and live
progress are deliberately separate so a run can resume mid-schedule:
``get_config`` returns the constructor arguments, while ``get_state`` /
``set_state`` carry the underlying scheduler's progress.
"""

import torch as T
from torch import optim
from torch.optim import lr_scheduler, Optimizer

class ScheduleWrapper:
    """Wrap a linear, cosine, or exponential schedule over an optimizer."""
    def __init__(
        self,
        schedule_type: str,
        steps: int,
        start_value: float,
        end_value: float,
        optimizer: Optimizer|None = None,
        **kwargs
    ):
        """Build a schedule, optionally attached to an optimizer.

        When ``optimizer`` is ``None``, a dummy SGD optimizer over a single
        non-trainable parameter is created so the schedule can still produce
        a scalar factor without owning a real training optimizer. Call
        [attach_optimizer][phoenx.schedulers.ScheduleWrapper.attach_optimizer] later to rebind.

        Args:
            schedule_type: Scheduler kind: ``"linear"``, ``"cosine"``, or
                ``"exponential"``.
            steps: Number of steps over which the schedule runs.
            start_value: Starting value of the schedule.
            end_value: Ending value of the schedule.
            optimizer: Optimizer whose learning rate is scheduled, or ``None``
                to use an internal dummy optimizer.
            kwargs (dict): Extra keyword arguments forwarded to the underlying
                PyTorch scheduler constructor.
        """
        self.schedule_type = schedule_type
        self.steps = steps
        self.start_value = start_value
        self.end_value = end_value
        self.kwargs = kwargs
        self.optimizer = optimizer
        self.scheduler = None
        self._param = None
        self._last_epoch = 0
        
        if optimizer is None:
            self._param = T.nn.Parameter(T.zeros(1), requires_grad=False)
            self.optimizer = optim.SGD([self._param], lr=1.0)

        self._create_scheduler()
        
    def _create_scheduler(self):
        """Creates scheduler."""
        # Map scheduler type to PyTorch's built-in schedulers
        if self.schedule_type == "linear":
            if self.end_value is None or self.start_value is None or self.steps is None:
                raise ValueError("End value, start value, and steps are required for linear scheduler.")
            end_factor = self.end_value / self.start_value
            self.scheduler = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=self.steps,
                **self.kwargs
            )

        elif self.schedule_type == "cosine":
            if self.end_value is None or self.start_value is None or self.steps is None:
                raise ValueError("End value, start value, and steps are required for cosine scheduler.")
            # end_factor = self.end_value / self.start_value
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.steps,
                eta_min=self.end_value,
                **self.kwargs
            )

        elif self.schedule_type == "exponential":
            if self.end_value is None or self.start_value is None or self.steps is None:
                raise ValueError("End value, start value, and steps are required for exponential scheduler.")
            gamma = (self.end_value / self.start_value) ** (1.0 / self.steps)
            self.scheduler = lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma,
                **self.kwargs
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {self.schedule_type}")

        # Restore last epoch if there was a previous state
        if hasattr(self.scheduler, 'last_epoch') and self._last_epoch > 0:
            self.scheduler.last_epoch = self._last_epoch

    def attach_optimizer(self, optimizer: Optimizer):
        """Attach an optimizer and rebuild the underlying scheduler.

        Preserves ``last_epoch`` from the current scheduler so progress is not
        reset when rebinding.

        Args:
            optimizer: Optimizer whose learning rate will be scheduled. A
                ``None`` value is ignored and leaves the wrapper unchanged.
        """
        if optimizer is None:
            return

        self.optimizer = optimizer

        # Save the current state of the optimizer before rebuilding it
        if self.scheduler is not None and hasattr(self.scheduler, 'last_epoch'):
            self._last_epoch = self.scheduler.last_epoch

        self._create_scheduler()

    def step(self, num_steps: int = 1):
        """Steps the scheduler for the given number of steps.

        Args:
            num_steps: The number of steps to step the scheduler for.
        """
        if self.scheduler:
            for _ in range(num_steps):
                self.scheduler.step()

    def get_factor(self):
        """Return the current schedule factor (last learning rate).

        Returns:
            factor (float): Current factor from the underlying scheduler, or
                ``1.0`` when no scheduler has been built.
        """
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return 1.0
    
    def get_config(self):
        """Return the constructor kwargs needed to rebuild this schedule.

        Returns:
            config (dict): Schedule type, steps, start/end values, plus any
                extra kwargs passed at construction.
        """
        config = {
            "schedule_type": self.schedule_type,
            "steps": self.steps,
            "start_value": self.start_value,
            "end_value": self.end_value,
        }

        config.update(**self.kwargs)

        return config

    @classmethod
    def from_config(cls, config: dict) -> "ScheduleWrapper":
        """Rebuild a ScheduleWrapper from a ``get_config()`` dict.

        Args:
            config: Mapping passed through as ``cls(**config)``. Use exactly
                the dict returned by [get_config][phoenx.schedulers.ScheduleWrapper.get_config]: required keys
                ``schedule_type``, ``steps``, ``start_value``, and
                ``end_value``, plus any extra keyword arguments that were
                forwarded to the underlying PyTorch scheduler at construction.
        """
        return cls(**config)

    def get_state(self) -> dict | None:
        """Return the scheduler progress (``last_epoch`` etc.) for resuming."""
        if self.scheduler is None:
            return None
        return self.scheduler.state_dict()

    def set_state(self, state: dict | None) -> None:
        """Restore scheduler progress produced by [get_state][phoenx.schedulers.ScheduleWrapper.get_state].

        Args:
            state: Scheduler ``state_dict`` as returned by [get_state][phoenx.schedulers.ScheduleWrapper.get_state]
                (the object from ``scheduler.state_dict()``, including
                ``last_epoch`` and related fields), or ``None`` to leave the
                scheduler unchanged.
        """
        if state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state)

    def clone(self):
        """Return a copy of this wrapper, including scheduler and optimizer state.

        Returns:
            wrapper (ScheduleWrapper): New wrapper whose scheduler and optimizer
                state match this instance.
        """
        new_wrapper = ScheduleWrapper(
            self.schedule_type,
            self.steps,
            self.start_value,
            self.end_value,
            self.optimizer,
            **self.kwargs
        )
        if self.scheduler:
            new_wrapper.scheduler.load_state_dict(self.scheduler.state_dict())
        if self.optimizer:
            new_wrapper.optimizer.load_state_dict(self.optimizer.state_dict())
        # new_wrapper.param = self.param.clone().detach().requires_grad_(False)
        return new_wrapper
