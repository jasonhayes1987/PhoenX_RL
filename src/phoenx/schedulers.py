import torch as T
from torch import optim
from torch.optim import lr_scheduler, Optimizer

class ScheduleWrapper:
    """Wrapper for schedule types.

    Args:
        schedule_type: The type of scheduler to use. Supported types: "linear", "cosine", "exponential".
        steps: The number of steps to run the scheduler for.
        start_value: The starting value of the scheduler.
        end_value: The ending value of the scheduler.
        kwargs: Additional keyword arguments to pass to the scheduler.
    """
    def __init__(
        self,
        schedule_type: str,
        steps: int,
        start_value: float,
        end_value: float,
        optimizer: Optimizer|None = None,
        **kwargs
    ):
        """Wrapper for schedule types.

        Args:
            schedule_type: The type of scheduler to use. Supported types: "linear", "cosine", "exponential".
            steps: The number of steps to run the scheduler for.
            start_value: The starting value of the scheduler.
            end_value: The ending value of the scheduler.
            kwargs: Additional keyword arguments to pass to the scheduler.
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
        """Attaches optimizer to scheduler."""
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
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return 1.0
    
    def get_config(self):
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
        """Rebuild a ScheduleWrapper from a ``get_config()`` dict."""
        # cfg = dict(config)
        # schedule_type = cfg.pop("schedule_type")
        # extra = cfg.pop("kwargs", {}) or {}
        # return cls(
        #     schedule_type=schedule_type,
        #     steps=cfg["steps"],
        #     start_value=cfg["start_value"],
        #     end_value=cfg["end_value"],
        #     **extra,
        # )

        return cls(**config)

    def get_state(self) -> dict | None:
        """Return the scheduler progress (``last_epoch`` etc.) for resuming."""
        if self.scheduler is None:
            return None
        return self.scheduler.state_dict()

    def set_state(self, state: dict | None) -> None:
        """Restore scheduler progress produced by :meth:`get_state`."""
        if state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state)

    def clone(self):
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
