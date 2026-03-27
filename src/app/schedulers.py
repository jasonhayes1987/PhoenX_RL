import torch as T
from torch import optim
from torch.optim import lr_scheduler

class ScheduleWrapper:
    def __init__(
        self,
        schedule_type: str,
        steps: int,
        end_value: float,
        start_value: float|None = None,
        **kwargs
    ):
        """
        Wrapper for schedule types.
        """
        self.schedule_type = schedule_type
        self.steps = steps
        self.end_value = end_value
        self.start_value = start_value
        self.kwargs = kwargs
        
        self.param = T.nn.Parameter(T.zeros(1), requires_grad=False)
        self.optimizer = optim.SGD([self.param], lr=1.0)
        
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
            if self.end_value is None or self.steps is None:
                raise ValueError("End value and steps are required for cosine scheduler.")
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

    def step(self):
        if self.scheduler:
            self.scheduler.step()

    def get_factor(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return 1.0
    
    def get_config(self):
        return {
            "type": self.schedule_type,
            "steps": self.steps,
            "end_value": self.end_value,
            "start_value": self.start_value,
            "kwargs": self.kwargs
        }

    def clone(self):
        new_wrapper = ScheduleWrapper(
            self.schedule_type,
            self.steps,
            self.end_value,
            self.start_value,
            **self.kwargs
        )
        if self.scheduler:
            new_wrapper.scheduler.load_state_dict(self.scheduler.state_dict())
        if self.optimizer:
            new_wrapper.optimizer.load_state_dict(self.optimizer.state_dict())
        new_wrapper.param = self.param.clone().detach().requires_grad_(False)
        return new_wrapper
