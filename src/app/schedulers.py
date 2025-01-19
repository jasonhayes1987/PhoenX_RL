import torch as T
from torch import optim
from torch.optim import lr_scheduler

class ScheduleWrapper:
    def __init__(self, schedule_config):
        if schedule_config is None:
            self.schedule_config = None
            self.scheduler = None
            return
        self.schedule_config = schedule_config
        
        self.param = T.nn.Parameter(T.zeros(1), requires_grad=False)
        self.optimizer = optim.SGD([self.param], lr=1.0)
        
        scheduler_type = schedule_config.get("type", "").lower()
        scheduler_params = schedule_config.get("params", {})
        
        # Map scheduler type to PyTorch's built-in schedulers
        if scheduler_type == "linear":
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "cosineannealing":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "exponential":
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def step(self):
        if self.scheduler:
            self.scheduler.step()

    def get_factor(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return 1.0
    
    def get_config(self):
        return self.schedule_config
