import torch as T
from torch import optim
from torch.optim import lr_scheduler

class ScheduleWrapper:
    def __init__(self, type: str, config: dict):
        """
        #TODO: define schedule config dict
        """
        self.type = type
        self.config = config
        
        self.param = T.nn.Parameter(T.zeros(1), requires_grad=False)
        self.optimizer = optim.SGD([self.param], lr=1.0)
        
        # Map scheduler type to PyTorch's built-in schedulers
        if self.type == "linear":
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, **self.config)
        elif self.type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **self.config)
        elif self.type == "cosineannealing":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config)
        elif self.type == "exponential":
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, **self.config)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.type}")

    def step(self):
        if self.scheduler:
            self.scheduler.step()

    def get_factor(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return 1.0
    
    def get_config(self):
        return {
            "type": self.type,
            "config": self.config
        }

    def clone(self):
        new_wrapper = ScheduleWrapper(self.type, self.config.copy())
        if self.scheduler:
            new_wrapper.scheduler.load_state_dict(self.scheduler.state_dict())
        if self.optimizer:
            new_wrapper.optimizer.load_state_dict(self.optimizer.state_dict())
        new_wrapper.param = self.param.clone().detach().requires_grad_(False)
        return new_wrapper
