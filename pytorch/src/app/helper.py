"""This module provides helper functions for configuring and using TensorFlow."""

# from tensorflow.keras import optimizers
# from tensorflow import random
import torch
from torch import optim
from torch.distributions import uniform, normal

import gymnasium as gym
import numpy as np

# random helper functions
def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The dictionary to flatten.
    - parent_key: The base key to use for the current level of recursion.
    - sep: The separator between nested keys.

    Returns:
    A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_optimizer_by_name(name: str):
    """Creates and returns an optimizer object by its string name.

    Args:
        name (str): The name of the optimizer.

    Returns:
        An instance of the requested optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    opts = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        # Add more optimizers as needed
    }

    if name not in opts:
        raise ValueError(
            f'Optimizer "{name}" is not recognized. Available options: {list(opts.keys())}'
        )

    return opts[name]


class Buffer():
    """Base class for replay buffers."""

    def __init__(self):
        pass

    def add(self):
        pass
    
    def sample(self):
        pass

    def config(self):
        pass

    @classmethod
    def create_instance(cls, buffer_class_name, **kwargs):
        """Creates an instance of the requested buffer class.

        Args:
        buffer_class_name (str): The name of the buffer class.

        Returns:
        Buffer: An instance of the requested buffer class.
        """

        buffer_classes = {
            "ReplayBuffer": ReplayBuffer,
        }

        
        if buffer_class_name in buffer_classes:
            return buffer_classes[buffer_class_name](**kwargs)
        else:
            raise ValueError(f"{buffer_class_name} is not a subclass of Buffer")




class ReplayBuffer(Buffer):
    def __init__(self, env:gym.Env, buffer_size:int=100000, goal_shape:tuple=None, device='cpu'):
        self.env = env
        self.buffer_size = buffer_size
        self.goal_shape = goal_shape
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.int8)
        
        if self.goal_shape is not None:
            self.desired_goals = np.zeros((buffer_size, *self.goal_shape), dtype=np.float32)
            self.state_achieved_goals = np.zeros((buffer_size, *self.goal_shape), dtype=np.float32)
            self.next_state_achieved_goals = np.zeros((buffer_size, *self.goal_shape), dtype=np.float32)
        
        self.counter = 0
        self.gen = np.random.default_rng()
        
    def add(self, state:np.ndarray, action:np.ndarray, reward:float, next_state:np.ndarray, done:bool,
            state_achieved_goal:np.ndarray=None, next_state_achieved_goal:np.ndarray=None, desired_goal:np.ndarray=None):
        """Add a transition to the replay buffer."""
        index = self.counter % self.buffer_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        
        if self.goal_shape is not None:
            if desired_goal is None or state_achieved_goal is None or next_state_achieved_goal is None:
                raise ValueError("Desired goal, state achieved goal, and next state achieved goal must be provided when use_goals is True.")
            self.state_achieved_goals[index] = state_achieved_goal
            self.next_state_achieved_goals[index] = next_state_achieved_goal
            self.desired_goals[index] = desired_goal
        
        self.counter = self.counter + 1
        
    def sample(self, batch_size:int):
        size = min(self.counter, self.buffer_size)
        indices = self.gen.integers(0, size, (batch_size,))
        
        if self.goal_shape is not None:
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                self.state_achieved_goals[indices],
                self.next_state_achieved_goals[indices],
                self.desired_goals[indices],
            )
        else:
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices]
            )
    
    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.spec.id,
                "buffer_size": self.buffer_size,
                "goal_shape": self.goal_shape
            }
        }

class Noise:
    """Base class for noise processes."""

    def __init__(self):
        pass

    def __call__(self, shape):
        pass

    def reset(self):
        pass

    def get_config(self):
        pass

    @classmethod
    def create_instance(cls, noise_class_name, **kwargs):
        """Creates an instance of the requested noise class.

        Args:
            noise_class_name (str): The name of the noise class.

        Returns:
            Noise: An instance of the requested noise class.
        """
        noise_classes = {
            "Ornstein-Uhlenbeck": OUNoise,
            "OUNoise": OUNoise,
            "Normal": NormalNoise,
            "NormalNoise": NormalNoise,
            "Uniform": UniformNoise,
            "UniformNoise": UniformNoise,
        }

        if noise_class_name in noise_classes:
            return noise_classes[noise_class_name](**kwargs)
        else:
            raise ValueError(f"{noise_class_name} is not a recognized noise class")

class UniformNoise(Noise):
    def __init__(self, minval=0, maxval=1, device=None):
        super().__init__()
        self.device = self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.minval = torch.tensor(minval, device=self.device)
        self.maxval = torch.tensor(maxval, device=self.device)
        
        self.noise_gen = uniform.Uniform(low=self.minval, high=self.maxval)

    def __call__(self, shape: torch.tensor):
        return self.noise_gen.sample(shape)

    def get_config(self):
        return {
            'class_name': 'UniformNoise',
            'config': {
                'minval': self.minval.item(),
                'maxval': self.maxval.item(),
            }
        }

class NormalNoise(Noise):
    def __init__(self, mean=0.0, stddev=1.0, device=None):
        super().__init__()
        self.device = self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)
        self.stddev = torch.tensor(stddev, device=self.device, dtype=torch.float32)

        self.noise_gen = normal.Normal(loc=self.mean, scale=self.stddev)

    def __call__(self, shape: torch.tensor):
        return self.noise_gen.sample(shape)

    def get_config(self):
        return {
            'class_name': 'NormalNoise',
            'config': {
                'mean': self.mean.item(),
                'stddev': self.stddev.item(),
            }
        }


class OUNoise(Noise):
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, shape: tuple, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2, device=None):
        """Initializes a new Ornstein-Uhlenbeck noise process."""
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape = shape
        self.mean = torch.tensor(mean, device=self.device)
        self.mu = torch.ones(self.shape, device=self.device) * self.mean
        self.theta = torch.tensor(theta, device=self.device)
        self.sigma = torch.tensor(sigma, device=self.device)
        self.dt = torch.tensor(dt, device=self.device)
        self.x_prev = torch.ones(self.shape, device=self.device) * self.mean

    def __call__(self):
        """Samples a new noise vector."""
        dx = self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * torch.randn(self.shape, device=self.device)
        x = self.x_prev + dx
        self.x_prev = x
        return x

    def reset(self, mu: torch.tensor = None):
        """Resets the noise process."""
        self.mu = torch.ones(self.shape, device=self.device) * self.mean if mu is None else torch.tensor(mu, device=self.device)
        self.x_prev = torch.ones(self.shape, device=self.device) * self.mu

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "shape": self.shape,
                "mean": self.mean.item(),
                "theta": self.theta.item(),
                "sigma": self.sigma.item(),
                "dt": self.dt.item(),
            }
        }
    

class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = clip_range

        self.local_sum = np.zeros(self.size, dtype=np.float32)
        self.local_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.local_cnt = np.zeros(1, dtype=np.float32)

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_cnt = 1

    def update_local_stats(self, new_data):
        self.local_sum += new_data.sum(axis=0)
        self.local_sum_sq += (np.square(new_data)).sum(axis=0)
        self.local_cnt[0] += new_data.shape[0]

    def normalize(self, v):
        clip_range = self.default_clip_range
        return np.clip((v - self.running_mean) / self.running_std,
                       -clip_range, clip_range).astype(np.float32)
    
    def update_global_stats(self):
        self.running_cnt += self.local_cnt
        self.running_sum += self.local_sum
        self.running_sum_sq += self.local_sum_sq

        self.local_cnt[...] = 0
        self.local_sum[...] = 0
        self.local_sum_sq[...] = 0

        self.running_mean = self.running_sum / self.running_cnt
        tmp = self.running_sum_sq / self.running_cnt - np.square(self.running_sum / self.running_cnt)
        self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))