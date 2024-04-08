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
    """Replay buffer for experience replay."""
    # needs to store state, action, reward, next_state, done
    def __init__(self, env: gym.Env, buffer_size: int = 100000, device='cpu'):
        """Initializes a new replay buffer.
        Args:
            env (gym.Env): The environment.
            buffer_size (int): The maximum number of transitions to store.
            device (torch.device): The device to use for storing tensors.
        """
        self.env = env
        self.buffer_size = buffer_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # self.states = torch.zeros((buffer_size, *env.observation_space.shape), dtype=torch.float32, device=self.device)
        self.states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        # self.actions = torch.zeros((buffer_size, *env.action_space.shape), dtype=torch.float32, device=self.device)
        self.actions = np.zeros((buffer_size, *env.action_space.shape), dtype=np.float32)
        # self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        # self.next_states = torch.zeros((buffer_size, *env.observation_space.shape), dtype=torch.float32, device=self.device)
        self.next_states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        # self.dones = torch.zeros((buffer_size,), dtype=torch.int, device=self.device)
        self.dones = np.zeros((buffer_size,), dtype=np.int8)
        
        self.counter = 0
        # self.gen = torch.Generator(device=self.device)
        self.gen = np.random.default_rng()
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Adds a new transition to the replay buffer.
        Args:
            state (np.ndarray): The state.
            action (np.ndarray): The action.
            reward (float): The reward.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        index = self.counter % self.buffer_size
        # self.states[index] = torch.from_numpy(state).to(self.device)
        self.states[index] = state
        # self.actions[index] = torch.from_numpy(action).to(self.device)
        self.actions[index] = action
        # self.rewards[index] = torch.tensor(reward).to(self.device)
        self.rewards[index] = reward
        # self.next_states[index] = torch.from_numpy(next_state).to(self.device)
        self.next_states[index] = next_state
        # self.dones[index] = torch.tensor(done).to(self.device)
        self.dones[index] = done
        self.counter = self.counter + 1
        
    def sample(self, batch_size: int):
        """Samples a batch of transitions from the replay buffer.
        Args:
            batch_size (int): The batch size.
        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        size = min(self.counter, self.buffer_size)
        indices = self.gen.integers(0, size, (batch_size,))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.spec.id,
                "buffer_size": self.buffer_size,
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
    
# def get_noise_by_name(name: str):
#     """Creates and returns a noise object by its string name.

#     Args:
#     name (str): The name of the noise.

#     Returns:
#     An instance of the requested noise.
# """
#     noises = {
#         "Ornstein-Uhlenbeck": OUNoise,


    