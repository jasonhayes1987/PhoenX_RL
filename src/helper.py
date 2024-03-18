"""This module provides helper functions for configuring and using TensorFlow."""

from tensorflow.keras import optimizers
from tensorflow import random

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
        "adam": optimizers.Adam,
        "sgd": optimizers.SGD,
        "rmsprop": optimizers.RMSprop,
        "adagrad": optimizers.Adagrad,
        # Add more optimizers as needed
    }

    if name.lower() not in opts:
        raise ValueError(
            f'Optimizer "{name}" is not recognized. Available options: {list(opts.keys())}'
        )

    return opts[name.lower()]()


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

    def __init__(self, env: gym.Env, buffer_size: int = 100000):
        """Initializes a new replay buffer.

        Args:
            env (gym.Env): The environment.
            buffer_size (int): The maximum number of transitions to store.
        """
        self.env = env
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=np.bool_)
        self.counter = 0
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

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.counter = self.counter + 1

    def sample(self, batch_size: int):
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): The batch size.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        # indices = np.random.choice(self.buffer_size, batch_size)
        # replace with a generator

        size = min(self.counter, self.buffer_size)
        indices = self.gen.choice(size, batch_size, replace=True)
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

    def __init__(self, shape):
        self.shape = shape

    def __call__(self):
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
    def __init__(self, shape, minval=0, maxval=1, dtype='float32'):
        super().__init__(shape)
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def __call__(self):
        return random.uniform(shape=self.shape, minval=self.minval, maxval=self.maxval, dtype=self.dtype)

    def get_config(self):
        return {
            'class_name': 'UniformNoise',
            'config': {
                'shape': self.shape,
                'minval': self.minval,
                'maxval': self.maxval,
                'dtype': self.dtype,
            }
        }

class NormalNoise(Noise):
    def __init__(self, shape, mean=0.0, stddev=1.0, dtype='float32'):
        super().__init__(shape)
        self.mean = mean
        self.stddev = stddev
        self.dtype = dtype

    def __call__(self):
        return random.normal(shape=self.shape, mean=self.mean, stddev=self.stddev, dtype=self.dtype)

    def get_config(self):
        return {
            'class_name': 'NormalNoise',
            'config': {
                'shape': self.shape,
                'mean': self.mean,
                'stddev': self.stddev,
                'dtype': self.dtype,
            }
        }


class OUNoise(Noise):
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, shape: tuple = (1,), mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        """Initializes a new Ornstein-Uhlenbeck noise process.

        Args:
            mu (ndarray): The mean of the noise process.
            theta (float, optional): The theta parameter. Defaults to 0.15.
            sigma (float, optional): The sigma parameter. Defaults to 0.2.
            dt (float, optional): The time step. Defaults to 1e-2.
        """
        self.size = shape
        self.mean = mean
        self.mu = np.ones(shape) * mean
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        # self.init_state = np.ones(size) * mu
        self.reset()

    def __call__(self):
        """Samples a new noise vector.

        Returns:
            A noise vector.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.size)
        self.x_prev = x

        return x
    
    def reset(self, mu: np.ndarray = None):
        """Resets the noise process."""
        self.mu = np.ones(self.size) * self.mean if mu is None else np.array(mu)
        self.x_prev = np.ones(self.mu.size) * self.mu

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "shape": self.size,
                "mean": self.mean,
                "theta": self.theta,
                "sigma": self.sigma,
                "dt": self.dt,
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


    