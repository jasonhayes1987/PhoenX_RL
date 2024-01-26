"""This module provides helper functions for configuring and using TensorFlow."""

from tensorflow.keras import optimizers
import gymnasium as gym
import numpy as np


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


class ReplayBuffer():
    """Replay buffer for experience replay."""
    # needs to store state, action, reward, next_state, done

    def __init__(self, env: gym.Env, buffer_size: int):
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
    
class OUNoise():
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, mu: np.ndarray, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        """Initializes a new Ornstein-Uhlenbeck noise process.

        Args:
            mu (ndarray): The mean of the noise process.
            theta (float, optional): The theta parameter. Defaults to 0.15.
            sigma (float, optional): The sigma parameter. Defaults to 0.2.
            dt (float, optional): The time step. Defaults to 1e-2.
        """
        self.mu = mu
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
        self.mu = np.zeros(self.mu.size) if mu is None else mu
        self.x_prev = np.ones(self.mu.size) * self.mu

    