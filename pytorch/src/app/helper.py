"""This module provides helper functions for configuring and using Pytorch."""

# from tensorflow.keras import optimizers
# from tensorflow import random
import torch as T
from torch import optim
from torch.distributions import uniform, normal
import threading
from multiprocessing import shared_memory #Manager
from mpi4py import MPI

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
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        
        # set internal attributes
        # get observation space
        if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            self._obs_space_shape = self.env.observation_space['observation'].shape
        else:
            self._obs_space_shape = self.env.observation_space.shape

        #DEBUG
        print(f'self._obs_space_shape: {self._obs_space_shape}')
        print(f'buffer size = {self.buffer_size}')

        self.states = np.zeros((buffer_size, *self._obs_space_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *self._obs_space_shape), dtype=np.float32)
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
    
    def clone(self):
        env = gym.make(self.env.spec)
        return ReplayBuffer(
            env,
            self.buffer_size,
            self.goal_shape,
            self.device
        )
    
class SharedReplayBuffer(Buffer):
    def __init__(self, manager, env:gym.Env, buffer_size:int=100000, goal_shape:tuple=None, device='cpu'):
        self.env = env
        self.buffer_size = buffer_size
        self.goal_shape = goal_shape
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')

        # self.lock = manager.Lock()
        self.lock = threading.Lock()

        # set internal attributes
        # get observation space
        if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            self._obs_space_shape = self.env.observation_space['observation'].shape
        else:
            self._obs_space_shape = self.env.observation_space.shape

        # Create shared data buffers to map ndarrays to
        # States
        # Calculate byte size
        state_byte_size = np.prod(self._obs_space_shape) * np.float32().itemsize
        state_total_bytes = self.buffer_size * state_byte_size
        # Create shared memory block for state and next states
        self.shared_states = shared_memory.SharedMemory(create=True, size=state_total_bytes)
        self.shared_next_states = shared_memory.SharedMemory(create=True, size=state_total_bytes)
        
        # Actions
        # Calculate byte size
        action_byte_size = np.prod(self.env.action_space.shape) * np.float32().itemsize
        action_total_bytes = self.buffer_size * action_byte_size
        # Create Shared memory block for actions
        self.shared_actions = shared_memory.SharedMemory(create=True, size=action_total_bytes)

        # Rewards
        # Calculate byte size
        reward_total_bytes = self.buffer_size * np.float32().itemsize
        self.shared_rewards = shared_memory.SharedMemory(create=True, size=reward_total_bytes)

        # Dones (byte size same as rewards)
        self.shared_dones = shared_memory.SharedMemory(create=True, size=reward_total_bytes)

        # If goal shape provided in constructor, create shared blocks for goals
        if self.goal_shape is not None:
            goal_byte_size = np.prod(self.goal_shape) * np.float32().itemsize
            goal_total_bytes = self.buffer_size * goal_byte_size
            # Create shared memory blocks for desired, state, and next state goals
            self.shared_desired_goals = shared_memory.SharedMemory(create=True, size=goal_total_bytes)
            self.shared_state_achieved_goals = shared_memory.SharedMemory(create=True, size=goal_total_bytes)
            self.shared_next_state_achieved_goals = shared_memory.SharedMemory(create=True, size=goal_total_bytes)

        # Create ndarrays to store data mapped to memory blocks
        self.states = np.ndarray((buffer_size, *self._obs_space_shape), dtype=np.float32, buffer=self.shared_states.buf)
        self.next_states = np.ndarray((buffer_size, *self._obs_space_shape), dtype=np.float32, buffer=self.shared_next_states.buf)
        self.actions = np.ndarray((buffer_size, *env.action_space.shape), dtype=np.float32, buffer=self.shared_actions.buf)
        self.rewards = np.ndarray((buffer_size,), dtype=np.float32, buffer=self.shared_rewards.buf)
        self.dones = np.ndarray((buffer_size,), dtype=np.int8, buffer=self.shared_dones.buf)
        
        # Fill ndarrays with zeros
        self.states.fill(0)
        self.next_states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)

        # If using goals, create ndarrays to store data mapped to memory blocks
        if self.goal_shape is not None:
            self.desired_goals = np.ndarray((buffer_size, *self.goal_shape), dtype=np.float32, buffer=self.shared_desired_goals.buf)
            self.state_achieved_goals = np.ndarray((buffer_size, *self.goal_shape), dtype=np.float32, buffer=self.shared_state_achieved_goals.buf)
            self.next_state_achieved_goals = np.ndarray((buffer_size, *self.goal_shape), dtype=np.float32, buffer=self.shared_next_state_achieved_goals.buf)
            # Fill ndarray with zeros
            self.desired_goals.fill(0)
            self.state_achieved_goals.fill(0)
            self.next_state_achieved_goals.fill(0)


        self.counter = 0
        self.gen = np.random.default_rng()
        
    def add(self, state:np.ndarray, action:np.ndarray, reward:float, next_state:np.ndarray, done:bool,
            state_achieved_goal:np.ndarray=None, next_state_achieved_goal:np.ndarray=None, desired_goal:np.ndarray=None):
        """Add a transition to the replay buffer."""
        with self.lock:
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
        with self.lock:
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
    
    def clone(self):
        env = gym.make(self.env.spec)
        return ReplayBuffer(
            env,
            self.buffer_size,
            self.goal_shape,
            self.device
        )
    
    def cleanup(self):
        # Close and unlink shared memory blocks
        try:
            if self.shared_states:
                self.shared_states.unlink()
                self.shared_states.close()
                self.shared_states = None
        except FileNotFoundError as e:
            print(f"Shared states already cleaned up: {e}")
        try:
            if self.shared_next_states:
                self.shared_next_states.unlink()
                self.shared_next_states.close()
                self.shared_next_states = None
        except FileNotFoundError as e:
            print(f"Shared next states already cleaned up: {e}")
        try:
            if self.shared_rewards:
                self.shared_rewards.unlink()
                self.shared_rewards.close()
                self.shared_rewards = None
        except FileNotFoundError as e:
            print(f"Shared rewards already cleaned up: {e}")
        try:
            if self.shared_dones:
                self.shared_dones.unlink()
                self.shared_dones.close()
                self.shared_dones = None
        except FileNotFoundError as e:
            print(f"Shared dones already cleaned up: {e}")
        
        if self.goal_shape is not None:
            try:
                if self.shared_desired_goals:
                    self.shared_desired_goals.unlink()
                    self.shared_desired_goals.close()
                    self.shared_desired_goals = None
            except FileNotFoundError as e:
                print(f"Shared desired goals already cleaned up: {e}")
            try:
                if self.shared_state_achieved_goals:
                    self.shared_state_achieved_goals.unlink()
                    self.shared_state_achieved_goals.close()
                    self.shared_state_achieved_goals = None
            except FileNotFoundError as e:
                print(f"Shared state achieved goals already cleaned up: {e}")
            try:
                if self.shared_next_state_achieved_goals:
                    self.shared_next_state_achieved_goals.unlink()
                    self.shared_next_state_achieved_goals.close()
                    self.shared_next_state_achieved_goals = None
            except FileNotFoundError as e:
                print(f"Shared next state achieved goals already cleaned up: {e}")

        print("SharedNormalizer resources have been cleaned up.")

    def __del__(self):
        self.cleanup()

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

    def clone(self):
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
    def __init__(self, shape, minval=0, maxval=1, device=None):
        super().__init__()
        self.shape = shape
        self.device = self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.minval = T.tensor(minval, device=self.device)
        self.maxval = T.tensor(maxval, device=self.device)
        
        self.noise_gen = uniform.Uniform(low=self.minval, high=self.maxval)

    def __call__(self):
        return self.noise_gen.sample(self.shape)

    def get_config(self):
        return {
            'class_name': 'UniformNoise',
            'config': {
                'shape': self.shape,
                'minval': self.minval.item(),
                'maxval': self.maxval.item(),
            }
        }
    
    def clone(self):
        return UniformNoise(
            self.shape,
            self.minval,
            self.maxval,
            self.device
        )

class NormalNoise:
    def __init__(self, shape, mean=0.0, stddev=1.0, device=None):
        super().__init__()
        self.shape = shape
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.mean = np.array(mean, dtype=np.float32)
        self.stddev = np.array(stddev, dtype=np.float32)

        # Initialize the noise generator here using the numpy arrays
        self.reset_noise_gen()

    def reset_noise_gen(self):
        # Convert numpy mean and stddev to tensors just for noise generation
        mean_tensor = T.tensor(self.mean, device=self.device)
        stddev_tensor = T.tensor(self.stddev, device=self.device)
        self.noise_gen = normal.Normal(loc=mean_tensor, scale=stddev_tensor)

    def __call__(self):
        # Directly sample using the noise generator
        return self.noise_gen.sample(self.shape)

    def __getstate__(self):
        # Only the numpy arrays are serialized
        state = self.__dict__.copy()
        # Remove the noise generator since it can't be pickled
        del state['noise_gen']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the noise generator after deserialization
        self.reset_noise_gen()

    def get_config(self):
        return {
            'class_name': 'NormalNoise',
            'config': {
                'shape': self.shape,
                'mean': self.mean.item(),
                'stddev': self.stddev.item(),
            }
        }
    
    def clone(self):
        return NormalNoise(
            self.shape,
            self.mean,
            self.stddev,
            self.device
        )
    
class OUNoise(Noise):
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, shape: tuple, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2, device=None):
        """Initializes a new Ornstein-Uhlenbeck noise process."""
        super().__init__()
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.shape = shape
        self.mean = T.tensor(mean, device=self.device)
        self.mu = T.ones(self.shape, device=self.device) * self.mean
        self.theta = T.tensor(theta, device=self.device)
        self.sigma = T.tensor(sigma, device=self.device)
        self.dt = T.tensor(dt, device=self.device)
        self.x_prev = T.ones(self.shape, device=self.device) * self.mean

    def __call__(self):
        """Samples a new noise vector."""
        dx = self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * T.randn(self.shape, device=self.device)
        x = self.x_prev + dx
        self.x_prev = x
        return x

    def reset(self, mu: T.tensor = None):
        """Resets the noise process."""
        self.mu = T.ones(self.shape, device=self.device) * self.mean if mu is None else T.tensor(mu, device=self.device)
        self.x_prev = T.ones(self.shape, device=self.device) * self.mu

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
        
    def clone(self):
        return OUNoise(
            self.shape,
            self.mean,
            self.theta,
            self.sigma,
            self.dt,
            self.device
        )
    

class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        self.local_sum = np.zeros(self.size, dtype=np.float32)
        self.local_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.local_cnt = np.zeros(1, dtype=np.int32)

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_cnt = np.zeros(1, dtype=np.int32)

        self.lock = threading.Lock()

    def normalize(self, v):
        clip_range = self.clip_range
        return np.clip((v - self.running_mean) / self.running_std,
                       -clip_range, clip_range).astype(np.float32)
    
    def update_local_stats(self, new_data):
        # print('SharedNormalizer update_local_stats fired...')
        try:
            with self.lock:
                # print('SharedNormalizer update_local_stats lock acquired...')
                # print(f'data: {new_data}')
                # print('previous local stats')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
                self.local_sum += new_data#.sum(axis=1)
                self.local_sum_sq += (np.square(new_data))#.sum(axis=1)
                self.local_cnt += 1 #new_data.shape[0]
                # print('new local values')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
        except Exception as e:
            print(f"Error during update: {e}")
    
    def update_global_stats(self):
        with self.lock:
            local_cnt = self.local_cnt.copy()
            local_sum = self.local_sum.copy()
            local_sum_sq = self.local_sum_sq.copy()

            self.local_cnt[...] = 0
            self.local_sum[...] = 0
            self.local_sum_sq[...] = 0

        sync_sum, sync_sum_sq, sync_cnt = self.sync_thread_stats(
                local_sum, local_sum_sq, local_cnt)

        self.running_cnt += sync_cnt
        self.running_sum += sync_sum
        self.running_sum_sq += sync_sum_sq

        self.running_mean = self.running_sum / self.running_cnt
        tmp = self.running_sum_sq / self.running_cnt -\
            np.square(self.running_sum / self.running_cnt)
        self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))

    def sync_thread_stats(self, local_sum, local_sum_sq, local_cnt):
        local_sum[...] = self.mpi_average(local_sum)
        local_sum_sq[...] = self.mpi_average(local_sum_sq)
        local_cnt[...] = self.mpi_average(local_cnt)
        return local_sum, local_sum_sq, local_cnt

    def mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf = buf.astype(np.float64) / MPI.COMM_WORLD.Get_size()
        return buf.astype(x.dtype)

    def get_config(self):
        return {
            "params":{
                'size':self.size,
                'eps':self.eps,
                'clip_range':self.clip_range,
            },
            "state":{
                'local_sum':self.local_sum,
                'local_sum_sq':self.local_sum_sq,
                'local_cnt':self.local_cnt,
                'running_mean':self.running_mean,
                'running_std':self.running_std,
                'running_sum':self.running_sum,
                'running_sum_sq':self.running_sum_sq,
                'running_cnt':self.running_cnt,
            },
        }


    def save_state(self, file_path):
        np.savez(
            file_path,
            local_sum=self.local_sum,
            local_sum_sq=self.local_sum_sq,
            local_cnt=self.local_cnt,
            running_mean=self.running_mean,
            running_std=self.running_std,
            running_sum=self.running_sum,
            running_sum_sq=self.running_sum_sq,
            running_cnt=self.running_cnt,
        )


    @classmethod
    def load_state(cls, file_path):
        with np.load(file_path) as data:
            normalizer = cls(size=data['running_mean'].shape)
            normalizer.local_sum = data['local_sum']
            normalizer.local_sum_sq = data['local_sum_sq']
            normalizer.local_cnt = data['local_cnt']
            normalizer.running_mean = data['running_mean']
            normalizer.running_std = data['running_std']
            normalizer.running_sum = data['running_sum']
            normalizer.running_sum_sq = data['running_sum_sq']
            normalizer.running_cnt = data['running_cnt']
        return normalizer
    
class SharedNormalizer:
    def __init__(self, manager, size, eps=1e-2, clip_range=5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        # self.lock = manager.Lock()
        self.lock = threading.Lock()

        # Create shared memory blocks
        total_byte_size = np.prod(self.size) * np.float32().itemsize
        self.shared_local_sum = shared_memory.SharedMemory(create=True, size=total_byte_size)
        self.shared_local_sum_sq = shared_memory.SharedMemory(create=True, size=total_byte_size)
        self.shared_local_cnt = shared_memory.SharedMemory(create=True, size=np.float32().itemsize)

        self.local_sum = np.ndarray(self.size, dtype=np.float32, buffer=self.shared_local_sum.buf)
        self.local_sum_sq = np.ndarray(self.size, dtype=np.float32, buffer=self.shared_local_sum_sq.buf)
        self.local_cnt = np.ndarray(1, dtype=np.int32, buffer=self.shared_local_cnt.buf)

        # Initiate shared arrays to zero
        self.local_sum.fill(0)
        self.local_sum_sq.fill(0)
        self.local_cnt.fill(0)

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_cnt = np.zeros(1, dtype=np.int32)

    def normalize(self, v):
        clip_range = self.clip_range
        return np.clip((v - self.running_mean) / self.running_std,
                       -clip_range, clip_range).astype(np.float32)
    
    def update_local_stats(self, new_data):
        # print('SharedNormalizer update_local_stats fired...')
        try:
            with self.lock:
                # print('SharedNormalizer update_local_stats lock acquired...')
                # print(f'data: {new_data}')
                # print('previous local stats')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
                self.local_sum += new_data#.sum(axis=1)
                self.local_sum_sq += (np.square(new_data))#.sum(axis=1)
                self.local_cnt += 1 #new_data.shape[0]
                # print('new local values')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
        except Exception as e:
            print(f"Error during update: {e}")
    
    def update_global_stats(self):
        with self.lock:
            # make copies of local stats
            local_cnt = self.local_cnt.copy()
            local_sum = self.local_sum.copy()
            local_sum_sq = self.local_sum_sq.copy()
            
            # Zero out local stats
            self.local_cnt[...] = 0
            self.local_sum[...] = 0
            self.local_sum_sq[...] = 0
            
            # Add local stats to global stats
            self.running_cnt += local_cnt
            self.running_sum += local_sum
            self.running_sum_sq += local_sum_sq

            # Calculate new mean, sum_sq, and std
            self.running_mean = self.running_sum / self.running_cnt
            tmp = self.running_sum_sq / self.running_cnt -\
                np.square(self.running_sum / self.running_cnt)
            self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))

    def get_config(self):
        return {
            "params":{
                'size':self.size,
                'eps':self.eps,
                'clip_range':self.clip_range,
            },
            "state":{
                'local_sum':self.local_sum,
                'local_sum_sq':self.local_sum_sq,
                'local_cnt':self.local_cnt,
                'running_mean':self.running_mean,
                'running_std':self.running_std,
                'running_sum':self.running_sum,
                'running_sum_sq':self.running_sum_sq,
                'running_cnt':self.running_cnt,
            },
        }


    def save_state(self, file_path):
        np.savez(
            file_path,
            local_sum=self.local_sum,
            local_sum_sq=self.local_sum_sq,
            local_cnt=self.local_cnt,
            running_mean=self.running_mean,
            running_std=self.running_std,
            running_sum=self.running_sum,
            running_sum_sq=self.running_sum_sq,
            running_cnt=self.running_cnt,
        )

    def cleanup(self):
        # Close and unlink shared memory blocks
        try:
            if self.shared_local_sum:
                self.shared_local_sum.unlink()
                self.shared_local_sum.close()
                self.shared_local_sum = None
        except FileNotFoundError as e:
            print(f"Shared local sum already cleaned up: {e}")
        try:
            if self.shared_local_sum_sq:
                self.shared_local_sum_sq.unlink()
                self.shared_local_sum_sq.close()
                self.shared_local_sum_sq = None
        except FileNotFoundError as e:
            print(f"Shared local sum sq already cleaned up: {e}")
        try:
            if self.shared_local_cnt:
                self.shared_local_cnt.unlink()
                self.shared_local_cnt.close()
                self.shared_local_cnt = None
        except FileNotFoundError as e:
            print(f"Shared local sum cnt already cleaned up: {e}")

        print("SharedNormalizer resources have been cleaned up.")

    def __del__(self):
        self.cleanup()


    @classmethod
    def load_state(cls, file_path):
        with np.load(file_path) as data:
            normalizer = cls(size=data['running_mean'].shape)
            normalizer.local_sum = data['local_sum']
            normalizer.local_sum_sq = data['local_sum_sq']
            normalizer.local_cnt = data['local_cnt']
            normalizer.running_mean = data['running_mean']
            normalizer.running_std = data['running_std']
            normalizer.running_sum = data['running_sum']
            normalizer.running_sum_sq = data['running_sum_sq']
            normalizer.running_cnt = data['running_cnt']
        return normalizer
    
# MULTITHREADING FUNCTIONALITY

class MPIHelper:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def is_main_process(self):
        return self.rank == 0

    def bcast(self, data, root=0):
        return self.comm.bcast(data, root)

    def gather(self, data, root=0):
        return self.comm.gather(data, root)

def sync_networks(network):
    comm = MPI.COMM_WORLD
    params = np.concatenate([getattr(p, 'data').cpu().numpy().flatten()
                             for p in network.parameters()])
    comm.Bcast(params)
    idx = 0
    for p in network.parameters():
        getattr(p, 'data').copy_(T.tensor(
            params[idx:idx + p.data.numel()]).view_as(p.data))
        idx += p.data.numel()

def sync_grads_sum(network):
    comm = MPI.COMM_WORLD
    grads = np.concatenate([getattr(p, 'grad').cpu().numpy().flatten()
                           for p in network.parameters()])
    global_grads = np.zeros_like(grads)
    comm.Allreduce(grads, global_grads, op=MPI.SUM)
    idx = 0
    for p in network.parameters():
        getattr(p, 'grad').copy_(T.tensor(
            global_grads[idx:idx + p.data.numel()]).view_as(p.data))
        idx += p.data.numel()

def sync_grads_avg(network):
    comm = MPI.COMM_WORLD
    workers = MPI.COMM_WORLD.Get_size()
    grads = np.concatenate([getattr(p, 'grad').cpu().numpy().flatten()
                           for p in network.parameters()])
    global_grads = np.zeros_like(grads)
    comm.Allreduce(grads, global_grads, op=MPI.SUM)
    global_grads /= workers
    idx = 0
    for p in network.parameters():
        getattr(p, 'grad').copy_(T.tensor(
            global_grads[idx:idx + p.data.numel()]).view_as(p.data))
        idx += p.data.numel()

def sync_metrics(config):
    # Create a dictionary to store the summed metrics
    summed_metrics = {}

    # Sum the metrics across all workers
    for key, value in config.items():
        if isinstance(value, (int, float)):
            # Create a buffer for MPI communication
            buffer = np.zeros(1, dtype=type(value))
            buffer[0] = value

            # Perform Allreduce to sum the metric values
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

            # Store the summed metric in the dictionary
            summed_metrics[key] = buffer[0]

    # Average the summed metrics
    num_workers = MPI.COMM_WORLD.Get_size()
    averaged_metrics = {key: value / num_workers for key, value in summed_metrics.items()}

    return averaged_metrics