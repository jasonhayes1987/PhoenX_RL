from abc import abstractmethod
import torch as T
import numpy as np
import gymnasium as gym
from .env_wrapper import Observation, EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
# from .utils import build_env_wrapper_obj
from .torch_utils import get_device
from typing import Optional, Tuple, List, Any, Dict
from collections import defaultdict
import math


class SumTree:
    """
    A binary sum tree for efficient sampling based on priorities.
    """
    def __init__(self, capacity: int, device: T.device):
        self.capacity = capacity
        self.device = get_device(device)
        # Initialize the tree with zeros
        self.tree = T.zeros(2 * capacity - 1, dtype=T.float32, device=self.device)
        self.next_idx = 0
        # self.size = 0
        self.max_priority = T.tensor(1.0, dtype=T.float32, device=self.device)
        # Add tracking for debugging
        self.debug_last_large_priority = None
        self.debug_last_large_priority_idx = None
    
    # def update(self, data_indices, priorities):
    #     # Cap priorities to prevent extreme values
    #     priorities = T.clamp(priorities, min=1e-6)

    #     # Track maximum priority
    #     if priorities.numel() > 0:
    #         self.max_priority = T.max(T.cat([self.max_priority.unsqueeze(0), T.max(priorities).unsqueeze(0)]))

    #     # Compute tree indices once
    #     tree_indices = data_indices + self.capacity - 1

    #     # Update leaf nodes in one operation
    #     self.tree[tree_indices] = priorities

    #     # Update parent nodes for each leaf individually - less vectorized but correct
    #     for idx in tree_indices:
    #         idx_item = idx.item()
    #         parent = (idx_item - 1) // 2

    #         # Traverse up to the root
    #         while parent >= 0:
    #             # Get children of this parent
    #             left = 2 * parent + 1
    #             right = 2 * parent + 2

    #             # Update the parent (handle case where right child might not exist)
    #             if right < self.tree.size(0):
    #                 self.tree[parent] = self.tree[left] + self.tree[right]
    #             else:
    #                 self.tree[parent] = self.tree[left]

    #             # Move to next parent up the tree
    #             parent = (parent - 1) // 2

    def update(self, data_indices, priorities):
        # Cap priorities to prevent extreme values
        priorities = T.clamp(priorities, min=1e-6)

        # Track maximum priority
        if priorities.numel() > 0:
            self.max_priority = T.max(T.cat([self.max_priority.unsqueeze(0), T.max(priorities).unsqueeze(0)]))

        # Compute tree indices once
        indices = data_indices + self.capacity - 1

        # Update leaf nodes in one operation
        self.tree[indices] = priorities

        # Vectorized propagation: process level-by-level up the tree
        while True:
            parents = (indices - 1) // 2
            unique_parents, _ = T.unique(parents, return_inverse=True)  # Get unique to avoid redundant updates
            if unique_parents.min() >= 0:  # Continue until we reach the root
                left_children = 2 * unique_parents + 1
                right_children = 2 * unique_parents + 2
                has_right = right_children < self.tree.size(0)
                
                # Sum children for each unique parent
                sums = self.tree[left_children].clone()  # Start with left
                sums[has_right] += self.tree[right_children[has_right]]  # Add right if exists
                
                self.tree[unique_parents] = sums
                indices = unique_parents  # Move up to parents for next level
            else:
                break

    @T.jit.script
    def _traverse_tree(p_values: T.Tensor, tree: T.Tensor, capacity: int) -> T.Tensor:
        batch_size = p_values.size(0)
        indices = T.zeros(batch_size, dtype=T.long, device=p_values.device)
        
        for i in range(batch_size):
            idx = 0  # Start at root
            p = p_values[i]
            
            # Binary search through the tree
            for _ in range(int(T.log2(T.tensor(capacity)).ceil().item())):
                left = 2 * idx + 1
                if left >= tree.size(0):
                    break
                    
                left_val = tree[left]
                if p <= left_val:
                    idx = left
                else:
                    p = p - left_val
                    idx = left + 1
                    
                if idx >= capacity - 1:  # Reached leaf nodes
                    break
                    
            indices[i] = idx
        
        return indices

    def get(self, p_values: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """Optimized sampling with JIT acceleration"""
        indices = self._traverse_tree(p_values, self.tree, self.capacity)
        
        # Ensure leaf node validity and get priorities
        indices = T.clamp(indices, 0, self.tree.size(0) - 1)
        priorities = self.tree[indices]
        
        # Convert to data indices
        data_indices = T.clamp(indices - (self.capacity - 1), 0, self.capacity - 1)
        
        return data_indices, priorities
    
    @property
    def total_priority(self) -> float:
        """Return the total priority (value at root)."""
        return self.tree[0].item() if self.tree.size(0) > 0 else 0.0

class Buffer:
    """
    Base class for replay buffers with N-step functionality.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        device: Optional[str] = None
    ):
        self.env = env
        self.buffer_size = buffer_size
        self.device = get_device(device)

        # Set observation, goal, and action space shapes
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            self.obs_space_shape = self.env.single_observation_space[self.env.obs_key].shape
            if self.env.goal_key is not None:
                self.goal_space_shape = self.env.single_observation_space[self.env.goal_key].shape
            else:
                self.goal_space_shape = None
        else:
            self.obs_space_shape = self.env.single_observation_space.shape
            self.goal_space_shape = None

        if isinstance(self.env.single_action_space, gym.spaces.Box):
            self.action_type = T.float32
            self.action_space_shape = self.env.single_action_space.shape
        else: # Discrete
            self.action_type = T.long
            self.action_space_shape = (1,)

    @abstractmethod
    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a transition to the buffer, including trajectory metadata.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    # @abstractmethod
    # def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
    #     """
    #     Sample a batch of transitions from the buffer.
    #     Abstract method to be implemented by subclasses.
    #     """
    #     raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "device": self.device.type
            }
        }

    @classmethod
    def create_instance(cls, buffer_type: str, **kwargs) -> 'Buffer':
        buffer_types = {
            "ReplayBuffer": ReplayBuffer,
            "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
            "RolloutBuffer": RolloutBuffer,
            "TrajectoryBuffer": TrajectoryBuffer,
        }
        if buffer_type in buffer_types:
            return buffer_types[buffer_type](**kwargs)
        else:
            raise ValueError(f"{buffer_type} is not a subclass of Buffer")

class ReplayBuffer(Buffer):
    """
    Off-Policy replay buffer with N-step sequence sampling.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100000,
        N: int = 1,
        device: str | T.device | None = None,
    ):
        super().__init__(env, buffer_size, device)
        self.N = N  # N-step hyperparameter
        self.counter = 0
        
        self.states = T.zeros((buffer_size, N, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, N, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, N, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.terminations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.trajectory_lengths = T.zeros((buffer_size,), dtype=T.int64, device=self.device)
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
        
        # self.counter = 0
        self.gen = np.random.default_rng()

    def record(self, cur_observation: Observation, **kwargs: Any) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            **kwargs: Any: Additional arguments to pass to the add method.
        """
        if cur_observation.n_step_trajectory is not None:
            self.add(**cur_observation.n_step_trajectory)
        else:
            raise ValueError("n-step trajectory is None. Must use VectorNStepReward wrapper when using ReplayBuffer.")

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        state_achieved_goals: Optional[T.Tensor] = None,
        next_state_achieved_goals: Optional[T.Tensor] = None,
        desired_goals: Optional[T.Tensor] = None,
        trajectory_lengths: Optional[T.Tensor] = None,
    ) -> None:
        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), T.arange(0, end_idx, device=self.device)])

        # Add N dimension of 1 at index 1 if values are 2d
        if states.ndim == 2:
            states = states[:, T.newaxis, :]
            # states = states.unsqueeze(1)
        if actions.ndim == 2:
            actions = actions[:, T.newaxis, :]
            # actions = actions.unsqueeze(1)
        if rewards.ndim == 1:
            rewards = rewards[:, T.newaxis]
            # rewards = rewards.unsqueeze(1)
        if next_states.ndim == 2:
            next_states = next_states[:, T.newaxis, :]
            # next_states = next_states.unsqueeze(1)
        if terminations.ndim == 1:
            terminations = terminations[:, T.newaxis]
        if truncations.ndim == 1:
            truncations = truncations[:, T.newaxis]

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            if state_achieved_goals.ndim == 2:
                state_achieved_goals = state_achieved_goals[:, T.newaxis, :]
                # state_achieved_goals = state_achieved_goals.unsqueeze(1)
            if next_state_achieved_goals.ndim == 2:
                next_state_achieved_goals = next_state_achieved_goals[:, T.newaxis, :]
                # next_state_achieved_goals = next_state_achieved_goals.unsqueeze(1)
            if desired_goals.ndim == 2:
                desired_goals = desired_goals[:, T.newaxis, :]

        # Store transitions (detach to avoid holding computation graphs)
        self.states[indices] = states.detach().to(device=self.device, dtype=T.float32)
        self.actions[indices] = actions.detach().to(device=self.device, dtype=self.action_type)
        self.rewards[indices] = rewards.detach().to(device=self.device, dtype=T.float32)
        self.next_states[indices] = next_states.detach().to(device=self.device, dtype=T.float32)
        self.terminations[indices] = terminations.detach().to(device=self.device, dtype=T.bool)
        self.truncations[indices] = truncations.detach().to(device=self.device, dtype=T.bool)
        self.trajectory_lengths[indices] = trajectory_lengths.detach().to(device=self.device, dtype=T.int64)
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.state_achieved_goals[indices] = state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.next_state_achieved_goals[indices] = next_state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.desired_goals[indices] = desired_goals.detach().to(device=self.device, dtype=T.float32)

        self.counter += batch_size

    def sample(self, batch_size: int) -> Dict[str, T.Tensor]:
        """Returns a dictionary of n-step sequences sampled from the buffer.
        
        Args:
            batch_size: int: The number of samples to draw from the buffer.
        
        Returns:
            Dict[str, T.Tensor]: A dictionary containing the sampled n-step sequences.
        """
        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = self.gen.integers(0, size, (batch_size,))

        sample = {
            "states": self.states[indices].clone(),
            "actions": self.actions[indices].clone(),
            "rewards": self.rewards[indices].clone(),
            "next_states": self.next_states[indices].clone(),
            "terminations": self.terminations[indices].clone(),
            "truncations": self.truncations[indices].clone(),
            "trajectory_lengths": self.trajectory_lengths[indices].clone(),
        }
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            sample.update({
                "state_achieved_goals": self.state_achieved_goals[indices].clone(),
                "next_state_achieved_goals": self.next_state_achieved_goals[indices].clone(),
                "desired_goals": self.desired_goals[indices].clone(),
            })
        else:
            sample.update({
                "state_achieved_goals": None,
                "next_state_achieved_goals": None,
                "desired_goals": None,
            })

        return sample
    
    def reset(self) -> None:
        """
        Reset the buffer to all zeros and the counter to zero.
        """
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.trajectory_lengths.zero_()
        self.counter = 0
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals.zero_()
            self.state_achieved_goals.zero_()
            self.next_state_achieved_goals.zero_()

    # def clone(self, device: Optional[str] = None) -> 'ReplayBuffer':
    #     """
    #     Clone the replay buffer.

    #     Returns:
    #         ReplayBuffer: A new instance of the replay buffer with the same configuration.
    #     """
    #     if device:
    #         device = get_device(device)
    #     else:
    #         device = self.device

    #     env = build_env_wrapper_obj(self.env.config)
    #     return ReplayBuffer(env, self.buffer_size, device)

    def is_ready(self, batch_size: int, warmup: int) -> bool:
        """
        Check if the buffer is ready to sample.
        """
        return (self.counter >= warmup) and (self.counter >= batch_size)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            "N": self.N
        })
        return config

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized off-policy replay buffer that samples transitions based on TD error.
    Supports both proportional and rank-based prioritization strategies.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_iter: int = 100_000,
        beta_update_freq: int = 10,
        priority: str = 'rank',
        normalize: bool = False,  # Only applies to proportional priority strategy
        epsilon: float = 1e-6,
        N: int = 1,
        device: Optional[str] = None,
    ):
        if priority not in ['proportional', 'rank']:
            raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

        super().__init__(env, buffer_size, N, device)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_iter = beta_iter
        self.priority = priority
        self.normalize = normalize
        self.epsilon = epsilon
        self.beta_update_freq = beta_update_freq
        self.beta = self.beta_start
        self._total_steps = 0
        # self.N = N  # Store N-step hyperparameter

        # Tensors for trajectory metadata
        # self.traj_ids = T.zeros(buffer_size, dtype=T.long, device=self.device)
        # self.step_indices = T.zeros(buffer_size, dtype=T.long, device=self.device)

        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank-based
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices = None

        # self.counter = 0

    def record(self, cur_observation: Observation, **kwargs: Any) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            **kwargs: Any: Additional arguments to pass to the add method.
        """
        if cur_observation.n_step_trajectory is not None:
            self.add(**cur_observation.n_step_trajectory)
        else:
            raise ValueError("n-step trajectory is None. Must use VectorNStepReward wrapper when using ReplayBuffer.")

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        dones: T.Tensor,
        state_achieved_goals: Optional[T.Tensor] = None,
        next_state_achieved_goals: Optional[T.Tensor] = None,
        desired_goals: Optional[T.Tensor] = None,
        trajectory_lengths: Optional[T.Tensor] = None,
    ) -> None:
        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), 
                             T.arange(0, end_idx, device=self.device)])

        # Add N dimension of 1 at index 1 if values are 2d
        if states.ndim == 2:
            states = states[:, T.newaxis, :]
            # states = states.unsqueeze(1)
        if actions.ndim == 2:
            actions = actions[:, T.newaxis, :]
            # actions = actions.unsqueeze(1)
        if rewards.ndim == 1:
            rewards = rewards[:, T.newaxis]
            # rewards = rewards.unsqueeze(1)
        if next_states.ndim == 2:
            next_states = next_states[:, T.newaxis, :]
            # next_states = next_states.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones[:, T.newaxis]


        if self.env.goal_key is not None and self.goal_space_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            if state_achieved_goals.ndim == 2:
                state_achieved_goals = state_achieved_goals[:, T.newaxis, :]
                # state_achieved_goals = state_achieved_goals.unsqueeze(1)
            if next_state_achieved_goals.ndim == 2:
                next_state_achieved_goals = next_state_achieved_goals[:, T.newaxis, :]
                # next_state_achieved_goals = next_state_achieved_goals.unsqueeze(1)
            if desired_goals.ndim == 2:
                desired_goals = desired_goals[:, T.newaxis, :]

        # Store transitions (detach to avoid holding computation graphs)
        self.states[indices] = states.detach().to(device=self.device, dtype=T.float32)
        self.actions[indices] = actions.detach().to(device=self.device, dtype=self.action_type)
        self.rewards[indices] = rewards.detach().to(device=self.device, dtype=T.float32)
        self.next_states[indices] = next_states.detach().to(device=self.device, dtype=T.float32)
        self.dones[indices] = dones.detach().to(device=self.device, dtype=T.bool)
        self.trajectory_lengths[indices] = trajectory_lengths.detach().to(device=self.device, dtype=T.int64)

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.state_achieved_goals[indices] = state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.next_state_achieved_goals[indices] = next_state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.desired_goals[indices] = desired_goals.detach().to(device=self.device, dtype=T.float32)

        # Set initial priorities (will be normalized in update)
        if self.priority == "proportional":
            priorities = T.ones(len(indices), device=self.device) * self.sum_tree.max_priority
            self.sum_tree.update(indices, priorities)
        else:  # rank-based
            self.priorities[indices] = T.ones(len(indices), device=self.device) * self.priorities.max()
            self.sorted_indices = None

        self.counter += batch_size
        self._total_steps += 1


    def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
        """Samples a batch of N-step transition sequences based on priority."""
        if self._total_steps % self.beta_update_freq == 0:
            self.update_beta()

        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, size)

        if self.priority == "proportional":
            total_priority = self.sum_tree.total_priority
            if total_priority <= 0:
                indices = T.randint(0, size, (batch_size,), device=self.device)
                weights = T.ones(batch_size, device=self.device)
                probs = T.ones(batch_size, device=self.device) / size
            else:
                segment_size = total_priority / batch_size
                segment_boundaries = T.arange(0, batch_size, device=self.device) * segment_size
                random_offsets = T.rand(batch_size, device=self.device) * segment_size
                p_values = segment_boundaries + random_offsets
                indices, priorities = self.sum_tree.get(p_values)
                probs = priorities / total_priority
                weights = (size * probs) ** (-self.beta)
                weights = weights / weights.max()
        else:  # rank-based
            self._prepare_rank_based()
            u = T.rand(batch_size, device=self.device)
            ranks = (u ** (1 / self.alpha) * size).long().clamp(max=size-1)
            indices = self.sorted_indices[ranks]
            cur_probs = 1 / ((ranks + 1) ** self.alpha)
            all_ranks = T.arange(size, device=self.device)
            sum_probs = T.sum(1 / (all_ranks + 1.0) ** self.alpha)
            probs = cur_probs / sum_probs
            weights = (size * probs) ** (-self.beta)
            weights = weights / weights.max()

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices],
            self.dones[indices], self.state_achieved_goals[indices], self.next_state_achieved_goals[indices],
            self.desired_goals[indices], self.trajectory_lengths[indices], weights, probs, indices)
        else:
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices],
            self.dones[indices], self.trajectory_lengths[indices], weights, probs, indices)

    def update_beta(self) -> None:
        """Anneal beta param"""
        progress = min(self._total_steps / self.beta_iter, 1.0)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)

    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor) -> None:
        """Updates priorities of sampled transitions"""
        if not isinstance(indices, T.Tensor):
            indices = T.tensor(indices, device=self.device)
        
        if not isinstance(priorities, T.Tensor):
            priorities = T.tensor(priorities, device=self.device)

        priorities = T.abs(priorities)

        if self.priority == "proportional":
            if priorities.numel() > 1 and self.normalize:
                mean = priorities.mean()
                std = priorities.std() + 1e-6
                normalized = (priorities - mean) / std
                priorities = T.clamp(normalized, -3.0, 3.0)
                priorities = ((normalized + 3.0) / 6.0) + self.epsilon
            else:
                priorities = T.clamp(priorities, min=self.epsilon)

            priorities = priorities ** self.alpha
            if T.isnan(priorities).any():
                nan_mask = T.isnan(priorities)
                mean_non_nan = priorities[~nan_mask].mean()
                priorities = T.where(nan_mask, mean_non_nan, priorities)

            self.sum_tree.update(indices, priorities)
        else:  # rank-based
            self.priorities[indices] = priorities
            self.sorted_indices = None

    def _prepare_rank_based(self) -> None:
        """Sorts priorities for rank-based sampling"""
        if self.sorted_indices is None:
            size = min(self.counter, self.buffer_size)
            if size > 0:
                self.sorted_indices = T.argsort(self.priorities[:size], descending=True)
            else:
                self.sorted_indices = T.tensor([], dtype=T.long, device=self.device)

    def get_config(self) -> Dict[str, Any]:
        """Get buffer config."""
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_iter": self.beta_iter,
            "beta_update_freq": self.beta_update_freq,
            "priority": self.priority,
            "normalize": self.normalize,
            "epsilon": self.epsilon,
        })
        return config
    
    def clone(self, device: Optional[str] = None) -> 'PrioritizedReplayBuffer':
        """Create a new instance with the same configuration."""
        if device:
            device = get_device(device)
        else:
            device = self.device.type

        env = build_env_wrapper_obj(self.env.config)
        return PrioritizedReplayBuffer(
            env, 
            self.buffer_size, 
            self.alpha, 
            self.beta_start, 
            self.beta_iter,
            self.beta_update_freq,
            self.priority, 
            self.normalize,
            self.epsilon,
            device,
            self.N
        )

class RolloutBuffer(Buffer):
    """
    On-Policy buffer for storing rollouts.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        device: Optional[str] = None
    ):
        super().__init__(env, buffer_size, device)
        self.cur_idx = T.zeros((env.num_envs,), dtype=T.long, device=self.device)

        #Instantiate buffers
        self.states = T.zeros((buffer_size, env.num_envs, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, env.num_envs, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.rewards = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, env.num_envs, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.terminations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        # Tracks initial steps of each environment (Phantom steps)
        self.first_steps = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)

    def record(self, cur_observation: Observation, prev_observation: Observation, actions: T.Tensor, prev_dones: T.Tensor) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: T.Tensor: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.add(
            states=prev_observation.states,
            actions=actions,
            rewards=cur_observation.rewards,
            next_states=cur_observation.states,
            terminations=cur_observation.terminations,
            truncations=cur_observation.truncations,
            state_achieved_goals=prev_observation.ach_goals if prev_observation.ach_goals is not None else None,
            next_state_achieved_goals=cur_observation.ach_goals if cur_observation.ach_goals is not None else None,
            desired_goals=prev_observation.goals if prev_observation.goals is not None else None,
            first_steps=prev_dones,
        )

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        state_achieved_goals: T.Tensor|None = None,
        next_state_achieved_goals: T.Tensor|None = None,
        desired_goals: T.Tensor|None = None,
        first_steps: T.Tensor|None = None,
    ) -> None:

        # Ensure actions are 2d tensors
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        # Check valid_steps and create if None
        if first_steps is None:
            first_steps = T.zeros((self.env.num_envs,), dtype=T.bool, device=self.device)
        else:
            first_steps = first_steps.to(device=self.device, dtype=T.bool)
            if first_steps.numel() != self.env.num_envs:
                raise ValueError(f"first_steps must have {self.env.num_envs} elements, got {first_steps.numel()}")
        # Get per environment indices to add values to
        idx = self.cur_idx.clone()
        # Get env id's
        env_ids = T.arange(self.env.num_envs, device=self.device)

        
        # Add values to buffers at indices
        self.states[idx, env_ids] = states.to(device=self.device, dtype=T.float32)
        self.actions[idx, env_ids] = actions.to(device=self.device, dtype=self.action_type)
        self.rewards[idx, env_ids] = rewards.to(device=self.device, dtype=T.float32)
        self.next_states[idx, env_ids] = next_states.to(device=self.device, dtype=T.float32)
        self.terminations[idx, env_ids] = terminations.to(device=self.device, dtype=T.bool)
        self.truncations[idx, env_ids] = truncations.to(device=self.device, dtype=T.bool)
        self.first_steps[idx, env_ids] = first_steps
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.state_achieved_goals[idx, env_ids] = state_achieved_goals.to(device=self.device, dtype=T.float32)
            self.next_state_achieved_goals[idx, env_ids] = next_state_achieved_goals.to(device=self.device, dtype=T.float32)
            self.desired_goals[idx, env_ids] = desired_goals.to(device=self.device, dtype=T.float32)
        
        # Increment step indices
        self.cur_idx += 1

    def sample(self) -> Dict[str, T.Tensor]:
        """
        Returns a dictionary of all buffer tensors up to the current index of each environment.
        Current index values will match across all tensors because all rollouts are of same length.
        """
        idx = int(self.cur_idx.max().item())
        if idx <= 0:
            raise ValueError("Cannot sample from empty buffer")

        # Create tensor of non first step (Phantom) indices for valid training samples
        first_steps = self.first_steps[:idx].clone()
        valid_indices = (first_steps.reshape(-1) == 0).nonzero()
        sample = {
            "states": self.states[:idx].clone(),
            "actions": self.actions[:idx].clone(),
            "rewards": self.rewards[:idx].clone(),
            "next_states": self.next_states[:idx].clone(),
            "terminations": self.terminations[:idx].clone(),
            "truncations": self.truncations[:idx].clone(),
            "first_steps": first_steps,
            "valid_indices": valid_indices,
        }
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            sample.update({
                "state_achieved_goals": self.state_achieved_goals[:idx].clone(),
                "next_state_achieved_goals": self.next_state_achieved_goals[:idx].clone(),
                "desired_goals": self.desired_goals[:idx].clone(),
            })
        else:
            sample.update({
                "state_achieved_goals": None,
                "next_state_achieved_goals": None,
                "desired_goals": None,
            })

        self.reset()
        return sample

    def reset(self) -> None:
        """Reset the current index of each environment to zero."""
        self.cur_idx.zero_()
        self.first_steps.zero_()

    def is_ready(self, **kwargs: Any) -> bool:
        """
        Check if the buffer is ready to sample. Always returns True.
        """
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get buffer config."""
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

class TrajectoryBuffer(RolloutBuffer):
    """
    On-Policy buffer for storing completedtrajectories
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        device: Optional[str] = None
    ):
        super().__init__(env, buffer_size, device)
        self.completed_trajectories: List[Dict[str, T.Tensor]] = []

    def record(self, cur_observation: Observation, prev_observation: Observation, actions: T.Tensor, prev_dones: T.Tensor) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: T.Tensor: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.add(
            states=prev_observation.states,
            actions=actions,
            rewards=cur_observation.rewards,
            next_states=cur_observation.states,
            terminations=cur_observation.terminations,
            truncations=cur_observation.truncations,
            state_achieved_goals=prev_observation.ach_goals if prev_observation.ach_goals is not None else None,
            next_state_achieved_goals=cur_observation.ach_goals if cur_observation.ach_goals is not None else None,
            desired_goals=prev_observation.goals if prev_observation.goals is not None else None,
            first_steps=prev_dones,
        )

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        state_achieved_goals: T.Tensor|None = None,
        next_state_achieved_goals: T.Tensor|None = None,
        desired_goals: T.Tensor|None = None,
        first_steps: T.Tensor|None = None,
    ) -> None:
        super().add(states, actions, rewards, next_states, terminations, truncations, state_achieved_goals, next_state_achieved_goals, desired_goals, first_steps)

        # Store completed trajectories if any last stored dones are True
        for i in range(self.env.num_envs):
            idx = int(self.cur_idx[i].item())
            if idx <= 0:
                continue
            done = bool(self.terminations[idx - 1, i].item() or self.truncations[idx - 1, i].item())
            if not done:
                continue
            # Check to make sure there are valid steps in the trajectory
            valid_steps = T.logical_not(self.first_steps[:idx, i])
            if not valid_steps.any():
                self.cur_idx[i] = 0
                continue

            trajectory = {
                "states": self.states[:idx, i][valid_steps].clone(),
                "actions": self.actions[:idx, i][valid_steps].clone(),
                "rewards": self.rewards[:idx, i][valid_steps].clone(),
                "next_states": self.next_states[:idx, i][valid_steps].clone(),
                "terminations": self.terminations[:idx, i][valid_steps].clone(),
                "truncations": self.truncations[:idx, i][valid_steps].clone(),
            }
            if self.env.goal_key is not None and self.goal_space_shape is not None:
                trajectory.update({
                    "state_achieved_goals": self.state_achieved_goals[:idx, i][valid_steps].clone(),
                    "next_state_achieved_goals": self.next_state_achieved_goals[:idx, i][valid_steps].clone(),
                    "desired_goals": self.desired_goals[:idx, i][valid_steps].clone(),
                })
            else:
                trajectory.update({
                    "state_achieved_goals": None,
                    "next_state_achieved_goals": None,
                    "desired_goals": None,
                })
            self.completed_trajectories.append(trajectory)
            # Reset step counter for done env
            self.cur_idx[i] = 0

    def sample(self) -> List[Dict[str, T.Tensor]]:
        """Returns a list of completed trajectories."""
        trajectories = self.completed_trajectories[:]
        # Clear trajectories
        self.completed_trajectories = []
        return trajectories

    def reset(self) -> None:
        super().reset()
        self.completed_trajectories = []