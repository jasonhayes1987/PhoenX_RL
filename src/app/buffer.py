import torch as T
import numpy as np
import gymnasium as gym
from env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from utils import build_env_wrapper_obj
from torch_utils import get_device
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
        self.tree = T.zeros(2 * capacity - 1, dtype=T.float32, device=self.device)
        self.next_idx = 0
        self.size = 0
        self.max_priority = T.tensor(0, dtype=T.float32, device=self.device)
        # self.epsilon = T.tensor(1e-6, dtype=T.float32, device=self.device)
    
    def update(self, data_indices, priorities):
        """Update the priorities of the given data indices."""
        #DEBUG
        # print(f'SumTree.update() data indices: {data_indices}')
        # print(f'SumTree.update() priorities: {priorities}')
        # print(f'SumTree.update() updating priorities')
        
        # Safety check for NaN values
        if T.isnan(priorities).any():
            print("WARNING: NaN values detected in SumTree.update priorities. Replacing with 1.0")
            priorities = T.nan_to_num(priorities, nan=1.0)
        
        # Compute tree indices (leaf nodes) from data indices
        tree_indices = T.tensor([idx + self.capacity - 1 for idx in data_indices], dtype=T.long, device=self.device)
        
        # Update leaf nodes with new priorities
        self.tree[tree_indices] = priorities
        
        # Update parent nodes to maintain the sum tree property
        self._update_tree(tree_indices)
        
        # Update max_priority
        if not T.isnan(priorities).all():  # Ensure we're not using all NaN values
            valid_values = priorities[~T.isnan(priorities)]
            if valid_values.numel() > 0:  # Check if there are any valid values
                max_new = T.max(valid_values)
                if max_new > self.max_priority:
                    self.max_priority = max_new
                    #DEBUG
                    print(f"SumTree.update() New max priority: {self.max_priority}")

        # print(f'SumTree.update() updated priorities')

    def _update_tree(self, indices):
        """Update parent nodes in the tree to maintain the sum property."""
        # Find all unique parent levels that need updating
        parent_indices = set()
        for idx in indices:
            # Get parent index
            parent_idx = (idx - 1) // 2
            while parent_idx >= 0:
                parent_indices.add(int(parent_idx))
                parent_idx = (parent_idx - 1) // 2
        
        # Sort parent indices in reverse order (bottom-up)
        parent_indices = sorted(list(parent_indices), reverse=True)
        
        # Update each parent node based on its children
        for parent_idx in parent_indices:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            
            # Ensure we don't go out of bounds
            if right_child < len(self.tree):
                # Sum the children
                self.tree[parent_idx] = self.tree[left_child] + self.tree[right_child]
            else:
                # If right child is out of bounds, use only left child
                self.tree[parent_idx] = self.tree[left_child]

    # def get(self, p: float) -> Tuple[int, float]:
    #     """Find sample based on a value p in range [0, total_priority)."""
    #     idx = self._retrieve(0, p)
    #     data_idx = idx - self.capacity + 1
    #     return int(data_idx), float(self.tree[idx].item())

    def get(self, p_values: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        Vectorized batch sampling from the SumTree.
        
        Args:
            p_values: Tensor of probability values to sample with [batch_size]
            
        Returns:
            Tuple of (data_indices, priorities)
        """
        batch_size = p_values.size(0)
        
        # Pre-allocate space for resulting indices
        tree_indices = T.zeros(batch_size, dtype=T.long, device=self.device)
        
        # Use an iterative approach for each sample in the batch
        for i in range(batch_size):
            p = p_values[i].item()
            idx = 0  # Start at root
            
            # Iterative traversal without recursion
            while True:
                left = 2 * idx + 1
                right = left + 1
                
                # If at leaf node, store index and break
                if left >= len(self.tree):
                    tree_indices[i] = idx
                    break
                
                # Get the left child value just once per iteration
                left_val = self.tree[left].item()
                
                # Determine direction
                if p <= left_val:
                    idx = left
                else:
                    p = p - left_val
                    idx = right
        
        # Convert tree indices to data indices and get priorities in one batch operation
        data_indices = tree_indices - (self.capacity - 1)
        priorities = self.tree[tree_indices]
        
        return data_indices, priorities
    
    def _retrieve(self, idx: int, p: float) -> int:
        """Traverse tree to find leaf node containing value s."""
        left = 2 * idx + 1
        right = left + 1
        
        # If at leaf node, return idx
        if left >= len(self.tree):
            return idx
        
        # Otherwise, traverse down
        if p <= self.tree[left].item():
            return self._retrieve(left, p)
        else:
            return self._retrieve(right, p - self.tree[left].item())
    
    @property
    def total_priority(self) -> float:
        """Get total priority (stored at root node)."""
        return float(self.tree[0].item())

class Buffer():
    """
    Base class for replay buffers.
    """

    def __init__(self, env: EnvWrapper, buffer_size: int, device: Optional[str] = None):
        self.device = get_device(device)
        self.env = env
        self.buffer_size = buffer_size

    def add(self, *args, **kwargs):
        """
        Add a transition to the buffer.
        """
        pass
    
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            Tuple: Sampled transitions.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration of the buffer.

        Returns:
            dict: Configuration details.
        """
        pass

    @classmethod
    def create_instance(cls, buffer_class_name: str, **kwargs) -> 'Buffer':
        """
        Create an instance of the requested buffer class.

        Args:
            buffer_class_name (str): Name of the buffer class.
            kwargs: Parameters for the buffer class.

        Returns:
            Buffer: An instance of the requested buffer class.

        Raises:
            ValueError: If the buffer class is not recognized.
        """
        buffer_classes = {
            "ReplayBuffer": ReplayBuffer,
        }

        if buffer_class_name in buffer_classes:
            return buffer_classes[buffer_class_name](**kwargs)
        else:
            raise ValueError(f"{buffer_class_name} is not a subclass of Buffer")
class ReplayBuffer(Buffer):
    """
    Replay buffer for storing transitions during reinforcement learning.

    Attributes:
        env (EnvWrapper): The environment wrapper associated with the buffer.
        buffer_size (int): Maximum size of the buffer.
        goal_shape (Optional[tuple]): Shape of goals (if used).
        device (str): Device to store the buffer ('cpu' or 'cuda').
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100000,
        goal_shape: Optional[Tuple[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the ReplayBuffer.

        Args:
            env (EnvWrapper): The environment wrapper object.
            buffer_size (int): Maximum size of the buffer.
            goal_shape (Optional[tuple]): Shape of goals, if applicable.
            device (Optional[str]): Device to store buffer data ('cpu' or 'cuda').
        """
        super().__init__(env, buffer_size, device)
        self.goal_shape = goal_shape
        
        # Determine observation space shape
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            self._obs_space_shape = self.env.single_observation_space['observation'].shape
        else:
            self._obs_space_shape = self.env.single_observation_space.shape

        self.states = T.zeros((buffer_size, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, *self.env.single_action_space.shape), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size,), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.dones = T.zeros((buffer_size,), dtype=T.int8, device=self.device)
        
        if self.goal_shape is not None:
            self.desired_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
        
        self.counter = 0
        self.gen = np.random.default_rng()

    def reset(self) -> None:
        """
        Reset the buffer to all zeros and the counter to zero.
        """
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.dones.zero_()
        self.counter = 0
        
        if self.goal_shape is not None:
            self.desired_goals.zero_()
            self.state_achieved_goals.zero_()
            self.next_state_achieved_goals.zero_()
        
        
    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        next_states: np.ndarray,
        dones: bool,
        state_achieved_goals: Optional[np.ndarray] = None,
        next_state_achieved_goals: Optional[np.ndarray] = None,
        desired_goals: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a transition to the replay buffer.

        Args:
            states (np.ndarray): Current states.
            actions (np.ndarray): Actions taken.
            rewards (float): Rewards received.
            next_states (np.ndarray): Next states.
            dones (bool): Whether the episode is done.
            state_achieved_goals (Optional[np.ndarray]): Achieved goals in the current state.
            next_state_achieved_goals (Optional[np.ndarray]): Achieved goals in the next state.
            desired_goals (Optional[np.ndarray]): Desired goals.
        """
        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        # Compute indices with wrapping
        if end_idx > start_idx:
            indices = np.arange(start_idx, end_idx)
        else:
            indices = np.concatenate([np.arange(start_idx, self.buffer_size), np.arange(0, end_idx)])

        # Convert lists to numpy arrays and then to tensors in one operation
        self.states[indices] = T.tensor(np.array(states), dtype=T.float32, device=self.device)
        self.actions[indices] = T.tensor(np.array(actions), dtype=T.float32, device=self.device)
        self.rewards[indices] = T.tensor(np.array(rewards), dtype=T.float32, device=self.device)
        self.next_states[indices] = T.tensor(np.array(next_states), dtype=T.float32, device=self.device)
        self.dones[indices] = T.tensor(np.array(dones), dtype=T.int8, device=self.device)

        if self.goal_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            self.state_achieved_goals[indices] = T.tensor(np.array(state_achieved_goals), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals[indices] = T.tensor(np.array(next_state_achieved_goals), dtype=T.float32, device=self.device)
            self.desired_goals[indices] = T.tensor(np.array(desired_goals), dtype=T.float32, device=self.device)

        self.counter += batch_size
        
    def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[T.Tensor, ...]: Sampled transitions.
        """
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
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration of the replay buffer.

        Returns:
            Dict[str, Any]: Configuration details.
        """
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "goal_shape": self.goal_shape,
                "device": self.device.type,
            }
        }
    
    def clone(self) -> 'ReplayBuffer':
        """
        Clone the replay buffer.

        Returns:
            ReplayBuffer: A new instance of the replay buffer with the same configuration.
        """
        env = build_env_wrapper_obj(self.env.config)
        return ReplayBuffer(env, self.buffer_size, self.goal_shape, self.device)

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer that samples transitions based on TD error.
    Supports both proportional and rank-based prioritization strategies.
    All tensor operations happen on the specified device to minimize data transfers.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_iter: int = 100_000, # Num iters to anneal beta from start to 1.0
        priority: str = 'proportional', # or 'rank'
        goal_shape: Optional[Tuple[int]] = None,
        epsilon: float = 1e-6,
        update_freq: int = 10,
        device: Optional[str] = None
    ):
        """ Initialize the PrioritizedReplayBuffer."""

        if priority not in ['proportional', 'rank']:
            raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

        super().__init__(env, buffer_size, goal_shape, device)
        self.alpha = T.tensor(alpha, dtype=T.float32, device=self.device)
        self.beta_start = T.tensor(beta_start, dtype=T.float32, device=self.device)
        self.beta_iter = T.tensor(beta_iter, dtype=T.int32, device=self.device)
        self.priority = priority
        self.goal_shape = goal_shape
        self.epsilon = T.tensor(epsilon, dtype=T.float32, device=self.device)
        self.update_freq = update_freq
        # Set initial self.beta value
        self.beta = self.beta_start

        # Determine observation space shape
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            self._obs_space_shape = self.env.single_observation_space['observation'].shape
        else:
            self._obs_space_shape = self.env.single_observation_space.shape

        # Initialize transition storage
        self.states = T.zeros((buffer_size, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, *self.env.single_action_space.shape), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size,), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.dones = T.zeros((buffer_size,), dtype=T.int8, device=self.device)
        
        # Handle goal-related data if needed
        if self.goal_shape is not None:
            self.desired_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
        
        # Initialize prioritization data structures
        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank-based
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices = None  # Will be computed when needed
        
        self.counter = 0

    def reset(self) -> None:
        """Resets the buffer to it's initial state, all zeros

        """
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.dones.zero_()

        if self.goal_shape is not None:
            self.desired_goals.zero_()
            self.state_achieved_goals.zero_()
            self.next_state_achieved_goals.zero_()

        if self.priority == "proportional":
            self.sum_tree = SumTree(self.buffer_size, self.device)
        else:
            self.priorities.zero_()
            self.sorted_indices = None

        self.counter = 0
        self.beta = self.beta_start

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        next_states: np.ndarray,
        dones: bool,
        state_achieved_goals: Optional[np.ndarray] = None,
        next_state_achieved_goals: Optional[np.ndarray] = None,
        desired_goals: Optional[np.ndarray] = None,
    ) -> None:
        """Adds a transition to the buffer and sets priority to max"""

        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        # Compute indices with wrapping
        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx)
        else:
            indices = T.concatenate([T.arange(start_idx, self.buffer_size), T.arange(0, end_idx)])

        # Add to buffer tensors
        self.states[indices] = T.tensor(np.array(states), dtype=T.float32, device=self.device)
        self.actions[indices] = T.tensor(np.array(actions), dtype=T.float32, device=self.device)
        self.rewards[indices] = T.tensor(np.array(rewards), dtype=T.float32, device=self.device)
        self.next_states[indices] = T.tensor(np.array(next_states), dtype=T.float32, device=self.device)
        self.dones[indices] = T.tensor(np.array(dones), dtype=T.int8, device=self.device)

        if self.goal_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            self.state_achieved_goals[indices] = T.tensor(np.array(state_achieved_goals), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals[indices] = T.tensor(np.array(next_state_achieved_goals), dtype=T.float32, device=self.device)
            self.desired_goals[indices] = T.tensor(np.array(desired_goals), dtype=T.float32, device=self.device)

        # Set priorities to max_priority for new transitions
        max_priority = self.sum_tree.max_priority if self.counter > 0 else T.tensor(1.0, device=self.device)
        priority_value = max_priority ** self.alpha
        priorities = priority_value.expand(batch_size)
        self.sum_tree.update(indices, priorities)

        self.counter += batch_size

    def update_beta(self, iter: int) -> None:
        """Anneal beta param"""

        progress = min(iter / self.beta_iter, 1.0)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)

    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor) -> None:
        """Updates priorities of sampled transitions"""

        # Check if priorities is numpy or tensor and handle accordingly
        # if isinstance(priorities, np.ndarray):
        #     priorities = T.tensor(priorities, device=self.device)

        # Check for NaN values in priorities
        if T.isnan(priorities).any():
            nan_count = T.isnan(priorities).sum().item()
            print(f"WARNING: Found {nan_count} NaN values in priorities. Replacing with 1.0")
            # Create a mask for NaN values
            nan_mask = T.isnan(priorities)
            # Clone priorities to avoid modifying the original tensor
            priorities = priorities.clone()
            # Replace NaN values with 1.0
            priorities[nan_mask] = 1.0
        
        # Set priorities to be max between abs of priority or epsilon
        priorities = T.max(priorities.abs(), self.epsilon)
        #DEBUG
        # print(f"Priorities: {priorities}")

        if self.priority == "proportional":
            self.sum_tree.update(indices, priorities ** self.alpha)
            # self.sum_tree.max_priority = max(self.sum_tree.max_priority, T.tensor(priority, device=self.device))
        else: # rank
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

    # def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
    #     """Samples a batch of transitions based on priority"""

    #     size = min(self.counter, self.buffer_size)
    #     if size == 0:
    #         raise ValueError("Cannot sample from empty buffer")

    #     batch_size = min(batch_size, size)
    #     indices = T.zeros(batch_size, dtype=T.long, device=self.device)
    #     weights = T.zeros(batch_size, dtype=T.float32, device=self.device)

    #     if self.priority == "proportional":
    #         #DEBUG
    #         # print(f"PER.sample() Total priority: {self.sum_tree.total_priority}")
    #         # Calculate segment boundaries
    #         segment_size = self.sum_tree.total_priority / batch_size
    #         segment_starts = T.arange(0, batch_size, device=self.device) * segment_size
    #         segment_ends = segment_starts + segment_size
            
    #         # Generate random values within each segment
    #         random_values = T.rand(batch_size, device=self.device) * (segment_ends - segment_starts) + segment_starts
    #         #DEBUG
    #         # print(f"PER.sample() p: {random_values}")
    #         # Sample indices from sum tree for each segment
    #         for i, p in enumerate(random_values):
    #             idx, priority = self.sum_tree.get(p)
    #             indices[i] = idx
            
    #         # Get corresponding priorities for all indices
    #         tree_indices = indices + self.buffer_size - 1
    #         priorities = self.sum_tree.tree[tree_indices]
            
    #         # Calculate weights (fully vectorized)
    #         probs = (priorities / self.sum_tree.total_priority).clamp(min=self.epsilon)
    #         weights = (size * probs).pow(-self.beta)

    def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
        """Samples a batch of transitions based on priority"""
        
        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, size)
        
        if self.priority == "proportional":
            # Calculate segment boundaries
            total_priority = self.sum_tree.tree[0].item()
            segment_size = total_priority / batch_size
            
            # Generate stratified random samples efficiently 
            segment_boundaries = T.arange(0, batch_size, device=self.device) * segment_size
            random_offsets = T.rand(batch_size, device=self.device) * segment_size
            p_values = segment_boundaries + random_offsets
            
            # Get indices and priorities in a single batch operation
            indices, priorities = self.sum_tree.get(p_values)
            
            # Calculate weights fully vectorized
            probs = (priorities / total_priority).clamp(min=self.epsilon)
            weights = (size * probs).pow(-self.beta)

        else:  # rank
            # sort priorities
            self._prepare_rank_based()
            
            # Inverse transform sampling
            u = T.rand(batch_size, device=self.device)
            rank_indices = (u.pow(1 / self.alpha) * size).long().clamp(max=size-1)
                    
            # Get actual indices from sorted indices
            indices = self.sorted_indices[rank_indices]
                    
            # Calculate probabilities and weights
            sample_probs = 1 / ((rank_indices + 1).float().pow(self.alpha))
            weights = (size * sample_probs).pow(-self.beta)
        
        # Normalize weights by max weight for stability
        if weights.numel() > 0:
            weights = weights / T.max(weights)
        
        # Convert indices to Python list for later use in update_priorities
        # indices_list = indices.cpu().tolist()
        
        # Return transition batch with importance weights
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
                weights,
                indices
            )
        else:
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                weights,
                indices
            )

    def get_config(self) -> Dict[str, Any]:
        """Get buffer config."""
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "alpha": self.alpha.item(),
                "beta_start": self.beta_start.item(),
                "beta_iter": self.beta_iter.item(),
                "priority": self.priority,
                "goal_shape": self.goal_shape,
                "epsilon": self.epsilon.item(),
                "update_freq": self.update_freq,
                "device": self.device.type
            }
        }
    
    def clone(self) -> 'PrioritizedReplayBuffer':
        """Create a new instance with the same configuration."""
        env = build_env_wrapper_obj(self.env.config)
        return PrioritizedReplayBuffer(
            env, 
            self.buffer_size, 
            self.alpha, 
            self.beta_start, 
            self.beta_frames, 
            self.prioritization_type, 
            self.goal_shape, 
            self.device.type, 
            self.epsilon
        )

#TODO Dont think shared replay buffer is needed.  If needed, update to EnvWrapper
class SharedReplayBuffer(Buffer):
    def __init__(self, env:gym.Env, buffer_size:int=100000, goal_shape:tuple=None, device='cpu'):
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