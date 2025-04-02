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
        # Initialize the tree with zeros
        self.tree = T.zeros(2 * capacity - 1, dtype=T.float32, device=self.device)
        self.next_idx = 0
        # self.size = 0
        self.max_priority = T.tensor(1.0, dtype=T.float32, device=self.device)
        # Add tracking for debugging
        self.debug_last_large_priority = None
        self.debug_last_large_priority_idx = None
    
    # def update(self, data_indices, priorities):
    #     """Update the priorities of the given data indices."""
        # Debug large priorities
        # if priorities.numel() > 0:
        #     max_val = T.max(priorities)
        #     if max_val > 1e6:  # Track suspiciously large priorities
        #         large_idx = T.argmax(priorities)
        #         self.debug_last_large_priority = max_val.item()
        #         self.debug_last_large_priority_idx = data_indices[large_idx].item()
        #         print(f"WARNING: Large priority detected: {max_val.item():.2e} at buffer index {data_indices[large_idx].item()}")
        
        # Safety check for NaN values
        # if T.isnan(priorities).any():
        #     priorities = T.nan_to_num(priorities, nan=1.0)
        #     print("WARNING: NaN priorities detected and replaced with 1.0")
        
        # Update max recorded priority if needed (before normalization)
        # if priorities.numel() > 0 and not T.isnan(priorities).all():
        #     new_max = T.max(priorities)
        #     if new_max > self.max_priority:
        #         old_max = self.max_priority.item()
        #         self.max_priority = new_max
        #         if new_max > old_max * 10:  # Log significant jumps
        #             print(f"WARNING: Large max priority increase: {old_max:.2e} -> {new_max.item():.2e}")
        
        # Normalize priorities globally using max_recorded_priority
        # priorities = priorities / self.max_priority
        
        # Debug normalization
        # if T.max(priorities) > 1.0:
        #     print(f"WARNING: Post-normalization priorities > 1.0: max = {T.max(priorities).item():.2e}")
        
        # Compute tree indices (leaf nodes) from data indices
        # tree_indices = data_indices + self.capacity - 1
        
        # Update leaf nodes with new priorities
        # self.tree[tree_indices] = priorities
        
        # Update parent nodes efficiently without using Python sets/lists
        # Directly update all parents in a bottom-up fashion
        # parent_indices = (tree_indices - 1) // 2
        
        # Handle case where parent_indices might be empty
        # while parent_indices.numel() > 0:
        #     # For each parent, calculate the sum of its children
        #     left_children = 2 * parent_indices + 1
        #     right_children = left_children + 1
            
            # Update parents with sum of children
            # Handle edge cases where right child might not exist
            # right_valid = right_children < len(self.tree)
            # self.tree[parent_indices] = self.tree[left_children] + \
            #                            T.where(right_valid, self.tree[right_children], 
            #                                   T.zeros_like(self.tree[right_children]))
            
            # # Move up to next level of parents, removing duplicates
            # parent_indices = (parent_indices - 1) // 2
            
            # # Use unique values but handle potential empty tensor
            # if parent_indices.numel() > 0:
            #     parent_indices = T.unique(parent_indices)
            
            # # Stop when we reach the root
            # if parent_indices.numel() == 0 or (parent_indices < 0).all():
            #     break

    def update(self, data_indices, priorities):
        # Cap priorities to prevent extreme values
        priorities = T.clamp(priorities, min=1e-6)
        
        # Track maximum priority only once
        if priorities.numel() > 0:
            self.max_priority = T.max(T.cat([self.max_priority.unsqueeze(0), T.max(priorities).unsqueeze(0)]))
        
        # Compute tree indices once
        tree_indices = data_indices + self.capacity - 1
        
        # Update leaf nodes in one operation
        self.tree[tree_indices] = priorities
        
        # Pre-compute all parent indices at once instead of loop
        nodes_to_update = tree_indices
        while nodes_to_update.numel() > 0 and T.min(nodes_to_update) > 0:
            # Get parent indices directly without loop
            parent_indices = (nodes_to_update - 1) // 2
            unique_parents = T.unique(parent_indices)
            
            # Update all parents in parallel using vectorized operations
            for level_start in range(0, unique_parents.numel(), 1024):  # Process in chunks to avoid memory issues
                level_end = min(level_start + 1024, unique_parents.numel())
                current_parents = unique_parents[level_start:level_end]
                
                left_children = 2 * current_parents + 1
                right_children = left_children + 1
                
                # Create mask for valid right children
                valid_right = right_children < self.tree.size(0)
                
                # Get left and right values
                left_values = self.tree[left_children]
                right_values = T.zeros_like(left_values)
                right_values[valid_right] = self.tree[right_children[valid_right]]
                
                # Update parents in one operation
                self.tree[current_parents] = left_values + right_values
            
            nodes_to_update = unique_parents

    # def get(self, p_values: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
    #     """
    #     Optimized vectorized batch sampling from the SumTree.
    #     """
    #     batch_size = p_values.size(0)
        
    #     # Pre-allocate space for resulting indices
    #     tree_indices = T.zeros(batch_size, dtype=T.long, device=self.device)
        
    #     # Use an iterative approach for batch traversal
    #     # This is still sequential per sample but avoids Python loop overhead
    #     @T.jit.script
    #     def traverse_tree(p_values: T.Tensor, tree: T.Tensor, capacity: int) -> T.Tensor:
    #         batch_size = p_values.size(0)
    #         tree_indices = T.zeros(batch_size, dtype=T.long, device=p_values.device)
            
    #         for i in range(batch_size):
    #             idx = 0  # Start at root
    #             p = p_values[i].item()
                
    #             # Traverse down the tree
    #             while idx < capacity - 1:  # Not a leaf node
    #                 left = 2 * idx + 1
                    
    #                 # If we would access beyond tree bounds, we've reached a leaf
    #                 if left >= tree.size(0):
    #                     break
                    
    #                 left_val = tree[left].item()
                    
    #                 # Choose direction
    #                 if p <= left_val:
    #                     idx = left
    #                 else:
    #                     p -= left_val
    #                     idx = left + 1
                
    #             tree_indices[i] = idx
            
    #         return tree_indices
            
    #     # Traverse tree for each sample in batch
    #     tree_indices = traverse_tree(p_values, self.tree, self.capacity)
        
    #     # Convert tree indices to data indices and get priorities
    #     data_indices = tree_indices - (self.capacity - 1)
    #     priorities = self.tree[tree_indices]
        
    #     return data_indices, priorities

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

        #DEBUG
        shape = (buffer_size,) + self._obs_space_shape
        print(f"shape: {shape}")

        self.states = T.zeros(shape, dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, *self.env.single_action_space.shape), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size,), dtype=T.float32, device=self.device)
        self.next_states = T.zeros(shape, dtype=T.float32, device=self.device)
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
        beta_iter: int = 100_000,
        beta_update_freq: int = 10,
        priority: str = 'proportional',
        normalize: bool = False,
        goal_shape: Optional[Tuple[int]] = None,
        epsilon: float = 1e-6,
        device: Optional[str] = None
    ):
        if priority not in ['proportional', 'rank']:
            raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

        super().__init__(env, buffer_size, goal_shape, device)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_iter = beta_iter
        self.priority = priority
        self.normalize = normalize
        self.goal_shape = goal_shape
        self.epsilon = epsilon
        self.beta_update_freq = beta_update_freq
        self.beta = self.beta_start
        self._total_steps = 0
        
        # Add debug tracking
        self.debug_last_td_error = None
        self.debug_last_priority = None
        self.debug_last_indices = None

        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank-based
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices = None

        
        self.counter = 0

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
        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), 
                             T.arange(0, end_idx, device=self.device)])

        # Add to buffer tensors
        self.states[indices] = T.tensor(states, dtype=T.float32, device=self.device)
        self.actions[indices] = T.tensor(actions, dtype=T.float32, device=self.device)
        self.rewards[indices] = T.tensor(rewards, dtype=T.float32, device=self.device)
        self.next_states[indices] = T.tensor(next_states, dtype=T.float32, device=self.device)
        self.dones[indices] = T.tensor(dones, dtype=T.int8, device=self.device)

        if self.goal_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            self.state_achieved_goals[indices] = T.tensor(state_achieved_goals, dtype=T.float32, device=self.device)
            self.next_state_achieved_goals[indices] = T.tensor(next_state_achieved_goals, dtype=T.float32, device=self.device)
            self.desired_goals[indices] = T.tensor(desired_goals, dtype=T.float32, device=self.device)

        # Set initial priorities (will be normalized in update)
        if self.priority == "proportional":
            priorities = T.ones(len(indices), device=self.device) * self.sum_tree.max_priority
            self.sum_tree.update(indices, priorities)
        else:  # rank-based
            self.priorities[indices] = T.ones(len(indices), device=self.device)
            self.sorted_indices = None

        self.counter += batch_size
        self._total_steps += 1

    def update_beta(self) -> None:
        """Anneal beta param more efficiently"""
        progress = min(self._total_steps / self.beta_iter, 1.0)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)

    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor) -> None:
        """Updates priorities of sampled transitions"""
        if not isinstance(indices, T.Tensor):
            indices = T.tensor(indices, device=self.device)
        
        if not isinstance(priorities, T.Tensor):
            priorities = T.tensor(priorities, device=self.device)

        # Ensure absolute value
        priorities = T.abs(priorities)

        if self.priority == "proportional":
            # Z-score normalization for extreme values
            if priorities.numel() > 1 and self.normalize:
                mean = priorities.mean()
                std = priorities.std() + 1e-6  # Avoid division by zero
                normalized = (priorities - mean) / std
                priorities = T.clamp(normalized, -3.0, 3.0)  # Limit to 3 standard deviations
                # Convert back to positive range
                priorities = ((normalized + 3.0) / 6.0) + self.epsilon
            else:
                # Just apply clipping
                priorities = T.clamp(priorities, min=self.epsilon)

            priorities = priorities ** self.alpha

            # If NaN values, replace with mean of non-NaN values
            if T.isnan(priorities).any():
                nan_mask = T.isnan(priorities)
                mean_non_nan = priorities[~nan_mask].mean()
                priorities = T.where(nan_mask, mean_non_nan, priorities)

            #DEBUG
            # print(f"Priorities: {priorities}")
    
            # Apply alpha power and update the tree
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

    def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
        """Samples a batch of transitions based on priority - optimized version"""

        # Anneal beta
        if self._total_steps % self.beta_update_freq == 0:
            self.update_beta()
            
        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, size)

        # create these tensors if don't exist to avoid repeated creation
        if not hasattr(self, '_segment_boundaries'):
            self._segment_boundaries = T.zeros(batch_size, device=self.device)  # For most common batch sizes
            self._random_offsets = T.zeros(batch_size, device=self.device)
            self._weights_buffer = T.zeros(batch_size, device=self.device)
        
        if self.priority == "proportional":
            # Calculate segment boundaries
            # total_priority = self.sum_tree.tree[0].item() if self.sum_tree.tree.numel() > 0 else 0
            total_priority = self.sum_tree.total_priority
            
            if total_priority <= 0:
                # If tree has no meaningful priorities, fall back to uniform sampling
                indices = T.randint(0, size, (batch_size,), device=self.device)
                weights = T.ones(batch_size, device=self.device)
                probs = T.ones(batch_size, device=self.device) / size
            else:
                # Prepare segment boundaries
                segment_size = total_priority / batch_size
                self._segment_boundaries[:batch_size] = T.arange(0, batch_size, device=self.device) * segment_size
                
                # Generate random offsets with pre-allocated tensor
                self._random_offsets[:batch_size].uniform_(0, 1)
                self._random_offsets[:batch_size].mul_(segment_size)
                
                # Compute p_values reusing memory
                p_values = self._segment_boundaries[:batch_size] + self._random_offsets[:batch_size]
                
                # Get indices and priorities
                indices, priorities = self.sum_tree.get(p_values)
                
                # Fast priority to probability calculation
                probs = priorities / total_priority
                
                # Compute weights with vectorized operations
                self._weights_buffer[:batch_size] = (size * probs) ** (-self.beta)
                weights = self._weights_buffer[:batch_size] / self._weights_buffer[:batch_size].max()

                #DEBUG
                # print(f"Weights: {weights}")
                # print(f"Probs: {probs}")
                # print(f"Priorities: {priorities}")
                # print(f"Total priority: {total_priority}")
                # print(f"Segment size: {segment_size}")
                # print(f"Indices: {indices}")
                # print(f"P_values: {p_values}")

                
                
        else:  # rank-based
            # Prepare ranks for sampling
            self._prepare_rank_based()
            
            # Efficient inverse transform sampling
            u = T.rand(batch_size, device=self.device)
            rank_indices = (u ** (1 / self.alpha) * size).long().clamp(max=size-1)
            
            # Get actual indices from sorted indices
            indices = self.sorted_indices[rank_indices]
            
            # Calculate weights directly
            probs = 1 / ((rank_indices + 1).float() ** self.alpha)
            weights = (size * probs) ** (-self.beta)
            weights = weights / weights.max()
        
        return self._create_batch_outputs(indices, weights, probs)
    
    def _create_batch_outputs(self, indices, weights, probs):
        """Helper to avoid duplicating return logic"""
        # Use advanced indexing to fetch all tensors at once
        if self.goal_shape is not None:
            return (
                self.states.index_select(0, indices),
                self.actions.index_select(0, indices),
                self.rewards.index_select(0, indices),
                self.next_states.index_select(0, indices),
                self.dones.index_select(0, indices),
                self.state_achieved_goals.index_select(0, indices),
                self.next_state_achieved_goals.index_select(0, indices),
                self.desired_goals.index_select(0, indices),
                weights,
                probs,
                indices
            )
        else:
            return (
                self.states.index_select(0, indices),
                self.actions.index_select(0, indices),
                self.rewards.index_select(0, indices),
                self.next_states.index_select(0, indices), 
                self.dones.index_select(0, indices),
                weights,
                probs,
                indices
            )

    def get_config(self) -> Dict[str, Any]:
        """Get buffer config."""
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "alpha": self.alpha,
                "beta_start": self.beta_start,
                "beta_iter": self.beta_iter,
                "beta_update_freq": self.beta_update_freq,
                "priority": self.priority,
                "normalize": self.normalize,
                "goal_shape": self.goal_shape,
                "epsilon": self.epsilon,
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
            self.beta_iter,
            self.beta_update_freq,
            self.priority, 
            self.normalize,
            self.goal_shape,
            self.device.type, 
            self.epsilon,
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