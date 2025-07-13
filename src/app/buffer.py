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
    
    def update(self, data_indices, priorities):
        # Cap priorities to prevent extreme values
        priorities = T.clamp(priorities, min=1e-6)

        # Track maximum priority
        if priorities.numel() > 0:
            self.max_priority = T.max(T.cat([self.max_priority.unsqueeze(0), T.max(priorities).unsqueeze(0)]))

        # Compute tree indices once
        tree_indices = data_indices + self.capacity - 1

        # Update leaf nodes in one operation
        self.tree[tree_indices] = priorities

        # Update parent nodes for each leaf individually - less vectorized but correct
        for idx in tree_indices:
            idx_item = idx.item()
            parent = (idx_item - 1) // 2

            # Traverse up to the root
            while parent >= 0:
                # Get children of this parent
                left = 2 * parent + 1
                right = 2 * parent + 2

                # Update the parent (handle case where right child might not exist)
                if right < self.tree.size(0):
                    self.tree[parent] = self.tree[left] + self.tree[right]
                else:
                    self.tree[parent] = self.tree[left]

                # Move to next parent up the tree
                parent = (parent - 1) // 2

    # def update(self, data_indices, priorities):
    #     # Cap priorities to prevent extreme values
    #     priorities = T.clamp(priorities, min=1e-6)
        
    #     # Track maximum priority only once
    #     if priorities.numel() > 0:
    #         self.max_priority = T.max(T.cat([self.max_priority.unsqueeze(0), T.max(priorities).unsqueeze(0)]))
        
    #     # Compute tree indices once
    #     tree_indices = data_indices + self.capacity - 1
        
    #     # Update leaf nodes in one operation
    #     self.tree[tree_indices] = priorities
        
    #     # Pre-compute all parent indices at once instead of loop
    #     nodes_to_update = tree_indices
    #     while nodes_to_update.numel() > 0 and T.min(nodes_to_update) > 0:
    #         # Get parent indices directly without loop
    #         parent_indices = (nodes_to_update - 1) // 2
    #         unique_parents = T.unique(parent_indices)
            
    #         # Update all parents in parallel using vectorized operations
    #         for level_start in range(0, unique_parents.numel(), 1024):  # Process in chunks to avoid memory issues
    #             level_end = min(level_start + 1024, unique_parents.numel())
    #             current_parents = unique_parents[level_start:level_end]
                
    #             left_children = 2 * current_parents + 1
    #             right_children = left_children + 1
                
    #             # Create mask for valid right children
    #             valid_right = right_children < self.tree.size(0)
                
    #             # Get left and right values
    #             left_values = self.tree[left_children]
    #             right_values = T.zeros_like(left_values)
    #             right_values[valid_right] = self.tree[right_children[valid_right]]
                
    #             # Update parents in one operation
    #             self.tree[current_parents] = left_values + right_values
            
    #         nodes_to_update = unique_parents

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

# class Buffer():
#     """
#     Base class for replay buffers.
#     """

#     def __init__(self, env: EnvWrapper, buffer_size: int, device: Optional[str] = None):
#         self.device = get_device(device)
#         self.env = env
#         self.buffer_size = buffer_size

#     def add(self, *args, **kwargs):
#         """
#         Add a transition to the buffer.
#         """
#         pass
    
#     def sample(self, batch_size: int):
#         """
#         Sample a batch of transitions from the buffer.

#         Args:
#             batch_size (int): The number of transitions to sample.

#         Returns:
#             Tuple: Sampled transitions.
#         """
#         pass

#     def get_config(self) -> Dict[str, Any]:
#         """
#         Retrieve the configuration of the buffer.

#         Returns:
#             dict: Configuration details.
#         """
#         pass

#     @classmethod
#     def create_instance(cls, buffer_class_name: str, **kwargs) -> 'Buffer':
#         """
#         Create an instance of the requested buffer class.

#         Args:
#             buffer_class_name (str): Name of the buffer class.
#             kwargs: Parameters for the buffer class.

#         Returns:
#             Buffer: An instance of the requested buffer class.

#         Raises:
#             ValueError: If the buffer class is not recognized.
#         """
#         buffer_classes = {
#             "ReplayBuffer": ReplayBuffer,
#             "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
#         }

#         if buffer_class_name in buffer_classes:
#             return buffer_classes[buffer_class_name](**kwargs)
#         else:
#             raise ValueError(f"{buffer_class_name} is not a subclass of Buffer")

class Buffer:
    """
    Base class for replay buffers with N-step functionality.
    """
    def __init__(self, env: EnvWrapper, buffer_size: int, N: int = 1, device: Optional[str] = None):
        self.device = get_device(device)
        self.env = env
        self.buffer_size = buffer_size
        self.N = N  # N-step hyperparameter
        self.counter = 0

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a transition to the buffer, including trajectory metadata.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    # def _get_sequence(self, start_idx: int) -> Tuple[List[T.Tensor], ...]:
    #     """Helper method to retrieve an N-step sequence starting from start_idx."""
    #     sequence_states, sequence_actions, sequence_rewards, sequence_next_states, sequence_dones = [], [], [], [], []
    #     if self.goal_shape is not None:
    #         sequence_sag, sequence_nsag, sequence_dg = [], [], []

    #     current_idx = start_idx
    #     traj_id = self.traj_ids[start_idx]
    #     start_step_idx = self.step_indices[start_idx]

    #     for i in range(self.N):
    #         #DEBUG
    #         # print(f'N:{i}')
    #         # print(f'current_idx: {current_idx}')
    #         # print(f'self.dones[current_idx]: {self.dones[current_idx]}')
    #         # print(f'self.traj_ids[current_idx]: {self.traj_ids[current_idx]}')
    #         # print(f'traj_id: {traj_id}')
    #         # print(f'self.step_indices[current_idx]: {self.step_indices[current_idx]}')
    #         # print(f'start_step_idx: {start_step_idx}')
    #         # print(f'i: {i}')
    #         if (current_idx >= self.buffer_size or 
    #             self.dones[current_idx] == 1 or 
    #             self.traj_ids[current_idx] != traj_id or 
    #             self.step_indices[current_idx] != start_step_idx + i):
    #             #DEBUG
    #             # print(f'BREAK')
    #             break

    #         sequence_states.append(self.states[current_idx])
    #         sequence_actions.append(self.actions[current_idx])
    #         sequence_rewards.append(self.rewards[current_idx])
    #         sequence_next_states.append(self.next_states[current_idx])
    #         sequence_dones.append(self.dones[current_idx])
    #         if self.goal_shape is not None:
    #             sequence_sag.append(self.state_achieved_goals[current_idx])
    #             sequence_nsag.append(self.next_state_achieved_goals[current_idx])
    #             sequence_dg.append(self.desired_goals[current_idx])

    #         current_idx = (current_idx + self.env.num_envs) % self.buffer_size

    #     # Pad sequences if shorter than N
    #     seq_len = len(sequence_states)
    #     if seq_len < self.N:
    #         pad_len = self.N - seq_len
    #         state_dim = self.states.shape[1:]
    #         action_dim = self.actions.shape[1:]
            
    #         sequence_rewards.extend([T.tensor(0.0, device=self.device)] * pad_len)
    #         sequence_dones.extend([T.tensor(1, dtype=T.int8, device=self.device)] * pad_len)
    #         sequence_states.extend([T.zeros(state_dim, device=self.device)] * pad_len)
    #         sequence_actions.extend([T.zeros(action_dim, device=self.device)] * pad_len)
    #         sequence_next_states.extend([T.zeros(state_dim, device=self.device)] * pad_len)

    #     #DEBUG
    #     # print(f'sequence_states: {sequence_states}')
    #     # print(f'sequence_actions: {sequence_actions}')
    #     # print(f'sequence_dones: {sequence_dones}')
    #     # print(f'sequence_rewards: {sequence_rewards}')
    #     # print(f'sequence_next_states: {sequence_next_states}')

    #     # Stack the sequences into tensors
    #     if self.goal_shape is not None:
    #         return (
    #             T.stack(sequence_states), T.stack(sequence_actions), T.stack(sequence_rewards),
    #             T.stack(sequence_next_states), T.stack(sequence_dones),
    #             T.stack(sequence_sag), T.stack(sequence_nsag), T.stack(sequence_dg)
    #         )
    #     return (
    #         T.stack(sequence_states), T.stack(sequence_actions), T.stack(sequence_rewards),
    #         T.stack(sequence_next_states), T.stack(sequence_dones)
    #     )

    def _get_sequence(self, start_idx: int) -> Tuple[List[T.Tensor], ...]:
        sequence_states, sequence_actions, sequence_rewards, sequence_next_states, sequence_dones, sequence_traj_ids, sequence_step_indices = [], [], [], [], [], [], []
        if self.goal_shape is not None:
            sequence_sag, sequence_nsag, sequence_dg = [], [], []

        traj_id = self.traj_ids[start_idx]
        start_step_idx = self.step_indices[start_idx]
        current_idx = start_idx

        for i in range(self.N):
            sequence_states.append(self.states[current_idx])
            sequence_actions.append(self.actions[current_idx])
            sequence_rewards.append(self.rewards[current_idx])
            sequence_next_states.append(self.next_states[current_idx])
            sequence_dones.append(self.dones[current_idx])
            sequence_traj_ids.append(self.traj_ids[current_idx])
            sequence_step_indices.append(self.step_indices[current_idx])
            
            if self.goal_shape is not None:
                sequence_sag.append(self.state_achieved_goals[current_idx])
                sequence_nsag.append(self.next_state_achieved_goals[current_idx])
                sequence_dg.append(self.desired_goals[current_idx])
            
            # if (current_idx >= self.buffer_size or 
            #     self.dones[current_idx] == 1 or 
            #     self.traj_ids[current_idx] != traj_id or 
            #     self.step_indices[current_idx] != start_step_idx + i):
            #     break
            
            # Stop if done or last step
            if self.dones[current_idx] == 1 or i == self.N - 1:
                break

            # Search for the next step
            mask = (self.traj_ids == traj_id) & (self.step_indices == abs(start_step_idx) + i + 1)
            idx = T.where(mask)[0]
            if len(idx) == 0:
                break
            current_idx = idx[0].item()

        seq_len = len(sequence_states)
        num_padded = 0
        if seq_len < self.N:
            num_padded += 1
            pad_len = self.N - seq_len
            state_dim = self.states.shape[1:]
            action_dim = self.actions.shape[1:]
            sequence_rewards.extend([T.tensor(0.0, device=self.device)] * pad_len)
            sequence_dones.extend([T.tensor(1, dtype=T.int8, device=self.device)] * pad_len)
            sequence_states.extend([T.zeros(state_dim, device=self.device)] * pad_len)
            sequence_actions.extend([T.zeros(action_dim, device=self.device)] * pad_len)
            sequence_next_states.extend([T.zeros(state_dim, device=self.device)] * pad_len)
            sequence_traj_ids.extend([T.tensor(traj_id, device=self.device)] * pad_len)
            sequence_step_indices.extend([T.tensor(start_step_idx + i + 1, device=self.device)] * pad_len)


        if self.goal_shape is not None:
            return (
                T.stack(sequence_states), T.stack(sequence_actions), T.stack(sequence_rewards),
                T.stack(sequence_next_states), T.stack(sequence_dones),
                T.stack(sequence_sag), T.stack(sequence_nsag), T.stack(sequence_dg),
                T.stack(sequence_traj_ids), T.stack(sequence_step_indices)
            )
        return (
            T.stack(sequence_states), T.stack(sequence_actions), T.stack(sequence_rewards),
            T.stack(sequence_next_states), T.stack(sequence_dones),
            T.stack(sequence_traj_ids), T.stack(sequence_step_indices)
        )

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def create_instance(cls, buffer_class_name: str, **kwargs) -> 'Buffer':
        buffer_classes = {
            "ReplayBuffer": ReplayBuffer,
            "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        }
        if buffer_class_name in buffer_classes:
            return buffer_classes[buffer_class_name](**kwargs)
        else:
            raise ValueError(f"{buffer_class_name} is not a subclass of Buffer")
        

# class ReplayBuffer(Buffer):
#     """
#     Replay buffer for storing transitions during reinforcement learning.

#     Attributes:
#         env (EnvWrapper): The environment wrapper associated with the buffer.
#         buffer_size (int): Maximum size of the buffer.
#         goal_shape (Optional[tuple]): Shape of goals (if used).
#         device (str): Device to store the buffer ('cpu' or 'cuda').
#     """
#     def __init__(
#         self,
#         env: EnvWrapper,
#         buffer_size: int = 100000,
#         goal_shape: Optional[Tuple[int]] = None,
#         device: Optional[str] = None,
#     ):
#         """
#         Initialize the ReplayBuffer.

#         Args:
#             env (EnvWrapper): The environment wrapper object.
#             buffer_size (int): Maximum size of the buffer.
#             goal_shape (Optional[tuple]): Shape of goals, if applicable.
#             device (Optional[str]): Device to store buffer data ('cpu' or 'cuda').
#         """
#         super().__init__(env, buffer_size, device)
#         self.goal_shape = goal_shape
        
#         # Determine observation space shape
#         if isinstance(self.env.single_observation_space, gym.spaces.Dict):
#             self._obs_space_shape = self.env.single_observation_space['observation'].shape
#         else:
#             self._obs_space_shape = self.env.single_observation_space.shape

#         #DEBUG
#         shape = (buffer_size,) + self._obs_space_shape

#         self.states = T.zeros(shape, dtype=T.float32, device=self.device)
#         self.actions = T.zeros((buffer_size, *self.env.single_action_space.shape), dtype=T.float32, device=self.device)
#         self.rewards = T.zeros((buffer_size,), dtype=T.float32, device=self.device)
#         self.next_states = T.zeros(shape, dtype=T.float32, device=self.device)
#         self.dones = T.zeros((buffer_size,), dtype=T.int8, device=self.device)
        
#         if self.goal_shape is not None:
#             self.desired_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
#             self.state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
#             self.next_state_achieved_goals = T.zeros((buffer_size, *self.goal_shape), dtype=T.float32, device=self.device)
        
#         self.counter = 0
#         self.gen = np.random.default_rng()

#     def reset(self) -> None:
#         """
#         Reset the buffer to all zeros and the counter to zero.
#         """
#         self.states.zero_()
#         self.actions.zero_()
#         self.rewards.zero_()
#         self.next_states.zero_()
#         self.dones.zero_()
#         self.counter = 0
        
#         if self.goal_shape is not None:
#             self.desired_goals.zero_()
#             self.state_achieved_goals.zero_()
#             self.next_state_achieved_goals.zero_()
        
        
#     def add(
#         self,
#         states: np.ndarray,
#         actions: np.ndarray,
#         rewards: float,
#         next_states: np.ndarray,
#         dones: bool,
#         state_achieved_goals: Optional[np.ndarray] = None,
#         next_state_achieved_goals: Optional[np.ndarray] = None,
#         desired_goals: Optional[np.ndarray] = None,
#     ) -> None:
#         """
#         Add a transition to the replay buffer.

#         Args:
#             states (np.ndarray): Current states.
#             actions (np.ndarray): Actions taken.
#             rewards (float): Rewards received.
#             next_states (np.ndarray): Next states.
#             dones (bool): Whether the episode is done.
#             state_achieved_goals (Optional[np.ndarray]): Achieved goals in the current state.
#             next_state_achieved_goals (Optional[np.ndarray]): Achieved goals in the next state.
#             desired_goals (Optional[np.ndarray]): Desired goals.
#         """
#         batch_size = len(states)
#         start_idx = self.counter % self.buffer_size
#         end_idx = (self.counter + batch_size) % self.buffer_size

#         # Compute indices with wrapping
#         if end_idx > start_idx:
#             indices = np.arange(start_idx, end_idx)
#         else:
#             indices = np.concatenate([np.arange(start_idx, self.buffer_size), np.arange(0, end_idx)])

#         # Convert lists to numpy arrays and then to tensors in one operation
#         self.states[indices] = T.tensor(np.array(states), dtype=T.float32, device=self.device)
#         self.actions[indices] = T.tensor(np.array(actions), dtype=T.float32, device=self.device)
#         self.rewards[indices] = T.tensor(np.array(rewards), dtype=T.float32, device=self.device)
#         self.next_states[indices] = T.tensor(np.array(next_states), dtype=T.float32, device=self.device)
#         self.dones[indices] = T.tensor(np.array(dones), dtype=T.int8, device=self.device)

#         if self.goal_shape is not None:
#             if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
#                 raise ValueError("Goal data must be provided when using goals")
#             self.state_achieved_goals[indices] = T.tensor(np.array(state_achieved_goals), dtype=T.float32, device=self.device)
#             self.next_state_achieved_goals[indices] = T.tensor(np.array(next_state_achieved_goals), dtype=T.float32, device=self.device)
#             self.desired_goals[indices] = T.tensor(np.array(desired_goals), dtype=T.float32, device=self.device)

#         self.counter += batch_size
        
#     def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
#         """
#         Sample a batch of transitions from the replay buffer.

#         Args:
#             batch_size (int): Number of transitions to sample.

#         Returns:
#             Tuple[T.Tensor, ...]: Sampled transitions.
#         """
#         size = min(self.counter, self.buffer_size)
#         if size == 0:
#             raise ValueError("Cannot sample from empty buffer")
        
#         indices = self.gen.integers(0, size, (batch_size,))
        
#         if self.goal_shape is not None:
#             return (
#                 self.states[indices],
#                 self.actions[indices],
#                 self.rewards[indices],
#                 self.next_states[indices],
#                 self.dones[indices],
#                 self.state_achieved_goals[indices],
#                 self.next_state_achieved_goals[indices],
#                 self.desired_goals[indices],
#             )
#         else:
#             return (
#                 self.states[indices],
#                 self.actions[indices],
#                 self.rewards[indices],
#                 self.next_states[indices],
#                 self.dones[indices]
#             )
    
#     def get_config(self) -> Dict[str, Any]:
#         """
#         Retrieve the configuration of the replay buffer.

#         Returns:
#             Dict[str, Any]: Configuration details.
#         """
#         return {
#             'class_name': self.__class__.__name__,
#             'config': {
#                 "env": self.env.to_json(),
#                 "buffer_size": self.buffer_size,
#                 "goal_shape": self.goal_shape,
#                 "device": self.device.type,
#             }
#         }
    
#     def clone(self, device: Optional[str] = None) -> 'ReplayBuffer':
#         """
#         Clone the replay buffer.

#         Returns:
#             ReplayBuffer: A new instance of the replay buffer with the same configuration.
#         """
#         if device:
#             device = get_device(device)
#         else:
#             device = self.device

#         env = build_env_wrapper_obj(self.env.config)
#         return ReplayBuffer(env, self.buffer_size, self.goal_shape, device)
    
class ReplayBuffer(Buffer):
    """
    Replay buffer with N-step sequence sampling.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100000,
        goal_shape: Optional[Tuple[int]] = None,
        N: int = 1,
        device: Optional[str] = None,
    ):
        super().__init__(env, buffer_size, N, device)
        self.goal_shape = goal_shape
        
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            self._obs_space_shape = self.env.single_observation_space['observation'].shape
        else:
            self._obs_space_shape = self.env.single_observation_space.shape

        self.states = T.zeros((buffer_size, N, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, N, *self.env.single_action_space.shape), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, N, *self._obs_space_shape), dtype=T.float32, device=self.device)
        self.dones = T.zeros((buffer_size, N), dtype=T.int8, device=self.device)
        
        if self.goal_shape is not None:
            self.desired_goals = T.zeros((buffer_size, N, *self.goal_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, N, *self.goal_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, N, *self.goal_shape), dtype=T.float32, device=self.device)
        
        # self.counter = 0
        self.gen = np.random.default_rng()

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        state_achieved_goals: Optional[np.ndarray] = None,
        next_state_achieved_goals: Optional[np.ndarray] = None,
        desired_goals: Optional[np.ndarray] = None,
    ) -> None:
        batch_size = len(states)
        start_idx = self.counter % self.buffer_size
        end_idx = (self.counter + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = np.arange(start_idx, end_idx)
        else:
            indices = np.concatenate([np.arange(start_idx, self.buffer_size), np.arange(0, end_idx)])

        #DEBUG
        # print(f'states: {states}')
        # print(f'actions: {actions}')
        # print(f'rewards: {rewards}')
        # print(f'next_states: {next_states}')
        # print(f'dones: {dones}')

        # Add N dimension of 1 at index 1 if values are 2d
        if states.ndim == 2:
            states = states[:, np.newaxis, :]
            # states = states.unsqueeze(1)
        if actions.ndim == 2:
            actions = actions[:, np.newaxis, :]
            # actions = actions.unsqueeze(1)
        if rewards.ndim == 1:
            rewards = rewards[:, np.newaxis]
            # rewards = rewards.unsqueeze(1)
        if next_states.ndim == 2:
            next_states = next_states[:, np.newaxis, :]
            # next_states = next_states.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones[:, np.newaxis]
            # dones = dones.unsqueeze(1)

        if self.goal_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            if state_achieved_goals.ndim == 2:
                state_achieved_goals = state_achieved_goals[:, np.newaxis, :]
                # state_achieved_goals = state_achieved_goals.unsqueeze(1)
            if next_state_achieved_goals.ndim == 2:
                next_state_achieved_goals = next_state_achieved_goals[:, np.newaxis, :]
                # next_state_achieved_goals = next_state_achieved_goals.unsqueeze(1)
            if desired_goals.ndim == 2:
                desired_goals = desired_goals[:, np.newaxis, :]
                # desired_goals = desired_goals.unsqueeze(1)

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

        self.counter += batch_size

    def sample(self, batch_size: int) -> List[Tuple[T.Tensor, ...]]:
        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # indices = T.randint(0, size, (batch_size,), device=self.device)
        indices = self.gen.integers(0, size, (batch_size,))
        # Retrieve N-step sequences for each sampled starting index
        # sequences = [self._get_sequence(idx.item()) for idx in indices]
        # sorted_sequences = zip(*sequences) # zips metrics together
        # sequence_stack = [T.stack(seq, dim=0) for seq in sorted_sequences]
        # return sequence_stack
        if self.goal_shape is not None:
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], self.state_achieved_goals[indices], self.next_state_achieved_goals[indices], self.desired_goals[indices])
        else:
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices])
    
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

    def clone(self, device: Optional[str] = None) -> 'ReplayBuffer':
        """
        Clone the replay buffer.

        Returns:
            ReplayBuffer: A new instance of the replay buffer with the same configuration.
        """
        if device:
            device = get_device(device)
        else:
            device = self.device

        env = build_env_wrapper_obj(self.env.config)
        return ReplayBuffer(env, self.buffer_size, self.goal_shape, device)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "goal_shape": self.goal_shape,
                "N": self.N,
                "device": self.device.type
            }
        }

# class PrioritizedReplayBuffer(ReplayBuffer):
#     """
#     Prioritized Experience Replay buffer that samples transitions based on TD error.
#     Supports both proportional and rank-based prioritization strategies.
#     All tensor operations happen on the specified device to minimize data transfers.
#     """
#     def __init__(
#         self,
#         env: EnvWrapper,
#         buffer_size: int = 100_000,
#         alpha: float = 0.6,
#         beta_start: float = 0.4,
#         beta_iter: int = 100_000,
#         beta_update_freq: int = 10,
#         priority: str = 'rank',
#         normalize: bool = False, # Only applies to proportional priority strategy
#         goal_shape: Optional[Tuple[int]] = None,
#         epsilon: float = 1e-6,
#         device: Optional[str] = None
#     ):
#         if priority not in ['proportional', 'rank']:
#             raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

#         super().__init__(env, buffer_size, goal_shape, device)
#         self.alpha = alpha
#         self.beta_start = beta_start
#         self.beta_iter = beta_iter
#         self.priority = priority
#         self.normalize = normalize
#         self.goal_shape = goal_shape
#         self.epsilon = epsilon
#         self.beta_update_freq = beta_update_freq
#         self.beta = self.beta_start
#         self._total_steps = 0

#         if self.priority == "proportional":
#             self.sum_tree = SumTree(buffer_size, self.device)
#         else:  # rank-based
#             self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
#             self.sorted_indices = None

        
#         self.counter = 0

#     def add(
#         self,
#         states: np.ndarray,
#         actions: np.ndarray,
#         rewards: np.ndarray,
#         next_states: np.ndarray,
#         dones: np.ndarray,
#         state_achieved_goals: Optional[np.ndarray] = None,
#         next_state_achieved_goals: Optional[np.ndarray] = None,
#         desired_goals: Optional[np.ndarray] = None,
#     ) -> None:
#         batch_size = len(states)
#         start_idx = self.counter % self.buffer_size
#         end_idx = (self.counter + batch_size) % self.buffer_size

#         if end_idx > start_idx:
#             indices = T.arange(start_idx, end_idx, device=self.device)
#         else:
#             indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), 
#                              T.arange(0, end_idx, device=self.device)])

#         # Add to buffer tensors
#         self.states[indices] = T.tensor(states, dtype=T.float32, device=self.device)
#         self.actions[indices] = T.tensor(actions, dtype=T.float32, device=self.device)
#         self.rewards[indices] = T.tensor(rewards, dtype=T.float32, device=self.device)
#         self.next_states[indices] = T.tensor(next_states, dtype=T.float32, device=self.device)
#         self.dones[indices] = T.tensor(dones, dtype=T.int8, device=self.device)

#         if self.goal_shape is not None:
#             if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
#                 raise ValueError("Goal data must be provided when using goals")
#             self.state_achieved_goals[indices] = T.tensor(state_achieved_goals, dtype=T.float32, device=self.device)
#             self.next_state_achieved_goals[indices] = T.tensor(next_state_achieved_goals, dtype=T.float32, device=self.device)
#             self.desired_goals[indices] = T.tensor(desired_goals, dtype=T.float32, device=self.device)

#         # Set initial priorities (will be normalized in update)
#         if self.priority == "proportional":
#             priorities = T.ones(len(indices), device=self.device) * self.sum_tree.max_priority
#             self.sum_tree.update(indices, priorities)
#         else:  # rank-based
#             self.priorities[indices] = T.ones(len(indices), device=self.device) * self.priorities.max()
#             self.sorted_indices = None

#         self.counter += batch_size
#         self._total_steps += 1

#     def update_beta(self) -> None:
#         """Anneal beta param more efficiently"""
#         progress = min(self._total_steps / self.beta_iter, 1.0)
#         self.beta = self.beta_start + progress * (1.0 - self.beta_start)

#     def update_priorities(self, indices: T.Tensor, priorities: T.Tensor) -> None:
#         """Updates priorities of sampled transitions"""
#         if not isinstance(indices, T.Tensor):
#             indices = T.tensor(indices, device=self.device)
        
#         if not isinstance(priorities, T.Tensor):
#             priorities = T.tensor(priorities, device=self.device)

#         # Ensure absolute value
#         priorities = T.abs(priorities)

#         if self.priority == "proportional":
#             # Z-score normalization for extreme values
#             if priorities.numel() > 1 and self.normalize:
#                 mean = priorities.mean()
#                 std = priorities.std() + 1e-6  # Avoid division by zero
#                 normalized = (priorities - mean) / std
#                 priorities = T.clamp(normalized, -3.0, 3.0)  # Limit to 3 standard deviations
#                 # Convert back to positive range
#                 priorities = ((normalized + 3.0) / 6.0) + self.epsilon
#             else:
#                 # Just apply clipping
#                 priorities = T.clamp(priorities, min=self.epsilon)

#             priorities = priorities ** self.alpha

#             # If NaN values, replace with mean of non-NaN values
#             if T.isnan(priorities).any():
#                 nan_mask = T.isnan(priorities)
#                 mean_non_nan = priorities[~nan_mask].mean()
#                 priorities = T.where(nan_mask, mean_non_nan, priorities)

#             #DEBUG
#             # print(f"Priorities: {priorities}")
    
#             # Apply alpha power and update the tree
#             self.sum_tree.update(indices, priorities)
        
#         else:  # rank-based
#             self.priorities[indices] = priorities
#             self.sorted_indices = None

#     def _prepare_rank_based(self) -> None:
#         """Sorts priorities for rank-based sampling"""

#         if self.sorted_indices is None:
#             size = min(self.counter, self.buffer_size)
#             if size > 0:
#                 self.sorted_indices = T.argsort(self.priorities[:size], descending=True)
#             else:
#                 self.sorted_indices = T.tensor([], dtype=T.long, device=self.device)

#     def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
#         """Samples a batch of transitions based on priority - optimized version"""

#         # Anneal beta
#         if self._total_steps % self.beta_update_freq == 0:
#             self.update_beta()

#         size = min(self.counter, self.buffer_size)
#         if size == 0:
#             raise ValueError("Cannot sample from empty buffer")

#         batch_size = min(batch_size, size)

#         if self.priority == "proportional":
#             # Instantiate tensors if don't exist to avoid repeated creation
#             if not hasattr(self, '_segment_boundaries'):
#                 self._segment_boundaries = T.zeros(batch_size, device=self.device)  # For most common batch sizes
#                 self._random_offsets = T.zeros(batch_size, device=self.device)
#                 self._weights_buffer = T.zeros(batch_size, device=self.device)

#             total_priority = self.sum_tree.total_priority

#             if total_priority <= 0:
#                 # If tree has no meaningful priorities, fall back to uniform sampling
#                 indices = T.randint(0, size, (batch_size,), device=self.device)
#                 weights = T.ones(batch_size, device=self.device)
#                 probs = T.ones(batch_size, device=self.device) / size
#             else:
#                 # Prepare segment boundaries
#                 segment_size = total_priority / batch_size
#                 self._segment_boundaries[:batch_size] = T.arange(0, batch_size, device=self.device) * segment_size

#                 # Generate random offsets with pre-allocated tensor
#                 self._random_offsets[:batch_size].uniform_(0, 1)
#                 self._random_offsets[:batch_size].mul_(segment_size)

#                 # Compute p_values reusing memory
#                 p_values = self._segment_boundaries[:batch_size] + self._random_offsets[:batch_size]

#                 # Get indices and priorities
#                 indices, priorities = self.sum_tree.get(p_values)

#                 # Fast priority to probability calculation
#                 probs = priorities / total_priority

#                 # Compute weights with vectorized operations
#                 self._weights_buffer[:batch_size] = (size * probs) ** (-self.beta)
#                 weights = self._weights_buffer[:batch_size] / self._weights_buffer[:batch_size].max()

#         else:  # rank-based
#             # Prepare ranks for sampling
#             self._prepare_rank_based()

#             # Inverse transform sampling
#             u = T.rand(batch_size, device=self.device)
#             ranks = (u ** (1 / self.alpha) * size).long().clamp(max=size-1)

#             # Get actual indices from sorted indices
#             indices = self.sorted_indices[ranks]

#             # Calculate weights directly
#             cur_probs = 1 / ((ranks + 1) ** self.alpha)
#             all_ranks = T.arange(size, device=self.device)
#             sum_probs = T.sum(1 / (all_ranks + 1.0) ** self.alpha)
#             probs = cur_probs / sum_probs
#             size = min(self.counter, self.buffer_size)
#             weights = (size * probs) ** (-self.beta)
#             weights = weights / weights.max()

#         return self._create_batch_outputs(indices, weights, probs)
    
#     def _create_batch_outputs(self, indices, weights, probs):
#         """Helper to avoid duplicating return logic"""
#         # Use advanced indexing to fetch all tensors at once
#         if self.goal_shape is not None:
#             return (
#                 self.states.index_select(0, indices),
#                 self.actions.index_select(0, indices),
#                 self.rewards.index_select(0, indices),
#                 self.next_states.index_select(0, indices),
#                 self.dones.index_select(0, indices),
#                 self.state_achieved_goals.index_select(0, indices),
#                 self.next_state_achieved_goals.index_select(0, indices),
#                 self.desired_goals.index_select(0, indices),
#                 weights,
#                 probs,
#                 indices
#             )
#         else:
#             return (
#                 self.states.index_select(0, indices),
#                 self.actions.index_select(0, indices),
#                 self.rewards.index_select(0, indices),
#                 self.next_states.index_select(0, indices), 
#                 self.dones.index_select(0, indices),
#                 weights,
#                 probs,
#                 indices
#             )

#     def get_config(self) -> Dict[str, Any]:
#         """Get buffer config."""
#         return {
#             'class_name': self.__class__.__name__,
#             'config': {
#                 "env": self.env.to_json(),
#                 "buffer_size": self.buffer_size,
#                 "alpha": self.alpha,
#                 "beta_start": self.beta_start,
#                 "beta_iter": self.beta_iter,
#                 "beta_update_freq": self.beta_update_freq,
#                 "priority": self.priority,
#                 "normalize": self.normalize,
#                 "goal_shape": self.goal_shape,
#                 "epsilon": self.epsilon,
#                 "device": self.device.type
#             }
#         }
    
#     def clone(self, device: Optional[str] = None) -> 'PrioritizedReplayBuffer':
#         """Create a new instance with the same configuration."""
#         # Set device if not None
#         if device:
#             device = get_device(device)
#         else:
#             device = self.device.type

#         env = build_env_wrapper_obj(self.env.config)
#         return PrioritizedReplayBuffer(
#             env, 
#             self.buffer_size, 
#             self.alpha, 
#             self.beta_start, 
#             self.beta_iter,
#             self.beta_update_freq,
#             self.priority, 
#             self.normalize,
#             self.goal_shape, 
#             self.epsilon,
#             device
#         )

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer that samples transitions based on TD error.
    Supports both proportional and rank-based prioritization strategies.
    Includes support for N-step returns using trajectory indices.
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
        priority: str = 'rank',
        normalize: bool = False,  # Only applies to proportional priority strategy
        goal_shape: Optional[Tuple[int]] = None,
        epsilon: float = 1e-6,
        N: int = 1,
        device: Optional[str] = None,
    ):
        if priority not in ['proportional', 'rank']:
            raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

        super().__init__(env, buffer_size, goal_shape, N, device)
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

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
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

        # Add N dimension of 1 at index 1 if values are 2d
        if states.ndim == 2:
            states = states[:, np.newaxis, :]
            # states = states.unsqueeze(1)
        if actions.ndim == 2:
            actions = actions[:, np.newaxis, :]
            # actions = actions.unsqueeze(1)
        if rewards.ndim == 1:
            rewards = rewards[:, np.newaxis]
            # rewards = rewards.unsqueeze(1)
        if next_states.ndim == 2:
            next_states = next_states[:, np.newaxis, :]
            # next_states = next_states.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones[:, np.newaxis]
            # dones = dones.unsqueeze(1)

        if self.goal_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            if state_achieved_goals.ndim == 2:
                state_achieved_goals = state_achieved_goals[:, np.newaxis, :]
                # state_achieved_goals = state_achieved_goals.unsqueeze(1)
            if next_state_achieved_goals.ndim == 2:
                next_state_achieved_goals = next_state_achieved_goals[:, np.newaxis, :]
                # next_state_achieved_goals = next_state_achieved_goals.unsqueeze(1)
            if desired_goals.ndim == 2:
                desired_goals = desired_goals[:, np.newaxis, :]
                # desired_goals = desired_goals.unsqueeze(1)

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
            self.priorities[indices] = T.ones(len(indices), device=self.device) * self.priorities.max()
            self.sorted_indices = None

        self.counter += batch_size
        self._total_steps += 1


    def sample(self, batch_size: int) -> Tuple[List[Tuple[T.Tensor, ...]], T.Tensor, T.Tensor, T.Tensor]:
        """Samples a batch of N-step transition sequences based on priority."""
        if self._total_steps % self.beta_update_freq == 0:
            self.update_beta()

        size = min(self.counter, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, size)

        if self.priority == "proportional":
            if not hasattr(self, '_segment_boundaries'):
                self._segment_boundaries = T.zeros(batch_size, device=self.device)
                self._random_offsets = T.zeros(batch_size, device=self.device)
                self._weights_buffer = T.zeros(batch_size, device=self.device)

            total_priority = self.sum_tree.total_priority
            if total_priority <= 0:
                indices = T.randint(0, size, (batch_size,), device=self.device)
                weights = T.ones(batch_size, device=self.device)
                probs = T.ones(batch_size, device=self.device) / size
            else:
                segment_size = total_priority / batch_size
                self._segment_boundaries[:batch_size] = T.arange(0, batch_size, device=self.device) * segment_size
                self._random_offsets[:batch_size].uniform_(0, 1)
                self._random_offsets[:batch_size].mul_(segment_size)
                p_values = self._segment_boundaries[:batch_size] + self._random_offsets[:batch_size]
                indices, priorities = self.sum_tree.get(p_values)
                probs = priorities / total_priority
                self._weights_buffer[:batch_size] = (size * probs) ** (-self.beta)
                weights = self._weights_buffer[:batch_size] / self._weights_buffer[:batch_size].max()
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

        # Retrieve N-step sequences for each sampled starting index
        # sequences = [self._get_sequence(idx.item()) for idx in indices]
        # sorted_sequences = zip(*sequences) # zips metrics together
        # sequence_stack = [T.stack(seq, dim=0) for seq in sorted_sequences]
        # return sequence_stack, weights, probs, indices

        if self.goal_shape is not None: 
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], self.state_achieved_goals[indices], self.next_state_achieved_goals[indices], self.desired_goals[indices], weights, probs, indices)
        else:
            return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], weights, probs, indices)

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
                "N": self.N,
                "device": self.device.type
            }
        }
    
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
            self.goal_shape, 
            self.epsilon,
            device,
            self.N
        )
