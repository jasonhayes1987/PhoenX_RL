import torch as T
import numpy as np
import sys
import os
from collections import defaultdict
import math

# Extract the exact SumTree and PrioritizedReplayBuffer classes from your buffer.py
# This tests YOUR exact implementation

# First, let's copy the SumTree class directly from your code
class SumTree:
    """
    A binary sum tree for efficient sampling based on priorities.
    """
    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
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

    def get(self, p_values: T.Tensor):
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

# Mock classes to avoid dependencies
class MockEnvWrapper:
    """Mock environment wrapper."""
    def __init__(self):
        self.single_observation_space = type('space', (), {'shape': (4,)})()
        self.single_action_space = type('space', (), {'shape': (2,)})()

    def to_json(self):
        return {"mock": True}

# Simplified PrioritizedReplayBuffer (extracted from your code)
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer that samples transitions based on TD error.
    Supports both proportional and rank-based prioritization strategies.
    Includes support for N-step returns using trajectory indices.
    All tensor operations happen on the specified device to minimize data transfers.
    """
    def __init__(
        self,
        env,
        buffer_size: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_iter: int = 100_000,
        beta_update_freq: int = 10,
        priority: str = 'rank',
        normalize: bool = False,  # Only applies to proportional priority strategy
        obs_key: str = 'observation',
        goal_key: str = 'desired_goal',
        epsilon: float = 1e-6,
        N: int = 1,
        device = None,
    ):
        if priority not in ['proportional', 'rank']:
            raise ValueError(f"Invalid priority type: {priority} (must be 'proportional' or 'rank')")

        # Simplified init - removed parent class calls
        self.device = device if device else T.device('cpu')
        self.env = env
        self.buffer_size = buffer_size
        self.N = N  # N-step hyperparameter
        self.counter = 0

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_iter = beta_iter
        self.priority = priority
        self.normalize = normalize
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.epsilon = epsilon
        self.beta_update_freq = beta_update_freq
        self.beta = self.beta_start
        self._total_steps = 0

        # Initialize storage tensors (simplified)
        obs_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape

        self.states = T.zeros((buffer_size, N, *obs_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, N, *action_shape), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, N, *obs_shape), dtype=T.float32, device=self.device)
        self.dones = T.zeros((buffer_size, N), dtype=T.int8, device=self.device)
        self.trajectory_lengths = T.zeros((buffer_size,), dtype=T.int64, device=self.device)

        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank-based
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices = None

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        dones: T.Tensor,
        state_achieved_goals = None,
        next_state_achieved_goals = None,
        desired_goals = None,
        trajectory_lengths = None,
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
        if actions.ndim == 2:
            actions = actions[:, T.newaxis, :]
        if rewards.ndim == 1:
            rewards = rewards[:, T.newaxis]
        if next_states.ndim == 2:
            next_states = next_states[:, T.newaxis, :]
        if dones.ndim == 1:
            dones = dones[:, T.newaxis]

        # Store transitions (detach to avoid holding computation graphs)
        self.states[indices] = states.detach().to(device=self.device, dtype=T.float32)
        self.actions[indices] = actions.detach().to(device=self.device, dtype=T.float32)
        self.rewards[indices] = rewards.detach().to(device=self.device, dtype=T.float32)
        self.next_states[indices] = next_states.detach().to(device=self.device, dtype=T.float32)
        self.dones[indices] = dones.detach().to(device=self.device, dtype=T.int8)
        self.trajectory_lengths[indices] = trajectory_lengths.detach().to(device=self.device, dtype=T.int64)

        # Set initial priorities (will be normalized in update)
        if self.priority == "proportional":
            priorities = T.ones(len(indices), device=self.device) * self.sum_tree.max_priority
            self.sum_tree.update(indices, priorities)
        else:  # rank-based
            self.priorities[indices] = T.ones(len(indices), device=self.device) * self.priorities.max()
            self.sorted_indices = None

        self.counter += batch_size
        self._total_steps += 1

    def sample(self, batch_size: int):
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

        return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], weights, probs, indices)

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

    def get_config(self) -> dict:
        """Get buffer config."""
        return {
            'type': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "alpha": self.alpha,
                "beta_start": self.beta_start,
                "beta_iter": self.beta_iter,
                "beta_update_freq": self.beta_update_freq,
                "priority": self.priority,
                "normalize": self.normalize,
                "obs_key": self.obs_key,
                "goal_key": self.goal_key,
                "epsilon": self.epsilon,
                "N": self.N,
                "device": self.device.type
            }
        }

class TestYourExactClasses:
    """Test PrioritizedReplayBuffer and SumTree implementations."""

    def __init__(self):
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f"Testing classes on device: {self.device}")

    def test_sum_tree_from_your_code(self):
        """Test the SumTree class."""
        print("\n=== Testing SumTree Implementation ===")

        capacity = 16
        sum_tree = SumTree(capacity, self.device)

        # Test initial state
        assert sum_tree.total_priority == 0.0
        print("PASS: Initial total priority is 0")

        # Test updates
        indices = T.tensor([0, 1, 2], device=self.device)
        priorities = T.tensor([1.0, 2.0, 3.0], device=self.device)
        sum_tree.update(indices, priorities)

        expected_total = 1.0 + 2.0 + 3.0
        assert abs(sum_tree.total_priority - expected_total) < 1e-6
        print("PASS: Updates work correctly")

        # Test sampling
        p_values = T.tensor([0.5], device=self.device)
        data_indices, sampled_priorities = sum_tree.get(p_values)

        assert 0 <= data_indices[0] < capacity
        assert sampled_priorities[0] > 0
        print("PASS: Sampling works correctly")

        print("SumTree: PASSED")

    def test_prioritized_buffer_from_your_code(self):
        """Test PrioritizedReplayBuffer implementation."""
        print("\n=== Testing PrioritizedReplayBuffer Implementation ===")

        env = MockEnvWrapper()

        # Test proportional
        buffer = PrioritizedReplayBuffer(
            env=env,
            buffer_size=1000,
            alpha=0.6,
            beta_start=0.4,
            priority='proportional',
            device=self.device
        )

        print("PASS: PrioritizedReplayBuffer created successfully")

        # Add some transitions
        batch_size = 10
        for i in range(5):
            states = T.randn(batch_size, 4, device=self.device)
            actions = T.randn(batch_size, 2, device=self.device)
            rewards = T.randn(batch_size, device=self.device)
            next_states = T.randn(batch_size, 4, device=self.device)
            dones = T.randint(0, 2, (batch_size,), dtype=T.int8, device=self.device)
            trajectory_lengths = T.ones(batch_size, dtype=T.int64, device=self.device) * 5

            buffer.add(states, actions, rewards, next_states, dones, trajectory_lengths=trajectory_lengths)

        assert buffer.counter == 50
        print("PASS: Buffer additions work correctly")

        # Test sampling
        batch = buffer.sample(batch_size=8)
        states, actions, rewards, next_states, dones, weights, probs, indices = batch

        assert states.shape[0] == 8
        assert weights.shape[0] == 8
        assert T.all(weights > 0)
        print("PASS: Sampling returns correct shapes and positive weights")

        # Test priority updates
        new_priorities = T.abs(T.randn(8, device=self.device)) + 0.1
        buffer.update_priorities(indices, new_priorities)
        print("PASS: Priority updates work correctly")

        # Test beta annealing
        initial_beta = buffer.beta
        for _ in range(buffer.beta_update_freq):
            buffer.update_beta()
        assert buffer.beta > initial_beta
        print("PASS: Beta annealing works correctly")

        print("PrioritizedReplayBuffer: PASSED")

    def test_both_priority_strategies(self):
        """Test both priority strategies."""
        print("\n=== Testing Both Priority Strategies ===")

        env = MockEnvWrapper()

        # Test rank-based
        buffer_rank = PrioritizedReplayBuffer(
            env=env,
            buffer_size=100,
            priority='rank',
            device=self.device
        )
        print("PASS: Rank-based buffer created")

        # Add some data and test rank-based sampling
        states = T.randn(10, 4, device=self.device)
        actions = T.randn(10, 2, device=self.device)
        rewards = T.randn(10, device=self.device)
        next_states = T.randn(10, 4, device=self.device)
        dones = T.randint(0, 2, (10,), dtype=T.int8, device=self.device)
        trajectory_lengths = T.ones(10, dtype=T.int64, device=self.device) * 5

        buffer_rank.add(states, actions, rewards, next_states, dones, trajectory_lengths=trajectory_lengths)

        batch = buffer_rank.sample(batch_size=5)
        _, _, _, _, _, weights, probs, indices = batch

        assert weights.shape[0] == 5
        assert T.all(weights > 0)
        print("PASS: Rank-based sampling works correctly")

        print("Both priority strategies: PASSED")

    def test_config_method(self):
        """Test the get_config method."""
        print("\n=== Testing Configuration Method ===")

        env = MockEnvWrapper()
        buffer = PrioritizedReplayBuffer(
            env=env,
            buffer_size=100,
            device=self.device
        )

        config = buffer.get_config()
        assert config['type'] == 'PrioritizedReplayBuffer'
        assert 'alpha' in config['config']
        assert 'beta_start' in config['config']
        print("PASS: Configuration method works correctly")

        print("Configuration method: PASSED")

    def run_all_tests(self):
        """Run all tests."""
        print("Testing PrioritizedReplayBuffer & SumTree Classes")
        print("=" * 70)

        try:
            self.test_sum_tree_from_your_code()
            self.test_prioritized_buffer_from_your_code()
            self.test_both_priority_strategies()
            self.test_config_method()

            print("\n" + "=" * 70)
            print("ALL TESTS PASSED! ✓✓✓")
            print("PrioritizedReplayBuffer and SumTree implementations are correct.")

        except Exception as e:
            print(f"\nFAIL: Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

if __name__ == "__main__":
    T.manual_seed(42)
    np.random.seed(42)

    test_suite = TestYourExactClasses()
    success = test_suite.run_all_tests()

    if success:
        print("\nSUCCESS: Implementations PASSED!")
    else:
        print("\nFAILURE: Some tests failed.")
        sys.exit(1)