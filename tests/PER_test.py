import torch as T
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from buffer import PrioritizedReplayBuffer, SumTree



def test_prioritized_replay_buffer_with_sumtree():
    """
    Test the PrioritizedReplayBuffer's add and update_priorities methods
    to ensure proper SumTree integration.
    """
    # Function to print the SumTree state in a readable format
    def print_tree_state(tree, buffer_size):
        total_nodes = 2 * buffer_size - 1
        leaf_start = buffer_size - 1

        print("\nSumTree state:")
        print(f"Root value (total priority): {tree.tree[0].item():.4f}")

        print("\nLeaf nodes (data priorities):")
        for i in range(buffer_size):
            tree_idx = i + leaf_start
            print(f"Buffer index {i}: {tree.tree[tree_idx].item():.4f}")

        # Verify tree integrity
        leaf_sum = tree.tree[leaf_start:total_nodes].sum().item()
        print(f"\nSum of leaves: {leaf_sum:.4f}")
        print(f"Tree integrity check: {abs(tree.tree[0].item() - leaf_sum) < 1e-5}")

    # Create a proper mock environment for testing
    class MockEnvWrapper:
        def __init__(self):
            # Create observation and action spaces
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            # Add the single_observation_space attribute that ReplayBuffer expects
            self.single_observation_space = self.observation_space
            self.single_action_space = self.action_space

            # Add any other attributes that might be needed
            self.config = {}

        def to_json(self):
            return {}

    # Test parameters
    buffer_size = 16
    alpha = 0.6

    # 1. Create a PrioritizedReplayBuffer with the SumTree implementation
    print("Test 1: Buffer initialization")
    env = MockEnvWrapper()
    buffer = PrioritizedReplayBuffer(
        env=env,
        buffer_size=buffer_size,
        alpha=alpha,
        beta_start=0.4,
        priority='proportional',
        normalize=False,
        device="cpu"
    )

    # Verify SumTree was created correctly
    assert hasattr(buffer, 'sum_tree'), "SumTree not created"
    assert buffer.sum_tree.capacity == buffer_size, f"Wrong tree capacity: {buffer.sum_tree.capacity}"
    print("✓ Buffer initialized correctly")

    # 2. Test adding experiences and initial priorities
    print("\nTest 2: Adding experiences")
    # Create a batch of experiences
    batch_size = 4
    states = np.random.rand(batch_size, 4).astype(np.float32)
    actions = np.random.rand(batch_size, 2).astype(np.float32)
    rewards = np.random.rand(batch_size).astype(np.float32)
    next_states = np.random.rand(batch_size, 4).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.uint8)

    # Add experiences to buffer
    buffer.add(states, actions, rewards, next_states, dones)

    # Check current state of SumTree
    print_tree_state(buffer.sum_tree, buffer_size)

    # Verify all new experiences have the max priority
    for i in range(batch_size):
        tree_idx = i + buffer_size - 1
        assert T.isclose(buffer.sum_tree.tree[tree_idx], buffer.sum_tree.max_priority), \
            f"Entry {i} doesn't have max priority"

    # 3. Test updating priorities
    print("\nTest 3: Updating priorities with TD errors")
    # Update priorities for the first batch
    indices = T.tensor([0, 1, 2, 3])
    td_errors = T.tensor([0.2, 0.5, 1.0, 2.0])

    print(f"TD errors: {td_errors}")
    buffer.update_priorities(indices, td_errors)

    # Check updated priorities
    print("\nAfter update:")
    print_tree_state(buffer.sum_tree, buffer_size)

    # Verify priorities after alpha power transformation
    expected_priorities = td_errors ** buffer.alpha
    for i, idx in enumerate(indices):
        tree_idx = idx + buffer_size - 1
        actual = buffer.sum_tree.tree[tree_idx].item()
        expected = expected_priorities[i].item()

        assert abs(actual - expected) < 1e-5, \
            f"Priority mismatch at {idx}: expected {expected}, got {actual}"

    print("\nPriority transformation verification (alpha =", buffer.alpha, "):")
    print("Index | TD Error | Expected | Actual")
    for i, idx in enumerate(indices):
        tree_idx = idx + buffer_size - 1
        actual = buffer.sum_tree.tree[tree_idx].item()
        print(f"{idx.item()} | {td_errors[i].item():.4f} | {expected_priorities[i].item():.4f} | {actual:.4f}")

    # 4. Test sampling distribution
    print("\nTest 4: Sampling distribution")
    # Sample multiple times and count frequencies
    samples = 1000
    sample_counts = {i: 0 for i in range(batch_size)}

    for _ in range(samples):
        batch = buffer.sample(1)
        idx = batch[-1].item()  # Indices are the last element
        if idx < batch_size:
            sample_counts[idx] += 1

    # Calculate observed frequencies
    frequencies = {i: sample_counts[i] / samples for i in range(batch_size)}

    # Calculate expected probabilities
    total_priority = buffer.sum_tree.total_priority
    expected_probs = {}
    for i in range(batch_size):
        tree_idx = i + buffer_size - 1
        expected_probs[i] = buffer.sum_tree.tree[tree_idx].item() / total_priority

    print("Sampling results after 1000 samples:")
    print("Index | Priority | Expected Prob | Observed Freq")
    for i in range(batch_size):
        tree_idx = i + buffer_size - 1
        priority = buffer.sum_tree.tree[tree_idx].item()
        print(f"{i} | {priority:.4f} | {expected_probs[i]:.4f} | {frequencies[i]:.4f}")

    # 5. Test buffer wraparound
    print("\nTest 5: Buffer wraparound behavior")
    # Fill more than half the buffer to cause wraparound
    more_batches = 3  # This will take us to 16 entries, causing wraparound

    for b in range(more_batches):
        states = np.random.rand(batch_size, 4).astype(np.float32)
        actions = np.random.rand(batch_size, 2).astype(np.float32)
        rewards = np.random.rand(batch_size).astype(np.float32)
        next_states = np.random.rand(batch_size, 4).astype(np.float32)
        dones = np.zeros(batch_size, dtype=np.uint8)

        buffer.add(states, actions, rewards, next_states, dones)

    print(f"Buffer counter: {buffer.counter}")
    print(f"Buffer filled: {min(buffer.counter, buffer_size)}/{buffer_size}")

    # Update priorities for indices that wrapped around
    wrap_indices = T.tensor([0, 1, 4, 8])  # Some were overwritten, some weren't
    wrap_td_errors = T.tensor([3.0, 4.0, 5.0, 6.0])

    buffer.update_priorities(wrap_indices, wrap_td_errors)

    print("\nPriorities after wraparound update:")
    print_tree_state(buffer.sum_tree, buffer_size)

    # 6. Test normalization option
    print("\nTest 6: Priority normalization")
    # Create a new buffer with normalization enabled
    norm_buffer = PrioritizedReplayBuffer(
        env=env,
        buffer_size=buffer_size,
        alpha=alpha,
        beta_start=0.4,
        priority='proportional',
        normalize=True,  # Enable normalization
        device="cpu"
    )

    # Add experiences
    states = np.random.rand(batch_size, 4).astype(np.float32)
    actions = np.random.rand(batch_size, 2).astype(np.float32)
    rewards = np.random.rand(batch_size).astype(np.float32)
    next_states = np.random.rand(batch_size, 4).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.uint8)

    norm_buffer.add(states, actions, rewards, next_states, dones)

    # Update with extreme priority values
    extreme_indices = T.tensor([0, 1, 2, 3])
    extreme_td_errors = T.tensor([0.01, 1.0, 10.0, 100.0])  # Very different scales

    print("Extreme TD errors:", extreme_td_errors)
    norm_buffer.update_priorities(extreme_indices, extreme_td_errors)

    print("\nNormalized priorities:")
    print_tree_state(norm_buffer.sum_tree, buffer_size)

    print("\nAll tests completed!")

    return buffer  # Return for further inspection if needed


if __name__ == "__main__":
    # Run the test function
    test_prioritized_replay_buffer_with_sumtree()

