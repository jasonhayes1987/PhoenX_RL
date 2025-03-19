import torch as T
import numpy as np
import matplotlib.pyplot as plt
from buffer import PrioritizedReplayBuffer, SumTree
from env_wrapper import GymnasiumWrapper
import gymnasium as gym
import os
import wandb
from collections import defaultdict

def log_priority_metrics(buffer, step, prefix=''):
    """Log priority-related metrics to wandb"""
    if buffer.priority == 'proportional':
        # Get all priorities from sum tree
        priorities = buffer.sum_tree.tree[buffer.sum_tree.capacity-1:].cpu().numpy()
        valid_priorities = priorities[priorities > 0]  # Filter out zero priorities
        
        metrics = {
            f'{prefix}priority_mean': np.mean(valid_priorities),
            f'{prefix}priority_median': np.median(valid_priorities),
            f'{prefix}priority_std': np.std(valid_priorities),
            f'{prefix}priority_max': float(buffer.sum_tree.max_priority),
            f'{prefix}total_priority': float(buffer.sum_tree.total_priority),
            f'{prefix}beta': float(buffer.beta)
        }
    else:  # rank-based
        priorities = buffer.priorities[:min(buffer.counter, buffer.buffer_size)].cpu().numpy()
        metrics = {
            f'{prefix}priority_mean': np.mean(priorities),
            f'{prefix}priority_median': np.median(priorities),
            f'{prefix}priority_std': np.std(priorities),
            f'{prefix}beta': float(buffer.beta)
        }
    
    wandb.log(metrics, step=step)

def test_sum_tree():
    """Test SumTree functionality"""
    print("\nTesting SumTree...")
    
    # Initialize wandb
    wandb.init(project="per_test", name="sum_tree_test")
    
    # Initialize SumTree
    capacity = 8
    tree = SumTree(capacity, T.device('cpu'))
    
    # Test 1: Basic priority updates
    print("\nTest 1: Basic priority updates")
    indices = T.tensor([0, 1, 2])
    priorities = T.tensor([1.0, 2.0, 3.0])
    tree.update(indices, priorities)
    total = tree.total_priority
    print(f"Total priority after update: {total} (should be 6.0)")
    print(f"Tree structure:\n{tree.tree}")
    
    # Log metrics
    wandb.log({
        'sum_tree/total_priority': total,
        'sum_tree/max_priority': float(tree.max_priority),
        'sum_tree/tree_values': wandb.Histogram(tree.tree.cpu().numpy())
    })
    
    # Test 2: Sampling
    print("\nTest 2: Sampling")
    p_values = T.tensor([1.5, 4.5])  # Should sample indices 1 and 2
    indices, priorities = tree.get(p_values)
    print(f"Sampled indices: {indices}")
    print(f"Sampled priorities: {priorities}")
    
    # Log sampling results
    wandb.log({
        'sum_tree/sampled_indices': wandb.Histogram(indices.cpu().numpy()),
        'sum_tree/sampled_priorities': wandb.Histogram(priorities.cpu().numpy())
    })
    
    # Test 3: Max priority update
    print("\nTest 3: Max priority tracking")
    print(f"Current max priority: {tree.max_priority}")
    tree.update(T.tensor([3]), T.tensor([5.0]))
    print(f"Max priority after update: {tree.max_priority}")
    
    wandb.log({
        'sum_tree/max_priority_after_update': float(tree.max_priority)
    })
    
    wandb.finish()

def test_prioritized_buffer():
    """Test PrioritizedReplayBuffer functionality"""
    print("\nTesting PrioritizedReplayBuffer...")
    
    # Initialize wandb
    wandb.init(project="per_test", name="prioritized_buffer_test")
    
    # Create environment and buffer
    env = gym.make('Pendulum-v1')
    env_spec = gym.spec('Pendulum-v1')
    env_wrapper = GymnasiumWrapper(env_spec)
    buffer_size = 1000
    buffer = PrioritizedReplayBuffer(
        env=env_wrapper,
        buffer_size=buffer_size,
        alpha=0.6,
        beta_start=0.4,
        beta_iter=10000,
        priority='proportional'
    )
    
    # Test 1: Adding transitions
    print("\nTest 1: Adding transitions")
    state = env.reset()[0]
    priorities = []
    sampled_priorities = defaultdict(list)
    
    for i in range(100):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Add transition to buffer
        buffer.add(
            states=np.array([state]),
            actions=np.array([action]),
            rewards=np.array([reward]),
            next_states=np.array([next_state]),
            dones=np.array([done])
        )
        
        if done:
            state = env.reset()[0]
        else:
            state = next_state
            
        # Record current max priority
        if buffer.priority == 'proportional':
            priorities.append(float(buffer.sum_tree.max_priority))
        
        # Log metrics every 10 steps
        if i % 10 == 0:
            log_priority_metrics(buffer, i, prefix='buffer/')
    
    print(f"Buffer size after adding: {min(buffer.counter, buffer.buffer_size)}")
    
    # Test 2: Sampling with importance sampling
    print("\nTest 2: Sampling with importance sampling")
    batch_size = 32
    num_samples = 1000  # Number of samples to collect for distribution analysis
    
    # Collect samples and their priorities
    for _ in range(num_samples):
        samples = buffer.sample(batch_size)
        if buffer.goal_shape is not None:
            states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, weights, indices = samples
        else:
            states, actions, rewards, next_states, dones, weights, indices = samples
            
        # Get priorities for sampled indices
        if buffer.priority == 'proportional':
            sampled_priorities['proportional'].extend(buffer.sum_tree.tree[buffer.sum_tree.capacity-1+indices].cpu().numpy())
        else:
            sampled_priorities['rank'].extend(buffer.priorities[indices].cpu().numpy())
    
    # Log sampling statistics
    wandb.log({
        'sampling/sampled_priorities_dist': wandb.Histogram(sampled_priorities['proportional' if buffer.priority == 'proportional' else 'rank']),
        'sampling/weights_dist': wandb.Histogram(weights.cpu().numpy()),
        'sampling/weights_mean': float(weights.mean()),
        'sampling/weights_std': float(weights.std())
    })
    
    # Test 3: Priority updates
    print("\nTest 3: Priority updates")
    print(f"Initial beta value: {buffer.beta:.3f}")
    
    # Update priorities and sample again
    new_priorities = T.rand(batch_size, device=buffer.device) * 2.0  # Random priorities between 0 and 2
    buffer.update_priorities(indices, new_priorities)
    
    # Log metrics after priority update
    log_priority_metrics(buffer, 100, prefix='buffer_after_update/')
    
    # Sample after priority update
    if buffer.goal_shape is not None:
        states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, weights, indices = buffer.sample(batch_size)
    else:
        states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size)
    
    print(f"\nWeight range after priority update: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"Updated beta value: {buffer.beta:.3f}")
    
    # Log final metrics
    wandb.log({
        'final/weights_range_min': float(weights.min()),
        'final/weights_range_max': float(weights.max()),
        'final/beta': float(buffer.beta)
    })
    
    # Plot weight distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(weights.cpu().numpy(), bins=20)
    plt.title('Importance Sampling Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(weights.cpu().numpy(), bins=20)
    plt.title('Weights Distribution After Priority Update')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Create test_results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    plt.savefig('test_results/priority_weights_distribution.png')
    plt.close()
    
    # Log the plot to wandb
    wandb.log({
        'plots/weight_distribution': wandb.Image('test_results/priority_weights_distribution.png')
    })
    
    wandb.finish()

if __name__ == "__main__":
    test_sum_tree()
    test_prioritized_buffer() 