import gymnasium as gym
import torch as T
import numpy as np
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from icm import ICM
from env_wrapper import EnvWrapper, GymnasiumWrapper
from schedulers import ScheduleWrapper  # If needed
from gymnasium.envs.registration import EnvSpec

class SimplePolicy(T.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = T.nn.Linear(obs_dim, 64)
        self.fc2 = T.nn.Linear(64, 64)
        self.out = T.nn.Linear(64, action_dim)

        self.to(T.device('cpu'))

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        return T.softmax(self.out(x), dim=-1)

# Setup environment
env_spec = EnvSpec(id='CartPole-v1', entry_point='gymnasium.envs.classic_control.cartpole:CartPoleEnv')
wrapped_env = GymnasiumWrapper(env_spec)

# Define ICM configurations
model_configs = {
    # 'encoder': {
    #     'layer_config': [
    #         {'type': 'dense', 'params': {'units': 64, 'kernel': 'ones', 'kernel params':{}}},
    #         {'type': 'relu'},
    #     ],
    #     'output_layer': [
    #         {'type': 'dense', 'params': {'units': 256, 'kernel': 'ones', 'kernel params':{}}}
    #     ]
    # },
    'inverse_model': {
        'layer_config': [
            {'type': 'dense', 'params': {'units': 64, 'kernel': 'ones', 'kernel params':{}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'kernel': 'ones', 'kernel params':{}}}
        ]
    },
    'forward_model': {
        'layer_config': [
            {'type': 'dense', 'params': {'units': 64, 'kernel': 'ones', 'kernel params':{}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'units': 4, 'kernel': 'ones', 'kernel params':{}}}
        ]
    }
}

optimizer_params = {'type': 'Adam', 'params': {'lr': 1e-3}}
icm = ICM(wrapped_env, model_configs, optimizer_params, reward_weight=0.2, beta=0.2, device=T.device('cpu'))

# Policy
obs_dim = 4  # CartPole
action_dim = 2
policy = SimplePolicy(obs_dim, action_dim)
policy_optimizer = T.optim.Adam(policy.parameters(), lr=0.001)

# Training parameters
gamma = 0.99
num_episodes = 2000

for episode in range(num_episodes):
    state, info = wrapped_env.reset()
    done = False
    trajectory = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    while not done:
        state_t = T.from_numpy(state).float()
        # print(f'state_t shape: {state_t.shape}')
        probs = policy(state_t)
        # print(f'probs shape: {probs.shape}')
        action = T.multinomial(probs, 1).numpy()
        action = wrapped_env.format_actions(action)
        # print(f'action: {action}')
        next_state, reward, done, _ = wrapped_env.step(action)

        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['next_states'].append(next_state)
        trajectory['rewards'].append(reward)
        trajectory['dones'].append(done)

        state = next_state

    # Convert to tensors
    states = T.from_numpy(np.array(trajectory['states'])).float().squeeze(1)
    next_states = T.from_numpy(np.array(trajectory['next_states'])).float().squeeze(1)
    actions = T.tensor(trajectory['actions']).squeeze(1)
    ext_rewards = T.tensor(trajectory['rewards'])

    # Compute intrinsic rewards (fix to scalar per transition)
    # print(f'states shape: {states.shape}')
    # print(f'next_states shape: {next_states.shape}')
    # print(f'actions shape: {actions.shape}')
    intr_rewards = icm.compute_intrinsic_reward(states, next_states, actions)
    # print(f'intr_rewards: {intr_rewards}')
    # print(f'intr_rewards shape: {intr_rewards.shape}')

    # Combine rewards (only intrinsic for this test)
    rewards = intr_rewards  # Or rewards = ext_rewards + intr_rewards for full ICM

    # Compute discounted returns
    returns = []
    R = 0.0
    for r in reversed(rewards.tolist()):
        R = r + gamma * R
        returns.insert(0, R)
    returns = T.tensor(returns)

    # Normalize returns (optional)
    if returns.std() > 0:
        returns = (returns - returns.mean()) / returns.std()

    # Compute policy loss (REINFORCE)
    log_probs = []
    for s, a in zip(trajectory['states'], trajectory['actions']):
        # print(f's: {s}')
        # print(f'a: {a}')
        s_t = T.from_numpy(s).float()
        # print(f's_t: {s_t}')
        probs = policy(s_t)  # No squeeze
        # print(f'probs: {probs}')
        log_prob = T.log(probs[0, a])  # Ensure [0, a] indexing
        log_probs.append(log_prob)
    log_probs = T.cat(log_probs)

    policy_loss = - (log_probs * returns).mean()

    # Update policy
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Train ICM
    icm_loss = icm.train(states, next_states, actions)

    # Logging
    total_ext_reward = ext_rewards.sum().item()
    print(f'Episode {episode + 1}: ICM Loss = {icm_loss:.4f}, Total Extrinsic Reward = {total_ext_reward:.2f}')

# To check convergence, observe if ICM Loss decreases over episodes. 