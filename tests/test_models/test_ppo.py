import pytest
import logging
import torch as T
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock
from rl_agents import PPO
from env_wrapper import EnvWrapper, GymnasiumWrapper

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class MockEnvWrapper(EnvWrapper):
    def __init__(self):
        super().__init__()

class MockGymnasiumWrapper(GymnasiumWrapper):
    def __init__(self):
        # super().__init__(MockEnvSpec(observation_space))
        # self.env = MockEnvWrapper(observation_space)
        self.env = MockVecEnv()
        # Implement other required methods here or leave them as is
        logger.info(f'env observation space:{self.env.observation_space}')
        logger.info(f'env single observation space:{self.env.single_observation_space}')
        logger.info(f'env action space:{self.env.action_space}')
        logger.info(f'env single action space:{self.env.single_action_space}')

    def reset(self):
        single_obs = self.single_observation_space.sample()
        return T.stack([single_obs for _ in range(self.observation_space.shape[0])]), {}
    
    def step(self, action):
        single_obs = T.tensor(self.single_observation_space.sample())
        observation = T.stack([single_obs for _ in range(self.observation_space.shape[0])])
        reward = T.zeros(self.observation_space.shape[0])
        terminated = T.zeros(self.observation_space.shape[0], dtype=T.bool)
        truncated = T.zeros(self.observation_space.shape[0], dtype=T.bool)
        info = {}
        return observation, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

# Mocking an environment
class MockVecEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 3))
        self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(2, 3))
        self.single_action_space = gym.spaces.Box(low=-2, high=2, shape=(3,))

    def reset(self):
        # Return an initial observation
        return self.observation_space.sample(), {}
    
    def step(self, action):
        # For simplicity, just return something
        observation = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

class MockPolicyModel:
    def __init__(self, device):
        self.device = device
        self.distribution = 'categorical'

class MockValueModel:
    def __init__(self, device):
        self.device = device

    def __call__(self, states):
        # Always returns 0.5 for any state input
        return T.full((states.shape[0],), 0.5, device=self.device)

@pytest.fixture
def ppo_instance():

    mock_env = MockGymnasiumWrapper()
    policy_model = MockPolicyModel(device='cpu')
    value_model = MockValueModel(device='cpu')
    
    ppo = PPO(
        env=mock_env,
        policy_model=policy_model,
        value_model=value_model,
        discount=0.99,
        gae_coefficient=0.95,
        normalize_advantages=False,  # Disabled for raw value checks
        device='cpu'
    )
    return ppo

def test_advantage_single_env_no_dones(ppo_instance):
    """Test advantages and returns for a single environment without any done flags."""
    rewards = T.tensor([[1.0], [1.0], [1.0]], dtype=T.float32)  # (3,1)
    states = T.rand(3, 1, 4, 84, 84)
    next_states = T.rand(3, 1, 4, 84, 84)
    dones = T.tensor([[0], [0], [0]], dtype=T.int)

    advantages, returns, values = ppo_instance.calculate_advantages_and_returns(rewards, states, next_states, dones)

    gamma = 0.99
    gae_lambda = 0.95
    value = 0.5

    # Reverse computation (since GAE is calculated backward)
    deltas = [1 + gamma * value - value] * 3  # All deltas are 0.995
    gae = 0.0
    expected_advantages = []
    for delta in reversed(deltas):
        gae = delta + gamma * gae_lambda * gae
        expected_advantages.insert(0, gae)
    expected_advantages = T.tensor(expected_advantages, dtype=T.float32).view(-1, 1)

    logger.info(f'advantages:{advantages}')
    logger.info(f'expected advantages:{expected_advantages}')
    logger.info(f'returns:{returns}')
    logger.info(f'expected returns:{expected_advantages + value}')
    logger.info(f'values:{values}')

    assert T.allclose(advantages, expected_advantages, atol=1e-4)
    assert T.allclose(returns, expected_advantages + value, atol=1e-4)
    assert T.allclose(values, T.full_like(values, 0.5), atol=1e-4)

def test_advantage_dones_mid_trajectory(ppo_instance):
    """Test when an episode ends mid-trajectory, resetting GAE."""
    rewards = T.tensor([[1.0], [1.0], [1.0]], dtype=T.float32)  # (3,1)
    states = T.rand(3, 1, 4, 84, 84)
    next_states = T.rand(3, 1, 4, 84, 84)
    dones = T.tensor([[0], [1], [0]], dtype=T.int)  # Episode ends at step 1

    advantages, returns, values = ppo_instance.calculate_advantages_and_returns(rewards, states, next_states, dones)

    # Manually compute expected advantages
    gamma = 0.99
    gae_lambda = 0.95
    value = 0.5

    # Process steps in reverse order (t=2,1,0)
    expected_advantages = []
    gae = 0.0

    # t=2: done=0, next_value=0.5
    delta = 1 + gamma * 0.5 - value
    gae = delta + gamma * gae_lambda * gae
    expected_advantages.insert(0, gae)

    # t=1: done=1, next_value=0
    delta = 1 + 0 - value
    gae = delta  # Reset GAE after done
    expected_advantages.insert(0, gae)

    # t=0: done=0, next_value=0.5
    delta = 1 + gamma * 0.5 - value
    gae = delta + gamma * gae_lambda * expected_advantages[0]  # Use previous GAE from t=1
    expected_advantages.insert(0, gae)

    expected_advantages = T.tensor(expected_advantages, dtype=T.float32).view(-1, 1)

    logger.info(f'advantages:{advantages}')
    logger.info(f'expected advantages:{expected_advantages}')
    logger.info(f'returns:{returns}')
    logger.info(f'expected returns:{expected_advantages + value}')
    logger.info(f'values:{values}')
    
    assert T.allclose(advantages, expected_advantages, atol=1e-3)
    assert T.allclose(returns, expected_advantages + value, atol=1e-4)
    assert T.allclose(values, T.full_like(values, 0.5), atol=1e-4)

def test_advantage_multi_env_dones(ppo_instance):
    """Test with two environments where done flags occur at different steps."""
    rewards = T.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=T.float32)  # (3,2)
    states = T.rand(3, 2, 4, 84, 84)
    next_states = T.rand(3, 2, 4, 84, 84)
    dones = T.tensor([[0, 0], [1, 0], [0, 1]], dtype=T.int)  # Env0 done at step1, Env1 done at step2

    advantages, returns, values = ppo_instance.calculate_advantages_and_returns(rewards, states, next_states, dones)

    # Expected advantages for each environment
    gamma = 0.99
    gae_lambda = 0.95
    value = 0.5

    # Env0: done at step1
    advantages_env0 = []
    gae = 0.0
    # t=2: done=0
    delta = 5 + gamma * value - value
    gae = delta + gamma * gae_lambda * gae
    advantages_env0.insert(0, gae)
    # t=1: done=1
    delta = 3 + 0 - value
    gae = delta
    advantages_env0.insert(0, gae)
    # t=0: done=0
    delta = 1 + gamma * value - value
    gae = delta + gamma * gae_lambda * advantages_env0[0]
    advantages_env0.insert(0, gae)

    # Env1: done at step2
    advantages_env1 = []
    gae = 0.0
    # t=2: done=1
    delta = 6 + 0 - value
    gae = delta
    advantages_env1.insert(0, gae)
    # t=1: done=0
    delta = 4 + gamma * value - value
    gae = delta + gamma * gae_lambda * advantages_env1[0]
    advantages_env1.insert(0, gae)
    # t=0: done=0
    delta = 2 + gamma * value - value
    gae = delta + gamma * gae_lambda * advantages_env1[0]
    advantages_env1.insert(0, gae)

    expected_advantages = T.tensor([advantages_env0, advantages_env1], dtype=T.float32).T

    logger.info(f'advantages:{advantages}')
    logger.info(f'expected advantages:{expected_advantages}')
    logger.info(f'returns:{returns}')
    logger.info(f'expected returns:{expected_advantages + value}')
    logger.info(f'values:{values}')
    
    assert T.allclose(advantages, expected_advantages, atol=1e-3)
    assert T.allclose(returns, expected_advantages + value, atol=1e-4)
    assert T.allclose(values, T.full_like(values, 0.5), atol=1e-4)

def test_normalized_advantages(ppo_instance):
    """Test if advantages are normalized when enabled."""
    ppo_instance.normalize_advantages = True
    rewards = T.tensor([[1.0], [2.0], [3.0]], dtype=T.float32)
    states = T.rand(3, 1, 4, 84, 84)
    next_states = T.rand(3, 1, 4, 84, 84)
    dones = T.tensor([[0], [0], [0]], dtype=T.int)

    advantages, _, _ = ppo_instance.calculate_advantages_and_returns(rewards, states, next_states, dones)

    # Manually compute expected advantages
    gamma = 0.99
    gae_lambda = 0.95
    value = 0.5

    # Process steps in reverse order (t=2,1,0)
    expected_advantages = []
    gae = 0.0

    # t=2: done=0, next_value=0.5
    delta = 3 + gamma * 0.5 - value
    gae = delta + gamma * gae_lambda * gae
    expected_advantages.insert(0, gae)

    # t=1: done=0, next_value=0
    delta = 2 + gamma * 0.5 - value
    gae = delta + gamma * gae_lambda * gae
    expected_advantages.insert(0, gae)

    # t=0: done=0, next_value=0.5
    delta = 1 + gamma * 0.5 - value
    gae = delta + gamma * gae_lambda * expected_advantages[0]  # Use previous GAE from t=1
    expected_advantages.insert(0, gae)

    expected_advantages = T.tensor(expected_advantages, dtype=T.float32).unsqueeze(-1)
    # Normalization check
    normalized = (expected_advantages - expected_advantages.mean()) / (expected_advantages.std() + 1e-4)
    logger.info(f'advantages:{advantages}')
    logger.info(f'expected advantages:{normalized}')
    assert T.allclose(advantages, normalized, atol=1e-5)

def test_action_adapter_normal(ppo_instance):
    # Set ppo_instance distribution to 'normal'
    ppo_instance.policy_model.distribution = 'normal'
    
    # Simulate raw actions from a Normal distribution (unbounded values).
    raw_actions = np.array([[ -3.0, 0.5, 2.0 ], [ -1.0, 1.0, 0.7 ]])
    
    # Call the adapter in your PPO class (assuming you have defined one as in previous examples).
    adapted_actions = ppo_instance.action_adapter(raw_actions)
    
    # Squash raw actions between 0 and 1 since distribution is normal
    squashed = 1/(1 + np.exp(-raw_actions))
    expected = ppo_instance.env.env.single_action_space.low + (
        ppo_instance.env.env.single_action_space.high - ppo_instance.env.env.single_action_space.low
    ) * squashed

    logger.info(f'actions:{adapted_actions}')
    logger.info(f'expected actions:{expected}')

    assert np.allclose(adapted_actions, expected, atol=1e-4)
    # Also check that the adapted actions are within the action space:
    assert (adapted_actions >= ppo_instance.env.env.single_action_space.low).all()
    assert (adapted_actions <= ppo_instance.env.env.single_action_space.high).all()

def test_action_adapter_beta(ppo_instance):
    # Set ppo_instance distribution to 'normal'
    ppo_instance.policy_model.distribution = 'beta'
    
    # Simulate raw actions from a Normal distribution (unbounded values).
    raw_actions = np.array([[ 0.2, 0.5, 0.3 ], [ 0.3, 1.0, 0.7 ]])
    
    # Call the adapter in your PPO class (assuming you have defined one as in previous examples).
    adapted_actions = ppo_instance.action_adapter(raw_actions)
    
    expected = ppo_instance.env.env.single_action_space.low + (
        ppo_instance.env.env.single_action_space.high - ppo_instance.env.env.single_action_space.low
    ) * raw_actions

    logger.info(f'actions:{adapted_actions}')
    logger.info(f'expected actions:{expected}')

    assert np.allclose(adapted_actions, expected, atol=1e-4)
    # Also check that the adapted actions are within the action space:
    assert (adapted_actions >= ppo_instance.env.env.single_action_space.low).all()
    assert (adapted_actions <= ppo_instance.env.env.single_action_space.high).all()
