import pytest
import logging
import torch as T
import torch.nn as nn
import gymnasium as gym

# Import your ActorModel (adjust the import path as needed)
from models import ActorModel

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Mock Environment Classes
# ------------------------------------------------------------------------------

class MockVecEnv:
    """
    A minimal vectorized environment for testing.
    It creates an observation space whose shape is (num_envs, *obs_shape)
    and a continuous action space (Box) with a given shape.
    """
    def __init__(self, num_envs, obs_shape, action_shape):
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(num_envs, *obs_shape), dtype=T.float32
        )
        self.single_observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=T.float32
        )
        # For continuous actions, we use a Box. (The ActorModel scales the output by single_action_space.high.)
        if isinstance(action_shape, int):
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(action_shape,), dtype=T.float32
            )
            self.single_action_space = self.action_space
        elif isinstance(action_shape, (tuple, list)):
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=tuple(action_shape), dtype=T.float32
            )
            self.single_action_space = self.action_space
        else:
            raise ValueError("Unsupported action shape format.")

    def reset(self):
        return self.observation_space.sample()


class MockGymnasiumWrapper:
    """
    A simple wrapper mimicking your GymnasiumWrapper.
    """
    def __init__(self, num_envs, obs_shape, action_shape):
        self.env = MockVecEnv(num_envs, obs_shape, action_shape)
        logger.info(f"Env observation space: {self.env.observation_space}")
        logger.info(f"Env action space: {self.env.action_space}")

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    def reset(self):
        return self.env.reset()

# ------------------------------------------------------------------------------
# ActorModel Fixture
# ------------------------------------------------------------------------------

@pytest.fixture
def actor_model(request):
    """
    Expects request.param as a tuple:
      (num_envs, obs_shape, action_shape, layer_config, goal_dim)
    where goal_dim==0 (or None) means no goal is passed to forward.
    """
    num_envs, obs_shape, action_shape, layer_config, goal_dim = request.param
    logger.info(f"Initializing ActorModel with num_envs={num_envs}, "
                f"obs_shape={obs_shape}, action_shape={action_shape}, "
                f"layer_config={layer_config}, goal_dim={goal_dim}")

    env = MockGymnasiumWrapper(num_envs, obs_shape, action_shape)
    model = ActorModel(env=env, layer_config=layer_config)
    
    # (Optional) Register forward hooks to log layer input/output shapes.
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Flatten, nn.Linear)):
            layer.register_forward_hook(
                lambda m, inp, out: logger.info(
                    f"Layer {m.__class__.__name__} input: {inp[0].shape}, output: {out.shape}"
                )
            )

    # Force lazy layers (if any) to initialize by running a dummy forward pass.
    dummy_obs = T.zeros((num_envs, *obs_shape))
    if goal_dim and goal_dim > 0:
        dummy_goal = T.zeros((num_envs, goal_dim))
        _ = model(dummy_obs, dummy_goal)
    else:
        _ = model(dummy_obs)

    # Manually set weights and biases to known values.
    with T.no_grad():
        # For layers built from layer_config
        for layer in model.layers.values():
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.fill_(0.5)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.fill_(0.1)
        # For the output layer (actor_mu) of ActorModel.
        if hasattr(model.output_layer["actor_mu"], "weight") and model.output_layer["actor_mu"].weight is not None:
            model.output_layer["actor_mu"].weight.fill_(0.5)
        if hasattr(model.output_layer["actor_mu"], "bias") and model.output_layer["actor_mu"].bias is not None:
            model.output_layer["actor_mu"].bias.fill_(0.1)
    return model

# ------------------------------------------------------------------------------
# Tests for ActorModel.forward (No Goal vs. With Goal)
# ------------------------------------------------------------------------------

# Test without supplying a goal.
@pytest.mark.parametrize("actor_model", [
    # (num_envs, obs_shape, action_shape, layer_config, goal_dim)
    (2, (3,), 1, [{'type': 'dense', 'params': {'units': 1, 'kernel': 'default', 'kernel params': {}}}], 0)
], indirect=True)
def test_actor_forward_no_goal(actor_model):
    # Define a simple observation tensor (2 environments, 3 features each)
    obs = T.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=T.float32)
    # Call forward without a goal.
    mu, pi = actor_model(obs)
    
    # --- Manual Computation ---
    # The network passes the input through a single dense layer (from layer_config):
    #   dense_out = 0.5 * (sum of obs elements) + 0.1
    # Then through the output layer "actor_mu":
    #   mu = 0.5 * (dense_out) + 0.1
    #
    # For the first sample: sum = 1+2+3 = 6.0
    #   dense_out = 0.5 * 6.0 + 0.1 = 3.0 + 0.1 = 3.1
    #   mu = 0.5 * 3.1 + 0.1 = 1.55 + 0.1 = 1.65
    #
    # For the second sample: sum = 4+5+6 = 15.0
    #   dense_out = 0.5 * 15.0 + 0.1 = 7.5 + 0.1 = 7.6
    #   mu = 0.5 * 7.6 + 0.1 = 3.8 + 0.1 = 3.9
    #
    # The actor_pi layer applies tanh and then scales by env.single_action_space.high.
    # Here, since the Box high is [1.0] (from MockVecEnv), the scaling does not change the tanh output.
    expected_mu = T.tensor([[1.65],
                            [3.9]], dtype=T.float32)
    expected_pi = T.tanh(expected_mu)
    
    logger.info(f"Computed mu: {mu}, Expected mu: {expected_mu}")
    logger.info(f"Computed pi: {pi}, Expected pi: {expected_pi}")
    
    assert T.allclose(mu, expected_mu, atol=1e-5), "Mu output (no goal) is incorrect."
    assert T.allclose(pi, expected_pi, atol=1e-5), "Pi output (no goal) is incorrect."


# Test when a goal is provided.
@pytest.mark.parametrize("actor_model", [
    # Here, goal_dim is 2. The forward method will concatenate the goal to the observation.
    (2, (3,), 1, [{'type': 'dense', 'params': {'units': 1, 'kernel': 'default', 'kernel params': {}}}], 2)
], indirect=True)
def test_actor_forward_with_goal(actor_model):
    # Define observations and goals.
    obs = T.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=T.float32)
    goal = T.tensor([[0.1, 0.2],
                     [0.3, 0.4]], dtype=T.float32)
    # Call forward with a goal.
    mu, pi = actor_model(obs, goal)
    
    # --- Manual Computation ---
    # For sample 1: concatenated input = [1, 2, 3, 0.1, 0.2] → sum = 6.3
    #   dense_out = 0.5 * 6.3 + 0.1 = 3.15 + 0.1 = 3.25
    #   mu = 0.5 * 3.25 + 0.1 = 1.625 + 0.1 = 1.725
    #
    # For sample 2: concatenated input = [4, 5, 6, 0.3, 0.4] → sum = 15.7
    #   dense_out = 0.5 * 15.7 + 0.1 = 7.85 + 0.1 = 7.95
    #   mu = 0.5 * 7.95 + 0.1 = 3.975 + 0.1 = 4.075
    expected_mu = T.tensor([[1.725],
                            [4.075]], dtype=T.float32)
    expected_pi = T.tanh(expected_mu)
    
    logger.info(f"Computed mu: {mu}, Expected mu: {expected_mu}")
    logger.info(f"Computed pi: {pi}, Expected pi: {expected_pi}")
    
    assert T.allclose(mu, expected_mu, atol=1e-5), "Mu output (with goal) is incorrect."
    assert T.allclose(pi, expected_pi, atol=1e-5), "Pi output (with goal) is incorrect."
