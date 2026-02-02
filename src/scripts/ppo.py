import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from agent_utils import *
from icm import ICM
from adaptive_kl import AdaptiveKL

##########################################
# Training Parameters
env_string = 'FetchReachDense-v4'
timesteps = 500_000
trajectory_length = 32
batch_size = 64
learning_epochs = 10
train_envs = 64
train_render_freq = 1000
seed = 42
device = 'cuda'

# Testing Parameters
num_episodes = 100
test_envs = 1
test_render_freq = 10

# Policy Model
distribution = 'normal'
policy_optimizer_type = 'Adam'
policy_optimizer_lr = 3e-4
policy_clip = 0.2
use_policy_clip_scheduler = False
policy_clip_scheduler_type = 'linear'
policy_clip_start_factor = 1.0
policy_clip_end_factor = 0.1
policy_clip_total_iters = 500_000
policy_layer_config = [
    # {'type': 'cnn', 'params': {'out_channels': 32, 'kernel_size': (8, 8), 'stride': 4, 'padding': 0}},
    # {'type': 'cnn', 'params': {'out_channels': 64, 'kernel_size': (4, 4), 'stride': 2, 'padding': 0}},
    # {'type': 'cnn', 'params': {'out_channels': 64, 'kernel_size': (3, 3), 'stride': 1, 'padding': 0}},
    # {'type': 'flatten'},
    {'type': 'dense', 'params': {'units': 64, 'kernel': 'default', 'kernel params':{}}},
    {'type': 'tanh'},
    {'type': 'dense', 'params': {'units': 64, 'kernel': 'default', 'kernel params':{}}},
    {'type': 'tanh'},
]
policy_output_layer_config = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]

# Value Model
value_optimizer_type = 'Adam'
value_optimizer_lr = 3e-4
value_loss_coefficient = 0.5
value_clip = 0.2
use_value_clip_scheduler = False
value_clip_scheduler_type = 'linear'
value_clip_start_factor = 1.0
value_clip_end_factor = 0.1
value_clip_total_iters = 500_000
value_layer_config = [
    # {'type': 'cnn', 'params': {'out_channels': 32, 'kernel_size': (8, 8), 'stride': 4, 'padding': 0}},
    # {'type': 'cnn', 'params': {'out_channels': 64, 'kernel_size': (4, 4), 'stride': 2, 'padding': 0}},
    # {'type': 'cnn', 'params': {'out_channels': 64, 'kernel_size': (3, 3), 'stride': 1, 'padding': 0}},
    # {'type': 'flatten'},
    {'type': 'dense', 'params': {'units': 64, 'kernel': 'default', 'kernel params':{}}},
    {'type': 'tanh'},
    {'type': 'dense', 'params': {'units': 64, 'kernel': 'default', 'kernel params':{}}},
    {'type': 'tanh'},
   
]
value_output_layer_config = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]

# Intrinsic Curiosity Module (ICM)
use_icm = False
icm_optimizer_type = 'Adam'
icm_optimizer_lr = 1e-3
icm_reward_weight = 0.01
use_icm_reward_scheduler = True
icm_reward_scheduler_type = 'linear'
icm_reward_scheduler_start_factor = 1.0
icm_reward_scheduler_end_factor = 0.1
icm_reward_scheduler_total_iters = 500_000
icm_beta = 0.2
icm_extrinsic_threshold = 0 # Number of steps to only use intrinsic rewards before adding extrinsic rewards
icm_model_configs = {
    # 'encoder': {
    #     'layer_config': [
    #         {'type': 'dense', 'params': {'units': 64, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
    #         {'type': 'relu'},
    #     ],
    #     'output_layer': [
    #         {'type': 'dense', 'params': {'units': 256, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}}
    #     ]
    # },
    'inverse_model': {
        'layer_config': [
            {'type': 'dense', 'params': {'units': 256, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
        ]
    },
    'forward_model': {
        'layer_config': [
            {'type': 'dense', 'params': {'units': 256, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
        ]
    }
}

# Entropy
entropy_coefficient = 0.0
use_entropy_scheduler = False
entropy_scheduler_type = 'linear'
entropy_start_factor = 1.0
entropy_end_factor = 0.0001
entropy_total_iters = 500_000

# KL Divergence
kl_coefficient = 0.0
use_adaptive_kl = False
initial_beta = 1.0
target_kl = 0.01
scale_up = 2.0
scale_down = 0.5
kl_tolerance_high = 1.5
kl_tolerance_low = 0.5

# Normalizers
normalize_advantages = True

use_state_normalizer = False
state_normalizer_momentum = 0.99
state_normalizer_update_freq = 100 # Number of steps between updates
state_normalizer_clip_range = 5.0

use_goal_normalizer = False
goal_normalizer_momentum = 0.99
goal_normalizer_update_freq = 200 # Number of steps between updates
goal_normalizer_clip_range = 5.0

# Clipping
grad_clip = 0.5
reward_clip = 10.0

# Other Params
discount = 0.99
gae_coefficient = 0.95
##########################################


# Instantiate EnvWrapper
env = gym.make(env_string)
env_spec = env.spec
env_wrap = GymnasiumWrapper(env_spec)

# Policy Model
policy_optimizer = {'type': policy_optimizer_type,'params': { 'lr': policy_optimizer_lr}}
policy = StochasticContinuousPolicy(env_wrap, policy_layer_config, policy_output_layer_config, policy_optimizer, distribution=distribution, device=device)

# Value Model
value_optimizer = {'type': value_optimizer_type,'params': { 'lr': value_optimizer_lr}}
value_function = ValueModel(env_wrap, value_layer_config, value_output_layer_config, value_optimizer, device=device)

# Intrinsic Curiosity Module (ICM)
if use_icm:
    icm_optimizer = {'type': icm_optimizer_type,'params': { 'lr': icm_optimizer_lr}}
    icm_reward_scheduler_config = {
        'type': icm_reward_scheduler_type,
        'params': {
            'start_factor': icm_reward_scheduler_start_factor,
            'end_factor': icm_reward_scheduler_end_factor,
            'total_iters': icm_reward_scheduler_total_iters
        }
    }
    icm_reward_scheduler = ScheduleWrapper(icm_reward_scheduler_config)
    icm = ICM(env_wrap, icm_model_configs, icm_optimizer, icm_reward_weight, icm_reward_scheduler,
              icm_beta, icm_extrinsic_threshold, device=device)
else:
    icm = None

# Schedulers
if use_policy_clip_scheduler:
    policy_clip_scheduler_config = {
        'type': policy_clip_scheduler_type,
        'params': {
            'start_factor': policy_clip_start_factor,
            'end_factor': policy_clip_end_factor,
            'total_iters': policy_clip_total_iters
        }
    }
    policy_clip_scheduler = ScheduleWrapper(policy_clip_scheduler_config)
else:
    policy_clip_scheduler = None

if use_value_clip_scheduler:
    value_clip_scheduler_config = {
        'type': value_clip_scheduler_type,
        'params': {
            'start_factor': value_clip_start_factor,
            'end_factor': value_clip_end_factor,
            'total_iters': value_clip_total_iters
        }
    }
    value_clip_scheduler = ScheduleWrapper(value_clip_scheduler_config)
else:
    value_clip_scheduler = None

if use_entropy_scheduler:
    entropy_scheduler_config = {
        'type': entropy_scheduler_type,
        'params': {
            'start_factor': entropy_start_factor,
            'end_factor': entropy_end_factor,
            'total_iters': entropy_total_iters
        }
    }
    entropy_scheduler = ScheduleWrapper(entropy_scheduler_config)
else:
    entropy_scheduler = None

if use_adaptive_kl:
    adaptive_kl = AdaptiveKL(initial_beta, target_kl, scale_up, scale_down, kl_tolerance_high, kl_tolerance_low)
else:
    adaptive_kl = None

# Normalizers
if isinstance(env_wrap.single_observation_space, gym.spaces.Dict):
    state_normalizer_size = env_wrap.single_observation_space['observation'].shape
    goal_normalizer_size = env_wrap.single_observation_space['desired_goal'].shape
else:
    state_normalizer_size = env_wrap.single_observation_space.shape

if use_state_normalizer:
    state_normalizer = Normalizer(
        size=state_normalizer_size,
        momentum=state_normalizer_momentum,
        update_freq=state_normalizer_update_freq,
        clip_range=state_normalizer_clip_range,
        device=device
    )
else:
    state_normalizer = None

if use_goal_normalizer:
    goal_normalizer = Normalizer(
        size=goal_normalizer_size,
        momentum=goal_normalizer_momentum,
        update_freq=goal_normalizer_update_freq,
        clip_range=goal_normalizer_clip_range,
        device=device
    )
else:
    goal_normalizer = None


# Set Save Dir
save_dir = f'/workspaces/PhoenX_RL/src/app/agents/{env_string}/PPO/'

# Create PPO Agent
agent_class = get_agent_class_from_type('PPO')
ppo = agent_class(
    env=env_wrap,
    policy_model=policy,
    value_model=value_function,
    discount=discount,
    gae_coefficient=gae_coefficient,
    policy_clip=policy_clip,
    policy_clip_schedule=policy_clip_scheduler,
    value_clip=value_clip,
    value_clip_schedule=value_clip_scheduler,
    value_loss_coefficient=value_loss_coefficient,
    entropy_coefficient=entropy_coefficient,
    entropy_schedule=entropy_scheduler,
    kl_coefficient=kl_coefficient,
    kl_adapter=adaptive_kl,
    normalize_advantages=normalize_advantages,
    curiosity=icm,
    state_normalizer=state_normalizer,
    goal_normalizer=goal_normalizer,
    obs_key='observation',
    goal_key='desired_goal',
    achieved_goal_key='achieved_goal',
    grad_clip=grad_clip,
    reward_clip=reward_clip,
    callbacks=[WandbCallback(env_string)],
    save_dir=save_dir,
    device=device,
    log_level='info'
)

# Save Agent
ppo.save()


# Set train config and path
train_config = {
    'timesteps': timesteps,
    'trajectory_length': trajectory_length,
    'batch_size': batch_size,
    'learning_epochs': learning_epochs,
    'num_envs': train_envs,
    'render_freq': train_render_freq,
    'seed': seed,
}
train_config_path = save_dir + 'train_config.json'
with open(train_config_path, 'w') as f:
    json.dump(train_config, f)

# Set test config and path
test_config = {
    'num_episodes': num_episodes,
    'num_envs': test_envs,
    'render_freq': test_render_freq,
    'seed': seed,
}
test_config_path = save_dir + 'test_config.json'
with open(test_config_path, 'w') as f:
    json.dump(test_config, f)