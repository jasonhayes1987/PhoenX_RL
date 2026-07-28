import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from agent_utils import *
from icm import ICM

base_agent_type = 'SAC'
device = 'cuda'
N = 3
env_string = 'FetchPush-v4'


env = gym.make(env_string)

wrappers = [
    {
        "type": "NStepReward",
        "params": {
            "n": N
        }
    }
]

env_spec = env.spec
env_wrap = GymnasiumWrapper(env_spec, wrappers)

# Build Actor (DDPG/TD3)
if base_agent_type in ['DDPG', 'TD3']:
    actor_optimizer = {'type': 'Adam','params': { 'lr': 0.001 }}

    layer_config = [
        # {'type': 'batchnorm1d'},
        {'type': 'dense', 'params': {'units': 400, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        # {'type': 'batchnorm1d'},
        {'type': 'relu'},
        {'type': 'dense', 'params': {'units': 300, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        # {'type': 'batchnorm1d'},
        {'type': 'relu'},
    ]
    # output_layer_config = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]
    output_layer_config = [{'type': 'dense', 'params': {'kernel': 'uniform', 'kernel params':{'a':-3e-3, 'b':3e-3}}}]

    actor = ActorModel(env_wrap, layer_config, output_layer_config, optimizer_params=actor_optimizer, device=device)

if base_agent_type == 'SAC':
    # Build Stochastic Continuous Actor
    scheduler_config = {
        "type": "linear",
        "params": {
            "start_factor": 1.0,
            "end_factor": 0.1,
            "total_iters": 1000000
        }
    }
    # scheduler = ScheduleWrapper(scheduler_config)
    scheduler = None
    optimizer = {'type': 'Adam','params': { 'lr': 3e-4 }}

    layer_config = [
        # {'type': 'batchnorm1d'},
        {'type': 'dense', 'params': {'units': 256, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        # {'type': 'batchnorm1d'},
        {'type': 'relu'},
        {'type': 'dense', 'params': {'units': 256, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        # {'type': 'batchnorm1d'},
        {'type': 'relu'},
    ]
    output_layer_config = [{'type': 'dense', 'params': {'kernel': 'uniform', 'kernel params':{'a':-3e-3, 'b':3e-3}}}]

    actor = StochasticContinuousPolicy(env_wrap, layer_config, output_layer_config, optimizer, scheduler, distribution='normal',device=device)


# Build Critic (DDPG & TD3)
if base_agent_type in ['DDPG', 'TD3']:
    critic_optimizer = {'type': 'Adam','params': { 'lr': 0.001, 'weight_decay':0.01}}
    critic_optimizer = {'type': 'Adam','params': { 'lr': 0.001}}

    state_layer_config = [
        # {'type': 'batchnorm1d'},
        {'type': 'dense', 'params': {'units': 400, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        # {'type': 'batchnorm1d'},
        {'type': 'relu'}
    ]

    merged_layer_config = [
        {'type': 'dense', 'params': {'units': 300, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
        {'type': 'relu'},
    ]
    # output_layer_config = {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}},

    critic = CriticModel(env_wrap, state_layers=state_layer_config, merged_layers=merged_layer_config,
                        output_layer_kernel=output_layer_config, optimizer_params=critic_optimizer, device=device)


# Build Crtic (SAC)
if base_agent_type == 'SAC':
    state_layer_config = [
    ]

    critic = CriticModel(env_wrap, state_layers=state_layer_config, merged_layers=layer_config,
                        output_layer_kernel=output_layer_config, optimizer_params=optimizer, lr_scheduler=scheduler, device=device)

    # critic_b = critic_a.clone()


# ICM
icm_optimizer = {'type': 'Adam', 'params': {'lr': 1e-3}}

# # Define ICM configurations
model_configs = {
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
            {'type': 'dense', 'params': {'units': 64, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
        ]
    },
    'forward_model': {
        'layer_config': [
            {'type': 'dense', 'params': {'units': 64, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
            {'type': 'relu'},
        ],
        'output_layer': [
            {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
        ]
    }
}

schedule_config = {'type':'linear', 'params':{'start_factor':1.0, 'end_factor':0.1, 'total_iters':10000}}
scheduler = ScheduleWrapper(schedule_config)
icm = ICM(env_wrap, model_configs, icm_optimizer, reward_weight=0.2, reward_scheduler=scheduler, beta=0.2, extrinsic_threshold=10000, warmup=1000, device=device)
# icm = None


# Shared parameters
replay_buffer = ReplayBuffer(env_wrap, 1000000, goal_shape=env_wrap.single_observation_space['desired_goal'].shape, N=N, device='cpu')
# replay_buffer = PrioritizedReplayBuffer(env_wrap, 100000, alpha=0.6, beta_start=0.4, beta_iter=10000, beta_update_freq=1, priority='rank',normalize=False, epsilon=0.01, N=N, device='cpu')
state_normalizer = Normalizer(size=env_wrap.single_observation_space['observation'].shape, update_freq=200, clip_range=5.0, device=device)
goal_normalizer = Normalizer(size=env_wrap.single_observation_space['desired_goal'].shape, update_freq=200, clip_range=5.0, device=device)

# DDPG and TD3 parameters
if base_agent_type in ['DDPG', 'TD3']:
    noise = NormalNoise(stddev=0.1, device=device)
    noise_schedule = None

# Set save directories
base_save_dir = f'/workspaces/PhoenX_RL/src/Trained_Models/{env_string}_N{N}/HER_{base_agent_type}/'
her_save_dir = f'{base_save_dir}her/'

# Create DDPG Agent
if base_agent_type == 'DDPG':
    agent_class = get_agent_class_from_type('DDPG')
    base_agent = agent_class(
        env=env_wrap,
        actor_model=actor,
        critic_model=critic,
        replay_buffer=replay_buffer,
        discount=0.99,
        tau=0.01,
        action_epsilon=0.2,
        batch_size=2048,
        noise=noise,
        noise_schedule=noise_schedule,
        grad_clip=40.0,
        warmup=1000,
        N=N,
        curiosity=icm,
        state_normalizer=state_normalizer,
        goal_normalizer=goal_normalizer,
        obs_key='observation',
        goal_key='desired_goal',
        achieved_goal_key='achieved_goal',
        callbacks=[WandbCallback(env_string)],
        save_dir=base_save_dir,
        device=device,
        log_level='info')

# Create TD3 Agent
if base_agent_type == 'TD3':
    agent_class = get_agent_class_from_type('TD3')
    target_noise = NormalNoise(stddev=0.1, device=device)
    target_noise_schedule = None
    target_noise_clip = 0.5

    base_agent = agent_class(
        env=env_wrap,
        actor_model=actor,
        critic_model_a=critic,
        discount=0.99,
        tau=0.005,
        action_epsilon=0.2,
        replay_buffer=replay_buffer,
        batch_size=128,
        noise=noise,
        noise_schedule=noise_schedule,
        target_noise=target_noise,
        target_noise_schedule=target_noise_schedule,
        target_noise_clip=target_noise_clip,
        actor_update_delay = 2,
        grad_clip=40.0,
        warmup=1000,
        N=N,
        curiosity=icm,
        state_normalizer=state_normalizer,
        goal_normalizer=goal_normalizer,
        obs_key='observation',
        goal_key='desired_goal',
        achieved_goal_key='achieved_goal',
        callbacks=[WandbCallback(env_string)],
        save_dir=base_save_dir,
        device=device
    )

# Create SAC Agent
if base_agent_type == 'SAC':
    value_model = ValueModel(env_wrap, layer_config, output_layer_config, optimizer_params=optimizer, lr_scheduler=scheduler, device=device)
    agent_class = get_agent_class_from_type('SAC')
    base_agent = agent_class(
        env=env_wrap,
        actor_model=actor,
        value_model=value_model,
        critic_model_a=critic,
        # critic_model_b=critic_b,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=False,
        alpha_lr=1e-4,
        replay_buffer=replay_buffer,
        batch_size=512,
        grad_clip=0.5,
        warmup=1000,
        N=N,
        curiosity=icm,
        state_normalizer=state_normalizer,
        goal_normalizer=goal_normalizer,
        obs_key='observation',
        goal_key='desired_goal',
        achieved_goal_key='achieved_goal',
        callbacks=[WandbCallback(env_string)],
        save_dir=base_save_dir,
        device=device
    )

# Create HER Agent
agent_class = get_agent_class_from_type('HER')
agent = agent_class(
    agent=base_agent,
    strategy='future',
    tolerance=0.05,
    num_goals=4,
    save_dir=her_save_dir
)

# Save Agent
agent.save()

# Get config
# config = agent.get_config()

# Set train config and path
train_config = {
    'num_epochs': 100,
    'num_cycles': 50,
    'num_episodes': 1,
    'num_updates': 40,
    'num_envs': 16,
    'render_freq': 1000,
    'seed': 42
}
train_config_path = her_save_dir + 'train_config.json'
with open(train_config_path, 'w') as f:
    json.dump(train_config, f)

# Set test config and path
test_config = {
    'num_episodes': 100,
    'num_envs': 1,
    'render_freq': 10,
    'seed': 42
}
test_config_path = her_save_dir + 'test_config.json'
with open(test_config_path, 'w') as f:
    json.dump(test_config, f)

