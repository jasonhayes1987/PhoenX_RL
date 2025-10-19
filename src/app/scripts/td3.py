import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from agent_utils import *
from icm import ICM

device = 'cuda'
num_envs = 4
N = 3
env_string = 'LunarLanderContinuous-v3'


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

# build actor
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

# build critic
# critic_optimizer = {'type': 'Adam','params': { 'lr': 0.001, 'weight_decay':0.01}}
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

replay_buffer = ReplayBuffer(env_wrap, 1000000, N=N, device='cpu')
# replay_buffer = PrioritizedReplayBuffer(env_wrap, 100000, alpha=0.6, beta_start=0.4, beta_iter=10000, beta_update_freq=1, priority='rank',normalize=False, epsilon=0.01, N=N, device='cpu')
noise = NormalNoise(stddev=0.1, device=device)
noise_schedule = None
target_noise = NormalNoise(stddev=0.1, device=device)
target_noise_schedule = None
target_noise_clip = 0.5
state_normalizer = Normalizer(size=env_wrap.single_observation_space.shape, update_freq=200, clip_range=5.0, device=device)
# state_normalizer = None

# ICM
# optimizer_params = {'type': 'Adam', 'params': {'lr': 1e-3}}

# # Define ICM configurations
# model_configs = {
#     # 'encoder': {
#     #     'layer_config': [
#     #         {'type': 'dense', 'params': {'units': 64, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}},
#     #         {'type': 'relu'},
#     #     ],
#     #     'output_layer': [
#     #         {'type': 'dense', 'params': {'units': 256, 'kernel': 'variance_scaling', 'kernel params':{"scale": 1.0, "mode": "fan_in", "distribution": "uniform"}}}
#     #     ]
#     # },
#     'inverse_model': {
#         'layer_config': [
#             {'type': 'dense', 'params': {'units': 64, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
#             {'type': 'relu'},
#         ],
#         'output_layer': [
#             {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
#         ]
#     },
#     'forward_model': {
#         'layer_config': [
#             {'type': 'dense', 'params': {'units': 64, 'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}},
#             {'type': 'relu'},
#         ],
#         'output_layer': [
#             {'type': 'dense', 'params': {'kernel': 'xavier_uniform', 'kernel params':{"gain": 2.0}}}
#         ]
#     }
# }

# schedule_config = {'type':'linear', 'params':{'start_factor':1.0, 'end_factor':0.1, 'total_iters':10000}}
# scheduler = ScheduleWrapper(schedule_config)

# icm = ICM(env_wrap, model_configs, optimizer_params, reward_weight=0.2, reward_scheduler=scheduler, beta=0.2, extrinsic_threshold=10000, warmup=1000, device=device)

icm = None

# Create TD3 Agent
agent_class = get_agent_class_from_type('TD3')

td3 = agent_class(
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
    callbacks=[WandbCallback(env_string)],
    save_dir=f'/workspaces/PhoenX_RL/src/app/agents/{env_string}_N{N}',
    device=device
)

# Save Agent
td3.save()

# Create train config
config = td3.get_config()

# Set train config and path
train_config = {
    'num_episodes': 1000,
    'num_envs': 4,
    'steps_per_learn': 1,
    'render_freq': 100,
    'seed': 42,
}
train_config_path = config["save_dir"] + 'train_config.json'
with open(train_config_path, 'w') as f:
    json.dump(train_config, f)

# Set test config and path
test_config = {
    'num_episodes': 100,
    'num_envs': 1,
    'seed': 42,
    'render_freq': 10
}
test_config_path = config["save_dir"] + 'test_config.json'
with open(test_config_path, 'w') as f:
    json.dump(test_config, f)