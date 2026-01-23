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

# scheduler_config = {
#     "type": "linear",
#     "params": {
#         "start_factor": 1.0,
#         "end_factor": 0.1,
#         "total_iters": 1000000
#     }
# }
# scheduler = ScheduleWrapper(scheduler_config)
scheduler = None

# build actor
optimizer = {'type': 'Adam','params': { 'lr': 3e-4 }}

layer_config = [
    # {'type': 'batchnorm1d'},
    {'type': 'dense', 'params': {'units': 256, 'kernel': 'default', 'kernel params':{}}},
    # {'type': 'batchnorm1d'},
    {'type': 'relu'},
    {'type': 'dense', 'params': {'units': 256, 'kernel': 'default', 'kernel params':{}}},
    # {'type': 'batchnorm1d'},
    {'type': 'relu'},
]
output_layer_config = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]

actor = StochasticContinuousPolicy(env_wrap, layer_config, output_layer_config, optimizer, scheduler, distribution='normal',device=device)

# build critic

state_layer_config = [

]

critic_a = CriticModel(env_wrap, state_layers=state_layer_config, merged_layers=layer_config,
                    output_layer_kernel=output_layer_config, optimizer_params=optimizer, lr_scheduler=scheduler, device=device)

# critic_b = critic_a.clone()


# Build Value model
value_model = ValueModel(env_wrap, layer_config, output_layer_config, optimizer_params=optimizer, lr_scheduler=scheduler, device=device)

replay_buffer = ReplayBuffer(env_wrap, 1000000, N=N, device='cpu')
# replay_buffer = PrioritizedReplayBuffer(env_wrap, 1000000, beta_start=0.4, beta_iter=100000, beta_update_freq=1, priority='rank',
#                                         normalize=False, goal_shape=env.observation_space['desired_goal'].shape, epsilon=0.01, N=N, device='cpu')
state_normalizer = Normalizer(size=env_wrap.single_observation_space.shape, update_freq=200, clip_range=5.0, device=device)

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

# schedule_config = {'type':'linear', 'params':{'start_factor':1.0, 'end_factor':0.1, 'total_iters':100000}}
# scheduler = ScheduleWrapper(schedule_config)

# icm = ICM(env_wrap, model_configs, optimizer_params, reward_weight=0.2, reward_scheduler=scheduler, beta=0.2, extrinsic_threshold=1000, warmup=1000, device=device)
icm = None

# Create SAC object
agent_class = get_agent_class_from_type('SAC')
sac = agent_class(
    env=env_wrap,
    actor_model=actor,
    value_model=value_model,
    critic_model_a=critic_a,
    # critic_model_b=critic_b,
    replay_buffer=replay_buffer,
    discount=0.99,
    tau=0.005,
    alpha=0.4,
    auto_entropy_tuning=False,
    alpha_lr=1e-4,
    batch_size=512,
    grad_clip=0.5,
    warmup=1000,
    N=N,
    curiosity=icm,
    state_normalizer=state_normalizer,
    callbacks=[WandbCallback(env_string)],
    save_dir=f'/workspaces/PhoenX_RL/src/app/agents/{env_string}_N{N}',
    device=device
)

sac.save()

config = sac.get_config()

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