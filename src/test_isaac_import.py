import nest_asyncio
nest_asyncio.apply()

import sys
import os

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

# from isaaclab.app import AppLauncher
import json
import torch
import numpy as np
from app.rl_agents import DDPG
from app.models import ActorModel, CriticModel
from app.buffer import ReplayBuffer, PrioritizedReplayBuffer
from app.normalizer import Normalizer, SharedNormalizer
from app.noise import NormalNoise, UniformNoise, OUNoise
from app.env_wrapper import IsaacSimWrapper
import app.rl_callbacks as rl_callbacks

# PARAMS
NUM_ENVS = 32
NUM_EPISODES = 100
DEVICE = "cuda:0"
SEED = 42
RENDER_MODE = 'headless' # 'headless' or 'gui'
N = 1 # N-step return
OBS_KEY = 'policy'
GOAL_KEY = None
ACH_GOAL_KEY = None

# Launch the simulator in headless mode
# app_launcher = AppLauncher(headless=True, device=DEVICE)
# simulation_app = app_launcher.app

# from isaaclab.envs import ManagerBasedRLEnv
# 
# from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

# Load config and customize (e.g., more envs for parallel training)
# cfg = CartpoleEnvCfg().to_dict()
# print(cfg)
# cfg.scene.num_envs = NUM_ENVS  # Scale for faster training (GPU handles it)
# cfg.sim.device = DEVICE
# cfg.seed = 42

cfg = 'isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg:CartpoleEnvCfg'

wrappers = [
    {
        "type": "VectorNStepReward",
        "params": {
            "n": N,
            "obs_key": OBS_KEY,
            "goal_key": GOAL_KEY,
            "ach_goal_key": ACH_GOAL_KEY,
        }
    }
]

# Create the env
# env = ManagerBasedRLEnv(cfg=cfg)
env = IsaacSimWrapper(cfg=cfg, num_envs=NUM_ENVS, wrappers=wrappers, render_mode=RENDER_MODE, seed=SEED, obs_key=OBS_KEY, goal_key=GOAL_KEY)


# Your PhoenX RL agent (pseudo-code - replace with your class)
# from your_phoenx_api import YourAgent  # Adjust to your module/agent class

# agent = YourAgent(
#     state_dim=env.observation_space.shape[0],
#     action_dim=env.action_space.shape[0],
#     device="cuda:0"
# )  # e.g., SAC, PPO, etc.

# Simple training loop (adapt to your API's train method)
# num_episodes = 1000
# obs, _ = env.reset()
# print(f'observation space: {env.observation_space}')
# print(f'observation space shape: {env.observation_space.shape}')
# print(f'observation: {obs}')
# print(f'action space: {env.action_space}')
# print(f'action space shape: {env.action_space.sample().shape}')
# print(f'action: {env.action_space.sample()}')
# episode_rewards = torch.zeros(NUM_ENVS, device=DEVICE)
# completed_episodes = torch.zeros(NUM_ENVS, device=DEVICE)

# while completed_episodes.sum() < NUM_EPISODES:
#     # done = False
#     # while not done:
#     # action = agent.act(obs)  # Your agent's policy
#     action = env.action_space.sample()
#     action = torch.Tensor(action)
#     states, rewards, dones, info = env.step(action)
#     #DEBUG
#     # print(f'reward: {reward}')
#     # print(f'episode_rewards: {episode_rewards}')
#     episode_rewards += rewards
#     # dones = torch.logical_or(terminated, truncated)
#     done_episodes = torch.nonzero(dones)
#     for episode in done_episodes:
#         completed_episodes[episode] += 1
#         print(f"Episode {completed_episodes.sum()}: Reward = {episode_rewards[episode].item()}")
#         episode_rewards[episode] = 0
#     obs = states

# # Close the env and app
# env.close()

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

actor = ActorModel(env, layer_config, output_layer_config, optimizer_params=actor_optimizer, obs_key=OBS_KEY, goal_key=GOAL_KEY, device=DEVICE)

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

critic = CriticModel(env, state_layers=state_layer_config, merged_layers=merged_layer_config,
                    output_layer_kernel=output_layer_config, optimizer_params=critic_optimizer, obs_key=OBS_KEY, goal_key=GOAL_KEY, device=DEVICE)

replay_buffer = ReplayBuffer(env, 1000000, obs_key=OBS_KEY, goal_key=GOAL_KEY, N=N, device=DEVICE)
# replay_buffer = PrioritizedReplayBuffer(env_wrap, 100000, alpha=0.6, beta_start=0.4, beta_iter=10000, beta_update_freq=1, priority='rank',normalize=False, epsilon=0.01, N=N, device='cpu')
noise = NormalNoise(stddev=0.1, device=DEVICE)
state_normalizer = Normalizer(size=env.single_observation_space[OBS_KEY].shape, update_freq=200, clip_range=5.0, device=DEVICE)

ddpg_agent = DDPG(
                env=env,
                actor_model=actor,
                critic_model=critic,
                replay_buffer=replay_buffer,
                discount=0.99,
                tau=0.005,
                action_epsilon=0.2,
                batch_size=128,
                noise=noise,
                grad_clip=40.0,
                warmup=100,
                N=N,
                state_normalizer=state_normalizer,
                obs_key=OBS_KEY,
                goal_key=GOAL_KEY,
                callbacks=[rl_callbacks.WandbCallback('Isaac_Cartpole-v1')],
                save_dir='Isaac_Cartpole_N3',
                device=DEVICE,
                log_level='info')

ddpg_agent.save()

config = ddpg_agent.get_config()

# Set train config and path
train_config = {
    'num_episodes': 100,
    'steps_per_learn': 1,
    'seed': 42,
}
train_config_path = config["save_dir"] + 'train_config.json'
with open(train_config_path, 'w') as f:
    json.dump(train_config, f)

ddpg_agent.train(**train_config)