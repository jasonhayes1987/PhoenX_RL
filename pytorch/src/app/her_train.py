import os
import random
import numpy as np
import torch as T
from mpi4py import MPI


import gymnasium as gym
import gymnasium_robotics as gym_robo
from rl_agents import HER, DDPG
from models import ActorModel, CriticModel
from helper import MPIHelper, ReplayBuffer, NormalNoise
from gym_helper import get_her_goal_functions
from rl_callbacks import WandbCallback

def main():
    # Set device
    device = 'cpu'

    # mpi_helper = MPIHelper()

    # Create gym environment
    env_id = "FetchPickAndPlace-v2"
    env = gym.make(env_id)

    # Get reward functions for environment
    desired_goal_func, achieved_goal_func, reward_func = get_her_goal_functions(env)

    # Get goal shape (requires gym.reset() to initialize env state)
    _,_ = env.reset()
    goal_shape = desired_goal_func(env).shape

    # Build actor model
    dense_layers = [
        (
            64,
            "relu",
            {
                "kaiming uniform": {
                }
            },
        ),
        (
            64,
            "relu",
            {
                "kaiming uniform": {\
                }
            },
        ),
        (
            64,
            "relu",
            {
                "kaiming uniform": {
                }
            },
        )
    ]

    actor = ActorModel(env, cnn_model=None, dense_layers=dense_layers, goal_shape=goal_shape,
                       optimizer='Adam', optimizer_params={'weight_decay':0.0},
                       learning_rate=0.00001, normalize_layers=False, device=device)

    # build critic
    state_layers = [
    ]

    merged_layers = [
        (
            64,
            "relu",
            {
                "kaiming uniform": {
                }
            },
        ),
        (
            64,
            "relu",
            {
                "kaiming uniform": {
                }
            },
        ),
        (
            64,
            "relu",
            {
                "kaiming uniform": {
                }
            },
        ),
    ]

    critic = CriticModel(env=env, cnn_model=None, state_layers=state_layers,
                         merged_layers=merged_layers, goal_shape=goal_shape,
                         optimizer="Adam", optimizer_params={'weight_decay':0.0},
                         learning_rate=0.00001, normalize_layers=False, device=device)

    # Instantiate replay buffer
    replay_buffer = ReplayBuffer(env, 100000, goal_shape)
    # Instantiate noise object
    noise = NormalNoise(shape=env.action_space.shape, mean=0.0, stddev=0.05, device=device)

    # Create DDPG agent
    ddpg_agent = DDPG(env=env,
                      actor_model=actor,
                      critic_model=critic,
                      discount=0.98,
                      tau=0.05,
                      action_epsilon=0.3,
                      replay_buffer=replay_buffer,
                      batch_size=128,
                      noise=noise,
                      callbacks=[WandbCallback("Fetch_Pick_Place_v2")],
                      save_dir="fetch_pick_place_v2/models/ddpg/"
                      )

    # Instantiate HER object
    her = HER(agent=ddpg_agent,
              strategy='future',
              tolerance=0.05,
              num_goals=4,
              desired_goal=desired_goal_func,
              achieved_goal=achieved_goal_func,
              reward_fn=reward_func,
              normalizer_clip=5.0,
              save_dir="fetch_pick_place_v2/models/her/"
            )

    # Run the training process
    her.train(epochs=200,
              num_cycles=50,
              num_episodes=16,
              num_updates=40,
              render=True,
              render_freq=1000)

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main()