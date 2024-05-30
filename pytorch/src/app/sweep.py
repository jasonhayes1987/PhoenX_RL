import sys
import os
import json
import time
import logging
import argparse
import subprocess

import random
import numpy as np
import torch as T
import wandb
import gymnasium as gym

from rl_agents import get_agent_class_from_type
from wandb_support import get_next_run_number, build_layers
from rl_callbacks import WandbCallback

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def train():
    config = wandb.config

    # load train config
    train_config_path = "sweep/train_config.json"
    train_config = load_config(train_config_path)

    agent_type = config.model_type
    load_weights = train_config.get('load_weights', False)
    num_episodes = train_config['num_episodes']
    render = train_config.get('render', False)
    render_freq = train_config.get('render_freq', 0)
    save_dir = train_config.get('save_dir', config.save_dir)
    seed = train_config['seed']
    run_number = train_config.get('run_number', None)
    use_mpi = train_config.get('use_mpi', False)

    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)

    assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER'], f"Unsupported agent type: {agent_type}"

    callbacks = []
    if wandb.run:
        callbacks.append(WandbCallback(project_name=wandb.run.project, run=wandb.run))

    env = gym.make(config.env_name)
    if agent_type == 'HER':
        actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(config)
        agent_class = get_agent_class_from_type(agent_type)
        rl_agent = agent_class.build(
            env=env,
            actor_cnn_layers=actor_cnn_layers,
            critic_cnn_layers=critic_cnn_layers,
            actor_layers=actor_layers,
            critic_state_layers=critic_state_layers,
            critic_merged_layers=critic_merged_layers,
            kernels=kernels,
            callbacks=callbacks,
            config=config,
        )

        if use_mpi:
            agent_config_path = rl_agent.save_dir + '/config.json'
            num_workers = train_config['num_workers']
            mpi_command = f"mpirun -np {num_workers} python train_her_mpi.py --agent_config {agent_config_path} --train_config {train_config_path}"
            subprocess.Popen(mpi_command, shell=True)
        else:
            num_epochs = train_config['num_epochs']
            num_cycles = train_config['num_cycles']
            num_updates = train_config['num_updates']
            rl_agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir, run_number)
    else:
        # Add similar blocks for other agent types here
        pass

if __name__ == "__main__":
    sweep_config_path = "sweep/sweep_config.json"
    train_config_path = "sweep/train_config.json"

    sweep_config = load_config(sweep_config_path)
    train_config = load_config(train_config_path)

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])

    for _ in range(train_config['num_agents']):
        wandb.agent(sweep_id=sweep_id, function=train)