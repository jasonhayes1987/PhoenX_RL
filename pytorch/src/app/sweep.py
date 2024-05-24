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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Run WandB Sweep')
parser.add_argument('--sweep_config', type=str, required=True, help='Path to the WandB configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the run configuration file')

args = parser.parse_args()

sweep_config_path = args.sweep_config
train_config_path = args.train_config

def _run_sweep(sweep_config, episodes_per_sweep, epochs_per_sweep, cycles_per_sweep, updates_per_sweep):

    # max_retries = 3
    # retry_delay = 10  # seconds
    #DEBUG
    # print(f'_run_sweep fired...')
    # get next run number
    # for attempt in range(max_retries):
    try:
        run_number = get_next_run_number(sweep_config["project"])
        #DEBUG
        # print(f'run number: {run_number}')
        run = wandb.init(
            project=sweep_config["project"],
            settings=wandb.Settings(start_method='thread'),
            job_type="train",
            name=f"train-{run_number}",
            tags=["train"],
            group=f"group-{run_number}",
        )
        # run.tags = run.tags + (wandb.config.model_type,)
        #DEBUG
        # print(f"creating env { {param: value['value'] for param, value in sweep_config['parameters']['env']['parameters'].items()} }")
        env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
        #DEBUG
        # print(f'env spec: {env.spec}')

        # check for agent type since constructors are different
        if wandb.config.model_type == "Reinforce" or wandb.config.model_type == "Actor Critic":
            policy_layers, value_layers = build_layers(wandb.config)
            agent = get_agent_class_from_type(wandb.config.model_type)
            rl_agent = agent.build(
                env=env,
                policy_layers=policy_layers,
                value_layers=value_layers,
                callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
                config=wandb.config,
                save_dir=wandb.config.save_dir,
            )
        elif wandb.config.model_type == "DDPG":
            actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(wandb.config)
            agent = get_agent_class_from_type(wandb.config.model_type)
            rl_agent = agent.build(
                env=env,
                actor_cnn_layers = actor_cnn_layers,
                critic_cnn_layers = critic_cnn_layers,
                actor_layers=actor_layers,
                critic_state_layers=critic_state_layers,
                critic_merged_layers=critic_merged_layers,
                kernels=kernels,
                callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
                config=wandb.config,
                save_dir=wandb.config.save_dir,
            )

        elif wandb.config.model_type == "HER_DDPG":
            #DEBUG
            # print(f'passed wandb config: {wandb.config}')
            actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(wandb.config)
            #DEBUG
            # print('build layers output')
            # print(f'kernels: {kernels}')
            agent = get_agent_class_from_type(wandb.config.model_type)
            rl_agent = agent.build(
                env=env,
                actor_cnn_layers = actor_cnn_layers,
                critic_cnn_layers = critic_cnn_layers,
                actor_layers=actor_layers,
                critic_state_layers=critic_state_layers,
                critic_merged_layers=critic_merged_layers,
                kernels=kernels,
                callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
                config=wandb.config,
            )

            #DEBUG
            # print(f'HER AGENT config: {rl_agent.get_config()}')
        
        rl_agent.save()

        agent_config_path = rl_agent.save_dir + '/config.json'
        train_config_path = os.path.join(os.getcwd(), 'sweep/train_config.json')
        # Import train config to add run number
        with open(train_config_path, 'r') as file:
            train_config = json.load(file)
        train_config['run_number'] = run_number
        # Save updated train config
        with open(train_config_path, 'w') as file:
            json.dump(train_config, file)

        run_command = f"python train.py --agent_config {agent_config_path} --train_config {train_config_path}"
        subprocess.Popen(run_command, shell=True)
        # break # exit loop if success
    except Exception as e:
        logging.error(f"Failed to start sweep run: {str(e)}")
        # time.sleep(retry_delay)
    # else:
    #     logging.error("Failed to start sweep run after multiple attempts.")

if __name__ == '__main__':
    try:
        with open(sweep_config_path, 'r', encoding="utf-8") as f:
            sweep_config = json.load(f)

        with open(train_config_path, 'r', encoding="utf-8") as f:
            train_config = json.load(f)

        sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])

        wandb.agent(
            sweep_id,
            function=lambda: _run_sweep(
                sweep_config,
                train_config["num_episodes"],
                train_config["num_epochs"],
                train_config["num_cycles"],
                train_config["num_updates"],
            ),
            count=train_config["num_sweeps"],
            project=sweep_config["project"],
        )

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {str(e)}")