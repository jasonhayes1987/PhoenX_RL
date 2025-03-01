import sys
import json
import time
from logging_config import logger
import argparse
import subprocess

import random
import numpy as np
import torch as T
import wandb

from rl_agents import load_agent_from_config

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument('--agent_config', type=str, required=True, help='Path to the agent configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the train configuration file')

args = parser.parse_args()

agent_config_path = args.agent_config
train_config_path = args.train_config

def train_agent(agent_config, train_config):

    # wandb_initialized = False  # Track if wandb is initialized
    try:
        agent_type = agent_config['agent_type']
        print(f'agent type:{agent_type}')
        # load_weights = train_config.get('load_weights', False)
        load_weights = train_config.get('load_weights', False)
        # render = train_config.get('render', False)
        render_freq = train_config.get('render_freq', 0)
        save_dir = train_config.get('save_dir', agent_config['save_dir'])
        #DEBUG
        # print(f'training save dir: {save_dir}')
        num_envs = train_config['num_envs']
        seed = train_config.get('seed', np.random.randint(1000))
        run_number = train_config.get('run_number', None)
        # num_runs = train_config.get('num_runs', 1)

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'TD3', 'HER', 'PPO'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = load_agent_from_config(agent_config, load_weights)
            # print('agent config loaded')
            # print(f'env:{agent.env.spec}')

            if agent_type == 'Reinforce':
                agent.train(train_config['num_episodes'], num_envs, train_config['batch_size'], seed, render_freq, save_dir)

            elif agent_type == 'ActorCritic':
                agent.train(train_config['num_episodes'], num_envs, seed, render_freq, save_dir)


            elif agent_type == 'HER':
                num_episodes = train_config['num_episodes']
                num_epochs = train_config['num_epochs']
                num_cycles = train_config['num_cycles']
                num_updates = train_config['num_updates']
                # for i in range(num_runs):
                agent.train(num_epochs, num_cycles, num_episodes, num_updates, render_freq, save_dir, run_number)
                # print(f'training run {i+1} initiated')
            
            elif agent_type == 'DDPG':
                num_episodes = train_config['num_episodes']
                agent.train(num_episodes, num_envs, seed, render_freq)

            elif agent_type == 'TD3':
                num_episodes = train_config['num_episodes']
                agent.train(num_episodes, num_envs, seed, render_freq)
            
            elif agent_type == 'PPO':
                timesteps = train_config['num_timesteps']
                traj_length = train_config['traj_length']
                batch_size = train_config['batch_size']
                learning_epochs = train_config['learning_epochs']
                agent.train(timesteps, traj_length, batch_size, learning_epochs, num_envs, seed, 10, render_freq, run_number=run_number)

    except KeyError as e:
        logger.error(f"Missing configuration parameter: {str(e)}")
        raise

    except AssertionError as e:
        logger.error(str(e))
        raise

    except Exception as e:
        logger.exception("An unexpected error occurred during training")
        raise
    # finally:
    #     # Ensure the WandB run is properly finished if it was initialized
    #     if wandb_initialized:
    #         wandb.finish()
    #         logging.info("WandB run finished")

if __name__ == '__main__':
    try:
        with open(agent_config_path, 'r', encoding="utf-8") as f:
            agent_config = json.load(f)

        with open(train_config_path, 'r', encoding="utf-8") as f:
            train_config = json.load(f)

        train_agent(agent_config, train_config)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file: {str(e)}")