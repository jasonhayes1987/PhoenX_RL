import sys
import json
import time
import logging
import argparse
import subprocess

import random
import numpy as np
import torch as T

from rl_agents import load_agent_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument('--agent_config', type=str, required=True, help='Path to the agent configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the train configuration file')

args = parser.parse_args()

agent_config_path = args.agent_config
train_config_path = args.train_config

def train_agent(agent_config, train_config):
    try:
        
        agent_type = agent_config['agent_type']
        load_weights = train_config.get('load_weights', False)
        num_episodes = train_config['num_episodes']
        render = train_config.get('render', False)
        render_freq = train_config.get('render_freq', 0)
        save_dir = train_config.get('save_dir', agent_config['save_dir'])
        #DEBUG
        print(f'training save dir: {save_dir}')
        seed = train_config['seed']
        run_number = train_config['run_number']
        num_runs = train_config.get('num_runs', 1)

        # MPI flag
        use_mpi = train_config.get('use_mpi', False)

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed(seed)

        print(f'seed: {seed}')

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = load_agent_from_config(agent_config, load_weights)
            print('agent config loaded')

            if agent_type == 'HER':

                if use_mpi:
                    num_workers = train_config['num_workers']
                    # Execute the MPI command for HER agent
                    mpi_command = f"mpirun -np {num_workers} python train_her_mpi.py --agent_config {agent_config_path} --train_config {train_config_path}"
                    for i in range(num_runs):
                        subprocess.Popen(mpi_command, shell=True)
                        print(f'training run {i+1} initiated')
                        # time.sleep(5)
                
                else:
                    num_epochs = agent_config['num_epochs']
                    num_cycles = agent_config['num_cycles']
                    num_updates = agent_config['num_updates']
                    for i in range(num_runs):
                        agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir, run_number)
                        print(f'training run {i+1} initiated')
            
            else:

                if use_mpi and agent_type == 'DDPG':
                    num_workers = train_config['num_workers']
                    mpi_command = f"mpirun -np {num_workers} python train_ddpg_mpi.py --agent_config {agent_config_path} --train_config {train_config_path}"
                    for i in range(num_runs):
                        subprocess.Popen(mpi_command, shell=True)
                        print(f'training run {i+1} initiated')
                
                else:
                    for i in range(num_runs):
                        agent.train(num_episodes, render, render_freq)
                        print(f'training run {i+1} initiated')

    except KeyError as e:
        logging.error(f"Missing configuration parameter: {str(e)}")
        raise

    except AssertionError as e:
        logging.error(str(e))
        raise

    except Exception as e:
        logging.exception("An unexpected error occurred during training")
        raise

if __name__ == '__main__':
    try:
        with open(agent_config_path, 'r', encoding="utf-8") as f:
            agent_config = json.load(f)

        with open(train_config_path, 'r', encoding="utf-8") as f:
            train_config = json.load(f)

        train_agent(agent_config, train_config)

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {str(e)}")