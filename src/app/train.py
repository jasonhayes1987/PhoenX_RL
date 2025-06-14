import sys
import json
import time
from logging_config import get_logger
import argparse
import subprocess
import ray
import random
import numpy as np
import torch as T
import wandb

from rl_agents import load_agent_from_config
from distributed_trainer import DistributedAgents

# Configure logging
logger = get_logger(__name__, 'info')

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument('--agent_config', type=str, required=True, help='Path to the agent configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the train configuration file')
parser.add_argument('--distributed_workers', type=int, default=1, help='Number of distributed workers (default: 1)')
parser.add_argument('--learner_device', type=str, default=None, help='Device for the learner (default: None)')
parser.add_argument('--learner_num_cpus', type=int, default=1, help='Number of CPUs for the learner (default: 1)')
parser.add_argument('--learner_num_gpus', type=float, default=1.0, help='Number of GPUs for the learner (default: 1)')
parser.add_argument('--worker_device', type=str, default='cpu', help='Device for the workers (default: cpu)')
parser.add_argument('--worker_num_cpus', type=int, default=1, help='Number of CPUs for the workers (default: 1)')
parser.add_argument('--worker_num_gpus', type=float, default=0.0, help='Number of GPUs for the workers (default: 0)')
parser.add_argument('--learn_iter', type=int, default=100, help='Learn frequency for the distributed workers (default: 100)')
parser.add_argument('--sync_iter', type=int, default=10, help='Sync interval for the distributed workers (default: 10)')

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
        
        # Use a specific seed if provided, otherwise generate a deterministic one based on current time
        # This ensures reproducibility while still giving different runs different seeds
        if 'seed' not in train_config:
            # Use current timestamp as seed for reproducible randomness
            seed = int(time.time()) % 10000
            logger.info(f"No seed provided, using generated seed: {seed}")
        else:
            seed = train_config['seed']
            logger.info(f"Using provided seed: {seed}")
        
        run_number = train_config.get('run_number', None)
        num_episodes = train_config['num_episodes']

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'TD3', 'HER', 'PPO'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            if agent_type in ['ActorCritic', 'DDPG', 'TD3']:
                if args.distributed_workers > 1:
                    distributed_agents = DistributedAgents(
                        agent_config,
                        args.distributed_workers,
                        args.learner_device,
                        args.learner_num_cpus,
                        args.learner_num_gpus,
                        args.worker_device,
                        args.worker_num_cpus,
                        args.worker_num_gpus,
                        args.learn_iter,
                    )
                    futures = distributed_agents.train(sync_iter=args.sync_iter, num_episodes=num_episodes, num_envs=num_envs, seed=seed, render_freq=render_freq)
                    if futures:
                        ray.get(futures)
                else:
                    agent = load_agent_from_config(agent_config, load_weights)
                    agent.train(num_episodes, num_envs, seed, render_freq)

            elif agent_type == 'Reinforce':
                trajectories_per_update = train_config['trajectories_per_update']
                agent.train(num_episodes, num_envs, trajectories_per_update, seed, render_freq)

            elif agent_type == 'HER':
                num_epochs = train_config['num_epochs']
                num_cycles = train_config['num_cycles']
                num_updates = train_config['num_epochs']
                if args.distributed_workers > 1:
                    distributed_agents = DistributedAgents(
                        agent_config,
                        args.distributed_workers,
                        args.learner_device,
                        args.learner_num_cpus,
                        args.learner_num_gpus,
                        args.worker_device,
                        args.worker_num_cpus,
                        args.worker_num_gpus,
                        args.learn_iter
                    )
                    futures = distributed_agents.train(
                        sync_iter=args.sync_iter,
                        num_epochs=num_epochs,
                        num_cycles=num_cycles,
                        num_episodes=num_episodes,
                        num_updates=num_updates,
                        render_freq=render_freq,
                        num_envs=num_envs,
                        seed=seed
                    )
                    if futures:
                        ray.get(futures)
                else:
                    agent = load_agent_from_config(agent_config, load_weights)
                    agent.train(num_epochs, num_cycles, num_episodes, num_updates, render_freq, num_envs, seed)
            
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