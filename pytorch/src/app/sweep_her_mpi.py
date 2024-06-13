
import os
import json
import logging
import argparse

import random
import numpy as np
import torch as T
import wandb
from mpi4py import MPI

from rl_agents import load_agent_from_config

parser = argparse.ArgumentParser(description='Sweep MPI')
parser.add_argument('--agent_config', type=str, required=True, help='Path to agent_config.json to load agent')
parser.add_argument('--train_config', type=str, required=True, help='Path to train_config.json to set training params')

args = parser.parse_args()

agent_config_path = args.agent_config
train_config_path = args.train_config

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def train(agent_config, train_config):
    try:
        logger.debug('mpi sweep train fired')

        # Load agent
        try:
            rl_agent = load_agent_from_config(agent_config)
            logger.debug('mpi sweep rl agent loaded')
        except Exception as e:
            logger.error(f"Error loading agent from config: {e}")
            raise

        # Set seeds
        try:
            random.seed(train_config['seed'])
            np.random.seed(train_config['seed'])
            T.manual_seed(train_config['seed'])
            T.cuda.manual_seed(train_config['seed'])
            logger.debug('mpi sweep seeds set')
        except Exception as e:
            logger.error(f"Error setting seeds: {e}")
            raise

        # Initialize wandb
        # try:
        #     if MPI.COMM_WORLD.Get_rank() == 0:
        #         wandb.init()
        #         logger.debug('mpi sweep wandb init called')
        # except Exception as e:
        #     logger.error(f"Error initializing wandb: {e}")
        #     raise

        # Train agent
        try:
            rl_agent.train(
                num_epochs=train_config['num_epochs'],
                num_cycles=train_config['num_cycles'],
                num_episodes=train_config['num_episodes'],
                num_updates=train_config['num_updates'],
                render=False,
                render_freq=0
            )
            logger.debug('mpi sweep training completed')
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    except Exception as e:
        logger.error(f"General error in train function: {e}")
        raise

        
if __name__ == "__main__":

    try:
        agent_config = load_config(agent_config_path)
        train_config = load_config(train_config_path)

        # Set the environment variable
        os.environ['WANDB_DISABLE_SERVICE'] = 'true'

        train(agent_config, train_config)

    except KeyError as e:
        logger.error(f"KeyError in W&B stream handling: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")