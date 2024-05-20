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
from wandb_support import hyperparameter_sweep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Run WandB Sweep')
parser.add_argument('--sweep_config', type=str, required=True, help='Path to the WandB configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the run configuration file')

args = parser.parse_args()

sweep_config_path = args.sweep_config
train_config_path = args.train_config


def sweep(sweep_config, train_config):
    try:

        
        hyperparameter_sweep(
            sweep_config,
            train_config['num_sweeps'],
            train_config['num_episodes'],
            train_config['num_epochs'],
            train_config['num_cycles'],
            train_config['num_updates'],
        )

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
        with open(sweep_config_path, 'r', encoding="utf-8") as f:
            sweep_config = json.load(f)

        with open(train_config_path, 'r', encoding="utf-8") as f:
            train_config = json.load(f)

        sweep(sweep_config, train_config)

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {str(e)}")