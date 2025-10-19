import sys
from pathlib import Path
import json
import logging
import argparse
import subprocess

import numpy as np
import torch as T

from rl_agents import load_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser(description='Test Agent')
parser.add_argument('--agent_dir', type=str, required=True, help='Path to the agent configuration directory')

args = parser.parse_args()
agent_config_dir = args.agent_dir

def test_agent(agent_config_dir):
    try:
        agent_config = json.load(open(Path(agent_config_dir) / 'config.json'))
        test_config = json.load(open(Path(agent_config_dir) / 'test_config.json'))
        agent_type = agent_config['agent_type']
        load_weights = test_config.get('load_weights', True)
        num_episodes = test_config['num_episodes']
        num_envs = test_config['num_envs']
        render_freq = test_config.get('render_freq', 0)
        seed = test_config.get('seed', np.random.randint(1000))

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'TD3', 'HER', 'PPO', 'SAC'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = load_agent(agent_config_dir, load_weights)
            if agent_type in ['ActorCritic', 'DDPG', 'TD3', 'SAC', 'HER', 'PPO']:
                agent.test(num_episodes, num_envs, render_freq, seed)

    except KeyError as e:
        logging.error(f"Missing configuration parameter: {str(e)}")
        raise

    except AssertionError as e:
        logging.error(str(e))
        raise

    except Exception as e:
        logging.exception("An unexpected error occurred during testing")
        raise

if __name__ == '__main__':
    try:
        test_agent(agent_config_dir)

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {str(e)}")