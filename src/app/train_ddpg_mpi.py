import sys
import json
from logging_config import logger
import argparse
from mpi4py import MPI

from rl_agents import DDPG

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument('--agent_config', type=str, required=True, help='Path to the agent configuration file')
parser.add_argument('--train_config', type=str, required=True, help='Path to the train configuration file')

args = parser.parse_args()

agent_config_path = args.agent_config
train_config_path = args.train_config

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def train_agent(agent_config, train_config):
    # print('mpi train agent fired')
    try:
        agent_type = agent_config['agent_type']
        load_weights = train_config.get('load_weights', False)
        num_episodes = train_config['num_episodes']
        render = train_config.get('render', False)
        render_freq = train_config.get('render_freq', 0)
        save_dir = train_config.get('save_dir', agent_config['save_dir'])
        run_number = train_config.get('run_number', None)

        assert agent_type == 'DDPG', f"Unsupported agent type: {agent_type}"

        if agent_type:
            #DEBUG
            # print(f'if agent passed in mpi')
            agent = DDPG.load(agent_config, load_weights)
            agent.use_mpi = True
            agent.comm = MPI.COMM_WORLD
            agent.rank = MPI.COMM_WORLD.Get_rank()
            logger.error(f'rank:{agent.rank}')
            # print(f'mpi agent built:{agent.get_config()}')
            agent.train(num_episodes, render, render_freq, save_dir, run_number)

    except KeyError as e:
        logger.error(f"Missing configuration parameter: {str(e)}")
        raise

    except AssertionError as e:
        logger.error(str(e))
        raise

    except Exception as e:
        logger.exception("An unexpected error occurred during training")
        raise

if __name__ == '__main__':
    # print('train_her_mpi fired')
    try:
        agent_config = load_config(agent_config_path)
        # print(f'mpi agent config loaded:{agent_config}')
        train_config = load_config(train_config_path)
        # print(f'mpi train config loaded:{train_config}')

        train_agent(agent_config, train_config)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file: {str(e)}")