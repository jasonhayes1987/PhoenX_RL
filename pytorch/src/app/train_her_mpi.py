import sys
import json
import logging
import argparse

from rl_agents import HER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    try:
        agent_type = agent_config['agent_type']
        load_weights = train_config.get('load_weights', False)
        num_epochs = train_config.get('num_epochs', None)
        num_cycles = train_config.get('num_cycles', None)
        num_episodes = train_config['num_episodes']
        num_updates = train_config.get('num_updates', None)
        render = train_config.get('render', False)
        render_freq = train_config.get('render_freq', 0)
        save_dir = train_config.get('save_dir', agent_config['save_dir'])
        run_number = train_config.get('run_number', None)

        assert agent_type == 'HER', f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = HER.load(agent_config, load_weights)
            agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir, run_number)

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
        agent_config = load_config(agent_config_path)
        train_config = load_config(train_config_path)

        train_agent(agent_config, train_config)

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {str(e)}")