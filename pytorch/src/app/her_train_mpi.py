import sys
import json
import logging

from rl_agents import HER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_agent(config):
    try:
        agent_type = config['agent_type']
        load_weights = config['load_weights']
        num_epochs = config['num_epochs']
        num_cycles = config['num_cycles']
        num_episodes = config['num_episodes']
        num_updates = config['num_updates']
        render = config['render']
        render_freq = config['render_freq']
        save_dir = config['save_dir']

        assert agent_type == 'HER', f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = HER.load(config, load_weights)
            agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir)

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
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)

            train_agent(config)

        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in configuration file: {config_path}")

    else:
        logging.error("Configuration file path not provided.")