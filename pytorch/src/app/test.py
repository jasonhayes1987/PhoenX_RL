import sys
import json
import logging
import subprocess

import random
import numpy as np
import torch as T

from rl_agents import load_agent_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_agent(config):
    try:
        
        agent_type = config['agent_type']
        load_weights = config['load_weights']
        num_episodes = config['num_episodes']
        render = config['render']
        render_freq = config['render_freq']
        seed = config['seed']
        run_number = config['run_number']

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed(seed)

        print(f'seed: {seed}')

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = load_agent_from_config(config, load_weights)
            print('agent config loaded')
            agent.test(num_episodes, render, render_freq, run_number)

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

            test_agent(config)

        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in configuration file: {config_path}")

    else:
        logging.error("Configuration file path not provided.")