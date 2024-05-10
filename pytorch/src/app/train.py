import sys
import json
import logging
import subprocess

from rl_agents import load_agent_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_agent(config):
    try:
        agent_type = config['agent_type']
        load_weights = config['load_weights']
        num_episodes = config['num_episodes']
        render = config['render']
        render_freq = config['render_freq']
        save_dir = config['save_dir']
        # MPI flag
        use_mpi = config.get('use_mpi', False)

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER'], f"Unsupported agent type: {agent_type}"

        if agent_type:
            agent = load_agent_from_config(config, load_weights)

        if agent_type == 'HER':

            if use_mpi:
                num_workers = config['num_workers']
                # Execute the MPI command for HER agent
                mpi_command = f"mpirun -np {num_workers} python her_train_mpi.py {sys.argv[1]}"
                subprocess.run(mpi_command, shell=True, check=True)
            
            else:
                num_epochs = config['num_epochs']
                num_cycles = config['num_cycles']
                num_updates = config['num_updates']
                agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir)
        
        else:

            if use_mpi and agent_type == 'DDPG':
                num_workers = config['num_workers']
                mpi_command = f"mpirun -np {num_workers} python train_ddpg.py {sys.argv[1]}"
                subprocess.run(mpi_command, shell=True, check=True)
            
            else:
                agent.train(num_episodes, render, render_freq, save_dir)

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