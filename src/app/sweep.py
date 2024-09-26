import os
from pathlib import Path
import argparse
import json
import logging
import multiprocessing
import subprocess
import shutil
import uuid
import time

import wandb
import gymnasium as gym
import numpy as np
import torch as T
# from dash_callbacks import run_agent
from wandb_support import get_next_run_number, build_layers
from rl_agents import init_sweep

print('sweep.py called')

# Initialize parser
parser = argparse.ArgumentParser(description='Sweep MPI')
parser.add_argument('--sweep_config', type=str, required=True, help='Path to sweep_config.json to load agent')
parser.add_argument('--num_sweeps', type=str, required=True, help='Number of sweeps to perform')
args = parser.parse_args()
sweep_config_path = args.sweep_config
num_sweeps = args.num_sweeps
print(f'num sweep = {num_sweeps}')
print(f'num sweeps type is {type(num_sweeps)}')
if num_sweeps == 0:
    num_sweeps = None

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

args = parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def main(sweep_config, num_sweeps):
    print('main fired...')
    try:
        # config_file_path = 'sweep/sweep_config.json'
        # with open(config_file_path, 'r') as file:
        #     sweep_config = json.load(file)

        # config_file_path = 'sweep/train_config.json'
        # with open(config_file_path, 'r') as file:
        #     train_config = json.load(file)
        print('attempting to create sweep id...')
        sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
        print('sweep id created')

        # num_sweep_agents = train_config['num_agents'] if train_config['num_agents'] is not None else 1
        # print(f'num sweep agents:{num_sweep_agents}')

        # if num_sweep_agents > 1:
        #     processes = []
        #     for agent in range(num_sweep_agents):
        #         p = multiprocessing.Process(target=run_agent, args=(sweep_id, sweep_config, train_config))
        #         p.start()
        #         processes.append(p)

        #     for p in processes:
        #         p.join()
        
        # else:
        run_agent(sweep_id, sweep_config, num_sweeps)

    except KeyError as e:
        logger.error(f"KeyError in W&B stream handling: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

def run_agent(sweep_id, sweep_config, num_sweeps):
    print('run agent fired...')
    wandb.agent(
        sweep_id,
        function=lambda: init_sweep(sweep_config),
        count=num_sweeps,
        project=sweep_config["project"],
    )

# def run_sweep(sweep_config):
#     print('run sweep fired...')

#     try:
#         init_sweep(sweep_config)

#         # if train_config['use_mpi']:
#         #     print('sweep test use mpi fired')
#         #     sweep_config_path = 'sweep/sweep_config.json'
#         #     train_config_path = 'sweep/train_config.json'
#         #     # Construct absolute paths
#         #     # base_dir = Path(__file__).resolve().parent
#         #     # sweep_config_path = base_dir / 'sweep' / 'sweep_config.json'
#         #     # train_config_path = base_dir / 'sweep' / 'train_config.json'
            
#         #     # Ensure the paths exist
#         #     # if not sweep_config_path.exists():
#         #     #     logger.error(f"Sweep config path does not exist: {sweep_config_path}")
#         #     # if not train_config_path.exists():
#         #     #     logger.error(f"Train config path does not exist: {train_config_path}")

#         #     # Find the path to mpirun
#         #     # mpi_path = shutil.which("mpirun")
#         #     # if not mpi_path:
#         #     #     logger.error("mpirun not found in PATH")
#         #     # else:
#         #     #     logger.debug(f"mpirun found at: {mpi_path}")

#         #     # mpi_command = [
#         #     #     mpi_path,
#         #     #     "-np", str(train_config['num_workers']),
#         #     #     "python", str(base_dir / "init_sweep.py"),
#         #     #     "--sweep_config", str(sweep_config_path),
#         #     #     "--train_config", str(train_config_path)
#         #     # ]
#         #     # mpi_command = f"{mpi_path} -np {str(train_config['num_workers'])} python {str(base_dir / 'init_sweep.py')} --sweep_config {str(sweep_config_path)} --train_config {str(train_config_path)}"
        
#         #     try:
#         #         # logger.debug(f"Running MPI command: {' '.join(mpi_command)}")

#         #         # mpi_process = subprocess.Popen(mpi_command, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
#         #         command = (
#         #             "mpiexec -np "
#         #             + str(train_config['num_workers'])
#         #             + " python init_sweep.py "
#         #             + "--sweep_config "
#         #             + str(sweep_config_path)
#         #             + " --train_config "
#         #             + str(train_config_path)
#         #         )

#         #         logger.debug(f"Running command: {command}")

#         #         result = subprocess.run(
#         #             command,
#         #             check=True,
#         #             stderr=subprocess.PIPE,
#         #             stdout=subprocess.PIPE,
#         #             universal_newlines=True,
#         #             shell=True
#         #         )

#         #         logger.debug("Standard Output:")
#         #         logger.debug(result.stdout)

#         #         logger.debug("Standard Error:")
#         #         logger.debug(result.stderr)

#         #     except subprocess.CalledProcessError as e:
#         #         logger.error(f"Subprocess failed with return code {e.returncode}")
#         #         logger.error(f"Standard Output: {e.stdout}")
#         #         logger.error(f"Standard Error: {e.stderr}")
#         #     except Exception as e:
#         #         logger.error(f"Error during subprocess execution: {str(e)}")

#         # else:
#             # init_sweep(sweep_config)

#     except Exception as e:
#         logging.error(f"Error during sweep run attempt: {str(e)}")

if __name__ == "__main__":
    try:
        # Set the environment variable
        # os.environ['WANDB_DISABLE_SERVICE'] = 'true'
        logger.debug("sweep.py fired")
        sweep_config = load_config(sweep_config_path)
        print(f'sweep config: {sweep_config}')
        logger.debug("sweep config loaded")
        main(sweep_config, num_sweeps)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
