import argparse
# import logging
from logging_config import logger
import json
import wandb
from mpi4py import MPI

from rl_agents import init_sweep

parser = argparse.ArgumentParser(description='Sweep MPI')
parser.add_argument('--sweep_config', type=str, required=True, help='Path to sweep_config.json to load agent')
parser.add_argument('--train_config', type=str, required=True, help='Path to train_config.json to set training params')

args = parser.parse_args()

sweep_config_path = args.sweep_config
train_config_path = args.train_config

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def main(sweep_config, train_config):
    logger.debug("init_sweep main fired")
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    num_agents = train_config['num_agents']
    assert size % num_agents == 0, "Number of workers must be divisible by number of agents."

    group_size = size // num_agents
    color = rank // group_size

    new_comm = MPI.COMM_WORLD.Split(color=color, key=rank)
    new_rank = new_comm.Get_rank()

    logger.debug(f"Global rank {rank} assigned to group {color} with new rank {new_rank}")

    if rank == 0:
        try:
            sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
            logger.debug(f"Sweep ID: {sweep_id}")
        except Exception as e:
            logger.error(f"error creating sweep ID: {e}", exc_info=True)
            sweep_id = None
    else:
        sweep_id = None

    # Broadcast the sweep ID from rank 0 of COMM_WORLD to all other ranks
    sweep_id = MPI.COMM_WORLD.bcast(sweep_id, root=0)
    logger.debug(f"Rank {rank} received Sweep ID: {sweep_id}")

    if new_rank == 0:
        try:
            sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
            logger.debug(f"Sweep ID: {sweep_id} for group {color}")
            wandb.agent(
                sweep_id,
                function=lambda: init_sweep(sweep_config, train_config, new_comm),
                count=train_config['num_sweeps'],
                project=sweep_config["project"],
            )
        except Exception as e:
            logger.error(f"error in init_sweep.py main process: {e}", exc_info=True)
    else:
        try:
            for _ in range(train_config['num_sweeps']):
                init_sweep(sweep_config, train_config, new_comm)
        except Exception as e:
            logger.error(f"error in init_sweep.py main process: {e}", exc_info=True)


if __name__ == "__main__":
        
        try:
            logger.debug("init_sweep.py fired")
            sweep_config = load_config(sweep_config_path)
            logger.debug("sweep config loaded")
            train_config = load_config(train_config_path)
            logger.debug("train config loaded")
            main(sweep_config, train_config)
            MPI.Finalize()
        except Exception as e:
              logger.error(f"error in init_sweep.py __main__ process: {e}", exc_info=True)
