import argparse
# import logging
from logging_config import logger
import json
import wandb
from mpi4py import MPI
import mpi_helper

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

    # group_size = size // num_agents
    group_size = mpi_helper.set_group_size(MPI.COMM_WORLD, num_agents)
    # group = rank // group_size
    group = mpi_helper.set_group(MPI.COMM_WORLD, group_size)
    
    if num_agents > 1:
        comm = MPI.COMM_WORLD.Split(color=group, key=rank)
        comm.Set_name(f"Group_{group}")
        new_rank = comm.Get_rank()

        logger.debug(f"Global rank {rank} assigned to {comm.Get_name()} with new rank {new_rank}")
    
    else:
        comm = MPI.COMM_WORLD
        new_rank = rank
        comm.Set_name(f"Group_{group}")
        
        logger.debug(f"Global rank {rank} assigned to {comm.Get_name()}")

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
            wandb.agent(
                sweep_id,
                function=lambda: init_sweep(sweep_config, train_config, comm),
                count=train_config['num_sweeps'],
                project=sweep_config["project"],
            )
        except Exception as e:
            logger.error(f"error in init_sweep.py main process: {e}", exc_info=True)
    else:
        try:
            for _ in range(train_config['num_sweeps']):
                init_sweep(sweep_config, train_config, comm)
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
