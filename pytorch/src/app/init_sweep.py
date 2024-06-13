import argparse
import logging
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

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def main(sweep_config, train_config):
        logger.debug("init_sweep main fired")
        try:
            if MPI.COMM_WORLD.rank == 0:
                sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
                wandb.agent(
                    sweep_id,
                    function=lambda: init_sweep(sweep_config, train_config),
                    count=train_config['num_sweeps'],
                    project=sweep_config["project"],
                )

            # try:
            #      logger.debug("calling MPI Barrier")
            #      MPI.COMM_WORLD.Barrier()
            #      logger.debug('Barrier passed')
            # except Exception as e:
            #      logger.error(f"error calling MPI Barrier")
            
            if MPI.COMM_WORLD.rank > 0:
                logger.debug(f"if rank > 0 fired: rank {MPI.COMM_WORLD.rank}")
                init_sweep(sweep_config, train_config)
        except Exception as e:
            logger.error(f"error in init_sweep.py main process: {e}")

if __name__ == "__main__":
        
        try:
            logger.debug("init_sweep.py fired")
            sweep_config = load_config(sweep_config_path)
            logger.debug("sweep config loaded")
            train_config = load_config(train_config_path)
            logger.debug("train config loaded")
            main(sweep_config, train_config)
        except Exception as e:
              logger.error(f"error in init_sweep.py __main__ process: {e}")
