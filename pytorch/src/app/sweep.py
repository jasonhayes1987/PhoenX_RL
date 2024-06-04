# import sys
# import os
# import json
# import time
# import logging
# import argparse
# import subprocess
# from multiprocessing import Process

# import random
# import numpy as np
# import torch as T
# import wandb
# import gymnasium as gym

# from rl_agents import get_agent_class_from_type
# from wandb_support import get_next_run_number, build_layers
# from rl_callbacks import WandbCallback

# def load_config(path):
#     with open(path, 'r', encoding="utf-8") as f:
#         return json.load(f)

# def train():

#     # load train config
#     train_config_path = "sweep/train_config.json"
#     train_config = load_config(train_config_path)

#     # load sweep config
#     sweep_config_path = "sweep/sweep_config.json"
#     sweep_config = load_config(sweep_config_path)

#     run_number = get_next_run_number(sweep_config["project"])

#     wandb.init(
#         project=sweep_config["project"],
#         settings=wandb.Settings(start_method='thread'),
#         job_type="train",
#         name=f"train-{run_number}",
#         tags=["train"],
#         group=f"group-{run_number}",
#     )

#     config = wandb.config
#     load_weights = train_config.get('load_weights', False)
#     num_episodes = train_config['num_episodes']
#     render = train_config.get('render', False)
#     render_freq = train_config.get('render_freq', 0)
#     save_dir = train_config.get('save_dir', config[config.model_type][f'{config.model_type}_save_dir'])
#     seed = train_config['seed']
#     run_number = train_config.get('run_number', None)
#     use_mpi = train_config.get('use_mpi', False)

#     random.seed(seed)
#     np.random.seed(seed)
#     T.manual_seed(seed)
#     T.cuda.manual_seed(seed)

#     assert config.model_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER_DDPG'], f"Unsupported agent type: {config.model_type}"

#     callbacks = []
#     if wandb.run:
#         callbacks.append(WandbCallback(project_name=wandb.run.project, run=wandb.run))

#     env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
#     if config.model_type == 'HER':
#         actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(config)
#         agent_class = get_agent_class_from_type(config.model_type)
#         rl_agent = agent_class.build(
#             env=env,
#             actor_cnn_layers=actor_cnn_layers,
#             critic_cnn_layers=critic_cnn_layers,
#             actor_layers=actor_layers,
#             critic_state_layers=critic_state_layers,
#             critic_merged_layers=critic_merged_layers,
#             kernels=kernels,
#             callbacks=callbacks,
#             config=config,
#         )

#         if use_mpi:
#             agent_config_path = rl_agent.save_dir + '/config.json'
#             num_workers = train_config['num_workers']
#             mpi_command = f"mpirun -np {num_workers} python train_her_mpi.py --agent_config {agent_config_path} --train_config {train_config_path}"
#             subprocess.Popen(mpi_command, env=os.environ.copy(), shell=True)
#         else:
#             num_epochs = train_config['num_epochs']
#             num_cycles = train_config['num_cycles']
#             num_updates = train_config['num_updates']
#             rl_agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir, run_number)
    
#     else:
#         # Add similar blocks for other agent types here
#         pass

# if __name__ == "__main__":
#     sweep_config_path = "sweep/sweep_config.json"
#     train_config_path = "sweep/train_config.json"

#     sweep_config = load_config(sweep_config_path)
#     train_config = load_config(train_config_path)

#     sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])

#     # for _ in range(train_config['num_agents']):
#     # wandb.agent(sweep_id, function=train)

#     wandb.agent(
#         sweep_id,
#         function=train,
#         count=train_config['num_sweeps'],
#         project=sweep_config["project"],
#     )


import sys
import os
import json
import time
import logging
import argparse
import subprocess
from multiprocessing import Process

import random
import numpy as np
import torch as T
import wandb
import gymnasium as gym

from rl_agents import get_agent_class_from_type
from wandb_support import get_next_run_number, build_layers
from rl_callbacks import WandbCallback

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def setup_wandb(sweep_config):
    try:
        run_number = get_next_run_number(sweep_config["project"])
        # logger.debug(f"Run number: {run_number}")

        wandb.init(
            project=sweep_config["project"],
            settings=wandb.Settings(start_method='thread'),
            job_type="train",
            name=f"train-{run_number}",
            tags=["train"],
            group=f"group-{run_number}",
        )
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}")

def train():
    # logger.debug("Entered train function")

    # load train config
    train_config_path = "sweep/train_config.json"
    train_config = load_config(train_config_path)
    # logger.debug(f"Loaded train config: {train_config}")

    # load sweep config
    sweep_config_path = "sweep/sweep_config.json"
    sweep_config = load_config(sweep_config_path)
    # logger.debug(f"Loaded sweep config: {sweep_config}")

    setup_wandb(sweep_config)

    config = wandb.config
    # logger.debug(f"Wandb config: {config}")

    load_weights = train_config.get('load_weights', False)
    num_episodes = train_config['num_episodes']
    render = train_config.get('render', False)
    render_freq = train_config.get('render_freq', 0)
    save_dir = train_config.get('save_dir', config[config.model_type][f'{config.model_type}_save_dir'])
    seed = train_config['seed']
    run_number = train_config.get('run_number', None)
    use_mpi = train_config.get('use_mpi', False)

    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)

    assert config.model_type in ['Reinforce', 'ActorCritic', 'DDPG', 'HER_DDPG'], f"Unsupported agent type: {config.model_type}"

    callbacks = []
    if wandb.run:
        # logger.debug("if wandb.run fired")
        callbacks.append(WandbCallback(project_name=sweep_config["project"], _sweep=True))

    env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
    # logger.debug(f"Environment created: {env}")

    if config.model_type == 'HER_DDPG':
        actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(config)
        agent_class = get_agent_class_from_type(config.model_type)
        rl_agent = agent_class.build(
            env=env,
            actor_cnn_layers=actor_cnn_layers,
            critic_cnn_layers=critic_cnn_layers,
            actor_layers=actor_layers,
            critic_state_layers=critic_state_layers,
            critic_merged_layers=critic_merged_layers,
            kernels=kernels,
            callbacks=callbacks,
            config=config,
        )
        # logger.debug("Agent built")

        if use_mpi:
            agent_config_path = rl_agent.save_dir + '/config.json'
            num_workers = train_config['num_workers']
            mpi_command = f"mpirun -np {num_workers} python train_her_mpi.py --agent_config {agent_config_path} --train_config {train_config_path}"
            # logger.debug(f"Running MPI command: {mpi_command}")
            mpi_process = subprocess.Popen(mpi_command, env=os.environ.copy(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = mpi_process.communicate()
            # logger.debug(f"MPI STDOUT: {stdout.decode()}")
            # logger.debug(f"MPI STDERR: {stderr.decode()}")
        else:
            num_epochs = train_config['num_epochs']
            num_cycles = train_config['num_cycles']
            num_updates = train_config['num_updates']
            # logger.debug("Starting single-process training")
            rl_agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq, save_dir, run_number)
    
    else:
        logger.debug(f"Unsupported model type: {config.model_type}")

if __name__ == "__main__":
    # logger.debug("Entered main")
    sweep_config_path = "sweep/sweep_config.json"
    train_config_path = "sweep/train_config.json"

    try:
        sweep_config = load_config(sweep_config_path)
        train_config = load_config(train_config_path)

        # logger.debug(f"Sweep config: {sweep_config}")
        # logger.debug(f"Train config: {train_config}")

        if sweep_config:
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
            # logger.debug(f"Sweep ID: {sweep_id}")

            wandb.agent(
                sweep_id,
                function=train,
                count=train_config['num_sweeps'],
                project=sweep_config["project"],
            )
        else:
            logger.error("Sweep configuration is None.")
    except KeyError as e:
        logger.error(f"KeyError in W&B stream handling: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


