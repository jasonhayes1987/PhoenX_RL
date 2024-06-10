
import os
import json
import logging

import random
import numpy as np
import torch as T
import wandb
import gymnasium as gym
from mpi4py import MPI

from rl_agents import get_agent_class_from_type
from wandb_support import get_next_run_number, build_layers
from rl_callbacks import WandbCallback

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def run_sweep(sweep_config, train_config):
    run_number = get_next_run_number(sweep_config["project"])
    print(f'run number:{run_number}')
    # Add run number to train config
    train_config['run_number'] = run_number
    #DEBUG
    # print(f'run number: {run_number}')
    run = wandb.init(
        project=sweep_config["project"],
        settings=wandb.Settings(start_method='thread'),
        job_type="train",
        name=f"train-{run_number}",
        tags=["train"],
        group=f"group-{run_number}",
    )
    print('wandb init called')
    print(f'wandb config:{wandb.config}')
    run.tags = run.tags + (wandb.config.model_type,)
    #DEBUG
    # print(f"creating env { {param: value['value'] for param, value in sweep_config['parameters']['env']['parameters'].items()} }")
    env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
    #DEBUG
    print(f'env spec: {env.spec}')
    # print(f"train config save dir:{train_config.get('save_dir', sweep_config[wandb.config.model_type][f'{wandb.config.model_type}_save_dir'])}")
    save_dir = sweep_config[wandb.config.model_type][f'{wandb.config.model_type}_save_dir']
    print('save dir set')
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    T.manual_seed(train_config['seed'])
    T.cuda.manual_seed(train_config['seed'])

    callbacks = []
    if wandb.run:
        print(f'if wandb run fired')
        # logger.debug("if wandb.run fired")
        callbacks.append(WandbCallback(project_name=sweep_config["project"], _sweep=True))
        #DEBUG
        # for callback in callbacks:
        #     print(callback.get_config())
        if wandb.config.model_type == "HER_DDPG":
            actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(wandb.config)
            print('layers built')
            agent_class = get_agent_class_from_type(wandb.config.model_type)
            print('agent class retrieved')
            rl_agent = agent_class.build(
                env=env,
                actor_cnn_layers=actor_cnn_layers,
                critic_cnn_layers=critic_cnn_layers,
                actor_layers=actor_layers,
                critic_state_layers=critic_state_layers,
                critic_merged_layers=critic_merged_layers,
                kernels=kernels,
                callbacks=callbacks,
                config=wandb.config,
            )
            # logger.debug("Agent built")
            #DEBUG
            print(f'agent built:{rl_agent.get_config()}')
            if MPI.COMM_WORLD.Get_rank() == 0:
                rl_agent.save()
                print(f'agent saved')
            
            # logger.debug("Starting single-process training")
            rl_agent.train(num_epochs = train_config['num_epochs'],
                            num_cycles = train_config['num_cycles'],
                            num_episodes = train_config['num_episodes'],
                            num_updates = train_config['num_updates'],
                            render = False,
                            render_freq = 0,
                            save_dir = save_dir,
                            run_number = run_number
                            )
        
if __name__ == "__main__":
    # logger.debug("Entered main")
    sweep_config_path = "sweep/sweep_config.json"
    train_config_path = "sweep/train_config.json"

    try:
        sweep_config = load_config(sweep_config_path)
        train_config = load_config(train_config_path)

        # logger.debug(f"Sweep config: {sweep_config}")
        # logger.debug(f"Train config: {train_config}")

        run_sweep(sweep_config, train_config)

    except KeyError as e:
        logger.error(f"KeyError in W&B stream handling: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")