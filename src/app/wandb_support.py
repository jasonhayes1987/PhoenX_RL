"""Adds support for W&B integration to the rl_agents package."""

# imports
import json
from pathlib import Path
import os
import ast
import subprocess
# import logging
from logging_config import logger
import time

import numpy as np
import random
# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback
import torch as T
import gymnasium as gym
import wandb
import pandas as pd
from scipy.stats import zscore
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.tools as tls

import rl_agents
import helper


def save_model_artifact(file_path: str, project_name: str, model_is_best: bool = False):
    """Save the model to W&B

    Args:
        file_path (str): The path to the model files.
        project_name (str): The name of the project.
        model_is_best (bool): Whether the model is the best model so far.
    """

    # Create a Model Version
    art = wandb.Artifact(f"{project_name}-{wandb.run.name}", type="model")
    # Add the serialized files
    art.add_dir(f"{file_path}", name="model")
    # art.add_file(f"{file_path}/obj_config.json", name="obj_config.json")
    # art.add_dir(f"{file_path}/policy_model", name="policy_model")
    # art.add_dir(f"{file_path}/value_model", name="value_model")
    # checks if there is a wandb config and if there is, log it as an artifact
    if os.path.exists(f"{file_path}/wandb_config.json"):
        art.add_file(f"{file_path}/wandb_config.json", name="wandb_config.json")
    if model_is_best:
        # If the model is the best model so far,
        #  add "best" to the aliases
        wandb.log_artifact(art, aliases=["latest", "best"])
    else:
        wandb.log_artifact(art)
    # Link the Model Version to the Collection
    wandb.run.link_artifact(art, target_path=project_name)


def load_model_from_artifact(artifact, load_weights: bool = True):
    """Loads the model from the specified artifact.

    Args:
        artifact (wandb.Artifact): The artifact to load the model from.

    Returns:
        rl_agents.Agent: The agent object.
    """

    # Download the artifact files to a directory
    artifact_dir = Path(artifact.download())

    return rl_agents.load_agent_from_config(artifact_dir, load_weights)


def build_layers(sweep_config):
    """formats sweep_config into policy and value layers.

    Args:
        sweep_config (dict): The sweep configuration.

    Returns:
        tuple: The policy layers and value layers.
    """
    model_type = list(sweep_config.keys())[0]
    if model_type == "Reinforce" or model_type == "ActorCritic":
        # get policy layers
        policy_layers = []
        if sweep_config[model_type][f"{model_type}_policy_num_layers"] > 0:
            for layer_num in range(1, sweep_config[model_type][f"{model_type}_policy_num_layers"] + 1):
                policy_layers.append(
                    (
                        sweep_config[model_type][f"policy_units_layer_{layer_num}_{model_type}"],
                        sweep_config[model_type][f"{model_type}_policy_activation"],
                    )
                )
        # get value layers
        value_layers = []
        if sweep_config[model_type][f"{model_type}_value_num_layers"] > 0:
            for layer_num in range(1, sweep_config[model_type][f"{model_type}_value_num_layers"] + 1):
                value_layers.append(
                    (
                        sweep_config[model_type][f"value_units_layer_{layer_num}_{model_type}"],
                        sweep_config[model_type][f"{model_type}_value_activation"],
                    )
                )

        return policy_layers, value_layers
    
    elif model_type in ["DDPG", "HER_DDPG", "TD3"]:
        
        # Get actor CNN layers if present
        actor_cnn_layers = []
        # Set variable to keep track of last out channel param in conv layer loop to use for batchnorm layer features param
        last_out_channels = 0

        # Set number of cnn layers to loop over and to check if 0
        num_layers = sweep_config[model_type][f"{model_type}_actor_num_cnn_layers"]
        
        # Loop over num layers if not 0
        if num_layers > 0:
            for layer_num in range(1, num_layers + 1):
                layer_type = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_types"]
                if layer_type == 'conv':
                    # get num filters
                    out_channels = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_conv_filters"]
                    # update last out channel param
                    last_out_channels = out_channels

                    # get kernel size
                    kernel_size = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_conv_kernel_size"]
                    # get stride
                    stride = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_conv_strides"]
                    # get padding
                    padding = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_conv_padding"]
                    # get bias
                    bias = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_conv_bias"]

                    # append to actor_cnn_layers
                    actor_cnn_layers.append({layer_type: {"out_channels": out_channels, "kernel_size": kernel_size, "stride": stride, "padding": padding, "bias": bias}})

                elif layer_type == 'pool':
                    # get pool size
                    kernel_size = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_pool_kernel_size"]
                    stride = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_pool_strides"]

                    # append to actor_cnn_layers
                    actor_cnn_layers.append({layer_type: {"kernel_size": kernel_size, "stride": stride}})

                elif layer_type == 'dropout':
                    # get dropout rate
                    rate = sweep_config[model_type][f"actor_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_actor_cnn_layer_{layer_num}_dropout_rate"]

                    # append to actor_cnn_layers
                    actor_cnn_layers.append({layer_type: {"p": rate}})

                elif layer_type == 'batchnorm':
                    if last_out_channels == 0:
                        raise ValueError("Batchnorm layer must come after a conv layer")
                    else:
                        num_features = last_out_channels
                    
                    # append to actor_cnn_layers
                    actor_cnn_layers.append({layer_type: {"num_features": num_features}})

        # Get critic CNN layers if present
        critic_cnn_layers = []
        # Set variable to keep track of last out channel param in conv layer loop to use for batchnorm layer features param
        last_out_channels = 0

        # Set number of cnn layers to loop over and to check if 0
        num_layers = sweep_config[model_type][f"{model_type}_critic_num_cnn_layers"]
        
        # Loop over num layers if not 0
        if num_layers > 0:
            for layer_num in range(1, num_layers + 1):
                layer_type = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_types"]
                if layer_type == 'conv':
                    # get num filters
                    out_channels = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_conv_filters"]
                    # update last out channel param
                    last_out_channels = out_channels

                    # get kernel size
                    kernel_size = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_conv_kernel_size"]
                    # get stride
                    stride = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_conv_strides"]
                    # get padding
                    padding = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_conv_padding"]
                    # get bias
                    bias = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_conv_bias"]

                    # append to critic_cnn_layers
                    critic_cnn_layers.append({layer_type: {"out_channels": out_channels, "kernel_size": kernel_size, "stride": stride, "padding": padding, "bias": bias}})

                elif layer_type == 'pool':
                    # get pool size
                    kernel_size = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_pool_kernel_size"]
                    stride = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_pool_strides"]

                    # append to critic_cnn_layers
                    critic_cnn_layers.append({layer_type: {"kernel_size": kernel_size, "stride": stride}})

                elif layer_type == 'dropout':
                    # get dropout rate
                    rate = sweep_config[model_type][f"critic_cnn_layer_{layer_num}_{model_type}"][f"{model_type}_critic_cnn_layer_{layer_num}_dropout_rate"]

                    # append to critic_cnn_layers
                    critic_cnn_layers.append({layer_type: {"p": rate}})

                elif layer_type == 'batchnorm':
                    if last_out_channels == 0:
                        raise ValueError("Batchnorm layer must come after a conv layer")
                    else:
                        num_features = last_out_channels
                    
                    # append to critic_cnn_layers
                    critic_cnn_layers.append({layer_type: {"num_features": num_features}})

        # Create empty dict to store kernel params
        kernels = {}
        # Create kernel initializer params
        for model in ['actor', 'critic']:
            for layer in ['hidden', 'output']:
                kernel = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_initializer"]
                params = {}
                if kernel == "constant":
                    params["val"] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_value"]

                elif kernel == 'variance_scaling':
                    params['scale'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_scale"]
                    params['mode'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_mode"]
                    params['distribution'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_distribution"]

                elif kernel == 'normal':
                    params['mean'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_mean"]
                    params['std'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_stddev"]

                elif kernel == 'uniform':
                    params['a'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_minval"]
                    params['b'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_maxval"]
                
                elif kernel == 'truncated_normal':
                    params['mean'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_mean"]
                    params['std'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_stddev"]

                elif kernel == "xavier_uniform":
                    params['gain'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_gain"]

                elif kernel == "xavier_normal":
                    params['gain'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_gain"]

                elif kernel == "kaiming_uniform":
                    params['mode'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_mode"]

                elif kernel == "kaiming_normal":
                    params['mode'] = sweep_config[model_type][f"{model_type}_{model}_{layer}_kernel_{kernel}"][f"{kernel}_mode"]

                # Create dict with kernel and params
                kernels[f'{model}_{layer}_kernel'] = {kernel:params}

        #DEBUG
        # print(f'kernels: {kernels}')

        
        
        # get actor hidden layers
        actor_layers = []
        for layer_num in range(1, sweep_config[model_type][f"{model_type}_actor_num_layers"] + 1):
            actor_layers.append(
                (
                    sweep_config[model_type][f"actor_units_layer_{layer_num}_{model_type}"],
                    sweep_config[model_type][f"{model_type}_actor_activation"],
                    kernels['actor_hidden_kernel'],
                )
            )
        # get critic state hidden layers
        critic_state_layers = []
        for layer_num in range(1, sweep_config[model_type][f"{model_type}_critic_state_num_layers"] + 1):
            critic_state_layers.append(
                (
                    sweep_config[model_type][f"critic_units_state_layer_{layer_num}_{model_type}"],
                    sweep_config[model_type][f"{model_type}_critic_activation"],
                    kernels['critic_hidden_kernel'],
                )
            )
        # get critic merged hidden layers
        critic_merged_layers = []
        for layer_num in range(1, sweep_config[model_type][f"{model_type}_critic_merged_num_layers"] + 1):
            critic_merged_layers.append(
                (
                    sweep_config[model_type][f"critic_units_merged_layer_{layer_num}_{model_type}"],
                    sweep_config[model_type][f"{model_type}_critic_activation"],
                    kernels['critic_hidden_kernel'],
                )
            )

        return actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels


def load_model_from_run(run_name: str, project_name: str, load_weights: bool = True):
    """Loads the model from the specified run.

    Args:
        run_name (str): The name of the run.
        project_name (str): The name of the project.

    Returns:
        rl_agents.Agent: The agent object.
    """

    artifact = get_artifact_from_run(project_name, run_name)

    return load_model_from_artifact(artifact, load_weights)


def hyperparameter_sweep(
    sweep_config,
    train_config,
    # num_sweeps: int,
    # episodes_per_sweep: int,
    # epochs_per_sweep: int = None,
    # cycles_per_sweep: int = None,
    # updates_per_sweep: int = None,
):
    """Runs a hyperparameter sweep of the specified agent.

    Args:
        rl_agent (rl_agents.Agent): The agent to train.
        sweep_config (dict): The sweep configuration.
        num_sweeps (int): The number of sweeps to run.
        episodes_per_sweep (int): The number of episodes to train per sweep.
    """
    #DEBUG
    # print(f'hyperparameter_sweep fired...')
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
    #DEBUG
    # print(f'sweep id: {sweep_id}')
    wandb.agent(
        sweep_id,
        function=lambda: _run_sweep(sweep_config, train_config,),
        count=train_config['num_sweeps'],
        project=sweep_config["project"],
    )
    # wandb.teardown()

def _run_sweep(sweep_config, train_config):
    """Runs a single sweep of the hyperparameter search.

    Args:
        sweep_config (dict): The sweep configuration.
        episodes_per_sweep (int): The number of episodes to train per sweep.
        save_dir (str): The directory to save the model to.

    Returns:
        dict: The sweep configuration.
    """
    try:
        from rl_callbacks import WandbCallback
        wandb.init()

        if wandb.config.model_type in ["Reinforce", "Actor Critic"]:
            run_number = get_next_run_number(sweep_config["project"])
            print(f'run number:{run_number}')
            train_config['run_number'] = run_number
            run = wandb.init(
                project=sweep_config["project"],
                settings=wandb.Settings(start_method='thread'),
                job_type="train",
                name=f"train-{run_number}",
                tags=["train"],
                group=f"group-{run_number}",
            )
            run.tags = run.tags + (wandb.config.model_type,)
            env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
            save_dir = train_config.get('save_dir', sweep_config[wandb.config.model_type][f'{wandb.config.model_type}_save_dir'])
            random.seed(train_config['seed'])
            np.random.seed(train_config['seed'])
            T.manual_seed(train_config['seed'])
            T.cuda.manual_seed(train_config['seed'])

            callbacks = []
            if wandb.run:
                print(f'if wandb run fired')
                callbacks.append(WandbCallback(project_name=sweep_config["project"], _sweep=True))
            policy_layers, value_layers = build_layers(wandb.config)
            agent = rl_agents.get_agent_class_from_type(wandb.config.model_type)
            rl_agent = agent.build(
                env=env,
                policy_layers=policy_layers,
                value_layers=value_layers,
                callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
                config=wandb.config,
                save_dir=wandb.config.save_dir,
            )
        elif wandb.config.model_type == "DDPG":
            run_number = get_next_run_number(sweep_config["project"])
            print(f'run number:{run_number}')
            train_config['run_number'] = run_number
            run = wandb.init(
                project=sweep_config["project"],
                settings=wandb.Settings(start_method='thread'),
                job_type="train",
                name=f"train-{run_number}",
                tags=["train"],
                group=f"group-{run_number}",
            )
            run.tags = run.tags + (wandb.config.model_type,)
            env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
            save_dir = train_config.get('save_dir', sweep_config[wandb.config.model_type][f'{wandb.config.model_type}_save_dir'])
            random.seed(train_config['seed'])
            np.random.seed(train_config['seed'])
            T.manual_seed(train_config['seed'])
            T.cuda.manual_seed(train_config['seed'])

            callbacks = []
            if wandb.run:
                print(f'if wandb run fired')
                callbacks.append(WandbCallback(project_name=sweep_config["project"], _sweep=True))
            actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(wandb.config)
            agent = rl_agents.get_agent_class_from_type(wandb.config.model_type)
            rl_agent = agent.build(
                env=env,
                actor_cnn_layers=actor_cnn_layers,
                critic_cnn_layers=critic_cnn_layers,
                actor_layers=actor_layers,
                critic_state_layers=critic_state_layers,
                critic_merged_layers=critic_merged_layers,
                kernels=kernels,
                callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
                config=wandb.config,
                save_dir=wandb.config.save_dir,
            )

        elif wandb.config.model_type == "HER_DDPG":
            print('her model creation fired')
            if train_config['use_mpi']:
                print('sweep use_mpi fired')
                mpi_command = [
                    "mpirun", 
                    "-np", str(train_config['num_workers']), 
                    "python", "sweep_mpi.py"
                ]
                
                mpi_process = subprocess.Popen(mpi_command, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = mpi_process.communicate()

                print("Standard Output:")
                print(stdout.decode())
                
                print("Standard Error:")
                print(stderr.decode())

                if mpi_process.returncode == 0:
                    print("Subprocess completed successfully")
                else:
                    print("Subprocess failed with return code", mpi_process.returncode)
            else:
                print('sweep not using mpi fired')
                print(f'wandb config:{wandb.config}')
                run_number = get_next_run_number(sweep_config["project"])
                print(f'run number:{run_number}')
                train_config['run_number'] = run_number
                run = wandb.init(
                    project=sweep_config["project"],
                    settings=wandb.Settings(start_method='thread'),
                    job_type="train",
                    name=f"train-{run_number}",
                    tags=["train"],
                    group=f"group-{run_number}",
                )
                print('wandb init called')
                run.tags = run.tags + (wandb.config.model_type,)
                print('wandb run tag complete')
                env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
                print(f'env spec: {env.spec}')
                save_dir = train_config.get('save_dir', wandb.config[wandb.config.model_type][f'{wandb.config.model_type}_save_dir'])
                print(f'save dir set:{save_dir}')
                random.seed(train_config['seed'])
                np.random.seed(train_config['seed'])
                T.manual_seed(train_config['seed'])
                T.cuda.manual_seed(train_config['seed'])

                callbacks = []
                print(f'if wandb run fired')
                callbacks.append(WandbCallback(project_name=sweep_config["project"], run_name=f"train-{run_number}", _sweep=True))
                actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = build_layers(wandb.config)
                agent_class = rl_agents.get_agent_class_from_type(wandb.config.model_type)
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
                print(f'agent built:{rl_agent.get_config()}')
                rl_agent.save()
                print(f'agent saved to {rl_agent.save_dir}')
                rl_agent.train(
                    num_epochs=train_config['num_epochs'],
                    num_cycles=train_config['num_cycles'],
                    num_episodes=train_config['num_episodes'],
                    num_updates=train_config['num_updates'],
                    render=False,
                    render_freq=0,
                    save_dir=save_dir,
                    run_number=run_number
                )

    except Exception as e:
        logging.error(f"Error during sweep run attempt: {str(e)}")


def get_run_id_from_name(project_name, run_name):
    """Returns the run ID for the specified run name.

    Args:
        project_name (str): The name of the project.
        run_name (str): The name of the run.

    Returns:
        str: The run ID.
    """

    api = wandb.Api()
    # Fetch all runs in the project
    runs = api.runs(f"{api.default_entity}/{project_name}")
    # Iterate over the runs and find the one with the matching name
    for run in runs:
        if run.name == run_name:
            return run.id

    # If we get here, no run has the given name
    return None


def get_run_number_from_name(run_name):
    """Extracts the run number from the run name.

    Args:
    run_name (str): The run name, e.g., 'train-4'.

    Returns:
    int: The extracted run number.
    """
    try:
        return int(run_name.split("-")[-1])

    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Invalid run name format. Expected format 'train-X' where X is an integer."
        ) from exc


def get_next_run_number(project_name):
    """Returns the next run number for the specified project.

    Args:
        project_name (str): The name of the project.

    Returns:
        int: The next run number.
    """
    api = wandb.Api()
    # Get the list of runs from the project
    runs = api.runs(f"jasonhayes1987/{project_name}")
    if runs:
        # Extract the run numbers and find the maximum
        run_numbers = [int(run.name.split("-")[-1]) for run in runs]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 1

    return next_run_number


def get_run(project_name, run_name):
    """Returns the specified run.

    Args:
    project_name (str): The name of the project.
    run_name (str): The name of the run.

    Returns:
    wandb.Run: The run object.
    """

    api = wandb.Api()
    # get the runs ID
    run_id = get_run_id_from_name(project_name, run_name)

    # Fetch the run using the project and run name
    run_path = f"{api.default_entity}/{project_name}/{run_id}"
    run = api.run(run_path)

    return run


def get_artifact_from_run(
    project_name, run_name, artifact_type: str = "model", version="best"
):
    """Returns the specified artifact from the specified run.

    Args:
    project_name (str): The name of the project.
    run_name (str): The name of the run.
    artifact_type (str): The type of artifact to fetch.
    version (str): The version of the artifact to fetch.

    Returns:
    wandb.Artifact: The artifact object.
    """
    api = wandb.Api()
    # Get the run
    run = get_run(project_name, run_name)

    # Get the list of artifacts linked to this run
    linked_artifacts = run.logged_artifacts()

    # Find the artifact of the specified type
    artifact_name = None
    for artifact in linked_artifacts:
        if artifact.type == artifact_type and version in artifact.aliases:
            artifact_name = artifact.name
            break
    if artifact_name is None:
        raise ValueError("No artifact of the specified type found in the run")

    # Construct the artifact path
    artifact_path = f"{api.default_entity}/{project_name}/{artifact_name}"

    # Fetch the artifact
    artifact = api.artifact(artifact_path, type=artifact_type)

    return artifact


def get_projects():
    """Returns the list of projects."""
    api = wandb.Api()
    projects = api.projects()

    return projects


def get_runs(project_name):
    """Returns the list of runs for the specified project.

    Args:
        project_name (str): The name of the project.

    Returns:
        list: The list of runs.
    """

    api = wandb.Api()

    runs = api.runs(f"{api.default_entity}/{project_name}")

    return runs


def delete_all_runs(project_name, delete_artifacts: bool = True):
    """Deletes all runs for the specified project.

    Args:
    project_name (str): The name of the project.
    delete_artifacts (bool): Whether to delete the artifacts associated with the runs.
    """

    api = wandb.Api()
    wandb.finish()
    runs = api.runs(f"{api.default_entity}/{project_name}")
    for run in runs:
        print(f"Deleting run: {run.name}")
        run.delete()

def delete_all_artifacts(project_name, artifact_type: str = "model"):
    """Deletes all artifacts for the specified project."""
    api = wandb.Api()

    # Fetch all artifacts in the project
    artifacts = api.artifact_collections(project_name='Pendulum-v1', type_name='model')

    for artifact in artifacts:
        # If filtering by type, uncomment the following lines
        # if artifact.type != artifact_type:
        #     continue
        
        print(f"Deleting artifact: {artifact.name}")
        artifact.delete()


## NOT WORKING
# def delete_all_sweeps(project_name):
#     api = wandb.Api()
#     wandb.finish()
#     project = api.project(f"{api.default_entity}/{project_name}")
#     print(f"Deleting all sweeps in project: {project_name}")
#     sweeps = project.sweeps()
#     print(f"sweeps: {sweeps}")
#     for sweep in sweeps:
#         print(f"Deleting sweep: {sweep.id}")
#         sweep.delete()

## NOT WORKING
# def delete_artifacts(project_name, only_empty: bool = True):
#     api = wandb.Api()
#     artifacts = api.artifacts(f"{api.default_entity}/{project_name}/model", per_page=1000)

#     for artifact in artifacts:
#         # Fetch the artifact to get detailed info
#         artifact = artifact.fetch()
#         if only_empty:
#             if len(artifact.manifest.entries) == 0:
#                 print(f"Deleting empty artifact: {artifact.name}")
#                 artifact.delete()
#             else:
#                 print(f"Artifact {artifact.name} is not empty and will not be deleted.")
#         else:
#             print(f"Deleting artifact: {artifact.name}")
#             artifact.delete()


def custom_wandb_init(*args, **kwargs):
    """Initializes a W&B run and prints the run ID."""
    print("Initializing W&B run...")
    run = wandb.init(*args, **kwargs)
    print(f"Run initialized with ID: {run.id}")

    return run


def custom_wandb_finish():
    """Finishes a W&B run."""
    print("Finishing W&B run...")
    wandb.finish()
    print("Run finished.")


def flush_cache():
    """Flushes the W&B cache."""
    api = wandb.Api()
    api.flush()


def get_sweep_from_name(project, sweep_name):
    try:
        api = wandb.Api()
        sweeps = api.project(project).sweeps()
        
        # Filter sweeps by name to find the matching sweep ID
        for sweep in sweeps:
            if sweep.name == sweep_name:
                return sweep
        else:
            print(f"No sweep found with the name '{sweep_name}'.")
    except Exception as e:
        print(f"An error occurred in get_sweep_from_name: {e}")
        return None

def get_sweeps_from_name(project):
    try:
        api = wandb.Api()
        sweeps = api.project(project).sweeps()

        # Filter sweeps by name to find the matching sweep ID
        sweeps_list = [sweep.name for sweep in sweeps]
        return sweeps_list
    except Exception as e:
        print(f"An error occurred in get_sweeps_from_name: {e}")
        return []

def get_runs_from_sweep(project, sweep):
    try:
        api = wandb.Api()
        runs = api.runs(path=project, filters={"sweep": sweep.id})
        return runs
    except Exception as e:
        print(f"An error occurred in get_runs_from_sweep: {e}")

def get_metrics(project: str, sweep_name: str = None):
    try:
        api = wandb.Api()
        # check if sweep is specified and if so, get the runs from the sweep
        if sweep_name is not None:
            sweep = get_sweep_from_name(project, sweep_name)
            runs = get_runs_from_sweep(project, sweep)
        else:
            runs = api.runs(project)

        summary_list, config_list, name_list = [], [], []

        for run in runs:       
            # call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # remove special values that start with _.
            config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

            name_list.append(run.name)

        runs_df = pd.DataFrame(
            {"summary": summary_list, "config": config_list, "name": name_list}
        )

        return runs_df
    except Exception as e:
        print(f"An error occurred in get_metrics: {e}")
        return pd.DataFrame()

def format_metrics(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug(f'format_metrics: data passed to format metrics: {data}')
        # Parse the 'config' column
        data['config'] = parse_dict_column(data['config'])
        logger.debug(f'format_metrics: data after parse dict column: {data}')

        # Extract hyperparameters and rewards from the dictionaries
        data['avg_reward'] = data['summary'].apply(lambda x: x.get('avg_reward') if isinstance(x, dict) else None)
        logger.debug(f'format_metrics: data after getting avg reward: {data}')
        data.to_csv(f"data.csv")

        # Filter out rows that do not have a 'config_dict' or 'avg reward'
        data_filtered = data.dropna(subset=['config', 'avg_reward'])
        logger.debug(f'format_metrics: data filtered: {data_filtered}')

        # Flatten the 'config_dict'
        hyperparams = data_filtered['config'].apply(lambda x: helper.flatten_dict(x))
        logger.debug(f'format_metrics: hyperparams after helper.flatten_dict: {hyperparams}')
        # parse hyperparameter names to remove dupe parts in name due to wandb config structure
        hyperparams = modify_keys(hyperparams)
        logger.debug(f'format_metrics: hyperparams after modify_keys: {hyperparams}')
        # and create a new DataFrame of hyperparameter values
        data_hyperparams = pd.DataFrame(hyperparams)
        logger.debug(f'format_metrics: data hyperparams columns after flatten config dict: {data_hyperparams.columns}')
        # Join the flattened hyperparameters with the avg reward column
        data_hyperparams = data_hyperparams.join(data_filtered['avg_reward'])
        logger.debug(f'format_metrics: data hyperparams after joining with data_filtered[avg_reward]: {data_hyperparams}')

        return data_hyperparams
    except Exception as e:
        logger.error(f"An error occurred in format_metrics: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_co_occurrence_matrix(data: pd.DataFrame, hyperparameters: list, avg_reward_threshold: int, bins: int, z_scores: bool = False) -> pd.DataFrame:
    try:
        logger.debug(f"calculate_co_occurence_matrix: passed param data: {data}")
        logger.debug(f"calculate_co_occurence_matrix: passed param hyperparameters: {hyperparameters}")
        # create an empty dict to store the bin ranges of each hyperparameter
        bin_ranges = {}
        # drop all columns that arent in hyperparameters list
        data = data[hyperparameters + ['avg_reward']]
        logger.debug(f'calculate_co_occurence_matrix: data:{data}')
        # Filter the DataFrame based on the avg reward_threshold
        data_heatmap = data[data['avg_reward'] >= avg_reward_threshold]
        logger.debug(f'calculate_co_occurence_matrix: data_heatmap created')
        # For continuous variables, bin them into the specified number of bins
        for hp in data_heatmap.columns:
            if data_heatmap[hp].dtype == float and hp not in ['avg_reward']:
                data_heatmap[hp], bin_edges  = pd.cut(data_heatmap[hp], bins, labels=range(bins), retbins=True)
                bin_ranges[hp] = bin_edges
        logger.debug(f'calculate_co_occurence_matrix: bin ranges created')

        # One-hot encode categorical variables
        data_one_hot = pd.get_dummies(data_heatmap.drop('avg_reward', axis=1).astype('category'), dtype=np.int8)
        logger.debug(f'calculate_co_occurence_matrix: categorical data one-hot encoded')
        # Calculate co-occurrence matrix
        co_occurrence_matrix = np.dot(data_one_hot.T, data_one_hot)
        logger.debug(f'calculate_co_occurence_matrix: co-occurrence matrix calculated')

        # calculate z-scores if z_scores is true
        if z_scores:
            # Calculate the z-scores of the co-occurrence counts
            co_occurrence_matrix = zscore(co_occurrence_matrix, axis=None)
            logger.debug(f'calculate_co_occurence_matrix: zscore co-occurrence matrix calculated')
        # Create a DataFrame from the co-occurrence matrix for easier plotting
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix, 
                                        index=data_one_hot.columns, 
                                        columns=data_one_hot.columns)
        logger.debug(f'calculate_co_occurence_matrix: co-occurrence dataframe created')
        co_occurrence_df.to_csv(f"co_occurrence_df.csv")
        logger.debug(f'calculate_co_occurence_matrix: co-occurrance dataframe saved to csv')

        return co_occurrence_df, bin_ranges
    except Exception as e:
        logger.error(f"An error occurred in calculate_co_occurrence_matrix: {e}", exc_info=True)
        return pd.DataFrame(), {}

def plot_co_occurrence_heatmap(co_occurrence_df: pd.DataFrame) -> go.Figure:
    try:
        # Create a layout
        layout = go.Layout(title='Co-occurrence Heatmap')

        # Create the figure
        fig = go.Figure(
            data=
                go.Heatmap(
                    z=co_occurrence_df.values,  # Heatmap values
                    x=co_occurrence_df.columns,  # X-axis categories
                    y=co_occurrence_df.index, # Y-axis categories
                ),
                layout=layout,
            )  

        return fig
    except Exception as e:
        print(f"An error occurred in plot_co_occurrence_heatmap: {e}")
        return go.Figure()

# Function to safely parse a Python dictionary string
def parse_dict_column(column):
    parsed_column = []
    for item in column:
        try:
            # Safely evaluate the dictionary string to a dictionary
            parsed_column.append(item)
        except ValueError:
            # In case of error, append a None or an empty dictionary
            parsed_column.append(None)
    return parsed_column

# Function to flatten the config dictionary and extract specified hyperparameters
def flatten_config(config_dict, hyperparameters):
    flat_config = {}
    for param in hyperparameters:
        # Extract the hyperparameter value from the nested dictionaries
        value = config_dict.get('DDPG', {}).get(param, None)
        # If a hyperparameter is not found, we try to get it from the top level (for env and model_type)
        if value is None:
            value = config_dict.get(param, None)
        flat_config[param] = value
    return flat_config

def fetch_sweep_hyperparameters_single_run(project_name, sweep_name):
    api = wandb.Api()
    sweep = get_sweep_from_name(project_name, sweep_name)
    runs = get_runs_from_sweep(project_name, sweep)
    hyperparameters = set()

    # Attempt to fetch a single run; if no runs, return empty list
    if runs is not None:
        run = runs.__getitem__(0)  # Take the first run from the sweep
        config = run.config
        flattened_config = helper.flatten_dict(config)
        for key in flattened_config.keys():
            if not key.startswith('_') and not key in ['wandb_version', 'state', 'env']:
                # print(f"key before:{key}")
                key = parse_parameter_name(key)
                # print(f"key after:{key}")
                hyperparameters.add(key)

    return list(hyperparameters)

def parse_parameter_name(parameter_string):
    # check if first 2 parts match the next 2 and if so, get rid of first 2 and return the rest
    parts = parameter_string.split('_') # update to split on _ without maxsplit
    if parts[:2] == parts[2:4]: # if the first string is repeated (model_type)
        return '_'.join(parts[2:]) # remove the repeated string and return the rest joined by '_'
                                            
    else:
        return parameter_string
    
def modify_keys(dicts):
    modified_dicts = []
    for d in dicts:
        new_dict = {}
        for key, value in d.items():
            new_key = parse_parameter_name(key)
            new_dict[new_key] = value
        modified_dicts.append(new_dict)
    return modified_dicts

