from zipfile import ZipFile
import os
import re
from natsort import natsorted
import requests
from pathlib import Path
import shutil
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import re

import wandb
import wandb_support
# import ray.tune as tune
import gymnasium as gym
import gymnasium.wrappers as base_wrappers
import gymnasium_robotics as gym_robo
# import tensorflow as tf
import numpy as np

import rl_agents
import models
from models import StochasticContinuousPolicy, StochasticDiscretePolicy, ActorModel, ValueModel, CriticModel
import rl_callbacks
from env_wrapper import GymnasiumWrapper, IsaacSimWrapper
from noise import *
from buffer import *

# Wrappers NOT in the user dropdown
EXCLUSION_LIST = {
    "RecordEpisodeStatistics",
    "RenderCollection",
    "RecordVideo",
    "HumanRendering",
    "DelayObservation",
    "MaxAndSkipObservation",
    "TransformAction",
    "ClipAction",
    "RescaleAction",
    "TransformObservation",
    "FilterObservation",
    "FlattenObservation",
    "ReshapeObservation",
    "RescaleObservation",
    "DtypeObservation",
    "AddRenderObservation",
    "Autoreset",
    "JaxToNumpy",
    "JaxToTorch",
    "NumpyToTorch",
    "OrderEnforcing",
    "PassiveEnvChecker",
    "vector"
}

# Wrappers that require extra parameter inputs:
WRAPPER_REGISTRY = {
    "AtariPreprocessing": {
        "cls": base_wrappers.AtariPreprocessing,
        "default_params": {
            "frame_skip": 1,
            "grayscale_obs": True,
            "scale_obs": True
        }
    },
    "TimeLimit": {
        "cls": base_wrappers.TimeLimit,
        "default_params": {
            "max_episode_steps": 1000
        }
    },
    "TimeAwareObservation": {
        "cls": base_wrappers.TimeAwareObservation,
        "default_params": {
            "flatten": False,
            "normalize_time": False
        }
    },
    "FrameStackObservation": {
        "cls": base_wrappers.FrameStackObservation,
        "default_params": {
            "stack_size": 4
        }
    },
    "ResizeObservation": {
        "cls": base_wrappers.ResizeObservation,
        "default_params": {
            "shape": 84
        }
    }
}

def camel_to_words(name: str) -> str:
    """
    Convert CamelCase to spaced words with each word capitalized.
    E.g. 'AtariPreprocessing' -> 'Atari Preprocessing'
    """
    # Insert a space before any capital letter that isn't the first character
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    # Now split on spaces and capitalize each token
    words = [word.capitalize() for word in spaced.split()]
    # Join them back into a single string
    return ' '.join(words)

def get_wrappers_dropdown_options():
    """
    Returns a sorted list of *all* wrappers from base_wrappers.__all__,
    minus those in EXCLUSION_LIST, for the user to select from.
    """
    all_wrappers = set(base_wrappers.__all__)
    valid_wrappers = all_wrappers - EXCLUSION_LIST
    return sorted(valid_wrappers)

# def create_wrappers_list(selected_wrappers, user_params):
#     """
#     Given a list of selected wrapper names and a dictionary of user-overridden
#     parameters, return a list of wrapper factory functions.

#     Any wrapper found in WRAPPER_REGISTRY uses user_params if provided.
#     Any wrapper not in WRAPPER_REGISTRY is applied with no arguments.
#     """
#     #DEBUG
#     if not selected_wrappers:
#         return []

#     wrappers_list = []

#     for w_name in selected_wrappers:
#         # If wrapper is in registry, it may have parameter overrides
#         if w_name in WRAPPER_REGISTRY:
#             wrapper_cls = WRAPPER_REGISTRY[w_name]["cls"]
#             default_params = WRAPPER_REGISTRY[w_name]["default_params"]
#             override_params = user_params.get(w_name, {}) if user_params else {}
#             final_params = {**default_params, **override_params}

#             # Example: special case for ResizeObservation
#             if w_name == "ResizeObservation" and isinstance(final_params.get("shape", None), int):
#                 side = final_params["shape"]
#                 final_params["shape"] = (side, side)

#             def wrapper_factory(env, cls=wrapper_cls, fparams=final_params):
#                 return cls(env, **fparams)

#         else:
#             # It's NOT in WRAPPER_REGISTRY => no-arg wrapper
#             wrapper_cls = getattr(base_wrappers, w_name, None)
#             # If the user only picks from a valid filtered list, wrapper_cls should not be None.
#             if wrapper_cls is None:
#                 # If somehow the user selected an invalid or excluded wrapper, skip or log
#                 continue

#             def wrapper_factory(env, cls=wrapper_cls):
#                 return cls(env)

#         wrappers_list.append(wrapper_factory)

#     return wrappers_list

def format_wrappers(wrapper_store):
    wrappers_dict = {}
    for key, value in wrapper_store.items():
        # Split the key into wrapper type and parameter name
        parts = key.split('_param:')
        # print(f'parts:{parts}')
        wrapper_type = parts[0].split('wrapper:')[1]
        # print(f'wrapper_type:{wrapper_type}')
        param_name = parts[1]
        # print(f'param name:{param_name}')
        
        # If the wrapper type already exists in the dictionary, append to its params
        if wrapper_type not in wrappers_dict:
            wrappers_dict[wrapper_type] = {'type': wrapper_type, 'params': {}}
        
        wrappers_dict[wrapper_type]['params'][param_name] = value
    
    # Convert the dictionary to a list of dictionaries
    formatted_wrappers = list(wrappers_dict.values())
    
    return formatted_wrappers

def get_key(id_dict, param=None):
    """
    Generate a key for the params_store dictionary based on the id_dict and an optional parameter.

    Args:
        id_dict (dict): The dictionary containing keys like 'type', 'model', 'agent', and optionally 'index'.
        param (str): An optional parameter name to append to the key.

    Returns:
        str: The formatted key as a string.
    """
    if param:
        #DEBUG
        base_key = "_".join(f"{k}:{v}" for k, v in list(id_dict.items())[1:])
        # print(f"key=type:{param}_{base_key}")
        return f"type:{param}_{base_key}"
    else:
        base_key = "_".join(f"{k}:{v}" for k, v in list(id_dict.items()))
        return base_key

def get_param_from_key(key):
    return key.split("_")[0].split(":")[-1].replace("-","_")

def get_specific_value(all_values, all_ids, id_type, model_type, agent_type):
    #DEBUG
    # print(f'get_specific_value fired...')
    #DEBUG
    # print(f'all values: {all_values}')
    # print(f'all ids: {all_ids}')
    for id_dict, value in zip(all_ids, all_values):
        # Check if this id dictionary matches the criteria
        #DEBUG
        # print(f'checking id_dict: {id_dict} and value: {value}')
        if id_dict.get('type') == id_type and id_dict.get('model') == model_type and id_dict.get('agent') == agent_type:
            #DEBUG
            # print(f"Found value {value} for {id_type} {model_type} {agent_type}")
            return value
    # Return None or some default value if not found
    return None

def get_specific_value_id(all_values, all_ids, value_type, model_type, agent_type, index):
    #DEBUG
    # print(f'get_specific_value id fired...')
    # print(f'all_values: {all_values}')
    # print(f'all_ids: {all_ids}')
    for id_dict, value in zip(all_ids, all_values):
        # print(f'checking id_dict: {id_dict} and value: {value}')
        if 'index' in id_dict.keys():
            # Check if this id dictionary matches the criteria
            if id_dict.get('type') == value_type and id_dict.get('model') == model_type and id_dict.get('agent') == agent_type and id_dict.get('index') == index:
                #DEBUG
                # print(f"Found value {value} for {value_type} {model_type} {agent_type}")
                return value
    # Return None or some default value if not found
    return None

def create_noise_object(env, model_type, agent_type, agent_params):
    noise_type = agent_params.get(get_key({'type':'noise-function', 'model':model_type, 'agent':agent_type}))

    if noise_type == "Normal":
        return Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            mean=agent_params.get(get_key({'type':'normal-mean', 'model':model_type, 'agent':agent_type})),
            stddev=agent_params.get(get_key({'type':'normal-stddv', 'model':model_type, 'agent':agent_type})),
            device=agent_params.get(get_key({'type':'device', 'model':model_type, 'agent':agent_type})),
        )

    elif noise_type == "Uniform":
        return Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            minval=agent_params.get(get_key({'type':'uniform-min', 'model':model_type, 'agent':agent_type})),
            maxval=agent_params.get(get_key({'type':'uniform-max', 'model':model_type, 'agent':agent_type})),
            device=agent_params.get(get_key({'type':'device', 'model':model_type, 'agent':agent_type})),
        )

    elif noise_type == "Ornstein-Uhlenbeck":
        return Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            mean=agent_params.get(get_key({'type':'ou-mean', 'model':model_type, 'agent':agent_type})),
            theta=agent_params.get(get_key({'type':'ou-theta', 'model':model_type, 'agent':agent_type})),
            sigma=agent_params.get(get_key({'type':'ou-sigma', 'model':model_type, 'agent':agent_type})),
            dt=agent_params.get(get_key({'type':'ou-dt', 'model':model_type, 'agent':agent_type})),
            device=agent_params.get(get_key({'type':'device', 'model':model_type, 'agent':agent_type})),
        )

def format_layers(model, agent, agent_params):
    # DEBUG
    print(f'Agent params: {agent_params}')
    
    layer_config = []

    # Extract keys related to the specified model and agent
    relevant_keys = {
        key: value for key, value in agent_params.items()
        if f"model:{model}" in key and f"agent:{agent}" in key
    }

    # Determine the indices of layers for the given model
    layer_indices = set(
        int(key.split("_")[-1].split(":")[-1])  # Extract the index
        for key in relevant_keys.keys()
        if "index:" in key
    )

    # DEBUG
    # print(f'Layer indices: {layer_indices}')

    # Iterate over each layer index and construct the layer configuration
    for index in sorted(layer_indices):
        if index > 0:
            layer_entry = {}

            # Construct keys for the current index
            type_key = f"type:layer-type-dropdown_model:{model}_agent:{agent}_index:{index}"
            if type_key in relevant_keys:
                layer_entry["type"] = relevant_keys[type_key]

            # Handle layers with parameters (e.g., dense, conv2d)
            if layer_entry["type"] == "dense":
                layer_entry["params"] = format_dense_layer(model, agent, index, relevant_keys)

            elif layer_entry["type"] == "conv2d":
                layer_entry["params"] = format_cnn_layer(model, agent, index, relevant_keys)

            # elif layer_entry["type"] == 


            # Handle layers with no params
            elif layer_entry["type"] in ["relu", "tanh", "flatten"]:
                layer_entry.pop("params", None)  # Remove params if present

            # Append the structured layer entry
            layer_config.append(layer_entry)
            # DEBUG
            print(f'Layer entry added: {layer_entry}')

    print(f'Final layer config: {layer_config}')
    return layer_config

def format_dense_layer(model, agent, index, keys):
    units_key = f"type:num-units_model:{model}_agent:{agent}_index:{index}"
    kernel_key = f"type:kernel-init_model:{model}_agent:{agent}_index:{index}"
    kernel_params_key = f"type:kernel-params_model:{model}_agent:{agent}_index:{index}"
    bias_key = f"type:bias_model:{model}_agent:{agent}_index:{index}"

    params = {
        "units": keys.get(units_key, None),
        "kernel": keys.get(kernel_key, "default"),
        "kernel params": keys.get(kernel_params_key, {}),
        "bias": keys.get(bias_key, True)
    }

    kernel_type = params["kernel"]
    KERNEL_PARAMS_MAP = get_kernel_params_map()
    if kernel_type in KERNEL_PARAMS_MAP:
        for param in KERNEL_PARAMS_MAP[kernel_type]:
            param_key = f"type:{param}_model:{model}_agent:{agent}_index:{index}"
            if param_key in keys:
                params["kernel params"][param] = keys[param_key]
    return params

def format_cnn_layer(model, agent, index, keys):
    out_channels_key = f"type:out-channels_model:{model}_agent:{agent}_index:{index}"
    kernel_size_key = f"type:kernel-size_model:{model}_agent:{agent}_index:{index}"
    stride_key = f"type:stride_model:{model}_agent:{agent}_index:{index}"
    padding_key = f"type:padding-dropdown_model:{model}_agent:{agent}_index:{index}"
    bias_key = f"type:bias_model:{model}_agent:{agent}_index:{index}"
    kernel_key = f"type:kernel-init_model:{model}_agent:{agent}_index:{index}"
    kernel_params_key = f"type:kernel-params_model:{model}_agent:{agent}_index:{index}"

    params = {
        "out_channels": keys.get(out_channels_key, 32),
        "kernel_size": keys.get(kernel_size_key, 2),
        "stride": keys.get(stride_key, 1),
        "padding": keys.get(padding_key, 0),
        "bias": keys.get(bias_key, True),
        "kernel": keys.get(kernel_key, "default"),
        "kernel params": keys.get(kernel_params_key, {}),
    }
    return params


#TODO Don't think I need this function anymore
# def format_cnn_layers(all_values, all_ids, all_indexed_values, all_indexed_ids, model_type, agent_type):
#     layers = []
#     # Get num CNN layers for model type
#     num_cnn_layers = get_specific_value(
#         all_values=all_values,
#         all_ids=all_ids,
#         id_type='conv-layers',
#         model_type=model_type,
#         agent_type=agent_type,
#     )
#     #DEBUG
#     # print(f'num_cnn_layers: {num_cnn_layers}')

#     # Loop through num of CNN layers
#     for index in range(1, num_cnn_layers+1):
#         # Get the layer type
#         layer_type = get_specific_value_id(
#             all_values=all_indexed_values,
#             all_ids=all_indexed_ids,
#             value_type='cnn-layer-type',
#             model_type=model_type,
#             agent_type=agent_type,
#             index=index
#         )
#         #DEBUG
#         # print(f'layer_type: {layer_type}')

#         # Parse layer types to set params
#         if layer_type == "conv":
#             params = {}
#             params['out_channels'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='conv-filters',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             params['kernel_size'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='conv-kernel-size',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             params['stride'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='conv-stride',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             padding = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='conv-padding',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             if padding == "custom":
#                 params['padding'] = get_specific_value_id(
#                     all_values=all_indexed_values,
#                     all_ids=all_indexed_ids,
#                     value_type='conv-padding-custom',
#                     model_type=model_type,
#                     agent_type=agent_type,
#                     index=index
#                 )

#             else:
#                 params['padding'] = padding


#             params['bias'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='conv-use-bias',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             # Append to layers list
#             layers.append({layer_type: params})
#             continue
        
#         elif layer_type == "batchnorm":
#             params = {}
#             params['num_features'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='batch-features',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             layers.append({layer_type: params})
#             continue
        
#         elif layer_type == "pool":
#             params = {}
#             params['kernel_size'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='pool-kernel-size',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             params['stride'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='pool-stride',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             layers.append({layer_type: params})
#             continue

#         elif layer_type == 'dropout':
#             params = {}
#             params['p'] = get_specific_value_id(
#                 all_values=all_indexed_values,
#                 all_ids=all_indexed_ids,
#                 value_type='dropout-prob',
#                 model_type=model_type,
#                 agent_type=agent_type,
#                 index=index
#             )
#             layers.append({layer_type: params})
#             continue

#         elif layer_type == 'relu':
#             params = {}
#             layers.append({layer_type: params})

#         elif layer_type == 'tanh':
#             params = {}
#             layers.append({layer_type: params})

#         else:
#             raise ValueError(f'Layer type {layer_type} not supported')
        
#     return layers




def get_projects():
    """Returns a list of projects from the W&B API."""

    api = wandb.Api()

    return [project.name for project in api.projects()]


def get_callbacks(callbacks, project):
    
    callback_objects = {
        "Weights & Biases": rl_callbacks.WandbCallback(project),
    }
    #DEBUG
    # print(f'callbacks: {callbacks}')

    callbacks_list = [rl_callbacks.DashCallback("http://127.0.0.1:8050")]
    # for id_dict, value in zip(all_ids, all_values):
    #     if id_dict.get('type') == 'callback':
    #         #DEBUG
    #         print(f'callback type found: {id_dict, value}')
    #         # Ensure value is a list before iterating over it
    #         selected_callbacks = value if isinstance(value, list) else [value] # make value variable a list
    #         #DEBUG
    #         print(f'selected_callbacks: {selected_callbacks}')
    #         for callback_name in selected_callbacks:
    #             #DEBUG
    #             print(f'callback_name: {callback_name}')
    #             if callback_name in callbacks:
    #                 #DEBUG
    #                 print(f'matched callback: {callback_name}')
    #                 callbacks_list.append(callbacks[callback_name])

    for callback in callbacks:
        if callback in callback_objects:
            callbacks_list.append(callback_objects[callback])

    #DEBUG
    # print(f'callback list: {callbacks_list}')
    return callbacks_list


def zip_agent_files(source_dir, output_zip):
    # Print the directory being zipped and list its contents
    # print(f"Zipping contents of: {source_dir}")
    files_found = os.listdir(source_dir)
    # print("Files found:", files_found)
    
    if not files_found:
        # print("No files found to zip. Exiting zip operation.")
        return
    
    with ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure the file exists before adding it
                if os.path.isfile(file_path):
                    zipf.write(file_path, os.path.relpath(file_path, source_dir))
                    # print(f"Added {file} to ZIP archive.")
                # else:
                    # print(f"Skipped {file}, not found or inaccessible.")


def load(agent_data, env_name):
    # check if the env name in agent data matches env_name var
    # check if the agent_data has the key 'agent'
    if 'agent' in agent_data:
        # if 'agent' key exists, use agent_data['agent']['env'] as the env
        agent_env = agent_data['agent']['env']
    else:
        # if 'agent' key doesn't exist, use agent_data['env'] as the env
        agent_env = agent_data['env']
    
    
    if agent_env == env_name:
        #DEBUG
        # print('env name matches!')
        # Load the agent
        return rl_agents.load_agent_from_config(agent_data)

    # else (they don't match) change params to match new environment action space
    # check what the agent type is to update params accordingly
    if agent_data['agent_type'] == 'Reinforce' or agent_data['agent_type'] == 'ActorCritic':
        env=gym.make(env_name)
        
        policy_optimizer = agent_data['policy_model']['optimizer']['class_name']
        
        value_optimizer = agent_data['value_model']['optimizer']['class_name']

        policy_layers = [(units, activation, initializer) for (units, activation, initializer) in agent_data['policy_model']['dense_layers']]

        ##DEBUG
        # print("Policy layers:", policy_layers)
        
        value_layers = [(units, activation, initializer) for (units, activation, initializer) in agent_data['value_model']['dense_layers']]

        ##DEBUG
        # print("Value layers:", value_layers)

        policy_model = models.StochasticDiscretePolicy(
                env=env,
                dense_layers=policy_layers,
                optimizer=policy_optimizer,
                learning_rate=agent_data['learning_rate'],
            )
        value_model = models.ValueModel(
            env=env,
            dense_layers=value_layers,
            optimizer=value_optimizer,
            learning_rate=agent_data['learning_rate'],
        )
        
        if agent_data['agent_type'] == "Reinforce":
            
            agent = rl_agents.Reinforce(
                env=env,
                policy_model=policy_model,
                value_model=value_model,
                discount=agent_data['discount'],
                callbacks = [rl_callbacks.load(callback['class_name'], callback['config']) for callback in agent_data['callbacks']],
                save_dir=agent_data['save_dir'],
            )
        elif agent_data['agent_type'] == "ActorCritic":
            
            agent = rl_agents.ActorCritic(
                env=gym.make(env),
                policy_model=policy_model,
                value_model=value_model,
                discount=agent_data['discount'],
                policy_trace_decay=agent_data['policy_trace_decay'],
                value_trace_decay=agent_data['policy_trace_decay'],
                callbacks = [rl_callbacks.load(callback['class_name'], callback['config']) for callback in agent_data['callbacks']],
                save_dir=agent_data['save_dir'],
            )
    elif agent_data['agent_type'] == "DDPG":
        # set defualt gym environment in order to build policy and value models and save
        env=gym.make(env_name)

        # set actor and critic model params
        actor_optimizer = agent_data['actor_model']['optimizer']['class_name']
        critic_optimizer = agent_data['critic_model']['optimizer']['class_name']
        
        actor_layers = [(units, activation, initializer) for (units, activation, initializer) in agent_data['actor_model']['dense_layers']]

        ##DEBUG
        # print("Actor layers:", actor_layers)
        
        critic_state_layers = [(units, activation, initializer) for (units, activation, initializer) in agent_data['critic_model']['state_layers']]
        ##DEBUG
        # print("Critic state layers:", critic_state_layers)
        critic_merged_layers = [(units, activation, initializer) for (units, activation, initializer) in agent_data['critic_model']['merged_layers']]
        ##DEBUG
        # print("Critic merged layers:", critic_merged_layers)

        actor_model = models.ActorModel(
            env=env,
            dense_layers=actor_layers,
            learning_rate=agent_data['actor_model']['learning_rate'],
            optimizer=actor_optimizer
        )
        ##DEBUG
        # print("Actor model:", actor_model.get_config())
        
        critic_model = models.CriticModel(
            env=env,
            state_layers=critic_state_layers,
            merged_layers=critic_merged_layers,
            learning_rate=agent_data['critic_model']['learning_rate'],
            optimizer=critic_optimizer
        )
        ##DEBUG
        # print("Critic model:", critic_model.get_config())
        agent = rl_agents.DDPG(
            env=env,
            actor_model=actor_model,
            critic_model=critic_model,
            discount=agent_data['discount'],
            tau=agent_data['tau'],
            replay_buffer=helper.ReplayBuffer(env, 100000),
            batch_size=agent_data['batch_size'],
            noise=helper.Noise.create_instance(agent_data["noise"]["class_name"], **agent_data["noise"]["config"]),
            callbacks = [rl_callbacks.load(callback['class_name'], callback['config']) for callback in agent_data['callbacks']],
            save_dir=agent_data['save_dir'],
        )

        #TODO: ADD IF HER
    
    return agent

# def train_model(agent_data, env_name, num_episodes, render, render_freq, num_epochs=None, num_cycles=None, num_updates=None, workers=None):  
#     # print('Training agent...')
#     # agent = rl_agents.load_agent_from_config(save_dir)
#     agent = load(agent_data, env_name)
#     # print('Agent loaded.')
#     if agent_data['agent_type'] == "HER":
#         agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq)
#     else:
#         agent.train(num_episodes, render, render_freq)
#     # print('Training complete.')

# def test_model(agent_data, env_name, num_episodes, render, render_freq):  
#     # print('Testing agent...')
#     # agent = rl_agents.load_agent_from_config(save_dir)
#     agent = load(agent_data, env_name)
#     # print('Agent loaded.')
#     agent.test(num_episodes, render, render_freq)
#     # print('Testing complete.')

def delete_renders(folder_path):
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .mp4 or .meta.json extension
        if filename.endswith(".mp4") or filename.endswith(".meta.json"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Remove the file
            os.remove(file_path)

def get_video_files(page, agent_data):

    # get correct model folder to look in for renders
    agent_type = agent_data['agent_type']
    if agent_type == "HER":
        agent_type = agent_data['agent']['agent_type']
    
    # check if path exists
    if os.path.exists(f'assets/models/{agent_type}/renders'):
        try:
            if page == "/train-agent":
                # Get video files from training renders folder
                return natsorted([f for f in os.listdir(Path(f"assets/models/{agent_type}/renders/training")) if f.endswith('.mp4')])
            elif page == "/test-agent":
                # Get video files from testing renders folder
                return natsorted([f for f in os.listdir(Path(f"assets/models/{agent_type}/renders/testing")) if f.endswith('.mp4')])
        except Exception as e:
            print(f"Failed to get video files: {e}")

def reset_agent_status_data(data):
    data['data'] = None
    data['progress'] = 0.0
    data['status'] = "Pending"
    
    return data


## LAYOUT COMPONENT GENERATORS

# Function to generate carousel items from video paths
def generate_video_items(video_files, page, agent_data):
    if page == "/train-agent":
        folder = 'training'
    elif page == "/test-agent":
        folder = 'testing'
    else:
        raise ValueError(f"Invalid page {page}")
    
    # determine agent type to determine folder to look in for renders folder
    agent_type = agent_data['agent_type']
    if agent_type == "HER":
        agent_type = agent_data['agent']['agent_type']
    return [
        html.Video(src=f'assets/models/{agent_type}/renders/{folder}/{video_file}', controls=True,
                   style={'width': '100%', 'height': 'auto'},
                   id={
                       'type':'video',
                       'page':page,
                       'index':index
                   })
        for index, video_file in enumerate(natsorted(video_files))
    ]



# Component for file upload
def upload_component(page):
    return dcc.Upload(
        id={
            'type': 'upload-agent-config',
            'page': page,
        },
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )

def instantiate_envwrapper_obj(library:str, env_id:str, wrappers:list = None):
    # Instantiates an EnvWrapper object for an env of library
    if library == 'gymnasium':
        env = gym.make(env_id)
        print(f'env spec:{env.spec}')
        return GymnasiumWrapper(env.spec, wrappers)
        
    elif library == 'isaacsim':
        # Placeholder for IsaacSim environments
        pass

# Environment dropdown components
def env_dropdown_component(page):
    allowed_wrappers = get_wrappers_dropdown_options()
    layout = html.Div([
        html.H2("Select Environment"),
        dcc.Dropdown(
            id={
                'type': 'library-select',
                'page': page,
            },
            options=[
                {'label': 'Gymnasium', 'value': 'gymnasium'},
                {'label': 'IsaacSim', 'value': 'isaacsim'}
            ],
            value=None,
            placeholder="Select Environment Library"
        ),
        dcc.Dropdown(
            id={
                'type': 'env-dropdown',
                'page': page,
            },
            options=[],
            placeholder="Select Gym Environment",
            style={'display': 'none'},
        ),
        html.H2("Select Wrappers"),
        dcc.Dropdown(
            id={
                'type': 'gym_wrappers_dropdown',
                'page': page,
            },
            options=[{"label": camel_to_words(w), "value": w} for w in allowed_wrappers],
            multi=True,
            placeholder="Select one or more wrappers..."
        ),
        dcc.Tabs(id={"type":"wrapper-tabs", "page":page}, children=[]),

        # This store will hold user param overrides for those wrappers in the registry
        dcc.Store(id={"type":"wrappers_params_store", "page":page}, data={}),
    ])
    print(f'layer:{layout}')
    return layout

def get_all_gym_envs():
    """
    Returns a list of the latest gym environment versions, excluding:
      - Certain directories (phys2d/, tabular/, etc.).
      - Atari games that contain 'Deterministic', 'NoFrameskip', or '-ram' in their IDs.
      - All older versions if there's a newer version available.
    """
    # 1) Directories to exclude entirely:
    exclude = ["Gym", "phys2d/", "tabular/"]

    # 2) Collect all env specs, skipping excluded directories
    all_specs = [
        spec for spec in gym.envs.registry.values()
        if not any(spec.id.startswith(ex) for ex in exclude)
    ]

    # 3) Exclude Atari variants with 'Deterministic', 'NoFrameskip', or '-ram'
    #    If you ONLY want to exclude these for Atari envs, check if env is "Atari-like".
    #    In practice, though, those suffixes only appear for Atari anyway.
    def is_unwanted_atari_variant(env_id: str) -> bool:
        return (
            "Deterministic" in env_id
            or "NoFrameskip" in env_id
            or "-ram" in env_id
        )
    
    filtered_specs = []
    for spec in all_specs:
        # skip if it has the "unwanted Atari variant" substrings
        if is_unwanted_atari_variant(spec.id):
            continue
        filtered_specs.append(spec)

    # 4) Now parse the final list to group them by (base_name) -> version
    def parse_env_id(env_id: str):
        """
        e.g., "ALE/Breakout-v5" -> ("Breakout", 5)
              "Breakout-v4"     -> ("Breakout", 4)
              "CartPole-v1"     -> ("CartPole", 1)
              "ALE/Crossbow-v5" -> ("Crossbow", 5)
        """
        # Remove the directory prefix if it exists: "ALE/Breakout-v5" -> "Breakout-v5"
        final_name = env_id.split("/")[-1]
        if "-v" in final_name:
            base_name, version_str = final_name.rsplit("-v", 1)
            try:
                version = int(version_str)
            except ValueError:
                version = None
        else:
            base_name = final_name
            version = None
        return base_name, version

    # Group by base_name
    grouped = {}
    for spec in filtered_specs:
        base_name, version = parse_env_id(spec.id)
        grouped.setdefault(base_name, []).append((spec.id, version))

    # Pick the environment that has the highest version for each base_name
    def version_or_neg1(v):
        return v if v is not None else -1

    latest_envs = []
    for base_name, items in grouped.items():
        best_item = max(items, key=lambda x: version_or_neg1(x[1]))  # max by version
        latest_envs.append(best_item[0])  # just the env_id

    # print(latest_envs)

    # Sort so the list is stable and predictable
    return sorted(latest_envs)

def get_env_data(env_name):
    env_data = {
    "CartPole-v1": {
        "description": "Similar to CartPole-v0 but with a different set of parameters for a harder challenge.",
        "gif_url": "https://gymnasium.farama.org/_images/cart_pole.gif",
    },
    "MountainCar-v0": {
        "description": "A car is on a one-dimensional track between two mountains; the goal is to drive up the mountain on the right.",
        "gif_url": "https://gymnasium.farama.org/_images/mountain_car.gif",
    },
    "MountainCarContinuous-v0": {
        "description": "A continuous control version of the MountainCar environment.",
        "gif_url": "https://gymnasium.farama.org/_images/mountain_car.gif",
    },
    "Pendulum-v1": {
        "description": "Control a frictionless pendulum to keep it upright.",
        "gif_url": "https://gymnasium.farama.org/_images/pendulum.gif",
    },
    "Acrobot-v1": {
        "description": "A 2-link robot that swings up to reach a given height.",
        "gif_url": "https://gymnasium.farama.org/_images/acrobot.gif",
    },
    "LunarLander-v2": {
        "description": "Lunar Lander description",
        "gif_url": "https://gymnasium.farama.org/_images/lunar_lander.gif",
    },
    "LunarLanderContinuous-v2": {
        "description": "A continuous control version of the LunarLander environment.",
        "gif_url": "https://gymnasium.farama.org/_images/lunar_lander.gif",
    },
    "BipedalWalker-v3": {
        "description": "Control a two-legged robot to walk through rough terrain without falling.",
        "gif_url": "https://gymnasium.farama.org/_images/bipedal_walker.gif",
    },
    "BipedalWalkerHardcore-v3": {
        "description": "A more challenging version of BipedalWalker with harder terrain and obstacles.",
        "gif_url": "https://gymnasium.farama.org/_images/bipedal_walker.gif",
    },
    "CarRacing-v2": {
        "description": "A car racing environment where the goal is to complete a track as quickly as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/car_racing.gif",
    },
    "Blackjack-v1": {
        "description": "A classic Blackjack card game environment.",
        "gif_url": "https://gymnasium.farama.org/_images/blackjack1.gif",
    },
    "FrozenLake-v1": {
        "description": "Navigate a grid world to reach a goal without falling into holes, akin to crossing a frozen lake.",
        "gif_url": "https://gymnasium.farama.org/_images/frozen_lake.gif",
    },
    "FrozenLake8x8-v1": {
        "description": "An 8x8 version of the FrozenLake environment, providing a larger and more complex grid.",
        "gif_url": "https://gymnasium.farama.org/_images/frozen_lake.gif",
    },
    "CliffWalking-v0": {
        "description": "A grid-based environment where the agent must navigate cliffs to reach a goal.",
        "gif_url": "https://gymnasium.farama.org/_images/cliff_walking.gif",
    },
    "Taxi-v3": {
        "description": "A taxi must pick up and drop off passengers at designated locations.",
        "gif_url": "https://gymnasium.farama.org/_images/taxi.gif",
    },
    "Reacher-v4": {
        "description": "Control a robotic arm to reach a target location",
        "gif_url": "https://gymnasium.farama.org/_images/reacher.gif",
    },
    "Reacher-v5": {
        "description": "Control a robotic arm to reach a target location",
        "gif_url": "https://gymnasium.farama.org/_images/reacher.gif",
    },
    "Pusher-v4": {
        "description": "A robot arm needs to push objects to a target location.",
        "gif_url": "https://gymnasium.farama.org/_images/pusher.gif",
    },
    "Pusher-v5": {
        "description": "A robot arm needs to push objects to a target location.",
        "gif_url": "https://gymnasium.farama.org/_images/pusher.gif",
    },
    "InvertedPendulum-v4": {
        "description": "Balance a pendulum in the upright position on a moving cart",
        "gif_url": "https://gymnasium.farama.org/_images/inverted_pendulum.gif",
    },
    "InvertedPendulum-v5": {
        "description": "Balance a pendulum in the upright position on a moving cart",
        "gif_url": "https://gymnasium.farama.org/_images/inverted_pendulum.gif",
    },
    "InvertedDoublePendulum-v4": {
        "description": "A more complex version of the InvertedPendulum with two pendulums to balance.",
        "gif_url": "https://gymnasium.farama.org/_images/inverted_double_pendulum.gif",
    },
    "InvertedDoublePendulum-v5": {
        "description": "A more complex version of the InvertedPendulum with two pendulums to balance.",
        "gif_url": "https://gymnasium.farama.org/_images/inverted_double_pendulum.gif",
    },
    "HalfCheetah-v4": {
        "description": "Control a 2D cheetah robot to make it run as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/half_cheetah.gif",
    },
    "HalfCheetah-v5": {
        "description": "Control a 2D cheetah robot to make it run as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/half_cheetah.gif",
    },
    "Hopper-v4": {
        "description": "Make a two-dimensional one-legged robot hop forward as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/hopper.gif",
    },
    "Hopper-v5": {
        "description": "Make a two-dimensional one-legged robot hop forward as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/hopper.gif",
    },
    "Swimmer-v4": {
        "description": "Control a snake-like robot to make it swim through water.",
        "gif_url": "https://gymnasium.farama.org/_images/swimmer.gif",
    },
    "Swimmer-v5": {
        "description": "Control a snake-like robot to make it swim through water.",
        "gif_url": "https://gymnasium.farama.org/_images/swimmer.gif",
    },
    "Walker2d-v4": {
        "description": "A bipedal robot walking simulation aiming to move forward as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/walker2d.gif",
    },
    "Walker2d-v5": {
        "description": "A bipedal robot walking simulation aiming to move forward as fast as possible.",
        "gif_url": "https://gymnasium.farama.org/_images/walker2d.gif",
    },
    "Ant-v4": {
        "description": "Control a four-legged robot to explore a terrain.",
        "gif_url": "https://gymnasium.farama.org/_images/ant.gif",
    },
    "Ant-v5": {
        "description": "Control a four-legged robot to explore a terrain.",
        "gif_url": "https://gymnasium.farama.org/_images/ant.gif",
    },
    "Humanoid-v4": {
        "description": "A two-legged humanoid robot that learns to walk and balance.",
        "gif_url": "https://gymnasium.farama.org/_images/humanoid.gif",
    },
    "Humanoid-v5": {
        "description": "A two-legged humanoid robot that learns to walk and balance.",
        "gif_url": "https://gymnasium.farama.org/_images/humanoid.gif",
    },
    "HumanoidStandup-v4": {
        "description": "The goal is to make a humanoid stand up from a prone position.",
        "gif_url": "https://gymnasium.farama.org/_images/humanoid_standup.gif",
    },
    "HumanoidStandup-v5": {
        "description": "The goal is to make a humanoid stand up from a prone position.",
        "gif_url": "https://gymnasium.farama.org/_images/humanoid_standup.gif",
    },
    "FetchReach-v2": {
        "description": "The goal in the environment is for a manipulator to move the end effector to a randomly \
            selected position in the robots workspace.",
        "gif_url": "https://robotics.farama.org/_images/reach.gif",
    },
    "FetchSlide-v2": {
        "description": "The task in the environment is for a manipulator hit a puck in order to reach a target \
            position on top of a long and slippery table.",
        "gif_url": "https://robotics.farama.org/_images/slide.gif",
    },
    "FetchPickAndPlace-v2": {
        "description": "The goal in the environment is for a manipulator to move a block to a target position on \
            top of a table or in mid-air.",
        "gif_url": "https://robotics.farama.org/_images/pick_and_place.gif",
    },
    "FetchPush-v2": {
        "description": "The goal in the environment is for a manipulator to move a block to a target position on \
            top of a table by pushing with its gripper.",
        "gif_url": "https://robotics.farama.org/_images/push.gif",
    },
    "HandReach-v1": {
        "description": "The goal of the task is for the fingertips of the hand to reach a predefined \
            target Cartesian position.",
        "gif_url": "https://robotics.farama.org/_images/reach1.gif",
    },
    "HandManipulateBlockRotateZ-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. There is a random target rotation \
            around the z axis of the block. No target position. Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlockRotateZ_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockRotateZ_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockRotateParallel-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. There is a random target rotation around \
            the z axis of the block and axis-aligned target rotations for the x and y axes. \
            No target position. Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlockRotateParallel_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockRotateXYZ-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. There is a random target rotation for all \
            axes of the block. No target position. Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlockFull-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. There is a Random target rotation \
            for all axes of the block. Random target position. Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlock-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. Rewards are Sparse",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlockDense-v1": {
        "description": "In this task a block is placed on the palm of the hand. The task is to then \
            manipulate the block such that a target pose is achieved. Rewards are Dense",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block.gif",
    },
    "HandManipulateBlock_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateBlock_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateBlock environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_block_touch_sensors.gif",
    },
    "HandManipulateEgg-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEggDense-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            Rewards are dense.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEggRotate-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            There is a random target rotation for all axes of the egg. No target position. \
            Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEggRotateDense-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            There is a random target rotation for all axes of the egg. No target position. \
            Rewards are dense.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEggFull-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            There is a random target rotation for all axes of the egg. Random target position. \
            Rewards are sparse.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEggFullDense-v1": {
        "description": "The task to be solved is very similar to that in the HandManipulateBlock \
            environment, but in this case an egg-shaped object is placed on the palm of the hand. \
            The task is to then manipulate the object such that a target pose is achieved. \
            There is a random target rotation for all axes of the egg. Random target position. \
            Rewards are dense.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg.gif",
    },
    "HandManipulateEgg_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggRotate_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            No target position. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggFull_ContinuousTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            Random target position. Rewards are sparse. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEgg_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggRotate_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            No target position. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggFull_BooleanTouchSensors-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            Random target position. Rewards are sparse. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEgg_ContinuousTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are dense. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggRotate_ContinuousTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            No target position. Rewards are dense. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggFull_ContinuousTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            Random target position. Rewards are dense. Continuous action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEgg_BooleanTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. Rewards are dense. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggRotate_BooleanTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            No target position. Rewards are dense. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
    "HandManipulateEggFull_BooleanTouchSensorsDense-v1": {
        "description": "The task to be solved is the same as in the HandManipulateEgg environment. \
            However, in this case the environment observation also includes tactile sensory information. \
            This is achieved by placing a total of 92 MuJoCo touch sensors in the palm and finger \
            phalanxes of the hand. There is a random target rotation for all axes of the egg. \
            Random target position. Rewards are dense. Discrete action space.",
        "gif_url": "https://robotics.farama.org/_images/manipulate_egg_touch_sensors.gif",
    },
}
    description = env_data[env_name]['description']
    gif_url = env_data[env_name]['gif_url']

    return description, gif_url

def update_run_options(agent_type, page):
    if agent_type == 'PPO':
        return create_ppo_run_options(page)
    elif agent_type == 'HER':
        return create_her_run_options(page)
    elif agent_type == 'Reinforce':
        return create_reinforce_run_options(page)
    elif agent_type == 'ActorCritic':
        return create_actor_critic_run_options(page)
    elif agent_type == 'DDPG':
        return create_ddpg_run_options(page)
    elif agent_type == 'TD3':
        return create_td3_run_options(page)
    else:
        # return some default or "unknown agent type" message
        return html.Div([
            html.P(f"No options found for agent type: {agent_type}")
        ])


def create_ppo_run_options(page):
    return html.Div(
        [
            create_num_timesteps_component(page),
            create_trajectory_length_component(page),
            create_batch_size_component(page),
            create_learning_epochs_component(page),
            *create_common_run_components(page)
        ],
    ),


def create_her_run_options(page):
    return html.Div([
        dcc.Input(
            id={'type': 'epochs', 'page': page},
            type='number', min=1, placeholder="Number of Epochs"
        ),
        dcc.Input(
            id={'type': 'cycles', 'page': page},
            type='number', min=1, placeholder="Number of Cycles"
        ),
        # ...
    ], style={'border': '1px solid #ccc', 'padding': '10px'})


def create_reinforce_run_options(page):
    components = []
    components.append(create_num_episodes_component(page))
    if page == '/train-agent':
        components.append(create_batch_size_component(page))
    
    return html.Div([
        *components,
        *create_common_run_components(page)
    ])

def create_actor_critic_run_options(page):
    components = []
    components.append(create_num_episodes_component(page))
    
    return html.Div([
        *components,
        *create_common_run_components(page)
    ])

def create_ddpg_run_options(page):
    components = []
    components.append(create_num_episodes_component(page))
    
    return html.Div([
        *components,
        *create_common_run_components(page)
    ])

def create_td3_run_options(page):
    components = []
    components.append(create_num_episodes_component(page))
    
    return html.Div([
        *components,
        *create_common_run_components(page)
    ])

def create_num_timesteps_component(page):
    return dcc.Input(
                id={
                    'type': 'num-timesteps',
                    'page': page,
                },
                type='number',
                placeholder="Number of Timesteps",
                min=1,
            )
def create_trajectory_length_component(page):
    return dcc.Input(
                id={
                    'type': 'traj-length',
                    'page': page,
                },
                type='number',
                placeholder="Trajectories Length (timesteps)",
                min=1,
            )

def create_batch_size_component(page):
    return dcc.Input(
                id={
                    'type': 'batch-size',
                    'page': page,
                },
                type='number',
                placeholder="Batch Size",
                min=1,
            )

def create_learning_epochs_component(page):
    return dcc.Input(
                id={
                    'type': 'learning-epochs',
                    'page': page,
                },
                type='number',
                placeholder="Learning Epochs",
                min=1,
            )

def create_num_episodes_component(page):
    return dcc.Input(
                    id={
                        'type': 'num-episodes',
                        'page': page,
                    },
                    type='number',
                    placeholder="Number of Episodes",
                    min=1,
                )

def create_num_envs_component(page):
    return dcc.Input(
            id={
                'type': 'num-envs',
                'page': page,
            },
            type='number',
            placeholder="Number of Envs",
            min=1,
        )

def create_load_weights_component(page):
    return dcc.Checklist(
            options=[
                {'label': 'Load Weights', 'value': True}
            ],
            id={
                'type': 'load-weights',
                'page': page,
            },
        )

def create_render_episode_component(page):
    return html.Div([
        html.Label('Render Episodes'),
        dcc.RadioItems(
            id={
                'type': 'render-option',
                'page': page,
            },
            options=[
                {'label': 'True', 'value': True},
                {'label': 'False', 'value': False},
            ],
            value=False,
            style={'margin-left': '10px'},
        ),
        html.Div(
            id = {
                'type':'render-block',
                'page':page,
            },
            children = [
                html.Label('Render Frequency'),
                dcc.Input(
                    id={
                        'type': 'render-freq',
                        'page': page,
                    },
                    type='number',
                    placeholder="Every 'n' Episodes",
                    min=1,
                )
            ],
            style={'margin-left': '10px', 'display':'none'}
        )
    ])
        

def create_common_run_components(page):
    common = []
    # if page == '/train-agent':
    common.append(create_num_envs_component(page))
    common.append(create_seed_component(page))
    common.append(create_load_weights_component(page))
    common.append(create_render_episode_component(page))
    # Create hidden div to serve as dummy output for train callback
    hidden = html.Div(
            id={
                'type':'hidden-div',
                'page':page,
            },
            style={'display': 'none'}
        )
    common.append(hidden)

    return common

# Training settings component
def run_agent_settings_component(page, agent_type=None):
    return html.Div([
        html.Div(
            id={
                'type': 'her-options',
                'page': page,
            },
            style={'display': 'none'},
            children=[
                dcc.Input(
                    id={
                        'type': 'epochs',
                        'page': page,
                    },
                    type='number',
                    placeholder="Number of Epochs",
                    min=1,
                ),
                dcc.Input(
                    id={
                        'type': 'cycles',
                        'page': page,
                    },
                    type='number',
                    placeholder="Number of Cycles",
                    min=1,
                ),
                dcc.Input(
                    id={
                        'type': 'learning-cycles',
                        'page': page,
                    },
                    type='number',
                    placeholder="Number of Learning Cycles",
                    min=1,
                ),
            ]
        ),
        html.Div(
            id={
                'type':'hidden-div',
                'page':page,
            },
            style={'display': 'none'}
        ),
    ])

# custom carousel for video playback
# def video_carousel_component(page, video_files=[]):
#     return html.Div([
#         html.Div(id={'type':'video-carousel', 'page':page}, children=[],
#                  style={'display': 'flex', 'overflowX': 'scroll', 'justifyContent': 'center', 'alignItems': 'center'}),
#         html.Div([
#             html.Button('Prev', id='prev-btn', n_clicks=0, style={'marginRight': '10px'}),
#             html.Button('Next', id='next-btn', n_clicks=0),
#         ], style={'textAlign': 'center', 'marginTop': '20px'}),
#     ], style={'width': '60%', 'margin': 'auto'})
def video_carousel_component(page):
    return html.Div([
        dcc.Store(id={'type': 'video-carousel-store', 'page': page}, data={'current_video': 0, 'video_list':[]}),
        html.Div(id={'type': 'video-carousel', 'page': page}),
        html.Div(id={'type': 'video-filename', 'page': page}, style={'textAlign': 'center', 'margin': '10px 0'}),
        html.Div([
            html.Button('Prev', id={'type': 'prev-btn', 'page': page}, n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Next', id={'type': 'next-btn', 'page': page}, n_clicks=0),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
    ], style={'width': '60%', 'margin': 'auto'})

# AGENT HYPERPARAM COMPONENTS #

def create_device_input(agent_type):
    return html.Div(
        [
            # html.Label('Computation Device', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'device',
                    'model':'none',
                    'agent':agent_type
                },
                options=[
                    {'label': 'CPU', 'value': 'cpu'},
                    {'label': 'CUDA', 'value': 'cuda'},
                ],
                placeholder='Computation Device'
            )
        ]
    )

def create_save_dir_input(agent_type):
    return html.Div(
        [
            html.Label('Save Directory:', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'save-dir',
                    'model':'none',
                    'agent':agent_type
                },
                type='text',
                placeholder='path/to/model'
            )
        ]
    )


def get_kernel_initializer_inputs(selected_initializer, initializer_id, agent_params):
    # Dictionary mapping the initializer names to the corresponding function
    #DEBUG
    print(f'selected initializer:{selected_initializer}, initializer_id:{initializer_id}')
    initializer_input_creators = {
        "variance_scaling": create_variance_scaling_inputs,
        "constant": create_constant_initializer_inputs,
        "normal": create_normal_initializer_inputs,
        "uniform": create_uniform_initializer_inputs,
        "truncated_normal": create_truncated_normal_initializer_inputs,
        "xavier_uniform": create_xavier_uniform_initializer_inputs,
        "xavier_normal": create_xavier_normal_initializer_inputs,
        "kaiming_uniform": create_kaiming_uniform_initializer_inputs,
        "kaiming_normal": create_kaiming_normal_initializer_inputs,
    }
    
    # Call the function associated with the selected_initializer,
    # or return an empty html.Div() if not found
    if selected_initializer in initializer_input_creators:
        # return initializer_input_creators.get(selected_initializer, lambda: html.Div())(initializer_id)
        return initializer_input_creators.get(selected_initializer)(initializer_id, agent_params)
    elif selected_initializer not in ['ones', 'zeros', 'orthogonal', 'default']:
        raise ValueError(f"{selected_initializer} not in initializer input creator dict")

def create_kaiming_normal_initializer_inputs(initializer_id, agent_params):
    """Component for kaiming uniform initializer hyperparameters"""
    # return html.Div(
    #     id={
    #         'type': 'kernel-params',
    #         'model': initializer_id['model'],
    #         'agent': initializer_id['agent']
    #         },
    children=[
        html.Label('Mode', style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id={
                'type':'mode',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
                },
            options=[
                    {'label': 'fan in', 'value': 'fan_in'},
                    {'label': 'fan out', 'value': 'fan_out'},
                ],
            value=agent_params.get(get_key(initializer_id, 'mode'), 'fan_in'),
        ),
        html.Hr(),
    ]
    return children
    


def create_kaiming_uniform_initializer_inputs(initializer_id, agent_params):
    """Component for kaiming uniform initializer hyperparameters"""
    children=[
        html.Label('Mode', style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id={
                'type':'mode',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
                },
            options=[
                    {'label': 'fan in', 'value': 'fan_in'},
                    {'label': 'fan out', 'value': 'fan_out'},
                ],
            value=agent_params.get(get_key(initializer_id, 'mode'), 'fan_in'),
        ),
        html.Hr(),
    ]
    return children
                    


def create_xavier_normal_initializer_inputs(initializer_id, agent_params):
    """Component for xavier uniform initializer hyperparameters"""
    children=[
        html.Label('Gain', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
                'type':'gain',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
            },
            type='number',
            min=1.0,
            max=3.0,
            step=1.0,
            value=agent_params.get(get_key(initializer_id, 'gain'), 1.0)
        ),
        html.Hr(),
    ]
    return children



def create_xavier_uniform_initializer_inputs(initializer_id, agent_params):
    """Component for xavier uniform initializer hyperparameters"""
    children=[
        html.Label('Gain', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
                'type':'gain',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
            },
            type='number',
            min=1.0,
            max=3.0,
            step=1.0,
            value=agent_params.get(get_key(initializer_id, 'gain'), 1.0)
        ),
        html.Hr(),
    ]
    return children

def create_truncated_normal_initializer_inputs(initializer_id, agent_params):
    """Component for truncated normal initializer hyperparameters"""
    children=[
        html.Label('Mean', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'mean',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=0.00,
            max=1.00,
            step=0.01,
            value=agent_params.get(get_key(initializer_id, 'mean'), 0.00)
        ),

        html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'std',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=0.01,
            max=3.00,
            step=0.01,
            value=agent_params.get(get_key(initializer_id, 'std'), 1.00)
        ),
        html.Hr(),
    ]
    return children


def create_uniform_initializer_inputs(initializer_id, agent_params):
    """Component for random uniform initializer hyperparameters"""
    children=[
        html.Label('Minimum', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'a',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=-1.000,
            max=1.000,
            step=0.001,
            value=agent_params.get(get_key(initializer_id, 'a'), -1.000)
        ),

        html.Label('Maximum', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'b',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=-1.000,
            max=1.000,
            step=0.001,
            value=agent_params.get(get_key(initializer_id, 'b'), 1.000)
        ),
        html.Hr(),
    ]
    return children


def create_normal_initializer_inputs(initializer_id, agent_params):
    """Component for random normal initializer hyperparameters"""
    children=[
        html.Label('Mean', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'mean',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=-1.00,
            max=1.00,
            step=0.01,
            value=agent_params.get(get_key(initializer_id, 'mean'), 0.0)
        ),

        html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'std',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=0.01,
            max=2.00,
            step=0.01,
            value=agent_params.get(get_key(initializer_id, 'std'), 1.0)
        ),
        html.Hr(),
    ]
    return children


def create_constant_initializer_inputs(initializer_id, agent_params):
    """Component for constant initializer hyperparameters"""
    children=[
        html.Label('Value', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'val',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=0.001,
            max=0.99,
            step=0.001,
            value=agent_params.get(get_key(initializer_id, 'val'), 1.0)
        ),
        html.Hr(),
    ]
    return children


def create_variance_scaling_inputs(initializer_id, agent_params):
    """Component for variance scaling initializer hyperparameters"""
    children=[
        html.Label('Scale', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
            'type':'scale',
            'model':initializer_id['model'],
            'agent':initializer_id['agent'],
            'index':initializer_id['index'],
            },
            type='number',
            min=1.0,
            max=5.0,
            step=1.0,
            value=agent_params.get(get_key(initializer_id, 'scale'), 2.0)
        ),
        
        html.Label('Mode', style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id={
                'type':'mode',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
            },
            options=[{'label': mode, 'value': mode} for mode in ['fan_in', 'fan_out', 'fan_avg']],
            placeholder="Mode",
            value=agent_params.get(get_key(initializer_id, 'mode'), 'fan_in')
        ),
        
        html.Label('Distribution', style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id={
                'type':'distribution',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                'index':initializer_id['index'],
            },
            options=[{'label': dist, 'value': dist} for dist in ['truncated_normal', 'uniform']],
            placeholder="Distribution",
            value=agent_params.get(get_key(initializer_id, 'distribution'), 'truncated_normal')
        ),
        html.Hr(),
    ]
    return children

def format_output_kernel_initializer_config(model_type, agent_type, agent_params):
    """Returns an initializer object based on initializer component values"""

    initializer_type = agent_params.get(get_key({'type':'kernel-init', 'model':model_type, 'agent':agent_type, 'index':0}))

    # create empty dictionary to store initializer config params
    config = {}
    # Iterate over initializer_type params list and get values
    if initializer_type not in ['zeros', 'ones', 'default']: 
        for param in get_kernel_params_map()[initializer_type]:
            config[param] = agent_params.get(get_key({'type':param, 'model':model_type, 'agent':agent_type, 'index':0}))
    
    # Get distribution type in order to format output correctly
    distribution = agent_params.get(get_key({'type':'distribution', 'model':'none', 'agent':agent_type}), None)
    # format
    if distribution is None or distribution == 'categorical':
        initializer_config = [{'type':'dense', 'params':{'kernel':initializer_type, 'kernel params':config}}]
    # elif distribution == 'categorical':
    #     initializer_config = [{'type':'dense', 'params':{'kernel':initializer_type, 'kernel params':config}}]
    elif distribution in ['beta','normal']:
        initializer_config = [{'type':'dense', 'params':{'kernel':initializer_type, 'kernel params':config}},
                              {'type':'dense', 'params':{'kernel':initializer_type, 'kernel params':config}}]

    return initializer_config

def create_discount_factor_input(agent_type):
    return html.Div(
        [
            html.Label('Discount Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'discount',
                    'model':'none',
                    'agent':agent_type
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99
            )
        ]
    )

def create_advantage_coeff_input(agent_type):
    return html.Div(
        [
            html.Label('Advantage Coefficient', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'advantage-coeff',
                    'model':'none',
                    'agent':agent_type
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.95
            )
        ]
    )

def create_surrogate_loss_clip_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Loss Clip', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip',
                    'model':model_type,
                    'agent':agent_type
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.2,
            ),
            create_surrogate_loss_clip_scheduler_input(agent_type, model_type)

        ]
    )

def create_surrogate_loss_clip_scheduler_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Clip Scheduler', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'surrogate-clip-scheduler',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Step', 'Exponential', 'CosineAnnealing', 'Linear', 'None']],
                placeholder=f"{model_type.capitalize()} Clip Scheduler",
            ),
            html.Div(
                id={
                    'type':'surrogate-clip-scheduler-options',
                    'model':model_type,
                    'agent':agent_type,
                }
            )
        ]
    )

def update_surrogate_loss_clip_scheduler_options(agent_type, model_type, surr_clip_scheduler):
    if surr_clip_scheduler == 'step':
        return surrogate_loss_clip_step_scheduler_options(agent_type, model_type)
    elif surr_clip_scheduler == 'exponential':
        return surrogate_loss_clip_exponential_scheduler_options(agent_type, model_type)
    elif surr_clip_scheduler == 'cosineannealing':
        return surrogate_loss_clip_cosineannealing_scheduler_options(agent_type, model_type)
    elif surr_clip_scheduler == 'linear':
        return surrogate_loss_clip_linear_scheduler_options(agent_type, model_type)
    return html.Div()

def surrogate_loss_clip_cosineannealing_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('T max (max iters)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-t-max',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=10000,
                step=1,
                value=1000,
            ),
            html.Label('Eta min (min LR)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-eta-min',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.000001,
                max=0.1,
                step=0.000001,
                value=0.0001,
            ),
        ]
    )

def surrogate_loss_clip_exponential_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def surrogate_loss_clip_step_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Step Size', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-step-size',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1000,
                step=1,
                value=100,
            ),
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def surrogate_loss_clip_linear_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Start Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-start-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=1.0,
                step=0.01,
                value=1.0,
            ),
            html.Label('End Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-end-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0001,
                max=1.0000,
                step=0.0001,
                value=0.0010,
            ),
            html.Label('Total Iterations', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'surrogate-clip-total-iters',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1e8,
                step=1,
                value=1e3,
            ),
        ]
    )

def get_surrogate_loss_clip_scheduler(model_type, agent_type, agent_params):

    scheduler = agent_params.get(get_key({'type':'surrogate-clip-scheduler', 'model':model_type, 'agent':agent_type}))
    if scheduler == 'none':
        return None
    params = {}
    if scheduler == 'step':
        params['step_size'] = agent_params.get(get_key({'type':'surrogate-clip-step-size', 'model':model_type, 'agent':agent_type}))
        params['gamma'] = agent_params.get(get_key({'type':'surrogate-clip-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'exponential':
        params['gamma'] = agent_params.get(get_key({'type':'surrogate-clip-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'cosineannealing':
        params['T_max'] = agent_params.get(get_key({'type':'surrogate-clip-t-max', 'model':model_type, 'agent':agent_type}))
        params['eta_min'] = agent_params.get(get_key({'type':'surrogate-clip-eta-min', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'linear':
        params['start_factor'] = agent_params.get(get_key({'type':'surrogate-clip-start-factor', 'model':model_type, 'agent':agent_type}))
        params['end_factor'] = agent_params.get(get_key({'type':'surrogate-clip-end-factor', 'model':model_type, 'agent':agent_type}))
        params['total_iters'] = agent_params.get(get_key({'type':'surrogate-clip-total-iters', 'model':model_type, 'agent':agent_type}))

    return {'type':scheduler, 'params':params}

# def create_value_clip_input(agent_type):
#     return html.Div(
#         [
#             html.Label('Value Loss Clip', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip',
#                     'model':'value',
#                     'agent':agent_type
#                 },
#                 type='number',
#                 min=0.01,
#                 max=0.99,
#                 step=0.01,
#                 value=0.2,
#             ),
#             create_value_clip_scheduler_input(agent_type)

#         ]
#     )

# def create_value_clip_scheduler_input(agent_type):
#     return html.Div(
#         [
#             html.Label('Value Clip Scheduler', style={'text-decoration': 'underline'}),
#             dcc.Dropdown(
#                 id={
#                     'type':'value-clip-scheduler',
#                     'model':'value',
#                     'agent':agent_type,
#                 },
#                 options=[{'label': i, 'value': i.lower()} for i in ['Step', 'Exponential', 'CosineAnnealing', 'Linear', 'None']],
#                 placeholder="Value Clip Scheduler",
#             ),
#             html.Div(
#                 id={
#                     'type':'value-clip-scheduler-options',
#                     'model':'value',
#                     'agent':agent_type,
#                 }
#             )
#         ]
#     )

# def update_value_clip_scheduler_options(agent_type, model_type, lr_scheduler):
#     if lr_scheduler == 'step':
#         return value_clip_step_scheduler_options(agent_type, model_type)
#     elif lr_scheduler == 'exponential':
#         return value_clip_exponential_scheduler_options(agent_type, model_type)
#     elif lr_scheduler == 'cosineannealing':
#         return value_clip_cosineannealing_scheduler_options(agent_type, model_type)
#     elif lr_scheduler == 'linear':
#         return value_clip_linear_scheduler_options(agent_type, model_type)
#     return html.Div()

# def value_clip_cosineannealing_scheduler_options(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label('T max (max iters)', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-t-max',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=1,
#                 max=10000,
#                 step=1,
#                 value=1000,
#             ),
#             html.Label('Eta min (min LR)', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-eta-min',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=0.000001,
#                 max=0.1,
#                 step=0.000001,
#                 value=0.0001,
#             ),
#         ]
#     )

# def value_clip_exponential_scheduler_options(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-gamma',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=0.01,
#                 max=0.99,
#                 step=0.01,
#                 value=0.99,
#             ),
#         ]
#     )

# def value_clip_step_scheduler_options(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label('Step Size', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-step-size',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=1,
#                 max=1000,
#                 step=1,
#                 value=100,
#             ),
#             html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-gamma',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=0.01,
#                 max=0.99,
#                 step=0.01,
#                 value=0.99,
#             ),
#         ]
#     )

# def value_clip_linear_scheduler_options(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label('Start Factor', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-start-factor',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=0.01,
#                 max=1.0,
#                 step=0.01,
#                 value=1.0,
#             ),
#             html.Label('End Factor', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-end-factor',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=0.01,
#                 max=1.00,
#                 step=0.01,
#                 value=0.01,
#             ),
#             html.Label('Total Iterations', style={'text-decoration': 'underline'}),
#             dcc.Input(
#                 id={
#                     'type':'value-clip-total-iters',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 type='number',
#                 min=1,
#                 max=1e8,
#                 step=1,
#                 value=1e3,
#             ),
#         ]
#     )

# def get_value_clip_scheduler(model_type, agent_type, agent_params):

#     scheduler = agent_params.get(get_key({'type':'value-clip-scheduler', 'model':model_type, 'agent':agent_type}))
#     if scheduler == 'none':
#         return None
#     params = {}
#     if scheduler == 'step':
#         params['step_size'] = agent_params.get(get_key({'type':'value-clip-step-size', 'model':model_type, 'agent':agent_type}))
#         params['gamma'] = agent_params.get(get_key({'type':'value-clip-gamma', 'model':model_type, 'agent':agent_type}))
#     elif scheduler == 'exponential':
#         params['gamma'] = agent_params.get(get_key({'type':'value-clip-gamma', 'model':model_type, 'agent':agent_type}))
#     elif scheduler == 'cosineannealing':
#         params['T_max'] = agent_params.get(get_key({'type':'value-clip-t-max', 'model':model_type, 'agent':agent_type}))
#         params['eta_min'] = agent_params.get(get_key({'type':'value-clip-eta-min', 'model':model_type, 'agent':agent_type}))
#     elif scheduler == 'linear':
#         params['start_factor'] = agent_params.get(get_key({'type':'value-clip-start-factor', 'model':model_type, 'agent':agent_type}))
#         params['end_factor'] = agent_params.get(get_key({'type':'value-clip-end-factor', 'model':model_type, 'agent':agent_type}))
#         params['total_iters'] = agent_params.get(get_key({'type':'value-clip-total-iters', 'model':model_type, 'agent':agent_type}))

#     return {'type':scheduler, 'params':params}

def create_grad_clip_input(agent_type, model_type):
    return html.Div(
        [
            # html.Label('Normalize Advantage', style={'text-decoration': 'underline'}),
            dcc.Checklist(
                id={'type':f'clip-grad',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': f'Clip {model_type.capitalize()} Gradient', 'value': True},
                ],
                # value=False
                style={'display':'inline-block'}
            ),
            html.Div(
                id = {
                    'type':f'grad-clip-block',
                    'model':model_type,
                    'agent':agent_type,
                },
                children = [
                    html.Label(f'{model_type.capitalize()} Gradient Clip'),
                    dcc.Input(
                        id={
                            'type': 'grad-clip',
                            'model': model_type,
                            'agent': agent_type
                        },
                        type='number',
                        min=0.01,
                        max=999.0,
                        value=999.0,
                        step=0.01,
                    )
                ],
                style={'display':'none', 'margin-left': '20px'}
            )
        ]
    )

def create_entropy_input(agent_type):
    return html.Div(
        [
            html.Label('Entropy Coefficient', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-coeff',
                    'model':'none',
                    'agent':agent_type
                },
                type='number',
                min=0.000,
                max=0.990,
                step=0.0001,
                value=0.001,
            ),
            create_entropy_scheduler_input(agent_type)
        ]
    )

def create_entropy_scheduler_input(agent_type):
    return html.Div(
        [
            html.Label('Entropy Scheduler', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'entropy-scheduler',
                    'model':'none',
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Step', 'Exponential', 'CosineAnnealing', 'Linear', 'None']],
                placeholder="Entropy Coefficient Scheduler",
            ),
            html.Div(
                id={
                    'type':'entropy-scheduler-options',
                    'model':'none',
                    'agent':agent_type,
                }
            )
        ]
    )

def update_entropy_scheduler_options(agent_type, model_type, lr_scheduler):
    if lr_scheduler == 'step':
        return entropy_step_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'exponential':
        return entropy_exponential_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'cosineannealing':
        return entropy_cosineannealing_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'linear':
        return entropy_linear_scheduler_options(agent_type, model_type)
    return html.Div()

def entropy_cosineannealing_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('T max (max iters)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-t-max',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=10000,
                step=1,
                value=1000,
            ),
            html.Label('Eta min (min LR)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-eta-min',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.000001,
                max=0.1,
                step=0.000001,
                value=0.0001,
            ),
        ]
    )


def entropy_exponential_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def entropy_step_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Step Size', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-step-size',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1000,
                step=1,
                value=100,
            ),
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def entropy_linear_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Start Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-start-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=1.0,
                step=0.01,
                value=1.0,
            ),
            html.Label('End Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-end-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=1.00,
                step=0.01,
                value=0.01,
            ),
            html.Label('Total Iterations', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'entropy-total-iters',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1e8,
                step=1,
                value=1e3,
            ),
        ]
    )

def get_entropy_scheduler(model_type, agent_type, agent_params):

    scheduler = agent_params.get(get_key({'type':'entropy-scheduler', 'model':model_type, 'agent':agent_type}))
    if scheduler == 'none':
        return None
    params = {}
    if scheduler == 'step':
        params['step_size'] = agent_params.get(get_key({'type':'entropy-step-size', 'model':model_type, 'agent':agent_type}))
        params['gamma'] = agent_params.get(get_key({'type':'entropy-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'exponential':
        params['gamma'] = agent_params.get(get_key({'type':'entropy-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'cosineannealing':
        params['T_max'] = agent_params.get(get_key({'type':'entropy-t-max', 'model':model_type, 'agent':agent_type}))
        params['eta_min'] = agent_params.get(get_key({'type':'entropy-eta-min', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'linear':
        params['start_factor'] = agent_params.get(get_key({'type':'entropy-start-factor', 'model':model_type, 'agent':agent_type}))
        params['end_factor'] = agent_params.get(get_key({'type':'entropy-end-factor', 'model':model_type, 'agent':agent_type}))
        params['total_iters'] = agent_params.get(get_key({'type':'entropy-total-iters', 'model':model_type, 'agent':agent_type}))

    return {'type':scheduler, 'params':params}

def create_kl_coeff_input(agent_type):
    return html.Div(
        [
            html.Label('KL Coefficient', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'kl-coeff',
                    'model':'none',
                    'agent':agent_type
                },
                type='number',
                min=0.00,
                max=10.00,
                step=0.01,
                value=3.00,
            ),
            html.Div(
                [
                    html.Label('Use Adaptive KL'),
                    dcc.RadioItems(
                        id={
                            'type': 'adaptive-kl',
                            'model': 'none',
                            'agent': agent_type,
                        },
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False},
                        ],
                        value=False,
                        style={'margin-left': '10px'},
                    ),
                    html.Div(
                        id = {
                            'type':'adaptive-kl-block',
                            'model':'none',
                            'agent':agent_type,
                        },
                        style={'margin-left': '20px'}
                    )
                ],
            )
        ]
    )

def create_adaptive_kl_options(agent_type):
    return html.Div([
        html.Label('Target KL'),
        dcc.Input(
            id={
                'type': 'adaptive-kl-target-kl',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0.0,
            max=1.0,
            value=0.01,
            step=0.0001,
        ),
        html.Label('Scale Up'),
        dcc.Input(
            id={
                'type': 'adaptive-kl-scale-up',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0.0,
            max=3.0,
            value=2.0,
            step=0.1,
        ),
        html.Label('Scale Down'),
        dcc.Input(
            id={
                'type': 'adaptive-kl-scale-down',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0.0,
            max=3.0,
            value=0.5,
            step=0.1,
        ),
        html.Label('High Tolerance'),
        dcc.Input(
            id={
                'type': 'adaptive-kl-tolerance-high',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0.0,
            max=3.0,
            value=1.5,
            step=0.1,
        ),
        html.Label('Low Tolerance'),
        dcc.Input(
            id={
                'type': 'adaptive-kl-tolerance-low',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0.0,
            max=3.0,
            value=0.5,
            step=0.1,
        )
    ])

def get_kl_adapter(model_type, agent_type, agent_params):
    adapt_kl = agent_params.get(get_key({'type':'adaptive-kl', 'model':model_type, 'agent':agent_type}))
    if not adapt_kl:
        return None
    params = {}
    params['initial_beta'] = agent_params.get(get_key({'type':'kl-coeff', 'model':model_type, 'agent':agent_type}))
    params['target_kl'] = agent_params.get(get_key({'type':'adaptive-kl-target-kl', 'model':model_type, 'agent':agent_type}))
    params['scale_up'] = agent_params.get(get_key({'type':'adaptive-kl-scale-up', 'model':model_type, 'agent':agent_type}))
    params['scale_down'] = agent_params.get(get_key({'type':'adaptive-kl-scale-down', 'model':model_type, 'agent':agent_type}))
    params['kl_tolerance_high'] = agent_params.get(get_key({'type':'adaptive-kl-tolerance-high', 'model':model_type, 'agent':agent_type}))
    params['kl_tolerance_low'] = agent_params.get(get_key({'type':'adaptive-kl-tolerance-low', 'model':model_type, 'agent':agent_type}))

    #DEBUG
    print(f'kl adapter params:{params}')

    return params

def create_value_model_coeff_input(agent_type):
    return html.Div(
        [
            html.Label('Value Model Coefficient', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'value-model-coeff',
                    'model':'value',
                    'agent':agent_type
                },
                type='number',
                min=0.01,
                max=2.00,
                step=0.01,
                value=0.5,
            )
        ]
    )

# def create_ppo_loss_type_input(agent_type):
#     return html.Div(
#         [
#             html.Label('Loss Type', style={'text-decoration': 'underline'}),
#             dcc.Dropdown(
#                 id={
#                     'type':'loss-type',
#                     'model':'none',
#                     'agent':agent_type,
#                 },
#                 options=[{'label': i, 'value': i.lower()} for i in ["KL", "Clipped", "Hybrid"]],
#                 placeholder="select loss type"
#             ),
#             html.Div(
#                 id={
#                     'type':'entropy-block',
#                     'model':'none',
#                     'agent':agent_type,
#                 },
#                 children = [
#                     html.Label('Entropy Coefficient', style={'text-decoration': 'underline'}),
#                     dcc.Input(
#                     id={
#                         'type':'entropy-value',
#                         'model':'none',
#                         'agent':agent_type,
#                     },
#                     type='number',
#                     # placeholder="Clamp Value",
#                     min=0.001,
#                     max=1.00,
#                     step=0.001,
#                     # value=0.01,
#                     ),
#                 ],
#                 style={'display': 'none'},
#             ),
#             html.Div(
#                 id={
#                     'type':'kl-block',
#                     'model':'none',
#                     'agent':agent_type,
#                 },
#                 children = [
#                     html.Label('KL Coefficient', style={'text-decoration': 'underline'}),
#                     dcc.Input(
#                     id={
#                         'type':'kl-value',
#                         'model':'none',
#                         'agent':agent_type,
#                     },
#                     type='number',
#                     # placeholder="Clamp Value",
#                     min=0.01,
#                     max=5.00,
#                     step=0.01,
#                     ),
#                 ],
#                 style={'display': 'none'},
#             ),
#             html.Div(
#                 id={
#                     'type':'lambda-block',
#                     'model':'none',
#                     'agent':agent_type,
#                 },
#                 children = [
#                     html.Label('Lambda Coefficient', style={'text-decoration': 'underline'}),
#                     dcc.Input(
#                     id={
#                         'type':'lambda-value',
#                         'model':'none',
#                         'agent':agent_type,
#                     },
#                     type='number',
#                     # placeholder="Clamp Value",
#                     min=0.00,
#                     max=1.00,
#                     step=0.01,
#                     # value=0.00,
#                     ),
#                 ],
#                 style={'display': 'none'},
#             ),
#         ]
#     )



def create_normalize_advantage_input(agent_type):
    return html.Div(
        [
            html.Label('Normalize Advantage'),
            dcc.RadioItems(
                id={
                    'type': 'norm-adv',
                    'model': 'none',
                    'agent': agent_type,
                },
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                value=True,
                style={'margin-left': '10px'},
            ),
        ]
    )

def create_normalize_values_input(agent_type):
    return html.Div(
        [
            html.Label('Normalize Values'),
            dcc.RadioItems(
                id={
                    'type': 'norm-values',
                    'model': 'none',
                    'agent': agent_type,
                },
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                value=False,
                style={'margin-left': '10px'},
            ),
            html.Div(
                id = {
                    'type':'norm-clip-block',
                    'model':'none',
                    'agent':agent_type,
                },
                children = [
                    html.Label('Norm Clip'),
                    dcc.Input(
                        id={
                            'type': 'norm-clip',
                            'model': 'none',
                            'agent': agent_type
                        },
                        type='number',
                        min=0.1,
                        max=10.0,
                        value=5.0,
                        step=0.1,
                    )
                ],
                style={'margin-left': '10px', 'display':'none'}
            )
        ]
    )

def create_clip_rewards_input(agent_type):
    return html.Div(
        [
            html.Label('Clip Rewards'),
            dcc.RadioItems(
                id={
                    'type': 'clip-rewards',
                    'model': 'none',
                    'agent': agent_type,
                },
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                value=True,
                style={'margin-left': '10px'},
            ),
            html.Div(
                id = {
                    'type':'clip-rewards-block',
                    'model':'none',
                    'agent':agent_type,
                },
                children = [
                    html.Label('Reward Clip'),
                    dcc.Input(
                        id={
                            'type': 'reward-clip',
                            'model': 'none',
                            'agent': agent_type
                        },
                        type='number',
                        min=0.1,
                        max=10.0,
                        value=1.0,
                        step=0.1,
                    )
                ],
                style={'margin-left': '10px', 'display':'none'}
            )
        ]
    )

def create_distribution_input(agent_type):
    return html.Div(
        [
            html.Label('Probability Distribution', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
            id={
                'type':'distribution',
                'model':'none',
                'agent':agent_type,
                },
                options=[{'label': i.capitalize(), 'value': i} for i in ["beta", "normal", "categorical"]],
                placeholder="select distribution"
            ),
        ]
    )

def create_warmup_input(agent_type):
    return html.Div([
        html.Label('Warmup Period (Steps)'),
        dcc.Input(
            id={
                'type': 'warmup',
                'model': 'none',
                'agent': agent_type
            },
            type='number',
            min=0,
            max=1000,
            value=0,
            step=500,
        )
    ])

def create_tau_input(agent_type):
    return html.Div(
        [
            html.Label('Tau', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'tau',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=0.001,
                max=0.999,
                value=0.005,
                step=0.001,
            ),
        ]
    )

def create_batch_size_input(agent_type):
    return html.Div(
        [
            html.Label('Batch Size', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'batch-size',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1024,
                step=1,
                value=64,
            ),
        ]
    )

def create_noise_function_input(agent_type):
    return html.Div(
        [
            dcc.Dropdown(
            id={
                'type':'noise-function',
                'model':'none',
                'agent':agent_type,
                },
                options=[{'label': i, 'value': i} for i in ["Ornstein-Uhlenbeck", "Normal", "Uniform"]],
                placeholder="Noise Function"
            ),
            html.Div(
                id={
                    'type':'noise-options',
                    'model':'none',
                    'agent':agent_type,
                }
            ),
        ]
    )

def create_target_noise_stddev_input(agent_type):
    return html.Div(
        [
            html.Label('Target Noise Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'target-noise-stddev',
                    'model':'actor',
                    'agent':agent_type,
                },
                type='number',
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.2,
            ),
        ]
    )

def create_target_noise_clip_input(agent_type):
    return html.Div(
        [
            html.Label('Target Noise Clip', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'target-noise-clip',
                    'model':'actor',
                    'agent':agent_type,
                },
                type='number',
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.5,
            ),
        ]
    )

def create_actor_delay_input(agent_type):
    return html.Div(
        [
            html.Label('Actor Update Delay (steps)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'actor-update-delay',
                    'model':'actor',
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=10,
                step=1,
                value=2,
            ),
        ]
    )

def create_convolution_layers_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Convolution Layers', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'conv-layers',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0,
                max=20,
                step=1,
                value=0,
            ),
            html.Div(
                id={
                    'type':'layer-types',
                    'model':model_type,
                    'agent':agent_type,
                }
            ),
        ]
    )

def create_dense_layers_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Dense Layers', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'dense-layers',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0,
                max=10,
                step=1,
                value=2,
            ),
            html.Div(
                id={
                    'type':'units-per-layer',
                    'model':model_type,
                    'agent':agent_type,
                }
            ),
        ]
    )

def create_kernel_input(agent_type, model_type, index, stored_values=None, key=None):

    #DEBUG
    print({
    'type': 'kernel-init',
    'model': model_type,
    'agent': agent_type,
    'index': index
    })
    
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type': 'kernel-init',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                    options=[
                        {"label": x, "value": x.replace(" ", "_")} for x in 
                        ["kaiming uniform", "kaiming normal", "xavier uniform", "xavier normal", "truncated normal", 
                        "uniform", "normal", "constant", "ones", "zeros", "variance scaling", "orthogonal", "default"
                        ]
                    ],
                placeholder="Kernel Initialization",
                value=stored_values.get(key) if stored_values else None,
            ),
            html.Div(
                id={
                    'type': 'kernel-params',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index
                }
            ),
        ]
    )

def get_kernel_params_map():
    return {
        "kaiming_normal": ["mode"],
        "kaiming_uniform": ["mode"],
        "xavier_normal": ["gain"],
        "xavier_uniform": ["gain"],
        "truncated_normal": ["mean", "std-dev"],
        "uniform": ["min", "max"],
        "normal": ["mean", "std-dev"],
        "constant": ["value"],
        "variance_scaling": ["scale", "mode", "distribution"]
    }

def create_activation_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'activation-function',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
                placeholder="Activation Function",
            ),
        ]
    )

def create_optimizer_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'optimizer',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
                placeholder="Optimizer",
            ),
            html.Div(
                id=
                {
                    'type':'optimizer-options',
                    'model':model_type,
                    'agent':agent_type,
                },
            ),
        ]
    )

def create_optimizer_params_input(agent_type, model_type, optimizer):
    if optimizer == 'Adam':
        return html.Div([
            html.Label("Weight Decay", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'adam-weight-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            )
        ])
    
    elif optimizer == 'Adagrad':
        return html.Div([
            html.Label("Weight Decay", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'adagrad-weight-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            ),
            html.Label("Learning Rate Decay", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'adagrad-lr-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            )
        ])
    
    elif optimizer == 'RMSprop':
        return html.Div([
            html.Label("Weight Decay", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'rmsprop-weight-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            ),
            html.Label("Momentum", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'rmsprop-momentum',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            )
        ])
    
    elif optimizer == 'SGD':
        return html.Div([
            html.Label("Weight Decay", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'sgd-weight-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            ),
            html.Label("Momentum", style={'text-decoration': 'underline', 'margin-left': '20px'}),
            dcc.Input(
                id=
                {
                    'type':'sgd-momentum',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.01,
            )
        ])
    
def get_optimizer(model_type, agent_type, agent_params):
    """Returns formatted dict of optimizer from params dict

    Args:
        model_type (str): model type the optimizer belongs to
        agent_type (str): agent type the optimizer belongs to
        params (dict): dict of all parameters belonging to the agent (from dcc.Store(id='agent-params-store')

    Returns:
        dict: dictionary of all {param:value} pairs belonging to the passed optimizer for the model of the agent
    """

    optimizer = agent_params.get(get_key({'type':'optimizer', 'model':model_type, 'agent':agent_type}))
    learning_rate_constant = agent_params.get(get_key({'type':'learning-rate-const', 'model':model_type, 'agent':agent_type}))
    learning_rate_exp = agent_params.get(get_key({'type':'learning-rate-exp', 'model':model_type, 'agent':agent_type}))
    learning_rate = learning_rate_constant * 10**learning_rate_exp

    # instantiate empty dict to store params
    params = {'lr':learning_rate}

    if optimizer == 'Adam':
        weight_decay = agent_params.get(get_key({'type':'adam-weight-decay', 'model':model_type, 'agent':agent_type}))
        params['weight_decay'] = weight_decay

    elif optimizer == 'Adagrad':
        weight_decay = agent_params.get(get_key({'type':'adagrad-weight-decay', 'model':model_type, 'agent':agent_type}))
        lr_decay = agent_params.get(get_key({'type':'adagrad-lr-decay', 'model':model_type, 'agent':agent_type}))
        params['weight_decay'] = weight_decay
        params['lr_decay'] = lr_decay

    elif optimizer == 'RMSprop':
        weight_decay = agent_params.get(get_key({'type':'rmsprop-weight-decay', 'model':model_type, 'agent':agent_type}))
        momentum = agent_params.get(get_key({'type':'rmsprop-momentum', 'model':model_type, 'agent':agent_type}))
        params['weight_decay'] = weight_decay
        params['momentum'] = momentum

    elif optimizer == 'SGD':
        weight_decay = agent_params.get(get_key({'type':'sgd-weight-decay', 'model':model_type, 'agent':agent_type}))
        momentum = agent_params.get(get_key({'type':'sgd-momentum', 'model':model_type, 'agent':agent_type}))
        params['weight_decay'] = weight_decay
        params['momentum'] = momentum
    
    else:
        raise ValueError(f"{optimizer} not found in utils.get_optimizer_params")
    
    return {'type':optimizer, 'params':params}

def create_learning_rate_constant_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Constant', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'learning-rate-const',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1.0,
                max=9.0,
                step=0.1,
                value=1.0,
            ),
        ]
    )
    
def create_learning_rate_exponent_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Exponent(10^x)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'learning-rate-exp',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=-9,
                max=-1,
                step=1,
                value=-4,
            ),
        ]
    )

def create_lr_scheduler_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Scheduler', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-scheduler',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Step', 'Exponential', 'CosineAnnealing', 'Linear', 'None']],
                placeholder="Learning Rate Scheduler",
            ),
            html.Div(
                id={
                    'type':'lr-scheduler-options',
                    'model':model_type,
                    'agent':agent_type,
                }
            )
        ]
    )

def update_lr_scheduler_options(agent_type, model_type, lr_scheduler):
    if lr_scheduler == 'step':
        return lr_step_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'exponential':
        return lr_exponential_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'cosineannealing':
        return lr_cosineannealing_scheduler_options(agent_type, model_type)
    elif lr_scheduler == 'linear':
        return lr_linear_scheduler_options(agent_type, model_type)
    return html.Div()

def lr_cosineannealing_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('T max (max iters)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-t-max',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=10000,
                step=1,
                value=1000,
            ),
            html.Label('Eta min (min LR)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-eta-min',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.000001,
                max=0.1,
                step=0.000001,
                value=0.0001,
            ),
        ]
    )


def lr_exponential_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def lr_step_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Step Size', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-step-size',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1000,
                step=1,
                value=100,
            ),
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-gamma',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
        ]
    )

def lr_linear_scheduler_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Start Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-start-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0001,
                max=1.0000,
                step=0.0001,
                value=1.0,
            ),
            html.Label('End Factor', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-end-factor',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0001,
                max=1.0000,
                step=0.0001,
                value=0.0010,
            ),
            html.Label('Total Iterations', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'lr-total-iters',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=1e8,
                step=1,
                value=1e3,
            ),
        ]
    )

def get_lr_scheduler(model_type, agent_type, agent_params):

    scheduler = agent_params.get(get_key({'type':'lr-scheduler', 'model':model_type, 'agent':agent_type}))
    if scheduler == 'none':
        return None
    params = {}
    if scheduler == 'step':
        params['step_size'] = agent_params.get(get_key({'type':'lr-step-size', 'model':model_type, 'agent':agent_type}))
        params['gamma'] = agent_params.get(get_key({'type':'lr-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'exponential':
        params['gamma'] = agent_params.get(get_key({'type':'lr-gamma', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'cosineannealing':
        params['T_max'] = agent_params.get(get_key({'type':'lr-t-max', 'model':model_type, 'agent':agent_type}))
        params['eta_min'] = agent_params.get(get_key({'type':'lr-eta-min', 'model':model_type, 'agent':agent_type}))
    elif scheduler == 'linear':
        params['start_factor'] = agent_params.get(get_key({'type':'lr-start-factor', 'model':model_type, 'agent':agent_type}))
        params['end_factor'] = agent_params.get(get_key({'type':'lr-end-factor', 'model':model_type, 'agent':agent_type}))
        params['total_iters'] = agent_params.get(get_key({'type':'lr-total-iters', 'model':model_type, 'agent':agent_type}))

    return {'type':scheduler, 'params':params}

def create_goal_strategy_input(agent_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'goal-strategy',
                    'model':'none',
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Future', 'Final', 'None']],
                placeholder="Goal Strategy",
            ),
            html.Div(
                id={
                    'type':'goal-strategy-options',
                    'model':'none',
                    'agent':agent_type,
                }
            )
        ]
    )

def update_goal_strategy_options(agent_type, goal_strategy):
    if goal_strategy == 'future':
        return future_goal_strategy_options(agent_type)
    return html.Div()

def future_goal_strategy_options(agent_type):
    return html.Div(
        [
            html.Label('Number of Future Goals', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'future-goals',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=10,
                step=1,
                value=4,
            ),
        ]
    )

def create_tolerance_input(agent_type):
    return html.Div(
        [
            html.Label('Goal Tolerance', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'goal-tolerance',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=0.001,
                max=10.000,
                step=0.001,
                value=0.05,
            ),
        ]
    )

def create_input_normalizer_options_input(agent_type):
    return html.Div(
        [
            html.Label(f'Minimum/Maximum Clip Value', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'clip-value',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=1,
                max=100,
                step=1,
                value=5,
            ),
        ]
    )

def create_input_normalizer_input(agent_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-input',
                    'model':'none',
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i} for i in ['True', 'False']],
                placeholder="Normalize Input",
            ),
            html.Div(
                id={
                    'type':'normalize-options',
                    'model':'none',
                    'agent':agent_type,
                }
            )
        ]
    )

def create_normalize_layers_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-layers',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i=='True'} for i in ['True', 'False']],
                placeholder="Normalize Layers",
            ),
        ]
    )

def create_clamp_output_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'clamp-output',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i=='True'} for i in ['True', 'False']],
                placeholder="Clamp Output",
            ),
            dcc.Input(
                id={
                    'type':'clamp-value',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                placeholder="Clamp Value",
                min=0.01,
                step=0.01,
                style={'display': 'none'},
            ),
        ]
    )

def create_epsilon_greedy_input(agent_type):
    return html.Div(
        [
            html.Label(f'Epsilon Greedy', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':f'epsilon-greedy',
                    'model':'none',
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.2,
            ),
        ]
    )


def create_trace_decay_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f"{model_type.capitalize()} Trace Decay", style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'trace-decay',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=0.9,
            ),
        ]
    )

def create_add_layer_button(agent_type, model_type):
    return html.Div([
        # dcc.Store(
        #     id={
        #         'type': 'layer-values-store',
        #         'model': model_type,
        #         'agent': agent_type,
        #     },
        #     data={}
        # ),
        html.Div(
            id={
                'type':'layer-dropdowns',
                'model':model_type,
                'agent':agent_type,
                }),
        html.Button("Add Layer",
            id={
                'type':'add-layer-btn',
                'model':model_type,
                'agent':agent_type,
            },
            n_clicks=0,
            style={'margin-top': '10px', 'margin-left': '20px'}
        )
    ])
    
def create_policy_model_type_input(agent_type):
    return html.Div(
        [
            html.Label("Policy Type", style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'policy-type',
                    'model':'policy',
                    'agent':agent_type,
                },
                options=[{'label': 'Stochastic Continuous', 'value': 'StochasticContinuousPolicy'},
                         {'label': 'Stochastic Discrete', 'value': 'StochasticDiscretePolicy'},
                         ],
                placeholder="Policy Model Type",
            ),
        ]
    )

def create_policy_model_input(agent_type):
    return html.Div(
        [
            html.H3("Policy Model Configuration"),
            # create_dense_layers_input(agent_type, 'policy'),
            # html.Label("Hidden Layers Kernel Initializers"),
            # create_kernel_input(agent_type, 'policy-hidden'),
            create_add_layer_button(agent_type, 'policy'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'policy', 0),
            # create_activation_input(agent_type, 'policy'),
            create_optimizer_input(agent_type, 'policy'),
            create_learning_rate_constant_input(agent_type, 'policy'),
            create_learning_rate_exponent_input(agent_type, 'policy'),
            create_lr_scheduler_input(agent_type, 'policy')
        ]
    )


def create_value_model_input(agent_type):
    return html.Div(
        [
            html.H3("Value Model Configuration"),
            # create_dense_layers_input(agent_type, 'value'),
            # html.Label("Hidden Layers Kernel Initializers"),
            # create_kernel_input(agent_type, 'value-hidden'),
            create_add_layer_button(agent_type, 'value'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'value', 0),
            # create_activation_input(agent_type, 'value'),
            create_optimizer_input(agent_type, 'value'),
            create_learning_rate_constant_input(agent_type, 'value'),
            create_learning_rate_exponent_input(agent_type, 'value'),
            create_lr_scheduler_input(agent_type, 'value')
        ]
    )
    

def create_actor_model_input(agent_type):
    return html.Div(
        [
            html.H3("Actor Model Configuration"),
            # create_dense_layers_input(agent_type, 'value'),
            # html.Label("Hidden Layers Kernel Initializers"),
            # create_kernel_input(agent_type, 'value-hidden'),
            create_add_layer_button(agent_type, 'actor'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'actor', 0),
            # create_activation_input(agent_type, 'value'),
            create_optimizer_input(agent_type, 'actor'),
            create_learning_rate_constant_input(agent_type, 'actor'),
            create_learning_rate_exponent_input(agent_type, 'actor'),
            create_lr_scheduler_input(agent_type, 'actor')
        ]
    )


def create_critic_model_input(agent_type):
    return html.Div(
        [
            html.H3("Critic State Layers Configuration"),
            create_add_layer_button(agent_type, 'critic-state'),
            html.H3("Critic State/Action Layers Configuration"),
            create_add_layer_button(agent_type, 'critic-merged'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'critic', 0),
            # create_activation_input(agent_type, 'value'),
            create_optimizer_input(agent_type, 'critic'),
            create_learning_rate_constant_input(agent_type, 'critic'),
            create_learning_rate_exponent_input(agent_type, 'critic'),
            create_lr_scheduler_input(agent_type, 'critic')
        ]
    )


def create_reinforce_parameter_inputs(agent_type):
    """Adds inputs for REINFORCE Agent"""
    return html.Div(
        id=f'{agent_type}-inputs',
        children=[
            dbc.Tabs([
                # Tab 1 Agent Parameters
                dbc.Tab(
                    label="Agent Parameters",
                    children=[
                        create_discount_factor_input(agent_type),
                    ]
                ),

                # Tab 2 Policy Model
                dbc.Tab(
                    label="Policy Model",
                    children=[
                        create_policy_model_input(agent_type),
                    ]
                ),

                # Tab 3 Value Model
                dbc.Tab(
                    label="Value Model",
                    children=[
                        create_value_model_input(agent_type),
                    ]
                ),

                # Tab 4 Agent Options
                dbc.Tab(
                    label="Agent Options",
                    children=[
                        create_device_input(agent_type),
                        create_save_dir_input(agent_type),
                    ]
                ),
            ])
        ]
    )

def create_actor_critic_parameter_inputs(agent_type):
    return html.Div(
        id=f'{agent_type}-inputs',
        children=[
            dbc.Tabs([
                # Tab 1 Agent Parameters
                dbc.Tab(
                    label="Agent Parameters",
                    children=[
                        create_discount_factor_input(agent_type),
                        create_trace_decay_input(agent_type, 'policy'),
                        create_trace_decay_input(agent_type, 'value'),
                    ]
                ),

                # Tab 2 Policy Model
                dbc.Tab(
                    label="Policy Model",
                    children=[
                        create_policy_model_input(agent_type),
                    ]
                ),

                # Tab 3 Value Model
                dbc.Tab(
                    label="Value Model",
                    children=[
                        create_value_model_input(agent_type),
                    ]
                ),

                # Tab 4 Agent Options
                dbc.Tab(
                    label="Agent Options",
                    children=[
                        create_device_input(agent_type),
                        create_save_dir_input(agent_type),
                    ]
                ),
            ])
        ]
    )

def create_ddpg_parameter_inputs(agent_type):
    """Adds inputs for DDPG Agent"""
    # return html.Div(
    #     id=f'{agent_type}-inputs',
    #     children=[
    #         create_device_input(agent_type),
            
    #         # Actor Model Configuration
    #         ,
    #         # Critic Model Configuration
    #         html.H3("Critic Model Configuration"),
            
    #         # Save dir
    #         create_save_dir_input(agent_type)
    #     ]
    # )
    return html.Div(
            id=f'{agent_type}-inputs',
            children=[
                dbc.Tabs([
                    # Tab 1 Agent Parameters
                    dbc.Tab(
                        label="Agent Parameters",
                        children=[
                            create_discount_factor_input(agent_type),
                            create_tau_input(agent_type),
                            create_epsilon_greedy_input(agent_type),
                            create_batch_size_input(agent_type),
                            create_noise_function_input(agent_type),
                            create_input_normalizer_input(agent_type),
                            create_warmup_input(agent_type),
                        ]
                    ),

                    # Tab 2 Policy Model
                    dbc.Tab(
                        label="Actor Model",
                        children=[
                            create_actor_model_input(agent_type),
                        ]
                    ),

                    # Tab 3 Value Model
                    dbc.Tab(
                        label="Critic Model",
                        children=[
                            create_critic_model_input(agent_type),
                        ]
                    ),

                    # Tab 4 Agent Options
                    dbc.Tab(
                        label="Agent Options",
                        children=[
                            create_device_input(agent_type),
                            create_save_dir_input(agent_type),
                        ]
                    ),
                ])
            ]
        )

def create_td3_parameter_inputs(agent_type):
    """Adds inputs for TD3 Agent"""
    return html.Div(
            id=f'{agent_type}-inputs',
            children=[
                dbc.Tabs([
                    # Tab 1 Agent Parameters
                    dbc.Tab(
                        label="Agent Parameters",
                        children=[
                            create_discount_factor_input(agent_type),
                            create_tau_input(agent_type),
                            create_epsilon_greedy_input(agent_type),
                            create_batch_size_input(agent_type),
                            create_noise_function_input(agent_type),
                            create_target_noise_stddev_input(agent_type),
                            create_target_noise_clip_input(agent_type),
                            create_input_normalizer_input(agent_type),
                            create_warmup_input(agent_type),
                        ]
                    ),

                    # Tab 2 Policy Model
                    dbc.Tab(
                        label="Actor Model",
                        children=[
                            create_actor_model_input(agent_type),
                            create_actor_delay_input(agent_type),
                        ]
                    ),

                    # Tab 3 Value Model
                    dbc.Tab(
                        label="Critic Model",
                        children=[
                            create_critic_model_input(agent_type),
                        ]
                    ),

                    # Tab 4 Agent Options
                    dbc.Tab(
                        label="Agent Options",
                        children=[
                            create_device_input(agent_type),
                            create_save_dir_input(agent_type),
                        ]
                    ),
                ])
            ]
        )


def create_her_ddpg_parameter_inputs(agent_type):
    """Adds inputs for Hindsight Experience Replay w/DDPG Agent"""
    return html.Div(
        id=f'{agent_type}-inputs',
        children=[
            create_device_input(agent_type),
            create_goal_strategy_input(agent_type),
            create_tolerance_input(agent_type),
            create_discount_factor_input(agent_type),
            create_tau_input(agent_type),
            create_epsilon_greedy_input(agent_type),
            create_batch_size_input(agent_type),
            create_noise_function_input(agent_type),
            html.H6("Input Normalizers"),
            create_input_normalizer_options_input(agent_type),
            # Actor Model Configuration
            create_actor_model_input(agent_type),
            # Critic Model Configuration
            create_critic_model_input(agent_type),
            # Save dir
            create_save_dir_input(agent_type),
        ]
    )

def create_ppo_parameter_inputs(agent_type):
    """Adds inputs for PPO Agent"""
    return html.Div(
        id=f'{agent_type}-inputs',
        children=[
            dbc.Tabs([
                # Tab 1 Agent Parameters
                dbc.Tab(
                    label="Agent Parameters",
                    children=[
                        create_distribution_input(agent_type),
                        create_discount_factor_input(agent_type),
                        create_advantage_coeff_input(agent_type),
                        create_entropy_input(agent_type),
                        create_kl_coeff_input(agent_type),
                        create_normalize_advantage_input(agent_type),
                        create_normalize_values_input(agent_type),
                        create_clip_rewards_input(agent_type),
                    ]
                ),

                # Tab 2 Policy Model
                dbc.Tab(
                    label="Policy Model",
                    children=[
                        create_policy_model_type_input(agent_type),
                        create_surrogate_loss_clip_input(agent_type, 'policy'),
                        create_grad_clip_input(agent_type, 'policy'),
                        create_policy_model_input(agent_type),
                    ]
                ),

                # Tab 3 Value Model
                dbc.Tab(
                    label="Value Model",
                    children=[
                        create_surrogate_loss_clip_input(agent_type, 'value'),
                        create_value_model_coeff_input(agent_type),
                        create_grad_clip_input(agent_type, 'value'),
                        create_value_model_input(agent_type),
                    ]
                ),

                # Tab 4 Agent Options
                dbc.Tab(
                    label="Agent Options",
                    children=[
                        create_device_input(agent_type),
                        create_save_dir_input(agent_type),
                    ]
                ),
            ])
        ]
    )

def create_agent_parameter_inputs(agent_type):
    """Component for agent hyperparameters"""
    if agent_type == 'Reinforce':
        return create_reinforce_parameter_inputs(agent_type)

    elif agent_type == 'ActorCritic':
        return create_actor_critic_parameter_inputs(agent_type)

    elif agent_type == 'DDPG':
        return create_ddpg_parameter_inputs(agent_type)
    
    elif agent_type == 'TD3':
        return create_td3_parameter_inputs(agent_type)
    
    elif agent_type == 'HER_DDPG':
        return create_her_ddpg_parameter_inputs(agent_type)
    
    elif agent_type == 'PPO':
        return create_ppo_parameter_inputs(agent_type)

    else:
        return html.Div("Select a model type to configure its parameters.")
    
## HYPERPARAMETER SEARCH FUNCTIONS

def create_reinforce_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            generate_learning_rate_hyperparam_component(agent_type, 'none'),
            generate_discount_hyperparam_component(agent_type, 'none'),
            dcc.Tabs([
                dcc.Tab([
                    generate_hidden_layer_hyperparam_component(agent_type, 'policy'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'policy'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'policy'),
                    generate_optimizer_hyperparam_component(agent_type, 'policy'),
                ],
                label="Policy Model"),
                dcc.Tab([
                    generate_hidden_layer_hyperparam_component(agent_type, 'value'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'value'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'value'),
                    generate_optimizer_hyperparam_component(agent_type, 'value'),
                ],
                label="Value Model"),
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def create_actor_critic_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            # utils.generate_actor_critic_hyperparam_component(),
            # html.H3('Actor Critic Hyperparameters'),
            generate_learning_rate_hyperparam_component(agent_type, 'none'),
            generate_discount_hyperparam_component(agent_type, 'none'),
            dcc.Tabs([
                dcc.Tab([
                    # html.H4("Policy Model Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'policy'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'policy'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'policy'),
                    generate_optimizer_hyperparam_component(agent_type, 'policy'),
                    generate_trace_decay_hyperparam_componenent(agent_type, 'policy'),
                ],
                label="Policy Model"),
                dcc.Tab([
                    # html.H4("Value Model Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'value'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'value'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'value'),
                    generate_optimizer_hyperparam_component(agent_type, 'value'),
                    generate_trace_decay_hyperparam_componenent(agent_type, 'value'),
                ],
                label="Value Model")
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def create_ddpg_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            create_device_input(agent_type),
            generate_discount_hyperparam_component(agent_type, 'none'),
            generate_tau_hyperparam_componenent(agent_type, 'none'),
            create_epsilon_greedy_hyperparam_input(agent_type, 'none'),
            generate_batch_hyperparam_componenent(agent_type, 'none'),
            generate_noise_hyperparam_componenent(agent_type, 'none'),
            create_replay_buffer_size_hyperparam_component(agent_type, 'none'),
            create_input_normalizer_hyperparam_input(agent_type, 'none'),
            generate_warmup_hyperparam_input(agent_type, 'none'),
            html.Hr(),
            dcc.Tabs([
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'actor'),
                    generate_cnn_layer_hyperparam_component(agent_type, 'actor'),
                    generate_hidden_layer_hyperparam_component(agent_type, 'actor'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'actor'),
                    generate_optimizer_hyperparam_component(agent_type, 'actor'),
                    create_normalize_layers_hyperparam_input(agent_type, 'actor'),
                ],
                label='Actor Model'),
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'critic'),
                    html.Hr(),
                    generate_cnn_layer_hyperparam_component(agent_type, 'critic'),
                    html.H4("Critic State Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-state'),
                    html.Hr(),
                    html.H4("Critic Merged (State + Action) Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-merged'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'critic'),
                    generate_optimizer_hyperparam_component(agent_type, 'critic'),
                    create_normalize_layers_hyperparam_input(agent_type, 'critic'),
                ],
                label='Critic Model')
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def create_td3_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            create_device_input(agent_type),
            generate_discount_hyperparam_component(agent_type, 'none'),
            generate_tau_hyperparam_componenent(agent_type, 'none'),
            create_epsilon_greedy_hyperparam_input(agent_type, 'none'),
            generate_batch_hyperparam_componenent(agent_type, 'none'),
            generate_noise_hyperparam_componenent(agent_type, 'none'),
            create_replay_buffer_size_hyperparam_component(agent_type, 'none'),
            generate_target_action_noise_stddev_hyperparam_component(agent_type, 'none'),
            generate_target_action_noise_clip_hyperparam_component(agent_type, 'none'),
            generate_actor_update_delay_hyperparam_component(agent_type, 'none'),
            create_input_normalizer_hyperparam_input(agent_type, 'none'),
            generate_warmup_hyperparam_input(agent_type, 'none'),
            html.Hr(),
            dcc.Tabs([
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'actor'),
                    generate_cnn_layer_hyperparam_component(agent_type, 'actor'),
                    generate_hidden_layer_hyperparam_component(agent_type, 'actor'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'actor'),
                    generate_optimizer_hyperparam_component(agent_type, 'actor'),
                    create_normalize_layers_hyperparam_input(agent_type, 'actor'),
                    # create_clamp_output_hyperparam_input(agent_type, 'actor'),
                ],
                label='Actor Model'),
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'critic'),
                    html.Hr(),
                    generate_cnn_layer_hyperparam_component(agent_type, 'critic'),
                    html.H4("Critic State Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-state'),
                    html.Hr(),
                    html.H4("Critic Merged (State + Action) Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-merged'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'critic'),
                    generate_optimizer_hyperparam_component(agent_type, 'critic'),
                    create_normalize_layers_hyperparam_input(agent_type, 'critic'),
                ],
                label='Critic Model')
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def create_her_ddpg_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            create_device_input(agent_type),
            create_goal_strategy_hyperparam_input(agent_type, 'none'),
            create_tolerance_hyperparam_input(agent_type, 'none'),
            generate_discount_hyperparam_component(agent_type, 'none'),
            generate_tau_hyperparam_componenent(agent_type, 'none'),
            create_epsilon_greedy_hyperparam_input(agent_type, 'none'),
            generate_batch_hyperparam_componenent(agent_type, 'none'),
            generate_noise_hyperparam_componenent(agent_type, 'none'),
            create_replay_buffer_size_hyperparam_component(agent_type, 'none'),
            html.H6("Input Normalizers"),
            create_input_normalizer_hyperparam_input(agent_type, 'none'),
            html.Hr(),
            # Actor config
            dcc.Tabs([
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'actor'),
                    generate_cnn_layer_hyperparam_component(agent_type, 'actor'),
                    generate_hidden_layer_hyperparam_component(agent_type, 'actor'),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'actor-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'actor'),
                    generate_optimizer_hyperparam_component(agent_type, 'actor'),
                    create_normalize_layers_hyperparam_input(agent_type, 'actor'),
                    # create_clamp_output_hyperparam_input(agent_type, 'actor'),
                ],
                label='Actor Model'),
                # Critic config
                dcc.Tab([
                    generate_learning_rate_hyperparam_component(agent_type, 'critic'),
                    html.Hr(),
                    generate_cnn_layer_hyperparam_component(agent_type, 'critic'),
                    html.H4("Critic State Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-state'),
                    html.Hr(),
                    html.H4("Critic Merged (State + Action) Input Layer Configuration"),
                    generate_hidden_layer_hyperparam_component(agent_type, 'critic-merged'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-hidden', 'Hidden Layers'),
                    html.Hr(),
                    generate_kernel_initializer_hyperparam_component(agent_type, 'critic-output', 'Output Layer'),
                    html.Hr(),
                    generate_activation_function_hyperparam_component(agent_type, 'critic'),
                    generate_optimizer_hyperparam_component(agent_type, 'critic'),
                    create_normalize_layers_hyperparam_input(agent_type, 'critic'),
                ],
                label='Critic Model')
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def create_ppo_hyperparam_input(agent_type):
    return dcc.Tab([
        html.Div([
            create_device_input(agent_type),
            generate_discount_hyperparam_component(agent_type, 'none'),
            create_advantage_coeff_hyperparam_input(agent_type, 'none'),
            create_entropy_coeff_hyperparam_input(agent_type, 'none'),
            create_kl_coeff_hyperparam_input(agent_type, 'none'),
            create_advantage_normalizer_hyperparam_input(agent_type, 'none'),
            create_reward_clip_hyperparam_input(agent_type, 'none'),
            html.Hr(),
            # Actor config
            dcc.Tabs([
                dcc.Tab(
                    html.Div(
                        [
                            create_model_type_hyperparam_input(agent_type, 'policy'),
                            create_distribution_hyperparam_input(agent_type, 'policy'),
                            create_surrogate_loss_clip_hyperparam_input(agent_type, 'policy'),
                            create_grad_clip_hyperparam_input(agent_type, 'policy'),
                            create_learning_rate_constant_hyperparam_input(agent_type, 'policy'),
                            create_learning_rate_exponent_hyperparam_input(agent_type, 'policy'),
                            # generate_cnn_layer_hyperparam_component(agent_type, 'policy'),
                            generate_hidden_layer_hyperparam_component(agent_type, 'policy'),
                            # generate_kernel_initializer_hyperparam_component(agent_type, 'policy-hidden', 'Hidden Layers'),
                            html.Hr(),
                            html.H5(f'Output Layer', style={'margin-right': '10px'}),
                            generate_kernel_initializer_hyperparam_component(agent_type, 'policy', 'output'),
                            html.Hr(),
                            # generate_activation_function_hyperparam_component(agent_type, 'policy'),
                            generate_optimizer_hyperparam_component(agent_type, 'policy'),
                            # create_normalize_layers_hyperparam_input(agent_type, 'policy'),
                            # create_clamp_output_hyperparam_input(agent_type, 'actor'),
                            create_learning_rate_scheduler_hyperparam_input(agent_type, 'policy')
                        ],
                        style={'padding-left': '10px', 'padding-bottom': '20px'}
                    ),
                    label='Policy Model'),
                # Critic config
                dcc.Tab(
                    html.Div(
                        [
                            create_loss_coeff_hyperparam_input(agent_type, 'value'),
                            create_surrogate_loss_clip_hyperparam_input(agent_type, 'value'),
                            create_grad_clip_hyperparam_input(agent_type, 'value'),
                            create_learning_rate_constant_hyperparam_input(agent_type, 'value'),
                            create_learning_rate_exponent_hyperparam_input(agent_type, 'value'),
                            # generate_cnn_layer_hyperparam_component(agent_type, 'critic'),
                            generate_hidden_layer_hyperparam_component(agent_type, 'value'),
                            # generate_kernel_initializer_hyperparam_component(agent_type, 'value-hidden', 'Hidden Layers'),
                            html.Hr(),
                            html.H5(f'Output Layer', style={'margin-right': '10px'}),
                            generate_kernel_initializer_hyperparam_component(agent_type, 'value', 'output'),
                            html.Hr(),
                            # generate_activation_function_hyperparam_component(agent_type, 'value'),
                            generate_optimizer_hyperparam_component(agent_type, 'value'),
                            html.H6("Value Normalizer"),
                            create_value_normalizer_hyperparam_input(agent_type, 'value'),
                            # create_normalize_layers_hyperparam_input(agent_type, 'critic'),
                            create_learning_rate_scheduler_hyperparam_input(agent_type, 'value')
                        ],
                        style={'padding-left': '10px', 'padding-bottom': '20px'}
                    ),
                    label='Value Model')
            ]),
            create_save_dir_input(agent_type),
        ])
    ],
    label=agent_type)

def generate_learning_rate_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Learning Rate'),
        dcc.Dropdown(
            id={
                'type': 'learning-rate-slider',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    {'label': '10^-2', 'value': 1e-2},
                    {'label': '10^-3', 'value': 1e-3},
                    {'label': '10^-4', 'value': 1e-4},
                    {'label': '10^-5', 'value': 1e-5},
                    {'label': '10^-6', 'value': 1e-6},
                    {'label': '10^-7', 'value': 1e-7},
            ],
            multi=True,
        )             
    ])

def create_learning_rate_constant_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Constant', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'learning-rate-const-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in range(0,10)],
                multi=True,
            ),
        ]
    )

def create_learning_rate_exponent_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Exponent(10^x)', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'learning-rate-exp-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in range(-9,0)],
                multi=True,
            ),
        ]
    )

# def create_learning_rate_scheduler_hyperparam_input(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label('Learning Rate Exponent(10^x)', style={'text-decoration': 'underline'}),
#             dcc.Dropdown(
#                 id={
#                     'type':'learning-rate-exp-hyperparam',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 options=[{'label': str(i), 'value': i} for i in range(-9,0)],
#                 multi=True,
#             ),
#         ]
#     )

def create_advantage_coeff_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Advantage Coefficient', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'advantage-coeff-hyperparam',
                    'model':model_type,
                    'agent':agent_type
                },
                options=[{'label': str(i), 'value': i} for i in np.arange(0.0, 1.0, 0.05).round(2)],
                multi=True,
            )
        ]
    )

def create_surrogate_loss_clip_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Surrogate Loss Clip', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'surrogate-clip-hyperparam',
                    'model':model_type,
                    'agent':agent_type
                },
                options=[{'label': str(i), 'value': i} for i in np.arange(0.0, 1.0, 0.05).round(2)],
                multi=True,
            )
        ]
    )

def create_entropy_coeff_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Entropy Coefficient', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'entropy-coeff-hyperparam',
                    'model':model_type,
                    'agent':agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.0001', 'value': 0.0001},
                    {'label': '0.0005', 'value': 0.0005},
                    {'label': '0.001', 'value': 0.001},
                    {'label': '0.005', 'value': 0.005},
                    {'label': '0.01', 'value': 0.01},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.1', 'value': 0.1},
                
                ],
                multi=True,
            )
        ]
    )

def create_kl_coeff_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('KL Coefficient', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'kl-coeff-hyperparam',
                    'model':model_type,
                    'agent':agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                
                ],
                multi=True,
            )
        ]
    )

def create_model_type_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Model Type', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'model-type-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i.title(), 'value': i} for i in ['stochastic discrete', 'stochastic continuous']],
                multi=True,
            ),
        ]
    )

def create_distribution_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Distribution', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'distribution-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i.title(), 'value': i} for i in ['beta', 'normal', 'categorical']],
                
                multi=True,
            ),
        ]
    )

def create_reward_clip_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Reward Clip', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'reward-clip-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [1,2,3,4,5,10,20,30,40,50,100,'Infinity']],
                
                multi=True,
            ),
        ]
    )

def generate_discount_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Discount'),
        dcc.Dropdown(
            id={
                'type': 'discount-slider',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '0.99', 'value': 0.99},
                    {'label': '1.0', 'value': 1.0},
            ],
            multi=True,
        )
    ])

def generate_warmup_hyperparam_input(agent_type, model_type):
    return html.Div([
        html.H5('Warmup Period (Steps)'),
        dcc.Dropdown(
            id={
                'type': 'warmup-slider-hyperparam',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    
                    {'label': '500', 'value': 500},
                    {'label': '1000', 'value': 1000},
                    {'label': '5000', 'value': 5000},
                    {'label': '10000', 'value': 10000},
            ],
            multi=True,
        )
    ])

def generate_actor_update_delay_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Actor Update Delay (Steps)'),
        dcc.Dropdown(
            id={
                'type': 'actor-update-delay-slider-hyperparam',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5},
                    {'label': '6', 'value': 6},
                    {'label': '7', 'value': 7},
                    {'label': '8', 'value': 8},
                    {'label': '9', 'value': 9},
                    {'label': '10', 'value': 10},
                    {'label': '20', 'value': 20},
                    {'label': '30', 'value': 30},
                    {'label': '40', 'value': 40},
                    {'label': '50', 'value': 50},
                    {'label': '60', 'value': 60},
                    {'label': '70', 'value': 70},
                    {'label': '80', 'value': 80},
                    {'label': '90', 'value': 90},
                    {'label': '100', 'value': 100},
            ],
            multi=True,
        )
    ])

def generate_target_action_noise_stddev_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Target Action Noise Stddev'),
        dcc.Dropdown(
            id={
                'type': 'target-action-noise-stddev-slider-hyperparam',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
            ],
            multi=True,
        )
    ])

def generate_target_action_noise_clip_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Target Action Noise Clip'),
        dcc.Dropdown(
            id={
                'type': 'target-action-noise-clip-slider-hyperparam',
                'model': model_type,
                'agent': agent_type
            },
            options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
            ],
            multi=True,
        )
    ])

def generate_cnn_layer_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('CNN Layers'),
        dcc.RangeSlider(
            id={
                'type': 'cnn-layers-slider-hyperparam',
                'model': model_type,
                'agent': agent_type
            },
            min=0,
            max=20,
            value=[1, 5],  # Default range
            marks={0: "0", 10: "10", 20: "20"},
            step=1,  # anchor slider to marks
            # pushable=1,  # allow pushing the slider,
            allowCross=True,  # allow selecting one value
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id={
            'type':'cnn-layer-types-hyperparam',
            'model':model_type,
            'agent':agent_type
        })
    ])

def generate_cnn_layer_type_hyperparam_component(agent_type, model_type, index):
    
        input_id = {
            'type': 'cnn-layer-type-hyperparam',
            'model': model_type,
            'agent': agent_type,
            'index': index,
        }
        return html.Div([
            html.Label(f'Layer Type for Conv Layer {index}', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id=input_id,
                options=[
                    {'label': 'Conv2D', 'value': 'conv'},
                    {'label': 'MaxPool2D', 'value': 'pool'},
                    {'label': 'Dropout', 'value': 'dropout'},
                    {'label': 'BatchNorm2D', 'value': 'batchnorm'},
                    {'label': 'Relu', 'value':'relu'},
                    {'label': 'Tanh', 'value': 'tanh'},
                ],
                multi=True,
            ),
            html.Div(
                id={
                    'type': 'cnn-layer-type-parameters-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
            ),
            html.Hr(),
            ])

def generate_cnn_layer_parameters_hyperparam_component(layer_type, agent_type, model_type, index):
    # loop over layer types to create the appropriate parameters
    if layer_type == 'conv':
        return html.Div([
            html.Label(f'Filters in Conv Layer {index}', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'conv-filters-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                options=[{'label': i, 'value': i} for i in [8, 16, 32, 64, 128, 256, 512, 1024]],
                multi=True,
            ),
            html.Label(f'Kernel Size in Conv Layer {index}', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'conv-kernel-size-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                min=1,
                max=10,
                step=1,
                value=[2, 4],
                marks={1:'1', 10:'10'},
                allowCross=True,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label(f'Kernel Stride in Conv Layer {index}', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'conv-stride-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                min=1,
                max=10,
                step=1,
                value=[2, 4],
                marks={1:'1', 10:'10'},
                allowCross=True,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label(f'Input Padding in Conv Layer {index}', style={'text-decoration': 'underline'}),
            dcc.RadioItems(
                id={
                    'type': 'conv-padding-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                options=[
                    {'label': 'Same', 'value': 'same'},
                    {'label': 'Valid', 'value': 'valid'},
                    {'label': 'Custom', 'value': 'custom'},
                ],
            ),
            html.Div(
                [
                    html.Label('Custom Padding (pixels)', style={'text-decoration': 'underline'}),
                    dcc.RangeSlider(
                        id={
                            'type': 'conv-padding-custom-hyperparam',
                            'model': model_type,
                            'agent': agent_type,
                            'index': index,
                        },
                        min=0,
                        max=10,
                        step=1,
                        value=[1,3],
                        marks={0:'0', 10:'10'},
                        allowCross=True,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                id={
                    'type': 'conv-padding-custom-container-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                style={'display': 'none'},  # Hide initially
            ),
            html.Hr(),
            html.Label('Use Bias Term', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'conv-use-bias-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                multi=True,
            ),
            html.Hr(),
        ])
    if layer_type == 'pool':
        return html.Div([
            html.Label(f'Kernel Size of Pooling Layer {index}', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'pool-kernel-size-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                min=1,
                max=10,
                step=1,
                value=[2, 4],
                marks={1:'1', 10:'10'},
                allowCross=True,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label(f'Kernel Stride in Pooling Layer {index}', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'pool-stride-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                min=1,
                max=10,
                step=1,
                value=[2,4],
                marks={1:'1', 10:'10'},
                allowCross=True,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Hr(),
        ])
    if layer_type == 'batchnorm':
        return html.Div([
            html.Label(f'Number of Features for BatchNorm Layer {index} Set by Previous Layer Input Channels', style={'text-decoration': 'underline'}),
            # dcc.RangeSlider(
            #     id={
            #         'type': 'batch-features-hyperparam',
            #         'model': model_type,
            #         'agent': agent_type,
            #         'index': index,
            #     },
            #     min=1,
            #     max=1024,
            #     step=1,
            #     value=32,
            #     marks={1:'1', 1024:'1024'},
            #     tooltip={"placement": "bottom", "always_visible": True},
            # ),
        ])
    if layer_type == 'dropout':
        return html.Div([
            html.Label(f'Probability of Zero-ed Element for Dropout Layer {index}', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'dropout-prob-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
            ],
            multi=True,
            ),
            html.Hr(),
        ])


def generate_hidden_layer_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Hidden Layers'),
        dcc.RangeSlider(
            id={
                'type': 'hidden-layers-slider',
                'model': model_type,
                'agent': agent_type
            },
            min=0,
            max=10,
            value=[1, 2],  # Default range
            marks={0: "0", 5: "5", 10: "10"},
            step=1,  # anchor slider to marks
            # pushable=1,  # allow pushing the slider,
            allowCross=True,  # allow selecting one value
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id={
            'type':'hidden-layers-hyperparam',
            'model':model_type,
            'agent':agent_type
        })
    ])

def generate_layer_hyperparam_component(agent_type, model_type, num_layers):
    tabs = []
    for layer_num in range(1, num_layers + 1):
        tab = dcc.Tab([
            generate_layer_type_hyperparam_component(agent_type, model_type, layer_num)
        ],
        label=f'Layer {layer_num}'
        )
        # Append tab to tabs array
        tabs.append(tab)
    
    return tabs

def generate_layer_type_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.Div(
                [
                    html.H6(f'Layer Type', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id={
                            'type': f'layer-type-hyperparam',
                            'model': model_type,
                            'agent': agent_type,
                            'index': layer_num,
                        },
                        options=[{'label': i.title(), 'value': i} for i in ['transformer encoder', 'dense', 'conv2d', 'maxpool2d', 'dropout', 'batchnorm2d', 'flatten', 'relu', 'tanh']],
                        multi=True,
                        style={'width': '200px'}
                    ),
                ],
                style={'width':'100px', 'display': 'flex', 'alignItems': 'center'}
            ),
            html.Div(
                id={
                    'type': 'layer-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num,
                },
                style={'margin-left': '10px'}
            ),
        ]
    )

def generate_layer_hyperparam_tab(agent_type, model_type, layer_num, layer_types):
    tabs = []
    for layer_type in layer_types:
        if layer_type == 'dense':
            tab = dcc.Tab(
                [item for item in generate_dense_layer_hyperparam_component(agent_type, model_type, layer_num)],
            label=f'{layer_type.title()}'
            )
        elif layer_type == 'conv2d':
            tab = dcc.Tab(
                [item for item in generate_conv_layer_hyperparam_component(agent_type, model_type, layer_num)],
            label=f'{layer_type.title()}'
            )
        tabs.append(tab)
    
    return dbc.Tabs(tabs)

def generate_dense_layer_hyperparam_component(agent_type, model_type, layer_num):
    params = []
    params.append(generate_hidden_units_per_layer_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_dense_bias_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_kernel_initializer_hyperparam_component(agent_type, model_type, layer_num))

    return params

def generate_hidden_units_per_layer_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Neurons'),
            dcc.Dropdown(
                id={
                    'type': f'layer-units-slider',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': i, 'value': i} for i in [8, 16, 32, 64, 128, 256, 512, 1024]],
                multi=True,
                style={'width':'200px'}
            ),
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_dense_bias_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Use Bias', style={'margin-right': '10px'}),
            dcc.Dropdown(
            id={
                'type': 'dense-bias-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num,
            },
            options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
            multi=True,
            style={'width': '200px'},
            )
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_kernel_initializer_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div([
        html.Div(
            id={
                'type': 'kernel-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num,
            },
            style={'display': 'flex', 'alignItems': 'center'},
            children=[
                html.H6(f'Kernel Function', style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id={
                        'type': 'kernel-function-hyperparam',
                        'model': model_type,
                        'agent': agent_type,
                        'index': layer_num,
                    },
                    options=[
                        {'label': "Default", 'value': "default"},
                        {'label': "Constant", 'value': "constant"},
                        {'label': "Xavier Uniform", 'value': "xavier_uniform"},
                        {'label': "Xavier Normal", 'value': "xavier_normal"},
                        {'label': "Kaiming Uniform", 'value': "kaiming_uniform"},
                        {'label': "Kaiming Normal", 'value': "kaiming_normal"},
                        {'label': "Zeros", 'value': "zeros"},
                        {'label': "Ones", 'value': "ones"},
                        {'label': "Uniform", 'value': "uniform"},
                        {'label': "Normal", 'value': "normal"},
                        {'label': "Truncated Normal", 'value': "truncated_normal"},
                        {'label': "Variance Scaling", 'value': "variance_scaling"},
                    ],
                    placeholder="Kernel Function",
                    multi=True,
                    style={'width':'200px'}
                ),
            ],
        ),
        html.Div(
            id={
                'type': 'hyperparam-kernel-options',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num,
            },
            style={'margin-left': '10px'},
            children=[
                html.H6(
                    id={
                        'type': 'kernel-options-header',
                        'model': model_type,
                        'agent': agent_type,
                        'index': layer_num,
                    },
                    children=['Kernel Options'],
                    hidden=True
                ),
                dcc.Tabs(
                    id={
                        'type': 'kernel-options-tabs',
                        'model': model_type,
                        'agent': agent_type,
                        'index': layer_num,
                    },
                ),
            ]
        )
    ])

def generate_kernel_options_hyperparam_component(agent_type, model_type, layer_num, selected_initializers):
    # Dictionary mapping the initializer names to the corresponding function
    kernel_input_creators = {
        "variance_scaling": generate_variance_scaling_hyperparam_inputs,
        "constant": generate_constant_kernel_hyperparam_inputs,
        "normal": generate_normal_kernel_hyperparam_inputs,
        "uniform": generate_uniform_kernel_hyperparam_inputs,
        "truncated_normal": generate_truncated_normal_kernel_hyperparam_inputs,
        "kaiming_normal": generate_kaiming_normal_hyperparam_inputs,
        "kaiming_uniform": generate_kaiming_uniform_hyperparam_inputs,
        "xavier_normal": generate_xavier_normal_hyperparam_inputs,
        "xavier_uniform": generate_xavier_uniform_hyperparam_inputs,
    }

    tabs = [] # empty list for adding tabs for each initializer in selected initializers
    
    for initializer in selected_initializers:
        if initializer in kernel_input_creators:
            #DEBUG
            print(f'generate kernel options hyperparam component: initializer {initializer} found')
            tabs.append(
                kernel_input_creators.get(initializer)(agent_type, model_type, layer_num)
            )
        
    return tabs


def generate_xavier_uniform_hyperparam_inputs(agent_type, model_type, layer_num):
    """Component for xavier uniform initializer hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num
                },
            children=[
                html.Label('Gain', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'xavier-uniform-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        'index': layer_num
                    },
                    options=[
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                    ],
                     multi=True,
                ),
                html.Hr(),
            ],
        )
        ],
        label='Xavier Uniform'
    )


def generate_xavier_normal_hyperparam_inputs(agent_type, model_type, layer_num):
    """Component for xavier normal initializer hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num
                },
            children=[
                html.Label('Gain', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'xavier-normal-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        'index': layer_num
                    },
                    options=[
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                    ],
                     multi=True,
                ),
                html.Hr(),
            ],
        )
        ],
        label="Xavier Normal"
    )

def generate_kaiming_uniform_hyperparam_inputs(agent_type, model_type, layer_num):
    """Component for kaiming uniform initializer sweep hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num
                },
            children=[
                html.Label('Mode', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'kaiming-uniform-mode-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        'index': layer_num
                        },
                    options=[
                            {'label': 'fan in', 'value': 'fan_in'},
                            {'label': 'fan out', 'value': 'fan_out'},
                        ],
                    multi=True,
                ),
                html.Hr(),
            ]
        )
        ],
        label='Kaiming Uniform'
    )

def generate_kaiming_normal_hyperparam_inputs(agent_type, model_type, layer_num):
    """Component for kaiming normal initializer sweep hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num
                },
            children=[
                html.Label('Mode', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'kaiming-normal-mode-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        'index': layer_num
                        },
                    options=[
                            {'label': 'fan in', 'value': 'fan_in'},
                            {'label': 'fan out', 'value': 'fan_out'},
                        ],
                    multi=True,
                ),
                html.Hr(),
            ]
        )
    ],
    label='Kaiming Normal'
    )

def generate_variance_scaling_hyperparam_inputs(agent_type, model_type, layer_num):
    return dcc.Tab([
        html.Div([
            html.Label('Scale', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'variance-scaling-scale-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                ],
                multi=True,
            ),
            html.Label('Mode', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'variance-scaling-mode-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': mode, 'value': mode} for mode in ['fan_in', 'fan_out', 'fan_avg']],
                placeholder="Mode",
                multi=True
            ),
            html.Label('Distribution', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'variance-scaling-distribution-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': dist, 'value': dist} for dist in ['truncated normal', 'uniform']],
                placeholder="Distribution",
                multi=True
            ),
        ])
    ],
    label='Variance Scaling')

def generate_constant_kernel_hyperparam_inputs(agent_type, model_type, layer_num):
    return dcc.Tab([
        html.Div([
            html.Label('Value', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'constant-value-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.001', 'value': 0.001},
                    {'label': '0.002', 'value': 0.002},
                    {'label': '0.003', 'value': 0.003},
                    {'label': '0.004', 'value': 0.004},
                    {'label': '0.005', 'value': 0.005},
                    {'label': '0.006', 'value': 0.006},
                    {'label': '0.007', 'value': 0.007},
                    {'label': '0.008', 'value': 0.008},
                    {'label': '0.009', 'value': 0.009},
                    {'label': '0.01', 'value': 0.01},
                    {'label': '0.02', 'value': 0.02},
                    {'label': '0.03', 'value': 0.03},
                    {'label': '0.04', 'value': 0.04},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.06', 'value': 0.06},
                    {'label': '0.07', 'value': 0.07},
                    {'label': '0.08', 'value': 0.08},
                    {'label': '0.09', 'value': 0.09},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                ],
                multi=True,
            ),
        ])
    ],
    label='Constant')

def generate_normal_kernel_hyperparam_inputs(agent_type, model_type, layer_num):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ])
    ],
    label='Random Normal')

def generate_uniform_kernel_hyperparam_inputs(agent_type, model_type, layer_num):
    return dcc.Tab([
        html.Div([ 
            html.Label('Minimum', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-uniform-minval-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '-0.1', 'value': -0.1},
                    {'label': '-0.2', 'value': -0.2},
                    {'label': '-0.3', 'value': -0.3},
                    {'label': '-0.4', 'value': -0.4},
                    {'label': '-0.5', 'value': -0.5},
                    {'label': '-0.6', 'value': -0.6},
                    {'label': '-0.7', 'value': -0.7},
                    {'label': '-0.8', 'value': -0.8},
                    {'label': '-0.9', 'value': -0.9},
                    {'label': '-1.0', 'value': -1.0},
                ],
                multi=True,
            ),
            html.Label('Maximum', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-uniform-maxval-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ]),
    ],
    label='Random Uniform')

def generate_truncated_normal_kernel_hyperparam_inputs(agent_type, model_type, layer_num):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'truncated-normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'truncated-normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ])
    ],
    label='Truncated Normal')

def generate_conv_layer_hyperparam_component(agent_type, model_type, layer_num):
    params = [] 
    params.append(generate_out_channels_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_kernel_size_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_stride_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_padding_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_conv_bias_hyperparam_component(agent_type, model_type, layer_num))
    params.append(generate_kernel_initializer_hyperparam_component(agent_type, model_type, layer_num))

    return params

def generate_out_channels_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Out Channels'),
            dcc.Dropdown(
                id={
                    'type': 'out-channels-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': i, 'value': i} for i in [8, 16, 32, 64, 128, 256, 512, 1024]],
                multi=True,
                style={'width':'200px'}
            ),
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_kernel_size_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Kernel Size'),
            dcc.Dropdown(
                id={
                    'type': 'kernel-size-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': i, 'value': i} for i in range(1, 11)],
                multi=True,
                style={'width':'200px'}
            ),
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_stride_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Stride'),
            dcc.Dropdown(
                id={
                    'type': 'stride-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': i, 'value': i} for i in range(1,6)],
                multi=True,
                style={'width':'200px'}
            ),
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_padding_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Padding'),
            dcc.Dropdown(
                id={
                    'type': 'padding-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': layer_num
                },
                options=[{'label': i, 'value': i} for i in range(6)],
                multi=True,
                style={'width':'200px'}
            ),
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_conv_bias_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div(
        [
            html.H6(f'Use Bias', style={'margin-right': '10px'}),
            dcc.Dropdown(
            id={
                'type': 'conv-bias-hyperparam',
                'model': model_type,
                'agent': agent_type,
                'index': layer_num,
            },
            options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
            multi=True,
            style={'width': '200px'},
            )
        ],
        style={'display': 'flex', 'alignItems': 'center'}
    )

def generate_activation_function_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Activation Function'),
        dcc.Dropdown(
            id={
                'type': 'activation-function-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
            placeholder="Activation Function",
            multi=True,
        ),
    ])

def generate_optimizer_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Optimizer'),
        dcc.Dropdown(
            id={
                'type': 'optimizer-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
            placeholder="Optimizer",
            multi=True,
        ),
        html.Div(
            id=
            {
                'type':'optimizer-options-hyperparams',
                'model': model_type,
                'agent': agent_type,
            }
        )
    ])

def generate_trace_decay_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5(f"{model_type.capitalize()} Trace Decay"),
        dcc.Dropdown(
            id={
                'type': 'trace-decay-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '0.99', 'value': 0.99},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
        ),
    ])

def generate_tau_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5("Tau"),
        dcc.Dropdown(
            id={
                'type': 'tau-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            options=[
                    {'label': '0.001', 'value': 0.001},
                    {'label': '0.005', 'value': 0.005},
                    {'label': '0.01', 'value': 0.01},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.5', 'value': 0.5},
                ],
                multi=True,
        ),
    ])

def generate_batch_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5("Batch Size"),
        dcc.Dropdown(
            id={
                'type': 'batch-size-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            options=[{'label': i, 'value': i} for i in [8, 16, 32, 64, 128, 256, 512, 1024]],
            placeholder="Batch Sizes",
            multi=True,
        ),
    ])

def generate_noise_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5("Noise"),
        dcc.Dropdown(
            id={
                'type': 'noise-function-hyperparam',
                'model': model_type,
                'agent': agent_type,
                },
                options=[{'label': i, 'value': i} for i in ["Ornstein-Uhlenbeck", "Normal", "Uniform"]],
                placeholder="Noise Function",
                multi=True,
            ),
        html.Div(
            id={
                'type':'noise-options-hyperparam',
                'model':model_type,
                'agent':agent_type
            },
            children=[
                html.H6(
                    id={
                        'type': 'noise-options-header',
                        'model': model_type,
                        'agent': agent_type
                    },
                    children=['Noise Options'],
                    hidden=True
                ),
                dcc.Tabs(
                    id={
                        'type': 'noise-options-tabs',
                        'model': model_type,
                        'agent': agent_type
                    }
                ),
            ]
        )
    ])




def generate_noise_options_hyperparams_component(agent_type, model_type, noise_functions):
    # Dictionary mapping the initializer names to the corresponding function
    noise_input_creators = {
        "Ornstein-Uhlenbeck": generate_OU_noise_hyperparam_inputs,
        "Uniform": generate_uniform_noise_hyperparam_inputs,
        "Normal": generate_normal_noise_hyperparam_inputs,        
    }


    tabs = [] # empty list for adding tabs for each initializer in selected initializers
    
    for noise in noise_functions:
        if noise in noise_input_creators:
            tabs.append(
                noise_input_creators.get(noise)(agent_type, model_type)
            )
        
    return tabs

def generate_OU_noise_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'ou-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Mean Reversion', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'ou-theta-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Volatility', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'ou-sigma-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ])
    ],
    label='Ornstein-Uhlenbeck')

def generate_uniform_noise_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Minimum Value', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'uniform-min-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '-0.1', 'value': -0.1},
                    {'label': '-0.2', 'value': -0.2},
                    {'label': '-0.3', 'value': -0.3},
                    {'label': '-0.4', 'value': -0.4},
                    {'label': '-0.5', 'value': -0.5},
                    {'label': '-0.6', 'value': -0.6},
                    {'label': '-0.7', 'value': -0.7},
                    {'label': '-0.8', 'value': -0.8},
                    {'label': '-0.9', 'value': -0.9},
                    {'label': '-1.0', 'value': -1.0},
                ],
                multi=True,
            ),
            html.Label('Maximum Value', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'uniform-max-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ])
    ],
    label='Uniform')

def generate_normal_noise_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.15', 'value': 0.15},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.25', 'value': 0.25},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.35', 'value': 0.35},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.45', 'value': 0.45},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.55', 'value': 0.55},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.65', 'value': 0.65},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.75', 'value': 0.75},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.85', 'value': 0.85},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '0.95', 'value': 0.95},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ])
    ],
    label='Normal')

def create_replay_buffer_hyperparam_component(agent_type, model_type):
    pass

def create_replay_buffer_size_hyperparam_component(agent_type, model_type):
    return html.Div(
        [
            html.Label('Buffer Size', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'buffer-size-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '100,000', 'value': 100_000},
                    {'label': '500,000', 'value': 500_000},
                    {'label': '1,000,000', 'value': 1_000_000}
                    ],
                placeholder="Replay Buffer Size",
                multi=True,
            ),
        ]
    )

def create_goal_strategy_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'goal-strategy-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Future', 'Final', 'None']],
                placeholder="Goal Strategy",
                multi=True,
            ),
            html.Div(
                id={
                    'type':'goal-strategy-options-hyperparam',
                    'model':'none',
                    'agent':agent_type,
                }
            )
        ]
    )

def create_tolerance_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Goal Tolerance', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'goal-tolerance-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.15', 'value': 0.15},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.25', 'value': 0.25},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.35', 'value': 0.35},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.45', 'value': 0.45},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.55', 'value': 0.55},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.65', 'value': 0.65},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.75', 'value': 0.75},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.85', 'value': 0.85},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '0.95', 'value': 0.95},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ]
    )

def create_epsilon_greedy_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'Epsilon Greedy', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':f'epsilon-greedy-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
        ]
    )

def create_input_normalizer_options_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'Minimum/Maximum Clip Value', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'norm-clip-value-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.0', 'value': 0.0},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                    {'label': '4.0', 'value': 4.0},
                    {'label': '5.0', 'value': 5.0},
                    {'label': '6.0', 'value': 6.0},
                    {'label': '7.0', 'value': 7.0},
                    {'label': '8.0', 'value': 8.0},
                    {'label': '9.0', 'value': 9.0},
                    {'label': '10.0', 'value': 10.0},
                ],
                multi=True,
            ),
        ]
    )

def create_input_normalizer_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-input-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i} for i in ['True', 'False']],
                placeholder="Normalize Input",
                multi=True,
            ),
            html.Div(
                id={
                    'type':'normalize-options-hyperparam',
                    'model':'none',
                    'agent':agent_type,
                }
            )
        ]
    )

# def create_value_normalizer_hyperparam_input(agent_type, model_type):
#     return html.Div(
#         [
#             dcc.Dropdown(
#                 id={
#                     'type':'normalize-values-hyperparam',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 options=[{'label': i, 'value': i} for i in ['True', 'False']],
#                 placeholder="Normalize Values",
#                 multi=True,
#             ),
#             html.Div(
#                 id={
#                     'type':'normalize-values-options-hyperparam',
#                     'model':model_type,
#                     'agent':agent_type,
#                 }
#             )
#         ]
#     )

# def create_value_normalizer_options_hyperparam_input(agent_type, model_type):
#     return html.Div(
#         [
#             html.Label(f'Minimum/Maximum Clip Value', style={'text-decoration': 'underline'}),
#             dcc.Dropdown(
#                 id={
#                     'type':'value-norm-clip-value-hyperparam',
#                     'model':model_type,
#                     'agent':agent_type,
#                 },
#                 options=[
#                     {'label': '0.0', 'value': 0.0},
#                     {'label': '0.1', 'value': 0.1},
#                     {'label': '0.2', 'value': 0.2},
#                     {'label': '0.3', 'value': 0.3},
#                     {'label': '0.4', 'value': 0.4},
#                     {'label': '0.5', 'value': 0.5},
#                     {'label': '0.6', 'value': 0.6},
#                     {'label': '0.7', 'value': 0.7},
#                     {'label': '0.8', 'value': 0.8},
#                     {'label': '0.9', 'value': 0.9},
#                     {'label': '1.0', 'value': 1.0},
#                     {'label': '2.0', 'value': 2.0},
#                     {'label': '3.0', 'value': 3.0},
#                     {'label': '4.0', 'value': 4.0},
#                     {'label': '5.0', 'value': 5.0},
#                     {'label': '6.0', 'value': 6.0},
#                     {'label': '7.0', 'value': 7.0},
#                     {'label': '8.0', 'value': 8.0},
#                     {'label': '9.0', 'value': 9.0},
#                     {'label': '10.0', 'value': 10.0},
#                     {'label': 'Infinity', 'value': 'infinity'},
#                 ],
#                 multi=True,
#             ),
#         ]
#     )

def create_grad_clip_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Gradient Clip', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'grad-clip-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                    {'label': '2.0', 'value': 2.0},
                    {'label': '3.0', 'value': 3.0},
                    {'label': '4.0', 'value': 4.0},
                    {'label': '5.0', 'value': 5.0},
                    {'label': '10.0', 'value': 10.0},
                    {'label': '20.0', 'value': 20.0},
                    {'label': '30.0', 'value': 30.0},
                    {'label': '40.0', 'value': 40.0},
                    {'label': '50.0', 'value': 50.0},
                    {'label': 'Infinity', 'value': 'infinity'},
                ],
                multi=True,
            ),
        ]
    )

def create_loss_coeff_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label(f'{model_type.capitalize()} Loss Coefficient', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'loss-coeff-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0', 'value': 1.0},
                    {'label': '1.1', 'value': 1.1},
                    {'label': '1.2', 'value': 1.2},
                    {'label': '1.3', 'value': 1.3},
                    {'label': '1.4', 'value': 1.4},
                    {'label': '1.5', 'value': 1.5},
                    {'label': '1.6', 'value': 1.6},
                    {'label': '1.7', 'value': 1.7},
                    {'label': '1.8', 'value': 1.8},
                    {'label': '1.9', 'value': 1.9},
                    {'label': '2.0', 'value': 2.0},
                ],
                multi=True,
            ),
        ]
    )

def create_advantage_normalizer_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-advantage-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [True, False]],
                placeholder="Normalize Advantages",
                multi=True,
            ),
        ]
    )

def create_value_normalizer_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-values-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [True, False]],
                placeholder="Normalize Values",
                multi=True,
            ),
            html.Div(
                id = {
                    'type':'norm-clip-hyperparam-block',
                    'model':model_type,
                    'agent':agent_type,
                },
                children = [
                    html.Label('Norm Clip'),
                    dcc.Dropdown(
                        id={
                            'type': 'norm-values-clip-hyperparam',
                            'model': model_type,
                            'agent': agent_type
                        },
                        options=[{'label': str(i), 'value': i} for i in [0.1, 0.5, 1.0, 5.0, 10.0]],
                    )
                ],
                style={'margin-left': '10px', 'display':'none'}
            )
        ]
    )

def create_normalize_layers_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'normalize-layers-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i=='True'} for i in ['True', 'False']],
                placeholder="Normalize Layers",
                multi=True,
            ),
        ]
    )

def create_clamp_output_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'clamp-value-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[
                    {'label': '0.01', 'value': 0.01},
                    {'label': '0.02', 'value': 0.02},
                    {'label': '0.03', 'value': 0.03},
                    {'label': '0.04', 'value': 0.04},
                    {'label': '0.05', 'value': 0.05},
                    {'label': '0.06', 'value': 0.06},
                    {'label': '0.07', 'value': 0.07},
                    {'label': '0.08', 'value': 0.08},
                    {'label': '0.09', 'value': 0.09},
                    {'label': '0.1', 'value': 0.1},
                    {'label': '0.2', 'value': 0.2},
                    {'label': '0.3', 'value': 0.3},
                    {'label': '0.4', 'value': 0.4},
                    {'label': '0.5', 'value': 0.5},
                    {'label': '0.6', 'value': 0.6},
                    {'label': '0.7', 'value': 0.7},
                    {'label': '0.8', 'value': 0.8},
                    {'label': '0.9', 'value': 0.9},
                    {'label': '1.0 (no clamp)', 'value': 1.0}, # Equals no clamp
                ],
                placeholder="Clamp Value",
                multi=True,
            ),
        ]
    )

def update_goal_strategy_hyperparam_options(agent_type, goal_strategy):
    if goal_strategy == 'future':
        return future_goal_strategy_hyperparam_options(agent_type)
    return html.Div()

def future_goal_strategy_hyperparam_options(agent_type):
    return html.Div(
        [
            html.Label('Number of Future Goals', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type':'future-goals-hyperparam',
                    'model':'none',
                    'agent':agent_type,
                },
                min=1,
                max=10,
                step=1,
                value=[4,8],  # Default position
                marks={1:'1', 10:'10'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
                allowCross=True,
            ),
        ]
    )

def create_learning_rate_scheduler_hyperparam_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate Scheduler', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-scheduler-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': i, 'value': i.lower()} for i in ['Step', 'Exponential', 'CosineAnnealing', 'None']],
                multi=True,
                placeholder="Select Learning Rate Scheduler(s)",
            ),
            html.Div(
                id={
                    'type':'lr-scheduler-tabs-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                }
            )
        ]
    )

def update_lr_scheduler_hyperparam_options(agent_type, model_type, lr_schedulers):
    #DEBUG
    print(f'lr_schedulers:{lr_schedulers}')
    if not lr_schedulers:
        return html.Div()

    tabs = []
    # tab_contents = []

    for scheduler in lr_schedulers:
        tab_label = scheduler.title()
        #DEBUG
        print(f'tab_label:{tab_label}')
        # tabs.append(dcc.Tab(label=tab_label, value=scheduler))

        if scheduler == 'step':
            tab_contents = lr_step_scheduler_hyperparam_options(agent_type, model_type)
            tab_label = 'Step'
        elif scheduler == 'exponential':
            tab_contents = lr_exponential_scheduler_hyperparam_options(agent_type, model_type)
            tab_label = 'Exponential'
        elif scheduler == 'cosineannealing':
            tab_contents = lr_cosineannealing_scheduler_hyperparam_options(agent_type, model_type)
            tab_label = 'Cosine Annealing'

        tab = dcc.Tab(
            label=tab_label,
            children=tab_contents
        )

        tabs.append(tab)

    return dcc.Tabs(
        id={
            'type': 'lr-scheduler-tabs-options-hyperparam',
            'model': model_type,
            'agent': agent_type,
        },
        value=lr_schedulers[0] if lr_schedulers else None,
        children=tabs,
        style={'margin-top': '10px'}
    )

def lr_cosineannealing_scheduler_hyperparam_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('T max (max iters)', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-t-max-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [100,500,1000,5000,10000,50000,100000]],
                multi=True,
                placeholder="Select T-Max Value(s)",

            ),
            html.Label('Eta min (min LR)', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-eta-min-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [0.001,0.0001,0.00001,0.000001]],
                multi=True,
                placeholder="Select Eta Min Value(s)",
            ),
        ]
    )

def lr_exponential_scheduler_hyperparam_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-gamma-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]],
                multi=True,
                placeholder="Select Gamma Value(s)",
            ),
        ]
    )

def lr_step_scheduler_hyperparam_options(agent_type, model_type):
    return html.Div(
        [
            html.Label('Step Size', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-step-size-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [0.001, 0.0001, 0.00001, 0.000001]],
                multi=True,
                placeholder="Select Step Size Value(s)",
            ),
            html.Label('Gamma (decay)', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'lr-gamma-hyperparam',
                    'model':model_type,
                    'agent':agent_type,
                },
                options=[{'label': str(i), 'value': i} for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]],
                multi=True,
                placeholder="Select Gamma Value(s)",
            ),
        ]
    )

def create_seed_component(page):
    # return html.Div([
        # html.Label('Seed', style={'text-decoration': 'underline'}),
    return dcc.Input(
            id={
                'type':'seed',
                'page': page,
            },
            type='number',
            min=1,
            placeholder="Blank for random"
        )
    # ])

## WEIGHTS AND BIASES FUNCTIONS
def create_wandb_login(page):
    return html.Div([
        dcc.Input(
            id={
                'type': 'wandb-api-key',
                'page': page,
            },
            type='password',
            placeholder='WandB API Key',
            value=''
        ),
        html.Button(
            'Login',
            id={
                'type': 'wandb-login',
                'page': page,
            },
            n_clicks=0,
        ),
        html.Div(id={
            'type': 'wandb-login-feedback',
            'page': page,
        })
    ])

def create_wandb_project_dropdown(page):
    projects = get_projects()
    return html.Div([
            # html.Label("Select a Project:"),
            dcc.Dropdown(
            id={'type':'projects-dropdown', 'page':page},
            options=[{'label': project, 'value': project} for project in projects],
            placeholder="Select a W&B Project",
            className="dropdown-field"
            )
        ])

def create_sweeps_dropdown(page):
    return html.Div([
        dcc.Dropdown(
            id={'type':'sweeps-dropdown', 'page':page},
            options=[],
            multi=True,
            placeholder="Select a W&B Sweep",
            )
    ])

def format_wandb_config_param(config, param_name, all_values, all_ids, dash_id, model, agent, index=None, is_range=False):
    """
    Formats a parameter for the Weights & Biases (wandb) config depending on its structure:
    single value, range (min/max), or list of values.

    Args:
        config (dict): The wandb config dictionary.
        param_name (str): Name of the parameter to format.
        all_values (list): List of all parameter values.
        all_ids (list): List of all parameter IDs.
        dash_id (str): The specific parameter ID to look up.
        model (str): The model name.
        agent (str): The agent name.
        index (int): The index number.
        is_range (bool): if parameter should be configured between a min/max range
        
    Returns:
        dict: The updated wandb config dictionary.
    """
    if index:
        value = get_specific_value_id(all_values, all_ids, dash_id, model, agent, index)
    else:
        value = get_specific_value(all_values, all_ids, dash_id, model, agent)

    if is_range:
        # Check if the range has a single value or a min/max range
        if value[0] == value[1]:
            param = {"value": value[0]}
        else:
            param = {"min": value[0], "max": value[1]}

    else:
        # Check if it's a single value or a list of values
        if isinstance(value, list):
            param = {"values": value}
        else:
            param = {"value": value}

    if index:
        config["parameters"][f"{agent}_{model}_{index}_{param_name}"] = param
    else:
        config["parameters"][f"{agent}_{model}_{param_name}"] = param

    return config

def format_wandb_optimizer_options(config, param_name, all_values, all_ids, model, agent):
    for value in config["parameters"][f"{agent}_{model}_{param_name}"]['values']:
        if value == 'Adam':
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_weight_decay"] = {"values": get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', model, agent)}

        elif value == 'Adagrad':
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_weight_decay"] = {"values": get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', model, agent)}
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_lr_decay"] = {"values": get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', model, agent)}

        elif value == 'RMSprop':
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_weight_decay"] = {"values": get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', model, agent)}
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_momentum"] = {"values": get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', model, agent)}

        elif value == 'SGD':
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_weight_decay"] = {"values": get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', model, agent)}
            config["parameters"][f"{agent}_{model}_{param_name}_{value}_momentum"] = {"values": get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', model, agent)}
    return config
    
def format_wandb_kernel(config, all_indexed_values, all_indexed_ids, model, agent, layer):
    if isinstance(layer, int):
        layer_types = get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-type-hyperparam', model, agent, layer)
        for layer_type in layer_types:
            if layer_type in ['dense', 'conv2d']:
                # Add selected kernels for layer to sweep config
                config = format_wandb_config_param(config, "kernel", all_indexed_values, all_indexed_ids, 'kernel-function-hyperparam', model, agent, layer)
                # Add params for each selected kernel to sweep config
                config = format_wandb_kernel_options(config, all_indexed_values, all_indexed_ids, model, agent, layer)
    else:
        # Output layer kernels
        config = format_wandb_config_param(config, "kernel", all_indexed_values, all_indexed_ids, "kernel-function-hyperparam", model, agent, 'output')
        # Output layer kernel options
        config = format_wandb_kernel_options(config, all_indexed_values, all_indexed_ids, model, agent, 'output')

    return config

def format_wandb_kernel_options(config, all_values, all_ids, model, agent, layer_num):
    for kernel in get_specific_value_id(all_values, all_ids, 'kernel-function-hyperparam', model, agent, layer_num):
        if kernel != 'default':
            # initialize empty config dictionary for parameters
            param = {}

            if kernel == "constant":
                config = format_wandb_config_param(config, "kernel", all_values, all_ids, 'constant-value-hyperparam', model, agent, layer_num)

            elif kernel == "variance_scaling":
                # scale
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_scale"] = {"values": get_specific_value_id(all_values, all_ids, 'variance-scaling-scale-hyperparam', model, agent, layer_num)}

                # mode
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_mode"] = {"values": get_specific_value_id(all_values, all_ids, 'variance-scaling-mode-hyperparam', model, agent, layer_num)}

                # distribution
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_distribution"] = {"values": get_specific_value_id(all_values, all_ids, 'variance-scaling-distribution-hyperparam', model, agent, layer_num)}

            elif kernel == "uniform":
                # maxval
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_maxval"] = {"values": get_specific_value_id(all_values, all_ids, 'random-uniform-maxval-hyperparam', model, agent, layer_num)}

                # minval
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_minval"] = {"values": get_specific_value_id(all_values, all_ids, 'random-uniform-minval-hyperparam', model, agent, layer_num)}

            elif kernel == "normal":
                # mean
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_mean"] = {"values": get_specific_value_id(all_values, all_ids, 'random-normal-mean-hyperparam', model, agent, layer_num)}

                # stddev
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_stddev"] = {"values": get_specific_value_id(all_values, all_ids, 'random-normal-stddev-hyperparam', model, agent, layer_num)}
    
            elif kernel == "truncated_normal":
                # mean
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_mean"] = {"values": get_specific_value_id(all_values, all_ids, 'truncated-normal-mean-hyperparam', model, agent, layer_num)}

                # stddev
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_stddev"] = {"values": get_specific_value_id(all_values, all_ids, 'truncated-normal-stddev-hyperparam', model, agent, layer_num)}

            elif kernel == "xavier_uniform":
                # gain
                config["parameters"][f"{agent}_{kernel}_{layer_num}_{model}_gain"] = {"values": get_specific_value_id(all_values, all_ids, 'xavier-uniform-gain-hyperparam', model, agent, layer_num)}

            elif kernel == "xavier_normal":
                # gain
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_gain"] = {"values": get_specific_value_id(all_values, all_ids, 'xavier-normal-gain-hyperparam', model, agent, layer_num)}

            elif kernel == "kaiming_uniform":
                # mode
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_mode"] = {"values": get_specific_value_id(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', model, agent, layer_num)}

            elif kernel == "kaiming_normal":
                # mode
                config["parameters"][f"{agent}_{model}_{layer_num}_{kernel}_mode"] = {"values": get_specific_value_id(all_values, all_ids, 'kaiming-normal-mode-hyperparam', model, agent, layer_num)}

            else:
                if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                    "uniform", "normal", "truncated_normal", "variance_scaling"]:
                    raise ValueError(f"Unknown kernel: {kernel}")

    return config

def format_wandb_lr_scheduler(config, all_values, all_ids, model, agent):
    schedulers = get_specific_value(all_values, all_ids, 'lr-scheduler-hyperparam', model, agent)
    for scheduler in schedulers:
        config = format_wandb_config_param(config, "scheduler", all_values, all_ids, 'lr-scheduler-hyperparam', model, agent)
        config = format_wandb_lr_scheduler_options(config, all_values, all_ids, model, agent, scheduler)

    return config

def format_wandb_lr_scheduler_options(config, all_values, all_ids, model, agent, scheduler):
    if scheduler == 'step':
        config = format_wandb_steplr_options(config, all_values, all_ids, model, agent)
    elif scheduler == 'exponential':
        config = format_wandb_exponentiallr_options(config, all_values, all_ids, model, agent)
    elif scheduler == 'cosineannealing':
        config = format_wandb_cosineannealinglr_options(config, all_values, all_ids, model, agent)
    
    return config

def format_wandb_steplr_options(config, all_values, all_ids, model, agent):
    config = format_wandb_config_param(config, "step_size", all_values, all_ids, 'lr-step-size-hyperparam', model, agent)
    config = format_wandb_config_param(config, "gamma", all_values, all_ids, 'lr-gamma-hyperparam', model, agent)

def format_wandb_exponentiallr_options(config, all_values, all_ids, model, agent):
    config = format_wandb_config_param(config, "step_size", all_values, all_ids, 'lr-gamma-hyperparam', model, agent)

def format_wandb_cosineannealinglr_options(config, all_values, all_ids, model, agent):
    config = format_wandb_config_param(config, "step_size", all_values, all_ids, 'lr-t-max-hyperparam', model, agent)
    config = format_wandb_config_param(config, "step_size", all_values, all_ids, 'lr-eta-min-hyperparam', model, agent)

def format_wandb_model_layers(config, all_values, all_ids, all_indexed_values, all_indexed_ids, model, agent):
    # Get num layers in model
    num_layers = get_specific_value(all_values, all_ids, 'hidden-layers-slider', model, agent)[1]
    # Add num_layers to wandb config
    config = format_wandb_config_param(config, 'num_layers', all_values, all_ids, 'hidden-layers-slider', model, agent, index=None, is_range=True)
    for layer in range(1, num_layers + 1):
        # Get layer types
        layer_types = get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-type-hyperparam', model, agent, layer)
        # Add layer_types to wandb config
        config = format_wandb_config_param(config, 'layer_types', all_indexed_values, all_indexed_ids, 'layer-type-hyperparam', model, agent, layer)
        # Loop through each layer type to assign correct params
        for layer_type in layer_types:
            if layer_type == 'dense':
                config = format_wandb_config_param(config, 'num_units', all_indexed_values, all_indexed_ids, 'layer-units-slider', model, agent, layer)
                config = format_wandb_config_param(config, 'bias', all_indexed_values, all_indexed_ids, 'dense-bias-hyperparam', model, agent, layer)
                config = format_wandb_kernel(config, all_indexed_values, all_indexed_ids, model, agent, layer)
            elif layer_type == 'cnn':
                config = format_wandb_config_param(config, 'out_channels', all_indexed_values, all_indexed_ids, 'layer-units-slider', model, agent, layer)

    return config


def format_wandb_layer_units(config, param_name, all_values, all_ids, all_indexed_values, all_indexed_ids, id, model, agent):
    print(f'num layers:{get_specific_value(all_values, all_ids, id, model, agent)}')
    for i in range(1, get_specific_value(all_values, all_ids, id, model, agent)[1] + 1):
        config["parameters"][agent]["parameters"][f"{model}_layer_{i}_{param_name}"] = {
            "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-units-slider', model, agent, i)   
        }
    return config


def create_wandb_config(method, project, sweep_name, metric_name, metric_goal, env_library, env, env_params, env_wrappers, agent, all_values, all_ids, all_indexed_values, all_indexed_ids):
    #DEBUG
    # print(f'create wandb config fired...')
    sweep_config = {
        "method": method,
        "project": project,
        "name": sweep_name,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": {
            "env_library": {"value": env_library},
            "env_id": {"value": env},
            **{f'env_{param}': {"value":value} for param, value in env_params.items()},
            "env_wrappers": {"values": env_wrappers},
            "model_type": {"value": agent},
        },
            
    }
    # set base config for each agent type
    # for agent in agent_selection:
        # Initialize the dictionary for the agent if it doesn't exist

    # if agent not in sweep_config["parameters"]:
    #     sweep_config["parameters"][agent] = {}
    
    if agent in ["DDPG", "TD3"]:
        sweep_config["parameters"][agent]["parameters"] = {}

        # actor learning rate
        value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'actor', agent)
        config = {"values": value_range}
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_learning_rate"] = config
        
        # critic learning rate
        value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'critic', agent)
        config = {"values": value_range}
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_learning_rate"] = config
        
        # discount
        value_range = get_specific_value(all_values, all_ids, 'discount-slider', 'none', agent)
        config = {"values": value_range}
        sweep_config["parameters"][agent]["parameters"][f"{agent}_discount"] = config
        
        # tau
        value_range = get_specific_value(all_values, all_ids, 'tau-hyperparam', 'none', agent)
        config = {"values": value_range}
        sweep_config["parameters"][agent]["parameters"][f"{agent}_tau"] = config
        
        # epsilon
        value_range = get_specific_value(all_values, all_ids, 'epsilon-greedy-hyperparam', 'none', agent)
        config = {"values": value_range}
        sweep_config["parameters"][agent]["parameters"][f"{agent}_epsilon_greedy"] = config

        # warmup
        sweep_config["parameters"][agent]["parameters"][f"{agent}_warmup"] = \
            {"values": get_specific_value(all_values, all_ids, 'warmup-slider-hyperparam', 'none', agent)}

        # normalize input
        sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_input"] = \
            {"values": get_specific_value(all_values, all_ids, 'normalize-input-hyperparam', 'none', agent)}

        # normalize input options
        # for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_input"]['values']:
        #     if value == 'True':
        #         value_range = get_specific_value(all_values, all_ids, 'norm-clip-value-hyperparam', 'none', agent)
        #         config = {"values": value_range}
        #     sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_clip"] = config

        value_range = get_specific_value(all_values, all_ids, 'norm-clip-value-hyperparam', 'none', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_normalizer_clip"] = config

        # replay buffer size
        sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer_size"] = \
            {"values": get_specific_value(all_values, all_ids, 'buffer-size-hyperparam', 'none', agent)}

        # Get Device
        value_range = get_specific_value(all_values, all_ids, 'device', 'none', agent)
        sweep_config["parameters"][agent]["parameters"][f"{agent}_device"] = {"value": value_range}
        
        # actor cnn layers
        value_range = get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_cnn_layers"] = config

        # actor num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'actor', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_layers"] = config

        # actor activation
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"] = \
            {"values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'actor', agent)}
        #DEBUG
        # print(f'DDPG actor activation set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"]}')

        # actor hidden layers kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent)}
        #DEBUG
        # print(f'DDPG actor kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"]}')

        # actor hidden kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent):
            if kernel != 'default':
                if f"{agent}_actor_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-hidden', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]["parameters"] = config


        # actor output layer kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent)}

        # actor output kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent):
            if kernel != 'default':
                if f"{agent}_actor_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-output', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]["parameters"] = config

        # actor optimizer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"] = \
            {"values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'actor', agent)}
        #DEBUG
        print(f'DDPG actor optimizer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"]}')

        # Actor optimizer options
        for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"]['values']:
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer_{value}_options"] = {'parameters': {}}
            config = {}
            if value == 'Adam':
                value_range = get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

            elif value == 'Adagrad':
                value_range = get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', 'actor', agent)
                config[f'{value}_lr_decay'] = {"values": value_range}

            elif value == 'RMSprop':
                value_range = get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', 'actor', agent)
                config[f'{value}_momentum'] = {"values": value_range}

            elif value == 'SGD':
                value_range = get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', 'actor', agent)
                config[f'{value}_momentum'] = {"values": value_range}
                
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer_{value}_options"]['parameters'] = config
                
        # actor normalize layers
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_normalize_layers"] = \
            {"values": get_specific_value(all_values, all_ids, 'normalize-layers-hyperparam', 'actor', agent)}

        # critic cnn layers
        value_range = get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'critic', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_num_cnn_layers"] = config
        #DEBUG
        # print(f'DDPG critic cnn layers set to {config}')


        # critic state num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-state', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_state_num_layers"] = config
        #DEBUG
        # print(f'DDPG critic state num layers set to {config}')

        # critic merged num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-merged', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_merged_num_layers"] = config
        #DEBUG
        # print(f'DDPG critic merged num layers set to {config}')

        # critic activation
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_activation"] = \
            {"values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'critic', agent)}
        #DEBUG
        # print(f'DDPG critic activation set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_activation"]}')

        # critic hidden layers kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent)}
        #DEBUG
        # print(f'DDPG critic kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"]}')

        # critic hidden kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent):
            if kernel != 'default':
                if f"{agent}_critic_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-hidden', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]["parameters"] = config

        # critic output layer kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent)}

        # critic output kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent):
            if kernel != "default":
                if f"{agent}_critic_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-output', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]["parameters"] = config

        # critic optimizer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"] = \
            {"values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'critic', agent)}
        #DEBUG
        # print(f'DDPG critic optimizer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"]}')

        # Critic optimizer options
        for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"]['values']:
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer_{value}_options"] = {'parameters': {}}
            config = {}
            if value == 'Adam':
                value_range = get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

            elif value == 'Adagrad':
                value_range = get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', 'critic', agent)
                config[f'{value}_lr_decay'] = {"values": value_range}

            elif value == 'RMSprop':
                value_range = get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', 'critic', agent)
                config[f'{value}_momentum'] = {"values": value_range}

            elif value == 'SGD':
                value_range = get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', 'critic', agent)
                config[f'{value}_momentum'] = {"values": value_range}
                
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer_{value}_options"]['parameters'] = config

        # critic normalize layers
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_normalize_layers"] = \
            {"values": get_specific_value(all_values, all_ids, 'normalize-layers-hyperparam', 'critic', agent)}
        
        # replay buffer ## NOT NEEDED
        # sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer"] = {"values": ["ReplayBuffer"]}
        #DEBUG
        # print(f'DDPG replay buffer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer"]}')

        # batch size
        sweep_config["parameters"][agent]["parameters"][f"{agent}_batch_size"] = \
            {"values": get_specific_value(all_values, all_ids, 'batch-size-hyperparam', 'none', agent)}
        #DEBUG
        # print(f'DDPG batch size set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_batch_size"]}')

        # noise
        sweep_config["parameters"][agent]["parameters"][f"{agent}_noise"] = \
            {"values": get_specific_value(all_values, all_ids, 'noise-function-hyperparam', 'none', agent)}
        #DEBUG
        # print(f'DDPG noise set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_noise"]}')

        # noise parameter options
        for noise in get_specific_value(all_values, all_ids, 'noise-function-hyperparam', 'none', agent):
            # Initialize the dictionary for the agent if it doesn't exist
            if f"{agent}_noise_{noise}" not in sweep_config["parameters"][agent]["parameters"]:
                sweep_config["parameters"][agent]["parameters"][f"{agent}_noise_{noise}"]={"parameters":{}}
            
            # initialize empty config dictionary for parameters
            config = {}
            
            if noise == "Ornstein-Uhlenbeck":
                # mean
                value_range = get_specific_value(all_values, all_ids, 'ou-mean-hyperparam', 'none', agent)
                config["mean"] = {"values": value_range}
                
                # theta
                value_range = get_specific_value(all_values, all_ids, 'ou-theta-hyperparam', 'none', agent)
                config["theta"] = {"values": value_range}

                # sigma
                value_range = get_specific_value(all_values, all_ids, 'ou-sigma-hyperparam', 'none', agent)
                config["sigma"] = {"values": value_range}
                
            elif noise == "Normal":
                # mean
                value_range = get_specific_value(all_values, all_ids, 'normal-mean-hyperparam', 'none', agent)
                config["mean"] = {"values": value_range}

                # stddev
                value_range = get_specific_value(all_values, all_ids, 'normal-stddev-hyperparam', 'none', agent)
                config["stddev"] = {"values": value_range}

            elif noise == "Uniform":
                # minval
                value_range = get_specific_value(all_values, all_ids, 'uniform-min-hyperparam', 'none', agent)
                config["minval"] = {"values": value_range}

                # maxval
                value_range = get_specific_value(all_values, all_ids, 'uniform-max-hyperparam', 'none', agent)
                config["maxval"] = {"values": value_range}

            sweep_config["parameters"][agent]["parameters"][f"{agent}_noise_{noise}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG noise set to {config}')

        
        #DEBUG
        # print(f'DDPG critic kernel set to {config}')

        # CNN layer params
        # Actor CNN layers
        for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_{agent}"] = {"parameters":{}}
            config = {}
            config[f"{agent}_actor_cnn_layer_{i}_types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'actor', agent, i)}

            # loop through each type in CNN layer and get the parameters to add to the sweep config
            for value in config[f"{agent}_actor_cnn_layer_{i}_types"]["values"]:
                if value == "conv":
                    config[f"{agent}_actor_cnn_layer_{i}_conv_filters"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'actor', agent, i)
                    }

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                    config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {
                        "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'actor', agent, i)
                    }

                    if config[f"{agent}_actor_cnn_layer_{i}_conv_padding"]["value"] == 'custom':
                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {"value": value_range[0]}
                        else:
                            config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                        
                        # val_config["conv_padding"]["parameters"] = pad_config
                    
                    config[f"{agent}_actor_cnn_layer_{i}_conv_bias"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'actor', agent, i)
                    }
                
                if value == "pool":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                if value == "dropout":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'actor', agent, i)
                    config[f"{agent}_actor_cnn_layer_{i}_dropout_prob"] = {"values": value_range}

                # config["parameters"] = val_config

            sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_{agent}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG actor CNN layers set to {config}')

        # Critic CNN layers
        for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'critic', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_{agent}"] = {"parameters":{}}
            config = {}
            config[f"{agent}_critic_cnn_layer_{i}_types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'critic', agent, i)}

            # loop through each type in CNN layer and get the parameters to add to the sweep config
            for value in config[f"{agent}_critic_cnn_layer_{i}_types"]["values"]:
                if value == "conv":
                    config[f"{agent}_critic_cnn_layer_{i}_conv_filters"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'critic', agent, i)
                    }

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                    config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {
                        "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'critic', agent, i)
                    }

                    if config[f"{agent}_critic_cnn_layer_{i}_conv_padding"]["value"] == 'custom':
                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {"value": value_range[0]}
                        else:
                            config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                    
                    config[f"{agent}_critic_cnn_layer_{i}_conv_bias"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'critic', agent, i)
                    }
                
                if value == "pool":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                if value == "dropout":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'critic', agent, i)
                    config[f"{agent}_critic_cnn_layer_{i}_dropout_prob"] = {"values": value_range}

            sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_{agent}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG critic CNN layers set to {config}')

        # layer units
        # actor layer units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'actor', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"actor_units_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'actor', agent)   
            }
        # critic state units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-state', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_units_state_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'critic-state', agent)
            }
        # critic merged units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-merged', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_units_merged_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'critic-merged', agent)
            }

        # Add save dir
        sweep_config["parameters"][agent]["parameters"][f"{agent}_save_dir"] = \
            {"value": get_specific_value(all_values, all_ids, 'save-dir', 'none', agent)}
        
        if agent == "TD3":
            # Add target action stddev
            sweep_config["parameters"][agent]["parameters"][f"{agent}_target_action_stddev"] = \
                {"values": get_specific_value(all_values, all_ids, 'target-action-noise-stddev-slider-hyperparam', 'none', agent)}
            
            # Add target action clip
            sweep_config["parameters"][agent]["parameters"][f"{agent}_target_action_clip"] = \
                {"values": get_specific_value(all_values, all_ids, 'target-action-noise-clip-slider-hyperparam', 'none', agent)}
            
            # Add actor update delay
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_update_delay"] = \
                {"values": get_specific_value(all_values, all_ids, 'actor-update-delay-slider-hyperparam', 'none', agent)}


    if agent == "HER_DDPG":
        sweep_config["parameters"][agent]["parameters"] = {}

        # actor learning rate
        value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'actor', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_learning_rate"] = config
        
        # critic learning rate
        value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'critic', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_learning_rate"] = config
        
        # goal strategy
        sweep_config["parameters"][agent]["parameters"][f"{agent}_goal_strategy"] = \
            {"values": get_specific_value(all_values, all_ids, 'goal-strategy-hyperparam', 'none', agent)}

        # number of goals
        for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_goal_strategy"]['values']:
            if value == 'future':
                value_range = get_specific_value(all_values, all_ids, 'future-goals-hyperparam', 'none', agent)
                if value_range[0] == value_range[1]:
                    config = {"value": value_range[0]}
                else:
                    config = {"min": value_range[0], "max": value_range[1]}

            sweep_config["parameters"][agent]["parameters"][f"{agent}_num_goals"] = config
        
        # goal tolerance
        value_range = get_specific_value(all_values, all_ids, 'goal-tolerance-hyperparam', 'none', agent)
        config = {"values": value_range}

        sweep_config["parameters"][agent]["parameters"][f"{agent}_goal_tolerance"] = config

        # discount
        value_range = get_specific_value(all_values, all_ids, 'discount-slider', 'none', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_discount"] = config
        
        # tau
        value_range = get_specific_value(all_values, all_ids, 'tau-hyperparam', 'none', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_tau"] = config
        
        # epsilon
        value_range = get_specific_value(all_values, all_ids, 'epsilon-greedy-hyperparam', 'none', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_epsilon_greedy"] = config

        # normalize input options
        value_range = get_specific_value(all_values, all_ids, 'norm-clip-value-hyperparam', 'none', agent)
        config = {"values": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_normalizer_clip"] = config

        # Get Device
        value_range = get_specific_value(all_values, all_ids, 'device', 'none', agent)
        sweep_config["parameters"][agent]["parameters"][f"{agent}_device"] = {"value": value_range}
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_normalizer_clip"] = config

        # actor cnn layers
        value_range = get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_cnn_layers"] = config

        # actor num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'actor', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_layers"] = config

        # actor activation
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"] = \
            {"values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'actor', agent)}
        #DEBUG
        # print(f'DDPG actor activation set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"]}')

        # actor hidden layer kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent)}
        #DEBUG
        # print(f'DDPG actor kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"]}')

        # actor hidden layer kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent)}

        # actor optimizer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"] = \
            {"values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'actor', agent)}
        #DEBUG
        print(f'DDPG actor optimizer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"]}')

        # Actor optimizer options
        for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"]['values']:
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer_{value}_options"] = {'parameters': {}}
            config = {}
            if value == 'Adam':
                value_range = get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

            elif value == 'Adagrad':
                value_range = get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', 'actor', agent)
                config[f'{value}_lr_decay'] = {"values": value_range}

            elif value == 'RMSprop':
                value_range = get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', 'actor', agent)
                config[f'{value}_momentum'] = {"values": value_range}

            elif value == 'SGD':
                value_range = get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', 'actor', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', 'actor', agent)
                config[f'{value}_momentum'] = {"values": value_range}
                
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer_{value}_options"]['parameters'] = config
                
        # actor normalize layers
        sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_normalize_layers"] = \
            {"values": get_specific_value(all_values, all_ids, 'normalize-layers-hyperparam', 'actor', agent)}

        # actor clamp output
        # sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_clamp_output"] = \
        #     {"values": get_specific_value(all_values, all_ids, 'clamp-value-hyperparam', 'actor', agent)}

        # critic cnn layers
        value_range = get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'critic', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_num_cnn_layers"] = config
        #DEBUG
        # print(f'DDPG critic cnn layers set to {config}')


        # critic state num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-state', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_state_num_layers"] = config
        #DEBUG
        # print(f'DDPG critic state num layers set to {config}')

        # critic merged num layers
        value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-merged', agent)
        if value_range[0] == value_range[1]:
            config = {"value": value_range[0]}
        else:
            config = {"min": value_range[0], "max": value_range[1]}           
        
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_merged_num_layers"] = config
        #DEBUG
        # print(f'DDPG critic merged num layers set to {config}')

        # critic activation
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_activation"] = \
            {"values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'critic', agent)}
        #DEBUG
        # print(f'DDPG critic activation set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_activation"]}')

        # critic hidden kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent)}
        #DEBUG
        # print(f'DDPG critic kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"]}')

        # critic output kernel initializer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_initializer"] = \
            {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent)}

        # critic optimizer
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"] = \
            {"values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'critic', agent)}
        #DEBUG
        # print(f'DDPG critic optimizer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"]}')

        # Critic optimizer options
        for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer"]['values']:
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer_{value}_options"] = {'parameters': {}}
            config = {}
            if value == 'Adam':
                value_range = get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

            elif value == 'Adagrad':
                value_range = get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', 'critic', agent)
                config[f'{value}_lr_decay'] = {"values": value_range}

            elif value == 'RMSprop':
                value_range = get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', 'critic', agent)
                config[f'{value}_momentum'] = {"values": value_range}

            elif value == 'SGD':
                value_range = get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', 'critic', agent)
                config[f'{value}_weight_decay'] = {"values": value_range}

                value_range = get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', 'critic', agent)
                config[f'{value}_momentum'] = {"values": value_range}
                
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_optimizer_{value}_options"]['parameters'] = config

        # critic normalize layers
        sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_normalize_layers"] = \
            {"values": get_specific_value(all_values, all_ids, 'normalize-layers-hyperparam', 'critic', agent)}
        
        
        # replay buffer size
        sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer_size"] = \
            {"values": get_specific_value(all_values, all_ids, 'buffer-size-hyperparam', 'none', agent)}
        #DEBUG
        # print(f'DDPG replay buffer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer"]}')

        # batch size
        sweep_config["parameters"][agent]["parameters"][f"{agent}_batch_size"] = \
            {"values": get_specific_value(all_values, all_ids, 'batch-size-hyperparam', 'none', agent)}
        #DEBUG
        # print(f'DDPG batch size set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_batch_size"]}')

        # noise
        sweep_config["parameters"][agent]["parameters"][f"{agent}_noise"] = \
            {"values": get_specific_value(all_values, all_ids, 'noise-function-hyperparam', 'none', agent)}
        #DEBUG
        # print(f'DDPG noise set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_noise"]}')

        # noise parameter options
        for noise in get_specific_value(all_values, all_ids, 'noise-function-hyperparam', 'none', agent):
            # Initialize the dictionary for the agent if it doesn't exist
            if f"{agent}_noise_{noise}" not in sweep_config["parameters"][agent]["parameters"]:
                sweep_config["parameters"][agent]["parameters"][f"{agent}_noise_{noise}"]={"parameters":{}}
            
            # initialize empty config dictionary for parameters
            config = {}
            
            if noise == "Ornstein-Uhlenbeck":
                # mean
                value_range = get_specific_value(all_values, all_ids, 'ou-mean-hyperparam', 'none', agent)
                config["mean"] = {"values": value_range}
                
                # theta
                value_range = get_specific_value(all_values, all_ids, 'ou-theta-hyperparam', 'none', agent)
                config["theta"] = {"values": value_range}

                # sigma
                value_range = get_specific_value(all_values, all_ids, 'ou-sigma-hyperparam', 'none', agent)
                config["sigma"] = {"values": value_range}
                
            elif noise == "Normal":
                # mean
                value_range = get_specific_value(all_values, all_ids, 'normal-mean-hyperparam', 'none', agent)
                config["mean"] = {"values": value_range}

                # stddev
                value_range = get_specific_value(all_values, all_ids, 'normal-stddev-hyperparam', 'none', agent)
                config["stddev"] = {"values": value_range}
            
            elif noise == "Uniform":
                # minval
                value_range = get_specific_value(all_values, all_ids, 'uniform-min-hyperparam', 'none', agent)
                config["minval"] = {"values": value_range}

                # maxval
                value_range = get_specific_value(all_values, all_ids, 'uniform-max-hyperparam', 'none', agent)
                config["maxval"] = {"values": value_range}

            sweep_config["parameters"][agent]["parameters"][f"{agent}_noise_{noise}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG noise set to {config}')

        # kernel options       
        # actor hidden kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent):
            if kernel != 'default':
                if f"{agent}_actor_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-hidden', agent)
                    config["value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-hidden', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG actor kernel set to {config}')

        # actor output kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent):
            if kernel != 'default':
                if f"{agent}_actor_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-output', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]["parameters"] = config

        # critic hidden kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent):
            if kernel != 'default':
                if f"{agent}_critic_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-hidden', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-hidden', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG critic kernel set to {config}')

        # critic output kernel options
        for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent):
            if kernel != "default":
                if f"{agent}_critic_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_value"] = {"values": value_range}
    
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_scale"] = {"values": value_range}

                    # mode
                    config[f"{kernel}_mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-output', agent)}

                    # distribution
                    config[f"{kernel}_distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-output', agent)
                    config[f"{kernel}_mode"] = {"values": values}

                    
                else:
                    if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG critic kernel set to {config}')

        # CNN layer params
        # Actor CNN layers
        for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_{agent}"] = {"parameters":{}}
            config = {}
            config[f"{agent}_actor_cnn_layer_{i}_types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'actor', agent, i)}

            # loop through each type in CNN layer and get the parameters to add to the sweep config
            for value in config[f"{agent}_actor_cnn_layer_{i}_types"]["values"]:
                if value == "conv":
                    config[f"{agent}_actor_cnn_layer_{i}_conv_filters"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'actor', agent, i)
                    }

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                    config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {
                        "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'actor', agent, i)
                    }

                    if config[f"{agent}_actor_cnn_layer_{i}_conv_padding"]["value"] == 'custom':
                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {"value": value_range[0]}
                        else:
                            config[f"{agent}_actor_cnn_layer_{i}_conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                        
                        # val_config["conv_padding"]["parameters"] = pad_config
                    
                    config[f"{agent}_actor_cnn_layer_{i}_conv_bias"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'actor', agent, i)
                    }
                
                if value == "pool":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'actor', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_actor_cnn_layer_{i}_pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                if value == "dropout":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'actor', agent, i)
                    config[f"{agent}_actor_cnn_layer_{i}_dropout_prob"] = {"values": value_range}

                # config["parameters"] = val_config

            sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_{agent}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG actor CNN layers set to {config}')

        # Critic CNN layers
        for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'critic', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_{agent}"] = {"parameters":{}}
            config = {}
            config[f"{agent}_critic_cnn_layer_{i}_types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'critic', agent, i)}

            # loop through each type in CNN layer and get the parameters to add to the sweep config
            for value in config[f"{agent}_critic_cnn_layer_{i}_types"]["values"]:
                if value == "conv":
                    config[f"{agent}_critic_cnn_layer_{i}_conv_filters"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'critic', agent, i)
                    }

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                    config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {
                        "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'critic', agent, i)
                    }

                    if config[f"{agent}_critic_cnn_layer_{i}_conv_padding"]["value"] == 'custom':
                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {"value": value_range[0]}
                        else:
                            config[f"{agent}_critic_cnn_layer_{i}_conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                    
                    config[f"{agent}_critic_cnn_layer_{i}_conv_bias"] = {
                        "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'critic', agent, i)
                    }
                
                if value == "pool":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_kernel_size"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'critic', agent, i)
                    if value_range[0] == value_range[1]:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_strides"] = {"value": value_range[0]}
                    else:
                        config[f"{agent}_critic_cnn_layer_{i}_pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                if value == "dropout":

                    value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'critic', agent, i)
                    config[f"{agent}_critic_cnn_layer_{i}_dropout_prob"] = {"values": value_range}

            sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_{agent}"]["parameters"] = config
        #DEBUG
        # print(f'DDPG critic CNN layers set to {config}')

        # layer units
        # actor layer units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'actor', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"actor_units_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'actor', agent)   
            }
        # critic state units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-state', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_units_state_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'critic-state', agent)
            }
        # critic merged units
        for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'critic-merged', agent)[1] + 1):
            sweep_config["parameters"][agent]["parameters"][f"critic_units_merged_layer_{i}_{agent}"] = {
                "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'critic-merged', agent)
            }

        # Add save dir
        sweep_config["parameters"][agent]["parameters"][f"{agent}_save_dir"] = \
            {"value": get_specific_value(all_values, all_ids, 'save-dir', 'none', agent)}

    if agent == "PPO":
        # sweep_config["parameters"][agent]["parameters"] = {}

        # Policy learning rate constant
        sweep_config = format_wandb_config_param(sweep_config, "learning_rate_constant", all_values, all_ids, 'learning-rate-const-hyperparam', 'policy', agent)

        # Policy learning rate exponent
        sweep_config = format_wandb_config_param(sweep_config, "learning_rate_exponent", all_values, all_ids, 'learning-rate-exp-hyperparam', 'policy', agent)
        
        # Value learning rate constant
        sweep_config = format_wandb_config_param(sweep_config, "learning_rate_constant", all_values, all_ids, 'learning-rate-const-hyperparam', 'value', agent)

        # Value learning rate exponent
        sweep_config = format_wandb_config_param(sweep_config, "learning_rate_exponent", all_values, all_ids, 'learning-rate-exp-hyperparam', 'value', agent)

        # Distribution
        sweep_config = format_wandb_config_param(sweep_config, "distribution", all_values, all_ids, "distribution-hyperparam", 'policy', agent)

        # Discount
        sweep_config = format_wandb_config_param(sweep_config, "discount", all_values, all_ids, 'discount-slider', 'none', agent)

        # Reward clip
        sweep_config = format_wandb_config_param(sweep_config, "reward_clip", all_values, all_ids, "reward-clip-hyperparam", 'none', agent)

        # Advantage Coeff
        sweep_config = format_wandb_config_param(sweep_config, "advantage", all_values, all_ids, 'advantage-coeff-hyperparam', 'none', agent)

        # Model type
        sweep_config = format_wandb_config_param(sweep_config, "model_type", all_values, all_ids, 'model-type-hyperparam', 'policy', agent)
        
        # Policy Surrogate Clip
        sweep_config = format_wandb_config_param(sweep_config, "clip_range", all_values, all_ids, 'surrogate-clip-hyperparam', 'policy', agent)

        # Policy Grad Clip
        sweep_config = format_wandb_config_param(sweep_config, "grad_clip", all_values, all_ids, 'grad-clip-hyperparam', 'policy', agent)

        # Entropy Coeff
        sweep_config = format_wandb_config_param(sweep_config, "entropy", all_values, all_ids, 'entropy-coeff-hyperparam', 'none', agent)

        # KL Coeff
        sweep_config = format_wandb_config_param(sweep_config, "kl", all_values, all_ids, 'kl-coeff-hyperparam', 'none', agent)

        # Normalize Advantage
        sweep_config = format_wandb_config_param(sweep_config, "normalize_advantage", all_values, all_ids, 'normalize-advantage-hyperparam', 'none', agent)

        # Normalize Values
        sweep_config = format_wandb_config_param(sweep_config, "normalize_values", all_values, all_ids, 'normalize-values-hyperparam', 'value', agent)

        # Normalize Value Clip
        sweep_config = format_wandb_config_param(sweep_config, "normalize_values_clip", all_values, all_ids, 'norm-values-clip-hyperparam', 'value', agent)

        # Get Device
        sweep_config = format_wandb_config_param(sweep_config, "device", all_values, all_ids, 'device', 'none', agent)

        # Policy num layers
        # sweep_config = format_wandb_config_param(sweep_config, "num_layers", all_values, all_ids, 'hidden-layers-slider', 'policy', agent, is_range=True)

        # Policy layers
        sweep_config = format_wandb_model_layers(sweep_config, all_values, all_ids, all_indexed_values, all_indexed_ids, 'policy', agent)
        
        # policy output layer kernel initializers
        sweep_config = format_wandb_kernel(sweep_config, all_indexed_values, all_indexed_ids, 'policy', agent, 'output')

        # Policy activation
        # sweep_config = format_wandb_config_param(sweep_config, "policy_activation", all_values, all_ids, 'activation-function-hyperparam', 'policy', agent)

        # Policy optimizer
        sweep_config = format_wandb_config_param(sweep_config, "optimizer", all_values, all_ids, 'optimizer-hyperparam', 'policy', agent)

        # Policy optimizer options
        sweep_config = format_wandb_optimizer_options(sweep_config, "optimizer", all_values, all_ids, "policy", agent)

        # Policy learning rate scheduler
        sweep_config = format_wandb_lr_scheduler(sweep_config, all_values, all_ids, "policy", agent)

        # Value num layers
        # sweep_config = format_wandb_config_param(sweep_config, "num_layers", all_values, all_ids, 'hidden-layers-slider', 'value', agent, is_range=True)

        # Value activation
        # sweep_config = format_wandb_config_param(sweep_config, "value_activation", all_values, all_ids, 'activation-function-hyperparam', 'value', agent)

        # Value layers
        sweep_config = format_wandb_model_layers(sweep_config, all_values, all_ids, all_indexed_values, all_indexed_ids, 'value', agent)

        # Value Loss Coefficient
        sweep_config = format_wandb_config_param(sweep_config, "loss_coeff", all_values, all_ids, 'loss-coeff-hyperparam', 'value', agent)

        # Value Surrogate Clip
        sweep_config = format_wandb_config_param(sweep_config, "clip_range", all_values, all_ids, 'surrogate-clip-hyperparam', 'value', agent)

        # Value Grad Clip
        sweep_config = format_wandb_config_param(sweep_config, "grad_clip", all_values, all_ids, 'grad-clip-hyperparam', 'value', agent)

        # value output layer kernel initializers
        sweep_config = format_wandb_kernel(sweep_config, all_indexed_values, all_indexed_ids, 'value', agent, 'output')

        # Value optimizer
        sweep_config = format_wandb_config_param(sweep_config, "optimizer", all_values, all_ids, 'optimizer-hyperparam', 'value', agent)

        # Value optimizer options
        sweep_config = format_wandb_optimizer_options(sweep_config, "optimizer", all_values, all_ids, "value", agent)

        # Value learning rate scheduler
        sweep_config = format_wandb_lr_scheduler(sweep_config, all_values, all_ids, "value", agent)

        # Layer units
        # Policy layer units
        # sweep_config = format_wandb_layer_units(sweep_config, 'policy_units_layer', all_values, all_ids, 'hidden-layers-slider', 'policy', agent)

        # Value layer units
        # sweep_config = format_wandb_layer_units(sweep_config, 'value_units_layer', all_values, all_ids, 'hidden-layers-slider', 'value', agent)

        # Add save dir
        sweep_config = format_wandb_config_param(sweep_config, 'save_dir', all_values, all_ids, 'save-dir', 'none', agent)
        
        # Add training parameters
        # Timesteps
        sweep_config = format_wandb_config_param(sweep_config, 'num_timesteps', all_values, all_ids, 'num-timesteps', 'none', agent)
        # Trajectory length
        sweep_config = format_wandb_config_param(sweep_config, 'trajectory_length', all_values, all_ids, 'trajectory-length', 'none', agent)
        # Batch size
        sweep_config = format_wandb_config_param(sweep_config, "batch_size", all_values, all_ids, 'batch-size', 'none', agent)
        # Learning epochs
        sweep_config = format_wandb_config_param(sweep_config, "learning_epochs", all_values, all_ids, 'learning-epochs', 'none', agent)
        # Num evns
        sweep_config = format_wandb_config_param(sweep_config, "num_envs", all_values, all_ids, 'num-envs', 'none', agent)
        # Seed
        sweep_config = format_wandb_config_param(sweep_config, 'seed', all_values, all_ids, 'seed', 'none', agent)

                                
    # elif agent == "Reinforce" or agent == "ActorCritic":
    #     sweep_config["parameters"][agent]["parameters"] = {
    #         f"{agent}_learning_rate": {
    #             "max": 10**(get_specific_value(all_values, all_ids, 'learning-rate-slider', 'none', agent)[1]),
    #             "min": 10**(get_specific_value(all_values, all_ids, 'learning-rate-slider', 'none', agent)[0]),
    #             },
    #         f"{agent}_discount": {
    #             "max": get_specific_value(all_values, all_ids, 'discount-slider', 'none', agent)[1],
    #             "min": get_specific_value(all_values, all_ids, 'discount-slider', 'none', agent)[0],
    #             },
    #         f"{agent}_policy_num_layers": {
    #             "max": get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'policy', agent)[1],
    #             "min": get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'policy', agent)[0],
    #             },
    #         f"{agent}_policy_activation": {
    #             "values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'policy', agent)
    #             },
    #         f"{agent}_policy_optimizer": {
    #             "values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'policy', agent)
    #             },
    #         f"{agent}_value_num_layers": {
    #             "max": get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'value', agent)[1],
    #             "min": get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'value', agent)[0],
    #             },
    #         f"{agent}_value_activation": {
    #             "values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'value', agent)
    #             },
    #         f"{agent}_value_optimizer": {
    #             "values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'value', agent)
    #             },
    #         },

    #     if agent == "ActorCritic":
    #         sweep_config["parameters"][agent]["parameters"]["policy_trace_decay"] = {
    #             "max": get_specific_value(all_values, all_ids, 'trace-decay-hyperparam', 'policy', agent)[1],
    #             "min": get_specific_value(all_values, all_ids, 'trace-decay-hyperparam', 'policy', agent)[0],
    #         }
    #         sweep_config["parameters"][agent]["parameters"]["value_trace_decay"] = {
    #             "max": get_specific_value(all_values, all_ids, 'trace-decay-hyperparam', 'value', agent)[1],
    #             "min": get_specific_value(all_values, all_ids, 'trace-decay-hyperparam', 'value', agent)[0],
    #         }

    #     # add kernel options to sweep config
    #     # policy kernel options
    #     for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'policy', agent):
    #         if f"{agent}_policy_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"] = {}
    #         if kernel == "constant":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"]["parameters"] = {
    #                 "value": {
    #                     "max": get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'policy', agent)[0],
    #                 },
    #             }
    #         elif kernel == "variance_scaling":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"]["parameters"] = {
    #                 "scale": {
    #                     "max": get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'policy', agent)[0],
    #                 },
    #                 "mode": {
    #                     "values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'policy', agent),
    #                 },
    #                 "distribution": {
    #                     "values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'policy', agent),
    #                 },
    #             }
    #         elif kernel == "random_uniform":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"]["parameters"] = {
    #                 "maxval": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'policy', agent)[0],
    #                 },
    #                 "minval": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'policy', agent)[0],
    #                 },
    #             }
    #         elif kernel == "random_normal":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"]["parameters"] = {
    #                 "mean": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'policy', agent)[0],
    #                 },
    #                 "stddev": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'policy', agent)[0],
    #                 },
    #             }
    #         elif kernel == "truncated_normal":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_policy_kernel_{kernel}"]["parameters"] = {
    #                 "mean": {
    #                     "max": get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'policy', agent)[0],
    #                 },
    #                 "stddev": {
    #                     "max": get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'policy', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'policy', agent)[0],
    #                 },
    #             }
    #         else:
    #             if kernel not in ["constant", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal", "zeros", "ones", \
    #                 "random_uniform", "random_normal", "truncated_normal", "variance_scaling"]:
    #                 raise ValueError(f"Unknown kernel: {kernel}")
            
    #     # value kernel options
    #     for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'value', agent):
    #         if f"{agent}_value_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"] = {}
    #         if kernel == "constant":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"]["parameters"] = {
    #                 "value": {
    #                     "max": get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'value', agent)[0],
    #                 },
    #             }
    #         elif kernel == "variance_scaling":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"]["parameters"] = {
    #                 "scale": {
    #                     "max": get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'value', agent)[0],
    #                 },
    #                 "mode": {
    #                     "values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'value', agent),
    #                 },
    #                 "distribution": {
    #                     "values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'value', agent),
    #                 },
    #             }
    #         elif kernel == "random_uniform":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"]["parameters"] = {
    #                 "maxval": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'value', agent)[0],
    #                 },
    #                 "minval": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'value', agent)[0],
    #                 },
    #             }
    #         elif kernel == "random_normal":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"]["parameters"] = {
    #                 "mean": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'value', agent)[0],
    #                 },
    #                 "stddev": {
    #                     "max": get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'value', agent)[0],
    #                 },
    #             }
    #         elif kernel == "truncated-normal":
    #             sweep_config["parameters"][agent]["parameters"][f"{agent}_value_kernel_{kernel}"]["parameters"] = {
    #                 "mean": {
    #                     "max": get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'value', agent)[0],
    #                 },
    #                 "stddev": {
    #                     "max": get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'value', agent)[1],
    #                     "min": get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'value', agent)[0],
    #                 },
    #             }
    #         else:
    #             raise ValueError(f"Unknown kernel: {kernel}")
        
    #     # add units per layer to sweep config
    #     for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'policy', agent)[1] + 1):
    #         sweep_config["parameters"][agent]["parameters"][f"policy_units_layer_{i}_{agent}"] = {
    #             "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'policy', agent) 
    #         }
    #     for i in range(1, get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'value', agent)[1] + 1):
    #         sweep_config["parameters"][agent]["parameters"][f"value_units_layer_{i}_{agent}"] = {
    #             "values": get_specific_value(all_values, all_ids, f'layer-{i}-units-slider', 'value', agent)
    #         }
        
    ##DEBUG
    print(f'Sweep Config: {sweep_config}')
        
    return sweep_config

def update_heatmap(data):
    if data is not None:

        #DEBUG
        print(f'update heatmap data: {data}')
        # Convert the data to a numpy array
        data_np = np.array(data['matrix_data']['data'])

        # Format hyperparameter strings (remove model type if appears twice)
        columns = [wandb_support.parse_parameter_name(col) for col in data['matrix_data']['columns']]
        index = [wandb_support.parse_parameter_name(col) for col in data['matrix_data']['index']]

        # Create a mask to set the upper triangular part to NaN
        mask = np.triu(np.ones_like(data_np, dtype=bool), k=1)
        matrix_data_masked = np.where(mask, np.nan, data_np)

        # create stacked bar charts for legend of show hyperparameter bins
        stacked_bar_graph = go.Figure()

        for hp, hp_bin_ranges in data['bin_ranges'].items():
            bin_labels = [f"Bin {i}" for i in range(len(hp_bin_ranges)-1)]
            bin_ranges = [hp_bin_ranges[i+1] - hp_bin_ranges[i] for i in range(len(hp_bin_ranges)-1)]
        
            for i in range(len(bin_labels)):
                stacked_bar_graph.add_trace(go.Bar(
                    y=[hp],
                    x=[hp_bin_ranges[i]],
                    name=f"{hp} - {bin_labels[i]}",
                    orientation="h",
                    text=[bin_labels[i]],
                    textposition="inside",
                    insidetextanchor="middle",
                    hoverinfo="text",
                    hovertext=[f"{hp_bin_ranges[i]} - {hp_bin_ranges[i+1]}"]
                ))
        # create the layout of the stacked bar graph
        stacked_bar_graph.update_layout(
            title="Hyperparameter Bin Ranges",
            barmode="stack",
            yaxis=dict(title="Hyperparameter"),
            xaxis_title="Range",
            showlegend=False,
        )

        # Create the heatmap figure using Plotly
        heatmap = px.imshow(
            img=matrix_data_masked,
            labels={'x':'Hyperparameters','y':'Hyperparameters'},
            title='Co-occurrence Matrix',
            x=columns,
            y=index,
            color_continuous_scale='RdBu_r',
            text_auto=True,
            aspect="auto",
        )

        return heatmap, stacked_bar_graph
    else:
        return None
    
def render_heatmap(page):
    return html.Div([
        html.Label('Bins', style={'text-decoration': 'underline'}),
        dcc.Slider(
            id={'type':'bin-slider', 'page':page},
            min=1,
            max=10,
            value=5,
            marks={i: str(i) for i in range(1, 11)},
            step=1,
        ),
        html.Div(id={'type':'legend-container', 'page':page}),
        html.Label('Reward Threshold', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={'type':'reward-threshold', 'page':page},
            type='number',
            value=0,
            style={'display':'inline-block'}
        ),
        dcc.Checklist(
            id={'type':'z-score-checkbox', 'page':page},
            options=[
                {'label': ' Display Z-Scores', 'value': 'zscore'},
            ],
            value=[],
            style={'display':'inline-block'}
        ),
        html.Div(id={'type':'heatmap-container', 'page':page}),
        html.Div(
            id={'type':'heatmap-placeholder', 'page':page},
            children=[
                html.P('The co-occurrence graph will load once enough data has been retrieved.'),
                html.Img(src='path/to/placeholder-image.png', alt='Placeholder Image')
            ],
            style={'display': 'none'}
        )
    ])

def create_sweep_options():
    return html.Div(
            [
                html.H4('Search Method'),
                dcc.RadioItems(
                    id='search-type',
                    options=[
                        {'label': 'Random Search', 'value': 'random'},
                        {'label': 'Grid Search', 'value': 'grid'},
                        {'label': 'Bayesian Search', 'value': 'bayes'},
                    ],
                    value='bayes',
                ),
                html.Div(
                    [
                        html.H6('Set Goal', style={'textAlign': 'left'}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.RadioItems(
                                            id='goal-type',
                                            options=[
                                                {'label': 'Maximize', 'value': 'maximize'},
                                                {'label': 'Minimize', 'value': 'minimize'},
                                            ],
                                        ),
                                    ],
                                    style={'display': 'flex'}
                                ),
                                html.Div(
                                    [
                                        dcc.RadioItems(
                                            id='goal-metric',
                                            options=[
                                                {'label': 'Episode Reward', 'value': 'episode_reward'},
                                                {'label': 'Value Loss', 'value': 'value_loss'},
                                                {'label': 'Policy Loss', 'value': 'policy_loss'},
                                            ],
                                        ),
                                    ],
                                    style={'display': 'flex'}
                                ),
                            ],
                            style={'display': 'flex'}
                        ),
                    ]
                ),
                dcc.Input(
                    id='sweep-name',
                    type='text',
                    placeholder='Sweep Name',
                ),
                dcc.Input(
                    id='num-sweeps',
                    type='number',
                    placeholder='Number of Sweeps',
                ),
            ]
        )

    
def create_agent_sweep_options(agent_type):
    """Returns Div of agent sweep options dependent on agent type

    Args:
        agent_type (str): agent selected in 'Agent Configuration' dropdown

    Returns:
        Div: Div object containing sweep options for agent_type selected
    """
    if agent_type == 'PPO':
        return create_ppo_sweep_options(agent_type)
    else:
        pass

def create_ppo_sweep_options(agent_type):
    """Returns Div of ppo agent sweep options

    Returns:
        Div: Div object of ppo agent sweep options
    """
    return html.Div(
                id='ppo-sweep-options',
                children=[
                    dcc.Input(
                        id={
                            'type':'num-timesteps',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        placeholder='Total Timesteps',
                    ),
                    dcc.Dropdown(
                        id={
                            'type':'trajectory-length',
                            'model':'none',
                            'agent': agent_type,
                        },
                        options=[{'label':str(i), 'value':i} for i in [100,200,300,400,500,1000,2000,5000,10000]],
                        placeholder='Trajectory Length',
                    ),
                    dcc.Dropdown(
                        id={
                            'type':'batch-size',
                            'model':'none',
                            'agent': agent_type,
                        },
                        options=[{'label':str(i), 'value':i} for i in [32,64,128,256,512,1024,2048,5096,10192]],
                        placeholder='Learning Batch Sizes',
                    ),
                    dcc.Dropdown(
                        id={
                            'type':'learning-epochs',
                            'model':'none',
                            'agent': agent_type,
                        },
                        options=[{'label':str(i), 'value':i} for i in [1,2,4,8,12,16,20]],
                        placeholder='Learning Updates per Epoch',
                    ),
                    dcc.Dropdown(
                        id={
                            'type':'num-envs',
                            'model':'none',
                            'agent': agent_type,
                        },
                        options=[{'label':str(i), 'value':i} for i in [1,2,4,8,12,16,20]],
                        placeholder='Number of Envs',
                    ),
                    dcc.Input(
                        id={
                            'type':'seed',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        placeholder='Random Seed',
                    ),
                ]
            ),

## TUNE FUNCTIONS ##
def create_tune_config(method, project, sweep_name, metric_name, metric_goal,
                       env_library, env, env_params, env_wrappers, agent,
                       all_values, all_ids, all_indexed_values, all_indexed_ids):
    # For Ray Tune, the "config" is usually just the hyperparameter search space.
    tune_config = {
        "env_library": env_library,
        "env_id": env,
    }
    # Add each environment parameter as a fixed value.
    for param, value in env_params.items():
        tune_config[f"env_{param}"] = value
    # For lists (like env_wrappers), you can either use the raw list or a search space (e.g., tune.choice)
    tune_config["env_wrappers"] = env_wrappers  # or: tune.choice(env_wrappers)
    tune_config["model_type"] = agent

    # Add agent-specific hyperparameters. Here we assume agent == "PPO".
    if agent == "PPO":
        # For each parameter, we use a helper that returns a Tune search space object.
        tune_config["policy_learning_rate_constant"] = format_tune_config_param(
            "learning_rate_constant", all_values, all_ids, "learning-rate-const-hyperparam", "policy", agent)
        tune_config["policy_learning_rate_exponent"] = format_tune_config_param(
            "learning_rate_exponent", all_values, all_ids, "learning-rate-exp-hyperparam", "policy", agent)
        tune_config["value_learning_rate_constant"] = format_tune_config_param(
            "learning_rate_constant", all_values, all_ids, "learning-rate-const-hyperparam", "value", agent)
        tune_config["value_learning_rate_exponent"] = format_tune_config_param(
            "learning_rate_exponent", all_values, all_ids, "learning-rate-exp-hyperparam", "value", agent)
        tune_config["distribution"] = format_tune_config_param(
            "distribution", all_values, all_ids, "distribution-hyperparam", "policy", agent)
        tune_config["discount"] = format_tune_config_param(
            "discount", all_values, all_ids, "discount-slider", "none", agent)
        tune_config["reward_clip"] = format_tune_config_param(
            "reward_clip", all_values, all_ids, "reward-clip-hyperparam", "none", agent)
        tune_config["advantage"] = format_tune_config_param(
            "advantage", all_values, all_ids, "advantage-coeff-hyperparam", "none", agent)
        tune_config["model_type_param"] = format_tune_config_param(
            "model_type", all_values, all_ids, "model-type-hyperparam", "policy", agent)
        tune_config["policy_clip_range"] = format_tune_config_param(
            "clip_range", all_values, all_ids, "surrogate-clip-hyperparam", "policy", agent)
        tune_config["policy_grad_clip"] = format_tune_config_param(
            "grad_clip", all_values, all_ids, "grad-clip-hyperparam", "policy", agent)
        tune_config["entropy"] = format_tune_config_param(
            "entropy", all_values, all_ids, "entropy-coeff-hyperparam", "none", agent)
        tune_config["kl"] = format_tune_config_param(
            "kl", all_values, all_ids, "kl-coeff-hyperparam", "none", agent)
        tune_config["normalize_advantage"] = format_tune_config_param(
            "normalize_advantage", all_values, all_ids, "normalize-advantage-hyperparam", "none", agent)
        tune_config["normalize_values"] = format_tune_config_param(
            "normalize_values", all_values, all_ids, "normalize-values-hyperparam", "value", agent)
        tune_config["normalize_values_clip"] = format_tune_config_param(
            "normalize_values_clip", all_values, all_ids, "norm-values-clip-hyperparam", "value", agent)
        tune_config["device"] = format_tune_config_param(
            "device", all_values, all_ids, "device", "none", agent)

        # Model architecture parameters.
        tune_config["policy_layers"] = format_tune_model_layers(
            all_values, all_ids, all_indexed_values, all_indexed_ids, "policy", agent)
        tune_config["policy_kernel_output"] = format_tune_kernel(
            all_indexed_values, all_indexed_ids, "policy", agent, "output")
        tune_config["policy_optimizer"] = format_tune_config_param(
            "optimizer", all_values, all_ids, "optimizer-hyperparam", "policy", agent)
        tune_config["policy_optimizer_options"] = format_tune_optimizer_options(
            "optimizer", all_values, all_ids, "policy", agent)
        tune_config["policy_lr_scheduler"] = format_tune_lr_scheduler(
            all_values, all_ids, "policy", agent)

        tune_config["value_layers"] = format_tune_model_layers(
            all_values, all_ids, all_indexed_values, all_indexed_ids, "value", agent)
        tune_config["loss_coeff"] = format_tune_config_param(
            "loss_coeff", all_values, all_ids, "loss-coeff-hyperparam", "value", agent)
        tune_config["value_clip_range"] = format_tune_config_param(
            "clip_range", all_values, all_ids, "surrogate-clip-hyperparam", "value", agent)
        tune_config["value_grad_clip"] = format_tune_config_param(
            "grad_clip", all_values, all_ids, "grad-clip-hyperparam", "value", agent)
        tune_config["value_kernel_output"] = format_tune_kernel(
            all_indexed_values, all_indexed_ids, "value", agent, "output")
        tune_config["value_optimizer"] = format_tune_config_param(
            "optimizer", all_values, all_ids, "optimizer-hyperparam", "value", agent)
        tune_config["value_optimizer_options"] = format_tune_optimizer_options(
            "optimizer", all_values, all_ids, "value", agent)
        tune_config["value_lr_scheduler"] = format_tune_lr_scheduler(
            all_values, all_ids, "value", agent)

        # Training parameters.
        tune_config["save_dir"] = format_tune_config_param(
            "save_dir", all_values, all_ids, "save-dir", "none", agent)
        tune_config["num_timesteps"] = format_tune_config_param(
            "num_timesteps", all_values, all_ids, "num-timesteps", "none", agent)
        tune_config["trajectory_length"] = format_tune_config_param(
            "trajectory_length", all_values, all_ids, "trajectory-length", "none", agent)
        tune_config["batch_size"] = format_tune_config_param(
            "batch_size", all_values, all_ids, "batch-size", "none", agent)
        tune_config["learning_epochs"] = format_tune_config_param(
            "learning_epochs", all_values, all_ids, "learning-epochs", "none", agent)
        tune_config["num_envs"] = format_tune_config_param(
            "num_envs", all_values, all_ids, "num-envs", "none", agent)
        tune_config["seed"] = format_tune_config_param(
            "seed", all_values, all_ids, "seed", "none", agent)
    return tune_config

def format_tune_config_param(config, param_name, all_values, all_ids, dash_id, model, agent, index=None, is_range=False):
    """
    For Ray Tune, returns a fixed value or search space object (e.g., tune.uniform or tune.choice)
    for a given parameter.
    """
    if index:
        value = get_specific_value_id(all_values, all_ids, dash_id, model, agent, index)
    else:
        value = get_specific_value(all_values, all_ids, dash_id, model, agent)
    
    if is_range:
        # If the two endpoints are equal, use the fixed value;
        # otherwise, use a uniform search space between min and max.
        if value[0] == value[1]:
            param = value[0]
        else:
            param = tune.uniform(value[0], value[1])
    else:
        # If value is a list, assume we want a choice over them.
        if isinstance(value, list):
            param = tune.choice(value)
        else:
            param = value

    if index:
        config[f"{agent}_{model}_{index}_{param_name}"] = param
    else:
        config[f"{agent}_{model}_{param_name}"] = param

    return config

def format_tune_optimizer_options(config, param_name, all_values, all_ids, model, agent):
    # Assume that config[f"{agent}_{model}_{param_name}"] is a list of optimizer names.
    for value in config[f"{agent}_{model}_{param_name}"]:
        if value == 'Adam':
            config[f"{agent}_{model}_{param_name}_{value}_weight_decay"] = tune.choice(
                get_specific_value(all_values, all_ids, 'adam-weight-decay-hyperparam', model, agent)
            )
        elif value == 'Adagrad':
            config[f"{agent}_{model}_{param_name}_{value}_weight_decay"] = tune.choice(
                get_specific_value(all_values, all_ids, 'adagrad-weight-decay-hyperparam', model, agent)
            )
            config[f"{agent}_{model}_{param_name}_{value}_lr_decay"] = tune.choice(
                get_specific_value(all_values, all_ids, 'adagrad-lr-decay-hyperparam', model, agent)
            )
        elif value == 'RMSprop':
            config[f"{agent}_{model}_{param_name}_{value}_weight_decay"] = tune.choice(
                get_specific_value(all_values, all_ids, 'rmsprop-weight-decay-hyperparam', model, agent)
            )
            config[f"{agent}_{model}_{param_name}_{value}_momentum"] = tune.choice(
                get_specific_value(all_values, all_ids, 'rmsprop-momentum-hyperparam', model, agent)
            )
        elif value == 'SGD':
            config[f"{agent}_{model}_{param_name}_{value}_weight_decay"] = tune.choice(
                get_specific_value(all_values, all_ids, 'sgd-weight-decay-hyperparam', model, agent)
            )
            config[f"{agent}_{model}_{param_name}_{value}_momentum"] = tune.choice(
                get_specific_value(all_values, all_ids, 'sgd-momentum-hyperparam', model, agent)
            )
    return config

def format_tune_kernel(config, all_indexed_values, all_indexed_ids, model, agent, layer):
    if isinstance(layer, int):
        layer_types = get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-type-hyperparam', model, agent, layer)
        for layer_type in layer_types:
            if layer_type in ['dense', 'conv2d']:
                config = format_tune_config_param(config, "kernel", all_indexed_values, all_indexed_ids,
                                                    'kernel-function-hyperparam', model, agent, layer)
                config = format_tune_kernel_options(config, all_indexed_values, all_indexed_ids, model, agent, layer)
    else:
        # For output layers, use index 'output'
        config = format_tune_config_param(config, "kernel", all_indexed_values, all_indexed_ids,
                                            "kernel-function-hyperparam", model, agent, 'output')
        config = format_tune_kernel_options(config, all_indexed_values, all_indexed_ids, model, agent, 'output')
    return config

def format_tune_kernel_options(config, all_values, all_ids, model, agent, layer_num):
    # For each kernel option, update config with additional parameters.
    for kernel in get_specific_value_id(all_values, all_ids, 'kernel-function-hyperparam', model, agent, layer_num):
        if kernel != 'default':
            if kernel == "constant":
                config = format_tune_config_param(config, "kernel", all_values, all_ids,
                                                    'constant-value-hyperparam', model, agent, layer_num)
            elif kernel == "variance_scaling":
                config[f"{agent}_{model}_{layer_num}_{kernel}_scale"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'variance-scaling-scale-hyperparam', model, agent, layer_num)
                )
                config[f"{agent}_{model}_{layer_num}_{kernel}_mode"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'variance-scaling-mode-hyperparam', model, agent, layer_num)
                )
                config[f"{agent}_{model}_{layer_num}_{kernel}_distribution"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'variance-scaling-distribution-hyperparam', model, agent, layer_num)
                )
            elif kernel == "uniform":
                config[f"{agent}_{model}_{layer_num}_{kernel}_maxval"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'random-uniform-maxval-hyperparam', model, agent, layer_num)
                )
                config[f"{agent}_{model}_{layer_num}_{kernel}_minval"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'random-uniform-minval-hyperparam', model, agent, layer_num)
                )
            elif kernel == "normal":
                config[f"{agent}_{model}_{layer_num}_{kernel}_mean"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'random-normal-mean-hyperparam', model, agent, layer_num)
                )
                config[f"{agent}_{model}_{layer_num}_{kernel}_stddev"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'random-normal-stddev-hyperparam', model, agent, layer_num)
                )
            elif kernel == "truncated_normal":
                config[f"{agent}_{model}_{layer_num}_{kernel}_mean"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'truncated-normal-mean-hyperparam', model, agent, layer_num)
                )
                config[f"{agent}_{model}_{layer_num}_{kernel}_stddev"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'truncated-normal-stddev-hyperparam', model, agent, layer_num)
                )
            elif kernel == "xavier_uniform":
                config[f"{agent}_{kernel}_{layer_num}_{model}_gain"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'xavier-uniform-gain-hyperparam', model, agent, layer_num)
                )
            elif kernel == "xavier_normal":
                config[f"{agent}_{model}_{layer_num}_{kernel}_gain"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'xavier-normal-gain-hyperparam', model, agent, layer_num)
                )
            elif kernel == "kaiming_uniform":
                config[f"{agent}_{model}_{layer_num}_{kernel}_mode"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', model, agent, layer_num)
                )
            elif kernel == "kaiming_normal":
                config[f"{agent}_{model}_{layer_num}_{kernel}_mode"] = tune.choice(
                    get_specific_value_id(all_values, all_ids, 'kaiming-normal-mode-hyperparam', model, agent, layer_num)
                )
            else:
                if kernel not in ["default", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform",
                                  "kaiming_normal", "zeros", "ones", "uniform", "normal", "truncated_normal", "variance_scaling"]:
                    raise ValueError(f"Unknown kernel: {kernel}")
    return config

def format_tune_lr_scheduler(config, all_values, all_ids, model, agent):
    schedulers = get_specific_value(all_values, all_ids, 'lr-scheduler-hyperparam', model, agent)
    for scheduler in schedulers:
        config = format_tune_config_param(config, "scheduler", all_values, all_ids,
                                          'lr-scheduler-hyperparam', model, agent)
        config = format_tune_lr_scheduler_options(config, all_values, all_ids, model, agent, scheduler)
    return config

def format_tune_lr_scheduler_options(config, all_values, all_ids, model, agent, scheduler):
    if scheduler == 'step':
        config = format_tune_steplr_options(config, all_values, all_ids, model, agent)
    elif scheduler == 'exponential':
        config = format_tune_exponentiallr_options(config, all_values, all_ids, model, agent)
    elif scheduler == 'cosineannealing':
        config = format_tune_cosineannealinglr_options(config, all_values, all_ids, model, agent)
    return config

def format_tune_steplr_options(config, all_values, all_ids, model, agent):
    config = format_tune_config_param(config, "step_size", all_values, all_ids,
                                      'lr-step-size-hyperparam', model, agent)
    config = format_tune_config_param(config, "gamma", all_values, all_ids,
                                      'lr-gamma-hyperparam', model, agent)
    return config

def format_tune_exponentiallr_options(config, all_values, all_ids, model, agent):
    config = format_tune_config_param(config, "step_size", all_values, all_ids,
                                      'lr-gamma-hyperparam', model, agent)
    return config

def format_tune_cosineannealinglr_options(config, all_values, all_ids, model, agent):
    config = format_tune_config_param(config, "step_size", all_values, all_ids,
                                      'lr-t-max-hyperparam', model, agent)
    config = format_tune_config_param(config, "step_size", all_values, all_ids,
                                      'lr-eta-min-hyperparam', model, agent)
    return config

def format_tune_model_layers(config, all_values, all_ids, all_indexed_values, all_indexed_ids, model, agent):
    num_layers = get_specific_value(all_values, all_ids, 'hidden-layers-slider', model, agent)[1]
    config = format_tune_config_param(config, 'num_layers', all_values, all_ids,
                                      'hidden-layers-slider', model, agent, index=None, is_range=True)
    for layer in range(1, num_layers + 1):
        layer_types = get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-type-hyperparam', model, agent, layer)
        config = format_tune_config_param(config, 'layer_types', all_indexed_values, all_indexed_ids,
                                          'layer-type-hyperparam', model, agent, layer)
        for layer_type in layer_types:
            if layer_type == 'dense':
                config = format_tune_config_param(config, 'num_units', all_indexed_values, all_indexed_ids,
                                                  'layer-units-slider', model, agent, layer)
                config = format_tune_config_param(config, 'bias', all_indexed_values, all_indexed_ids,
                                                  'dense-bias-hyperparam', model, agent, layer)
                config = format_tune_kernel(config, all_indexed_values, all_indexed_ids, model, agent, layer)
            elif layer_type == 'cnn':
                config = format_tune_config_param(config, 'out_channels', all_indexed_values, all_indexed_ids,
                                                  'layer-units-slider', model, agent, layer)
    return config

def format_tune_layer_units(config, param_name, all_values, all_ids, all_indexed_values, all_indexed_ids, id, model, agent):
    num_layers = get_specific_value(all_values, all_ids, id, model, agent)
    for i in range(1, num_layers[1] + 1):
        config[f"{agent}_{model}_layer_{i}_{param_name}"] = tune.choice(
            get_specific_value_id(all_indexed_values, all_indexed_ids, 'layer-units-slider', model, agent, i)
        )
    return config



## GYMNASIUM FUNCTIONS
def get_extra_gym_params(env_spec_id):
    extra_params = {}

    if env_spec_id == 'CarRacing-v2':
        extra_params = {
            'lap_complete_percent': {
                'type': 'float',
                'default': 0.95,
                'description': 'Percentage of tiles that must be visited by the agent before a lap is considered complete'
            },
            'domain_randomize': {
                'type': 'boolean',
                'default': False,
                'description': 'Enable domain randomization (background and track colors are different on every reset)'
            },
            'continuous': {
                'type': 'boolean',
                'default': True,
                'description': 'Use continuous action space (False for discrete action space)'
            }
        }

    elif env_spec_id == 'LunarLander-v2':
        extra_params = {
            'continuous': {
                'type': 'boolean',
                'default': False,
                'description': 'Use continuous actions'
            },
            'gravity': {
                'type': 'float',
                'default': -10.0,
                'description': 'Gravitational acceleration (negative number)'
            },
            'enable_wind': {
                'type': 'boolean',
                'default': False,
                'description': 'Add wind force to the lunar lander'
            },
            'wind_power': {
                'type': 'float',
                'default': 15.0,
                'description': 'Strength of wind force if enable_wind is True'
            },
            'turbulence_power': {
                'type': 'float',
                'default': 1.5,
                'description': 'Strength of turbulence if enable_wind is True'
            }
        }

    elif env_spec_id == 'BipedalWalker-v3':
        extra_params = {
            'hardcore': {
                'type': 'boolean',
                'default': False,
                'description': 'Use the hardcore version with obstacles'
            }
        }

    elif env_spec_id.startswith('FetchReach') or env_spec_id.startswith('FetchPush') or \
            env_spec_id.startswith('FetchSlide') or env_spec_id.startswith('FetchPickAndPlace'):
        extra_params = {
            'max_episode_steps': {
                'type': 'int',
                'default': 50,
                'description': "Number of steps per episode"
            }
        }

    elif env_spec_id.startswith('HandReach') or env_spec_id.startswith('HandManipulate'):
        extra_params = {
            'max_episode_steps': {
                'type': 'int',
                'default': 50,
                'description': "Number of steps per episode"
            }
        }

    elif env_spec_id == 'FrozenLake-v1':
        extra_params = {
            'map_name': {
                'type': 'string',
                'default': '4x4',
                'description': 'Name of the map layout'
            },
            'is_slippery': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether the environment is slippery'
            }
        }

    return extra_params

def generate_gym_extra_params_container(env_spec_id):
    extra_params = get_extra_gym_params(env_spec_id)

    if not extra_params:
        return None

    params_container = html.Div([
        html.H4('Extra Parameters'),
        html.Div([
            generate_gym_param_input(param_name, param_info)
            for param_name, param_info in extra_params.items()
        ])
    ])

    return params_container


def generate_gym_param_input(param_name, param_info):
    param_type = param_info['type']
    param_default = param_info['default']
    param_description = param_info['description']

    if param_type == 'boolean':
        input_component = dcc.Dropdown(
            id=f'env-param-{param_name}',
            options=[
                {'label': 'True', 'value': True},
                {'label': 'False', 'value': False}
            ],
            value=param_default
        )
    elif param_type == 'float':
        input_component = dcc.Input(
            id=f'env-param-{param_name}',
            type='number',
            value=param_default
        )
    else:  # Assuming 'string' type
        input_component = dcc.Input(
            id=f'env-param-{param_name}',
            type='text',
            value=param_default
        )

    return html.Div([
        html.Label(param_name, style={'text-decoration': 'underline'}),
        html.Div(param_description, style={'fontSize': '12px', 'color': 'gray'}),
        input_component
    ], style={'marginBottom': '10px'})

def extract_gym_params(component):
    print('extract gym params called..')
    gym_params = {}

    if isinstance(component, dict):
        if component['type'] == 'Label' and 'children' in component['props']:
            param_name = component['props']['children']
            gym_params[param_name] = None  # Initialize parameter value as None

        if 'children' in component['props']:
            children = component['props']['children']
            if isinstance(children, list):
                for child in children:
                    child_params = extract_gym_params(child)
                    gym_params.update(child_params)
            elif isinstance(children, dict):
                child_params = extract_gym_params(children)
                gym_params.update(child_params)

        if component['type'] in ['Input', 'Dropdown', 'Slider'] and 'id' in component['props']:
            param_id = component['props']['id']
            if param_id.startswith('env-param-'):
                param_name = param_id.replace('env-param-', '')
                param_value = component['props'].get('value')
                gym_params[param_name] = param_value

    return gym_params