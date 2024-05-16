from zipfile import ZipFile
import os
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
import gymnasium as gym
import gymnasium_robotics as gym_robo
# import tensorflow as tf
import numpy as np

import rl_agents
import models
import helper
import rl_callbacks

def get_specific_value(all_values, all_ids, id_type, model_type, agent_type):
    #DEBUG
    # print(f'get_specific_value fired...')
    for id_dict, value in zip(all_ids, all_values):
        # Check if this id dictionary matches the criteria
        if id_dict.get('type') == id_type and id_dict.get('model') == model_type and id_dict.get('agent') == agent_type:
            #DEBUG
            # print(f"Found value {value} for {value_type} {value_model} {agent_type}")
            return value
    # Return None or some default value if not found
    return None

def get_specific_value_id(all_values, all_ids, value_type, value_model, agent_type, index):
    #DEBUG
    # print(f'get_specific_value fired...')
    # print(f'all_values: {all_values}')
    # print(f'all_ids: {all_ids}')
    for id_dict, value in zip(all_ids, all_values):
        if 'index' in id_dict.keys():
            # print(f'id_dict: {id_dict}')
            # Check if this id dictionary matches the criteria
            if id_dict.get('type') == value_type and id_dict.get('model') == value_model and id_dict.get('agent') == agent_type and id_dict.get('index') == index:
                #DEBUG
                # print(f"Found value {value} for {value_type} {value_model} {agent_type}")
                return value
    # Return None or some default value if not found
    return None

def create_noise_object(env, all_values, all_ids, agent_type):
    noise_type = get_specific_value(
        all_values=all_values,
        all_ids=all_ids,
        id_type='noise-function',
        model_type='none',
        agent_type=agent_type,
    )

    if noise_type == "Normal":
        return helper.Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            mean=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normal-mean',
                model_type='none',
                agent_type=agent_type,
            ),
            stddev=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normal-stddv',
                model_type='none',
                agent_type=agent_type,
            ),

        )

    elif noise_type == "Uniform":
        return helper.Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            minval=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='uniform-min',
                model_type='none',
                agent_type=agent_type,
            ),
            maxval=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='uniform-max',
                model_type='none',
                agent_type=agent_type,
            ),

        )

    elif noise_type == "Ornstein-Uhlenbeck":
        return helper.Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            mean=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='ou-mean',
                model_type='none',
                agent_type=agent_type,
            ),
            theta=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='ou-theta',
                model_type='none',
                agent_type=agent_type,
            ),
            sigma=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='ou-sigma',
                model_type='none',
                agent_type=agent_type,
            ),
            dt=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='ou-dt',
                model_type='none',
                agent_type=agent_type,
            ),

        )

def format_layers(all_values, all_ids, layer_units_values, layer_units_ids, value_type, value_model, agent_type):
    #DEBUG
    # print(f'format_layers fired...')
    num_layers = get_specific_value(
        all_values=all_values,
        all_ids=all_ids,
        id_type='dense-layers',
        model_type=value_model,
        agent_type=agent_type,
    )
    #DEBUG
    # print(f'{num_layers} for {value_type}, {value_model}, {agent_type}')

    #DEBUG
    # print(f'layer units ids: {layer_units_ids}')
    # print(f'layer units values: {layer_units_values}')
    
    units = []
    for index in range(num_layers):
        for id_dict, value in zip(layer_units_ids, layer_units_values):
            # Check if this id dictionary matches the criteria
            if id_dict.get('type') == value_type \
            and id_dict.get('model') == value_model \
            and id_dict.get('agent') == agent_type \
            and id_dict.get('index') == index+1:
                # print('added unit')
                # print(f'id_dict: {id_dict}')
                # print(f'value: {value}')
                units.append(value)
     #DEBUG
    # print(f'{units} for {value_type}, {value_model}, {agent_type}')     
    return units

def format_cnn_layers(all_values, all_ids, all_indexed_values, all_indexed_ids, model_type, agent_type):
    layers = []
    # Get num CNN layers for model type
    num_cnn_layers = get_specific_value(
        all_values=all_values,
        all_ids=all_ids,
        id_type='conv-layers',
        model_type=model_type,
        agent_type=agent_type,
    )
    #DEBUG
    # print(f'num_cnn_layers: {num_cnn_layers}')

    # Loop through num of CNN layers
    for index in range(1, num_cnn_layers+1):
        # Get the layer type
        layer_type = get_specific_value_id(
            all_values=all_indexed_values,
            all_ids=all_indexed_ids,
            value_type='cnn-layer-type',
            value_model=model_type,
            agent_type=agent_type,
            index=index
        )
        #DEBUG
        # print(f'layer_type: {layer_type}')

        # Parse layer types to set params
        if layer_type == "conv":
            params = {}
            params['out_channels'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='conv-filters',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            params['kernel_size'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='conv-kernel-size',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            params['stride'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='conv-stride',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            padding = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='conv-padding',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            if padding == "custom":
                params['padding'] = get_specific_value_id(
                    all_values=all_indexed_values,
                    all_ids=all_indexed_ids,
                    value_type='conv-padding-custom',
                    value_model=model_type,
                    agent_type=agent_type,
                    index=index
                )

            else:
                params['padding'] = padding


            params['bias'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='conv-use-bias',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            # Append to layers list
            layers.append({layer_type: params})
            continue
        
        elif layer_type == "batchnorm":
            params = {}
            params['num_features'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='batch-features',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            layers.append({layer_type: params})
            continue
        
        elif layer_type == "pool":
            params = {}
            params['kernel_size'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='pool-kernel-size',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            params['stride'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='pool-stride',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            layers.append({layer_type: params})
            continue

        elif layer_type == 'dropout':
            params = {}
            params['p'] = get_specific_value_id(
                all_values=all_indexed_values,
                all_ids=all_indexed_ids,
                value_type='dropout-prob',
                value_model=model_type,
                agent_type=agent_type,
                index=index
            )
            layers.append({layer_type: params})
            continue

        elif layer_type == 'relu':
            params = {}
            layers.append({layer_type: params})

        elif layer_type == 'tanh':
            params = {}
            layers.append({layer_type: params})

        else:
            raise ValueError(f'Layer type {layer_type} not supported')
        
    return layers




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

        policy_model = models.PolicyModel(
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

def train_model(agent_data, env_name, num_episodes, render, render_freq, num_epochs=None, num_cycles=None, num_updates=None, workers=None):  
    # print('Training agent...')
    # agent = rl_agents.load_agent_from_config(save_dir)
    agent = load(agent_data, env_name)
    # print('Agent loaded.')
    if agent_data['agent_type'] == "HER":
        agent.train(num_epochs, num_cycles, num_episodes, num_updates, render, render_freq)
    else:
        agent.train(num_episodes, render, render_freq)
    # print('Training complete.')

def test_model(agent_data, env_name, num_episodes, render, render_freq):  
    # print('Testing agent...')
    # agent = rl_agents.load_agent_from_config(save_dir)
    agent = load(agent_data, env_name)
    # print('Agent loaded.')
    agent.test(num_episodes, render, render_freq)
    # print('Testing complete.')

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

# Environment dropdown components
def env_dropdown_component(page):
    env_options = [
        {'label': env_name, 'value': env_name} for env_name in get_all_gym_envs()
    ]
    
    return dcc.Dropdown(
        id={
            'type': 'env-dropdown',
            'page': page,
        },
        options=env_options,
        placeholder="Select Gym Environment",
    )

def get_all_gym_envs():
    """Returns a list of all gym environments."""

    exclude_list = [
        "/",
        "Gym",
        "CartPole-v0",
        "Reacher-v2",
        "Reacher-v4",
        "Pusher-v2",
        "Pusher-v4",
        "InvertedPendulum-v2",
        "InvertedPendulum-v4",
        "InvertedDoublePendulum-v2",
        "InvertedDoublePendulum-v4",
        "HalfCheetah-v2",
        "HalfCheetah-v3",
        "HalfCheetah-v4",
        "Hopper-v2",
        "Hopper-v3",
        "Hopper-v4",
        "Walker2d-v2",
        "Walker2d-v3",
        "Walker2d-v4",
        "Swimmer-v2",
        "Swimmer-v3",
        "Swimmer-v4",
        "Ant-v2",
        "Ant-v3",
        "Ant-v4",
        "Humanoid-v2",
        "Humanoid-v3",
        "Humanoid-v4",
        "HumanoidStandup-v2",
        "HumanoidStandup-v4",
        "BipedalWalkerHardcore-v3",
        "LunarLanderContinuous-v2",
        "FrozenLake8x8-v1",
        "MountainCarContinuous-v0",
        "FetchReach-v1",
        "FetchSlide-v1",
        "FetchPush-v1",
        "FetchPickAndPlace-v1",
    ]
    return [
        env_spec
        for env_spec in gym.envs.registration.registry
        if not any(exclude in env_spec for exclude in exclude_list)
    ]

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
}
    description = env_data[env_name]['description']
    gif_url = env_data[env_name]['gif_url']

    return description, gif_url

# Training settings component
def run_agent_settings_component(page, agent_type=None):
    return html.Div([
        html.Div(
            id={
                'type': 'mpi-options',
                'page': page,
            },
            style={'display': 'none'},
            children=[
                html.Label('Use MPI', style={'text-decoration': 'underline'}),
                dcc.RadioItems(
                    id={
                        'type': 'mpi',
                        'page': page,
                    },
                    options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False},
                    ],
                ),
                dcc.Input(
                    id={
                        'type': 'workers',
                        'page': page,
                    },
                    type='number',
                    placeholder="Number of Workers",
                    min=1,
                    style={'display': 'none'},
                ),
            ]
        ),
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
        dcc.Input(
            id={
                'type': 'num-episodes',
                'page': page,
            },
            type='number',
            placeholder="Number of Episodes",
            min=1,
        ),
        dcc.Checklist(
            options=[
                {'label': 'Load Weights', 'value': True}
            ],
            id={
                'type': 'load-weights',
                'page': page,
            },
            value=[True]
        ),
        dcc.Checklist(
            options=[
                {'label': 'Render Episodes', 'value': 'RENDER'}
            ],
            id={
                'type': 'render-option',
                'page': page,
            },
            value=[]
        ),
        dcc.Input(
            id={
                'type': 'render-freq',
                'page': page,
            },
            type='number',
            placeholder="Every 'n' Episodes",
            min=1,
            disabled=True,
        ),
        dcc.Input(
            id={
                'type': 'seed',
                'page': page,
            },
            type='number',
            placeholder="Random Seed",
            min=1,
        ),
        dcc.Input(
            id={
                'type': 'run-number',
                'page': page,
            },
            type='number',
            placeholder="WANDB Run Number (blank for None)",
        ),
        dcc.Input(
            id={
                'type': 'num-runs',
                'page': page,
            },
            type='number',
            placeholder="Number of Runs",
            min=1,
        ),
        html.Label('Save Directory:', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
                'type':'save-dir',
                'page':page,
            },
            type='text',
            placeholder='path/to/model'
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


def get_kernel_initializer_inputs(selected_initializer, initializer_id):
    # Dictionary mapping the initializer names to the corresponding function
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
        return initializer_input_creators.get(selected_initializer)(initializer_id)
    elif selected_initializer not in ['ones', 'zeros', 'default']:
        raise ValueError(f"{selected_initializer} not in initializer input creator dict")

def create_kaiming_normal_initializer_inputs(initializer_id):
    """Component for kaiming uniform initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
            },
        children=[
            html.Label('Mode', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'mode',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                    },
                options=[
                        {'label': 'fan in', 'value': 'fan_in'},
                        {'label': 'fan out', 'value': 'fan_out'},
                    ],
                value='fan_in',
            ),
            html.Hr(),
        ]
    )


def create_kaiming_uniform_initializer_inputs(initializer_id):
    """Component for kaiming uniform initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
            },
        children=[
            html.Label('Mode', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'mode',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                    },
                options=[
                        {'label': 'fan in', 'value': 'fan_in'},
                        {'label': 'fan out', 'value': 'fan_out'},
                    ],
                value='fan_in',
            ),
            html.Hr(),
        ]
    )
                    


def create_xavier_normal_initializer_inputs(initializer_id):
    """Component for xavier uniform initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
            },
        children=[
            html.Label('Gain', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'gain',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                },
                type='number',
                min=1.0,
                max=3.0,
                step=1.0,
                value=1.0,
            ),
            html.Hr(),
        ],
    )


def create_xavier_uniform_initializer_inputs(initializer_id):
    """Component for xavier uniform initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
            },
        children=[
            html.Label('Gain', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'gain',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                },
                type='number',
                min=1.0,
                max=3.0,
                step=1.0,
                value=1.0,
            ),
            html.Hr(),
        ],
    )

def create_truncated_normal_initializer_inputs(initializer_id):
    """Component for truncated normal initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
        },
        children=[
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'mean',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),

            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'std',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                },
                type='number',
                min=0.01,
                max=2.99,
                step=0.01,
                value=0.99,  # Default position
            ),
            html.Hr(),
    ])


def create_uniform_initializer_inputs(initializer_id):
    """Component for random uniform initializer hyperparameters"""
    return html.Div(
        id={
            'type': 'kernel-params',
            'model': initializer_id['model'],
            'agent': initializer_id['agent']
        },
        children=[
            html.Label('Minimum', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'a',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=-0.99,
                max=0.99,
                step=0.001,
                value=-0.99,  # Default position
            ),

            html.Label('Maximum', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'b',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=-0.99,
                max=0.99,
                step=0.001,
                value=0.99
            ),
            html.Hr(),
    ])


def create_normal_initializer_inputs(initializer_id):
    """Component for random normal initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'mean',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),

            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'std',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=0.01,
                max=1.99,
                step=0.01,
                value=0.99,
            ),
            html.Hr(),
    ])


def create_constant_initializer_inputs(initializer_id):
    """Component for constant initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Value', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'val',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,
            ),
            html.Hr(),
        ]
    )


def create_variance_scaling_inputs(initializer_id):
    """Component for variance scaling initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Scale', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                'type':'scale',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                type='number',
                min=1.0,
                max=5.0,
                step=1.0,
                value=2.0,  # Default position
            ),
            
            html.Label('Mode', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'mode',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent']
                },
                options=[{'label': mode, 'value': mode} for mode in ['fan_in', 'fan_out', 'fan_avg']],
                placeholder="Mode",
                value='fan_in'
            ),
            
            html.Label('Distribution', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'distribution',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent']
                },
                options=[{'label': dist, 'value': dist} for dist in ['truncated_normal', 'uniform']],
                placeholder="Distribution",
                value='truncated_normal'
            ),
            html.Hr(),
        ]
    )

def format_kernel_initializer_config(all_values, all_ids, value_model, agent_type):
    """Returns an initializer object based on initializer component values"""
    # Define an initializer config dictionary listing params needed for each initializer
    initializer_configs = {
        'variance_scaling': ['scale', 'mode', 'distribution'],
        'constant': ['val'],
        'normal': ['mean', 'std'],
        'uniform': ['a', 'b'],
        'truncated_normal': ['mean', 'std'],
        'xavier_uniform': ['gain'],
        'xavier_normal': ['gain'],
        'kaiming_uniform': ['mode'],
        'kaiming_normal': ['mode'],
        'zeros': [],
        'ones': [],
        'default': [],
    }

    # Get initializer type
    initializer_type = get_specific_value(all_values, all_ids, 'kernel-function', value_model, agent_type)

    # create empty dictionary to store initializer config params
    config = {}
    # Iterate over initializer_type params list and get values
    for param in initializer_configs[initializer_type]:
        config[param] = get_specific_value(all_values, all_ids, param, value_model, agent_type)
    
    # format 
    initializer_config = {initializer_type: config}

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
                value=0.99,
            )
        ]
    )

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

def create_kernel_input(agent_type, model_type):
    return html.Div(
        [
            dcc.Dropdown(
                id={
                    'type':'kernel-function',
                    'model':model_type,
                    'agent':agent_type,
                },
                options = [{'label': i, 'value': i.replace(' ', '_')} for i in 
                           ["default", "constant", "xavier uniform", "xavier normal", "kaiming uniform",
                            "kaiming normal", "zeros", "ones", "uniform", "normal",
                            "truncated normal","variance scaling"]],
                placeholder="Kernel Function",
            ),
            html.Div(
                id={
                    'type': 'kernel-initializer-options',
                    'model': model_type,
                    'agent': agent_type,
                }
            ),
        ]
    )

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
            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
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
            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
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
            html.Label("Learning Rate Decay", style={'text-decoration': 'underline'}),
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
            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
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
            html.Label("Momentum", style={'text-decoration': 'underline'}),
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
            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
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
            html.Label("Momentum", style={'text-decoration': 'underline'}),
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
    
def get_optimizer_params(agent_type, model_type, all_values, all_ids):
    optimizer = get_specific_value(all_values, all_ids, 'optimizer', model_type, agent_type)

    # instantiate empty dict to store params
    params = {}

    if optimizer == 'Adam':
        weight_decay = get_specific_value(all_values, all_ids, 'adam-weight-decay', model_type, agent_type)
        params['weight_decay'] = weight_decay

    elif optimizer == 'Adagrad':
        weight_decay = get_specific_value(all_values, all_ids, 'adagrad-weight-decay', model_type, agent_type)
        lr_decay = get_specific_value(all_values, all_ids, 'adagrad-lr-decay', model_type, agent_type)
        params['weight_decay'] = weight_decay
        params['lr_decay'] = lr_decay

    elif optimizer == 'RMSprop':
        weight_decay = get_specific_value(all_values, all_ids, 'rmsprop-weight-decay', model_type, agent_type)
        momentum = get_specific_value(all_values, all_ids, 'rmsprop-momentum', model_type, agent_type)
        params['weight_decay'] = weight_decay
        params['momentum'] = momentum

    elif optimizer == 'SGD':
        weight_decay = get_specific_value(all_values, all_ids, 'sgd-weight-decay', model_type, agent_type)
        momentum = get_specific_value(all_values, all_ids, 'sgd-momentum', model_type, agent_type)
        params['weight_decay'] = weight_decay
        params['momentum'] = momentum
    
    else:
        raise ValueError(f"{optimizer} not found in utils.get_optimizer_params")
    
    return params

    
def create_learning_rate_input(agent_type, model_type):
    return html.Div(
        [
            html.Label('Learning Rate (10^x)', style={'text-decoration': 'underline'}),
            dcc.Input(
                id={
                    'type':'learning-rate',
                    'model':model_type,
                    'agent':agent_type,
                },
                type='number',
                min=-6,
                max=-2,
                step=1,
                value=-4,
            ),
        ]
    )

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
    

def create_policy_model_input(agent_type):
    return html.Div(
        [
            html.H3("Policy Model Configuration"),
            create_dense_layers_input(agent_type, 'policy'),
            create_kernel_input(agent_type, 'policy'),
            create_activation_input(agent_type, 'policy'),
            create_optimizer_input(agent_type, 'policy'),
        ]
    )


def create_value_model_input(agent_type):
    return html.Div(
        [
            html.H3("Policy Model Configuration"),
            create_dense_layers_input(agent_type, 'value'),
            create_kernel_input(agent_type, 'value'),
            create_activation_input(agent_type, 'value'),
            create_optimizer_input(agent_type, 'value'),
        ]
    )
    

def create_actor_model_input(agent_type):
    return html.Div(
        [
            html.H3("Actor Model Configuration"),
            create_convolution_layers_input(agent_type, 'actor'),
            create_dense_layers_input(agent_type, 'actor'),
            create_normalize_layers_input(agent_type, 'actor'),
            create_clamp_output_input(agent_type, 'actor'),
            html.Label("Hidden Layers Kernel Initializers"),
            create_kernel_input(agent_type, 'actor-hidden'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'actor-output'),
            create_activation_input(agent_type, 'actor'),
            create_optimizer_input(agent_type, 'actor'),
            create_learning_rate_input(agent_type, 'actor'),
        ]
    )


def create_critic_model_input(agent_type):
    return html.Div(
        [
            html.H3("Critic Model Configuration"),
            create_convolution_layers_input(agent_type, 'critic'),
            create_dense_layers_input(agent_type, 'critic-state'),
            create_dense_layers_input(agent_type, 'critic-merged'),
            create_normalize_layers_input(agent_type, 'critic'),
            html.Label("Hidden Layers Kernel Initializers"),
            create_kernel_input(agent_type, 'critic-hidden'),
            html.Label("Output Layer Kernel Initializer"),
            create_kernel_input(agent_type, 'critic-output'),
            create_activation_input(agent_type, 'critic'),
            create_optimizer_input(agent_type, 'critic'),
            create_learning_rate_input(agent_type, 'critic'),
        ]
    )


def create_reinforce_parameter_inputs(agent_type):
    return html.Div(
        [
            create_learning_rate_input(agent_type, 'none'),
            create_discount_factor_input(agent_type),
            # Policy Model
            create_policy_model_input(agent_type),
            # Value Mode
            create_value_model_input(agent_type),
            # Save dir
            create_save_dir_input(agent_type),
        ]
    )

def create_actor_critic_parameter_inputs(agent_type):
    return html.Div(
        [
            create_learning_rate_input(agent_type, 'none'),
            create_discount_factor_input(agent_type),
            create_trace_decay_input(agent_type, 'policy'),
            create_trace_decay_input(agent_type, 'value'),
            # Policy Model
            create_policy_model_input(agent_type),
            # Value Model
            create_value_model_input(agent_type),
            # Save dir
            create_save_dir_input(agent_type),
        ]
    )

def create_ddpg_parameter_inputs(agent_type):
    """Adds inputs for Hindsight Experience Replay w/DDPG Agent"""
    return html.Div(
        id=f'{agent_type}-inputs',
        children=[
            create_device_input(agent_type),
            create_discount_factor_input(agent_type),
            create_tau_input(agent_type),
            create_epsilon_greedy_input(agent_type),
            create_batch_size_input(agent_type),
            create_noise_function_input(agent_type),
            create_input_normalizer_input(agent_type),
            # Actor Model Configuration
            create_actor_model_input(agent_type),
            # Critic Model Configuration
            html.H3("Critic Model Configuration"),
            create_critic_model_input(agent_type),
            # Save dir
            create_save_dir_input(agent_type)
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


def create_agent_parameter_inputs(agent_type):
    """Component for agent hyperparameters"""
    if agent_type == 'Reinforce':
        return create_reinforce_parameter_inputs(agent_type)

    elif agent_type == 'ActorCritic':
        return create_actor_critic_parameter_inputs(agent_type)

    elif agent_type == 'DDPG':
        return create_ddpg_parameter_inputs(agent_type)
    
    elif agent_type == 'HER_DDPG':
        return create_her_ddpg_parameter_inputs(agent_type)

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
            create_input_normalizer_hyperparam_input(agent_type, 'none'),
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
            html.Hr(),
            html.H6("Input Normalizers"),
            create_input_normalizer_options_hyperparam_input(agent_type, 'none'),
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
                    create_clamp_output_hyperparam_input(agent_type, 'actor'),
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
            'type':'hyperparam-units-per-layer',
            'model':model_type,
            'agent':agent_type
        })
    ])

def generate_hidden_units_per_layer_hyperparam_component(agent_type, model_type, layer_num):
    return html.Div([
        html.H6(f'Neurons in Hidden Layer {layer_num}'),
        dcc.Dropdown(
            id={
                'type': f'layer-{layer_num}-units-slider',
                'model': model_type,
                'agent': agent_type,
                # 'index': layer_num
            },
            options=[{'label': i, 'value': i} for i in [8, 16, 32, 64, 128, 256, 512, 1024]],
            multi=True,
        )
    ])

def generate_kernel_initializer_hyperparam_component(agent_type, model_type, layer_type):
    return html.Div([
        html.H5(f'{layer_type} Kernel Initializer'),
        dcc.Dropdown(
            id={
                'type':'kernel-function-hyperparam',
                'model':model_type,
                'agent':agent_type,
            },
            options=[{'label': "Default", 'value': "default"},
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
        ),
        html.Div(
            id={
                'type':'hyperparam-kernel-options',
                'model':model_type,
                'agent':agent_type
            },
            children=[
                html.H6(
                    id={
                        'type': 'kernel-options-header',
                        'model': model_type,
                        'agent': agent_type
                    },
                    children=['Kernel Options'],
                    hidden=True
                ),
                dcc.Tabs(
                    id={
                        'type': 'kernel-options-tabs',
                        'model': model_type,
                        'agent': agent_type
                    }
                ),
            ]
        )
    ])

def generate_kernel_options_hyperparam_component(agent_type, model_type, selected_initializers):
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
            tabs.append(
                kernel_input_creators.get(initializer)(agent_type, model_type)
            )
        
    return tabs


def generate_xavier_uniform_hyperparam_inputs(agent_type, model_type):
    """Component for xavier uniform initializer hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type
                },
            children=[
                html.Label('Gain', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'xavier-uniform-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
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


def generate_xavier_normal_hyperparam_inputs(agent_type, model_type):
    """Component for xavier normal initializer hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type
                },
            children=[
                html.Label('Gain', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'xavier-normal-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
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

def generate_kaiming_uniform_hyperparam_inputs(agent_type, model_type):
    """Component for kaiming uniform initializer sweep hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                },
            children=[
                html.Label('Mode', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'kaiming-uniform-mode-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        },
                    options=[
                            {'label': 'fan in', 'value': 'fan_in'},
                            {'label': 'fan out', 'value': 'fan_out'},
                        ],
                    value='fan_in',
                    multi=True,
                ),
                html.Hr(),
            ]
        )
        ],
        label='Kaiming Uniform'
    )

def generate_kaiming_normal_hyperparam_inputs(agent_type, model_type):
    """Component for kaiming normal initializer sweep hyperparameters"""
    return dcc.Tab([
        html.Div(
            id={
                'type': 'kernel-params-hyperparam',
                'model': model_type,
                'agent': agent_type,
                },
            children=[
                html.Label('Mode', style={'text-decoration': 'underline'}),
                dcc.Dropdown(
                    id={
                        'type':'kaiming-normal-mode-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                        },
                    options=[
                            {'label': 'fan in', 'value': 'fan_in'},
                            {'label': 'fan out', 'value': 'fan_out'},
                        ],
                    value=['fan_in'],
                    multi=True,
                ),
                html.Hr(),
            ]
        )
    ],
    label='Kaiming Normal'
    )

def generate_variance_scaling_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Scale', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'variance-scaling-scale-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
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
                    'agent': agent_type
                },
                options=[{'label': mode, 'value': mode} for mode in ['fan_in', 'fan_out', 'fan_avg']],
                placeholder="Mode",
                value=['fan_in'],
                multi=True
            ),
            html.Label('Distribution', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type':'variance-scaling-distribution-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                options=[{'label': dist, 'value': dist} for dist in ['truncated normal', 'uniform']],
                placeholder="Distribution",
                value=['truncated normal'],
                multi=True
            ),
        ])
    ],
    label='Variance Scaling')

def generate_constant_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Value', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'constant-value-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
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
                ],
                multi=True,
            ),
        ])
    ],
    label='Constant')

def generate_normal_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-normal-mean-hyperparam',
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
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-normal-stddev-hyperparam',
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
    label='Random Normal')

def generate_uniform_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([ 
            html.Label('Minimum', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-uniform-minval-hyperparam',
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
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Maximum', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'random-uniform-maxval-hyperparam',
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
        ]),
    ],
    label='Random Uniform')

def generate_truncated_normal_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'truncated-normal-mean-hyperparam',
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
                    {'label': '1.0', 'value': 1.0},
                ],
                multi=True,
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Dropdown(
                id={
                    'type': 'truncated-normal-stddev-hyperparam',
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
    label='Truncated Normal')

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

def generate_replay_buffer_hyperparam_component(agent_type, model_type):
    pass

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

def generate_seed_component(agent_type, model_type):
    return html.Div([
        html.Label('Seed', style={'text-decoration': 'underline'}),
        dcc.Input(
            id={
                'type':'seed',
                'model': model_type,
                'agent': agent_type
            },
            type='number',
            placeholder="Leave blank for random seed"
        ),
    ])

## WEIGHTS AND BIASES FUNCTIONS
def generate_wandb_login(page):
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

def generate_wandb_project_dropdown(page):
    projects = get_projects()
    return html.Div([
            # html.Label("Select a Project:"),
            dcc.Dropdown(
            id={'type':'projects-dropdown', 'page':page},
            options=[{'label': project, 'value': project} for project in projects],
            placeholder="Select a W&B Project",
            )
        ])

def generate_sweeps_dropdown(page):
    return html.Div([
        dcc.Dropdown(
            id={'type':'sweeps-dropdown', 'page':page},
            options=[],
            multi=True,
            placeholder="Select a W&B Sweep",
            )
    ])


def create_wandb_config(method, project, sweep_name, metric_name, metric_goal, env, env_params, agent_selection, all_values, all_ids, all_indexed_values, all_indexed_ids):
    #DEBUG
    # print(f'create wandb config fired...')
    sweep_config = {
        "method": method,
        "project": project,
        "name": sweep_name,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": {
            "env": {
                "parameters":{
                    "id": {"value": env},
                    **{param: {"value":value} for param, value in env_params.items()},
                },
            },
            "model_type": {"values": agent_selection},
        }
    }
    # set base config for each agent type
    for agent in agent_selection:
        # Initialize the dictionary for the agent if it doesn't exist

        if agent not in sweep_config["parameters"]:
            sweep_config["parameters"][agent] = {}
        
        if agent == "DDPG":
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

            # normalize input
            sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_input"] = \
                {"values": get_specific_value(all_values, all_ids, 'normalize-input-hyperparam', 'none', agent)}

            # normalize input options
            for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_input"]['values']:
                if value == 'True':
                    value_range = get_specific_value(all_values, all_ids, 'norm-clip-value-hyperparam', 'none', agent)
                    config = {"values": value_range}
                sweep_config["parameters"][agent]["parameters"][f"{agent}_normalize_clip"] = config
            
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

            # actor hidden layer kernel params
            for value in sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"]['values']:
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer_{value}_options"] = {'parameters': {}}
                config = {}
                if value == 'variance_scaling':
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_scale'] = {"values": value_range}

                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_mode'] = {"values": value_range}

                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_distribution'] = {"values": value_range}

                elif value == 'constant':
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_value'] = {"values": value_range}

                elif value == 'normal':
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_mean'] = {"values": value_range}

                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_stddev'] = {"values": value_range}

                elif value == 'uniform':
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_minval'] = {"values": value_range}

                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_maxval'] = {"values": value_range}

                elif value == 'truncated_normal':
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_mean'] = {"values": value_range}

                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_stddev'] = {"values": value_range}

                elif value == "kaiming_normal":
                    value_range = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_mode'] = {"values": value_range}

                elif value == "kaiming_uniform":
                    value_range = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_mode'] = {"values": value_range}

                elif value == "xavier_normal":
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_gain'] = {"values": value_range}

                elif value == "xavier_uniform":
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-hidden', agent)
                    config[f'{value}_gain'] = {"values": value_range}

                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer_{value}_options"]['parameters'] = config


            # actor output layer kernel initializer
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
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"] = \
                {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent)}
            #DEBUG
            # print(f'DDPG critic kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"]}')

            # critic output layer kernel initializer
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"] = \
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

            # kernel options       
            # actor hidden kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent):
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
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-hidden', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-hidden', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-hidden', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-hidden', agent)
                    config["mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-hidden', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG actor kernel set to {config}')

            # actor output kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent):
                if f"{agent}_actor_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-output', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-output', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-output', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-output', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-output', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-output', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-output', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-output', agent)
                    config["mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-output', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]["parameters"] = config

            # critic hidden kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent):
                if f"{agent}_critic_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-hidden', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-hidden', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-hidden', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-hidden', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-hidden', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-hidden', agent)
                    config["mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-hidden', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG critic kernel set to {config}')

            # critic output kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent):
                if f"{agent}_critic_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-output', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-output', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-output', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-output', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-output', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-output', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-output', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-output', agent)
                    config["mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-output', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]["parameters"] = config

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
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"] = \
                {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor', agent)}
            #DEBUG
            # print(f'DDPG actor kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"]}')

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
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_clamp_output"] = \
                {"values": get_specific_value(all_values, all_ids, 'clamp-value-hyperparam', 'actor', agent)}

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

            # critic kernel initializer
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"] = \
                {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic', agent)}
            #DEBUG
            # print(f'DDPG critic kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_initializer"]}')

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
            
            
            # replay buffer # NOT NEEDED
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

            # kernel options       
            # actor hidden kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-hidden', agent):
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
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-hidden', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-hidden', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-hidden', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-hidden', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-hidden', agent)
                    config["mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-hidden', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_hidden_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG actor kernel set to {config}')

            # actor output kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor-output', agent):
                if f"{agent}_actor_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor-output', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor-output', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor-output', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor-output', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor-output', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor-output', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor-output', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor-output', agent)
                    config["mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor-output', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_output_kernel_{kernel}"]["parameters"] = config

            # critic hidden kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-hidden', agent):
                if f"{agent}_critic_hidden_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-hidden', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-hidden', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-hidden', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-hidden', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-hidden', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-hidden', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-hidden', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-hidden', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-hidden', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-hidden', agent)
                    config["mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-hidden', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_hidden_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG critic kernel set to {config}')

            # critic output kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic-output', agent):
                if f"{agent}_critic_output_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_output_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic-output', agent)
                    config["value"] = {"values": value_range}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic-output', agent)
                    config["scale"] = {"values": value_range}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic-output', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic-output', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic-output', agent)
                    config["maxval"] = {"values": value_range}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic-output', agent)
                    config["minval"] = {"values": value_range}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic-output', agent)
                    config["stddev"] = {"values": value_range}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic-output', agent)
                    config["mean"] = {"values": value_range}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic-output', agent)
                    config["stddev"] = {"values": value_range}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic-output', agent)
                    config["gain"] = {"values": value_range}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic-output', agent)
                    config["mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic-output', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
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
            'reward_type': {
                'type': 'string',
                'default': 'sparse',
                'description': "Type of reward function ('sparse' or 'dense')"
            }
        }

    elif env_spec_id.startswith('HandReach') or env_spec_id.startswith('HandManipulate'):
        extra_params = {
            'distance_threshold': {
                'type': 'float',
                'default': 0.01,
                'description': 'Threshold for the distance to the goal'
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