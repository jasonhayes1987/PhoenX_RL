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
# import tensorflow as tf
import numpy as np

import rl_agents
import models
import helper
import rl_callbacks

def get_specific_value(all_values, all_ids, value_type, value_model, agent_type):
    #DEBUG
    # print(f'get_specific_value fired...')
    for id_dict, value in zip(all_ids, all_values):
        # Check if this id dictionary matches the criteria
        if id_dict.get('type') == value_type and id_dict.get('model') == value_model and id_dict.get('agent') == agent_type:
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
        value_type='noise-function',
        value_model='none',
        agent_type=agent_type,
    )

    if noise_type == "Normal":
        return helper.Noise.create_instance(
            noise_class_name=noise_type,
            shape=env.action_space.shape,
            mean=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='normal-mean',
                value_model='none',
                agent_type=agent_type,
            ),
            stddev=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='normal-stddv',
                value_model='none',
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
                value_type='uniform-min',
                value_model='none',
                agent_type=agent_type,
            ),
            maxval=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='uniform-max',
                value_model='none',
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
                value_type='ou-mean',
                value_model='none',
                agent_type=agent_type,
            ),
            theta=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='ou-theta',
                value_model='none',
                agent_type=agent_type,
            ),
            sigma=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='ou-sigma',
                value_model='none',
                agent_type=agent_type,
            ),
            dt=get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                value_type='ou-dt',
                value_model='none',
                agent_type=agent_type,
            ),

        )

def format_layers(all_values, all_ids, layer_units_values, layer_units_ids, value_type, value_model, agent_type):
    #DEBUG
    # print(f'format_layers fired...')
    num_layers = get_specific_value(
        all_values=all_values,
        all_ids=all_ids,
        value_type='dense-layers',
        value_model=value_model,
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
        value_type='conv-layers',
        value_model=model_type,
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

def get_all_gym_envs():
    """Returns a list of all gym environments."""

    exclude_list = [
        "/",
        "Gym",
        "Reacher-v2",
        "Pusher-v2",
        "InvertedPendulum-v2",
        "InvertedDoublePendulum-v2",
        "HalfCheetah-v2",
        "HalfCheetah-v3",
        "Hopper-v2",
        "Hopper-v3",
        "Walker2d-v2",
        "Walker2d-v3",
        "Swimmer-v2",
        "Swimmer-v3",
        "Ant-v2",
        "Ant-v3",
        "Humanoid-v2",
        "Humanoid-v3",
        "HumanoidStandup-v2",
        "BipedalWalkerHardcore-v3",
        "LunarLanderContinuous-v2",
        "FrozenLake8x8-v1",
        "MountainCarContinuous-v0",
    ]
    return [
        env_spec
        for env_spec in gym.envs.registration.registry
        if not any(exclude in env_spec for exclude in exclude_list)
    ]

def load(agent_data, env_name):
    # check if the env name in agent data matches env_name var
    if agent_data['env'] == env_name:
        #DEBUG
        # print('env name matches!')
        # Load the agent
        return rl_agents.load_agent_from_config(agent_data['save_dir'])

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
    
    return agent

def train_model(agent_data, env_name, num_episodes, render, render_freq):  
    # print('Training agent...')
    # agent = rl_agents.load_agent_from_config(save_dir)
    agent = load(agent_data, env_name)
    # print('Agent loaded.')
    agent.train(num_episodes, render, render_freq)
    # print('Training complete.')

def test_model(agent_data, env_name, num_episodes, render, render_freq):  
    # print('Testing agent...')
    # agent = rl_agents.load_agent_from_config(save_dir)
    agent = load(agent_data, env_name)
    # print('Agent loaded.')
    agent.test(num_episodes, render, render_freq)
    # print('Testing complete.')

def clear_renders_folder(folder_path):
    # Make sure folder_path is a Path object
    folder_path = Path(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_video_files(page):
    if os.path.exists('assets/renders'):
        try:
            if page == "/train-agent":
                # Get video files from training renders folder
                return natsorted([f for f in os.listdir(Path("assets/renders/training")) if f.endswith('.mp4')])
            elif page == "/test-agent":
                # Get video files from testing renders folder
                return natsorted([f for f in os.listdir(Path("assets/renders/testing")) if f.endswith('.mp4')])
        except Exception as e:
            print(f"Failed to get video files: {e}")


## LAYOUT COMPONENT GENERATORS

# Function to generate carousel items from video paths
def generate_video_items(video_files, page):
    if page == "/train-agent":
        folder = 'training'
    elif page == "/test-agent":
        folder = 'testing'
    else:
        raise ValueError(f"Invalid page {page}")
    return [
        html.Video(src=f'assets/renders/{folder}/{video_file}', controls=True,
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

# Environment dropdown component
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

# Testing settings component
def parameter_settings_component(page):
    return html.Div([
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
                {'label': 'Render Testing Episodes', 'value': 'RENDER'}
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
        return initializer_input_creators.get(selected_initializer, lambda: html.Div())(initializer_id)


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
            dcc.Slider(
                id={
                    'type':'gain',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                },
                min=1.0,
                max=3.0,
                step=1.0,
                value=1.0,
                marks={1.0:'1.0', 3.0:'3.0'},
                tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.Slider(
                id={
                    'type':'gain',
                    'model':initializer_id['model'],
                    'agent':initializer_id['agent'],
                },
                min=1.0,
                max=3.0,
                step=1.0,
                value=1.0,
                marks={1.0:'1.0', 3.0:'3.0'},
                tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.Slider(
                id={
                'type':'mean',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                },
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),

            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'std',
                'model':initializer_id['model'],
                'agent':initializer_id['agent'],
                },
                min=0.01,
                max=2.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 1.00:'1.00', 1.99:'1.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
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
            dcc.Slider(
                id={
                'type':'a',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),

            html.Label('Maximum', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'b',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),
            html.Hr(),
    ])


def create_normal_initializer_inputs(initializer_id):
    """Component for random normal initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'mean',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),

            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'std',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=0.01,
                max=1.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 1.00:'1.00', 1.99:'1.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),
            html.Hr(),
    ])


def create_constant_initializer_inputs(initializer_id):
    """Component for constant initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Value', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'val',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.99,  # Default position
                marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            ),
            html.Hr(),
        ]
    )


def create_variance_scaling_inputs(initializer_id):
    """Component for variance scaling initializer hyperparameters"""
    return html.Div(
        children=[
            html.Label('Scale', style={'text-decoration': 'underline'}),
            dcc.Slider(
                id={
                'type':'scale',
                'model':initializer_id['model'],
                'agent':initializer_id['agent']
                },
                min=1.0,
                max=5.0,
                step=1.0,
                value=2.0,  # Default position
                marks={0.0:'0.0', 5.0:'5.0'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
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
        'variance scaling': ['scale', 'mode', 'distribution'],
        'constant': ['val'],
        'normal': ['mean', 'std'],
        'uniform': ['a', 'b'],
        'truncated normal': ['mean', 'std'],
        'xavier uniform': ['gain'],
        'xavier normal': ['gain'],
        'kaiming uniform': ['mode'],
        'kaiming normal': ['mode'],
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


def create_agent_parameter_inputs(agent_type):
    """Component for agent hyperparameters"""
    if agent_type in ['Reinforce', 'ActorCritic']:
        return html.Div(
            id=f'{agent_type}-inputs',
            children=[
                html.Label('Learning Rate', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'learning-rate',
                        'model':'none',
                        'agent':agent_type,
                    }, 
                    min=-6,
                    max=-2,
                    step=None,
                    value=-4,  # Default position
                    marks={i: f'10^{i}' for i in range(-6, -1)},
                    # tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Label('Discount Factor', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'discount',
                        'model':'none',
                        'agent':agent_type
                    },
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    value=0.99,  # Default position
                    marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                # Policy Model Configuration for Reinforce and Actor Critic
                html.H3("Policy Model Configuration"),
                html.Label('Policy Hidden Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'hidden-layers',
                        'model':'policy',
                        'agent':agent_type,
                    },
                    min=1,
                    max=10,
                    step=1,
                    value=2,  # Default position
                    marks={1:'1', 10:'10'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'units-per-layer',
                        'model':'policy',
                        'agent':agent_type,
                    }
                ),
                html.Hr(),
                dcc.Dropdown(
                    id={
                        'type':'kernel-function',
                        'model':'policy',
                        'agent':agent_type,
                    },
                    options=[{'label': "Constant", 'value': "constant"},
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
                    placeholder="Kernel Function"
                ),
                html.Div(
                    id={
                        'type': 'kernel-initializer-options',
                        'model': 'policy',
                        'agent': agent_type,
                    }
                ),
                dcc.Dropdown(
                    id={
                        'type':'activation-function',
                        'model':'policy',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
                    placeholder="Activation Function",
                ),
                dcc.Dropdown(
                    id={
                        'type':'optimizer',
                        'model':'policy',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
                    placeholder="Optimizer",
                ),
                # Value Model Configuration for Reinforce and Actor Critic
                html.H3("Value Model Configuration"),
                html.Label('Value Hidden Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'hidden-layers',
                        'model':'value',
                        'agent':agent_type,
                    },
                    min=1,
                    max=10,
                    step=1,
                    value=2,  # Default position
                    marks={1:'1', 10:'10'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'units-per-layer',
                        'model':'value',
                        'agent':agent_type,
                    }
                ),
                html.Hr(),
                dcc.Dropdown(
                    id={
                        'type':'kernel-function',
                        'model':'value',
                        'agent':agent_type,
                    },
                    options=[{'label': 'Constant', 'value': 'constant'},
                             {'label': 'Xavier Uniform', 'value': 'xavier_uniform'},
                             {'label': 'Xavier Normal', 'value': 'xavier_normal'},
                             {'label': 'Kaiming Uniform', 'value': 'kaiming_uniform'},
                             {'label': 'Kaiming Normal', 'value': 'kaiming_normal'},
                             {'label': 'Zeros', 'value': 'zeros'},
                             {'label': 'Ones', 'value': 'ones'},
                             {'label': 'Uniform', 'value': 'uniform'},
                             {'label': 'Normal', 'value': 'normal'},
                             {'label': 'Truncated Normal', 'value': 'truncated_normal'},
                             {'label': 'Variance Scaling', 'value': 'variance_scaling'},
                    ],
                    placeholder="Kernel Function",
                ),
                html.Div(
                    id={
                        'type': 'kernel-initializer-options',
                        'model': 'value',
                        'agent': agent_type,
                    }
                ),
                dcc.Dropdown(
                    id={
                        'type':'activation-function',
                        'model':'value',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
                    placeholder="Activation Function",
                ),
                dcc.Dropdown(
                    id={
                        'type':'optimizer',
                        'model':'value',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
                    placeholder="Optimizer",
                ),
                # Additional configuration for Actor Critic
                html.Div(id='actor-critic-config-container'),  # Container to hold the generated inputs
            ])
    elif agent_type == 'DDPG':
        return html.Div(
            id=f'{agent_type}-inputs',
            children=[
                # DDPG specific inputs
                html.Label('Discount Factor', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'discount',
                        'model':'none',
                        'agent':agent_type
                    },
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    value=0.99,  # Default position
                    marks={0.01:'0.01', 0.50:'0.50', 0.99:'0.99'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Label('Tau', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'tau',
                        'model':'none',
                        'agent':agent_type,
                    },
                    min=0.001,
                    max=0.999,
                    value=0.005,
                    step=0.001,
                    marks={0.001: "0.001", 0.500: "0.500", 0.999: "0.999"},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Label('Batch Size', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'batch-size',
                        'model':'none',
                        'agent':agent_type,
                    },
                    min=1,
                    max=1024,
                    step=1,
                    value=64,  # Default position
                    marks={1:'1', 1024:'1024'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
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
                # Actor Model Configuration
                html.H3("Actor Model Configuration"),
                html.Label('Actor Convolution Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'conv-layers',
                        'model':'actor',
                        'agent':agent_type,
                    }, 
                    min=0,
                    max=20,
                    step=1,
                    value=0,  # Default position
                    marks={0:'0', 20:'20'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'layer-types',
                        'model':'actor',
                        'agent':agent_type,
                    }
                ),

                html.Label('Actor Dense Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'dense-layers',
                        'model':'actor',
                        'agent':agent_type,
                    }, 
                    min=1,
                    max=10,
                    step=1,
                    value=2,  # Default position
                    marks={1:'1', 10:'10'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'units-per-layer',
                        'model':'actor',
                        'agent':agent_type,
                    }
                ),
                html.Hr(),
                dcc.Dropdown(
                    id={
                        'type':'kernel-function',
                        'model':'actor',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ["constant", "xavier uniform", "xavier normal", "kaiming uniform", "kaiming normal", "zeros", "ones", \
                        "uniform", "normal", "truncated normal", "variance scaling"]],
                    placeholder="Kernel Function",
                ),
                html.Div(
                    id={
                        'type': 'kernel-initializer-options',
                        'model': 'actor',
                        'agent': agent_type,
                    }
                ),
                dcc.Dropdown(
                    id={
                        'type':'activation-function',
                        'model':'actor',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
                    placeholder="Activation Function",
                ),
                dcc.Dropdown(
                    id={
                        'type':'optimizer',
                        'model':'actor',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
                    placeholder="Optimizer",
                ),
                html.Label('Learning Rate', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'learning-rate',
                        'model':'actor',
                        'agent':agent_type,
                    },
                    min=-6,
                    max=-2,
                    step=None,
                    value=-4,  # Default position
                    marks={i: f'10^{i}' for i in range(-6, -1)},
                    # tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),

                # Critic Model Configuration
                html.H3("Critic Model Configuration"),
                html.Label('Critic Convolution Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'conv-layers',
                        'model':'critic',
                        'agent':agent_type,
                    }, 
                    min=0,
                    max=20,
                    step=1,
                    value=0,  # Default position
                    marks={0:'0', 20:'20'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'layer-types',
                        'model':'critic',
                        'agent':agent_type,
                    }
                ),
                
                html.Label('Critic State Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'dense-layers',
                        'model':'critic-state',
                        'agent':agent_type,
                    }, 
                    min=1,
                    max=10,
                    step=1,
                    value=1,  # Default position
                    marks={1:'1', 10:'10'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'units-per-layer',
                        'model':'critic-state',
                        'agent':agent_type,
                    }
                ),
                html.Hr(),
                html.Label('Critic Merged Layers', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'dense-layers',
                        'model':'critic-merged',
                        'agent':agent_type,
                    }, 
                    min=1,
                    max=10,
                    step=1,
                    value=2,  # Default position
                    marks={1:'1', 10:'10'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
                html.Div(
                    id={
                        'type':'units-per-layer',
                        'model':'critic-merged',
                        'agent':agent_type,
                    }
                ),
                html.Hr(),
                dcc.Dropdown(
                    id={
                        'type':'kernel-function',
                        'model':'critic',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ["constant", "xavier uniform", "xavier normal", "kaiming uniform", "kaiming normal", "zeros", "ones", \
                        "uniform", "normal", "truncated normal", "variance scaling"]],
                    placeholder="Kernel Function",
                ),
                html.Div(
                    id={
                        'type': 'kernel-initializer-options',
                        'model': 'critic',
                        'agent': agent_type,
                    }
                ),
                dcc.Dropdown(
                    id={
                        'type':'activation-function',
                        'model':'critic',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['relu', 'tanh', 'sigmoid']],
                    placeholder="Activation Function",
                ),
                dcc.Dropdown(
                    id={
                        'type':'optimizer',
                        'model':'critic',
                        'agent':agent_type,
                    },
                    options=[{'label': i, 'value': i} for i in ['Adam', 'SGD', 'RMSprop', 'Adagrad']],
                    placeholder="Optimizer",
                ),
                html.Label('Learning Rate', style={'text-decoration': 'underline'}),
                dcc.Slider(
                    id={
                        'type':'learning-rate',
                        'model':'critic',
                        'agent':agent_type,
                        },
                    min=-6,
                    max=-2,
                    step=None,
                    value=-4,  # Default position
                    marks={i: f'10^{i}' for i in range(-6, -1)},
                    # tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                ),
            ])
    else:
        return html.Div("Select a model type to configure its parameters.")
    
## HYPERPARAMETER SEARCH FUNCTIONS

def generate_reinforce_hyperparam_component():
    return html.Div([html.H3('Reinforce Hyperparameters')
                     ,
                    ])

def generate_actor_critic_hyperparam_component():
    return html.Div([html.H3('Actor Critic Hyperparameters'),])





def generate_learning_rate_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Learning Rate'),
        dcc.RangeSlider(
            id={
                'type': 'learning-rate-slider',
                'model': model_type,
                'agent': agent_type
            },
            min=-6,  # For 10^-6
            max=-2,  # For 10^-2
            value=[-6, -2],  # Default range
            marks={i: f'10^{i}' for i in range(-6, -1)},  # marks for better readability
            step=None,  # anchor slider to marks
            # pushable=1,  # allow pushing the slider
            allowCross=True, # allow selecting single value
            # tooltip={"placement": "bottom", "always_visible": True}
        )             
    ])



def generate_discount_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Discount'),
        dcc.RangeSlider(
            id={
                'type': 'discount-slider',
                'model': model_type,
                'agent': agent_type
            },
            min=0.01,
            max=0.99,
            value=[0.01, 0.50],
            step=0.01,
            marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
            # pushable=0.01,  # allow pushing the slider
            allowCross=True, # allow selecting single value
            tooltip={"placement": "bottom", "always_visible": True},
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
            min=1,
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
            dcc.RangeSlider(
                id={
                    'type': 'dropout-prob-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                    'index': index,
                },
                min=0.0,
                max=1.0,
                step=0.1,
                value=[0.2, 0.5],
                marks={0.0:'0.0', 1.0:'1.0'},
                allowCross=True,
                tooltip={"placement": "bottom", "always_visible": True},
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
            min=1,
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

def generate_kernel_initializer_hyperparam_component(agent_type, model_type):
    return html.Div([
        html.H5('Hidden Layers Kernel Initializer'),
        dcc.Dropdown(
            id={
                'type':'kernel-function-hyperparam',
                'model':model_type,
                'agent':agent_type,
            },
            options=[{'label': "Constant", 'value': "constant"},
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
                dcc.RangeSlider(
                    id={
                        'type':'xavier-uniform-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                    },
                    min=1.0,
                    max=3.0,
                    step=1.0,
                    value=[1.0, 3.0],
                    marks={1.0:'1.0', 3.0:'3.0'},
                    allowCross=True,
                    tooltip={"placement": "bottom", "always_visible": True},
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
                dcc.RangeSlider(
                    id={
                        'type':'xavier-normal-gain-hyperparam',
                        'model':model_type,
                        'agent':agent_type,
                    },
                    min=1.0,
                    max=3.0,
                    step=1.0,
                    value=[1.0, 3.0],
                    marks={1.0:'1.0', 3.0:'3.0'},
                    allowCross=True,
                    tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.RangeSlider(
                id={
                    'type': 'variance-scaling-scale-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                },
                min=1.0,
                max=5.0,
                value=[1.0, 2.0],  # Default range
                marks={1.0: "0", 5.0: "5.0"},
                step=1.0,  # anchor slider to marks
                # pushable=1,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.RangeSlider(
                id={
                    'type': 'constant-value-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.1,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    label='Constant')

def generate_normal_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'random-normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.1,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'random-normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=1.99,
                value=[0.01, 1.99],  
                marks={0.01: "0.01", 1.0: "1.0", 1.99: "1.99"},
                step=.01,  # anchor slider to marks
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    label='Random Normal')

def generate_uniform_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([ 
            html.Label('Minimum', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'random-uniform-minval-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=0.01,
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Maximum', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'random-uniform-maxval-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=0.01,
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ]),
    ],
    label='Random Uniform')

def generate_truncated_normal_kernel_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'truncated-normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type,
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.1,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'truncated-normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=1.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 1.00: "1.00", 1.99: "1.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True}
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
    ])

def generate_trace_decay_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5(f"{model_type.capitalize()} Trace Decay"),
        dcc.RangeSlider(
            id={
                'type': 'trace-decay-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            min=0.01,
            max=0.99,
            value=[0.01, 0.50],
            step=0.01,
            marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
            # pushable=.01,  # allow pushing the slider
            allowCross=True, # allow selecting single value
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ])

def generate_tau_hyperparam_componenent(agent_type, model_type):
    return html.Div([
        html.H5("Tau"),
        dcc.RangeSlider(
            id={
                'type': 'tau-hyperparam',
                'model': model_type,
                'agent': agent_type,
            },
            min=0.001,
            max=0.999,
            value=[0.001, 0.500],
            step=0.001,
            marks={0.001: "0.001", 0.500: "0.500", 0.999: "0.999"},
            # pushable=.001,  # allow pushing the slider
            allowCross=True, # allow selecting single value
            tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.RangeSlider(
                id={
                    'type': 'ou-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Mean Reversion', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'ou-theta-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=False,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Volatility', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'ou-sigma-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=0.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    label='Ornstein-Uhlenbeck')

def generate_uniform_noise_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Minimum Value', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'uniform-min-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.49],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Maximum Value', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'uniform-max-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.50, 0.99],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    label='Uniform')

def generate_normal_noise_hyperparam_inputs(agent_type, model_type):
    return dcc.Tab([
        html.Div([
            html.Label('Mean', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'normal-mean-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
            dcc.RangeSlider(
                id={
                    'type': 'normal-stddev-hyperparam',
                    'model': model_type,
                    'agent': agent_type
                },
                min=0.01,
                max=0.99,
                value=[0.01, 0.50],
                step=0.01,
                marks={0.01: "0.01", 0.50: "0.50", 0.99: "0.99"},
                # pushable=.01,  # allow pushing the slider
                allowCross=True, # allow selecting single value
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    label='Normal')

def generate_replay_buffer_hyperparam_component(agent_type, model_type):
    pass

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
            #DEBUG
            # print(f'DDPG agent selected...')
            sweep_config["parameters"][agent]["parameters"] = {}

            # set parameters based on selection(s)

            # actor learning rate
            value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'actor', agent)
            if value_range[0] == value_range[1]:
                config = {"value": 10**value_range[0]}
            else:
                config = {"min": 10**value_range[0], "max": 10**value_range[1]}         
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_learning_rate"] = config
            #DEBUG
            # print(f'DDPG actor learning rate set to {config}')


            # critic learning rate
            value_range = get_specific_value(all_values, all_ids, 'learning-rate-slider', 'critic', agent)
            if value_range[0] == value_range[1]:
                config = {"value": 10**value_range[0]}
            else:
                config = {"min": 10**value_range[0], "max": 10**value_range[1]}           
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_learning_rate"] = config
            #DEBUG
            # print(f'DDPG critic learning rate set to {config}')


            # discount
            value_range = get_specific_value(all_values, all_ids, 'discount-slider', 'none', agent)
            if value_range[0] == value_range[1]:
                config = {"value": value_range[0]}
            else:
                config = {"min": value_range[0], "max": value_range[1]}           
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_discount"] = config
            #DEBUG
            # print(f'DDPG discount set to {config}')


            # tau
            value_range = get_specific_value(all_values, all_ids, 'tau-hyperparam', 'none', agent)
            if value_range[0] == value_range[1]:
                config = {"value": value_range[0]}
            else:
                config = {"min": value_range[0], "max": value_range[1]}           
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_tau"] = config
            #DEBUG
            # print(f'DDPG tau set to {config}')


            # actor cnn layers
            value_range = get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)
            if value_range[0] == value_range[1]:
                config = {"value": value_range[0]}
            else:
                config = {"min": value_range[0], "max": value_range[1]}           
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_cnn_layers"] = config
            #DEBUG
            # print(f'DDPG actor cnn layers set to {config}')


            # actor num layers
            value_range = get_specific_value(all_values, all_ids, 'hidden-layers-slider', 'actor', agent)
            if value_range[0] == value_range[1]:
                config = {"value": value_range[0]}
            else:
                config = {"min": value_range[0], "max": value_range[1]}           
            
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_num_layers"] = config
            #DEBUG
            # print(f'DDPG actor num layers set to {config}')


            # actor activation
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"] = \
                {"values": get_specific_value(all_values, all_ids, 'activation-function-hyperparam', 'actor', agent)}
            #DEBUG
            # print(f'DDPG actor activation set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_activation"]}')

            # actor kernel initializer
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"] = \
                {"values": get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor', agent)}
            #DEBUG
            # print(f'DDPG actor kernel initializer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_initializer"]}')

            # actor optimizer
            sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"] = \
                {"values": get_specific_value(all_values, all_ids, 'optimizer-hyperparam', 'actor', agent)}
            #DEBUG
            print(f'DDPG actor optimizer set to {sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_optimizer"]}')

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

            # replay buffer
            sweep_config["parameters"][agent]["parameters"][f"{agent}_replay_buffer"] = {"values": ["ReplayBuffer"]}
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
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}
                    
                    # theta
                    value_range = get_specific_value(all_values, all_ids, 'ou-theta-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["theta"] = {"value": value_range[0]}
                    else:
                        config["theta"] = {"min": value_range[0], "max": value_range[1]}

                    # sigma
                    value_range = get_specific_value(all_values, all_ids, 'ou-sigma-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["sigma"] = {"value": value_range[0]}
                    else:
                        config["sigma"] = {"min": value_range[0], "max": value_range[1]}
                    
                elif noise == "Normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'normal-mean-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'normal-stddev-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["stddev"] = {"value": value_range[0]}
                    else:
                        config["stddev"] = {"min": value_range[0], "max": value_range[1]}

                
                elif noise == "Uniform":
                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'uniform-min-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["minval"] = {"value": value_range[0]}
                    else:
                        config["minval"] = {"min": value_range[0], "max": value_range[1]}

                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'uniform-max-hyperparam', 'none', agent)
                    if value_range[0] == value_range[1]:
                        config["maxval"] = {"value": value_range[0]}
                    else:
                        config["maxval"] = {"min": value_range[0], "max": value_range[1]}

                sweep_config["parameters"][agent]["parameters"][f"{agent}_noise_{noise}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG noise set to {config}')

            # kernel options       
            # actor kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'actor', agent):
                if f"{agent}_actor_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config = {"value": value_range[0]}
                    else:
                        config = {"min": value_range[0], "max": value_range[1]}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["scale"] = {"value": value_range[0]}
                    else:
                        config["scale"] = {"min": value_range[0], "max": value_range[1]}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'actor', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'actor', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["maxval"] = {"value": value_range[0]}
                    else:
                        config["maxval"] = {"min": value_range[0], "max": value_range[1]}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["minval"] = {"value": value_range[0]}
                    else:
                        config["minval"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["stddev"] = {"value": value_range[0]}
                    else:
                        config["stddev"] = {"min": value_range[0], "max": value_range[1]}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["stddev"] = {"value": value_range[0]}
                    else:
                        config["stddev"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["gain"] = {"value": value_range[0]}
                    else:
                        config["gain"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'actor', agent)
                    if value_range[0] == value_range[1]:
                        config["gain"] = {"value": value_range[0]}
                    else:
                        config["gain"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'actor', agent)
                    config["mode"] = {"values": values}


                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'actor', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_actor_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG actor kernel set to {config}')

            # critic kernel options
            for kernel in get_specific_value(all_values, all_ids, 'kernel-function-hyperparam', 'critic', agent):
                if f"{agent}_critic_kernel_{kernel}" not in sweep_config["parameters"][agent]["parameters"]:
                    sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_{kernel}"]={"parameters":{}}

                # initialize empty config dictionary for parameters
                config = {}

                if kernel == "constant":
                    value_range = get_specific_value(all_values, all_ids, 'constant-value-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config = {"value": value_range[0]}
                    else:
                        config = {"min": value_range[0], "max": value_range[1]}
       
                elif kernel == "variance_scaling":
                    # scale
                    value_range = get_specific_value(all_values, all_ids, 'variance-scaling-scale-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["scale"] = {"value": value_range[0]}
                    else:
                        config["scale"] = {"min": value_range[0], "max": value_range[1]}

                    # mode
                    config["mode"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-mode-hyperparam', 'critic', agent)}

                    # distribution
                    config["distribution"] = {"values": get_specific_value(all_values, all_ids, 'variance-scaling-distribution-hyperparam', 'critic', agent)}

                elif kernel == "uniform":
                    # maxval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-maxval-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["maxval"] = {"value": value_range[0]}
                    else:
                        config["maxval"] = {"min": value_range[0], "max": value_range[1]}

                    # minval
                    value_range = get_specific_value(all_values, all_ids, 'random-uniform-minval-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["minval"] = {"value": value_range[0]}
                    else:
                        config["minval"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-mean-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'random-normal-stddev-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["stddev"] = {"value": value_range[0]}
                    else:
                        config["stddev"] = {"min": value_range[0], "max": value_range[1]}
        
                elif kernel == "truncated_normal":
                    # mean
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-mean-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["mean"] = {"value": value_range[0]}
                    else:
                        config["mean"] = {"min": value_range[0], "max": value_range[1]}

                    # stddev
                    value_range = get_specific_value(all_values, all_ids, 'truncated-normal-stddev-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["stddev"] = {"value": value_range[0]}
                    else:
                        config["stddev"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "xavier_uniform":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-uniform-gain-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["gain"] = {"value": value_range[0]}
                    else:
                        config["gain"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "xavier_normal":
                    # gain
                    value_range = get_specific_value(all_values, all_ids, 'xavier-normal-gain-hyperparam', 'critic', agent)
                    if value_range[0] == value_range[1]:
                        config["gain"] = {"value": value_range[0]}
                    else:
                        config["gain"] = {"min": value_range[0], "max": value_range[1]}

                elif kernel == "kaiming_uniform":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-uniform-mode-hyperparam', 'critic', agent)
                    config["mode"] = {"values": values}

                elif kernel == "kaiming_normal":
                    # mode
                    values = get_specific_value(all_values, all_ids, 'kaiming-normal-mode-hyperparam', 'critic', agent)
                    config["mode"] = {"values": values}

                    
                else:
                    if kernel not in ["constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones", \
                        "uniform", "normal", "truncated_normal", "variance_scaling"]:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                sweep_config["parameters"][agent]["parameters"][f"{agent}_critic_kernel_{kernel}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG critic kernel set to {config}')

            # CNN layer params
            # Actor CNN layers
            for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'actor', agent)[1] + 1):
                sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_types_{agent}"] = {"parameters":{}}
                config = {}
                config["types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'actor', agent, i)}

                # loop through each type in CNN layer and get the parameters to add to the sweep config
                for value in config["types"]["values"]:
                    if value == "conv":
                        config["conv_filters"] = {
                            "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'actor', agent, i)
                        }

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config["conv_kernel_size"] = {"value": value_range[0]}
                        else:
                            config["conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config["conv_strides"] = {"value": value_range[0]}
                        else:
                            config["conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                        config["conv_padding"] = {
                            "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'actor', agent, i)
                        }

                        if config["conv_padding"]["value"] == 'custom':
                            value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'actor', agent, i)
                            if value_range[0] == value_range[1]:
                                config["conv_padding"] = {"value": value_range[0]}
                            else:
                                config["conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                            
                            # val_config["conv_padding"]["parameters"] = pad_config
                        
                        config["conv_bias"] = {
                            "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'actor', agent, i)
                        }
                    
                    if value == "pool":

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config["pool_kernel_size"] = {"value": value_range[0]}
                        else:
                            config["pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config["pool_strides"] = {"value": value_range[0]}
                        else:
                            config["pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                    if value == "dropout":

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'actor', agent, i)
                        if value_range[0] == value_range[1]:
                            config["dropout_prob"] = {"value": value_range[0]}
                        else:
                            config["dropout_prob"] = {"min": value_range[0], "max": value_range[1]}

                    # config["parameters"] = val_config

                sweep_config["parameters"][agent]["parameters"][f"actor_cnn_layer_{i}_types_{agent}"]["parameters"] = config
            #DEBUG
            # print(f'DDPG actor CNN layers set to {config}')

            # Critic CNN layers
            for i in range(1, get_specific_value(all_values, all_ids, 'cnn-layers-slider-hyperparam', 'critic', agent)[1] + 1):
                sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_types_{agent}"] = {"parameters":{}}
                config = {}
                config["types"] = {"values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'cnn-layer-type-hyperparam', 'critic', agent, i)}

                # loop through each type in CNN layer and get the parameters to add to the sweep config
                for value in config["types"]["values"]:
                    if value == "conv":
                        config["conv_filters"] = {
                            "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-filters-hyperparam', 'critic', agent, i)
                        }

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-kernel-size-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config["conv_kernel_size"] = {"value": value_range[0]}
                        else:
                            config["conv_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-stride-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config["conv_strides"] = {"value": value_range[0]}
                        else:
                            config["conv_strides"] = {"min": value_range[0], "max": value_range[1]}

                        config["conv_padding"] = {
                            "value": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-hyperparam', 'critic', agent, i)
                        }

                        if config["conv_padding"]["value"] == 'custom':
                            value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-padding-custom-hyperparam', 'critic', agent, i)
                            if value_range[0] == value_range[1]:
                                config["conv_padding"] = {"value": value_range[0]}
                            else:
                                config["conv_padding"] = {"min": value_range[0], "max": value_range[1]}
                        
                        config["conv_bias"] = {
                            "values": get_specific_value_id(all_indexed_values, all_indexed_ids, 'conv-use-bias-hyperparam', 'critic', agent, i)
                        }
                    
                    if value == "pool":

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-kernel-size-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config["pool_kernel_size"] = {"value": value_range[0]}
                        else:
                            config["pool_kernel_size"] = {"min": value_range[0], "max": value_range[1]}

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'pool-stride-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config["pool_strides"] = {"value": value_range[0]}
                        else:
                            config["pool_strides"] = {"min": value_range[0], "max": value_range[1]}

                    if value == "dropout":

                        value_range = get_specific_value_id(all_indexed_values, all_indexed_ids, 'dropout-prob-hyperparam', 'critic', agent, i)
                        if value_range[0] == value_range[1]:
                            config["dropout_prob"] = {"value": value_range[0]}
                        else:
                            config["dropout_prob"] = {"min": value_range[0], "max": value_range[1]}

                sweep_config["parameters"][agent]["parameters"][f"critic_cnn_layer_{i}_types_{agent}"]["parameters"] = config
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
        # print(f'Sweep Config: {sweep_config}')
        
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
    
def render_heatmap():
    return html.Div([
        html.Label('Bins', style={'text-decoration': 'underline'}),
        dcc.Slider(
            id='bin-slider',
            min=1,
            max=10,
            value=5,
            marks={i: str(i) for i in range(1, 11)},
            step=1,
        ),
        html.Div(id='legend-container'),
        html.Label('Reward Threshold', style={'text-decoration': 'underline'}),
        dcc.Input(
            id='reward-threshold',
            type='number',
            value=0,
            style={'display':'inline-block'}
        ),
        dcc.Checklist(
                id='z-score-checkbox',
                options=[
                    {'label': ' Display Z-Scores', 'value': 'zscore'},
                ],
                value=[],
                style={'display':'inline-block'}
        ),
        html.Div(id='heatmap-container'),
        html.Div(
            id='heatmap-placeholder',
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