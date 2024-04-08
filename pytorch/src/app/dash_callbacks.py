import os
from pathlib import Path
import multiprocessing
from multiprocessing import Queue
from queue import Empty
import threading
import time
import json
import base64
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask import request
import numpy as np

import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.tools as tls
import plotly.express as px



import gymnasium as gym
import wandb
# import tensorflow as tf
import pandas as pd

import layouts
import helper
import utils
import models
import cnn_models
import rl_agents
import wandb_support
# import tasks

# Create a queue to store the formatted data
formatted_data_queue = Queue()

def fetch_data_process(project, sweep_name, shared_data):
    while True:
        try:
            # print("Fetching data from wandb...")
            metrics_data = wandb_support.get_metrics(project, sweep_name)
            formatted_data = wandb_support.format_metrics(metrics_data)
            shared_data['formatted_data'] = formatted_data
            # print("Data fetched and formatted successfully.")
            time.sleep(10)  # Wait for 60 seconds before fetching data again
        except Exception as e:
            print(f"Error in fetch_data_process: {str(e)}")
            time.sleep(5)


def update_heatmap_process(shared_data, hyperparameters, bins, z_score, reward_threshold):
    # while True:
    try:
        if 'formatted_data' in shared_data:
            #DEBUG
            print(f'shared data: {shared_data}')
            # print("Calculating co-occurrence matrix...")
            formatted_data = shared_data['formatted_data']
            #DEBUG
            print(f'formatted data passed to wandb_support: {formatted_data}')
            matrix_data, bin_ranges = wandb_support.calculate_co_occurrence_matrix(formatted_data, hyperparameters, reward_threshold, bins, z_score)
            #DEBUG
            print(f'bin ranges returned from wandb_support: {bin_ranges}')
            shared_data['matrix_data'] = matrix_data.to_dict(orient='split')
            shared_data['bin_ranges'] = bin_ranges
            #DEBUG
            print(f'data in shared data: {shared_data}')
            print("Co-occurrence matrix calculated successfully.")
        # time.sleep(5)  # Wait for 5 seconds before updating the heatmap again
    except Exception as e:
        print(f"Error in update_heatmap_process: {str(e)}")
        # time.sleep(5)


def register_callbacks(app, shared_data):
    @app.callback(
            Output('page-content', 'children'),
            Input('url', 'pathname')
    )
    def display_page(page):
        if page == '/':
            return layouts.home(page)
        elif page == '/build-agent':
            return layouts.build_agent(page)
        elif page == '/train-agent':
            return layouts.train_agent(page)
        elif page == '/test-agent':
            return layouts.test_agent(page)
        elif page == '/hyperparameter-search':
            return layouts.hyperparameter_search(page)
        elif page == '/wandb-utils':
            return layouts.wandb_utils(page)
        # Add additional conditions for other pages
        else:
            return '404'
    
    @app.callback(
        Output('agent-parameters-inputs', 'children'),
        Input({'type':'agent-type-dropdown', 'page':'/build-agent'}, 'value'),
    )
    def update_agent_parameters_inputs(agent_type):
        if agent_type:
            return utils.create_agent_parameter_inputs(agent_type)
        else:
            return "Select a model type to configure its parameters."
        
    @app.callback(
    Output({'type': 'units-per-layer', 'model': MATCH, 'agent': MATCH}, 'children'),
    Input({'type': 'dense-layers', 'model': MATCH, 'agent': MATCH}, 'value'),
    State({'type': 'dense-layers', 'model': MATCH, 'agent': MATCH}, 'id'),
)
    def update_units_per_layer_inputs(num_layers, id):
        if num_layers is not None:
            model_type = id['model']
            agent_type = id['agent']
            inputs = []
            for i in range(1, num_layers + 1):
                input_id = {
                    'type': 'layer-units',
                    'model': model_type,
                    'agent': agent_type,
                    'index': i,
                }
                inputs.append(html.Div([
                    html.Label(f'Neurons in Hidden Layer {i}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id=input_id,
                        min=1,
                        max=1024,
                        step=1,
                        value=512,  # Default position
                        marks={0:'0', 1024:'1024'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                ]))

            return inputs
        
        
    @app.callback(
    Output({'type': 'layer-types', 'model': MATCH, 'agent': MATCH}, 'children'),
    Input({'type': 'conv-layers', 'model': MATCH, 'agent': MATCH}, 'value'),
    State({'type': 'conv-layers', 'model': MATCH, 'agent': MATCH}, 'id'),
)
    def update_cnn_layer_type_inputs(num_layers, id):
        if num_layers is not None:
            model_type = id['model']
            agent_type = id['agent']

            layer_types = []
            for i in range(1, num_layers + 1):
                input_id = {
                    'type': 'cnn-layer-type',
                    'model': model_type,
                    'agent': agent_type,
                    'index': i,
                }
                layer_types.append(html.Div([
                    html.Label(f'Layer Type for Conv Layer {i}', style={'text-decoration': 'underline'}),
                    dcc.Dropdown(
                        id=input_id,
                        options=[
                            {'label': 'Conv2D', 'value': 'conv'},
                            {'label': 'MaxPool2D', 'value': 'pool'},
                            {'label': 'Dropout', 'value': 'dropout'},
                            {'label': 'BatchNorm2D', 'value': 'batchnorm'},
                            {'label': 'Relu', 'value':'relu'},
                            {'label': 'Tanh', 'value': 'tanh'},
                        ]
                    ),
                    html.Div(
                        id={
                            'type': 'cnn-layer-type-parameters',
                            'model': model_type,
                            'agent': agent_type,
                            'index': i,
                        },
                    )
                    ])
                )

            return layer_types
        
    
    @app.callback(
    Output({'type': 'conv-padding-custom-container', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'style'),
    Input({'type': 'conv-padding', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'value')
    )
    def show_hide_custom_padding(padding_value):
        if padding_value == 'custom':
            return {'display': 'block'}
        else:
            return {'display': 'none'}
        
    
    @app.callback(
    Output({'type': 'conv-padding-custom-container-hyperparam', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'style'),
    Input({'type': 'conv-padding-hyperparam', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'value')
    )
    def show_hide_custom_padding_hyperparams(padding_value):
        if padding_value == 'custom':
            return {'display': 'block'}
        else:
            return {'display': 'none'}
        
        
    @app.callback(
    Output({'type': 'cnn-layer-type-parameters', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'children'),
    Input({'type': 'cnn-layer-type', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'value'),
    State({'type': 'cnn-layer-type', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'id'),
    )
    def update_layer_type_params(layer_type, id):
        if layer_type is not None:
            model_type = id['model']
            agent = id['agent']
            index = id['index']

            # loop over layer types to create the appropriate parameters
            if layer_type == 'conv':
                return html.Div([
                    html.Label(f'Filters in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'conv-filters',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=1024,
                        step=1,
                        value=32,
                        marks={1:'1', 1024:'1024'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Label(f'Kernel Size in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'conv-kernel-size',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={1:'1', 10:'10'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Label(f'Kernel Stride in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'conv-stride',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={1:'1', 10:'10'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Label(f'Input Padding in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.RadioItems(
                        id={
                            'type': 'conv-padding',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        options=[
                            {'label': 'Same', 'value': 'same'},
                            {'label': 'Valid', 'value': 'valid'},
                            {'label': 'Custom', 'value': 'custom'},
                        ],
                        value='same',  # Default value
                    ),
                    html.Div(
                        [
                            html.Label('Custom Padding (pixels)', style={'text-decoration': 'underline'}),
                            dcc.Slider(
                                id={
                                    'type': 'conv-padding-custom',
                                    'model': model_type,
                                    'agent': agent,
                                    'index': index,
                                },
                                min=0,
                                max=10,
                                step=1,
                                value=1,
                                marks={0:'0', 10:'10'},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        id={
                            'type': 'conv-padding-custom-container',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        style={'display': 'none'},  # Hide initially
                    ),
                    dcc.Checklist(
                        id={
                            'type': 'conv-use-bias',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        options=[
                            {'label': 'Use Bias', 'value': True},
                        ]
                    )
                ])
            if layer_type == 'pool':
                return html.Div([
                    html.Label(f'Kernel Size of Pooling Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'pool-kernel-size',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={1:'1', 10:'10'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Label(f'Kernel Stride in Pooling Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'pool-stride',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={1:'1', 10:'10'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ])
            if layer_type == 'batchnorm':
                return html.Div([
                    html.Label(f'Number of Features for BatchNorm Layer {index} (set to number of input channels)', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'batch-features',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=1,
                        max=1024,
                        step=1,
                        value=32,
                        marks={1:'1', 1024:'1024'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ])
            if layer_type == 'dropout':
                return html.Div([
                    html.Label(f'Probability of Zero-ed Element for Dropout Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type': 'dropout-prob',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=0.5,
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ])
    
    
    @app.callback(
        Output('actor-critic-config-container', 'children'),
        Input({'type':'agent-type-dropdown', 'page':'/build-agent'}, 'value')
    )
    def update_actor_critic_parameters_inputs(agent_type):
        """Adds additional trace decay options for actor critic model"""
        if agent_type == 'Actor Critic':
            return html.Div([
                html.H3("Actor Critic Configuration"),
                dcc.Input(
                    id={
                        'type':'trace-decay',
                        'model':'policy',
                        'agent':agent_type,
                    },
                    type='number',
                    placeholder="Policy Trace Decay",
                    min=0.0,
                    max=1.0,
                    step=0.01
                ),
                dcc.Input(
                    id={
                        'type':'trace-decay',
                        'model':'value',
                        'agent':agent_type,
                    },
                    type='number',
                    placeholder="Value Trace Decay",
                    min=0.0,
                    max=1.0,
                    step=0.01
                ),
            ])
        else:
            return None
        
    @app.callback(
        Output({'type': 'noise-options', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'noise-function', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'noise-function', 'model': MATCH, 'agent': MATCH}, 'id'),
    )
    def update_noise_inputs(noise_type, id):
        if noise_type is not None:
            agent_type = id['agent']
            if noise_type == "Ornstein-Uhlenbeck":
                inputs = html.Div([
                    html.Label('Mean', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'ou-mean',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.0,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                    html.Label('Mean Reversion', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'ou-sigma',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.15,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                    html.Label('Volatility', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'ou-theta',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.2,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                    html.Label('Time Delta', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'ou-dt',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=1.0,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                ])

            elif noise_type == "Normal":
                inputs = html.Div([
                    html.Label('Mean', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'normal-mean',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=0.0,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                    html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'normal-stddv',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=1.0,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                ])

            elif noise_type == "Uniform":
                inputs = html.Div([
                    html.Label('Minimum Value', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'uniform-min',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=0.1,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                    html.Label('Maximum Value', style={'text-decoration': 'underline'}),
                    dcc.Slider(
                        id={
                            'type':'uniform-max',
                            'model':'none',
                            'agent': agent_type,
                        },
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=1.0,  # Default position
                        marks={0.0:'0.0', 1.0:'1.0'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    ),
                ])
            return inputs
        
    # Callback that updates the placeholder div based on the selected kernel initializer
    @app.callback(
        Output({'type': 'kernel-initializer-options', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'kernel-function', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'kernel-function', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_kernel_initializer_options(selected_initializer, initializer_id):
        # Use the utility function to get the initializer inputs
        return utils.get_kernel_initializer_inputs(selected_initializer, initializer_id)
        
    @app.callback(
        Output('callback-selection', 'children'),
        Input({'type':'agent-type-dropdown', 'page':'/build-agent'}, 'value'),
        # State('url', 'pathname'),
    )
    def update_callback_inputs(agent_type):
        if agent_type is not None:
            return html.Div([
                html.H3("Callback Selection"),
                dcc.Dropdown(
                    id={
                        'type': 'callback',
                        'page': '/build-agent',
                    },
                    options=[
                        {'label': 'Weights & Biases', 'value': 'Weights & Biases'},
                    ],
                    placeholder='Select Callbacks to Use',
                    clearable=True,
                    multi=True,
                ),
            ])
        else:
            return None
        
    @app.callback(
        Output('build-agent-status', 'children'),
        [Input('build-agent-button', 'n_clicks')],
        [State({'type': ALL, 'model': ALL, 'agent': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'id'),
        State({'type':'agent-type-dropdown', 'page':'/build-agent'}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'id'),
        State({'type':'projects-dropdown', 'page': '/build-agent'}, 'value'),
        State({'type':'callback', 'page':'/build-agent'}, 'value'),
        State({'type': 'env-dropdown', 'page': '/build-agent'}, 'value')],
        prevent_initial_call=True,
)
    def build_agent_model(n_clicks, all_values, all_ids, agent_type_dropdown_value, layer_index_values, layer_index_ids, project, callbacks, env_selection):
        if n_clicks is None or n_clicks < 1:
            raise PreventUpdate

        # Set environment parameter
        env = gym.make(env_selection)

        # set params if agent is reinforce or actor critic
        if agent_type_dropdown_value == "Reinforce" or agent_type_dropdown_value == "ActorCritic":
            # set defualt gym environment in order to build policy and value models and save
            # env = gym.make("CartPole-v1")

            learning_rate=10**utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        value_type='learning-rate',
                        value_model='none',
                        agent_type=agent_type_dropdown_value,
                    )

            policy_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='optimizer',
                    value_model='policy',
                    agent_type=agent_type_dropdown_value,
                )
            
            policy_initializer = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='policy',
                agent_type=agent_type_dropdown_value
            )

            policy_layers = models.build_layers(
                utils.format_layers(
                    all_values=all_values,
                    all_ids=all_ids,
                    layer_units_values=layer_index_values,
                    layer_units_ids=layer_index_ids,
                    value_type='layer-units',
                    value_model='policy',
                    agent_type=agent_type_dropdown_value,
                ),
                utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='activation-function',
                    value_model='policy',
                    agent_type=agent_type_dropdown_value,
                ),
                policy_initializer,
            )

            ##DEBUG
            # print("Policy layers:", policy_layers)

            policy_model = models.PolicyModel(
                    env=gym.make(env),
                    dense_layers=policy_layers,
                    optimizer=policy_optimizer,
                    learning_rate=learning_rate,
                )

            value_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='optimizer',
                    value_model='value',
                    agent_type=agent_type_dropdown_value,
                )

            value_initializer = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='value',
                agent_type=agent_type_dropdown_value
            )
            
            value_layers = models.build_layers(
                utils.format_layers(
                    all_values=all_values,
                    all_ids=all_ids,
                    layer_units_values=layer_index_values,
                    layer_units_ids=layer_index_ids,
                    value_type='layer-units',
                    value_model='value',
                    agent_type=agent_type_dropdown_value,
                ),
                utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='activation-function',
                    value_model='value',
                    agent_type=agent_type_dropdown_value,
                ),
                value_initializer,
            )

            ##DEBUG
            # print("Value layers:", value_layers)

            
            value_model = models.ValueModel(
                env=gym.make(env),
                dense_layers=value_layers,
                optimizer=value_optimizer,
                learning_rate=learning_rate,
            )

            
            if agent_type_dropdown_value == "Reinforce":

                discount=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        value_type='discount',
                        value_model='none',
                        agent_type=agent_type_dropdown_value,
                    ),

                
                agent = rl_agents.Reinforce(
                    env=gym.make(env),
                    policy_model=policy_model,
                    value_model=value_model,
                    discount=discount,
                    callbacks=utils.get_callbacks(callbacks, project),
                    save_dir=os.path.join(os.getcwd(), 'assets'),
                )

            elif agent_type_dropdown_value == "Actor Critic":

                discount=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        value_type='discount',
                        value_model='none',
                        agent_type=agent_type_dropdown_value,
                    ),

                policy_trace_decay=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        value_type='trace-decay',
                        value_model='policy',
                        agent_type=agent_type_dropdown_value,
                    )
                value_trace_decay=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        value_type='trace-decay',
                        value_model='value',
                        agent_type=agent_type_dropdown_value,
                    )

                agent = rl_agents.ActorCritic(
                    env=gym.make(env),
                    policy_model=policy_model,
                    value_model=value_model,
                    discount=discount,
                    policy_trace_decay=policy_trace_decay,
                    value_trace_decay=value_trace_decay,
                    callbacks=utils.get_callbacks(callbacks, project),
                    save_dir=os.path.join(os.getcwd(), 'assets'),
                )

        elif agent_type_dropdown_value == "DDPG":
            # set defualt gym environment in order to build policy and value models and save
            # env = gym.make("Pendulum-v1")

            # Set actor params
            
            # Set actor learning rate
            actor_learning_rate=10**utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='learning-rate',
                    value_model='actor',
                    agent_type=agent_type_dropdown_value,
                )

            actor_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='optimizer',
                    value_model='actor',
                    agent_type=agent_type_dropdown_value,
                )

            actor_initializer = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor',
                agent_type=agent_type_dropdown_value
            )

            actor_conv_layers = utils.format_cnn_layers(
                all_values,
                all_ids,
                layer_index_values,
                layer_index_ids,
                'actor',
                agent_type_dropdown_value
            )

            if actor_conv_layers:
                actor_cnn = cnn_models.CNN(actor_conv_layers, env)


            actor_dense_layers = models.build_layers(
                utils.format_layers(
                    all_values=all_values,
                    all_ids=all_ids,
                    layer_units_values=layer_index_values,
                    layer_units_ids=layer_index_ids,
                    value_type='layer-units',
                    value_model='actor',
                    agent_type=agent_type_dropdown_value,
                ),
                utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='activation-function',
                    value_model='actor',
                    agent_type=agent_type_dropdown_value,
                ),
                actor_initializer,
            )

            # Create actor model
            actor_model = models.ActorModel(
                env=env,
                cnn_model=actor_cnn,
                dense_layers=actor_dense_layers,
                learning_rate=actor_learning_rate,
                optimizer=actor_optimizer
            )
            
            #DEBUG
            # print(f'actor cnn model: {actor_cnn}')
            # print(f'actor dense layers: {actor_dense_layers}')
            # print(f'actor optimizer: {actor_optimizer}')
            # print(f'actor learning rate: {actor_learning_rate}')
            # print(f'actor model: {actor_model}')
            
            # Set critic params

            critic_learning_rate=10**utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='learning-rate',
                    value_model='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='optimizer',
                    value_model='critic',
                    agent_type=agent_type_dropdown_value,
                )

            critic_initializer = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic',
                agent_type=agent_type_dropdown_value
            )
            
            critic_state_layers = models.build_layers(
                utils.format_layers(
                    all_values=all_values,
                    all_ids=all_ids,
                    layer_units_values=layer_index_values,
                    layer_units_ids=layer_index_ids,
                    value_type='layer-units',
                    value_model='critic-state',
                    agent_type=agent_type_dropdown_value,
                ),
                utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='activation-function',
                    value_model='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_initializer,
            )
           
            critic_conv_layers = utils.format_cnn_layers(
                all_values,
                all_ids,
                layer_index_values,
                layer_index_ids,
                'critic',
                agent_type_dropdown_value
            )

            if critic_conv_layers:
                critic_cnn = cnn_models.CNN(critic_conv_layers, env)

            critic_merged_layers = models.build_layers(
                utils.format_layers(
                    all_values=all_values,
                    all_ids=all_ids,
                    layer_units_values=layer_index_values,
                    layer_units_ids=layer_index_ids,
                    value_type='layer-units',
                    value_model='critic-merged',
                    agent_type=agent_type_dropdown_value,
                ),
                utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='activation-function',
                    value_model='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_initializer,
            )
           
           
            critic_model = models.CriticModel(
                env=env,
                cnn_model=critic_cnn,
                state_layers=critic_state_layers,
                merged_layers=critic_merged_layers,
                learning_rate=critic_learning_rate,
                optimizer=critic_optimizer
            )

            #DEBUG
            # print(f'critic cnn model: {critic_cnn}')
            # print(f'critic state layers: {critic_state_layers}')
            # print(f'critic merged layers: {critic_merged_layers}')
            # print(f'critic optimizer: {critic_optimizer}')
            # print(f'critic learning rate: {critic_learning_rate}')
            # print(f'critic model: {critic_model}')


            # Set DDPG params

            discount=utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='discount',
                    value_model='none',
                    agent_type=agent_type_dropdown_value,
                )
            
            tau=utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='tau',
                    value_model='none',
                    agent_type=agent_type_dropdown_value,
                )
            
            batch_size=utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    value_type='batch-size',
                    value_model='none',
                    agent_type=agent_type_dropdown_value,
                )
            
            noise=utils.create_noise_object(
                    env=env,
                    all_values=all_values,
                    all_ids=all_ids,
                    agent_type=agent_type_dropdown_value,
                )
            
            agent = rl_agents.DDPG(
                env=env,
                actor_model=actor_model,
                critic_model=critic_model,
                discount=discount,
                tau=tau,
                replay_buffer=helper.ReplayBuffer(env, 100000),
                batch_size=batch_size,
                noise=noise,
                callbacks=utils.get_callbacks(callbacks, project),
                save_dir=os.path.join(os.getcwd(), 'assets'),
            )
            
        # save agent
        agent.save()

        # source_dir = os.path.join(os.getcwd(), 'assets')  # Directory containing the agent's files
        # output_zip = os.path.join('assets', 'agent_model.zip')  # Path to save the ZIP archive
        # # Package the agent files into a ZIP archive
        # utils.zip_agent_files(source_dir, output_zip)


        return html.Div([
        dbc.Alert("Agent built successfully!", color="success")
        # html.P("Agent built successfully."),
        # html.A('Download Agent Model', href='/assets/agent_model.zip', download='agent_model.zip'),
        ])
    

    @app.callback(
        Output({'type':'wandb-login-container', 'page':'/build-agent'}, 'children'),
        Input({'type':'callback', 'page':'/build-agent'}, 'value'),
        # State({'type':'callback', 'page':'/build-agent'}, 'id'),
        # prevent_initial_call=True,
)
    def show_wandb_login(callbacks):
        if callbacks is not None:
            # page = callback_id['page']
            # Flatten the list if it's a list of lists, which can happen with ALL
            flattened_values = [item for sublist in callbacks for item in (sublist if isinstance(sublist, list) else [sublist])]
            if 'Weights & Biases' in flattened_values:
                return utils.generate_wandb_login('/build-agent')
            else:
                return html.Div()
        
    @app.callback(
        Output({'type':'wandb-login-feedback', 'page':MATCH}, 'children'),
        Input({'type':'wandb-login', 'page':MATCH}, 'n_clicks'),
        State({'type':'wandb-api-key', 'page':MATCH}, 'value'),
        State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def login_to_wandb(n_clicks, api_key, page):
        if not n_clicks or n_clicks < 1:
            raise PreventUpdate

        # Attempt to log in to Weights & Biases
        try:
            wandb.login(key=api_key)
            return dbc.Alert("Login successful!", color="success", is_open=True,
                             id={'type':'wandb-login-success', 'page':page})
        
        # html.Div(
        #             "Logged in to Weights & Biases successfully.", 
        #             style={'color': 'green'},
        #             id={'type':'wandb-login-success', 'page':page},
        #         ),
                
        except (ValueError, ConnectionError) as e:
            return html.Div(f"Failed to log in to Weights & Biases: {e}", style={'color': 'red'})
        
    @app.callback(
        Output({'type': 'hyperparameter-inputs', 'page':MATCH}, 'hidden'),
        Input({'type': 'wandb-login-success', 'page':MATCH}, 'is_open'),
        State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def show_hyperparam_inputs(wandb_login, page):
            # print(wandb_login)
            return not wandb_login

    @app.callback(
        Output({'type':'project-selection-container', 'page':MATCH}, 'children'),
        Input({'type':'wandb-login-success', 'page':MATCH}, 'is_open'),
        State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def show_project_selection(wandb_login, page):
        
        if wandb_login:
            return utils.generate_wandb_project_dropdown(page)
    
    @app.callback(
        [Output({'type':'env-description', 'page':MATCH}, 'children'),
        Output({'type':'env-gif', 'page':MATCH}, 'src'),
        Output({'type':'gym-params', 'page':MATCH}, 'children')],
        [Input({'type':'env-dropdown', 'page':MATCH}, 'value')],
        State({'type':'env-dropdown', 'page':MATCH}, 'id'),
        prevent_initial_call=True,
    )
    def update_env_info(env_name, id):
        if env_name:
            env_data = {
        "CartPole-v0": {
            "description": "Balance a pole on a cart; the goal is to prevent it from falling.",
            "gif_url": "https://gymnasium.farama.org/_images/cart_pole.gif",
        },
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
        "Pusher-v4": {
            "description": "A robot arm needs to push objects to a target location.",
            "gif_url": "https://gymnasium.farama.org/_images/pusher.gif",
        },
        "InvertedPendulum-v4": {
            "description": "Balance a pendulum in the upright position on a moving cart",
            "gif_url": "https://gymnasium.farama.org/_images/inverted_pendulum.gif",
        },
        "InvertedDoublePendulum-v4": {
            "description": "A more complex version of the InvertedPendulum with two pendulums to balance.",
            "gif_url": "https://gymnasium.farama.org/_images/inverted_double_pendulum.gif",
        },
        "HalfCheetah-v4": {
            "description": "Control a 2D cheetah robot to make it run as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/half_cheetah.gif",
        },
        "Hopper-v4": {
            "description": "Make a two-dimensional one-legged robot hop forward as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/hopper.gif",
        },
        "Swimmer-v4": {
            "description": "Control a snake-like robot to make it swim through water.",
            "gif_url": "https://gymnasium.farama.org/_images/swimmer.gif",
        },
        "Walker2d-v4": {
            "description": "A bipedal robot walking simulation aiming to move forward as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/walker2d.gif",
        },
        "Ant-v4": {
            "description": "Control a four-legged robot to explore a terrain.",
            "gif_url": "https://gymnasium.farama.org/_images/ant.gif",
        },
        "Humanoid-v4": {
            "description": "A two-legged humanoid robot that learns to walk and balance.",
            "gif_url": "https://gymnasium.farama.org/_images/humanoid.gif",
        },
        "HumanoidStandup-v4": {
            "description": "The goal is to make a humanoid stand up from a prone position.",
            "gif_url": "https://gymnasium.farama.org/_images/humanoid_standup.gif",
        },
    }
            description = env_data[env_name]['description']
            gif_url = env_data[env_name]['gif_url']
            gym_params = {}
            if id['page'] != "/build-agent":
                gym_params = utils.generate_gym_extra_params_container(env_name)
            return description, gif_url, gym_params
        return "", "", gym_params  # Default empty state

    @app.callback(
        Output({'type':'hidden-div', 'page':MATCH}, 'children'),
        Input({'type':'start', 'page':MATCH}, 'n_clicks'),
        State({'type':'start', 'page':MATCH}, 'id'),
        State('agent-store', 'data'),
        State({'type':'storage', 'page':MATCH}, 'data'),
        State({'type':'env-dropdown', 'page':MATCH}, 'value'),
        State({'type':'num-episodes', 'page':MATCH}, 'value'),
        State({'type':'render-option', 'page':MATCH}, 'value'),
        State({'type':'render-freq', 'page':MATCH}, 'value'),
    )
    def start_agent(n_clicks, id, agent_data, storage_data, env_name, num_episodes, render_option, render_freq):
        #DEBUG
        # print("Start callback called.")
        if n_clicks > 0:
            # Use the agent_data['save_dir'] to load your agent
            if agent_data:  # Check if agent_data is not empty
                # save_dir = agent_data['save_dir']
                #DEBUG
                # print(f"agent data: {agent_data}")
                # task = train_model_task.delay(save_dir, num_episodes, render_option, render_freq)
                render = 'RENDER' in render_option
                # set target for thread (train/test) dependent on page id
                if id['page'] == '/train-agent':
                    target = utils.train_model
                    #DEBUG
                    # print("Training target set")
                elif id['page'] == '/test-agent':
                    target = utils.test_model
                    #DEBUG
                    # print("Testing target set")
                else:
                    raise ValueError("Invalid page")

                thread = threading.Thread(target=target, args=(agent_data, env_name, num_episodes, render, render_freq))
                #DEBUG
                # print("thread set")
                thread.daemon = True  # This ensures the thread will be automatically cleaned up when the main process exits
                thread.start()
                #DEBUG
                # print("thread started")
      
        raise PreventUpdate
            
    
    @app.callback(
        Output('agent-store', 'data'),
        Output('output-agent-load', 'children'),
        Input({'type':'upload-agent-config', 'page':ALL}, 'contents'),
        prevent_initial_call=True,
)
    def store_agent(contents):
        # Uploads and stores the agent's obj_config.json to reference to load
        # into memory when required
        for content in contents:
            #DEBUG
            # print(f'content: {content}')
            if content is not None:
                # Split the content into metadata and the base64 encoded string
                _, encoded = content.split(',')
                decoded = base64.b64decode(encoded)           
                config = json.loads(decoded.decode('utf-8'))
                #DEBUG
                # print(f'config: {config}')
                # save_dir = config.get('save_dir')
                
                # Here, instead of loading the agent, save its config or directory to dcc.Store
                return config, html.Div([
                    "Config loaded successfully from: ", html.Code(config['save_dir'])
                ])
        return {}, "Please upload a file."
    

    @app.callback(
    Output({'type':'render-freq', 'page':MATCH}, 'disabled'),  
    Input({'type':'render-option', 'page':MATCH}, 'value')
)
    def toggle_render_freq(render_option):
        if 'RENDER' in render_option:
            return False
        return True
    

    @app.callback(
    Output({'type':'storage', 'page':MATCH}, 'data'),
    [Input({'type':'interval-component', 'page':MATCH}, 'n_intervals')],
    [State({'type':'storage', 'page':MATCH}, 'data'),
     State({'type':'num-episodes', 'page':MATCH}, 'value'),
     State('url', 'pathname')]
)
    def update_data(n, storage_data, num_episodes, pathname):
        if num_episodes is not None:
            if pathname == '/train-agent':
                file_name = 'training_data.json'
                process = 'Training'
            elif pathname == '/test-agent':
                file_name = 'testing_data.json'
                process = 'Testing'
            else:
                # If the pathname does not match expected values, prevent update
                raise PreventUpdate
            # Read the latest training progress data from the JSON file
            try:
                #DEBUG
                # print('update data callback called...')
                with open(Path("assets") / file_name, 'r') as f:
                    data = json.load(f)
                    # print(f"Updating {process} data")
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}  # Use an empty dict if there's an issue reading the file
            
            #DEBUG
            # print(f'current data: {storage_data}')
            # print(f'new data: {data}')

            # if the new data dict isn't empty, update storage data
            if data != {}:
                #DEBUG
                # print('updating data...')
                storage_data['data'] = data
                storage_data['progress'] = round(data['episode']/num_episodes, ndigits=2)
                if storage_data['progress'] == 1.0:
                    storage_data['status'] = f"{process} Completed"
                else:
                    storage_data['status'] = f"{process} in Progress..."
        
        return storage_data
    
    
    @app.callback(
        Output({'type':'status', 'page':MATCH}, 'children'),
        Input({'type':'interval-component', 'page':MATCH}, 'n_intervals'),
        State({'type':'storage', 'page':MATCH}, 'data'),
        State({'type':'start', 'page':MATCH}, 'n_clicks'),
    )
    def update_status_and_progress(n, storage_data, start_clicks):
        if start_clicks > 0:
            # Extract status message
            status_message = storage_data.get('status', "Status not found.")

            # Extract progress and calculate percentage
            progress = storage_data.get('progress', 0) * 100  # Assuming progress is a fraction

            # Format data metrics for display
            data_metrics = storage_data.get('data', {})
            metrics_display = html.Ul([html.Li(f"{key}: {value}") for key, value in data_metrics.items()])

            # Create progress bar component
            progress_bar = dbc.Progress(value=progress, max=100, striped=True, animated=True, style={"margin": "20px 0"})

            # Combine all components to be displayed in a single div
            combined_display = html.Div([
                html.P(status_message),
                metrics_display,
                progress_bar
            ])

            return combined_display
        
        PreventUpdate
    
    # @app.callback(
    #     Output({'type':'video-carousel-store', 'page':MATCH}, 'data'),
    #     [Input({'type':'interval-component', 'page':MATCH}, 'n_intervals')],
    #     State({'type':'interval-component', 'page':MATCH}, 'id'),
    #     State({'type':'start', 'page':MATCH}, 'n_clicks'),
    #     State({'type':'storage', 'page':MATCH}, 'data'),
    # )
    # def update_video_carousel(n_intervals, interval_id, start_clicks, video_storage):
    #     if start_clicks > 0:
    #         video_filenames = utils.get_video_files(interval_id['page'])
    #         #DEBUG
    #         # print(f'video filenames: {video_filenames}')
    #         # check if video_filenames is not empty
    #         video_storage['video_list'] = video_filenames
            
    #         return video_storage
    
    #     PreventUpdate
    @app.callback(
        Output({'type': 'video-carousel-store', 'page': MATCH}, 'data'),
        Output({'type': 'video-carousel', 'page': MATCH}, 'children'),
        Output({'type': 'video-filename', 'page': MATCH}, 'children'),
        [Input({'type': 'prev-btn', 'page': MATCH}, 'n_clicks'),
        Input({'type': 'next-btn', 'page': MATCH}, 'n_clicks'),
        Input({'type': 'interval-component', 'page': MATCH}, 'n_intervals')],
        [State({'type': 'video-carousel-store', 'page': MATCH}, 'data'),
        State({'type': 'interval-component', 'page': MATCH}, 'id'),
        State({'type': 'start', 'page': MATCH}, 'n_clicks')],
        State({'type': 'video-carousel', 'page': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_video_data(prev_clicks, next_clicks, n_intervals, data, interval_id, start_clicks, carousel_id):
        # check if start has been clicked before trying to update        
        ctx = dash.callback_context
        triggered_id, prop = ctx.triggered[0]['prop_id'].split('.')
        #DEBUG
        # print(f'trigger id: {triggered_id}')

        # Initial load or automatic update triggered by the interval component
        if 'interval-component' in triggered_id and start_clicks > 0:
            video_filenames = utils.get_video_files(interval_id['page'])
            if video_filenames:
                data['video_list'] = video_filenames
                # Optionally reset current video to 0 or keep it as is
                # data['current_video'] = 0
                # video_files = data['video_list']
                current_video = data.get('current_video', 0)
                current_filename = video_filenames[current_video]
                video_item = utils.generate_video_items([current_filename], carousel_id['page'])[0]
                
                #DEBUG
                # print(f'data: {data}')
                return data, video_item, current_filename
            else:
                raise PreventUpdate

        # Navigation button clicks
        elif 'prev-btn' in triggered_id or 'next-btn' in triggered_id:
            current_video = data.get('current_video', 0)
            video_list = data.get('video_list', [])
            #DEBUG
            # print(f'current video: {current_video}')
            # print(f'video_list: {video_list}')
            
            if 'prev-btn' in triggered_id:
                current_video = max(0, current_video - 1)
            elif 'next-btn' in triggered_id:
                current_video = min(len(video_list) - 1, current_video + 1)

            data['current_video'] = current_video
            #DEBUG
            # print(f'current video: {current_video}')

            video_files = data['video_list']
            current_filename = video_files[current_video]
            video_item = utils.generate_video_items([current_filename], carousel_id['page'])[0]

            return data, video_item, current_filename

        raise PreventUpdate

    # Callback to update video display based on navigation
    # @app.callback(
    #     Output({'type': 'video-carousel', 'page': MATCH}, 'children'),
    #     Output({'type': 'video-filename', 'page': MATCH}, 'children'),
    #     [Input({'type': 'prev-btn', 'page': MATCH}, 'n_clicks'),
    #     Input({'type': 'next-btn', 'page': MATCH}, 'n_clicks')],
    #     [State({'type': 'video-carousel-store', 'page': MATCH}, 'data'),
    #     State({'type': 'video-carousel', 'page': MATCH}, 'id')],
    #     prevent_initial_call=True
    # )
    # def update_video(n_clicks_prev, n_clicks_next, video_storage, id):
    #     ctx = dash.callback_context

    #     if not ctx.triggered:
    #         button_id = 'No clicks yet'
    #     else:
    #         button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    #     video_files = video_storage['video_list']
    #     current_video = video_storage['current_video']
    #     current_filename = video_files[current_video]

        # if 'prev-btn' in button_id:
        #     current_video = max(0, current_video - 1)
        # elif 'next-btn' in button_id:
        #     current_video = min(len(video_files) - 1, current_video + 1)

        # video_storage['current_video']=current_video
        # video_item = utils.generate_video_items([video_files[current_video]], id['page'])[0]

        # return video_item, current_filename

    # Callback to enable/disable navigation buttons
    @app.callback(
        [Output({'type': 'prev-btn', 'page': MATCH}, 'disabled'),
        Output({'type': 'next-btn', 'page': MATCH}, 'disabled')],
        [Input({'type': 'video-carousel-store', 'page': MATCH}, 'data')],
        [State({'type': 'video-carousel', 'page': MATCH}, 'id')]
    )
    def toggle_navigation_buttons(data, id):
        video_files = data['video_list']
        current_video = data['current_video']

        prev_disabled = current_video <= 0
        next_disabled = current_video >= len(video_files) - 1

        return prev_disabled, next_disabled
        
        ## HYPERPARAMETER SEARCH CALLBACKS
    
    @app.callback(
        Output('agent-options-tabs', 'children'),
        Input('agent-type-selector', 'value')
    )
    def update_hyperparam_inputs(selected_agent_types):
        # This function updates the inputs based on selected agent types
        tabs = []
        for agent_type in selected_agent_types:
            if agent_type == 'Reinforce':
                tabs.append(
                    dcc.Tab([
                        html.Div([
                            # utils.generate_reinforce_hyperparam_component(),
                            # html.H3('Reinforce Hyperparameters'),
                            utils.generate_learning_rate_hyperparam_component(agent_type, 'none'),
                            utils.generate_discount_hyperparam_component(agent_type, 'none'),
                            dcc.Tabs([
                                dcc.Tab([
                                    # html.H4("Policy Model Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'policy'),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'policy'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'policy'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'policy'),
                                ],
                                label="Policy Model"),
                                dcc.Tab([
                                    # html.H4("Value Model Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'value'),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'value'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'value'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'value'),
                                ],
                                label="Value Model"),
                            ])
                        ])
                    ],
                    label=agent_type)
                )
            elif agent_type == 'ActorCritic':
                tabs.append(
                    dcc.Tab([
                        html.Div([
                            # utils.generate_actor_critic_hyperparam_component(),
                            # html.H3('Actor Critic Hyperparameters'),
                            utils.generate_learning_rate_hyperparam_component(agent_type, 'none'),
                            utils.generate_discount_hyperparam_component(agent_type, 'none'),
                            dcc.Tabs([
                                dcc.Tab([
                                    # html.H4("Policy Model Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'policy'),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'policy'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'policy'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'policy'),
                                    utils.generate_trace_decay_hyperparam_componenent(agent_type, 'policy'),
                                ],
                                label="Policy Model"),
                                dcc.Tab([
                                    # html.H4("Value Model Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'value'),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'value'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'value'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'value'),
                                    utils.generate_trace_decay_hyperparam_componenent(agent_type, 'value'),
                                ],
                                label="Value Model")
                            ])
                        ])
                    ],
                    label=agent_type)
                )
            elif agent_type == 'DDPG':
                tabs.append(
                    dcc.Tab([
                        html.Div([
                            # utils.generate_actor_critic_hyperparam_component(),
                            # html.H3('DDPG Hyperparameters'),
                            utils.generate_batch_hyperparam_componenent(agent_type, 'none'),
                            # utils.generate_replay_buffer_hyperparam_component(agent_type, 'none'), # under development
                            utils.generate_noise_hyperparam_componenent(agent_type, 'none'),
                            html.Hr(),
                            utils.generate_tau_hyperparam_componenent(agent_type, 'none'),
                            utils.generate_discount_hyperparam_component(agent_type, 'none'),
                            dcc.Tabs([
                                dcc.Tab([
                                    # html.H4("Actor Model Configuration"),
                                    utils.generate_learning_rate_hyperparam_component(agent_type, 'actor'),
                                    utils.generate_cnn_layer_hyperparam_component(agent_type, 'actor'),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'actor'),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'actor'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'actor'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'actor'),
                                ],
                                label='Actor Model'),
                                dcc.Tab([
                                    # html.H4("Critic Model Configuration"),
                                    utils.generate_learning_rate_hyperparam_component(agent_type, 'critic'),
                                    html.Hr(),
                                    utils.generate_cnn_layer_hyperparam_component(agent_type, 'critic'),
                                    html.H4("Critic State Input Layer Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'critic-state'),
                                    html.Hr(),
                                    html.H4("Critic Merged (State + Action) Input Layer Configuration"),
                                    utils.generate_hidden_layer_hyperparam_component(agent_type, 'critic-merged'),
                                    html.Hr(),
                                    utils.generate_kernel_initializer_hyperparam_component(agent_type, 'critic'),
                                    html.Hr(),
                                    utils.generate_activation_function_hyperparam_component(agent_type, 'critic'),
                                    utils.generate_optimizer_hyperparam_component(agent_type, 'critic'),
                                ],
                                label='Critic Model')
                            ])
                        ])
                    ],
                    label=agent_type)
                )
        
        return tabs
    

    @app.callback(
        Output({'type': 'hyperparam-units-per-layer', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'hidden-layers-slider', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'hidden-layers-slider', 'model': MATCH, 'agent': MATCH}, 'id'),
    )
    def update_hyperparam_units_per_layer_inputs(num_layers, id):
        if num_layers is not None:
            model_type = id['model']
            agent_type = id['agent']
            inputs = []
            for i in range(1, num_layers[1] + 1):
                inputs.append(
                    utils.generate_hidden_units_per_layer_hyperparam_component(agent_type, model_type, i)
                )

            return inputs
        

    @app.callback(
        Output({'type': 'cnn-layer-types-hyperparam', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'cnn-layers-slider-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'cnn-layers-slider-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
    )
    def update_hyperparam_units_types_inputs(num_layers, layer_id):
        if num_layers is not None:
            model_type = layer_id['model']
            agent_type = layer_id['agent']
            layer_types = []
            for i in range(1, num_layers[1] + 1):
                layer_types.append(
                    utils.generate_cnn_layer_type_hyperparam_component(agent_type, model_type, i)
                )

            return layer_types
        

    @app.callback(
    Output({'type': 'cnn-layer-type-parameters-hyperparam', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'children'),
    Input({'type': 'cnn-layer-type-hyperparam', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'value'),
    State({'type': 'cnn-layer-type-hyperparam', 'model': MATCH, 'agent': MATCH, 'index': MATCH}, 'id'),
    prevent_initial_call=True
    )
    def update_layer_type_params_hyperparams(layer_types, layer_id):
        if layer_types is not None:
            layer_params = []
            for layer_type in layer_types:
                model = layer_id['model']
                agent = layer_id['agent']
                index = layer_id['index']

                layer_params.append(utils.generate_cnn_layer_parameters_hyperparam_component(layer_type, agent, model, index))
            
        return layer_params

 
    @app.callback(
        Output({'type': 'kernel-options-tabs', 'model': MATCH, 'agent': MATCH}, 'children'),
        Output({'type': 'kernel-options-header' , 'model': MATCH, 'agent': MATCH}, 'hidden'),
        Input({'type': 'kernel-function-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'kernel-function-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_kernel_hyperparam_options(kernel_functions, id):
        model_type = id['model']
        agent_type = id['agent']

        tabs = utils.generate_kernel_options_hyperparam_component(agent_type, model_type, kernel_functions)

        # Hide header if no kernel options
        hide_header = True if not kernel_functions else False

        return tabs, hide_header
    
    @app.callback(
        Output({'type': 'noise-options-tabs', 'model': MATCH, 'agent': MATCH}, 'children'),
        Output({'type': 'noise-options-header' , 'model': MATCH, 'agent': MATCH}, 'hidden'),
        Input({'type': 'noise-function-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'noise-function-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_noise_hyperparam_options(noise_functions, id):
        model_type = id['model']
        agent_type = id['agent']

        tabs = utils.generate_noise_options_hyperparams_component(agent_type, model_type, noise_functions)

        # Hide header if no kernel options
        hide_header = True if not noise_functions else False

        return tabs, hide_header

    @app.callback(
        Output({'type':'hidden-div-hyperparam', 'page':'/hyperparameter-search'}, 'children'),
        Input({'type':'start', 'page':'/hyperparameter-search'}, 'n_clicks'),
        State({'type':'storage', 'page':'/hyperparameter-search'}, 'data'),
        State('search-type', 'value'),
        State({'type': 'projects-dropdown', 'page': '/hyperparameter-search'}, 'value'),
        State('sweep-name', 'value'),
        State('goal-metric', 'value'),
        State('goal-type', 'value'),
        State({'type': 'env-dropdown', 'page': '/hyperparameter-search'}, 'value'),
        State({'type':'gym-params', 'page':'/hyperparameter-search'}, 'children'),
        State('agent-type-selector', 'value'),
        State('num-sweeps', 'value'),
        State('num-episodes', 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'id'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'id'),
        prevent_initial_call=True
    )
    def begin_sweep(num_clicks, data, method, project, sweep_name, metric_name, metric_goal, env, env_params, agent_selection, num_sweeps, num_episodes, all_values, all_ids, all_indexed_values, all_indexed_ids):

        # extract any additional gym env params
        params = utils.extract_gym_params(env_params)

        if num_clicks > 0:
            sweep_config = utils.create_wandb_config(
                method,
                project,
                sweep_name,
                metric_name,
                metric_goal,
                env,
                params,
                agent_selection,
                all_values,
                all_ids,
                all_indexed_values,
                all_indexed_ids
            )
            #DEBUG
            print(f'wandb config: {sweep_config}')
            thread = threading.Thread(target=wandb_support.hyperparameter_sweep, args=(sweep_config, num_sweeps, num_episodes, os.path.join(os.getcwd(), 'assets')))
            #DEBUG
            # print("thread set")
            thread.daemon = True  # This ensures the thread will be automatically cleaned up when the main process exits
            thread.start()
            #DEBUG
            # print("thread started")


    @app.callback(
        Output('hyperparameter-selector', 'options'),
        Input('update-hyperparam-selector', 'n_intervals'),
        State({'type':'start', 'page':'/hyperparameter-search'}, 'n_clicks'),
        State({'type':'projects-dropdown', 'page':'/hyperparameter-search'}, 'value'),
        State('sweep-name', 'value'),
        prevent_initial_call=True,
    )
    def update_hyperparameter_list(n_intervals, num_clicks, project_name, sweep_id):
        if num_clicks > 0:
            # print('update hyperparameter list')
            hyperparameters = wandb_support.fetch_sweep_hyperparameters_single_run(project_name, sweep_id)
            # print(f'retrieved params: {hyperparameters}')
            # print('')
            # task = tasks.test_task.delay()
            return [{'label': hp, 'value': hp} for hp in hyperparameters]
        else:
            return []
        
            
    @app.callback(
        Output('heatmap-data-store', 'data'),
        Input('heatmap-store-data-interval', 'n_intervals'),
        State('heatmap-data-store', 'data'),
    )
    def update_heatmap_data(n, data):
        if 'matrix_data' not in shared_data:
            raise PreventUpdate
        matrix_data = shared_data.get('matrix_data')
        bin_ranges = shared_data.get('bin_ranges')
        if matrix_data:
            # print(f'heatmap data: {data}')
            new_data = {'matrix_data': matrix_data, 'bin_ranges': bin_ranges}
            return new_data
        else:
            return None
        
    
    @app.callback(
        Output('heatmap-container', 'children'),
        Output('legend-container' , 'children'),
        Input('heatmap-data-store', 'data')
    )
    def update_heatmap_container(data):
        heatmap, bar_chart = utils.update_heatmap(data)
        if heatmap is None:
            return dash.no_update
        return dcc.Graph(figure=heatmap), dcc.Graph(figure=bar_chart)
    
    @app.callback(
        Output('heatmap-placeholder', 'style'),
        Input('heatmap-container', 'children')
    )
    def toggle_placeholder(heatmap):
        if heatmap is None:
            return {'display': 'block'}
        return {'display': 'none'}
        
    @app.callback(
        Output('hidden-div-fetch-process', 'children'),
        Input({'type':'start', 'page':'/hyperparameter-search'}, 'n_clicks'),
        State({'type': 'projects-dropdown', 'page': '/hyperparameter-search'}, 'value'),
        State('sweep-name', 'value'),
    )
    def start_data_fetch_processes(n_clicks, project, sweep_name):
        if n_clicks > 0:
            # Create and start the fetch_data_process
            # print('start data fetch process called')
            fetch_data_thread = threading.Thread(target=fetch_data_process, args=(project, sweep_name, shared_data))
            fetch_data_thread.start()

        return None

    @app.callback(
        Output('hidden-div-matrix-process', 'children'),
        Input('start-matrix-process-interval', 'n_intervals'),
        State('hyperparameter-selector', 'value'),
        State('bin-slider', 'value'),
        State('z-score-checkbox', 'value'),
        State('reward-threshold', 'value'),
        State({'type':'start', 'page':'/hyperparameter-search'}, 'n_clicks'),
    )
    def start_matrix_process(n, hyperparameters, bins, zscore_option, reward_threshold, n_clicks):
    # Create and start the update_heatmap_process
        if n_clicks > 0:
            z_score = 'zscore' in zscore_option
            # print('start matrix process callback called')
            update_heatmap_thread = threading.Thread(target=update_heatmap_process, args=(shared_data, hyperparameters, bins, z_score, reward_threshold))
            update_heatmap_thread.start()
        
        return None

