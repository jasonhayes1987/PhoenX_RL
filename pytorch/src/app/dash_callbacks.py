import os
from pathlib import Path
import subprocess
import multiprocessing
from multiprocessing import Queue
from queue import Empty
import threading
import time
import json
# import logging
from logging_config import logger
import base64
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import io
import ast

import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.tools as tls
import plotly.express as px



import gymnasium as gym
import gymnasium_robotics as gym_robo
import wandb
# import tensorflow as tf
import pandas as pd

import layouts
import helper
import gym_helper
import utils
import models
from models import StochasticDiscretePolicy, StochasticContinuousPolicy, ValueModel, CriticModel, ActorModel
import cnn_models
import rl_agents
import wandb_support


# Create a queue to store the formatted data
formatted_data_queue = Queue()

def fetch_data_process(project, sweep_name, shared_data):
    while True:
        try:
            # print("Fetching data from wandb...")
            metrics_data = wandb_support.get_metrics(project, sweep_name)
            logger.debug(f'fetch_data_process metrics data:{metrics_data}')
            formatted_data = wandb_support.format_metrics(metrics_data)
            logger.debug(f'fetch_data_process formatted data:{formatted_data}')
            shared_data['formatted_data'] = formatted_data
            logger.debug(f"fetch_data_process shared_data[formatted_data]:{shared_data['formatted_data']}")
            time.sleep(10)  # Wait before fetching data again
        except Exception as e:
            logger.error(f"Error in fetch_data_process: {str(e)}", exc_info=True)
            time.sleep(5)


def update_heatmap_process(shared_data, hyperparameters, bins, z_score, reward_threshold):
    print(f'update heatmap process shared data: {shared_data}')
    try:
        if 'formatted_data' in shared_data:
            #DEBUG
            formatted_data = shared_data['formatted_data']
            logger.debug(f'update_heatmap_process: formatted_data:{formatted_data}')
            # Convert the JSON string back to a pandas DataFrame
            # formatted_data = pd.read_json(data, orient='split')
            # print(f'formatted data passed to wandb_support: {formatted_data}')
            matrix_data, bin_ranges = wandb_support.calculate_co_occurrence_matrix(formatted_data, hyperparameters, reward_threshold, bins, z_score)
            logger.debug(f'update_heatmap_process: matrix_data:{matrix_data}')
            logger.debug(f'update_heatmap_process: bin_ranges:{bin_ranges}')
            shared_data['matrix_data'] = matrix_data.to_dict(orient='split')
            logger.debug(f"update_heatmap_process: shared_data[matrix_data]:{shared_data['matrix_data']}")
            shared_data['bin_ranges'] = bin_ranges
            logger.debug(f"update_heatmap_process: shared_data[bin_ranges]:{shared_data['bin_ranges']}")
        # time.sleep(5)  # Wait for 5 seconds before updating the heatmap again
    except Exception as e:
        print(f"Error in update_heatmap_process: {str(e)}")
        # time.sleep(5)

def run_agent(sweep_id, sweep_config, train_config):
    wandb.agent(
        sweep_id,
        function=lambda: wandb_support._run_sweep(sweep_config, train_config),
        count=train_config['num_sweeps'],
        project=sweep_config["project"],
    )


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
        elif page == '/co-occurrence-analysis':
            return layouts.co_occurrence_analysis(page)
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
        Output({'type':'optimizer-options', 'model':MATCH, 'agent':MATCH}, 'children'),
        Input({'type':'optimizer', 'model':MATCH, 'agent':MATCH}, 'value'),
        State({'type':'optimizer-options', 'model':MATCH, 'agent':MATCH}, 'id'),
        prevent_initial_call=True,
    )
    def update_agent_optimizer_params(optimizer, optimizer_id):
        agent_type = optimizer_id['agent']
        model_type = optimizer_id['model']
        return utils.create_optimizer_params_input(agent_type, model_type, optimizer)

        
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
                    dcc.Input(
                        id=input_id,
                        type='number',
                        min=1,
                        max=1024,
                        step=1,
                        value=512,
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
                    dcc.Input(
                        id={
                            'type': 'conv-filters',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=1024,
                        step=1,
                        value=32,
                    ),
                    html.Label(f'Kernel Size in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type': 'conv-kernel-size',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                    ),
                    html.Label(f'Kernel Stride in Conv Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type': 'conv-stride',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
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
                            dcc.Input(
                                id={
                                    'type': 'conv-padding-custom',
                                    'model': model_type,
                                    'agent': agent,
                                    'index': index,
                                },
                                type='number',
                                min=0,
                                max=10,
                                step=1,
                                value=1,
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
                    dcc.Input(
                        id={
                            'type': 'pool-kernel-size',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                    ),
                    html.Label(f'Kernel Stride in Pooling Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type': 'pool-stride',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                    ),
                ])
            if layer_type == 'batchnorm':
                return html.Div([
                    html.Label(f'Number of Features for BatchNorm Layer {index} (set to number of input channels)', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type': 'batch-features',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=1,
                        max=1024,
                        step=1,
                        value=32,
                    ),
                ])
            if layer_type == 'dropout':
                return html.Div([
                    html.Label(f'Probability of Zero-ed Element for Dropout Layer {index}', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type': 'dropout-prob',
                            'model': model_type,
                            'agent': agent,
                            'index': index,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=0.5,
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
                    dcc.Input(
                        id={
                            'type':'ou-mean',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.0,
                    ),
                    html.Label('Mean Reversion', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'ou-sigma',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.15,
                    ),
                    html.Label('Volatility', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'ou-theta',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.2,
                    ),
                    html.Label('Time Delta', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'ou-dt',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=1.0,
                    ),
                ])

            elif noise_type == "Normal":
                inputs = html.Div([
                    html.Label('Mean', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'normal-mean',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.0,
                    ),
                    html.Label('Standard Deviation', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'normal-stddv',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=1.0,
                    ),
                ])

            elif noise_type == "Uniform":
                inputs = html.Div([
                    html.Label('Minimum Value', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'uniform-min',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.1,
                    ),
                    html.Label('Maximum Value', style={'text-decoration': 'underline'}),
                    dcc.Input(
                        id={
                            'type':'uniform-max',
                            'model':'none',
                            'agent': agent_type,
                        },
                        type='number',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=1.0,
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
        Output({'type': 'goal-strategy-options', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'goal-strategy', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'goal-strategy', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_goal_strategy_options(strategy, strategy_id):
        agent_type = strategy_id['agent']
        return utils.update_goal_strategy_options(agent_type, strategy)
    
    @app.callback(
        Output({'type': 'goal-strategy-options-hyperparam', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'goal-strategy-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'goal-strategy-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_goal_strategy_hyperparam_options(strategy, strategy_id):
        agent_type = strategy_id['agent']
        options = []
        for strat in strategy:
            options.append(utils.update_goal_strategy_hyperparam_options(agent_type, strat))
        
        return options
    
    @app.callback(
        Output({'type': 'normalize-options', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'normalize-input', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'normalize-input', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_normalize_options(normalize, normalize_id):
        if normalize == 'True':
            agent_type = normalize_id['agent']
            if agent_type == 'DDPG':
                return utils.create_input_normalizer_options_input(agent_type)
        return html.Div()
    
    @app.callback(
        Output({'type': 'normalize-options-hyperparam', 'model': MATCH, 'agent': MATCH}, 'children'),
        Input({'type': 'normalize-input-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
        State({'type': 'normalize-input-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
        prevent_initial_call=True
    )
    def update_normalize_hyperparam_options(normalize, normalize_id):
        for norm in normalize:
            if norm == 'True':
                agent_type = normalize_id['agent']
                model_type = normalize_id['model']
                # if agent_type == 'DDPG':
                return utils.create_input_normalizer_options_hyperparam_input(agent_type, model_type)
        return html.Div()
        
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
        print(f'env spec: {env.spec}')

        # Get device
        device = utils.get_specific_value(
            all_values=all_values,
            all_ids=all_ids,
            id_type='device',
            model_type='none',
            agent_type=agent_type_dropdown_value,
        )

        # set params if agent is reinforce or actor critic
        if agent_type_dropdown_value in ["Reinforce", "ActorCritic", "PPO"]:
            # set defualt gym environment in order to build policy and value models and save
            # env = gym.make("CartPole-v1")

            learning_rate=10**utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='learning-rate',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )

            policy_optimizer = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='optimizer',
                model_type='policy',
                agent_type=agent_type_dropdown_value,
            )
            
            policy_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='policy',
                all_values=all_values,
                all_ids=all_ids
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
                    id_type='activation-function',
                    model_type='policy',
                    agent_type=agent_type_dropdown_value,
                ),
                policy_initializer,
            )

            policy_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='policy-output',
                agent_type=agent_type_dropdown_value
            )

            ##DEBUG
            # print("Policy layers:", policy_layers)
            if agent_type_dropdown_value == "PPO":
                policy_type = utils.get_specific_value(
                    all_values=all_values,
                    add_ids=all_ids,
                    model_type='policy',
                    agent_type=agent_type_dropdown_value,
                )

                if policy_type == 'StochasticContinuousPolicy':
                    model = StochasticContinuousPolicy
                else:
                    model = StochasticDiscretePolicy
            else:
                model = StochasticDiscretePolicy

            policy_model = model(
                    env=gym.make(env),
                    dense_layers=policy_layers,
                    output_layer_kernel=policy_output_kernel,
                    optimizer=policy_optimizer,
                    optimizer_params=policy_opt_params,
                    learning_rate=learning_rate,
                )

            value_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='value',
                    agent_type=agent_type_dropdown_value,
                )
            
            value_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='value',
                all_values=all_values,
                all_ids=all_ids
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
                    id_type='activation-function',
                    model_type='value',
                    agent_type=agent_type_dropdown_value,
                ),
                value_initializer,
            )

            value_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='value-output',
                agent_type=agent_type_dropdown_value
            )
            
            value_model = ValueModel(
                env=gym.make(env),
                dense_layers=value_layers,
                output_layer_kernel=value_output_kernel,
                optimizer=value_optimizer,
                optimizer_params=value_opt_params,
                learning_rate=learning_rate,
            )

            discount=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        id_type='discount',
                        model_type='none',
                        agent_type=agent_type_dropdown_value,
                    ),
            
            save_dir = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='save-dir',
                model_type='none',
                agent_type=agent_type_dropdown_value,
                )
            
            if agent_type_dropdown_value == "Reinforce":

                agent = rl_agents.Reinforce(
                    env=gym.make(env),
                    policy_model=policy_model,
                    value_model=value_model,
                    discount=discount,
                    callbacks=utils.get_callbacks(callbacks, project),
                    save_dir=os.path.join(os.getcwd(), save_dir),
                )

            elif agent_type_dropdown_value == "ActorCritic":

                policy_trace_decay=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        id_type='trace-decay',
                        model_type='policy',
                        agent_type=agent_type_dropdown_value,
                    )
                value_trace_decay=utils.get_specific_value(
                        all_values=all_values,
                        all_ids=all_ids,
                        id_type='trace-decay',
                        model_type='value',
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
                    save_dir=os.path.join(os.getcwd(), save_dir),
                )

            # elif agent_type_dropdown_value == "PPO":

            #     dist = 

        elif agent_type_dropdown_value == "DDPG":
            # set defualt gym environment in order to build policy and value models and save
            # env = gym.make("Pendulum-v1")

            # # Get device
            # device = utils.get_specific_value(
            #     all_values=all_values,
            #     all_ids=all_ids,
            #     id_type='device',
            #     model_type='none',
            #     agent_type=agent_type_dropdown_value,
            # )

            # Set actor params
            # Set actor learning rate
            actor_learning_rate=10**utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='learning-rate',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )

            actor_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )
            
            actor_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='actor',
                all_values=all_values,
                all_ids=all_ids
            )

            actor_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-hidden',
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
            else:
                actor_cnn = None


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
                    id_type='activation-function',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                ),
                actor_hidden_kernel,
            )

            actor_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-output',
                agent_type=agent_type_dropdown_value
            )

            actor_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='actor',
                agent_type=agent_type_dropdown_value,
            )

            # Create actor model
            actor_model = ActorModel(
                env=env,
                cnn_model=actor_cnn,
                dense_layers=actor_dense_layers,
                output_layer_kernel=actor_output_kernel,
                optimizer=actor_optimizer,
                optimizer_params=actor_opt_params,
                learning_rate=actor_learning_rate,
                normalize_layers=actor_normalize_layers,
                device=device,
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
                    id_type='learning-rate',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='critic',
                all_values=all_values,
                all_ids=all_ids
            )

            critic_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-hidden',
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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
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
            else:
                critic_cnn = None

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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
            )

            critic_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-output',
                agent_type=agent_type_dropdown_value
            )
           
            critic_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='critic',
                agent_type=agent_type_dropdown_value,
            )
           
            critic_model = CriticModel(
                env=env,
                cnn_model=critic_cnn,
                state_layers=critic_state_layers,
                merged_layers=critic_merged_layers,
                output_layer_kernel=critic_output_kernel,
                learning_rate=critic_learning_rate,
                optimizer=critic_optimizer,
                optimizer_params=critic_opt_params,
                normalize_layers=critic_normalize_layers,
                device=device,
            )

            #DEBUG
            # print(f'critic cnn model: {critic_cnn}')
            # print(f'critic state layers: {critic_state_layers}')
            # print(f'critic merged layers: {critic_merged_layers}')
            # print(f'critic optimizer: {critic_optimizer}')
            # print(f'critic learning rate: {critic_learning_rate}')
            # print(f'critic model: {critic_model}')


            # Set DDPG params

            discount = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'discount',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            tau=utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'tau',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            epsilon = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'epsilon-greedy',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )
            
            batch_size = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'batch-size',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            noise=utils.create_noise_object(
                    env = env,
                    all_values = all_values,
                    all_ids = all_ids,
                    agent_type = agent_type_dropdown_value,
                )
            
            normalize_inputs = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'normalize-input',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            clip_value = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'clip-value',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            warmup = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'warmup',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            save_dir = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='save-dir',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )
            
            agent = rl_agents.DDPG(
                env = env,
                actor_model = actor_model,
                critic_model = critic_model,
                discount = discount,
                tau = tau,
                action_epsilon = epsilon,
                replay_buffer = helper.ReplayBuffer(env, 100000, device=device),
                batch_size = batch_size,
                noise = noise,
                normalize_inputs = normalize_inputs,
                normalizer_clip = clip_value,
                warmup = warmup,
                callbacks = utils.get_callbacks(callbacks, project),
                save_dir = os.path.join(os.getcwd(), save_dir),
                device=device,
            )

        elif agent_type_dropdown_value == "TD3":
            # set defualt gym environment in order to build policy and value models and save
            # env = gym.make("Pendulum-v1")

            # # Get device
            # device = utils.get_specific_value(
            #     all_values=all_values,
            #     all_ids=all_ids,
            #     id_type='device',
            #     model_type='none',
            #     agent_type=agent_type_dropdown_value,
            # )

            # Set actor params
            # Set actor learning rate
            actor_learning_rate=10**utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='learning-rate',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )

            actor_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )
            
            actor_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='actor',
                all_values=all_values,
                all_ids=all_ids
            )

            actor_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-hidden',
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
            else:
                actor_cnn = None


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
                    id_type='activation-function',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                ),
                actor_hidden_kernel,
            )

            actor_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-output',
                agent_type=agent_type_dropdown_value
            )

            actor_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='actor',
                agent_type=agent_type_dropdown_value,
            )

            # Create actor model
            actor_model = ActorModel(
                env=env,
                cnn_model=actor_cnn,
                dense_layers=actor_dense_layers,
                output_layer_kernel=actor_output_kernel,
                optimizer=actor_optimizer,
                optimizer_params=actor_opt_params,
                learning_rate=actor_learning_rate,
                normalize_layers=actor_normalize_layers,
                device=device,
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
                    id_type='learning-rate',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='critic',
                all_values=all_values,
                all_ids=all_ids
            )

            critic_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-hidden',
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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
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
            else:
                critic_cnn = None

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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
            )

            critic_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-output',
                agent_type=agent_type_dropdown_value
            )
           
            critic_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='critic',
                agent_type=agent_type_dropdown_value,
            )
           
            critic_model = CriticModel(
                env=env,
                cnn_model=critic_cnn,
                state_layers=critic_state_layers,
                merged_layers=critic_merged_layers,
                output_layer_kernel=critic_output_kernel,
                learning_rate=critic_learning_rate,
                optimizer=critic_optimizer,
                optimizer_params=critic_opt_params,
                normalize_layers=critic_normalize_layers,
                device=device,
            )

            #DEBUG
            # print(f'critic cnn model: {critic_cnn}')
            # print(f'critic state layers: {critic_state_layers}')
            # print(f'critic merged layers: {critic_merged_layers}')
            # print(f'critic optimizer: {critic_optimizer}')
            # print(f'critic learning rate: {critic_learning_rate}')
            # print(f'critic model: {critic_model}')


            # Set DDPG params

            discount = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'discount',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            tau=utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'tau',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            epsilon = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'epsilon-greedy',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )
            
            batch_size = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'batch-size',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            noise=utils.create_noise_object(
                    env = env,
                    all_values = all_values,
                    all_ids = all_ids,
                    agent_type = agent_type_dropdown_value,
                )
            
            target_noise_stddev = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'target-noise-stddev',
                model_type = 'actor',
                agent_type = agent_type_dropdown_value,
            )

            target_noise_clip = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'target-noise-clip',
                model_type = 'actor',
                agent_type = agent_type_dropdown_value,
            )

            actor_update_delay = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'actor-update-delay',
                model_type = 'actor',
                agent_type = agent_type_dropdown_value,
            )
            
            normalize_inputs = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'normalize-input',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            clip_value = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'clip-value',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            warmup = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'warmup',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            save_dir = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='save-dir',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )
            
            agent = rl_agents.TD3(
                env = env,
                actor_model = actor_model,
                critic_model = critic_model,
                discount = discount,
                tau = tau,
                action_epsilon = epsilon,
                replay_buffer = helper.ReplayBuffer(env, 100000, device=device),
                batch_size = batch_size,
                noise = noise,
                target_noise_stddev = target_noise_stddev,
                target_noise_clip = target_noise_clip,
                actor_update_delay = actor_update_delay,
                normalize_inputs = normalize_inputs,
                normalizer_clip = clip_value,
                warmup = warmup,
                callbacks = utils.get_callbacks(callbacks, project),
                save_dir = os.path.join(os.getcwd(), save_dir),
                device=device,
            )

        elif agent_type_dropdown_value == "HER_DDPG":

            # # Get device
            # device = utils.get_specific_value(
            #     all_values=all_values,
            #     all_ids=all_ids,
            #     id_type='device',
            #     model_type='none',
            #     agent_type=agent_type_dropdown_value,
            # )

            # get goal and reward functions and goal shape
            desired_goal_func, achieved_goal_func, reward_func = gym_helper.get_her_goal_functions(env)
            
            # Reset env in order to instantiate and get goal shape
            _,_ = env.reset()
            goal_shape = desired_goal_func(env).shape

            print(f'desired goal: {desired_goal_func}')
            print(f'achieved goal: {achieved_goal_func}')
            print(f'reward func: {reward_func}')
            print(f'goal shape: {goal_shape}')

            # Set actor params
            # Set actor learning rate
            actor_learning_rate=10**utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='learning-rate',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )

            actor_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                )
            
            actor_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='actor',
                all_values=all_values,
                all_ids=all_ids
            )

            actor_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-hidden',
                agent_type=agent_type_dropdown_value
            )

            actor_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='actor-output',
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
                actor_cnn=cnn_models.CNN(actor_conv_layers, env)
            else:
                actor_cnn=None


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
                    id_type='activation-function',
                    model_type='actor',
                    agent_type=agent_type_dropdown_value,
                ),
                actor_hidden_kernel,
            )

            actor_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='actor',
                agent_type=agent_type_dropdown_value,
            )

            #DEBUG
            print(f'actor dense layers: {actor_dense_layers}')
            print(f'actor output kernel: {actor_output_kernel}, type: {type(actor_output_kernel)}')
            # Create actor model
            actor_model = ActorModel(
                env=env,
                cnn_model=actor_cnn,
                dense_layers=actor_dense_layers,
                output_layer_kernel=actor_output_kernel,
                goal_shape=goal_shape,
                optimizer=actor_optimizer,
                optimizer_params=actor_opt_params,
                learning_rate=actor_learning_rate,
                normalize_layers=actor_normalize_layers,
                device=device,
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
                    id_type='learning-rate',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_optimizer = utils.get_specific_value(
                    all_values=all_values,
                    all_ids=all_ids,
                    id_type='optimizer',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                )
            
            critic_opt_params = utils.get_optimizer_params(
                agent_type=agent_type_dropdown_value,
                model_type='critic',
                all_values=all_values,
                all_ids=all_ids
            )

            critic_hidden_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-hidden',
                agent_type=agent_type_dropdown_value
            )

            critic_output_kernel = utils.format_kernel_initializer_config(
                all_values=all_values,
                all_ids=all_ids,
                value_model='critic-output',
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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
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
            else:
                critic_cnn = None

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
                    id_type='activation-function',
                    model_type='critic',
                    agent_type=agent_type_dropdown_value,
                ),
                critic_hidden_kernel,
            )
           
            critic_normalize_layers = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='normalize-layers',
                model_type='critic',
                agent_type=agent_type_dropdown_value,
            )
           
            critic_model = CriticModel(
                env=env,
                cnn_model=critic_cnn,
                state_layers=critic_state_layers,
                merged_layers=critic_merged_layers,
                output_layer_kernel=critic_output_kernel,
                goal_shape=goal_shape,
                learning_rate=critic_learning_rate,
                optimizer=critic_optimizer,
                optimizer_params=critic_opt_params,
                normalize_layers=critic_normalize_layers,
                device=device,
            )

            #DEBUG
            # print(f'critic cnn model: {critic_cnn}')
            # print(f'critic state layers: {critic_state_layers}')
            # print(f'critic merged layers: {critic_merged_layers}')
            # print(f'critic optimizer: {critic_optimizer}')
            # print(f'critic learning rate: {critic_learning_rate}')
            # print(f'critic model: {critic_model}')


            # Set DDPG params

            discount = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'discount',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            tau=utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'tau',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            epsilon = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'epsilon-greedy',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )
            
            batch_size = utils.get_specific_value(
                    all_values = all_values,
                    all_ids = all_ids,
                    id_type = 'batch-size',
                    model_type = 'none',
                    agent_type = agent_type_dropdown_value,
                )
            
            noise=utils.create_noise_object(
                    env = env,
                    all_values = all_values,
                    all_ids = all_ids,
                    agent_type = agent_type_dropdown_value,
                )
            
            normalize_inputs = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'normalize-input',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            clip_value = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'clip-value',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )
            
            ddpg_agent = rl_agents.DDPG(
                env = env,
                actor_model = actor_model,
                critic_model = critic_model,
                discount = discount,
                tau = tau,
                action_epsilon = epsilon,
                replay_buffer = helper.ReplayBuffer(env, 100000, goal_shape, device=device),
                batch_size = batch_size,
                noise = noise,
                normalize_inputs = normalize_inputs,
                normalizer_clip = clip_value,
                callbacks = utils.get_callbacks(callbacks, project),
                # save_dir = os.path.join(os.getcwd(), 'assets/models/ddpg/'),
                device = device
            )

            # set HER specific hyperparams
            strategy = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='goal-strategy',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )

            num_goals = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='future-goals',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )

            tolerance = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='goal-tolerance',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )

            normalizer_clip = utils.get_specific_value(
                all_values = all_values,
                all_ids = all_ids,
                id_type = 'clip-value',
                model_type = 'none',
                agent_type = agent_type_dropdown_value,
            )

            save_dir = utils.get_specific_value(
                all_values=all_values,
                all_ids=all_ids,
                id_type='save-dir',
                model_type='none',
                agent_type=agent_type_dropdown_value,
            )

            # create HER object
            agent = rl_agents.HER(
                ddpg_agent,
                strategy=strategy,
                tolerance=tolerance,
                num_goals=num_goals,
                desired_goal=desired_goal_func,
                achieved_goal=achieved_goal_func,
                reward_fn=reward_func,
                normalizer_clip=normalizer_clip,
                device=device,
                save_dir = os.path.join(os.getcwd(), save_dir),
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
            description, gif_url = utils.get_env_data(env_name)
            gym_params = {}
            if id['page'] != "/build-agent":
                gym_params = utils.generate_gym_extra_params_container(env_name)
            return description, gif_url, gym_params
        return "", "", gym_params  # Default empty state

    @app.callback(
        Output({'type':'hidden-div', 'page':'/train-agent'}, 'data'),
        Input({'type':'start', 'page':'/train-agent'}, 'n_clicks'),
        State({'type':'start', 'page':'/train-agent'}, 'id'),
        State({'type':'agent-store', 'page':'/train-agent'}, 'data'),
        State({'type':'storage', 'page':'/train-agent'}, 'data'),
        State({'type':'env-dropdown', 'page':'/train-agent'}, 'value'),
        State({'type':'num-episodes', 'page':'/train-agent'}, 'value'),
        State({'type':'render-option', 'page':'/train-agent'}, 'value'),
        State({'type':'render-freq', 'page':'/train-agent'}, 'value'),
        State({'type':'epochs', 'page':'/train-agent'}, 'value'),
        State({'type':'cycles', 'page':'/train-agent'}, 'value'),
        State({'type':'learning-cycles', 'page':'/train-agent'}, 'value'),
        State({'type':'mpi', 'page':'/train-agent'}, 'value'),
        State({'type':'workers', 'page':'/train-agent'}, 'value'),
        State({'type':'load-weights', 'page':'/train-agent'}, 'value'),
        State({'type':'seed', 'page':'/train-agent'}, 'value'),
        State({'type':'run-number', 'page':'/train-agent'}, 'value'),
        State({'type':'num-runs', 'page':'/train-agent'}, 'value'),
        State({'type':'save-dir', 'page':'/train-agent'}, 'value'),
        prevent_initial_call=True,
    )
    def train_agent(n_clicks, id, agent_data, storage_data, env_name, num_episodes, render_option, render_freq, epochs, cycles, num_updates, use_mpi, workers, load_weights, seed, run_number, num_runs, save_dir):
        #DEBUG
        # print("Start callback called.")
        if n_clicks > 0:
            
            # Use the agent_data['save_dir'] to load agent
            if agent_data:  # Check if agent_data is not empty
                # set save dir to agent config save dir if save dir == None
                if not save_dir:
                    save_dir = agent_data['save_dir']
                
                # Create an empty dict for train_config.json
                train_config = {}
                render = 'RENDER' in render_option
                # use_mpi = agent_data.get('use_mpi', False)
                
                # clear the renders in the train folder
                if render:
                    if os.path.exists(save_dir + '/renders/training'):
                        utils.delete_renders(save_dir + '/renders/training')

                # Update the configuration with render settings
                train_config['num_episodes'] = num_episodes
                train_config['render'] = render
                train_config['render_freq'] = render_freq
                train_config['load_weights'] = load_weights
                train_config['seed'] = seed
                train_config['run_number'] = run_number
                train_config['num_runs'] = num_runs
                train_config['save_dir'] = save_dir

                # Add MPI settings to config if DDPG or HER
                if agent_data['agent_type'] in ['DDPG','TD3','HER']:
                    train_config['use_mpi'] = use_mpi
                    train_config['num_workers'] = workers
                
                # Update additional settings for HER agent
                if agent_data['agent_type'] == 'HER':
                    train_config['num_epochs'] = epochs
                    train_config['num_cycles'] = cycles
                    train_config['num_updates'] = num_updates
            
                # Save the updated configuration to a train config file
                train_config_path = save_dir + '/train_config.json'
                print(f'agent train config path:{train_config_path}')
                with open(train_config_path, 'w') as f:
                    json.dump(train_config, f)

                # Set the config path of the agent
                agent_config_path = save_dir + '/config.json'
                print(f'agent config path:{agent_config_path}')

                script_path = 'train.py'
                run_command = f"python {script_path} --agent_config {agent_config_path} --train_config {train_config_path}"
                subprocess.Popen(run_command, shell=True)

        raise PreventUpdate
    
    @app.callback(
        Output({'type':'hidden-div', 'page':'/test-agent'}, 'data'),
        Input({'type':'start', 'page':'/test-agent'}, 'n_clicks'),
        State({'type':'start', 'page':'/test-agent'}, 'id'),
        State({'type':'agent-store', 'page':'/test-agent'}, 'data'),
        State({'type':'storage', 'page':'/test-agent'}, 'data'),
        State({'type':'env-dropdown', 'page':'/test-agent'}, 'value'),
        State({'type':'num-episodes', 'page':'/test-agent'}, 'value'),
        State({'type':'render-option', 'page':'/test-agent'}, 'value'),
        State({'type':'render-freq', 'page':'/test-agent'}, 'value'),
        State({'type':'load-weights', 'page':'/test-agent'}, 'value'),
        State({'type':'seed', 'page':'/test-agent'}, 'value'),
        State({'type':'run-number', 'page':'/test-agent'}, 'value'),
        State({'type':'num-runs', 'page':'/test-agent'}, 'value'),
        prevent_initial_call=True,
    )
    def test_agent(n_clicks, id, agent_data, storage_data, env_name, num_episodes, render_option, render_freq, load_weights, seed, run_number, num_runs):

        print('test agent fired...')
        try:
            if n_clicks > 0 and agent_data:
                #DEBUG
                print('n clicks and agent data passed')
                # clear the renders in the train folder
                agent_type = agent_data['agent_type']
                if agent_type == "HER":
                    agent_type = agent_data['agent']['agent_type']
                if os.path.exists(f'assets/models/{agent_type}/renders/testing'):
                    utils.delete_renders(f"assets/models/{agent_type}/renders/testing")

                # Create empty dict for test_config.json
                test_config = {}
                # Update the configuration with render settings
                render = 'RENDER' in render_option
                test_config['render'] = render
                test_config['num_episodes'] = num_episodes
                test_config['render_freq'] = render_freq
                test_config['load_weights'] = load_weights
                test_config['seed'] = seed
                test_config['run_number'] = run_number
                test_config['num_runs'] = num_runs

                # Save the updated configuration to a file
                test_config_path = agent_data['save_dir'] + '/test_config.json'
                with open(test_config_path, 'w') as f:
                    json.dump(test_config, f)

                # Set the config path of the agent
                agent_config_path = agent_data['save_dir'] + '/config.json'
                
                script_path = 'test.py'
                #DEBUG
                print(f'script set to {script_path}')
                run_command = f"python {script_path} --agent_config {agent_config_path} --test_config {test_config_path}"
                subprocess.Popen(run_command, shell=True)

            raise PreventUpdate

        except KeyError as e:
            print(f"KeyError: {str(e)}")
            # Handle the case when a required key is missing in agent_data
            # You can choose to raise an exception, return an error message, or take appropriate action

        except FileNotFoundError as e:
            print(f"FileNotFoundError: {str(e)}")
            # Handle the case when the specified file or directory is not found
            # You can choose to raise an exception, return an error message, or take appropriate action

        except subprocess.SubprocessError as e:
            print(f"SubprocessError: {str(e)}")
            # Handle the case when there is an error executing the subprocess
            # You can choose to raise an exception, return an error message, or take appropriate action

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
    # Handle any other unexpected exceptions
    # You can choose to raise an exception, return an error message, or take appropriate action

        # if n_clicks > 0:
        #     # clear the renders in the train folder
        #     agent_type = agent_data['agent_type']
        #     if agent_type == "HER":
        #         agent_type = agent_data['agent']['agent_type']
        #     if os.path.exists(f'assets/models/{agent_type}/renders/testing'):
        #         utils.delete_renders(f"assets/models/{agent_type}/renders/testing")
            
        #     # Use the agent_data['save_dir'] to load agent
        #     if agent_data:  # Check if agent_data is not empty
        #         render = 'RENDER' in render_option

        #         # Update the configuration with render settings
        #         agent_data['num_episodes'] = num_episodes
        #         agent_data['render'] = render
        #         agent_data['render_freq'] = render_freq
        #         agent_data['load_weights'] = load_weights
                
        #         # Save the updated configuration to a file
        #         config_path = agent_data['save_dir'] + '/test_config.json'
        #         with open(config_path, 'w') as f:
        #             json.dump(agent_data, f)

        #         script_path = 'test.py'
        #         run_command = f"python {script_path} {config_path}"
        #         subprocess.Popen(run_command, shell=True)

        # raise PreventUpdate
            
    
    @app.callback(
    Output({'type':'agent-store', 'page':MATCH}, 'data'),
    Output({'type':'output-agent-load', 'page':MATCH}, 'children'),
    Input({'type':'upload-agent-config', 'page':MATCH}, 'contents'),
    State({'type':'upload-agent-config', 'page':MATCH}, 'id'),
    prevent_initial_call=True,
    )
    def store_agent(contents, upload_id):
        # for content, page in zip(contents, id):
        if contents is not None:
            _, encoded = contents.split(',')
            decoded = base64.b64decode(encoded)           
            config = json.loads(decoded.decode('utf-8'))
            
            success_message = html.Div([
            dbc.Alert(f"Config loaded successfully from: {config['save_dir']}", color="success")
            ])
            
            return config, success_message
    
        return {}, "Please upload a file.", {'display': 'none'}
    
    @app.callback(
    Output({'type': 'her-options', 'page': '/train-agent'}, 'style'),
    Input({'type':'agent-store', 'page': '/train-agent'}, 'data'),
    prevent_initial_call = True,
    )
    def update_her_options(agent_data):

        if agent_data['agent_type'] == 'HER':
            her_options_style = {'display': 'block'}
        else:
            her_options_style = {'display': 'none'}
        
        return her_options_style
    
    @app.callback(
    Output({'type': 'mpi-options', 'page': '/train-agent'}, 'style'),
    Input({'type':'agent-store', 'page': '/train-agent'}, 'data'),
    prevent_initial_call = True,
    )
    def update_mpi_options(agent_data):

        if agent_data['agent_type'] in ['DDPG', 'TD3', 'HER']:
            mpi_options_style = {'display': 'block'}
        
        else:
            mpi_options_style = {'display': 'none'}
        
        return mpi_options_style
    
    @app.callback(
    Output({'type': 'mpi-options', 'page': '/hyperparameter-search'}, 'style'),
    Input({'type':'agent-type-selector', 'page': '/hyperparameter-search'}, 'value'),
    prevent_initial_call = True,
    )
    def update_mpi_hyperparam_options(agent_types):

        print(f'agent types: {agent_types}')

        if any(agent in ['DDPG', 'TD3', 'HER_DDPG'] for agent in agent_types):
        # if agent_types:
            print(f'hyperparam options true')
            mpi_options = {'display': 'block'}
        
        else:
            print(f'hyperparam options false')
            mpi_options = {'display': 'none'}
        
        print(f'mpi options: {mpi_options}')
        return mpi_options
    
    @app.callback(
    Output({'type': 'workers', 'page': MATCH}, 'style'),
    Input({'type':'mpi', 'page': MATCH}, 'value'),
    prevent_initial_call = True,
    )
    def update_workers_option(use_mpi):

        if use_mpi:
            workers_style = {'display': 'block'}
        else:
            workers_style = {'display': 'none'}
        
        return workers_style
    
    @app.callback(
    Output({'type': 'sweep-options', 'page': '/hyperparameter-search'}, 'style'),
    Input({'type':'device', 'model': 'none', 'agent':ALL}, 'value'),
    State('url', 'pathname'),
    prevent_initial_call = True,
    )
    def update_sweep_option(device, page):

        if page == '/hyperparameter-search':

            if any([d=='cpu' for d in device]):
                options = {'display': 'block'}
            else:
                options = {'display': 'none'}
            
            return options
    
    @app.callback(
    Output({'type': 'clamp-value', 'model':MATCH, 'agent':MATCH}, 'style'),
    Input({'type':'clamp-output', 'model':MATCH, 'agent':MATCH}, 'value'),
    prevent_initial_call = True,
    )
    def update_clamp_options(clamp_output):

        if clamp_output:
            clamp_options_style = {'display': 'block'}
        else:
            clamp_options_style = {'display': 'none'}
        
        return clamp_options_style
    
    # @app.callback(
    # Output({'type': 'ddpg-workers', 'page': '/train-agent'}, 'style'),
    # Input({'type':'ddpg-mpi', 'page': '/train-agent'}, 'value'),
    # prevent_initial_call = True,
    # )
    # def update_ddpg_workers_option(use_mpi):

    #     if use_mpi:
    #         workers_style = {'display': 'block'}
    #     else:
    #         workers_style = {'display': 'none'}
        
    #     return workers_style
    

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
     State({'type':'agent-store', 'page':MATCH}, 'data'),
     State({'type':'num-episodes', 'page':MATCH}, 'value'),
     State({'type':'epochs', 'page':MATCH}, 'value'),
     State({'type':'cycles', 'page':MATCH}, 'value'),
     State('url', 'pathname')],
     prevent_initial_call = True,
)
    def update_data(n, storage_data, agent_config, num_episodes, num_epochs, num_cycles, pathname):
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
            

            # if the new data dict isn't empty, update storage data
            if data != {}:
                # Need to determine agent type in order to correctly calculate num_episodes and progress
                if agent_config['agent_type'] == 'HER':
                    num_episodes = num_epochs * num_cycles * num_episodes
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
        State({'type':'agent-store', 'page':MATCH}, 'data'),
        prevent_initial_call=True
    )
    def update_video_data(prev_clicks, next_clicks, n_intervals, data, interval_id, start_clicks, carousel_id, agent_data):
        # check if start has been clicked before trying to update        
        ctx = dash.callback_context
        triggered_id, prop = ctx.triggered[0]['prop_id'].split('.')
        #DEBUG
        # print(f'trigger id: {triggered_id}')

        # Initial load or automatic update triggered by the interval component
        if 'interval-component' in triggered_id and start_clicks > 0:
            video_filenames = utils.get_video_files(interval_id['page'], agent_data)
            if video_filenames:
                data['video_list'] = video_filenames
                # Optionally reset current video to 0 or keep it as is
                # data['current_video'] = 0
                # video_files = data['video_list']
                current_video = data.get('current_video', 0)
                current_filename = video_filenames[current_video]
                video_item = utils.generate_video_items([current_filename], carousel_id['page'], agent_data)[0]
                
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
            video_item = utils.generate_video_items([current_filename], carousel_id['page'], agent_data)[0]

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
        Input({'type':'agent-type-selector', 'page':'/hyperparameter-search'}, 'value')
    )
    def update_hyperparam_inputs(selected_agent_types):
        # This function updates the inputs based on selected agent types
        tabs = []
        for agent_type in selected_agent_types:
            if agent_type == 'Reinforce':
                tabs.append(utils.create_reinforce_hyperparam_input(agent_type))
            
            elif agent_type == 'ActorCritic':
                tabs.append(utils.create_actor_critic_hyperparam_input(agent_type))
            
            elif agent_type == 'DDPG':
                tabs.append(utils.create_ddpg_hyperparam_input(agent_type))

            elif agent_type == 'TD3':
                tabs.append(utils.create_td3_hyperparam_input(agent_type))

            elif agent_type == 'HER_DDPG':
                tabs.append(utils.create_her_ddpg_hyperparam_input(agent_type))

        return tabs
    

    @app.callback(
    Output({'type': 'optimizer-options-hyperparams', 'model': MATCH, 'agent': MATCH}, 'children'),
    Input({'type': 'optimizer-hyperparam', 'model': MATCH, 'agent': MATCH}, 'value'),
    State({'type': 'optimizer-hyperparam', 'model': MATCH, 'agent': MATCH}, 'id'),
    prevent_initial_call=True
    )
    def update_optimizer_options_hyperparams(optimizers, optimizer_id):
        tabs = []
        for optimizer in optimizers:
            if optimizer == 'Adam':
                tab = dcc.Tab(
                    label='Adam',
                    children=[
                        html.Div([
                            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'adam-weight-decay-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                    ]
                )
            elif optimizer == 'Adagrad':
                tab = dcc.Tab(
                    label='Adagrad',
                    children=[
                        html.Div([
                            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'adagrad-weight-decay-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                            html.Label("Learning Rate Decay", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'adagrad-lr-decay-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                    ]
                )
            elif optimizer == 'RMSprop':
                tab = dcc.Tab(
                    label='RMSprop',
                    children=[
                        html.Div([
                            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'rmsprop-weight-decay-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                            html.Label("Momentum", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'rmsprop-momentum-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                    ]
                )
            elif optimizer == 'SGD':
                tab = dcc.Tab(
                    label='SGD',
                    children=[
                        html.Div([
                            html.Label("Weight Decay", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'sgd-weight-decay-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                            html.Label("Momentum", style={'text-decoration': 'underline'}),
                            dcc.Dropdown(
                                id=
                                {
                                    'type': 'sgd-momentum-hyperparam',
                                    'model': optimizer_id['model'],
                                    'agent': optimizer_id['agent']
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
                    ]
                )
            
            tabs.append(tab)
        
        return dcc.Tabs(tabs)
    

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
        Output({'type': 'entropy-block', 'model': 'none', 'agent': MATCH},'style'),
        Output({'type': 'kl-block', 'model': 'none', 'agent': MATCH},'style'),
        Output({'type': 'lambda-block', 'model': 'none', 'agent': MATCH},'style'),
        Input({'type':'loss-type', 'model': 'none', 'agent': MATCH}, 'value'),
        prevent_initial_call=True
    )
    def update_ppo_loss_options(loss_type):
        if loss_type == 'KL':
            kl_block = {'display': 'block'}
            ep_block = {'display': 'none'}
            lm_block = {'display': 'none'}
            return ep_block, kl_block, lm_block
        elif loss_type == 'Clipped':
            kl_block = {'display': 'none'}
            ep_block = {'display': 'block'}
            lm_block = {'display': 'none'}
            return ep_block, kl_block, lm_block
        kl_block = {'display': 'block'}
        ep_block = {'display': 'block'}
        lm_block = {'display': 'block'}
        return ep_block, kl_block, lm_block

    @app.callback(
        Output({'type': 'norm-clip-block', 'model': 'none', 'agent': MATCH},'style'),
        Input({'type':'norm-values', 'model': 'none', 'agent': MATCH}, 'value'),
        prevent_initial_call=True
    )
    def update_ppo_loss_options(norm_values):
        if norm_values:
            norm_block = {'display': 'block'}
            return norm_block
        
        norm_block = {'display': 'none'}
        return norm_block
    
    @app.callback(
        Output('her-options-hyperparam', 'hidden'),
        Input({'type':'agent-type-selector', 'page':'/hyperparameter-search'}, 'value'),
        prevent_initial_call=True
    )
    def update_her_hyperparam_options(agent_types):
        if 'HER_DDPG' in agent_types:
            return False
        return True
    
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
        State({'type':'seed', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'agent-type-selector', 'page':'/hyperparameter-search'}, 'value'),
        State('num-sweeps', 'value'),
        State('num-episodes', 'value'),
        State('num-epochs', 'value'),
        State('num-cycles', 'value'),
        State('num-updates', 'value'),
        State({'type':'mpi', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'workers', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'num-sweep-agents', 'page':'/hyperparameter-search'}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'id'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'id'),
        prevent_initial_call=True
    )
    def begin_sweep(num_clicks, data, method, project, sweep_name, metric_name, metric_goal, env, env_params, seed, agent_selection, num_sweeps, num_episodes, num_epochs, num_cycles, num_updates, use_mpi, num_workers, num_agents, all_values, all_ids, all_indexed_values, all_indexed_ids):

        try:
            if num_clicks > 0:
                # extract any additional gym env params
                params = utils.extract_gym_params(env_params)
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

                if sweep_config:  # Check if sweep_config is not empty
                    # Create an empty dict for sweep_config.json
                    train_config = {}
                    # Add config options to run_config
                    train_config['num_sweeps'] = num_sweeps
                    train_config['num_episodes'] = num_episodes
                    train_config['seed'] = seed if seed is not None else None

                    # Add MPI config if not None else None
                    train_config['use_mpi'] = use_mpi if use_mpi is not None else None
                    train_config['num_workers'] = num_workers if num_workers is not None else None
                    train_config['num_agents'] = num_agents if num_agents is not None else None
                    
                    # Update additional settings for HER agent
                    train_config['num_epochs'] = num_epochs if num_epochs is not None else None
                    train_config['num_cycles'] = num_cycles if num_cycles is not None else None
                    train_config['num_updates'] = num_updates if num_updates is not None else 1
                
                    # Save the updated configuration to a train config file
                    os.makedirs('sweep', exist_ok=True)
                    train_config_path = os.path.join(os.getcwd(), 'sweep/train_config.json')
                    with open(train_config_path, 'w') as f:
                        json.dump(train_config, f)

                    # Save the sweep config and set the sweep config path
                    sweep_config_path = os.path.join(os.getcwd(), 'sweep/sweep_config.json')
                    with open(sweep_config_path, 'w') as f:
                        json.dump(sweep_config, f)

                    # Construct and run the MPI command
                    command = [
                        'mpiexec', '-np', str(num_workers), 'python', 'init_sweep.py',
                        '--sweep_config', sweep_config_path,
                        '--train_config', train_config_path
                    ]

                    subprocess.Popen(command)
                    


        except KeyError as e:
            logger.error(f"KeyError in W&B stream handling: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
                
                
                

                # command = ['python', 'sweep.py']

                
                # subprocess.Popen(command)
            

        raise PreventUpdate


    @app.callback(
        Output({'type':'hyperparameter-selector', 'page':'/hyperparameter-search'}, 'options'),
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
        Output({'type':'heatmap-data-store', 'page':'/hyperparameter-search'}, 'data'),
        Input('heatmap-store-data-interval', 'n_intervals'),
        State({'type':'heatmap-data-store', 'page':'/hyperparameter-search'}, 'data'),
    )
    def update_heatmap_data(n, data):
        if 'matrix_data' not in shared_data:
            raise PreventUpdate
        matrix_data = shared_data.get('matrix_data')
        bin_ranges = shared_data.get('bin_ranges')
        if matrix_data:
            # print(f'heatmap data: {data}')
            new_data = {'matrix_data': matrix_data, 'bin_ranges': bin_ranges}
            print(f'new data:{new_data}')
            return new_data
        else:
            return None
        
    
    @app.callback(
        Output({'type':'heatmap-container', 'page':'/hyperparameter-search'}, 'children'),
        Output({'type':'legend-container', 'page':'/hyperparameter-search'}, 'children'),
        Input({'type':'heatmap-data-store', 'page':'/hyperparameter-search'}, 'data')
    )
    def update_heatmap_container(data):
        heatmap, bar_chart = utils.update_heatmap(data)
        if heatmap is None:
            return dash.no_update
        return dcc.Graph(figure=heatmap), dcc.Graph(figure=bar_chart)
    
    @app.callback(
        Output({'type':'heatmap-placeholder', 'page':'/hyperparameter-search'}, 'style'),
        Input({'type':'heatmap-container', 'page':'/hyperparameter-search'}, 'children')
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
        State({'type':'hyperparameter-selector', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'bin-slider', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'z-score-checkbox', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'reward-threshold', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'start', 'page':'/hyperparameter-search'}, 'n_clicks'),
    )
    def start_matrix_process(n, hyperparameters, bins, zscore_option, reward_threshold, n_clicks):
    # Create and start the update_heatmap_process
        if n_clicks > 0:
            z_score = 'zscore' in zscore_option
            # print('start matrix process callback called')
            # print(f'hyperparameters passed to start matrix callback: {hyperparameters}')
            update_heatmap_thread = threading.Thread(target=update_heatmap_process, args=(shared_data, hyperparameters, bins, z_score, reward_threshold))
            update_heatmap_thread.start()
        
        return None


    @app.callback(
        Output({'type':'sweeps-dropdown', 'page':'/co-occurrence-analysis'}, 'options'),
        Input({'type': 'projects-dropdown', 'page': '/co-occurrence-analysis'}, 'value'),
        prevent_initial_call=True,
    )
    def update_sweeps_dropdown(project):
        if project is not None:
            sweep_names = wandb_support.get_sweeps_from_name(project)
            return [{'label': name, 'value': name} for name in sweep_names]
        
    @app.callback(
        Output({'type':'heatmap-data-store', 'page':'/co-occurrence-analysis'}, 'data'),
        Output({'type':'output-data-upload', 'page':'/co-occurrence-analysis'}, 'children'),
        Input({'type':'sweep-data-button', 'page': '/co-occurrence-analysis'}, 'n_clicks'),
        State({'type':'projects-dropdown', 'page': '/co-occurrence-analysis'}, 'value'),
        State({'type':'sweeps-dropdown', 'page': '/co-occurrence-analysis'}, 'value'),
        State({'type':'heatmap-data-store', 'page':'/co-occurrence-analysis'}, 'data'),
        prevent_initial_call=True,
    )
    def get_sweep_data(n_clicks, project, sweeps, co_occurrence_data):
        try:
            if n_clicks > 0:
                dfs = []
                for sweep in sweeps:
                    metrics_data = wandb_support.get_metrics(project, sweep)
                    logger.debug(f'get_sweep_data: metrics data: {metrics_data}')
                    formatted_data = wandb_support.format_metrics(metrics_data)
                    logger.debug(f'get_sweep_data: formatted data: {formatted_data}')
                    dfs.append(formatted_data)
                data = pd.concat(dfs, ignore_index=True)
                data_json = data.to_json(orient='split')
                co_occurrence_data['formatted_data'] = data_json
                logger.debug(f'get_sweep_data: co_occurrence_data[formatted_data]: {co_occurrence_data["formatted_data"]}')
                # create a Div containing a success message to return
                success_message = html.Div([
                    dbc.Alert("Data Loaded.", color="success")
                ])
                return co_occurrence_data, success_message

            return None
        except Exception as e:
            logger.error(f"Error in get_sweep_data: {e}", exc_info=True)
    
    @app.callback(
        Output({'type':'hyperparameter-selector', 'page':'/co-occurrence-analysis'}, 'options'),
        Input({'type':'sweeps-dropdown', 'page':'/co-occurrence-analysis'}, 'value'),
        State({'type':'projects-dropdown', 'page':'/co-occurrence-analysis'}, 'value'),
        prevent_initial_call=True,
    )
    def update_co_occurance_hyperparameter_dropdown(sweeps, project):
        hyperparameters = wandb_support.fetch_sweep_hyperparameters_single_run(project, sweeps[0])

        return [{'label': hp, 'value': hp} for hp in hyperparameters]
    
    # @app.callback(
    #     Output({'type':'hyperparameter-selector', 'page':'/hyperparameter-search'}, 'options'),
    #     Input({'type':'sweeps-dropdown', 'page':'/hyperparameter-search'}, 'value'),
    #     State({'type':'projects-dropdown', 'page':'/hyperparameter-search'}, 'value'),
    #     prevent_initial_call=True,
    # )
    # def update_wandb_sweep_hyperparameter_dropdown(sweeps, project):
    #     hyperparameters = wandb_support.fetch_sweep_hyperparameters_single_run(project, sweeps[0])

    #     return [{'label': hp, 'value': hp} for hp in hyperparameters]


    @app.callback(
        Output({'type':'heatmap-container', 'page':'/co-occurrence-analysis'}, 'children'),
        Output({'type':'legend-container', 'page':'/co-occurrence-analysis'}, 'children'),
        Input({'type':'heatmap-data-store', 'page':'/co-occurrence-analysis'}, 'data'),
        Input({'type':'hyperparameter-selector', 'page':'/co-occurrence-analysis'}, 'value'),
        Input({'type':'bin-slider', 'page':'/co-occurrence-analysis'}, 'value'),
        Input({'type':'z-score-checkbox', 'page':'/co-occurrence-analysis'}, 'value'),
        Input({'type':'reward-threshold', 'page':'/co-occurrence-analysis'}, 'value'),
        prevent_initial_call=True,
    )
    def update_co_occurrence_graphs(data, hyperparameters, bins, zscore_option, reward_threshold):
        try:
            logger.debug(f'update_co_occurrence_graphs: data: {data}')
            # f_data = data['formatted_data']
            # Convert the JSON string back to a pandas DataFrame
            formatted_data = pd.read_json(data['formatted_data'], orient='split')
            logger.debug(f"update_co_occurrence_graphs: formatted_data dataframe: {formatted_data}")
            z_score = 'zscore' in zscore_option
            matrix_data, bin_ranges = wandb_support.calculate_co_occurrence_matrix(formatted_data, hyperparameters, reward_threshold, bins, z_score)
            data['matrix_data'] = matrix_data.to_dict(orient='split')
            logger.debug(f'update_co_occurrence_graphs: matrix data: {data["matrix_data"]}')
            data['bin_ranges'] = bin_ranges
            logger.debug(f'update_co_occurrence_graphs: bin_ranges: {data["bin_ranges"]}')
            heatmap, bar_chart = utils.update_heatmap(data)
            
            return dcc.Graph(figure=heatmap), dcc.Graph(figure=bar_chart)
        except Exception as e:
            logger.error(f"Error in update_co_occurrence_graphs: {e}", exc_info=True)

    @app.callback(
    Output("download-wandb-config", "data"),
    [Input("download-wandb-config-button", "n_clicks")],
    [
        State('search-type', 'value'),
        State({'type': 'projects-dropdown', 'page': '/hyperparameter-search'}, 'value'),
        State('sweep-name', 'value'),
        State('goal-metric', 'value'),
        State('goal-type', 'value'),
        State({'type': 'env-dropdown', 'page': '/hyperparameter-search'}, 'value'),
        State({'type':'gym-params', 'page':'/hyperparameter-search'}, 'children'),
        State({'type':'agent-type-selector', 'page':'/hyperparameter-search'}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL}, 'id'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'value'),
        State({'type': ALL, 'model': ALL, 'agent': ALL, 'index': ALL}, 'id'),
    ],
    prevent_initial_call=True,
    )
    def download_wandb_config(num_clicks, method, project, sweep_name, metric_name, metric_goal, env, env_params, agent_selection, all_values, all_ids, all_indexed_values, all_indexed_ids):
        # extract any additional gym env params
        params = utils.extract_gym_params(env_params)

        if num_clicks > 0:
            wandb_config = utils.create_wandb_config(
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
            config_json = json.dumps(wandb_config, indent=4)

            return dict(content=config_json, filename="wandb_config.json")
        
    @app.callback(
    Output("download-sweep-config", "data"),
    [Input("download-sweep-config-button", "n_clicks")],
    [
        State('num-sweeps', 'value'),
        State('num-episodes', 'value'),
        State('num-epochs', 'value'),
        State('num-cycles', 'value'),
        State('num-updates', 'value'),
        State({'type':'mpi', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'workers', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'num-sweep-agents', 'page':'/hyperparameter-search'}, 'value'),
        State({'type':'seed', 'page':'/hyperparameter-search'}, 'value'),
    ],
    prevent_initial_call=True,
    )
    def download_train_config(n_clicks, num_sweeps, num_episodes, num_epochs, num_cycles, num_updates, use_mpi, num_workers, num_agents, seed):
        if n_clicks > 0:
            # Create an empty dict for sweep_config.json
            sweep_config = {}
            # Add config options to run_config
            sweep_config['num_sweeps'] = num_sweeps
            sweep_config['num_episodes'] = num_episodes
            sweep_config['seed'] = seed if seed is not None else None

            # Add MPI config if not None else None
            sweep_config['use_mpi'] = use_mpi if use_mpi is not None else None
            sweep_config['num_workers'] = num_workers if num_workers is not None else None
            sweep_config['num_agents'] = num_agents if num_agents is not None else None
            
            # Update additional settings for HER agent
            sweep_config['num_epochs'] = num_epochs if num_epochs is not None else None
            sweep_config['num_cycles'] = num_cycles if num_cycles is not None else None
            sweep_config['num_updates'] = num_updates if num_updates is not None else 1

            config_json = json.dumps(sweep_config, indent=4)

            return dict(content=config_json, filename="sweep_config.json")