from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.tools as tls

import utils

# Placeholder for page content functions
def home(page):
    return html.Div("Home Page Content")

def build_agent(page):
    return html.Div([
        dbc.Container([
            html.H1("Build Agent", style={'textAlign': 'center'}),
            utils.env_dropdown_component(page),
            html.Div(id={'type':'gym-params', 'page':page}), # will be empty since just building agent
            html.Div(id={'type':'env-description', 'page':page}),
            html.Img(id={'type':'env-gif', 'page':page}, style={'width': '300px'}),
            dcc.Dropdown(
                id={
                    'type': 'agent-type-dropdown',
                    'page': page
                },
                options=[
                    {'label': 'Reinforce', 'value': 'Reinforce'},
                    {'label': 'Actor Critic', 'value': 'ActorCritic'},
                    {'label': 'Deep Deterministic Policy Gradient', 'value': 'DDPG'},
                    {'label': 'Hindsight Experience Replay (DDPG)', 'value': 'HER_DDPG'},
                    {'label': 'Proximal Policy Optimization', 'value': 'PPO'},
                ],
                placeholder="Select Agent Type",
            ),
            html.Div(id='agent-parameters-inputs'),
            html.Div(id='callback-selection'),
            html.Div(id={
                'type': 'wandb-login-container',
                'page': page,
            }),
            html.Div(id={'type':'project-selection-container', 'page':page}),
            html.Div(id='save-directory-selection'),
            dbc.Button("Build Model", id='build-agent-button', n_clicks=0),
            html.Div(id='build-agent-status')
        ])
    ])

def train_agent(page):
   
    return dbc.Container([
        html.H1("Train Agent", style={'textAlign': 'center'}),
        dcc.Store(id={'type':'agent-store', 'page':page}),
        utils.upload_component(page),
        html.Div(id={'type':'output-agent-load', 'page':page}),
        utils.env_dropdown_component(page),
        html.Div(id={'type':'env-description', 'page':page}),
        html.Img(id={'type':'env-gif', 'page':page}, style={'width': '300px'}),
        html.Div(id={'type':'gym-params', 'page':page}),
        utils.run_agent_settings_component(page),
        dbc.Button("Start",
            id={
                'type': 'start',
                'page': page,
            },
            n_clicks=0
        ),
        # dcc.Loading(
        #     id={
        #         'type':'loading',
        #         'page':page,
        #         },
        #     type="default", 
        #     children=html.Div(
        #         id={
        #             'type':'loading-output',
        #             'page':page,
        #         },
        #     ),
        # ),
        dcc.Store(
            id={
                'type':'storage',
                'page':page,
            },
            data={
                'status': " Training Initiated...",
                'progress': 0,
                'data': {},
            },
        ),
        html.Div(id={'type':'status', 'page':page}),
        utils.video_carousel_component(page),  # Initially empty list of video paths
        dcc.Interval(
            id={
                'type':'interval-component',
                'page':page
            },
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
    ], fluid=True)

def test_agent(page):
    
    return dbc.Container([
        html.H1("Test Agent", style={'textAlign': 'center'}),
        dcc.Store(id={'type':'agent-store', 'page':page}),
        utils.upload_component(page),
        html.Div(id={'type':'output-agent-load', 'page':page}),
        utils.env_dropdown_component(page),
        html.Div(id={'type':'env-description', 'page':page}),
        html.Img(id={'type':'env-gif', 'page':page}, style={'width': '300px'}),
        html.Div(id={'type':'gym-params', 'page':page}),
        utils.run_agent_settings_component(page),
        dbc.Button("Start",
            id={
                'type': 'start',
                'page': page,
            },
            n_clicks=0
        ),
        dcc.Store(
            id={
                'type':'storage',
                'page':page,
            },
            data={
                'status': "Testing Initiated...",
                'progress': 0,
                'data': {},
            },
        ),
        html.Div(id={'type':'status', 'page':page}),
        utils.video_carousel_component(page),  # Initially empty list of video paths
        dcc.Interval(
            id={
                'type':'interval-component',
                'page':page
            },
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
    ], fluid=True)


def hyperparameter_search(page):
    left_column = dbc.Col(
        [
            html.H3('Wandb Project'),
            utils.generate_wandb_project_dropdown(page),
            html.Hr(),
            html.H3("Search Configuration"),
            utils.env_dropdown_component(page),
            html.Div(id={'type':'gym-params', 'page':page}),
            html.Div(id={'type': 'env-description', 'page': page}),
            html.Img(id={'type': 'env-gif', 'page': page}, style={'width': '300px'}),
            html.H6('Search Method'),
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
            dcc.Input(
                id='num-episodes',
                type='number',
                placeholder='Episodes per Sweep',
            ),
            html.Div(
                id='her-options-hyperparam',
                hidden=True,
                children=[
                    dcc.Input(
                        id='num-epochs',
                        type='number',
                        placeholder='Epochs per Sweep',
                    ),
                    dcc.Input(
                        id='num-cycles',
                        type='number',
                        placeholder='Cycles per Episode',
                    ),
                    dcc.Input(
                        id='num-updates',
                        type='number',
                        placeholder='Updates per Episode',
                    ),
                ]
            ),
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
            utils.generate_seed_component(page),
            html.Hr(),
        ],
        md=6,
    )

    right_column = dbc.Col(
        [
            html.H3('Agent Configuration'),
            dcc.Dropdown(
                id={
                    'type':'agent-type-selector',
                    'page':page,
                    },
                options=[
                    {'label': 'Reinforce', 'value': 'Reinforce'},
                    {'label': 'Actor Critic', 'value': 'ActorCritic'},
                    {'label': 'Deep Deterministic Policy Gradient', 'value': 'DDPG'},
                    {'label': 'Hindsight Experience Replay (DDPG)', 'value': 'HER_DDPG'},
                    {'label': 'Proximal Policy Optimization', 'value': 'PPO'},
                   
                ],    
                value=[],
                multi=True,
                placeholder="Select Agent(s)",
            ),
            html.Div(
                id='agent-hyperparameters-inputs',
                children=[
                    dcc.Tabs(
                        id='agent-options-tabs',
                    ),
                ]
            ),
            html.Hr(),
        ]
    )

    row = dbc.Row([left_column, right_column])

    inputs = html.Div(
        id={
            'type': 'hyperparameter-inputs',
            'page': page,
        },
        hidden=True,
        children=[
            row,
            dbc.Button("Start Sweep",
                id={
                    'type': 'start',
                    'page': page,
                },
                n_clicks=0
            ),
            dcc.Store(
                id={
                    'type': 'storage',
                    'page': page,
                },
                data={
                    'status': "Sweep Initiated...",
                    'progress': 0,
                    'data': {},
                },
            ),
            html.Div(id={'type': 'status', 'page': page}),
            html.Div(
                id={
                    'type': 'hidden-div-hyperparam',
                    'page': page,
                },
                style={'display': 'none'}
            ),
            dcc.Dropdown(
                id={'type':'hyperparameter-selector', 'page':page},
                options=[],
                multi=True,
                placeholder="Select Hyperparameters",
            ),
            dcc.Interval(
                id='update-hyperparam-selector',
                interval=1*1000,
                n_intervals=0
            ),
            
            utils.render_heatmap(page),
            dcc.Store(id={'type':'heatmap-data-store', 'page':page}),
            dcc.Interval(
                id='heatmap-store-data-interval',
                interval=10*1000,
                n_intervals=0
            ),
            dcc.Interval(
                id='start-fetch-process-interval',
                interval=10*1000,
                n_intervals=0,
                max_intervals=1,
            ),
            dcc.Interval(
                id='start-matrix-process-interval',
                interval=10*1000,
                n_intervals=0,
            ),
            html.Div(id='hidden-div-fetch-process', style={'display': 'none'}),
            html.Div(id='hidden-div-matrix-process', style={'display': 'none'}),
        ]
    )

    return dbc.Container(
        [
            html.Div(
                [
                    html.H1("Hyperparameter Search", style={'textAlign': 'center'}),
                    html.Div(
                        [  
                            utils.generate_wandb_login(page),
                        ],
                        style={'textAlign': 'center'},
                    ),
                ],
            ),
            inputs
        ]
    )


def co_occurrence_analysis(page):
    return html.Div([
        html.H1("Co-Occurrence Analysis", style={'textAlign': 'center'}),
        utils.generate_wandb_project_dropdown(page),
        utils.generate_sweeps_dropdown(page),
        html.Button('Get Data', id={'type':'sweep-data-button', 'page':page}, n_clicks=0),
        html.Div(id={'type':'output-data-upload', 'page':page}),
        dcc.Dropdown(
            id={'type':'hyperparameter-selector', 'page':page},
            options=[],
            multi=True,
            placeholder="Select Hyperparameters",
        ),
        utils.render_heatmap(page),
        dcc.Store(id={'type':'heatmap-data-store', 'page':page},
                  data={'formatted_data':None, 'matrix_data':None, 'bin_ranges':None}),
    ])


def wandb_utils(page):
    return html.Div([
        dbc.Container([
            html.H1("wandb-utils"),
            html.Div(id='wandb-utils'),
        ])
    ])