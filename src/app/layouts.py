from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.tools as tls

import dash_utils

# Placeholder for page content functions
def home(page):
    return html.Div("Home Page Content")

def build_agent(page):
    return html.Div([
        dbc.Container([
            dcc.Store(id="agent-params-store", data={}),
            html.H1("Build Agent", style={'textAlign': 'center'}),
            # Define tabs
            dbc.Tabs([
                # Tab 1: Environment
                dbc.Tab(label="Environment", children=[
                    dash_utils.env_dropdown_component(page),
                    html.Div(id={'type': 'gym-params', 'page': page}),  # empty for building agent
                    html.Div(id={'type': 'env-description', 'page': page}),
                    html.Img(id={'type': 'env-gif', 'page': page}, style={'width': '300px'})
                ]),

                # Tab 2: Agent
                dbc.Tab(label="Agent", children=[
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
                            {'label': 'TD3', 'value': 'TD3'},
                            {'label': 'Proximal Policy Optimization', 'value': 'PPO'},
                        ],
                        placeholder="Select Agent Type",
                    ),
                    html.Div(id='agent-parameters-inputs'),
                ]),

                # Tab 3: WANDB
                dbc.Tab(label="WANDB", children=[
                    html.Div(id='callback-selection'),
                    html.Div(id={
                        'type': 'wandb-login-container',
                        'page': page
                    }),
                    html.Div(id={'type': 'project-selection-container', 'page': page}),
                ]),

                # Tab 4: Build
                dbc.Tab(label="Build", children=[
                    html.Div(id='save-directory-selection'),
                    dbc.Button("Build Model", id='build-agent-button', n_clicks=0),
                    html.Div(id='build-agent-status')
                ]),
            ])
        ])
    ])

def train_agent(page):
   
    return dbc.Container([
        dcc.Store(id={"type":"run-params-store", "page":page}, data={}),
        html.H1("Train Agent", style={'textAlign': 'center'}),
        dcc.Store(id={'type':'agent-store', 'page':page}),
        dash_utils.upload_component(page),
        html.Div(id={'type':'output-agent-load', 'page':page}),
        # dash_utils.env_dropdown_component(page),
        # html.Div(id={'type':'env-description', 'page':page}),
        # html.Img(id={'type':'env-gif', 'page':page}, style={'width': '300px'}),
        # html.Div(id={'type':'gym-params', 'page':page}),
        # dash_utils.run_agent_settings_component(page),
        html.Div(id={'type': 'run-options', 'page': page}),
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
        dash_utils.video_carousel_component(page),  # Initially empty list of video paths
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
        dcc.Store(id={"type":"run-params-store", "page":page}, data={}),
        html.H1("Test Agent", style={'textAlign': 'center'}),
        dcc.Store(id={'type':'agent-store', 'page':page}),
        dash_utils.upload_component(page),
        html.Div(id={'type':'output-agent-load', 'page':page}),
        # dash_utils.env_dropdown_component(page),
        # html.Div(id={'type':'env-description', 'page':page}),
        # html.Img(id={'type':'env-gif', 'page':page}, style={'width': '300px'}),
        # html.Div(id={'type':'gym-params', 'page':page}),
        html.Div(id={'type': 'run-options', 'page': page}),
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
        dash_utils.video_carousel_component(page),  # Initially empty list of video paths
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
    right_column = dbc.Col(
        [
            html.H3('Wandb Project'),
            dash_utils.create_wandb_project_dropdown(page),
            html.Hr(),
            html.H3("Search Configuration"),
            dash_utils.env_dropdown_component(page),
            html.Div(id={'type':'gym-params', 'page':page}),
            html.Div(id={'type': 'env-description', 'page': page}),
            html.Img(id={'type': 'env-gif', 'page': page}, style={'width': '300px'}),
            dash_utils.create_sweep_options(),
            # dcc.Input(
            #     id='num-episodes',
            #     type='number',
            #     placeholder='Episodes per Sweep',
            # ),
            html.Div(
                id='agent-sweep-options',
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
            html.Div(
                id={
                    'type': 'sweep-options',
                    'page': page,
                },
                style={'display': 'none'},
                children=[
                    dcc.Input(
                        id={
                            'type': 'num-sweep-agents',
                            'page': page,
                        },
                        type='number',
                        placeholder="Number of Sweep Agents",
                        min=1,
                    ),
                ],
            ),
            dash_utils.create_seed_component(page),
            html.Hr(),
            dbc.Button("Download WandB Config", id="download-wandb-config-button", color="primary", className="mr-2"),
            dcc.Download(id="download-wandb-config"),
            dbc.Button("Download Sweep Config", id="download-sweep-config-button", color="primary", className="mr-2"),
            dcc.Download(id="download-sweep-config"),
        ],
        md=6,
    )

    left_column = dbc.Col(
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
                    {'label': 'TD3', 'value': 'TD3'},
                    {'label': 'Hindsight Experience Replay (DDPG)', 'value': 'HER_DDPG'},
                    {'label': 'Proximal Policy Optimization', 'value': 'PPO'},
                ],
                placeholder="Select Agent",
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
            dcc.Store(id={'type':'agent-store', 'page':page}),
            dash_utils.upload_component(page),
            html.Div(id={'type':'output-agent-load', 'page':page}),
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
            
            dash_utils.render_heatmap(page),
            dcc.Store(id={'type':'heatmap-data-store', 'page':page},
                  data={'formatted_data':None, 'matrix_data':None, 'bin_ranges':None}),
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
            dcc.Interval(
                id={
                    'type':'interval-component',
                    'page':page
                },
                interval=1*1000,  # in milliseconds
                n_intervals=0
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
                            dash_utils.create_wandb_login(page),
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
        dash_utils.create_wandb_project_dropdown(page),
        dash_utils.create_sweeps_dropdown(page),
        html.Button('Get Data', id={'type':'sweep-data-button', 'page':page}, n_clicks=0),
        html.Div(id={'type':'output-data-upload', 'page':page}),
        dcc.Dropdown(
            id={'type':'hyperparameter-selector', 'page':page},
            options=[],
            multi=True,
            placeholder="Select Hyperparameters",
        ),
        dash_utils.render_heatmap(page),
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