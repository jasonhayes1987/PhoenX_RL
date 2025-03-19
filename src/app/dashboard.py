import json
import logging
import multiprocessing
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import ale_py


import layouts
import dash_callbacks
import dash_utils
import gymnasium as gym
import gymnasium_robotics as gym_robo



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Suppress INFO logs; show only errors

navbar = html.Div(
    [
        dbc.NavLink("Home", href="/", className="nav-link"),
        dbc.NavLink("Build Agent", href="/build-agent", className="nav-link"),
        dbc.NavLink("Train Agent", href="/train-agent", className="nav-link"),
        dbc.NavLink("Test Agent", href="/test-agent", className="nav-link"),
        dbc.NavLink("Hyperparameter Search", href="/hyperparameter-search", className="nav-link"),
        dbc.NavLink("Co-Occurrence Analysis", href="/co-occurrence-analysis", className="nav-link"),
        dbc.NavLink("WandB Utils", href="/wandb-utils", className="nav-link"),
    ],
    className="navbar",
)

banner = html.Div([
    html.Div([
        # html.Img(src='/assets/banner_edit.png', className='banner_img'),
        # html.Div("Phoenix AI", className="header-title"),
    ], className="banner"),
])

app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        banner,
        navbar,
        html.Div(id="page-content"),
        # dcc.Store(id='task-store')
    ],
    fluid=True,
)
if __name__ == "__main__":
    # Register Gym Robotics envs
    gym_robo.register_robotics_envs()
    # Register Ale Atari envs
    gym.register_envs(ale_py)
    # Create a multiprocessing manager
    manager = multiprocessing.Manager()
    # Create a shared dictionary to store data between processes
    shared_data = manager.dict()

    # Pass the shared data to the register_callbacks function
    dash_callbacks.register_callbacks(app, shared_data)
    app.run(debug=True, dev_tools_ui=True, dev_tools_props_check=True)