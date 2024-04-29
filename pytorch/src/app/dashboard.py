import json
import flask
from flask import request
import multiprocessing
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
# from celery import Celery


import layouts
import dash_callbacks
import utils
# from celery_config import broker_url



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# celery_app = Celery('tasks', broker=broker_url)
# celery_app.config_from_object('celery_config')

# dash_callbacks.register_callbacks(app)  # Function to register callbacks

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Build Agent", href="/build-agent")),
        dbc.NavItem(dbc.NavLink("Train Agent", href="/train-agent")),
        dbc.NavItem(dbc.NavLink("Test Agent", href="/test-agent")),
        dbc.NavItem(dbc.NavLink("Hyperparameter Search", href="/hyperparameter-search")),
        dbc.NavItem(dbc.NavLink("Co-Occurrence Analysis", href="/co-occurrence-analysis")),
        dbc.NavItem(dbc.NavLink("WandB Utils", href="/wandb-utils")),
    ],
    brand="RL Agent Training and Testing App",
    brand_href="/",
    color="primary",
    dark=True,
    sticky="top",
)

banner = html.Div([
    dbc.Container(
        [
            html.Img(src="https://img.freepik.com/free-photo/ai-machine-learning-hand-robot-human_587448-4824.jpg?t=st=1708716455~exp=1708720055~hmac=420ba1f82041709af10980bc6e9f9106b911414152f1a2b76154190a48e73e05&w=2000",
                     style={"width": "100%", "height": "150px"}),
        ],
        fluid=True,
        className="py-2",
    )
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
    # Create a multiprocessing manager
    manager = multiprocessing.Manager()
    # Create a shared dictionary to store data between processes
    shared_data = manager.dict()

    # Pass the shared data to the register_callbacks function
    dash_callbacks.register_callbacks(app, shared_data)
    app.run(debug=False, dev_tools_ui=False, dev_tools_props_check=False)