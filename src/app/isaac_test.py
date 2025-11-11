from isaaclab.app import AppLauncher
import torch
import numpy as np

# PARAMS
NUM_ENVS = 16
NUM_EPISODES = 100
DEVICE = "cuda:0"

# Launch the simulator in headless mode
app_launcher = AppLauncher(headless=True, device=DEVICE)
# simulation_app = app_launcher.app

# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from isaaclab.envs import ManagerBasedRLEnvCfg



print("success")