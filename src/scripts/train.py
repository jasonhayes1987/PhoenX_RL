import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import numpy as np
from pathlib import Path
from app.logging_config import get_logger
from app.agent_utils import load_agent

# import os
# from pathlib import Path
# import json
# import time
# import subprocess
# import ray
# import random
# import numpy as np
# import torch as T
# import wandb
# from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# from .distributed_trainer import DistributedAgents

# Configure logging
logger = get_logger(__name__, 'info')

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument(
    '--agent_dir',
    type=str,
    required=True,
    help='Path to the agent configuration directory')
parser.add_argument(
    '--load_weights',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='Load weights from the agent configuration directory (default: None)')
parser.add_argument(
    '--render_freq',
    type=int,
    default=None,
    help='Render frequency (default: None)'
)
parser.add_argument(
    '--num_episodes',
    type=int,
    default=None,
    help='Number of episodes (default: None)'
)
parser.add_argument(
    '--steps_per_learn',
    type=int,
    default=None,
    help='Steps per learn (default: None)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Seed (default: None)'
)
parser.add_argument(
    '--trajectories_per_update',
    type=int,
    default=None,
    help='Trajectories per update (Reinforce only) (default: None)'
)
# parser.add_argument('--distributed_workers', type=int, default=1, help='Number of distributed workers (default: 1)')
# parser.add_argument('--learner_device', type=str, default=None, help='Device for the learner (default: None)')
# parser.add_argument('--learner_num_cpus', type=int, default=1, help='Number of CPUs for the learner (default: 1)')
# parser.add_argument('--learner_num_gpus', type=float, default=1.0, help='Number of GPUs for the learner (default: 1)')
# parser.add_argument('--worker_device', type=str, default='cpu', help='Device for the workers (default: cpu)')
# parser.add_argument('--worker_num_cpus', type=int, default=1, help='Number of CPUs for the workers (default: 1)')
# parser.add_argument('--worker_num_gpus', type=float, default=0.0, help='Number of GPUs for the workers (default: 0)')
# parser.add_argument('--learn_iter', type=int, default=100, help='Learn frequency for the distributed workers (default: 100)')
# parser.add_argument('--sync_iter', type=int, default=10, help='Sync interval for the distributed workers (default: 10)')

args = parser.parse_args()
agent_config_dir = args.agent_dir

def train_agent(agent_config_dir):

    # wandb_initialized = False  # Track if wandb is initialized
    try:
        agent_config = json.load(open(Path(agent_config_dir) / 'config.json'))
        train_config = json.load(open(Path(agent_config_dir) / 'train_config.json'))
        agent_type = agent_config.get('agent_type')
        load_weights = args.load_weights if args.load_weights is not None else train_config.get('load_weights', False)
        render_freq = args.render_freq if args.render_freq is not None else train_config.get('render_freq', 0)
        num_episodes = args.num_episodes if args.num_episodes is not None else train_config.get('num_episodes')
        seed = args.seed if args.seed is not None else train_config.get('seed', np.random.randint(1000))

        assert agent_type in ['Reinforce', 'ActorCritic', 'DDPG', 'TD3', 'HER', 'PPO', 'SAC'], f"Unsupported agent type: {agent_type}"

        # Load Agent
        agent = load_agent(agent_config_dir, load_weights)

        if agent_type:
            if agent_type in ['DDPG', 'TD3', 'SAC']:
                # if args.distributed_workers > 1:
                #     distributed_agents = DistributedAgents(
                #         agent_config,
                #         args.distributed_workers,
                #         args.learner_device,
                #         args.learner_num_cpus,
                #         args.learner_num_gpus,
                #         args.worker_device,
                #         args.worker_num_cpus,
                #         args.worker_num_gpus,
                #         args.learn_iter,
                #     )
                #     futures = distributed_agents.train(sync_iter=args.sync_iter, num_episodes=num_episodes, num_envs=num_envs, seed=seed, render_freq=render_freq)
                #     if futures:
                #         ray.get(futures)
                # else:
                steps_per_learn = args.steps_per_learn if args.steps_per_learn is not None else train_config.get('steps_per_learn', 1)
                agent.train(num_episodes, steps_per_learn, render_freq, seed)

            elif agent_type == 'ActorCritic':
                agent.train(num_episodes, render_freq, seed)

            elif agent_type == 'Reinforce':
                trajectories_per_update = train_config['trajectories_per_update']
                agent.train(num_episodes, trajectories_per_update, render_freq, seed)

            elif agent_type == 'HER':
                num_epochs = train_config['num_epochs']
                num_cycles = train_config['num_cycles']
                num_updates = train_config['num_updates']
                # if args.distributed_workers > 1:
                #     distributed_agents = DistributedAgents(
                #         agent_config,
                #         args.distributed_workers,
                #         args.learner_device,
                #         args.learner_num_cpus,
                #         args.learner_num_gpus,
                #         args.worker_device,
                #         args.worker_num_cpus,
                #         args.worker_num_gpus,
                #         args.learn_iter
                #     )
                #     futures = distributed_agents.train(
                #         sync_iter=args.sync_iter,
                #         num_epochs=num_epochs,
                #         num_cycles=num_cycles,
                #         num_episodes=num_episodes,
                #         num_updates=num_updates,
                #         render_freq=render_freq,
                #         num_envs=num_envs,
                #         seed=seed
                #     )
                #     if futures:
                #         ray.get(futures)
                # else:
                # agent = load_agent(agent_config_dir, load_weights)
                agent.train(num_epochs, num_cycles, num_episodes, num_updates, num_envs, render_freq, seed)

                # Export a Chrome trace for manual viewing (optional) -- MOVED OUTSIDE THE WITH BLOCK
                # try:
                #     prof.export_chrome_trace(os.path.join(report_dir, "torch_profiler_logs/torch_profile.json"))
                # except RuntimeError as e:
                #     logger.error(f"Failed to export profiler trace: {e}")
            
            elif agent_type == 'PPO':
                timesteps = train_config['timesteps']
                trajectory_length = train_config['trajectory_length']
                batch_size = train_config['batch_size']
                learning_epochs = train_config['learning_epochs']
                agent.train(timesteps, trajectory_length, batch_size, learning_epochs, num_envs, render_freq, seed)

    except KeyError as e:
        logger.error(f"Missing configuration parameter: {str(e)}")
        raise

    except AssertionError as e:
        logger.error(str(e))
        raise

    except Exception as e:
        logger.exception("An unexpected error occurred during training")
        raise
    # finally:
    #     # Ensure the WandB run is properly finished if it was initialized
    #     if wandb_initialized:
    #         wandb.finish()
    #         logging.info("WandB run finished")

def main():
    """Main entry point for the training script."""
    try:
        train_agent(agent_config_dir)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file: {str(e)}")


if __name__ == '__main__':
    main()