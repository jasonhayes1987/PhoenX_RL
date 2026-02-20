"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
from abc import abstractmethod
import json
import os
from typing import Optional, Dict, List, TypeAlias
from pathlib import Path
import time
from collections import deque
import logging

from torch.return_types import all_return_types
from .logging_config import get_logger
import copy
from .encoder import CustomJSONEncoder, serialize_env_spec
# from moviepy.editor import ImageSequenceClip
from umap import UMAP
import plotly.express as px

from .icm import ICM
from .rl_callbacks import WandbCallback, Callback
from .rl_callbacks import load as callback_load
from .models import select_policy_model, StochasticContinuousPolicy, StochasticDiscretePolicy, ValueModel, ContinuousCritic, DiscreteCritic, ActorModel
from .schedulers import ScheduleWrapper
from .adaptive_kl import AdaptiveKL
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, Buffer
from .normalizer import Normalizer, SharedNormalizer
from .noise import Noise, NormalNoise, UniformNoise, OUNoise
import wandb
from . import wandb_support
from .torch_utils import set_seed, get_device, move_to_device, VarianceScaling_
from .env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
# from utils import render_video, build_env_wrapper_obj, check_for_inf_or_NaN
from .utils import *

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Beta, Normal, kl_divergence
from torch.profiler import profile
import gymnasium as gym
import gymnasium_robotics
from gymnasium.envs.registration import EnvSpec
import numpy as np

from isaaclab.app import AppLauncher


from app.agent_utils import load_agent, get_agent_class_from_type, compute_n_step_return, compute_full_return, compute_gae


ActionOutput: TypeAlias = tuple[np.ndarray|T.Tensor, T.Tensor|None, Distribution|None]

# Agent class
class Agent:
    """Base class for all RL agents."""

    def __init__(self,
                 env: EnvWrapper,
                 callbacks: Optional[list[Callback]] = None,
                 obs_key: str | None = None,
                 goal_key: str | None = None,
                 save_dir: str = "models/",
                 device: Optional[str | T.device] = None,
                 log_level: str = 'info',
                 **kwargs):

        self.logger = get_logger(__name__, log_level)
        self.kwargs = kwargs
        try:
            self.save_dir = self._setup_save_dir(save_dir)
            self.env = env
            self.callbacks = self._initialize_callbacks(callbacks)
            self.obs_key = obs_key
            self.goal_key = goal_key
            self.device = get_device(device)

            # Set internal attributes
            self._initialized = False
            # self._distributed = False
           
        except Exception as e:
            self.logger.error(f"Error in Agent init: {e}", exc_info=True)

    @property
    def base_agent(self):
        """Return the base agent"""
        return self

    def _setup_save_dir(self, save_dir: str):
        """
        Setup the save directory for the agent.
        If save_dir doesn't end with the agent's name, append it.
        
        Args:
            save_dir (str): Base save directory path
        """
        # agent_name = self.__class__.__name__.lower()
        # if f"/{agent_name}/" not in save_dir:
        #     return save_dir + f"/{agent_name}/"
        # else:
        return save_dir

    def _initialize_callbacks(self, callbacks):
        """
        Initialize and configure callbacks for logging and monitoring.

        Args:
            callbacks (list): List of callback objects.
        """
        try:
            if callbacks:
                for callback in callbacks:
                    callback._config(self)
                    if isinstance(callback, WandbCallback):
                        self._wandb = True
            else:
                self._wandb = False

            return callbacks
        except Exception as e:
            self.logger.error(f"Error initializing callbacks: {e}", exc_info=True)

    def _initialize_run(self, seed: int | None = None, training: bool = True, **kwargs):
        """
        Initializes the environment, seeds, and tracking variables for training.
        Args:
            seed (int | None): Seed for the environment. If None, a random seed is used.
            training (bool): Whether the agent is training or testing.
            **kwargs: Additional keyword arguments.
        """
        if getattr(self, '_initialized', False):
            return
        
        # Set models to train mode if training, else evaluation mode
        for name in ['actor_model', 'critic_model', 'critic_model_a', 'critic_model_b',
                    'value_model', 'policy_model']:
            model = getattr(self.base_agent, name, None)
            if model and training: model.train()
            elif model: model.eval()

        # Set target models to eval mode
        for name in ['target_actor_model', 'target_critic_model', 'target_critic_model_a', 'target_critic_model_b',
                     'target_value_model', 'target_policy_model']:
            model = getattr(self.base_agent, name, None)
            if model: model.eval()
        
        # Set internal attributes
        seed = seed if seed else np.random.randint(1000)
        set_seed(seed)
        states, _ = self.base_agent.env.reset()

        # Add initial states to normalizer if it exists and training
        if training:
            if getattr(self.base_agent, 'state_normalizer', None):
                if isinstance(states, dict):
                    self.base_agent.state_normalizer.add(T.tensor(states[self.base_agent.obs_key], dtype=T.float32, device=self.base_agent.state_normalizer.device.type))
                else:
                    self.base_agent.state_normalizer.add(T.tensor(states, dtype=T.float32, device=self.base_agent.state_normalizer.device.type))

            if getattr(self.base_agent, 'goal_normalizer', None):
                self.base_agent.goal_normalizer.add(T.tensor(states[self.base_agent.goal_key], dtype=T.float32, device=self.base_agent.goal_normalizer.device.type))

        # Set callbacks
        if self.base_agent.callbacks:
            config = self.get_config()
            config.update({'num_envs': self.env.num_envs, 'seed': seed})
            config.update(kwargs)

            models = [model for model in [getattr(self.base_agent, "actor_model", None),
                                          getattr(self.base_agent, "critic_model", None),
                                          getattr(self.base_agent, "critic_model_a", None),
                                          getattr(self.base_agent, "critic_model_b", None),
                                          getattr(self.base_agent, "value_model", None),
                                          getattr(self.base_agent, "policy_model", None),
                                          self.base_agent.curiosity if hasattr(self.base_agent, 'curiosity') else None] if model is not None]
            
            func_name = 'on_train_begin' if training else 'on_test_begin'

            for callback in self.base_agent.callbacks:
                callback_func = getattr(callback, func_name)
                if isinstance(callback, WandbCallback):
                    run_number = callback.run_name.split("-")[-1] if callback.run_name else None
                    if func_name == 'on_train_begin':
                        callback_func(config, run_number, tuple(models))
                    else:
                        callback_func(config, run_number)
                else:
                    callback_func(config)
        self._initialized = True

        # Return dict with initialized params
        return {
            # 'env': env,
            'step': 0,
            'states': states,
            'best_reward': -T.inf,
            'completed_episodes': T.zeros(self.env.num_envs, dtype=T.int32, device=self.device),
            'episode_scores': T.zeros(self.env.num_envs, dtype=T.float32, device=self.device),
            'score_history': deque(maxlen=100)
        }

    def extract_states_goals(self, states: np.ndarray | T.Tensor | dict | list[dict]):
        """Extract the states and goals from the passed states argument and returns them as Tensors.
        
        Args:
            states (np.ndarray | T.Tensor | dict | list[dict]): States to extract from.
        
        Returns:
            tuple: Tuple of states and goals as Tensors.
        """
        if isinstance(states, list):
            obs_list = []
            goals_list = []
            for step_data in states:
                if isinstance(step_data, dict):
                    if not self.obs_key:
                        raise ValueError("Goal-aware observation spaces require obs_key to be set")
                    step_obs = step_data.get(self.obs_key)
                    if self.goal_key:
                        step_goal = step_data.get(self.goal_key)
                    else:
                        step_goal = None
                else:
                    break

                # Convert to tensor if needed
                if not isinstance(step_obs, T.Tensor):
                    step_obs = T.tensor(step_obs, dtype=T.float32, device=self.device)
                if self.goal_key:
                    if not isinstance(step_goal, T.Tensor):
                        step_goal = T.tensor(step_goal, dtype=T.float32, device=self.device)
                    goals_list.append(step_goal)
                obs_list.append(step_obs)
            obs = T.stack(obs_list, dim=0)
            
            if self.goal_key:
                goals = T.stack(goals_list, dim=0)
            else:
                goals = None

        elif isinstance(states, dict):
            if not self.obs_key:
                raise ValueError("Goal-aware observation spaces require obs_key to be set")
            obs = states.get(self.obs_key)
            if self.goal_key:
                goals = states.get(self.goal_key)
            else:
                goals = None
        else:
            obs = states
            goals = None

        if not isinstance(obs, T.Tensor):
            obs = T.tensor(obs, dtype=T.float32, device=self.device)
        if goals is not None and not isinstance(goals, T.Tensor):
            goals = T.tensor(goals, dtype=T.float32, device=self.device)
        
        return obs, goals

    def _preprocess_inputs(self, states: np.ndarray | T.Tensor | dict | list[dict]):
        """Preprocess the inputs for the agent by separating states into observation and desired goal
        if the observation space is goal-aware.

        Args:
            states (np.ndarray | dict | list[dict]): States to preprocess.

        Returns:
            tuple: Tuple of observation and desired goal as Tensors.
        """
        obs, goals = self.extract_states_goals(states)
        if self.state_normalizer:
            obs = self.state_normalizer.normalize(obs)
        if hasattr(self, 'goal_normalizer') and self.goal_normalizer:
            goals = self.goal_normalizer.normalize(goals)
        
        return obs, goals

    def _step(self, env: EnvWrapper, step: int, states: np.ndarray, num_episodes: int, episode_scores: np.ndarray, completed_episodes: np.ndarray, score_history: deque, learn: bool = True, training: bool = True):
        """Step function for the agent."""
        raise NotImplementedError("Subclasses must implement _step.")

    def render_episode(self, episode: int, step: int, context: str = 'train', num_envs:int=1, **kwargs):
        """Render a single episode in a temporary environment, collect metrics, create video, and log to wandb.
            **Only works with Gymnasium Environments**
        Args:
            episode (int): Episode number to render.
            step (int): Current training/testing step (for wandb logging).
            context (str): Context of the episode (train or test). Defaults to 'train'.
            **kwargs: Additional keyword arguments to pass to the environment wrapper to override original values.
        """
        if isinstance(self.base_agent.env, IsaacSimWrapper):
            raise ValueError("Rendering episodes is not supported for IsaacSim environments. Test using one environment with render_mode='gui' instead.")
        self.base_agent.logger.info(f"Rendering episode {episode} in {context} with kwargs: {kwargs}")
        env = self.base_agent.env.clone(num_envs, **kwargs)
        states, _ = env.reset()
        frames = []
        done = False
        local_step = 0
        episode_reward = 0

        agent_type = self.base_agent.__class__.__name__

        while not done:
            local_step += 1
            obs, goals = self._preprocess_inputs(states)
            # Condition get_action call on agent type
            # if agent_type in ['ActorCritic', 'Reinforce', 'SAC']:
            #     actions, _, _ = self.base_agent.get_action(obs, goals, context='test')
            # elif agent_type == 'PPO':
            #     actions, _ = self.base_agent.get_action(obs, goals, context='test')
            # else:
            actions, _, _ = self.base_agent.get_action(obs, goals, context='test')

            if hasattr(self, 'action_adapter'):
                actions = self.action_adapter(env, actions)
            

            actions = env.format_actions(actions)
            states, rewards, dones, _ = env.step(actions)
            episode_reward += rewards[0]
            frame = env.render_frame()
            frames.append(frame)
            done = dones[0]
            # states = next_states

        metrics = {
            'render_episode_reward': episode_reward,
            'render_episode_length': local_step
        }

        render_video(frames, episode, self.save_dir, context)
        video_path = os.path.join(self.save_dir, f"renders/{context}/episode_{episode}.mp4")
        if self.base_agent.callbacks:
            for callback in self.base_agent.callbacks:
                if isinstance(callback, WandbCallback):
                    caption = (f"{context.capitalize()} render episode {episode}")
                    wandb.log({
                        f"{context}_video": wandb.Video(video_path, caption=caption, format="mp4"),
                        **metrics
                    }, step=step)
        env.close()


    @abstractmethod
    def _distributed_learn(self, *args, **kwargs):
        """Handle distributed learning for both on-policy and off-policy agents."""
        raise NotImplementedError("Subclasses must implement _distributed_learn.")
    
    @abstractmethod
    def get_parameters(self):
        """Return a dictionary of model parameters: {model_name: params}."""
        raise NotImplementedError("Subclasses must implement get_parameters.")

    @abstractmethod
    def apply_parameters(self, params):
        """Apply the provided parameters to the agent's models."""
        raise NotImplementedError("Subclasses must implement apply_parameters.")

    def clone(self, device: Optional[str | T.device] = None) -> 'Agent':
        """
        Create a deep copy of the agent, optionally moving it to a new device.
        
        Args:
            device (str or T.device, optional): Target device for the cloned agent. If None, uses the current device.
        
        Returns:
            Agent: A cloned instance of the agent with all components correctly copied and moved.
        """
        # Perform a deep copy of the agent
        clone = copy.deepcopy(self)

        if clone.__class__.__name__ == 'HER':
            cloned_agent = clone.agent
        else:
            cloned_agent = clone

        if device:
            # Determine the target device
            target_device = get_device(device)
            device_str = str(target_device).split(':')[0]  # Get 'cuda' or 'cpu' part
            # Update the cloned agent's device attribute
            cloned_agent.device = target_device
            
            # Explicitly update model configurations to use the target device
            if hasattr(cloned_agent, '_config') and isinstance(cloned_agent._config, dict):
                # Update top-level device
                if 'device' in cloned_agent._config:
                    cloned_agent._config['device'] = device_str
                    
                # Update devices in model configs
                for model_key in ['actor_model', 'critic_model', 'critic_model_a', 'critic_model_b', 'value_model', 'policy_model']:
                    if model_key in cloned_agent._config and isinstance(cloned_agent._config[model_key], dict):
                        if 'device' in cloned_agent._config[model_key]:
                            cloned_agent._config[model_key]['device'] = device_str
            
            # Explicitly update model device attributes
            for model_name in ['actor_model', 'critic_model', 'critic_model_a', 'critic_model_b', 'value_model', 'policy_model']:
                if hasattr(cloned_agent, model_name):
                    model = getattr(cloned_agent, model_name)
                    if hasattr(model, 'device'):
                        setattr(model, 'device', target_device)
            
            # Explicitly handle target networks for algorithms like DDPG and TD3
            for attr_name in dir(cloned_agent):
                # Look for attributes starting with 'target_' that might be models
                if attr_name.startswith('target_') and hasattr(cloned_agent, attr_name):
                    target_model = getattr(cloned_agent, attr_name)
                    # Check if it has a device attribute to update
                    if hasattr(target_model, 'device'):
                        setattr(target_model, 'device', target_device)
            
            # Now use move_to_device to handle all tensors and other components
            cloned_agent = move_to_device(cloned_agent, target_device)
        
        if clone.__class__.__name__ == 'HER':
            clone.agent = cloned_agent
        else:
            clone = cloned_agent

        return clone

    @abstractmethod
    def get_action(self, states: np.ndarray|T.Tensor, goals: np.ndarray|T.Tensor|None=None, step: int|None=None, context: str = 'train')->tuple[np.ndarray|T.Tensor, T.Tensor, Distribution|None]:
        """Returns an action given a state."""
        raise NotImplementedError("Subclasses must implement get_action.")

    @abstractmethod
    def train(self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
        """Trains the model for 'episodes' number of episodes."""
        raise NotImplementedError("Subclasses must implement train.")
    
    @abstractmethod
    def learn(self):
        """Updates the model."""
        raise NotImplementedError("Subclasses must implement learn.")

    @abstractmethod
    def test(self, num_episodes=None, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""
        raise NotImplementedError("Subclasses must implement test.")

    @abstractmethod
    def get_config(self):
        raise NotImplementedError("Subclasses must implement get_config.")

    @abstractmethod
    def save(self):
        """Saves the model."""
        raise NotImplementedError("Subclasses must implement save.")
    
    @classmethod
    def load(cls, folder: str = "models"):
        """Loads the model."""
        raise NotImplementedError("Subclasses must implement load.")


class ActorCritic(Agent):
    """Actor Critic Agent."""

    def __init__(
        self,
        env: EnvWrapper,
        policy_model: StochasticDiscretePolicy,
        value_model: ValueModel,
        discount: float=0.99,
        policy_trace_decay: float=0.0,
        value_trace_decay: float=0.0,
        entropy_coefficient: float=0.01,
        entropy_schedule: Optional[ScheduleWrapper] = None,
        gae_coefficient: float=0.95,
        trajectory_length: int=10,
        state_normalizer: Optional[Normalizer] = None,
        advantage_normalizer: Normalizer|None=None,
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models/",
        device: Optional[str | T.device] = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, save_dir=save_dir, device=device, log_level=log_level)
            self.policy_model = policy_model
            self.value_model = value_model
            self.discount = discount
            self.policy_trace_decay = policy_trace_decay
            self.value_trace_decay = value_trace_decay
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.advantage_normalizer = advantage_normalizer
            self.gae_coefficient = gae_coefficient
            self.trajectory_length = trajectory_length
            self.state_normalizer = state_normalizer
            
            # instantiate trajectory buffers for rollouts
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.log_probs = []
            self.entropies = []
            
            # instantiate and set policy and value traces
            self.policy_trace = []
            self.value_trace = []
            self._set_traces()
        except Exception as e:
            self.logger.error(f"Error in ActorCritic init: {e}", exc_info=True)

    def _set_traces(self):
        for weights in self.policy_model.parameters():
            self.policy_trace.append(T.zeros_like(weights, device=self.device))

        for weights in self.value_model.parameters():
            self.value_trace.append(T.zeros_like(weights, device=self.device))

    def _update_traces(self):
        with T.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                self.policy_trace[i] = (self.discount * self.policy_trace_decay * self.policy_trace[i]) + weights.grad

            for i, weights in enumerate(self.value_model.parameters()):
                self.value_trace[i] = (self.discount * self.value_trace_decay * self.value_trace[i]) + weights.grad

    def get_action(self, states:np.ndarray|T.Tensor, step=None, context='train')->tuple[np.ndarray|T.Tensor, T.Tensor, Distribution|None]:
        states, _ = self._preprocess_inputs(states)
        if context == 'train':
            dist, logits = self.policy_model(states)
            actions = dist.sample()
            actions = actions.detach()
            return actions, logits, dist
        elif context == 'test':
            with T.no_grad():
                dist, logits = self.policy_model(states)
                actions = dist.mode
                return actions, logits, dist

    def _step(self, step: int, states: np.ndarray | T.Tensor, max_episodes: int, episode_scores: np.ndarray,
              completed_episodes: np.ndarray, score_history: deque[float], best_reward: float, training: bool = True):
        """
        Performs a single step of training/testing.
        
        Args:
        step: int: The current step.
        states: np.ndarray | T.Tensor: The current states.
        max_episodes: int: The maximum number of episodes.
        episode_scores: np.ndarray: The current episode scores.
        completed_episodes: np.ndarray: The current completed episodes.
        score_history: deque[float]: The current score history.
        best_reward: float: The current best reward.
        training: bool: Whether the step is for training or testing.
        
        Returns:
        dict: A dictionary containing the step metrics.
        """
        step_log = {}
        episode_logs = []

        actions, logits, dist = self.get_action(states, context='train' if training else 'test')
        actions = self.env.format_actions(actions)
        next_states, rewards, dones, infos = self.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        states, actions, rewards, next_states, dones = (
            T.tensor(states, dtype=T.float32, device=self.device) if isinstance(states, np.ndarray) else states,
            T.tensor(actions, dtype=T.long, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )

        # Get log probs and entropy values from distribution
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy().to(self.device)

        # Add transitions to rollout buffers
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.entropies.append(entropies)

        episode_scores += rewards.flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': rewards.mean().item()
        })
        
        if training:
            # Update normalizer if state_normalizer
            if self.state_normalizer:
                self.state_normalizer.add(next_states)
            # Check if trajectory length has been reached
            if len(self.states) >= self.trajectory_length:
                # Learn from the trajectory buffers
                learn_metrics = self.learn()
                step_log.update(learn_metrics)
                # Clear the trajectory buffers
                self.states = []
                self.actions = []
                self.rewards = []
                self.next_states = []
                self.dones = []
                self.log_probs = []
                self.entropies = []

        done_episodes = T.nonzero(dones, as_tuple=False).flatten()

        for i in done_episodes:
            # Increment completed episodes for env by 1
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': round(float(episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }
            if training:
                # check if best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save()
                episode_log.update({
                    'best_reward': round(float(best_reward), 2),
                    'best': 1 if avg_reward > best_reward else 0
                })
            episode_logs.append(episode_log)

        return{
        'episode_scores': episode_scores,
        'completed_episodes': completed_episodes,
        'score_history': score_history,
        'next_states': next_states,
        'step_log': step_log,
        'episode_logs': episode_logs,
        'done': completed_episodes.sum() >= max_episodes
    }    
              

    def train(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Trains the model for 'episodes' number of episodes."""
        
        init_dict = self._initialize_run(seed=seed, num_episodes=num_episodes)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        
        while completed_episodes.sum() < num_episodes:
            # # If distributed, sync to shared agent
            # if self._distributed and self._step % self._sync_iter == 0:
            #     params = self.get_parameters()
            #     self.apply_parameters(params)
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward)
            # Update states, episode scores, completed episodes, and score history
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    # Call the test function to render an episode
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)

        self.env.close()

    def learn(self):

        learn_metrics = {}

        self.policy_model.optimizer.zero_grad()
        self.value_model.optimizer.zero_grad()

        # Convert trajectory buffers to tensors
        states = T.stack(self.states, dim=0)
        actions = T.stack(self.actions, dim=0)
        rewards = T.stack(self.rewards, dim=0)
        next_states = T.stack(self.next_states, dim=0)
        dones = T.stack(self.dones, dim=0)
        log_probs = T.stack(self.log_probs, dim=0)
        entropies = T.stack(self.entropies, dim=0)

        # Get entropy coefficient
        entropy_coefficient = self.entropy_coefficient
        if self.entropy_schedule:
            entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get trajectory length and num_envs dims
        trajectory_length, num_envs, obs_dim = states.shape

        # Normalize states if self.normalize_inputs
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)

        state_values = self.value_model(states.view(trajectory_length * num_envs, obs_dim))
        next_state_values = self.value_model(next_states.view(trajectory_length * num_envs, obs_dim)).detach()
        td_errors = (
            rewards + self.discount * next_state_values.reshape(trajectory_length, num_envs, -1).squeeze() * (1 - dones) - state_values.reshape(trajectory_length, num_envs, -1).squeeze())
        
        advantages = compute_gae(td_errors, dones, self.discount, self.gae_coefficient, device=self.device)
        if self.advantage_normalizer:
            self.advantage_normalizer.add(advantages.view(trajectory_length * num_envs, 1))
            advantages = self.advantage_normalizer.normalize(advantages.view(trajectory_length * num_envs, 1))
            advantages = advantages.view(trajectory_length, num_envs)
        returns = (advantages.detach() + state_values.reshape(trajectory_length, num_envs, -1).squeeze())
        value_loss = (state_values.reshape(trajectory_length, num_envs, -1).squeeze() - returns.detach()).square().mean()
        # value_loss = returns.square().mean()
        # value_loss = advantages.square().mean()
        # value_loss = td_errors.square().mean()

        value_loss.backward()

        policy_loss = -(log_probs * advantages.detach() + entropy_coefficient * entropies).mean()
        policy_loss.backward()

        self._update_traces()

        #copy traces to weight gradients
        with T.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                weights.grad = self.policy_trace[i]

            for i, weights in enumerate(self.value_model.parameters()):
                weights.grad = self.value_trace[i]

        self.value_model.optimizer.step()
        self.policy_model.optimizer.step()

        # Update entropy schedule
        if self.entropy_schedule:
            self.entropy_schedule.step()

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'temporal_difference': td_errors.mean().item(),
            'advantages': advantages.mean().item(),
            'returns': returns.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
        })

        return learn_metrics

    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""
        
        init_dict = self._initialize_run(seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, training = False)
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])

            render = True
            for i, episode_log in enumerate(step_result['episode_logs']):
                # Print complete episode metrics to console
                print(f"Testing Environment {i}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[i] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)
            
                if render and render_freq > 0 and completed_episodes.sum() % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "discount": self.discount,
            "policy_trace_decay": self.policy_trace_decay,
            "value_trace_decay": self.value_trace_decay,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "gae_coefficient": self.gae_coefficient,
            "trajectory_length": self.trajectory_length,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir
        }


    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy_model.save(self.save_dir)
        self.value_model.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.advantage_normalizer:
            self.advantage_normalizer.save(self.save_dir + "advantage_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        policy_model = StochasticDiscretePolicy.load(config_dir, 'policy_model', load_weights, env=env_wrapper)
        value_model = ValueModel.load(config_dir, 'value_model', load_weights, env=env_wrapper)
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        advantage_normalizer = Normalizer.load(config["advantage_normalizer"], config["save_dir"] + "advantage_normalizer.pt") if config["advantage_normalizer"] else None
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None
        agent = cls(
            env=env_wrapper,
            policy_model=policy_model,
            value_model=value_model,
            discount=config["discount"],
            policy_trace_decay=config["policy_trace_decay"],
            value_trace_decay=config["value_trace_decay"],
            entropy_coefficient=config["entropy_coefficient"],
            entropy_schedule=ScheduleWrapper(config["entropy_schedule"]) if config["entropy_schedule"] else None,
            gae_coefficient=config["gae_coefficient"],
            trajectory_length=config["trajectory_length"],
            state_normalizer=state_normalizer,
            advantage_normalizer=advantage_normalizer,
            callbacks=callbacks,
            save_dir=config["save_dir"],
        )

        return agent


class Reinforce(Agent):
    def __init__(
        self,
        env: EnvWrapper,
        policy: StochasticDiscretePolicy,
        value: ValueModel|None = None,
        discount: float = 0.99,
        state_normalizer: Optional[Normalizer] = None,
        entropy_coefficient: float = 0.01,
        entropy_schedule: Optional[ScheduleWrapper] = None,
        max_trajectory_length: int = 1000,
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: str = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, save_dir=save_dir, device=device, log_level=log_level)
            self.policy = policy
            self.value = value
            self.discount = discount
            self.state_normalizer = state_normalizer
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.max_trajectory_length = max_trajectory_length

            # Instantiate internal buffers
            obs_dim = self.env.single_observation_space.shape[0]
            self.states_buffer = T.zeros((self.max_trajectory_length, self.env.num_envs, obs_dim), dtype=T.float32, device=self.device)
            self.actions_buffer = T.zeros((self.max_trajectory_length, self.env.num_envs), dtype=T.long, device=self.device)
            self.rewards_buffer = T.zeros((self.max_trajectory_length, self.env.num_envs), dtype=T.float32, device=self.device)
            self.next_states_buffer = T.zeros((self.max_trajectory_length, self.env.num_envs, obs_dim), dtype=T.float32, device=self.device)
            self.dones_buffer = T.zeros((self.max_trajectory_length, self.env.num_envs), dtype=T.int8, device=self.device)
            # Track step index per env
            self.step_indices = T.zeros((self.env.num_envs), dtype=T.long, device=self.device)
            # Track completed trajectories
            self.completed_trajectories = []
            
        except Exception as e:
            self.logger.error(f"Error in Reinforce.__init__: {e}", exc_info=True)

    
    def get_action(self,
                   states: np.ndarray|T.Tensor,
                   goals: np.ndarray|T.Tensor|None=None,
                   step: int|None=None,
                   context: str = 'train'
                   )->ActionOutput:
        """
        Select an action based on the current policy.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            step: Optional[int]: The current step.
            context: str: The context of the action (train, test, or learn).
        
        Returns:
            tuple[np.ndarray | T.Tensor, T.Tensor | None, Distribution | None]: actions, raw actions, and distribution.
        """

        actions = None
        raw_actions = None
        dist = None
        
        if context == 'train':
            dist, logits = self.policy(states)
            actions = dist.sample()
            actions = actions.detach()
            return actions, logits, dist
        elif context == 'test':
            with T.no_grad():
                dist, logits = self.policy(states)
                actions = dist.mode
                return actions, logits, dist
        else: # learn
            dist, logits = self.policy(states)
            return actions, logits, dist

    def _step(self,
              step: int,
              states: np.ndarray | T.Tensor,
              max_episodes: int,
              episode_scores: np.ndarray,
              completed_episodes: np.ndarray,
              score_history: deque[float],
              best_reward: float,
              trajectories_per_update: int,
              training: bool = True
              ):
        """
        Performs a single step of training/testing.
        
        Args:
        step: int: The current step.
        states: np.ndarray | T.Tensor: The current states.
        max_episodes: int: The maximum number of episodes.
        episode_scores: np.ndarray: The current episode scores.
        completed_episodes: np.ndarray: The current completed episodes.
        score_history: deque[float]: The current score history.
        best_reward: float: The current best reward.
        trajectories_per_update: int: The number of trajectories to update the model on
        training: bool: Whether the step is for training or testing.
        
        Returns:
        dict: A dictionary containing the step metrics.
        """
        step_log = {}
        episode_logs = []

        obs, _ = self._preprocess_inputs(states)
        actions, _, _ = self.get_action(obs, context='train' if training else 'test')
        actions = self.env.format_actions(actions)
        next_states, rewards, dones, infos = self.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        obs, actions, rewards, next_states, dones = (
            T.tensor(obs, dtype=T.float32, device=self.device) if isinstance(obs, np.ndarray) else obs,
            T.tensor(actions, dtype=T.long, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int8, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )

        # Take snapshot of current step indices per env
        current_indices = self.step_indices.clone()
        # Add transitions to trajectory buffers, overwriting current values
        self.states_buffer[current_indices, T.arange(self.env.num_envs), :] = obs
        self.actions_buffer[current_indices, T.arange(self.env.num_envs)] = actions
        self.rewards_buffer[current_indices, T.arange(self.env.num_envs)] = rewards
        self.next_states_buffer[current_indices, T.arange(self.env.num_envs), :] = next_states
        self.dones_buffer[current_indices, T.arange(self.env.num_envs)] = dones
        # Increment step indices
        self.step_indices += 1

        episode_scores += rewards.flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': rewards.mean().item()
        })
        
        # Check if any env is done
        done_episodes = T.nonzero(dones, as_tuple=False).flatten()

        for i in done_episodes:
            # Append episode trajectory to completed trajectories
            self.completed_trajectories.append(
                {
                    'states': self.states_buffer[:current_indices[i].item(), i, :].clone(),
                    'actions': self.actions_buffer[:current_indices[i].item(), i].clone(),
                    'rewards': self.rewards_buffer[:current_indices[i].item(), i].clone(),
                    'next_states': self.next_states_buffer[:current_indices[i].item(), i, :].clone(),
                    'dones': self.dones_buffer[:current_indices[i].item(), i].clone(),
                }
            )
            # Reset step counter for done env
            self.step_indices[i] = 0

            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)
            # check if best reward
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': round(float(episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }
            if training:
                # Update normalizer if state_normalizer
                if self.state_normalizer:
                    self.state_normalizer.add(next_states)
                # Check if best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save()
                # Add best reward to episode log
                episode_log.update({
                    'best_reward': round(float(best_reward), 2),
                    'best': 1 if avg_reward > best_reward else 0
                })
                # Check if time to learn (len trajectories)
                if len(self.completed_trajectories) >= trajectories_per_update:
                    learn_metrics = self.learn()
                    step_log.update(learn_metrics)
                    # Clear completed trajectories
                    self.completed_trajectories = []
            episode_logs.append(episode_log)

        return{
        'episode_scores': episode_scores,
        'completed_episodes': completed_episodes,
        'score_history': score_history,
        'next_states': next_states,
        'step_log': step_log,
        'episode_logs': episode_logs,
        'done': completed_episodes.sum() >= max_episodes
    }

    def train(self, num_episodes:int, trajectories_per_update:int=10, render_freq:int = 0, seed:int|None=None):
        """Trains the model for 'episodes' number of episodes."""

        init_dict = self._initialize_run(seed=seed, num_episodes=num_episodes, trajectories_per_update=trajectories_per_update)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, trajectories_per_update)
            # Update states, episode scores, completed episodes, and score history
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    # Call the test function to render an episode
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)

        self.env.close()

    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""

        init_dict = self._initialize_run(seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, trajectories_per_update=0, training = False)
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])

            render = True
            for i, episode_log in enumerate(step_result['episode_logs']):
                # Print complete episode metrics to console
                print(f"Testing Environment {i}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[i] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)
            
                if render and render_freq > 0 and completed_episodes.sum() % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        self.env.close()

    def learn(self):
        learn_metrics = {}

        # Instantiate lists to append trajectory data to
        all_states = []
        all_actions = []
        all_returns = []

        # Clear gradients
        self.policy.optimizer.zero_grad()
        self.value.optimizer.zero_grad()

        # Iterate over completed trajectories
        for trajectory in self.completed_trajectories:
            # Append states to list
            all_states.append(trajectory['states'])
            # Append actions to list
            all_actions.append(trajectory['actions'])
            # Compute returns for trajectory
            returns = compute_full_return(trajectory['rewards'], self.discount)
            # Append returns to list
            all_returns.append(T.tensor(returns, dtype=T.float32, device=self.device))
            

        # Use T.cat to concatenate all tensors in list into single tensor of shape [total_steps, obs_dim]
        states = T.cat(all_states, dim=0)
        actions = T.cat(all_actions, dim=0)
        returns = T.cat(all_returns, dim=0)

        # Normalize states if using normalizer
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)

        # Get state values (if using value model) and calculate value loss
        if self.value:
            state_values = self.value(states)
            advantages = returns.detach() - state_values.squeeze(-1)
            value_loss = (advantages ** 2).mean()
        else:
            advantages = returns.detach()
            value_loss = 0
        
        # dist, logits = self.policy_model(states)
        _, _, dist = self.get_action(states, context='learn')
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        # Get entropy coefficient
        entropy_coefficient = self.entropy_coefficient
        if self.entropy_schedule:
            entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get policy loss
        policy_loss = -(log_probs * advantages.detach() + entropy_coefficient * entropies).mean()

        # Calculate gradients
        total_loss = policy_loss + value_loss
        total_loss.backward()

        # Update weights
        self.policy.optimizer.step()
        if self.value:
            self.value.optimizer.step()

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantages': advantages.mean().item(),
            'returns': returns.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
        })

        return learn_metrics

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "policy": self.policy.get_config(),
            "value": self.value.get_config() if self.value is not None else None,
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "max_trajectory_length": self.max_trajectory_length,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir
        }

    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        if self.value:
            self.value.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        policy_model = StochasticDiscretePolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
        value_model = ValueModel.load(config_dir, 'value', load_weights, env=env_wrapper) if config.get('value', None) else None
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "/state_normalizer.pt") if config.get('state_normalizer', None) else None
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

        # return reinforce agent
        agent = cls(
            env=env_wrapper,
            policy=policy_model,
            value=value_model,
            discount=config["discount"],
            state_normalizer=state_normalizer,
            entropy_coefficient=config["entropy_coefficient"],
            entropy_schedule=ScheduleWrapper(config["entropy_schedule"]) if config["entropy_schedule"] else None,
            max_trajectory_length=config["max_trajectory_length"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
        )

        return agent
    

class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        env: EnvWrapper,
        policy: ActorModel,
        critic: ContinuousCritic,
        *,
        replay_buffer: Buffer,
        discount: float=0.99,
        tau: float=0.001,
        action_epsilon: float = 0.2,
        batch_size: int = 64,
        noise: Optional[Noise]=None,
        noise_schedule: Optional[ScheduleWrapper]=None,
        noise_clip: float = 0.5,
        grad_clip: Optional[float]=None,
        warmup: int=1000,
        N: int=1, # N-steps
        curiosity: Optional[ICM] = None,
        state_normalizer: Optional[Normalizer] = None,
        goal_normalizer: Optional[Normalizer] = None,
        obs_key: str = 'observation',
        goal_key: str | None = None,
        achieved_goal_key: str = 'achieved_goal', # For HER
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: Optional[str | T.device] = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, obs_key, goal_key, save_dir, device, log_level)
            self.policy = policy
            self.critic = critic
            # set target actor and critic models
            self.target_policy = self.policy.clone(device=self.policy.device)
            self.target_critic = self.critic.clone(device=self.critic.device)
            self.discount = discount
            self.tau = tau
            self.action_epsilon = action_epsilon
            self.replay_buffer = replay_buffer
            self.batch_size = batch_size
            self.noise = noise
            self.noise_schedule = noise_schedule
            self.noise_clip = noise_clip
            self.grad_clip = grad_clip
            self.warmup = warmup
            self.N = N
            self.curiosity = curiosity
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.obs_key = obs_key
            self.goal_key = goal_key
            self.achieved_goal_key = achieved_goal_key
        except Exception as e:
            self.logger.error(f"Error in DDPG init: {e}", exc_info=True)
        
        # set internal attributes
        try:
            self._use_her = False

            # Set learn_iter and sync_iter to 0. For distributed training
            # self._learn_iter = 0
            # self._sync_iter = 0

        except Exception as e:
            self.logger.error(f"Error in DDPG init internal attributes: {e}", exc_info=True)
    
    def _initialize_wandb(self, run_number:Optional[str]=None, run_name_prefix:Optional[str]=None, learn_iter:Optional[int]=None):
        """Initialize WandbCallback if using WandbCallback"""
        try:
            if self._wandb:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if not callback.initialized:
                            models = [self.policy, self.critic]
                            if self.curiosity is not None:
                                models.append(self.curiosity)
                            config = self.get_config()
                            # if learn_iter:
                            #     self._learn_iter = learn_iter
                            #     config['learn_interval'] = learn_iter
                            callback.initialize_run(models, config, run_number=run_number, run_name_prefix=run_name_prefix)
        except Exception as e:
            self.logger.error(f"Error in _initialize_wandb: {e}", exc_info=True)

    def _init_her(self):
        self._use_her = True

    # def _distributed_learn(self, step: int, run_number:Optional[str]=None, learn_iter:Optional[int]=None, num_updates:int=1,
    #                       state_normalizer:Optional[Normalizer]=None, goal_normalizer:Optional[Normalizer]=None):
    #     """Used in distributed training to update the shared models.
    #     This function is overridden by the Worker class to point to the Learner class.
    #     """
    #     previous_step = self._step
    #     # Set current step to step if greater than current step
    #     if step > previous_step:
    #         self._step = step
    #         # Initialize wandb check
    #         self._initialize_wandb(run_number=run_number, run_name_prefix="train", learn_iter=learn_iter)
    #         for _ in range(num_updates):
    #             actor_loss, critic_loss = self.learn(goal_normalizer)
    #         # Only store log if current step greater than previous and self._wandb
    #         if self._wandb:
    #             self._train_step_config["actor_loss"] = actor_loss
    #             self._train_step_config["critic_loss"] = critic_loss
    #             for callback in self.callbacks:
    #                 if isinstance(callback, WandbCallback):
    #                     callback.on_train_step_end(step, self._train_step_config)
    #     else:
    #         for _ in range(num_updates):
    #             actor_loss, critic_loss = self.learn(goal_normalizer)

    # def get_parameters(self):
    #     """Get the parameters of all models, ensuring they are on CPU for Ray serialization."""
    #     return {
    #         'actor_model': {k: v.cpu() for k, v in self.actor_model.state_dict().items()},
    #         'critic_model': {k: v.cpu() for k, v in self.critic_model.state_dict().items()},
    #         'target_actor_model': {k: v.cpu() for k, v in self.target_actor_model.state_dict().items()},
    #         'target_critic_model': {k: v.cpu() for k, v in self.target_critic_model.state_dict().items()},
    #     }

    # def apply_parameters(self, params:Dict[str, Dict[str, T.Tensor]]):
    #     """Apply params to a model. Used in distributed training."""
    #     self.actor_model.load_state_dict(params['actor_model'])
    #     self.critic_model.load_state_dict(params['critic_model'])
    #     self.target_actor_model.load_state_dict(params['target_actor_model'])
    #     self.target_critic_model.load_state_dict(params['target_critic_model'])

    def get_action(self,
                   states: np.ndarray|T.Tensor,
                   goals: np.ndarray|T.Tensor|None=None,
                   step: int|None=None,
                   context: str = 'train'
                   )->ActionOutput:
        """
        Select an action based on the current policy.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            step: Optional[int]: The current step.
            context: str: The context of the action (train, test, or learn).
        
        Returns:
            tuple[np.ndarray | T.Tensor, T.Tensor | None, Distribution | None]: actions, raw actions, and distribution.
        """

        raw_actions = None
        dist = None

        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= self.warmup):
                return self.env.action_space.sample(), raw_actions, dist
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                return self.env.action_space.sample(), raw_actions, dist
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    raw_actions, squashed_actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                action_space_high = T.tensor(self.env.action_space.high, dtype=T.float32, device=self.policy.device)
                action_space_low = T.tensor(self.env.action_space.low, dtype=T.float32, device=self.policy.device)
                actions = (squashed_actions + noise).clip(action_space_low, action_space_high)

                return actions.detach(), raw_actions, dist

        # check if get action is for testing
        if context == 'test':
            with T.no_grad():
                raw_actions, squashed_actions = self.target_policy(states, goals)
            return squashed_actions.detach(), raw_actions, dist

        else: # learn
            raw_actions, squashed_actions = self.policy(states, goals)
            return squashed_actions, raw_actions, dist

    def learn(self, step: int):
        
        # self._learn_iter += 1
        # self.logger.debug(f"DDPG learn iteration: {self._learn_iter}")

        learn_metrics = {}
            
        if self.replay_buffer.get_config()['type'] == 'PrioritizedReplayBuffer':
            if self._use_her:  # HER with prioritized replay
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
            else:  # Just prioritized replay
                states, actions, rewards, next_states, dones, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
                
            # Log PER-specific metrics
            if self._wandb:
                # Get the actual size of used buffer (not the full capacity)
                actual_size = min(self.replay_buffer.counter, self.replay_buffer.buffer_size)
                # Get indices for all actual entries in the buffer
                valid_indices = T.arange(actual_size, device=self.replay_buffer.device)
                # Get priority info for logging
                if hasattr(self.replay_buffer, 'sum_tree') and self.replay_buffer.sum_tree is not None:
                    indices_tensor = T.tensor(indices, device=self.replay_buffer.device)
                    # Get tree indices for sampled transitions
                    tree_indices = indices_tensor + self.replay_buffer.sum_tree.capacity - 1
                    # Get priorities for sampled transitions
                    sampled_priorities = self.replay_buffer.sum_tree.tree[tree_indices].cpu().numpy()
                    valid_tree_indices = valid_indices + self.replay_buffer.sum_tree.capacity - 1
                    buffer_priorities = self.replay_buffer.sum_tree.tree[valid_tree_indices].cpu().numpy()

                else:
                    buffer_priorities = self.replay_buffer.priorities[valid_indices].cpu().numpy()
                    sampled_priorities = self.replay_buffer.priorities[indices].cpu().numpy()
                    
                # Add priority buffer metrics to learn_metrics dict
                learn_metrics.update({
                    'PER/beta': self.replay_buffer.beta,
                    'PER/sampled_priorities': sampled_priorities,
                    'PER/buffer_priorities': buffer_priorities,
                    'PER/weights': weights,
                    'PER/probs': probs,
                    'PER/mean_sampled_priority': np.mean(sampled_priorities),
                    'PER/mean_buffer_priority': np.mean(buffer_priorities),
                    'PER/max_sampled_priority': np.max(sampled_priorities),
                    'PER/max_buffer_priority': np.max(buffer_priorities),
                    'PER/weight_mean': np.mean(weights.cpu().numpy()) if weights is not None else 0.0,
                    'PER/weight_std': np.std(weights.cpu().numpy()) if weights is not None else 0.0
                })
        else:  # Standard replay buffer
            if self._use_her:
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            else:
                states, actions, rewards, next_states, dones, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            
            weights = None
            indices = None

        # Normalize states if self.normalize_inputs
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            desired_goals = self.goal_normalizer.normalize(desired_goals)
        else:
            desired_goals = None

        # Ensure training data is on correct device
        actions = actions.to(self.target_policy.device)
        rewards = rewards.to(self.target_critic.device)
        dones = dones.to(self.target_critic.device)
        trajectory_lengths = trajectory_lengths.to(self.target_critic.device)

        # Train ICM if curiosity and update _use_extrinsic flag
        if self.curiosity:
            # Reshape arrays to (batch_size * N, -1) to train on all steps in N
            dones_reshaped = dones.view(self.batch_size * self.N)
            mask = (dones_reshaped == 0)
            states_reshaped = states.view(self.batch_size * self.N, -1)
            next_states_reshaped = next_states.view(self.batch_size * self.N, -1)
            actions_reshaped = actions.view(self.batch_size * self.N, -1)
            # Replace next state with state value where done = True b/c next state value could be reset observation
            # returned by environment (IsaacSim)
            next_states_reshaped = T.where(mask.unsqueeze(1), next_states_reshaped, states_reshaped)
            curiosity_loss = self.curiosity.train(states_reshaped, next_states_reshaped, actions_reshaped)
            if step > self.curiosity.extrinsic_threshold:
                self.curiosity._use_extrinsic = True
            else:
                self.curiosity._use_extrinsic = False

        # Get target values
        with T.no_grad():
            # Compute intrinsic reward if using ICM
            if self.curiosity:
                intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                    states_reshaped,
                    next_states_reshaped,
                    actions_reshaped
                )
                intrinsic_reward = intrinsic_reward.view(self.batch_size, self.N)
                if self.curiosity._use_extrinsic:
                    rewards += intrinsic_reward
                else:
                    rewards = intrinsic_reward

            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic.device
            ).squeeze()

            target_actions, _, _ = self.get_action(
                next_states[:,-1,:],
                desired_goals[:,-1,:] if desired_goals is not None else None,
                context='test'
            )

            target_critic_values = self.target_critic(
                next_states[:,-1,:],
                target_actions,
                desired_goals[:,-1,:] if desired_goals is not None else None
            ).squeeze()

            no_dones_mask = (dones.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values

            # Apply HER-specific clamping if needed
            if self._use_her:
                if self.curiosity and not self.curiosity._use_extrinsic:
                    pass
                else:
                    targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions = self.critic(
            states[:,0,:],
            actions[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None
        ).squeeze()

        # Calculate TD errors
        error = targets - predictions
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss = (weights.to(self.critic.device) * error.pow(2)).mean()
        else:
            critic_loss = error.pow(2).mean()
            # critic_loss = F.mse_loss(predictions, targets)

        # Update critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic.optimizer.step()

        # Get actor's action predictions
        action_values, raw_actions, _ = self.get_action(
            states[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None,
            context='learn'
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic(states[:,0,:], action_values, desired_goals[:,0,:] if desired_goals is not None else None)
        if weights is not None:
            actor_loss = -(weights.to(self.policy.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()

        # Add HER-specific regularization if needed
        if self._use_her:
            actor_loss += raw_actions.pow(2).mean()

        # Update actor
        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy.optimizer.step()

        # Perform soft update on target networks
        if not self._use_her:
            self.soft_update(self.policy, self.target_policy)
            self.soft_update(self.critic, self.target_critic)

        # Update priorities if using prioritized replay - only on update_freq steps
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:# and hasattr(self.replay_buffer, 'beta_update_freq'):
            self.replay_buffer.update_priorities(indices, error.detach().flatten().to(self.replay_buffer.device))

        learn_metrics.update({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": error.mean().item(),
            "actor_predictions": action_values.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_actor_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
        })
        if self.curiosity:
            learn_metrics.update({
                "curiosity_loss": curiosity_loss,
                "intrinsic_reward": intrinsic_reward.mean().item(),
                "use_extrinsic": self.curiosity._use_extrinsic,
                "reward_weight": self.curiosity.reward_weight * self.curiosity.reward_scheduler.get_factor() \
                    if self.curiosity.reward_scheduler else self.curiosity.reward_weight
            })
        # Step noise schedule if present
        if self.noise_schedule:
            learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})
            self.noise_schedule.step()

        return learn_metrics
        
    
    def soft_update(self, current, target):
        with T.no_grad():
            for current_params, target_params in zip(current.parameters(), target.parameters()):
                target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)

            # Copy buffers (running_mean, running_var)
            main_buffers = dict(current.named_buffers())
            target_buffers = dict(target.named_buffers())
            for name in main_buffers:
                if name in target_buffers:
                    target_buffers[name].copy_(main_buffers[name])

    def _step(self,
              step: int,
              states: np.ndarray | T.Tensor | dict,
              max_episodes: int,
              episode_scores: np.ndarray,
              completed_episodes: np.ndarray,
              score_history: deque[float],
              best_reward: float,
              learn: bool = True,
              training: bool = True
              ):

        # reset noise if training
        if training:
            if type(self.noise) == OUNoise:
                self.noise.reset()
        obs, goals = self._preprocess_inputs(states)
        actions, _, _ = self.get_action(obs, goals, step, context='train' if training else 'test')
        actions = self.env.format_actions(actions)
        next_states, rewards, dones, infos = self.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        obs, actions, rewards, next_states, dones = (
            T.tensor(obs, dtype=T.float32, device=self.device) if isinstance(obs, np.ndarray) else obs,
            T.tensor(actions, dtype=T.float32, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int8, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )

        episode_scores += rewards.flatten()

        buffer_trajectories = {
            'states': infos['n-step trajectory']['states'],
            'actions': infos['n-step trajectory']['actions'],
            'rewards': infos['n-step trajectory']['rewards'],
            'next_states': infos['n-step trajectory']['next_states'],
            'dones': infos['n-step trajectory']['dones'],
            'trajectory_lengths': infos['n-step trajectory']['trajectory_lengths']
        }
        if self.goal_key:
            buffer_trajectories['desired_goals'] = infos['n-step trajectory']['desired_goals']
            buffer_trajectories['state_achieved_goals'] = infos['n-step trajectory']['state_achieved_goals']
            buffer_trajectories['next_state_achieved_goals'] = infos['n-step trajectory']['next_state_achieved_goals']

        if training:
            self.replay_buffer.add(**buffer_trajectories)
            # Update normalizer if state_normalizer
            next_obs, next_goals = self.extract_states_goals(next_states)
            if self.state_normalizer:
                self.state_normalizer.add(next_obs)
            if self.goal_normalizer:
                self.goal_normalizer.add(next_goals)

        done_episodes = T.nonzero(dones, as_tuple=False).flatten()
        episode_logs = []

        for i in done_episodes:
            # Increment completed episodes for env by 1
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)

            # check if best reward
            if training and avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': round(float(episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }

            if training:
                episode_log.update({
                    'best_reward': round(float(best_reward), 2),
                    'best': 1 if avg_reward > best_reward else 0
                })
            episode_logs.append(episode_log)

        step_log = {} # Collect step metrics

        # Check if past warmup
        if learn and step > self.warmup and self.replay_buffer.counter > self.batch_size:
            # Check if distributed
            # if self._distributed:
            #     self._distributed_learn(step, self._run_number)
            # else:
            learn_metrics = self.learn(step)
            step_log.update({**learn_metrics})

        # self._train_step_config["step_reward"] = rewards.mean()
        step_log.update({
            'step_reward': rewards.mean(),
        })

        # Step schedulers
        if self.noise_schedule is not None:
            self.noise_schedule.step()

        return{
            'episode_scores': episode_scores,
            'completed_episodes': completed_episodes,
            'score_history': score_history,
            'next_states': next_states,
            'step_log': step_log,
            'episode_logs': episode_logs,
            'done': completed_episodes.sum() >= max_episodes
        }

    def train(self, num_episodes: int, steps_per_learn: int = 1, render_freq: int = 0, seed: int | None = None):
        """Trains the model for 'episodes' number of episodes."""

        init_dict = self._initialize_run(seed=seed, num_episodes=num_episodes)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step count
            step += 1
            # If distributed, sync to shared agent
            # if self._distributed and _step % self._sync_iter == 0:
            #     params = self.get_parameters()
            #     self.apply_parameters(params)

            # Determine if step should perform update
            learn = step % steps_per_learn == 0
            # Perform train step
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn)
            # Update states, episode scores, completed episodes, and score history
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            # log to callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    # Call the test function to render an episode
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)
        
        self.env.close()
       
    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""

        init_dict = self._initialize_run(seed=seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn = False, training = False)
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])

            render = True
            for i, episode_log in enumerate(step_result['episode_logs']):
                # Print complete episode metrics to console
                print(f"Testing Environment {i}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)
            
                if render and render_freq > 0 and completed_episodes.sum() % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "actor_model": self.policy.get_config(),
            "critic_model": self.critic.get_config(),
            "replay_buffer": self.replay_buffer.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "batch_size": self.batch_size,
            "noise": self.noise.get_config(),
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            'grad_clip': self.grad_clip,
            'warmup': self.warmup,
            'N': self.N,
            "curiosity": self.curiosity.get_config() if self.curiosity is not None else None,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "obs_key": self.obs_key,
            "goal_key": self.goal_key,
            "achieved_goal_key": self.achieved_goal_key,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir,
            "device": self.device.type,
            "log_level": logging.getLevelName(self.logger.getEffectiveLevel()).lower()
        }


    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.critic.save(self.save_dir)
        if self.curiosity:
            self.curiosity.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        policy = ActorModel.load(config_dir, 'policy', load_weights, env=env_wrapper)
        critic = ContinuousCritic.load(config_dir, 'critic', load_weights, env=env_wrapper)
        if config['replay_buffer'] is not None:
            config['replay_buffer']['config']['env'] = env_wrapper
            if config['replay_buffer']['type'] == 'PrioritizedReplayBuffer':
                replay_buffer = PrioritizedReplayBuffer(**config["replay_buffer"]["config"])
            else:
                replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
        else:
            replay_buffer = None
        noise = Noise.create_instance(config["noise"]["type"], **config["noise"]["config"])
        curiosity = ICM.load(config["save_dir"], env=env_wrapper) if config["curiosity"] else None
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = Normalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

        agent = cls(
            env = env_wrapper,
            policy = policy,
            critic = critic,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            replay_buffer=replay_buffer,
            batch_size=config["batch_size"],
            noise=noise,
            noise_schedule=ScheduleWrapper(config["noise_schedule"]) if config["noise_schedule"] else None,
            noise_clip=config["noise_clip"],
            grad_clip=config['grad_clip'],
            warmup = config['warmup'],
            N = config['N'],
            curiosity=curiosity,
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            obs_key=config["obs_key"],
            goal_key=config["goal_key"],
            achieved_goal_key=config["achieved_goal_key"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
            log_level=config["log_level"]
        )

        return agent
    

class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""
    
    def __init__(
        self,
        env: EnvWrapper,
        policy: ActorModel,
        critic_a: ContinuousCritic,
        critic_b: Optional[ContinuousCritic] = None,
        *,
        replay_buffer: Optional[Buffer] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        action_epsilon: float = 0.0,
        batch_size: int = 256,
        noise: Optional[Noise] = None,
        noise_schedule: Optional[ScheduleWrapper] = None,
        target_noise: Optional[Noise] = None,
        target_noise_schedule: Optional[ScheduleWrapper] = None,
        noise_clip: float = 0.5,
        policy_update_delay: int = 2,
        grad_clip: float = 40.0,
        warmup: int = 1000,
        N: int=1, # N-steps
        curiosity: Optional[ICM] = None,
        state_normalizer: Optional[Normalizer] = None,
        goal_normalizer: Optional[Normalizer] = None,
        obs_key: str = 'observation',
        goal_key: str = 'desired_goal',
        achieved_goal_key: str = 'achieved_goal', # For HER
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: Optional[str | T.device] = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, obs_key, goal_key, save_dir, device, log_level)
            self.policy = policy
            self.critic_a = critic_a
            self.critic_b = critic_b
            # clone second critic (do not copy weights) if critic_b None
            if not critic_b:
                self.critic_b = self.critic_a.clone(copy_weights=False, device=self.critic_a.device)
            self.target_policy = self.policy.clone(device=self.policy.device)
            self.target_critic_a = self.critic_a.clone(device=self.critic_a.device)
            self.target_critic_b = self.critic_b.clone(device=self.critic_b.device)
            self.discount = discount
            self.tau = tau
            self.action_epsilon = action_epsilon
            self.replay_buffer = replay_buffer
            self.batch_size = batch_size
            self.noise = noise
            self.noise_schedule = noise_schedule
            if target_noise is None:
                target_noise = NormalNoise(self.env.single_action_space.shape, stddev=0.2, device=device)
            self.target_noise = target_noise
            self.target_noise_schedule = target_noise_schedule
            self.noise_clip = noise_clip
            self.policy_update_delay = policy_update_delay
            self.grad_clip = grad_clip
            self.warmup = warmup
            self.N = N
            self.curiosity = curiosity
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.obs_key = obs_key
            self.goal_key = goal_key
            self.achieved_goal_key = achieved_goal_key

        except Exception as e:
            self.logger.error(f"Error in TD3 init: {e}", exc_info=True)
        
        # set internal attributes
        try:
            self._use_her = False
            self._learn_iter = 0

            # Set sync_iter to 0. For distributed training
            # self._sync_iter = 0

        except Exception as e:
            self.logger.error(f"Error in TD3 init internal attributes: {e}", exc_info=True)
    
    def _initialize_wandb(self, run_number:str=None, run_name_prefix:str=None, learn_iter:int=None):
        """Initialize WandbCallback if using WandbCallback"""
        try:
            if self._wandb:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if not callback.initialized:
                            models = (self.policy, self.critic_a, self.critic_b)
                            config = self.get_config()
                            # if learn_iter:
                            #     self._learn_iter = learn_iter
                            #     config['learn_interval'] = learn_iter
                            callback.initialize_run(models, config, run_number=run_number, run_name_prefix=run_name_prefix)
        except Exception as e:
            self.logger.error(f"Error in _initialize_wandb: {e}", exc_info=True)

    def _init_her(self):
        self._use_her = True

    # def _distributed_learn(self, step: int, run_number:str=None, learn_iter:int=None):
    #     """Used in distributed training to update the shared models.
    #     This function is overridden by the Worker class to point to the Learner class.
    #     """
    #     previous_step = self._step
    #     # Set current step to step if greater than current step
    #     if step > previous_step:
    #         self._step = step
    #         # Initialize wandb check
    #         self._initialize_wandb(run_number=run_number, run_name_prefix="train", learn_iter=learn_iter)
    #         actor_loss, critic_loss = self.learn()
    #         # Only store log if current step greater than previous and self._wandb
    #         if self._wandb:
    #             self._train_step_config["actor_loss"] = actor_loss
    #             self._train_step_config["critic_loss"] = critic_loss
    #             for callback in self.callbacks:
    #                 if isinstance(callback, WandbCallback):
    #                     callback.on_train_step_end(step, self._train_step_config)
    #     else:
    #         actor_loss, critic_loss = self.learn()

    # def get_parameters(self):
    #     """Get the parameters of all models, ensuring they are on CPU for Ray serialization."""
    #     return {
    #         'actor_model': {k: v.cpu() for k, v in self.actor_model.state_dict().items()},
    #         'critic_model_a': {k: v.cpu() for k, v in self.critic_model_a.state_dict().items()},
    #         'critic_model_b': {k: v.cpu() for k, v in self.critic_model_b.state_dict().items()},
    #         'target_actor_model': {k: v.cpu() for k, v in self.target_actor_model.state_dict().items()},
    #         'target_critic_model_a': {k: v.cpu() for k, v in self.target_critic_model_a.state_dict().items()},
    #         'target_critic_model_b': {k: v.cpu() for k, v in self.target_critic_model_b.state_dict().items()},
    #     }

    # def apply_parameters(self, params:Dict[str, Dict[str, T.Tensor]]):
    #     """Apply params to a model. Used in distributed training."""
    #     self.actor_model.load_state_dict(params['actor_model'])
    #     self.critic_model_a.load_state_dict(params['critic_model_a'])
    #     self.critic_model_b.load_state_dict(params['critic_model_b'])
    #     self.target_actor_model.load_state_dict(params['target_actor_model'])
    #     self.target_critic_model_a.load_state_dict(params['target_critic_model_a'])
    #     self.target_critic_model_b.load_state_dict(params['target_critic_model_b'])
    

    def get_action(self,
                   states: np.ndarray|T.Tensor,
                   goals: np.ndarray|T.Tensor|None=None,
                   step: int|None=None,
                   context: str = 'train'
                   )->ActionOutput:
        """
        Select an action based on the current policy.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            step: Optional[int]: The current step.
            context: str: The context of the action (train, test, or learn).
        
        Returns:
            tuple[np.ndarray | T.Tensor, T.Tensor | None, Distribution | None]: actions, raw actions, and distribution.
        """

        raw_actions = None
        dist = None

        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= self.warmup):
                return self.env.action_space.sample(), raw_actions, dist
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                return self.env.action_space.sample(), raw_actions, dist
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    raw_actions, squashed_actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                action_space_high = T.tensor(self.env.action_space.high, dtype=T.float32, device=self.policy.device)
                action_space_low = T.tensor(self.env.action_space.low, dtype=T.float32, device=self.policy.device)
                actions = (squashed_actions + noise).clip(action_space_low, action_space_high)

                return actions.detach(), raw_actions, dist

        # check if get action is for testing
        if context == 'test':
            with T.no_grad():
                raw_actions, squashed_actions = self.target_policy(states, goals)
            return squashed_actions.detach(), raw_actions, dist

        else: # learn
            raw_actions, squashed_actions = self.policy(states, goals)
            return squashed_actions, raw_actions, dist
            
    def learn(self, step: int):
        self._learn_iter += 1

        learn_metrics = {}
            
        if self.replay_buffer.get_config()['type'] == 'PrioritizedReplayBuffer':
            if self._use_her:  # HER with prioritized replay
                #DEBUG
                # print(f"HER with prioritized replay")
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
            else:  # Just prioritized replay
                #DEBUG
                # print(f"Just prioritized replay")
                states, actions, rewards, next_states, dones, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
                
            # Log PER-specific metrics
            if self._wandb:
                # Get the actual size of used buffer (not the full capacity)
                actual_size = min(self.replay_buffer.counter, self.replay_buffer.buffer_size)
                # Get indices for all actual entries in the buffer
                valid_indices = T.arange(actual_size, device=self.replay_buffer.device)
                # Get priority info for logging
                if hasattr(self.replay_buffer, 'sum_tree') and self.replay_buffer.sum_tree is not None:
                    indices_tensor = T.tensor(indices, device=self.replay_buffer.device)
                    # Get tree indices for sampled transitions
                    tree_indices = indices_tensor + self.replay_buffer.sum_tree.capacity - 1
                    # Get priorities for sampled transitions
                    sampled_priorities = self.replay_buffer.sum_tree.tree[tree_indices].cpu().numpy()
                    valid_tree_indices = valid_indices + self.replay_buffer.sum_tree.capacity - 1
                    buffer_priorities = self.replay_buffer.sum_tree.tree[valid_tree_indices].cpu().numpy()

                else:
                    buffer_priorities = self.replay_buffer.priorities[valid_indices].cpu().numpy()
                    sampled_priorities = self.replay_buffer.priorities[indices].cpu().numpy()
                    
                # Add priority buffer metrics to learn_metrics dict
                learn_metrics.update({
                    'PER/beta': self.replay_buffer.beta,
                    'PER/sampled_priorities': sampled_priorities,
                    'PER/buffer_priorities': buffer_priorities,
                    'PER/weights': weights,
                    'PER/probs': probs,
                    'PER/mean_sampled_priority': np.mean(sampled_priorities),
                    'PER/mean_buffer_priority': np.mean(buffer_priorities),
                    'PER/max_sampled_priority': np.max(sampled_priorities),
                    'PER/max_buffer_priority': np.max(buffer_priorities),
                    'PER/weight_mean': np.mean(weights.cpu().numpy()) if weights is not None else 0.0,
                    'PER/weight_std': np.std(weights.cpu().numpy()) if weights is not None else 0.0
                })

        else:  # Standard replay buffer
            if self._use_her:
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            else:
                states, actions, rewards, next_states, dones, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            
            weights = None
            indices = None


        # Normalize states if self.normalize_inputs
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            desired_goals = self.goal_normalizer.normalize(desired_goals)
        else:
            desired_goals = None

        # Ensure training data is on correct device
        actions = actions.to(self.target_policy.device)
        rewards = rewards.to(self.target_critic_a.device)
        dones = dones.to(self.target_critic_a.device)
        trajectory_lengths = trajectory_lengths.to(self.target_critic_a.device)

        # Train ICM if curiosity and update _use_extrinsic flag
        if self.curiosity:
            # Reshape arrays to (batch_size * N, -1) to train on all steps in N
            dones_reshaped = dones.view(self.batch_size * self.N)
            mask = (dones_reshaped == 0)
            states_reshaped = states.view(self.batch_size * self.N, -1)
            next_states_reshaped = next_states.view(self.batch_size * self.N, -1)
            actions_reshaped = actions.view(self.batch_size * self.N, -1)
            # Replace next state with state value where done = True b/c next state value could be reset observation
            # returned by environment (IsaacSim)
            next_states_reshaped = T.where(mask.unsqueeze(1), next_states_reshaped, states_reshaped)
            curiosity_loss = self.curiosity.train(states_reshaped, next_states_reshaped, actions_reshaped)
            if step > self.curiosity.extrinsic_threshold:
                self.curiosity._use_extrinsic = True
            else:
                self.curiosity._use_extrinsic = False

        # Get target values
        with T.no_grad():
            # Compute intrinsic reward if using ICM
            if self.curiosity:
                intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                    states_reshaped,
                    next_states_reshaped,
                    actions_reshaped
                )
                intrinsic_reward = intrinsic_reward.view(self.batch_size, self.N)
                if self.curiosity._use_extrinsic:
                    rewards += intrinsic_reward
                else:
                    rewards = intrinsic_reward

            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic_a.device
            ).squeeze()

            target_actions, _, _ = self.get_action(
                next_states[:,-1,:],
                desired_goals[:,-1,:] if desired_goals is not None else None,
                context='test'
            )

            noise = self.target_noise(target_actions.shape)
            # Apply noise clipping if needed
            if self.noise_clip > 0:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            # Apply noise schedule if needed
            if self.target_noise_schedule is not None:
                noise *= self.target_noise_schedule.get_factor()
                learn_metrics.update({'target_noise_anneal': self.target_noise_schedule.get_factor()})   

            if self.noise_schedule:
                learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})
                
            # Add noise to target actions and clamp to action space
            target_actions = (target_actions + noise).clamp(float(self.env.action_space.low.min()), float(self.env.action_space.high.max()))
            
            target_critic_values_a = self.target_critic_a(
                next_states[:,-1,:],
                target_actions,
                desired_goals[:,-1,:] if desired_goals is not None else None).squeeze()
            target_critic_values_b = self.target_critic_b(
                next_states[:,-1,:],
                target_actions,
                desired_goals[:,-1,:] if desired_goals is not None else None).squeeze()
            target_critic_values = T.minimum(target_critic_values_a, target_critic_values_b)
            no_dones_mask = (dones.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values
            
            # Apply HER-specific clamping if needed
            if self._use_her:
                if self.curiosity and not self.curiosity._use_extrinsic:
                    pass
                else:
                    targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions_a = self.critic_a(
            states[:,0,:],
            actions[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None
        ).squeeze()

        predictions_b = self.critic_b(
            states[:,0,:],
            actions[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None
        ).squeeze()

        # Calculate TD errors (use average of both critic networks for PER)
        error_a = targets - predictions_a
        error_b = targets - predictions_b
        # error = (error_a.abs() + error_b.abs()) / 2  # Average of absolute errors for priorities
        error = T.minimum(error_a, error_b)

        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss_a = (weights.to(self.critic_a.device) * error_a.pow(2)).mean()
            critic_loss_b = (weights.to(self.critic_b.device) * error_b.pow(2)).mean()
            critic_loss = critic_loss_a + critic_loss_b
        else:
            # critic_loss_a = F.mse_loss(predictions_a, targets)
            critic_loss_a = error_a.pow(2).mean()
            # critic_loss_b = F.mse_loss(predictions_b, targets)
            critic_loss_b = error_b.pow(2).mean()
            critic_loss = critic_loss_a + critic_loss_b

        # Update critics
        self.critic_a.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        critic_loss_a.backward()
        critic_loss_b.backward()

        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.critic_a.parameters(), self.grad_clip)
            T.nn.utils.clip_grad_norm_(self.critic_b.parameters(), self.grad_clip)
        self.critic_a.optimizer.step()
        self.critic_b.optimizer.step()
        
         # Get actor's action predictions
        action_values, raw_actions, _ = self.get_action(
            states[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None,
            context='learn'
        )
        
        # Calculate actor loss based on critic A
        critic_values = self.critic_a(
            states[:,0,:],
            action_values,
            desired_goals[:,0,:] if desired_goals is not None else None
        ).squeeze()

        if weights is not None:
            actor_loss = -(weights.to(self.policy.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()
        
        # Add HER-specific regularization if using HER
        if self._use_her:
            actor_loss += raw_actions.pow(2).mean()

        
        # Update actor
        # Only update actor every actor_update_delay steps
        if self._learn_iter % self.policy_update_delay == 0:
            self.policy.optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.policy.optimizer.step()

            if not self._use_her:
                # Perform soft update on target networks
                self.soft_update(self.policy, self.target_policy)
                self.soft_update(self.critic_a, self.target_critic_a)
                self.soft_update(self.critic_b, self.target_critic_b)

        # Update priorities if using prioritized replay - only on update_freq steps
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:
            self.replay_buffer.update_priorities(indices, error.detach().flatten().to(self.replay_buffer.device))

        # Add metrics to step_logs
        learn_metrics.update({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": error.mean().item(),
            "actor_predictions": action_values.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_actor_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
        })
        if self.curiosity:
            learn_metrics.update({
                "curiosity_loss": curiosity_loss.item(),
                "intrinsic_reward": intrinsic_reward.mean().item(),
                "use_extrinsic": self.curiosity._use_extrinsic,
                "reward_weight": self.curiosity.reward_weight * self.curiosity.reward_scheduler.get_factor() \
                    if self.curiosity.reward_scheduler else self.curiosity.reward_weight
            })
        
        return learn_metrics
        
    
    def soft_update(self, current, target):
        with T.no_grad():
            for current_params, target_params in zip(current.parameters(), target.parameters()):
                target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)

        # Copy buffers (running_mean, running_var)
        main_buffers = dict(current.named_buffers())
        target_buffers = dict(target.named_buffers())
        for name in main_buffers:
            if name in target_buffers:
                target_buffers[name].copy_(main_buffers[name])
        
    def _step(self,
              step: int,
              states: np.ndarray | T.Tensor | dict,
              max_episodes: int,
              episode_scores: np.ndarray,
              completed_episodes: np.ndarray,
              score_history: deque[float],
              best_reward: float,
              learn: bool = True,
              training: bool = True
              ):

        # reset noise if training
        if training:
            if type(self.noise) == OUNoise:
                self.noise.reset()
        obs, goals = self._preprocess_inputs(states)
        actions, _, _ = self.get_action(obs, goals, step, context='train' if training else 'test')
        actions = self.env.format_actions(actions)
        next_states, rewards, dones, infos = self.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        obs, actions, rewards, next_states, dones = (
            T.tensor(obs, dtype=T.float32, device=self.device) if isinstance(obs, np.ndarray) else obs,
            T.tensor(actions, dtype=T.float32, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int8, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )

        episode_scores += rewards.flatten()

        buffer_trajectories = {
            'states': infos['n-step trajectory']['states'],
            'actions': infos['n-step trajectory']['actions'],
            'rewards': infos['n-step trajectory']['rewards'],
            'next_states': infos['n-step trajectory']['next_states'],
            'dones': infos['n-step trajectory']['dones'],
            'trajectory_lengths': infos['n-step trajectory']['trajectory_lengths']
        }
        if self.goal_key:
            buffer_trajectories['desired_goals'] = infos['n-step trajectory']['desired_goals']
            buffer_trajectories['state_achieved_goals'] = infos['n-step trajectory']['state_achieved_goals']
            buffer_trajectories['next_state_achieved_goals'] = infos['n-step trajectory']['next_state_achieved_goals']

        if training:
            self.replay_buffer.add(**buffer_trajectories)
            # Update normalizer if state_normalizer
            next_obs, next_goals = self.extract_states_goals(next_states)
            if self.state_normalizer:
                self.state_normalizer.add(next_obs)
            if self.goal_normalizer:
                self.goal_normalizer.add(next_goals)

        done_episodes = T.nonzero(dones, as_tuple=False).flatten()
        episode_logs = []

        for i in done_episodes:
            # Increment completed episodes for env by 1
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)
            
            # check if best reward
            if training and avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': round(float(episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }
            
            if training:
                episode_log.update({
                    'best_reward': round(float(best_reward), 2),
                    'best': 1 if avg_reward > best_reward else 0
                })
            episode_logs.append(episode_log)

        step_log = {} # Collect step metrics

        # Check if past warmup
        if learn and step > self.warmup and self.replay_buffer.counter > self.batch_size:
            # Check if distributed
            # if self._distributed:
            #     self._distributed_learn(step, self._run_number)
            # else:
            learn_metrics = self.learn(step)
            # self._train_step_config["actor_loss"] = actor_loss
            # self._train_step_config["critic_loss"] = critic_loss
            step_log.update({**learn_metrics})

        step_log.update({
            'step_reward': rewards.mean()
        })

        # Step schedulers
        if self.noise_schedule is not None:
            self.noise_schedule.step()
        if self.target_noise_schedule is not None:
            self.target_noise_schedule.step()

        return{
            'episode_scores': episode_scores,
            'completed_episodes': completed_episodes,
            'score_history': score_history,
            'next_states': next_states,
            'step_log': step_log,
            'episode_logs': episode_logs,
            'done': completed_episodes.sum() >= max_episodes
        }

    def train(self, num_episodes: int, steps_per_learn: int = 1, render_freq: int = 0, seed: int | None = None):
        """Trains the TD3 agent for a given number of episodes."""

        init_dict = self._initialize_run(seed=seed, num_episodes=num_episodes)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step count
            step += 1
            # If distributed, sync to shared agent
            # if self._distributed and _step % self._sync_iter == 0:
            #     params = self.get_parameters()
            #     self.apply_parameters(params)

            # Determine if step should perform update
            learn = step % steps_per_learn == 0
            # Perform train step
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn)
            # Update states, episode scores, completed episodes, and score history
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            # log to callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    # Call the test function to render an episode
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)

        self.env.close()

    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""

        init_dict = self._initialize_run(seed=seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(env, step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn = False, training = False)
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])
            
            render = True
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Testing Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)
            
                if render and render_freq > 0 and completed_episodes.sum() % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        self.env.close()

    def get_config(self):
        return {
            "agent_type": "TD3",
            "env": self.env.to_json(),
            "policy": self.policy.get_config(),
            "critic_a": self.critic_a.get_config(),
            "critic_b": self.critic_b.get_config(),
            "replay_buffer": self.replay_buffer.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "batch_size": self.batch_size,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "target_noise": self.target_noise.get_config() if self.target_noise is not None else None,
            "target_noise_schedule": self.target_noise_schedule.get_config() if self.target_noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "policy_update_delay": self.policy_update_delay,
            "grad_clip": self.grad_clip,
            "warmup": self.warmup,
            "N": self.N,
            "curiosity": self.curiosity.get_config() if self.curiosity is not None else None,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "obs_key": self.obs_key,
            "goal_key": self.goal_key,
            "achieved_goal_key": self.achieved_goal_key,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir,
            "device": self.device.type,
            "log_level": logging.getLevelName(self.logger.getEffectiveLevel()).lower()
        }

    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.critic_a.save(self.save_dir, 'critic_a')
        self.critic_b.save(self.save_dir, 'critic_b')
        if self.curiosity:
            self.curiosity.save(self.save_dir)
        # save state normalizer
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool=True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        policy = ActorModel.load(config_dir, 'policy', load_weights, env=env_wrapper)
        critic_a = ContinuousCritic.load(config_dir, 'critic_a', load_weights, env=env_wrapper)
        critic_b = ContinuousCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        if config['replay_buffer'] is not None:
            config['replay_buffer']['config']['env'] = env_wrapper
            if config['replay_buffer']['type'] == 'PrioritizedReplayBuffer':
                replay_buffer = PrioritizedReplayBuffer(**config["replay_buffer"]["config"])
            else:
                replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
        else:
            replay_buffer = None
        # load curiosity
        curiosity = ICM.load(config["save_dir"], env=env_wrapper) if config["curiosity"] else None
        # load state normalizer
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = Normalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        noise = Noise.create_instance(config["noise"]["type"], **config["noise"]["config"])
        target_noise = Noise.create_instance(config["target_noise"]["type"], **config["target_noise"]["config"])
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

        agent = cls(
            env=env_wrapper,
            policy=policy,
            critic_a=critic_a,
            critic_b=critic_b,
            replay_buffer=replay_buffer,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            batch_size=config["batch_size"],
            noise=noise,
            noise_schedule=ScheduleWrapper(config["noise_schedule"]["type"], config["noise_schedule"]["config"]) if config["noise_schedule"] else None,
            target_noise=target_noise,
            target_noise_schedule=ScheduleWrapper(config["target_noise_schedule"]["type"], config["target_noise_schedule"]["config"]) if config["target_noise_schedule"] else None,
            noise_clip=config["noise_clip"],
            policy_update_delay=config["policy_update_delay"],
            grad_clip=config["grad_clip"],
            warmup=config["warmup"],
            N=config["N"],
            curiosity=curiosity,
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            obs_key=config["obs_key"],
            goal_key=config["goal_key"],
            achieved_goal_key=config["achieved_goal_key"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
        )
        return agent

class SAC(Agent):
    """Soft Actor Critic Agent."""

    def __init__(
        self,
        env: EnvWrapper,
        policy: StochasticDiscretePolicy|StochasticContinuousPolicy,
        # value_model: ValueModel,
        critic_a: ContinuousCritic|DiscreteCritic,
        critic_b: ContinuousCritic|DiscreteCritic|None = None,
        *,
        replay_buffer: Buffer,
        discount: float=0.99,
        tau: float=0.005,
        alpha: float=0.2, # Auto set to 1.0 if auto-tuning
        auto_entropy_tuning: bool=True,
        alpha_lr: float=3e-4, # Only used if auto entropy = True
        batch_size: int = 256,
        grad_clip: Optional[float]=None,
        warmup: int=1000,
        N: int=1,
        curiosity: Optional[ICM] = None,
        state_normalizer: Optional[Normalizer] = None,
        goal_normalizer: Optional[Normalizer] = None,
        obs_key: str = 'observation',
        goal_key: str = 'desired_goal',
        achieved_goal_key: str = 'achieved_goal',
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: Optional[str | T.device] = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, obs_key, goal_key, save_dir, device, log_level)
            self.policy = policy
            self.critic_a = critic_a
            self.critic_b = critic_b
            # clone second critic (do not copy weights) if critic_model_b None
            if not critic_b:
                self.critic_b = self.critic_a.clone(copy_weights=False)
            # self.value_model = value_model
            # self.target_value_model = self.clone_model(value_model)
            self.target_critic_model_a = self.critic_a.clone()
            self.target_critic_model_b = self.critic_b.clone()
            self.discount = discount
            self.tau = tau
            self.alpha = alpha
            self.auto_entropy_tuning = auto_entropy_tuning
            self.alpha_lr = alpha_lr
            if self.auto_entropy_tuning:
                if self.policy.distribution in ['normal', 'beta']:
                    self.target_entropy = -float(self.env.single_action_space.shape[-1])
                else: # Discrete actor
                    self.target_entropy = 0.98 * T.log(T.tensor(self.env.single_action_space.n, dtype=T.float32, device=self.device)).item()
                self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.replay_buffer = replay_buffer
            self.batch_size = batch_size
            self.grad_clip = grad_clip
            self.warmup = warmup
            self.N = N
            self.curiosity = curiosity
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.obs_key = obs_key
            self.goal_key = goal_key
            self.achieved_goal_key = achieved_goal_key
        except Exception as e:
            self.logger.error(f"Error in SAC init: {e}", exc_info=True)
        
        # set internal attributes
        try:
            # Instantiate internal attribute use_her to be switched by HER class if using DDPG
            self._use_her = False
            # Set learn_iter and sync_iter to 0. For distributed training
            # self._learn_iter = 0
            # self._sync_iter = 0

        except Exception as e:
            self.logger.error(f"Error in DDPG init internal attributes: {e}", exc_info=True)

    def clone_model(self, model, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Clones a model."""
        if device:
            device = get_device(device)
        else:
            device = self.device

        return model.clone(copy_weights, device)
    
    def _initialize_wandb(self, run_number:Optional[str]=None, run_name_prefix:Optional[str]=None, learn_iter:Optional[int]=None):
        """Initialize WandbCallback if using WandbCallback"""
        try:
            if self._wandb:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if not callback.initialized:
                            models = (self.policy, self.critic_a, self.critic_b)
                            config = self.get_config()
                            # if learn_iter:
                            #     self._learn_iter = learn_iter
                            #     config['learn_interval'] = learn_iter
                            callback.initialize_run(models, config, run_number=run_number, run_name_prefix=run_name_prefix)
        except Exception as e:
            self.logger.error(f"Error in _initialize_wandb: {e}", exc_info=True)

    def _init_her(self):
            self._use_her = True

    # def _distributed_learn(self, step: int, run_number:Optional[str]=None, learn_iter:Optional[int]=None, num_updates:int=1,
    #                       state_normalizer:Optional[Normalizer]=None, goal_normalizer:Optional[Normalizer]=None):
    #     """Used in distributed training to update the shared models.
    #     This function is overridden by the Worker class to point to the Learner class.
    #     """
    #     previous_step = self._step
    #     # Set current step to step if greater than current step
    #     if step > previous_step:
    #         self._step = step
    #         # Initialize wandb check
    #         self._initialize_wandb(run_number=run_number, run_name_prefix="train", learn_iter=learn_iter)
    #         for _ in range(num_updates):
    #             actor_loss, critic_loss = self.learn(state_normalizer, goal_normalizer)
    #         # Only store log if current step greater than previous and self._wandb
    #         if self._wandb:
    #             self._train_step_config["actor_loss"] = actor_loss
    #             self._train_step_config["critic_loss"] = critic_loss
    #             for callback in self.callbacks:
    #                 if isinstance(callback, WandbCallback):
    #                     callback.on_train_step_end(step, self._train_step_config)
    #     else:
    #         for _ in range(num_updates):
    #             actor_loss, critic_loss = self.learn(state_normalizer, goal_normalizer)

    # def get_parameters(self):
    #     """Get the parameters of all models, ensuring they are on CPU for Ray serialization."""
    #     return {
    #         'actor_model': {k: v.cpu() for k, v in self.actor_model.state_dict().items()},
    #         'critic_model_a': {k: v.cpu() for k, v in self.critic_model_a.state_dict().items()},
    #         'critic_model_b': {k: v.cpu() for k, v in self.critic_model_b.state_dict().items()},
    #         'value_model': {k: v.cpu() for k, v in self.value_model.state_dict().items()},
    #         'target_value_model': {k: v.cpu() for k, v in self.target_value_model.state_dict().items()},
    #     }

    # def apply_parameters(self, params:Dict[str, Dict[str, T.Tensor]]):
    #     """Apply params to a model. Used in distributed training."""
    #     self.actor_model.load_state_dict(params['actor_model'])
    #     self.critic_model_a.load_state_dict(params['critic_model_a'])
    #     self.critic_model_b.load_state_dict(params['critic_model_b'])
    #     self.value_model.load_state_dict(params['value_model'])
    #     self.target_value_model.load_state_dict(params['target_value_model'])

    def get_action(self,
                   states: np.ndarray|T.Tensor,
                   goals: np.ndarray|T.Tensor|None=None,
                   step: int|None=None,
                   context: str = 'train'
                   )->ActionOutput:
        """
        Select an action based on the current policy.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            step: Optional[int]: The current step.
            context: str: The context of the action (train, test, or learn).
        
        Returns:
            tuple[np.ndarray | T.Tensor, T.Tensor, Distribution | None]: Selected actions, log probabilities, and distribution.
        """
        # Set log probs and dist to None (overwritten if used)
        log_probs = None
        dist = None

        # If warmup, sample random action from action space
        if (context == 'train') and (step is not None) and (step <= self.warmup):
            return self.env.action_space.sample(), log_probs, dist

        else:
            if self.policy.distribution in ['normal', 'beta']:
                # Get action space bounds
                action_space_high = T.tensor(self.env.single_action_space.high, dtype=T.float32, device=self.policy.device)
                action_space_low = T.tensor(self.env.single_action_space.low, dtype=T.float32, device=self.policy.device)
                dist, _, _ = self.policy(states, goals)
                if context == 'learn':
                    raw_actions = dist.rsample()
                elif context == 'test':
                    raw_actions = dist.mean
                else: # train
                    raw_actions = dist.sample()
                # Use squash (Normal/Gaussian) and scale technique
                if self.policy.distribution == 'normal':
                    squashed_actions = T.tanh(raw_actions)
                    actions = action_space_low + (action_space_high - action_space_low) * (squashed_actions + 1) / 2
                    log_probs = dist.log_prob(raw_actions).sum(-1)
                    log_probs -= T.log(1 - squashed_actions.pow(2) + 1e-6).sum(-1)
                    log_probs -= T.log((action_space_high - action_space_low) / 2).sum()
                elif self.policy.distribution == 'beta':
                    actions = action_space_low + (action_space_high - action_space_low) * raw_actions
                    log_probs = dist.log_prob(raw_actions).sum(-1)
                    log_probs -= T.log(action_space_high - action_space_low).sum()
            else: # Discrete actor
                dist, _ = self.policy(states, goals)
                if context == 'learn':
                    actions = dist.probs.argmax(dim=-1)
                    probs = dist.probs
                    log_probs = T.log(probs + 1e-6)
                elif context == 'test':
                    actions = dist.probs.argmax(dim=-1)
                else: # train
                    actions = dist.sample()
            
            if context != 'learn':
                actions = actions.detach()
                if log_probs is not None:
                    log_probs = log_probs.detach()
            
            return actions, log_probs, dist

    # TODO: Add support for stochastic discrete policy
    def learn(self, step: int):
        
        # self._learn_iter += 1
        # Create learn_metrics dict
        learn_metrics = {}
            
        if self.replay_buffer.get_config()['type'] == 'PrioritizedReplayBuffer':
            if self._use_her:  # HER with prioritized replay
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
            else:  # Just prioritized replay
                states, actions, rewards, next_states, dones, trajectory_lengths, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
                
            # Log PER-specific metrics
            if self._wandb:
                # Get the actual size of used buffer (not the full capacity)
                actual_size = min(self.replay_buffer.counter, self.replay_buffer.buffer_size)
                # Get indices for all actual entries in the buffer
                valid_indices = T.arange(actual_size, device=self.replay_buffer.device)
                # Get priority info for logging
                if hasattr(self.replay_buffer, 'sum_tree') and self.replay_buffer.sum_tree is not None:
                    indices_tensor = T.tensor(indices, device=self.replay_buffer.device)
                    # Get tree indices for sampled transitions
                    tree_indices = indices_tensor + self.replay_buffer.sum_tree.capacity - 1
                    # Get priorities for sampled transitions
                    sampled_priorities = self.replay_buffer.sum_tree.tree[tree_indices].cpu().numpy()
                    valid_tree_indices = valid_indices + self.replay_buffer.sum_tree.capacity - 1
                    buffer_priorities = self.replay_buffer.sum_tree.tree[valid_tree_indices].cpu().numpy()

                else:
                    buffer_priorities = self.replay_buffer.priorities[valid_indices].cpu().numpy()
                    sampled_priorities = self.replay_buffer.priorities[indices].cpu().numpy()
                    
                # Add priority buffer metrics to learn_metrics dict
                learn_metrics.update({
                    'PER/beta': self.replay_buffer.beta,
                    'PER/sampled_priorities': sampled_priorities,
                    'PER/buffer_priorities': buffer_priorities,
                    'PER/weights': weights,
                    'PER/probs': probs,
                    'PER/mean_sampled_priority': np.mean(sampled_priorities),
                    'PER/mean_buffer_priority': np.mean(buffer_priorities),
                    'PER/max_sampled_priority': np.max(sampled_priorities),
                    'PER/max_buffer_priority': np.max(buffer_priorities),
                    'PER/weight_mean': np.mean(weights.cpu().numpy()) if weights is not None else 0.0,
                    'PER/weight_std': np.std(weights.cpu().numpy()) if weights is not None else 0.0
                })
        else:  # Standard replay buffer
            if self._use_her:
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            else:
                states, actions, rewards, next_states, dones, trajectory_lengths = self.replay_buffer.sample(self.batch_size)
            
            weights = None
            indices = None

        # Normalize states if self.normalize_inputs
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            desired_goals = self.goal_normalizer.normalize(desired_goals)
        else:
            desired_goals = None

        # Ensure training data is on correct device
        actions = actions.to(self.policy.device)
        rewards = rewards.to(self.target_critic_model_a.device)
        dones = dones.to(self.target_critic_model_a.device)
        trajectory_lengths = trajectory_lengths.to(self.target_critic_model_a.device)

        # action_space_high = T.tensor(self.env.single_action_space.high, dtype=T.float32, device=self.policy_model.device)
        # print(f'action_space_high: {action_space_high}')

        # Train ICM if curiosity and update _use_extrinsic flag
        if self.curiosity:
            # Reshape arrays to (batch_size * N, -1) to train on all steps in N
            dones_reshaped = dones.view(self.batch_size * self.N)
            mask = (dones_reshaped == 0)
            states_reshaped = states.view(self.batch_size * self.N, -1)
            next_states_reshaped = next_states.view(self.batch_size * self.N, -1)
            actions_reshaped = actions.view(self.batch_size * self.N, -1)
            # Replace next state with state value where done = True b/c next state value could be reset observation
            # returned by environment (IsaacSim)
            next_states_reshaped = T.where(mask.unsqueeze(1), next_states_reshaped, states_reshaped)
            curiosity_loss = self.curiosity.train(states_reshaped, next_states_reshaped, actions_reshaped)
            if step > self.curiosity.extrinsic_threshold:
                self.curiosity._use_extrinsic = True
            else:
                self.curiosity._use_extrinsic = False

        # with T.no_grad():
        #     # Get target values
        #     dist, _, _ = self.actor_model(states[:,-1,:], desired_goals[:,-1,:] if desired_goals is not None else None)
        #     new_actions = dist.rsample()
        #     squashed_actions = T.tanh(new_actions)
        #     scaled_actions = squashed_actions * action_space_high
        #     log_probs = dist.log_prob(new_actions).sum(-1)
        #     log_probs -= T.log(1-squashed_actions.pow(2) + 1e-6).sum(-1)
        #     log_probs -= T.log(action_space_high).sum()
        #     q_1 = self.critic_model_a(states[:,-1,:], scaled_actions, desired_goals[:,-1,:] if desired_goals is not None else None).view(-1)
        #     q_2 = self.critic_model_b(states[:,-1,:], scaled_actions, desired_goals[:,-1,:] if desired_goals is not None else None).view(-1)
        #     min_q = T.minimum(q_1, q_2)
        #     if self.auto_entropy_tuning:
        #         current_alpha = self.log_alpha.exp()
        #     else:
        #         current_alpha = self.alpha
        #     v_targ = min_q - current_alpha * log_probs

        # v_preds = self.value_model(states[:,-1,:], desired_goals[:,-1,:] if desired_goals is not None else None).view(-1)
        # self.value_model.optimizer.zero_grad()
        # value_loss = 0.5 * F.mse_loss(v_preds, v_targ)
        # value_loss.backward()
        # if self.grad_clip:
        #     T.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_clip)
        # self.value_model.optimizer.step()

        if self.auto_entropy_tuning:
            current_alpha = self.log_alpha.exp()
        else:
            current_alpha = self.alpha

        with T.no_grad():
            # Compute intrinsic reward if using ICM
            if self.curiosity:
                intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                    states_reshaped,
                    next_states_reshaped,
                    actions_reshaped
                )
                intrinsic_reward = intrinsic_reward.view(self.batch_size, self.N)
                if self.curiosity._use_extrinsic:
                    rewards += intrinsic_reward
                else:
                    rewards = intrinsic_reward
            
            q_targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic_model_a.device
            ).squeeze()

            # Sample new actions from current policy 
            # dist, _, _ = self.actor_model(next_states[:,-1,:], desired_goals[:,-1,:] if desired_goals is not None else None)
            # new_actions = dist.rsample()
            # squashed_actions = T.tanh(new_actions)
            # scaled_actions = squashed_actions * action_space_high
            # log_probs = dist.log_prob(new_actions).sum(-1)
            # log_probs -= T.log(1-squashed_actions.pow(2) + 1e-6).sum(-1)
            # log_probs -= T.log(action_space_high).sum()

            ## Critic Update ##
            target_actions, log_probs, dist = self.get_action(next_states[:,-1,:], desired_goals[:,-1,:] if desired_goals is not None else None, context='learn')

            # Continuous critic target values
            if self.policy.distribution in ['normal', 'beta']:
                target_values_1 = self.target_critic_model_a(
                    next_states[:,-1,:],
                    target_actions,
                    desired_goals[:,-1,:] if desired_goals is not None else None
                    ).squeeze()
                target_values_2 = self.target_critic_model_b(
                    next_states[:,-1,:],
                    target_actions,
                    desired_goals[:,-1,:] if desired_goals is not None else None
                    ).squeeze()
                target_values = T.minimum(target_values_1, target_values_2) - current_alpha * log_probs

            else: # Discrete critic target values
                target_values_1 = self.target_critic_model_a(
                    next_states[:,-1,:],
                    desired_goals[:,-1,:] if desired_goals is not None else None
                )
                target_values_2 = self.target_critic_model_b(
                    next_states[:,-1,:],
                    desired_goals[:,-1,:] if desired_goals is not None else None
                )
                target_values = (dist.probs * (T.minimum(target_values_1, target_values_2) - current_alpha * log_probs)).sum(-1)

            no_dones_mask = (dones.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            q_targets += no_dones_mask * gamma_pow * target_values

            # Apply HER-specific clamping if needed
            if self._use_her:
                if self.curiosity and not self.curiosity._use_extrinsic:
                    pass
                else:
                    q_targets = T.clamp(q_targets, min=-1/(1-self.discount))

        # Continuous critic predictions
        if self.policy.distribution in ['normal', 'beta']:
            q1_preds = self.critic_a(
                states[:,0,:],
                actions[:,0,:],
                desired_goals[:,0,:] if desired_goals is not None else None).squeeze()
            q2_preds = self.critic_b(
                states[:,0,:],
                actions[:,0,:],
                desired_goals[:,0,:] if desired_goals is not None else None).squeeze()

        else: # Discrete critic predictions
            q1 = self.critic_a(
                states[:,0,:],
                desired_goals[:,0,:] if desired_goals is not None else None
            )
            q2 = self.critic_b(
                states[:,0,:],
                desired_goals[:,0,:] if desired_goals is not None else None
            )
            buffer_actions = actions[:,0,:].squeeze(-1).long().unsqueeze(1)
            q1_preds = q1.gather(1, buffer_actions).squeeze(1)
            q2_preds = q2.gather(1, buffer_actions).squeeze(1)

        # Calculate TD errors
        q1_loss = (q1_preds - q_targets.detach()).pow(2)
        q2_loss = (q2_preds - q_targets.detach()).pow(2)
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            q1_loss = weights.to(self.critic_a.device) * q1_loss
            q2_loss = weights.to(self.critic_b.device) * q2_loss
        critic_loss = 0.5 * (q1_loss.mean() + q2_loss.mean())

        self.critic_a.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.critic_a.parameters(), self.grad_clip)
            T.nn.utils.clip_grad_norm_(self.critic_b.parameters(), self.grad_clip)
        self.critic_a.optimizer.step()
        self.critic_b.optimizer.step()

        ## Update Policy ##
        # dist, _, _ = self.actor_model(states[:,0,:], desired_goals[:,0,:] if desired_goals is not None else None)
        # new_actions = dist.rsample()
        # squashed_actions = T.tanh(new_actions)
        # scaled_actions = squashed_actions * action_space_high
        # log_probs = dist.log_prob(new_actions).sum(-1)
        # log_probs -= T.log(1-squashed_actions.pow(2) + 1e-6).sum(-1)

        new_actions, log_probs, dist = self.get_action(states[:,0,:], desired_goals[:,0,:] if desired_goals is not None else None, context='learn')
        # Continuous policy update
        if self.policy.distribution in ['normal', 'beta']:
            q1 = self.critic_a(states[:,0,:], new_actions, desired_goals[:,0,:] if desired_goals is not None else None).squeeze()
            q2 = self.critic_b(states[:,0,:], new_actions, desired_goals[:,0,:] if desired_goals is not None else None).squeeze()
            min_q = T.minimum(q1, q2)
            actor_loss = current_alpha * log_probs - min_q
        
        else: # Discrete policy update
            q1 = self.critic_a(states[:,0,:], desired_goals[:,0,:] if desired_goals is not None else None)
            q2 = self.critic_b(states[:,0,:], desired_goals[:,0,:] if desired_goals is not None else None)
            min_q = T.minimum(q1, q2)
            actor_loss = (dist.probs * (current_alpha * log_probs - min_q)).sum(-1)


        if weights is not None:
            actor_loss = weights.to(self.policy.device) * actor_loss

        actor_loss = actor_loss.mean()

        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy.optimizer.step()

        if self.auto_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            if self.policy.distribution in ['normal', 'beta']:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            else: # Discrete actor
                alpha_loss = -(self.log_alpha * ((dist.probs * log_probs).sum(dim=-1) + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Perform soft update on target networks
        if not self._use_her:
            self.soft_update(self.critic_a, self.target_critic_model_a)
            self.soft_update(self.critic_b, self.target_critic_model_b)

        # Calculate TD Error to update priorities and for logging
        error = q_targets - T.minimum(q1_preds, q2_preds)

        # Update priorities if using prioritized replay - only on update_freq steps
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:# and hasattr(self.replay_buffer, 'beta_update_freq'):
            #DEBUG
            # print(f'indices shape: {indices.shape}')
            # print(f'error shape: {error.flatten().shape}')
            self.replay_buffer.update_priorities(indices, error.detach().flatten().to(self.replay_buffer.device))

        learn_metrics.update({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": error.mean().item(),
            # "actor_predictions": new_actions.mean().item(),
            "critic_predictions": min_q.mean().item(),
            "target_critic_predictions": target_values.mean().item(),
            "alpha": current_alpha,
            "entropy": float(-log_probs.mean().item())
        })
        if self.curiosity:
            learn_metrics.update({
                "curiosity_loss": curiosity_loss.item(),
                "intrinsic_reward": intrinsic_reward.mean().item(),
                "use_extrinsic": self.curiosity._use_extrinsic,
                "reward_weight": self.curiosity.reward_weight * self.curiosity.reward_scheduler.get_factor() \
                    if self.curiosity.reward_scheduler else self.curiosity.reward_weight
            })

        return learn_metrics

    def soft_update(self, current, target):
        with T.no_grad():
            for current_params, target_params in zip(current.parameters(), target.parameters()):
                target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)

            # Copy buffers (running_mean, running_var)
            main_buffers = dict(current.named_buffers())
            target_buffers = dict(target.named_buffers())
            for name in main_buffers:
                if name in target_buffers:
                    target_buffers[name].copy_(main_buffers[name])

    def _step(self,
              step: int,
              states: np.ndarray | T.Tensor | dict,
              max_episodes: int,
              episode_scores: np.ndarray,
              completed_episodes: np.ndarray,
              score_history: deque[float],
              best_reward: float,
              learn: bool = True,
              training: bool = True
              ):

        obs, goals = self._preprocess_inputs(states)
        actions, _, _ = self.get_action(obs, goals, step, context='train' if training else 'test')
        # Format actions
        actions = self.env.format_actions(actions)
        next_states, rewards, dones, infos = self.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        obs, actions, rewards, next_states, dones = (
            T.tensor(obs, dtype=T.float32, device=self.device) if isinstance(obs, np.ndarray) else obs,
            T.tensor(actions, dtype=T.float32, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int8, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )


        episode_scores += rewards.flatten()

        buffer_trajectories = {
            'states': infos['n-step trajectory']['states'],
            'actions': infos['n-step trajectory']['actions'],
            'rewards': infos['n-step trajectory']['rewards'],
            'next_states': infos['n-step trajectory']['next_states'],
            'dones': infos['n-step trajectory']['dones'],
            'trajectory_lengths': infos['n-step trajectory']['trajectory_lengths']
        }
        if self.goal_key:
            buffer_trajectories['desired_goals'] = infos['n-step trajectory']['desired_goals']
            buffer_trajectories['state_achieved_goals'] = infos['n-step trajectory']['state_achieved_goals']
            buffer_trajectories['next_state_achieved_goals'] = infos['n-step trajectory']['next_state_achieved_goals']

        if training:
            self.replay_buffer.add(**buffer_trajectories)
            # Update normalizer if state_normalizer
            next_obs, next_goals = self.extract_states_goals(next_states)
            if self.state_normalizer:
                self.state_normalizer.add(next_obs)
            if self.goal_normalizer:
                self.goal_normalizer.add(next_goals)

        done_episodes = T.nonzero(dones, as_tuple=False).flatten()
        episode_logs = []
        
        for i in done_episodes:
            # Increment completed episodes for env by 1
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)

            # check if best reward
            if training and avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': round(float(episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }

            if training:
                episode_log.update({
                    'best_reward': round(float(best_reward), 2),
                    'best': 1 if avg_reward > best_reward else 0
                })
            episode_logs.append(episode_log)

        step_log = {} # Collect step metrics

        # Check if past warmup
        if learn and step > self.warmup and self.replay_buffer.counter > self.batch_size:
            # Check if distributed
            # if self._distributed:
            #     self._distributed_learn(step, self._run_number)
            # else:
            learn_metrics = self.learn(step)
            step_log.update({**learn_metrics})

        # self._train_step_config["step_reward"] = rewards.mean()
        step_log.update({
            'step_reward': rewards.mean(),
        })

        return{
            'episode_scores': episode_scores,
            'completed_episodes': completed_episodes,
            'score_history': score_history,
            'next_states': next_states,
            'step_log': step_log,
            'episode_logs': episode_logs,
            'done': completed_episodes.sum() >= max_episodes
        }

    def train(self, num_episodes: int, steps_per_learn: int = 1, render_freq: int = 0, seed: int | None = None):
        """Trains the model for 'episodes' number of episodes."""

        init_dict = self._initialize_run(seed=seed, num_episodes=num_episodes)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        
        while completed_episodes.sum() < num_episodes:
            # Increment step count
            step += 1
            # If distributed, sync to shared agent
            # if self._distributed and self._step % self._sync_iter == 0:
            #     params = self.get_parameters()
            #     self.apply_parameters(params)
            # Determine if step should perform update
            learn = step % steps_per_learn == 0
            # Perform train step
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn)
            # Update states, episode scores, completed episodes, and score history
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            # log to callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)

        self.env.close()

    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""

        init_dict = self._initialize_run(seed=seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']

        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            step += 1
            step_result = self._step(step, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, learn = False, training = False)
            states = step_result['next_states']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])
            
            render = True
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Testing Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)
            
                if render and render_freq > 0 and completed_episodes.sum() % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "policy_model": self.policy.get_config(),
            # "value_model": self.value_model.get_config(),
            "critic_model_a": self.critic_a.get_config(),
            "critic_model_b": self.critic_b.get_config(),
            "replay_buffer": self.replay_buffer.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "alpha": self.alpha,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "alpha_lr": self.alpha_lr,
            "batch_size": self.batch_size,
            'grad_clip': self.grad_clip,
            'warmup': self.warmup,
            'N': self.N,
            "curiosity": self.curiosity.get_config() if self.curiosity else None,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer else None,
            "obs_key": self.obs_key,
            "goal_key": self.goal_key,
            "achieved_goal_key": self.achieved_goal_key,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir,
            "device": self.device.type,
            "log_level": logging.getLevelName(self.logger.getEffectiveLevel()).lower()
        }


    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        # self.value_model.save(self.save_dir)
        self.critic_a.save(self.save_dir, 'critic_a')
        self.critic_b.save(self.save_dir, 'critic_b')
        if self.curiosity:
            self.curiosity.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool=True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        distribution = config['policy_model']['distribution']
        if distribution == 'categorical':
            policy = StochasticDiscretePolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
            critic_a = DiscreteCritic.load(config_dir, 'critic_a', load_weights, env=env_wrapper)
            critic_b = DiscreteCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        elif distribution in ['beta', 'normal']:
            policy = StochasticContinuousPolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
            critic_a = ContinuousCritic.load(config_dir, 'critic_a', load_weights, env=env_wrapper)
            critic_b = ContinuousCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        # value_model = ValueModel.load(config_dir, 'value_model', load_weights, env=env_wrapper)
        
        if config['replay_buffer'] is not None:
            config['replay_buffer']['config']['env'] = env_wrapper
            if config['replay_buffer']['type'] == 'PrioritizedReplayBuffer':
                replay_buffer = PrioritizedReplayBuffer(**config["replay_buffer"]["config"])
            else:
                replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
        else:
            replay_buffer = None
        curiosity = ICM.load(config["save_dir"], env=env_wrapper) if config["curiosity"] else None
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "/state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = Normalizer.load(config["goal_normalizer"], config["save_dir"] + "/goal_normalizer.pt") if config["goal_normalizer"] else None
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

        agent = cls(
            env = env_wrapper,
            policy = policy,
            # value_model = value_model,
            critic_a = critic_a,
            critic_b = critic_b,
            replay_buffer=replay_buffer,
            discount=config["discount"],
            tau=config["tau"],
            alpha=config["alpha"],
            auto_entropy_tuning=config["auto_entropy_tuning"],
            alpha_lr=config["alpha_lr"],
            batch_size=config["batch_size"],
            grad_clip=config['grad_clip'],
            warmup = config['warmup'],
            N = config['N'],
            curiosity=curiosity,
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            obs_key=config["obs_key"],
            goal_key=config["goal_key"],
            achieved_goal_key=config["achieved_goal_key"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
            log_level=config["log_level"]
        )

        return agent

class HER(Agent):
    """Hindsight Experience Replay Agent wrapper."""

    def __init__(
        self,
        agent: DDPG|TD3|SAC,
        strategy: str = 'final',
        tolerance: float = 0.5,
        num_goals: int = 4,
        save_dir: str = "models",
    ):
        """
        Initializes the HER agent wrapper.
        
        Args:
            agent (Agent): The underlying agent (e.g., DDPG, TD3) to wrap with HER.
            strategy (str): HER strategy for goal sampling ('final', 'future', etc.).
            tolerance (float): Distance threshold for success determination.
            num_goals (int): Number of goals to sample for hindsight replay.
            save_dir (str): Directory to save models and logs.
            # callbacks (Optional[list[Callback]]): List of callbacks for training.
        """
        try:
            self.agent = agent
            self.strategy = strategy
            self.tolerance = tolerance
            self.num_goals = num_goals
            self.save_dir = self._setup_save_dir(save_dir)

            # Set learn iter to 0. For distributed training
            self._learn_iter = 0

        except Exception as e:
            self.logger.error(f"Error in HER init: {e}", exc_info=True)

        # Internal attributes
        try:
            # Initialize HER flag in agent
            self.base_agent._init_her()
            
            # Set distance threshold based on environment type
            if isinstance(self.agent.env.env, gym.vector.SyncVectorEnv):
                # Vectorized environment: set distance_threshold for each sub-environment
                for i in range(len(self.agent.env.env.envs)):
                    base_env = self.agent.env.get_base_env(i)
                    if hasattr(base_env, "distance_threshold"):
                        base_env.distance_threshold = self.tolerance
                    else:
                        self.logger.warning(f"Environment {base_env} does not have distance_threshold attribute")
            else:
                # Non-vectorized environment: set directly if attribute exists
                if hasattr(self.agent.env.env, "distance_threshold"):
                    self.agent.env.env.distance_threshold = self.tolerance
                else:
                    self.logger.warning("Underlying environment does not have distance_threshold attribute")

            # self._sync_iter = 1
            # self._learn_iter = 0

        except Exception as e:
            self.logger.error(f"Error in HER init internal attributes: {e}", exc_info=True)
        
    @property
    def base_agent(self):
        """Return the base agent"""
        return self.agent

    # def get_parameters(self):
    #     """Get the parameters of all models, ensuring they are on CPU for Ray serialization."""
    #     return self.agent.get_parameters()

    # def apply_parameters(self, params:Dict[str, Dict[str, T.Tensor]]):
    #     """Apply params to a model. Used in distributed training."""
    #     self.agent.apply_parameters(params)

    # def _distributed_learn(self, step: int, run_number:str=None, learn_iter:int=None, num_updates:int=1):
    #     """Used in distributed training to update the shared models.
    #     This function is overridden by the Worker class to point to the Learner class.
    #     """
    #     #DEBUG
    #     state_actor_id = str(self.state_normalizer.shared_normalizer)
    #     goal_actor_id = str(self.goal_normalizer.shared_normalizer)
    #     self.agent.logger.debug(f"Learner using normalizer actors:")
    #     self.agent.logger.debug(f"  State normalizer actor ID: {state_actor_id}")
    #     self.agent.logger.debug(f"  Goal normalizer actor ID: {goal_actor_id}")
    #     self.agent._distributed_learn(step, run_number, learn_iter, num_updates, self.state_normalizer, self.goal_normalizer)
    #     # Update target networks
    #     if isinstance(self.agent, DDPG):
    #         print(f"Updating DDPG target networks")
    #         self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
    #         self.agent.soft_update(self.agent.critic_model, self.agent.target_critic_model)
    #     elif isinstance(self.agent, TD3):
    #         self.agent.logger.info(f"Updating TD3 target networks")
    #         self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
    #         self.agent.soft_update(self.agent.critic_model_a, self.agent.target_critic_model_a)
    #         self.agent.soft_update(self.agent.critic_model_b, self.agent.target_critic_model_b)

    # def format_trajectory(self, n_step_data):
    #     # Extract data
    #     states = n_step_data['states']
    #     next_states = n_step_data['next_states']
    #     actions = n_step_data['actions']
    #     rewards = n_step_data['rewards']
    #     dones = n_step_data['dones']
        
    #     # Get dimensions
    #     # num_envs, num_steps = states.shape
    #     # obs_dim = len(states[0, 0]['observation'])  # Get actual observation dimension
    #     obs_dim = self.agent.env.single_observation_space[self.obs_key].shape[-1]
    #     # goal_dim = len(states[0, 0]['achieved_goal']) # Get actual goal dimension
    #     goal_dim = self.agent.env.single_observation_space[self.achieved_goal_key].shape[-1]
        
    #     # Initialize arrays for rearranged data
    #     obs_array = np.zeros((self.num_envs, self.agent.N, obs_dim))
    #     achieved_goals_array = np.zeros((self.num_envs, self.agent.N, goal_dim))
    #     desired_goals_array = np.zeros((self.num_envs, self.agent.N, goal_dim))
        
    #     next_obs_array = np.zeros((self.num_envs, self.agent.N, obs_dim))
    #     next_achieved_goals_array = np.zeros((self.num_envs, self.agent.N, goal_dim))
        
    #     # Fill arrays by extracting from dictionaries
    #     for env_idx in range(self.num_envs):
    #         for step_idx in range(self.agent.N):
    #             # Current states
    #             state_dict = states[env_idx, step_idx]
    #             #DEBUG
    #             print(f'state_dict: {state_dict}')
    #             obs_array[env_idx, step_idx] = state_dict['observation']
    #             achieved_goals_array[env_idx, step_idx] = state_dict['achieved_goal']
    #             desired_goals_array[env_idx, step_idx] = state_dict['desired_goal']
                
    #             # Next states
    #             next_state_dict = next_states[env_idx, step_idx]
    #             next_obs_array[env_idx, step_idx] = next_state_dict['observation']
    #             next_achieved_goals_array[env_idx, step_idx] = next_state_dict['achieved_goal']
        
    #     return {
    #         'states': obs_array,
    #         'achieved_goals': achieved_goals_array,
    #         'desired_goals': desired_goals_array,
    #         'next_states': next_obs_array,
    #         'next_achieved_goals': next_achieved_goals_array,
    #         'actions': actions,
    #         'rewards': rewards,
    #         'dones': dones
    #     }

    def _step(self, env: EnvWrapper, step: int, trajectories: list[list[tuple]], states: dict, max_episodes: int, episode_scores: np.ndarray,
              completed_episodes: np.ndarray, score_history: deque[float], best_reward: float, success_counter: float, training: bool = True):
        """
        Perform a single training step.
        """
        # Get actions for all environments
        actions = self.base_agent.get_action(env, states, step=step, testing=not training)
        actions = env.format_actions(actions)
        next_states, rewards, dones, infos = env.step(actions)
        episode_scores += rewards
        step_logs = {f'step_reward': rewards.mean()}
        
        # Store transitions in the env trajectory
        for i in range(env.num_envs):
            trajectories[i].append(
                (
                    infos['n-step trajectory']['states'][i],
                    infos['n-step trajectory']['actions'][i],
                    infos['n-step trajectory']['rewards'][i],
                    infos['n-step trajectory']['next_states'][i],
                    infos['n-step trajectory']['dones'][i],
                    infos['n-step trajectory']['state_achieved_goals'][i],
                    infos['n-step trajectory']['next_state_achieved_goals'][i],
                    infos['n-step trajectory']['desired_goals'][i]
                )
            )

        if training:
            # Update normalizers
            if self.base_agent.state_normalizer:
                self.base_agent.state_normalizer.add(
                    T.tensor(next_states['observation'], dtype=T.float32, device=self.agent.state_normalizer.device.type)
                )
            if self.base_agent.goal_normalizer:
                self.base_agent.goal_normalizer.add(
                    T.tensor(next_states['achieved_goal'], dtype=T.float32, device=self.base_agent.goal_normalizer.device.type)
                )

        done_episodes = np.flatnonzero(dones) # Get indices of completed episodes
        episode_logs = []
        successes = 0
        for i in done_episodes:
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)
            
            # Store hindsight trajectory if training
            if training:
                self.store_hindsight_trajectory(trajectories[i])
                # check if best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save()
            # Calculate success rate
            goal_distance = np.linalg.norm(next_states['achieved_goal'][i] - states['desired_goal'][i], axis=-1)
            # successes += (goal_distance <= self.tolerance).astype(np.int32)
            success_counter += (goal_distance <= self.tolerance).astype(np.int32)
            success_perc = (success_counter / completed_episodes.sum())
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': episode_scores[i].round(2),
                'avg_reward': avg_reward.round(2),
                'goal_distance': goal_distance.round(2),
                # 'Successes': successes,
                'success_rate': success_perc.round(2)
            }
            if training:
                episode_log.update({
                    'best_reward': best_reward.round(2),
                    'best': 1 if avg_reward > best_reward else 0,
                })
            episode_logs.append(episode_log)
            trajectories[i] = []

        return{
            'episode_scores': episode_scores,
            'completed_episodes': completed_episodes,
            'score_history': score_history,
            'trajectories': trajectories,
            'next_states': next_states,
            'success_counter': success_counter,
            'step_logs': step_logs,
            'episode_logs': episode_logs,
            'done': completed_episodes.sum() >= max_episodes
        }

    def train(self, num_epochs: int, num_cycles: int, num_episodes: int, num_updates: int, num_envs: int = 1, render_freq: int = 0, seed: int | None = None):
        """
        Train the HER agent with a vectorized environment setup, following the HER paper's experiment structure.

        Args:
            num_epochs (int): Number of training epochs.
            num_cycles (int): Number of cycles per epoch.
            num_episodes (int): Number of episodes to collect per cycle across all environments.
            num_updates (int): Number of optimization steps per cycle after collecting episodes.
            num_envs (int): Number of parallel environments (default: 1).
            render_freq (int): Frequency of rendering (in total completed episodes).
            seed (int, optional): Random seed for reproducibility.
        """
        
        init_dict = self._initialize_run(num_envs, seed, num_episodes=num_episodes, num_epochs=num_epochs, num_cycles=num_cycles, num_updates=num_updates)
        env = init_dict['env']
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        trajectories = [[] for _ in range(num_envs)]
        success_counter = 0.0

        # Training loop
        for epoch in range(num_epochs):

            for cycle in range(num_cycles):
                completed_before_cycle = completed_episodes.sum()

                # Collect episodes until num_episodes_per_cycle are completed
                while completed_episodes.sum() < completed_before_cycle + num_episodes:
                    # If distributed, sync to shared agent
                    # if self.agent._distributed and self.agent._step % self._sync_iter == 0:
                    #     params = self.get_parameters()
                    #     self.apply_parameters(params)
                    step += 1

                    step_result = self._step(env, step, trajectories, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, success_counter)
                    states = step_result['next_states']
                    trajectories = step_result['trajectories']
                    episode_scores = step_result['episode_scores']
                    completed_episodes = step_result['completed_episodes']
                    score_history = step_result['score_history']
                    success_counter = step_result['success_counter']

                    if self.base_agent.callbacks:
                        for callback in self.base_agent.callbacks:
                            callback.on_train_step_end(step=step, logs=step_result['step_logs'])

                    render = True # Flag to keep track of render status to avoid rendering multiple times per step
                    for episode_log in step_result['episode_logs']:
                        # Print complete episode metrics to console
                        print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                        # Reset episode score
                        episode_scores[episode_log['env']] = 0
                        best_reward = episode_log['best_reward']

                        if self.base_agent.callbacks:
                            for callback in self.base_agent.callbacks:
                                callback.on_train_epoch_end(epoch=step, logs=episode_log)

                        # Check if number of completed episodes should trigger render
                        if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                            print(f"Rendering episode {episode_log['episode']} during training...")
                            self.render_episode(episode_log['episode'], step, context='train')
                            render = False

                # Perform optimization after collecting episodes
                if step > self.base_agent.warmup and self.base_agent.replay_buffer.counter > self.base_agent.batch_size:
                    # Check if distributed
                    # if self.base_agent._distributed:
                    #     self._distributed_learn(self.agent._step, self._run_number, num_updates)
                    # else:
                    for _ in range(num_updates):
                        learn_logs = self.base_agent.learn(step)

                    # Update target networks
                    if isinstance(self.base_agent, DDPG):
                        # self.agent.logger.debug(f"Updating DDPG target networks")
                        self.agent.soft_update(self.base_agent.policy, self.base_agent.target_policy)
                        self.agent.soft_update(self.base_agent.critic, self.base_agent.target_critic)
                    elif isinstance(self.base_agent, TD3):
                        # self.agent.logger.debug(f"Updating TD3 target networks")
                        self.agent.soft_update(self.base_agent.policy, self.base_agent.target_policy_model)
                        self.agent.soft_update(self.base_agent.critic_model_a, self.base_agent.target_critic_model_a)
                        self.agent.soft_update(self.base_agent.critic_model_b, self.base_agent.target_critic_model_b)
                    elif isinstance(self.base_agent, SAC):
                        self.agent.soft_update(self.base_agent.value_model, self.base_agent.target_value_model)

                else:
                    learn_logs = None

                if self.base_agent.callbacks:
                    for callback in self.base_agent.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=learn_logs)

        if self.base_agent.callbacks:
            for callback in self.base_agent.callbacks:
                callback.on_train_end(logs=episode_log)

        env.close()
    
    def test(self, num_episodes: int, num_envs: int = 1, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""
        
        init_dict = self._initialize_run(num_envs, seed, training=False)
        env = init_dict['env']
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        trajectories = [[] for _ in range(num_envs)]
        success_counter = 0.0

        while completed_episodes.sum() < num_episodes:
            step += 1
            step_result = self._step(env, step, trajectories, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, success_counter, training=False)
            states = step_result['next_states']
            trajectories = step_result['trajectories']
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']
            success_counter = step_result['success_counter']

            if self.base_agent.callbacks:
                for callback in self.base_agent.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_logs'])

            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Testing Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    # Call the test function to render an episode
                    self.render_episode(episode_log['episode'], step, context='test')
                    render = False


                if self.base_agent.callbacks:
                    for callback in self.base_agent.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)

        if self.base_agent.callbacks:
            for callback in self.base_agent.callbacks:
                callback.on_test_end()

        env.close()

    def store_hindsight_trajectory(self, trajectory):
        """
        Store hindsight-augmented transitions from a completed trajectory into the replay buffer with n-step rewards.
        
        Args:
            trajectory (list): List of tuples or dicts with transition data:
                (state, action, reward, next_state, done, achieved_goal, next_achieved_goal, desired_goal)
        """
        states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals = zip(*trajectory)
        
        # Convert to NumPy arrays for efficiency
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        achieved_goals = np.array(achieved_goals)
        next_achieved_goals = np.array(next_achieved_goals)
        desired_goals = np.array(desired_goals)

        # experiences = []
        goals = []
        
        if self.strategy == "final":
            # New desired goal is the final achieved goal
            new_desired_goal = next_achieved_goals[-1]
            
            # Compute new rewards for all transitions
            new_rewards = [self.agent.env.get_base_env().compute_reward(achieved_goals[i], new_desired_goal, {}) 
                        for i in range(len(trajectory))]
            
            # Compute n-step returns and store transitions
            for t in range(len(trajectory)):
                # Number of steps for n-step return (limited by trajectory length)
                k = min(self.agent.N, len(trajectory) - t)
                
                # Calculate n-step return
                # n_step_reward = sum([self.agent.discount**i * new_rewards[t + i] for i in range(k)])
                
                states = np.append(states, np.expand_dims(states[t], axis=0), axis=0)
                actions = np.append(actions, np.expand_dims(actions[t], axis=0), axis=0)
                rewards = np.append(rewards, np.expand_dims(new_rewards[t], axis=0), axis=0)
                next_states = np.append(next_states, np.expand_dims(next_states[t], axis=0), axis=0)
                dones = np.append(dones, np.expand_dims(dones[t], axis=0), axis=0)
                achieved_goals = np.append(achieved_goals, np.expand_dims(achieved_goals[t], axis=0), axis=0)
                next_achieved_goals = np.append(next_achieved_goals, np.expand_dims(next_achieved_goals[t], axis=0), axis=0)
                desired_goals = np.append(desired_goals, np.expand_dims(new_desired_goal, axis=0), axis=0)
        
        elif self.strategy == "future":
            # For "future" strategy, n-step rewards are tricky due to changing goals.
            # Using single-step rewards as a fallback (modify if needed)
            for t in range(len(trajectory)):
                step_goals = []
                for _ in range(self.num_goals):
                    if t + 1 >= len(trajectory):
                        break
                    goal_idx = np.random.randint(t + 1, len(trajectory))
                    step_goals.append(next_achieved_goals[goal_idx])
                goals.append(step_goals)
            
            for t in range(len(trajectory)):
                for goal in goals[t]:
                    goal_distance = np.linalg.norm(next_achieved_goals[t] - goal, axis=-1)
                    new_reward = self.agent.env.get_base_env().compute_reward(next_achieved_goals[t], goal, {})
                    states = np.append(states, np.expand_dims(states[t], axis=0), axis=0)
                    actions = np.append(actions, np.expand_dims(actions[t], axis=0), axis=0)
                    rewards = np.append(rewards, np.expand_dims(new_reward, axis=0), axis=0)
                    next_states = np.append(next_states, np.expand_dims(next_states[t], axis=0), axis=0)
                    dones = np.append(dones, np.expand_dims(dones[t], axis=0), axis=0)
                    achieved_goals = np.append(achieved_goals, np.expand_dims(achieved_goals[t], axis=0), axis=0)
                    next_achieved_goals = np.append(next_achieved_goals, np.expand_dims(next_achieved_goals[t], axis=0), axis=0)
                    desired_goals = np.append(desired_goals, np.expand_dims(goal, axis=0), axis=0)
                
        
        elif self.strategy == "none":
            pass  # No hindsight replay
        
        # Add all experiences to the replay buffer
        self.agent.replay_buffer.add(states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals)
                
        

    # def set_normalizer_state(self, config):
    #     self.agent.state_normalizer.set_state(config)

    def get_config(self):
        config = {
            "agent_type": self.__class__.__name__,
            "agent": self.agent.get_config(),
            "strategy": self.strategy,
            "tolerance":self.tolerance,
            "num_goals": self.num_goals,
            "device": self.agent.device.type,
            "save_dir": self.save_dir,
        }

        return config
    
    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        self.agent.save()

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        agent = load_agent(Path(config["agent"]["save_dir"]), load_weights)
        her = cls(agent, config["strategy"], config["tolerance"], config["num_goals"], config["save_dir"])
        
        return her
    
class PPO(Agent):
    """
    Proximal Policy Optimization (PPO) agent implementation.

    This agent uses policy and value networks to learn an optimal policy for a given environment
    using the PPO algorithm. It supports features such as Generalized Advantage Estimation (GAE),
    reward clipping, and gradient clipping for stable learning.

    Attributes:
        env (EnvWrapper): The environment wrapper for the agent.
        policy_model: The policy model used for action selection.
        value_model: The value model used for state-value prediction.
        discount (float): Discount factor for future rewards.
        gae_coefficient (float): GAE smoothing coefficient.
        policy_clip (float): Clipping value for policy ratio updates.
        policy_clip_schedule (ScheduleWrapper): Rate at which to decay policy clip per learn epoch.
        value_clip (float): Clipping value for value model updates.
        value_clip_schedule (ScheduleWrapper): Rate at which to decay value clip per learn epoch.
        value_loss_coefficient (float): value to weight the value loss by.
        entropy_coefficient (float): Coefficient for entropy regularization.
        entropy_schedule (ScheduleWrapper): Rate at which to decay entropy coefficient per learn epoch.
        kl_coefficient (float): Coefficient for KL divergence penalty.
        kl_adapter (AdaptiveKL): Adjusts kl_coefficient to keep KL Divergence near target.
        normalize_advantages (bool): Whether to normalize advantages.
        state_normalizer (Normalizer): Normalizer for state inputs.
        goal_normalizer (Normalizer): Normalizer for goal inputs.
        obs_key (str): Key for observation in the state space dict.
        goal_key (str): Key for desired goal in the state space dict.
        achieved_goal_key (str): Key for achieved goal in the state space dict.
        grad_clip (float): Maximum norm for gradients.
        reward_clip (float): Maximum absolute value for reward clipping.
        callbacks (List): List of callback objects for logging and monitoring.
        save_dir (str): Directory to save models and configurations.
        device (str): Device for computations ('cpu' or 'cuda').
    """

    def __init__(self,
                 env: EnvWrapper,
                 policy_model: StochasticContinuousPolicy | StochasticDiscretePolicy,
                 value_model: ValueModel,
                 discount: float = 0.99,
                 gae_coefficient: float = 0.95,
                 policy_clip: float = 0.2,
                 policy_clip_schedule: Optional[ScheduleWrapper] = None,
                 value_clip: float = 0.2,
                 value_clip_schedule: Optional[ScheduleWrapper] = None,
                 value_loss_coefficient: float = 1.0,
                 entropy_coefficient: float = 0.01,
                 entropy_schedule: Optional[ScheduleWrapper] = None,
                 kl_coefficient: float = 0.0,
                 kl_adapter: Optional[AdaptiveKL] = None,
                 normalize_advantages: bool = True,
                 curiosity: Optional[ICM] = None,
                 state_normalizer: Optional[Normalizer] = None,
                 goal_normalizer: Optional[Normalizer] = None,
                 obs_key: str = 'observation',
                 goal_key: str = 'desired_goal',
                 achieved_goal_key: str = 'achieved_goal',
                 grad_clip: float = float('inf'),
                 reward_clip: float = float('inf'),
                 callbacks: Optional[list[Callback]] = None,
                 save_dir: str = 'models',
                 device: str = None,
                 log_level: str = 'info'
                 ):
        """
        Initialize the PPO agent.

        Args:
            env (EnvWrapper): The environment wrapper for the agent.
            policy_model: The policy model used for action selection.
            value_model: The value model used for state-value prediction.
            discount (float): Discount factor for future rewards (default: 0.99).
            gae_coefficient (float): GAE smoothing coefficient (default: 0.95).
            policy_clip (float): Clipping value for policy ratio updates (default: 0.2).
            policy_clip_schedule (ScheduleWrapper): Rate at which to decay policy clip per learn epoch (default: None).
            value_clip (float): Clipping value for value model updates (default: 0.2).
            value_clip_schedule (ScheduleWrapper): Rate at which to decay value clip per learn epoch (default: None).
            value_loss_coefficient (float): value to weight the value loss by (default: 1.0).
            entropy_coefficient (float): Coefficient for entropy regularization (default: 0.01).
            entropy_schedule (ScheduleWrapper): Rate at which to decay entropy coefficient per learn epoch (default: None).
            kl_coefficient (float): Coefficient for KL divergence penalty (default: 0.01).
            kl_adapter (AdaptiveKL): Adjusts kl_coefficient to keep KL Divergence near target (default: None).
            normalize_advantages (bool): Whether to normalize advantages (default: True).
            curiosity (ICM): ICM model for curiosity-driven learning (default: None).
            state_normalizer (Normalizer): Normalizer for state inputs (default: None).
            goal_normalizer (Normalizer): Normalizer for goal inputs (default: None).
            obs_key (str): Key for observation in the state space dict (default: 'observation').
            goal_key (str): Key for desired goal in the state space dict (default: 'desired_goal').
            achieved_goal_key (str): Key for achieved goal in the state space dict (default: 'achieved_goal').
            grad_clip (float): Maximum norm for policy gradients (default: inf).
            reward_clip (float): Maximum absolute value for reward clipping (default: inf).
            callbacks (list): List of callback objects for logging and monitoring (default: []).
            save_dir (str): Directory to save models and configurations (default: 'models').
            device (str): Device for computations ('cpu' or 'cuda', default: 'cuda').
        """
        try:
            super().__init__(env, callbacks, obs_key, goal_key, save_dir, device, log_level)
            self.policy_model = policy_model
            self.value_model = value_model
            self.discount = discount
            self.gae_coefficient = gae_coefficient
            self.policy_clip = policy_clip
            self.policy_clip_schedule = policy_clip_schedule
            self.value_clip = value_clip
            self.value_clip_schedule = value_clip_schedule
            self.value_loss_coefficient = value_loss_coefficient
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.kl_coefficient = kl_coefficient
            self.kl_adapter = kl_adapter
            self.normalize_advantages = normalize_advantages
            self.curiosity = curiosity
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.obs_key = obs_key
            self.goal_key = goal_key
            self.achieved_goal_key = achieved_goal_key
            self.grad_clip = grad_clip
            self.reward_clip = reward_clip
        except Exception as e:
            self.logger.error(f"Error in PPO.__init__: {e}", exc_info=True)
    

    def calculate_advantages_and_returns(self, rewards, states, next_states, dones, goals=None):
        """
        Compute advantages and returns using GAE, correctly handling episode terminations.
        """
        num_steps, num_envs = rewards.shape
        all_advantages = []
        all_returns = []
        all_values = []

        for env_idx in range(num_envs):
            with T.no_grad():
                rewards_env = rewards[:, env_idx]
                states_env = states[:, env_idx, ...]
                next_states_env = next_states[:, env_idx, ...]
                goals_env = goals[:, env_idx, ...] if goals is not None else None
                dones_env = dones[:, env_idx]
                values = self.value_model(states_env, goals_env).squeeze(-1)
                next_values = self.value_model(next_states_env, goals_env).squeeze(-1)
                advantages = T.zeros_like(rewards_env)
                returns = T.zeros_like(rewards_env)
                gae = 0.0

                # Calculate deltas across the trajectory
                deltas = rewards_env + self.discount * next_values * (1.0 - dones_env) - values

                for t in reversed(range(num_steps)):
                    gae = deltas[t] + self.discount * self.gae_coefficient * gae * (1.0 - dones_env[t])
                    advantages[t] = gae
                    returns[t] = gae + values[t]

                all_advantages.append(advantages)
                all_returns.append(returns)
                all_values.append(values)

        # Stack results across environments
        advantages = T.stack(all_advantages, dim=1)
        returns = T.stack(all_returns, dim=1)
        values = T.stack(all_values, dim=1)

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns, values

    def get_action(self, states:np.ndarray|T.Tensor|dict|list[dict], step=None, testing:bool=False)->tuple[np.ndarray, np.ndarray]:
        """
        Select an action based on the current policy.

        Args:
            env: EnvWrapper: The environment wrapper.
            states: np.ndarray: The current states.
            goal_normalizer: Optional[Normalizer]: The goal normalizer.
            testing: bool: if testing the action.
            rendering: bool: if rendering the action.
            obs_key: str | None: The observation key.
            goal_key: str | None: The goal key.
        
        Returns:
            Tuple[array, array]: Selected actions and their log probabilities.
        """

        states, goals = self._preprocess_inputs(states)

        if testing:
            with T.no_grad():
                if self.policy_model.distribution == 'categorical':
                    dist, logits = self.policy_model(states, goals)
                    actions = dist.probs.argmax(dim=-1)
                else:
                    dist, _, _ = self.policy_model(states, goals)
                    actions = dist.mean
                log_probs = dist.log_prob(actions)


        else:
            with T.no_grad():
                if self.policy_model.distribution == 'categorical':
                    dist, logits = self.policy_model(states, goals)
                else:
                    dist, _, _ = self.policy_model(states, goals)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
        
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def action_adapter(self, env: EnvWrapper, actions):
        """
        Adapt actions to match the environment's action space.

        Args:
            actions (array): Actions to adapt.

        Returns:
            array: Adapted actions.
        """
        if isinstance(env, GymnasiumWrapper):
            if isinstance(env.single_action_space, gym.spaces.Box):
                action_space_low = env.single_action_space.low
                action_space_high = env.single_action_space.high
                # Map action values to be between 0-1 if using beta distribution
                if self.policy_model.distribution == 'beta':
                    actions = 1/(1 + np.exp(-actions))
                # Map from [0, 1] to [action_space_low, action_space_high]
                adapted_actions = action_space_low + (action_space_high - action_space_low) * actions
                return adapted_actions
            elif isinstance(env.single_action_space, gym.spaces.Discrete):
                n = env.single_action_space.n
                # Map actions from [0, 1] to [0, n-1]
                adapted_actions = (actions * n).astype(int)
                adapted_actions = np.clip(adapted_actions, 0, n - 1)
                return adapted_actions
        elif isinstance(env, IsaacSimWrapper):
            pass
        else:
            raise NotImplementedError(f"Action adaptation not implemented for environment type: {type(self.env)}")

        raise NotImplementedError("Unsupported action space type for the current environment")
    

    # def clip_reward(self, reward):
    #     """
    #     Clip rewards to the specified range.

    #     Args:
    #         reward (float): Reward to clip.

    #     Returns:
    #         float: Clipped reward.
    #     """
    #     if reward > self.reward_clip:
    #         return self.reward_clip
    #     elif reward < -self.reward_clip:
    #         return -self.reward_clip
    #     else:
    #         return reward

    def learn(self, step:int, trajectories:dict, batch_size:int, learning_epochs:int):
        """
        Perform learning updates using the collected trajectory.

        Args:
            trajectories (dict): Collected trajectories containing states, actions, etc.
            batch_size (int): Batch size for training.
            learning_epochs (int): Number of epochs per update.

        Returns:
            Tuple: policy loss, value loss, entropy, and KL divergence.
        """
        # Unpack trajectory
        states_traj, next_states_traj, actions_traj, log_probs_traj, rewards_traj, dones_traj = trajectories.values()

        # Convert states and next states trajectories to numpy arrays
        # states_traj = np.ndarray(states_traj)
        # next_states_traj = np.ndarray(next_states_traj)

        states, goals = self._preprocess_inputs(states_traj)
        next_states, next_goals = self._preprocess_inputs(next_states_traj)
        # Convert actions to T.long values if categorical, else floats
        if self.policy_model.distribution == 'categorical':
            actions = T.stack([T.tensor(a, dtype=T.long, device=self.policy_model.device) for a in actions_traj])
        else:
            actions = T.stack([T.tensor(a, dtype=T.float32, device=self.policy_model.device) for a in actions_traj])
        log_probs = T.stack([T.tensor(lp, dtype=T.float32, device=self.policy_model.device) for lp in log_probs_traj])
        rewards = T.stack([T.tensor(r, dtype=T.float32, device=self.value_model.device) for r in rewards_traj])
        dones = T.stack([T.tensor(d, dtype=T.int, device=self.policy_model.device) for d in dones_traj])

        # Clip rewards
        if np.isfinite(self.reward_clip):
            rewards = T.clamp(rewards, min=-self.reward_clip, max=self.reward_clip)

        if self.curiosity:
            # Flatten trajectory data
            num_steps, num_envs = rewards.shape
            total_samples = num_steps * num_envs
            states_flat = states.reshape(total_samples, -1)
            next_states_flat = next_states.reshape(total_samples, -1)
            if self.policy_model.distribution == 'categorical':
                actions_flat = actions.flatten()
            else:
                actions_flat = actions.reshape(total_samples, -1)

            curiosity_loss = self.curiosity.train(states_flat, next_states_flat, actions_flat)
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(states_flat, next_states_flat, actions_flat)
            intrinsic_reward = intrinsic_reward.reshape(num_steps, num_envs)
            if step > self.curiosity.extrinsic_threshold:
                rewards += intrinsic_reward
            else:
                rewards = intrinsic_reward

        # Calculate advantages and returns (also normalizes via _preprocess_inputs func if using normalizers)
        advantages, returns, all_values = self.calculate_advantages_and_returns(rewards, states, next_states, dones, goals)

        # Flatten the tensors along the time and environment dimensions for batching
        num_steps, num_envs = rewards.shape
        total_samples = num_steps * num_envs

        # Reshape observations
        states = states.reshape(total_samples, -1)
        next_states = next_states.reshape(total_samples, -1)
        goals = goals.reshape(total_samples, -1) if goals is not None else None

        # Reshape tensors for batching
        all_values = all_values.reshape(total_samples, -1) # Shape: (total_samples, 1)
        actions = actions.reshape(total_samples, -1)     # Shape: (total_samples, action_space)
        log_probs = log_probs.reshape(total_samples, -1) # Shape: (total_samples, action_dim)
        advantages = advantages.reshape(total_samples, 1) # Shape: (total_samples, 1)
        returns = returns.reshape(total_samples, 1)      # Shape: (total_samples, 1)

        # Create random indices for shuffling
        indices = T.randperm(total_samples)
        num_batches = total_samples // batch_size

        # Create instance of policy to serve as old policy
        if isinstance(self.policy_model, StochasticDiscretePolicy):
            policy = StochasticDiscretePolicy
        else:
            policy = StochasticContinuousPolicy
        
        old_policy = policy(
            env = self.env, 
            layer_config = self.policy_model.layer_config,
            output_layer_kernel = self.policy_model.output_config,
            optimizer_params = self.policy_model.optimizer_params,
            lr_scheduler = self.policy_model.lr_scheduler,
            distribution = self.policy_model.distribution,
            device = self.policy_model.device
        )
        old_policy.load_state_dict(self.policy_model.state_dict())
        old_policy.eval()

        # Create instance of value model to serve as old value func
        old_value_model = ValueModel(
            env = self.env,
            layer_config = self.value_model.layer_config,
            output_layer_kernel = self.value_model.output_config,
            optimizer_params = self.value_model.optimizer_params,
            lr_scheduler = self.value_model.lr_scheduler,
            device = self.value_model.device
        )
        old_value_model.load_state_dict(self.value_model.state_dict())
        old_value_model.eval()

        # Get current values of policy clip and entropy/kl coefficients
        policy_clip = self.policy_clip
        if self.policy_clip_schedule:
            policy_clip *= self.policy_clip_schedule.get_factor()

        value_clip = self.value_clip
        if self.value_clip_schedule:
            value_clip *= self.value_clip_schedule.get_factor()

        entropy_coefficient = self.entropy_coefficient
        if self.entropy_schedule:
            entropy_coefficient *= self.entropy_schedule.get_factor()

        kl_coefficient = self.kl_coefficient
        if self.kl_adapter:
            kl_coefficient *= self.kl_adapter.get_beta()

        # Training loop
        for epoch in range(learning_epochs):

            for batch_num in range(num_batches):
                batch_indices = indices[batch_num * batch_size : (batch_num + 1) * batch_size]
                states_batch = states[batch_indices]
                goals_batch = goals[batch_indices] if goals is not None else None
                actions_batch = actions[batch_indices]
                log_probs_batch = log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                

                # Create new distribution
                if self.policy_model.distribution == 'categorical':
                    # New distribution
                    new_dist, logits = self.policy_model(states_batch, goals_batch)
                    new_log_probs = new_dist.log_prob(actions_batch.view(-1))
                    # Old distribution
                    old_dist, old_logits = old_policy(states_batch, goals_batch)
                    old_log_probs = old_dist.log_prob(actions_batch.view(-1))
                else: # Continuous Distributions
                    # New distribution
                    new_dist, param1, param2 = self.policy_model(states_batch, goals_batch)
                    new_log_probs = new_dist.log_prob(actions_batch).sum(dim=-1)
                    # Old distribution
                    old_dist, old_param1, old_param2 = old_policy(states_batch, goals_batch)
                    old_log_probs = old_dist.log_prob(actions_batch).sum(dim=-1)


                # Calculate the ratios of new to old probabilities of actions
                if new_log_probs.dim() == 1:
                    new_log_probs = new_log_probs.unsqueeze(-1)
                    old_log_probs = old_log_probs.unsqueeze(-1)
                    advantages_batch = advantages_batch.view(-1,1)
                prob_ratio = T.exp(new_log_probs - old_log_probs)

                # Calculate Surrogate Loss
                surr1 = prob_ratio * advantages_batch
                surr2 = T.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantages_batch
                surrogate_loss = -T.min(surr1, surr2).mean()

                # Calculate Entropy penalty
                entropy = new_dist.entropy().sum(dim=-1).mean()
                entropy_penalty = entropy * -entropy_coefficient 

                # Calculate the KL penalty
                kl = kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
                kl_penalty = kl * kl_coefficient
                
                policy_loss = surrogate_loss + entropy_penalty + kl_penalty
                
                # Update the policy
                self.policy_model.optimizer.zero_grad()
                policy_loss.backward()
                if self.grad_clip:
                    T.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.grad_clip)
                self.policy_model.optimizer.step()
                
                    
                # Update the value function
                values = self.value_model(states_batch, goals_batch)
                loss = (values - returns_batch).pow(2)
                old_values = old_value_model(states_batch, goals_batch)
                clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
                clipped_value_loss = (clipped_values - returns_batch).pow(2)
                value_loss = self.value_loss_coefficient * (0.5 * T.max(loss, clipped_value_loss).mean())
                self.value_model.optimizer.zero_grad()
                value_loss.backward()
                if self.grad_clip:
                    T.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=self.grad_clip)
                self.value_model.optimizer.step()

                
        # Step schedulers
        if self.policy_model.lr_scheduler:
            policy_learning_rate = self.policy_model.lr_scheduler.get_last_lr()[0] * self.policy_model.optimizer.param_groups[0]['lr']
            self.policy_model.lr_scheduler.step()
        else:
            policy_learning_rate = self.policy_model.optimizer.param_groups[0]['lr']
        if self.value_model.lr_scheduler:
            value_learning_rate = self.value_model.lr_scheduler.get_last_lr()[0] * self.value_model.optimizer.param_groups[0]['lr']
            self.value_model.lr_scheduler.step()
        else:
            value_learning_rate = self.value_model.optimizer.param_groups[0]['lr']
        if self.policy_clip_schedule:
            self.policy_clip_schedule.step()
        if self.value_clip_schedule:
            self.value_clip_schedule.step()
        if self.entropy_schedule:
            self.entropy_schedule.step()
        if self.kl_adapter:
            self.kl_adapter.step(kl)

        learn_metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item(),
            'log_probs': log_probs.detach().cpu().flatten().mean(),
            'advantages': advantages.detach().cpu().flatten().mean(),
            'returns': returns.detach().cpu().flatten().mean(),
            'policy_clip': policy_clip,
            'value_clip': value_clip,
            'entropy_coefficient': entropy_coefficient,
            'kl_coefficient': kl_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate
        }

        if self.policy_model.distribution == 'categorical':
            learn_metrics['logits'] = logits.detach().cpu().flatten().mean()
        else:
            learn_metrics['param1'] = param1.detach().cpu().flatten().mean()
            learn_metrics['param2'] = param2.detach().cpu().flatten().mean()
        
        return learn_metrics

    def _step(self, env: EnvWrapper, states: np.ndarray, trajectories: dict, max_episodes: int, episode_scores: np.ndarray,
              completed_episodes: np.ndarray, score_history: deque[float], best_reward: float, training: bool = True):

        actions, log_probs = self.get_action(env, states, testing=not training)
        # if self.policy_model.distribution == 'beta':
        acts = self.action_adapter(env, actions)
        # else:
        #     acts = actions
        acts = env.format_actions(acts)
        next_states, rewards, dones, infos = env.step(acts)
        # Add step data to trajectories
        trajectories['states'].append(states)
        trajectories['next_states'].append(next_states)
        trajectories['actions'].append(actions)
        trajectories['log_probs'].append(log_probs)
        trajectories['rewards'].append(rewards)
        trajectories['dones'].append(dones)

        episode_scores += rewards # Add rewards to episode scores
        step_log = {'step_reward': rewards.mean()} # Add step reward to step log

        done_episodes = np.flatnonzero(dones) # Get indices of completed episodes
        episode_logs = []
        for i in done_episodes:
            completed_episodes[i] += 1
            score_history.append(float(episode_scores[i].item()))
            avg_reward = sum(score_history) / len(score_history)

            # check if best reward
            if training and avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
            episode_log = {
                'env': i,
                'episode': int(completed_episodes.sum()),
                'episode_reward': episode_scores[i].round(2),
                'avg_reward': avg_reward.round(2)
            }

            if training:
                episode_log.update({
                    'best_reward': best_reward.round(2),
                    'best': 1 if avg_reward > best_reward else 0
                })
            episode_logs.append(episode_log)

        return {
            'trajectories': trajectories,
            'episode_scores': episode_scores,
            'completed_episodes': completed_episodes,
            'score_history': score_history,
            'step_log': step_log,
            'episode_logs': episode_logs,
            'done': completed_episodes.sum() >= max_episodes
        }


    def train(self, timesteps:int, trajectory_length:int, batch_size:int, learning_epochs:int, num_envs:int, render_freq:int=0, seed:int=None):
        """
        Train the PPO agent.
        
        Args:
            timesteps (int): Total number of timesteps to train.
            trajectory_length (int): Number of timesteps per update.
            batch_size (int): Batch size for training.
            learning_epochs (int): Number of epochs per update.
            num_envs (int): Number of parallel environments.
            seed (int, optional): Random seed for reproducibility.
            render_freq (int): Frequency of rendering episodes.
            save_dir (str, optional): directory to save the model. Defaults to self.save_dir
        """

        metrics = self._initialize_run(num_envs, seed, timesteps=timesteps, trajectory_length=trajectory_length, batch_size=batch_size, learning_epochs=learning_epochs)
        env = metrics['env']
        step = metrics['step']
        states = metrics['states']
        episode_scores = metrics['episode_scores']
        completed_episodes = metrics['completed_episodes']
        score_history = metrics['score_history']
        best_reward = metrics['best_reward']
        trajectories = {
            'states': [],
            'next_states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

        while step < timesteps:
            step += 1
            step_result = self._step(env, states, trajectories, timesteps, episode_scores, completed_episodes, score_history, best_reward)
            # Update states, episode scores, completed episodes, and score history
            trajectories = step_result['trajectories']
            states = trajectories['next_states'][-1]
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']

            if step % trajectory_length == 0:
                # Learn
                learn_metrics = self.learn(step, trajectories, batch_size, learning_epochs)
                step_result['step_log'].update({**learn_metrics})
                # Clear trajectory data
                trajectories['states'] = []
                trajectories['next_states'] = []
                trajectories['actions'] = []
                trajectories['log_probs'] = []
                trajectories['rewards'] = []
                trajectories['dones'] = []
            
            # log to callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=step, logs=step_result['step_log'])


            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0
                best_reward = episode_log['best_reward']

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during training...")
                    self.render_episode(episode_log['episode'], step, context='train', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=episode_log)

        env.close()

    def test(self, num_episodes:int, num_envs:int=1, render_freq:int=0, seed:int|None=None):
        """
        Test the PPO agent in the environment.

        Args:
            num_episodes (int): Number of episodes to test.
            num_envs (int): Number of parallel environments.
            render_freq (int): Frequency of rendering episodes.
            seed (int, optional): Random seed for reproducibility.
        Returns:
            dict: Test metrics including scores and log probabilities.
        """
        metrics = self._initialize_run(num_envs, seed, training=False)
        env = metrics['env']
        step = metrics['step']
        states = metrics['states']
        episode_scores = metrics['episode_scores']
        completed_episodes = metrics['completed_episodes']
        score_history = metrics['score_history']
        best_reward = metrics['best_reward']
        trajectories = {
            'states': [],
            'next_states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

        while completed_episodes.sum() < num_episodes:
            step += 1
            step_result = self._step(env, states, trajectories, num_episodes, episode_scores, completed_episodes, score_history, best_reward, training=False)
            # Update states, episode scores, completed episodes, and score history
            trajectories = step_result['trajectories']
            states = trajectories['next_states'][-1]
            episode_scores = step_result['episode_scores']
            completed_episodes = step_result['completed_episodes']
            score_history = step_result['score_history']
            
            # log to callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=step, logs=step_result['step_log'])


            render = True # Flag to keep track of render status to avoid rendering multiple times per step
            for episode_log in step_result['episode_logs']:
                # Print complete episode metrics to console
                print(f"Training Environment {episode_log['env']}: Episode {episode_log['episode']}, Score {episode_log['episode_reward']}, Avg_Score {episode_log['avg_reward']}")
                # Reset episode score
                episode_scores[episode_log['env']] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=step, logs=episode_log)

                # Check if number of completed episodes should trigger render
                if render and render_freq > 0 and episode_log['episode'] % render_freq == 0:
                    print(f"Rendering episode {episode_log['episode']} during testing...")
                    self.render_episode(episode_log['episode'], step, context='test', render_mode='rgb_array', seed=np.random.randint(0, 1000000))
                    render = False

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=episode_log)

        env.close()

    def get_config(self):
        """
        Get the current configuration of the PPO agent.

        Returns:
            dict: Configuration dictionary.
        """
        return {
                "agent_type": self.__class__.__name__,
                "env": self.env.to_json(),
                "policy_model": self.policy_model.get_config(),
                "value_model": self.value_model.get_config(),
                "discount": self.discount,
                "gae_coefficient": self.gae_coefficient,
                "policy_clip": self.policy_clip,
                "policy_clip_schedule": self.policy_clip_schedule.get_config() if self.policy_clip_schedule else None,
                "value_clip": self.value_clip,
                "value_clip_schedule": self.value_clip_schedule.get_config() if self.value_clip_schedule else None,
                "value_loss_coefficient": self.value_loss_coefficient,
                "entropy_coefficient": self.entropy_coefficient,
                "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule else None,
                "kl_coefficient": self.kl_coefficient,
                "kl_adapter": self.kl_adapter.get_config() if self.kl_adapter else None,
                "normalize_advantages":self.normalize_advantages,
                "curiosity": self.curiosity.get_config() if self.curiosity else None,
                "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer else None,
                "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer else None,
                "obs_key": self.obs_key,
                "goal_key": self.goal_key,
                "achieved_goal_key": self.achieved_goal_key,
                "grad_clip": self.grad_clip,
                "reward_clip": self.reward_clip,
                "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
                "save_dir": self.save_dir,
                "device": self.device.type,
            }

    def save(self, save_dir=None):
        """
        Save the model and its configuration.

        Args:
            save_dir (str, optional): Directory to save the model. Defaults to self.save_dir.
        """
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy_model.save(self.save_dir)
        self.value_model.save(self.save_dir)
        if self.curiosity:
            self.curiosity.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool=True):
        """
        Load a PPO agent from a saved configuration.

        Args:
            config (dict): Configuration dictionary.
            load_weights (bool): Whether to load model weights.

        Returns:
            PPO: Loaded PPO agent.
        """
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        distribution = config['policy_model']['distribution']
        if distribution == 'categorical':
            policy_model = StochasticDiscretePolicy.load(Path(config_dir) / 'policy_model', load_weights, env=env_wrapper)
        elif distribution in ['beta', 'normal']:
            policy_model = StochasticContinuousPolicy.load(Path(config_dir) / 'policy_model', load_weights, env=env_wrapper)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        value_model = ValueModel.load(Path(config_dir) / 'value_model', load_weights, env=env_wrapper)
        curiosity = ICM.load(config["save_dir"], env=env_wrapper) if config["curiosity"] else None
        state_normalizer = Normalizer.load(config["state_normalizer"], config["save_dir"] + "/state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = Normalizer.load(config["goal_normalizer"], config["save_dir"] + "/goal_normalizer.pt") if config["goal_normalizer"] else None
        callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None
        agent = cls(
            env_wrapper,
            policy_model = policy_model,
            value_model = value_model,
            discount=config["discount"],
            gae_coefficient = config["gae_coefficient"],
            policy_clip = config["policy_clip"],
            policy_clip_schedule = ScheduleWrapper(config["policy_clip_schedule"]) if config["policy_clip_schedule"] else None,
            value_clip = config["value_clip"],
            value_clip_schedule = ScheduleWrapper(config["value_clip_schedule"]) if config["value_clip_schedule"] else None,
            value_loss_coefficient = config["value_loss_coefficient"],
            entropy_coefficient = config["entropy_coefficient"],
            entropy_schedule = ScheduleWrapper(config["entropy_schedule"]) if config["entropy_schedule"] else None,
            kl_coefficient = config["kl_coefficient"],
            kl_adapter = AdaptiveKL(**config["kl_adapter"]) if config["kl_adapter"] else None,
            normalize_advantages = config["normalize_advantages"],
            curiosity = curiosity,
            state_normalizer = state_normalizer,
            goal_normalizer = goal_normalizer,
            obs_key = config["obs_key"],
            goal_key = config["goal_key"],
            achieved_goal_key = config["achieved_goal_key"],
            grad_clip = config["grad_clip"],
            reward_clip = config['reward_clip'],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
        )

        return agent
    

    

# class MAPPO(Agent):

#     def __init__(self,
#                  env: EnvWrapper,
#                  policy_model,
#                  value_model,
#                  distribution: str = 'beta',
#                  discount: float = 0.99,
#                  gae_coefficient: float = 0.95,
#                  policy_clip: float = 0.2,
#                  entropy_coefficient: float = 0.01,
#                  loss:str = 'clipped',
#                  kl_coefficient: float = 0.01,
#                  normalize_advantages: bool = True,
#                  normalize_values: bool = False,
#                  value_normalizer_clip: float = np.inf,
#                  policy_grad_clip:float = np.inf,
#                  reward_clip:float = np.inf,
#                  callbacks: List = [],
#                  save_dir = 'models',
#                  device = 'cuda',
#                 #  seed: float = None,
#                  ):
#         self.env = env
#         self.policy_model = policy_model
#         self.value_model = value_model
#         self.distribution = distribution
#         self.discount = discount
#         self.gae_coefficient = gae_coefficient
#         self.policy_clip = policy_clip
#         self.entropy_coefficient = entropy_coefficient
#         self.loss = loss
#         self.kl_coefficient = kl_coefficient
#         self.normalize_advantages = normalize_advantages
#         self.normalize_values = normalize_values
#         self.value_norm_clip = value_normalizer_clip
#         if self.normalize_values:
#             self.normalizer = Normalizer((1), clip_range=self.value_norm_clip, device=device)
#         self.policy_grad_clip = policy_grad_clip
#         self.reward_clip = reward_clip
#         self.callbacks = callbacks
#         self.device = device
#         # if seed is None:
#         #     seed = np.random.randint(100)
#         # self.seed = seed

#         # self.save_dir = save_dir + "/ddpg/"
#         if save_dir is not None and "/ppo/" not in save_dir:
#                 self.save_dir = save_dir + "/ppo/"
#         elif save_dir is not None and "/ppo/" in save_dir:
#                 self.save_dir = save_dir


#         # self.lambda_param = 0.5
#         if self.loss == 'hybrid':
#             # Instantiate learnable parameter to blend Clipped and KL loss objectives
#             self.lambda_param = T.nn.Parameter(T.tensor(self.lambda_))
#             # # Add lambda param to policy optimizer
#             self.policy_model.optimizer.add_param_group({'params': [self.lambda_param]})

#         # Set callbacks
#         try:
#             self.callbacks = callbacks
#             if callbacks:
#                 for callback in self.callbacks:
#                     self._config = callback._config(self)
#                     if isinstance(callback, WandbCallback):
#                         self._wandb = True

#             else:
#                 self.callback_list = None
#                 self._wandb = False
#             # if self.use_mpi:
#             #     logger.debug(f"rank {self.rank} TD3 init: callbacks set")
#             # else:
#             #     logger.debug(f"TD3 init: callbacks set")
#         except Exception as e:
#             logger.error(f"Error in TD3 init set callbacks: {e}", exc_info=True)

#         self._train_config = {}
#         self._train_episode_config = {}
#         self._train_step_config = {}
#         self._test_config = {}
#         self._test_step_config = {}
#         self._test_episode_config = {}

#         self._step = None
        
#     def calculate_advantages_and_returns(self, rewards, states, next_states, dones):
#         num_steps, num_envs = rewards.shape
#         all_advantages = []
#         all_returns = []
#         all_values = []

#         for env_idx in range(num_envs):
#             with T.no_grad():
#                 rewards_env = rewards[:, env_idx]
#                 states_env = states[:, env_idx, :]
#                 next_states_env = next_states[:, env_idx, :]
#                 dones_env = dones[:, env_idx]

#                 values = self.value_model(states_env).squeeze(-1)
#                 next_values = self.value_model(next_states_env).squeeze(-1)

#                 advantages = T.zeros_like(rewards_env)
#                 returns = T.zeros_like(rewards_env)
#                 gae = 0
#                 for t in reversed(range(len(rewards_env))):
#                     delta = rewards_env[t] + self.discount * next_values[t] * (1 - dones_env[t]) - values[t]
#                     gae = delta + self.discount * self.gae_coefficient * (1 - dones_env[t]) * gae
#                     # gae = T.tensor(gae, dtype=T.float32, device=self.value_model.device)
#                     # print(f'rewards env shape:{rewards_env.shape}')
#                     # print(f'values shape:{values.shape}')
#                     # print(f'next values shape:{next_values.shape}')
#                     # print(f'dones env shape:{dones_env.shape}')
#                     # print(f'gae shape:{gae.shape}')
#                     # print(f'advantages shape:{advantages.shape}')
#                     advantages[t] = gae
#                     returns[t] = gae + values[t]
#                     # print(f'advantages[t]:{advantages[t]}')
#                     # print(f'returns[t]:{returns[t]}')

#                 if self.normalize_advantages:
#                     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#                 all_advantages.append(advantages.unsqueeze(-1))
#                 all_returns.append(returns.unsqueeze(-1))
#                 all_values.append(values.unsqueeze(-1))

#         all_advantages = T.stack(all_advantages, dim=1)
#         all_returns = T.stack(all_returns, dim=1)
#         all_values = T.stack(all_values, dim=1)

#         self._train_episode_config["values"] = values.mean().item()
#         self._train_episode_config["advantages"] = all_advantages.mean().item()
#         self._train_episode_config["returns"] = all_returns.mean().item()

#         return all_advantages, all_returns, all_values


#     # def get_action(self, states):
#     #     # Run states through each Policy to get distribution params
#     #     actions = []
#     #     log_probs = []
#     #     # print(f'states sent to get action: {states.shape}')
#     #     for state in states:
#     #         with T.no_grad():
#     #             # make sure state is a tensor and on correct device
#     #             state = T.tensor(state, dtype=T.float32, device=self.policy_model.device).unsqueeze(0)
#     #             #DEBUG
#     #             # print(f'state shape in get_action:{state.shape}')
#     #             # print(f'get action state:{state}')
#     #             if self.distribution == 'categorical':
#     #                 dist, logits = self.policy_model(state)
#     #             else:
#     #                 dist, _, _ = self.policy_model(state)
#     #             action = dist.sample()
#     #             log_prob = dist.log_prob(action)
#     #             actions.append(action.detach().cpu().numpy().flatten())
#     #             log_probs.append(log_prob.detach().cpu().numpy().flatten())

#     #     return np.array(actions), np.array(log_probs)

#     def get_action(self, states):
#         with T.no_grad():
#             states = T.tensor(states, dtype=T.float32, device=self.policy_model.device)
#             # print(f'states shape:{states.shape}')
#             # if len(states.shape) == 4:
#             #     print('states len == 4 fired...')
#             #     states = states.permute(0, 3, 1, 2)
#             # print(f'new states shape:{states.shape}')
#             if self.distribution == 'categorical':
#                 dist, logits = self.policy_model(states)
#             else:
#                 dist, _, _ = self.policy_model(states)
#             actions = dist.sample()
#             log_probs = dist.log_prob(actions)
#             actions = actions.detach().cpu().numpy()
#             log_probs = log_probs.detach().cpu().numpy()
#         return actions, log_probs

#     def action_adapter(self, actions, env):
#         if isinstance(env.single_action_space, gym.spaces.Box):
#             action_space_low = env.single_action_space.low  # Array of lows per dimension
#             action_space_high = env.single_action_space.high  # Array of highs per dimension
#             # Ensure actions are in [0, 1]
#             actions = np.clip(actions, 0, 1)
#             # Map from [0, 1] to [action_space_low, action_space_high]
#             adapted_actions = action_space_low + (action_space_high - action_space_low) * actions
#             return adapted_actions
#         elif isinstance(env.single_action_space, gym.spaces.Discrete):
#             n = env.single_action_space.n
#             # Map actions from [0, 1] to [0, n-1]
#             adapted_actions = (actions * n).astype(int)
#             adapted_actions = np.clip(adapted_actions, 0, n - 1)
#             return adapted_actions
#         else:
#             raise NotImplementedError(f"Unsupported action space type: {type(env.single_action_space)}")
    
#     # def action_adapter(self, action):
#     #     # print(f'action adpater action:{action}')
#     #     # print(f'action adpater action shape:{action.shape}')
#     #     return 2 * (action.reshape(1,-1) -0.5 * self.env.action_space.high[0])
#     #     # print(f'action adpater a:{a}')
#     #     # print(f'action adpater a shape:{a.shape}')
#     #     # return a

#     def clip_reward(self, reward):
#         if reward > self.reward_clip:
#             return self.reward_clip
#         elif reward < -self.reward_clip:
#             return -self.reward_clip
#         else:
#             return reward

#     @classmethod
#     def sweep_train(
#         cls,
#         config, # wandb.config,
#         # train_config,
#         env_spec,
#         callbacks,
#         run_number,
#         # comm=None,
#     ):
#         """Builds and trains agents from sweep configs"""
#         # Import necessary functions directly from wandb_support
#         from wandb_support import get_wandb_config_value, get_wandb_config_optimizer_params

#         logger.debug(f"init_sweep fired")
#         try:
#             # Instantiate env from env_spec
#             env = gym.make(gym.envs.registration.EnvSpec.from_json(env_spec))

#             # logger.debug(f"train config: {train_config}")
#             logger.debug(f"env spec id: {env.spec.id}")
#             logger.debug(f"callbacks: {callbacks}")
#             logger.debug(f"run number: {run_number}")
#             logger.debug(f"config set: {config}")
#             model_type = list(config.keys())[0]
#             logger.debug(f"model type: {model_type}")

#             # Get device
#             device = get_wandb_config_value(config, model_type, 'device')

#             # Format policy and value layers, and kernels
#             policy_layers, value_layers, kernels = wandb_support.format_layers(config)
#             # logger.debug(f"layers built")

#             # Policy
#             # Learning Rate
#             policy_learning_rate_const = get_wandb_config_value(config, model_type, 'policy_learning_rate_constant')
#             policy_learning_rate_exp = get_wandb_config_value(config, model_type, 'policy_learning_rate_exponent')
#             policy_learning_rate = policy_learning_rate_const * (10 ** policy_learning_rate_exp)
#             logger.debug(f"policy learning rate set to {policy_learning_rate}")
#             # Distribution
#             distribution = get_wandb_config_value(config, model_type, 'distribution')
#             # Optimizer
#             policy_optimizer = get_wandb_config_value(config, model_type, 'policy_optimizer')
#             logger.debug(f"policy optimizer set to {policy_optimizer}")
#             # Get optimizer params
#             policy_optimizer_params = get_wandb_config_optimizer_params(config, model_type, 'policy_optimizer')
#             logger.debug(f"policy optimizer params set to {policy_optimizer_params}")
#             # Get correct policy model for env action space
#             if isinstance(env.action_space, gym.spaces.Discrete):
#                 policy_model = StochasticDiscretePolicy(
#                     env = env,
#                     dense_layers = policy_layers,
#                     output_layer_kernel = kernels[f'policy_output_kernel'],
#                     optimizer = policy_optimizer,
#                     optimizer_params = policy_optimizer_params,
#                     learning_rate = policy_learning_rate,
#                     device = device,
#                 )
#             # Check if the action space is continuous
#             elif isinstance(env.action_space, gym.spaces.Box):
#                 policy_model = StochasticContinuousPolicy(
#                     env = env,
#                     dense_layers = policy_layers,
#                     output_layer_kernel = kernels[f'policy_output_kernel'],
#                     optimizer = policy_optimizer,
#                     optimizer_params = policy_optimizer_params,
#                     learning_rate = policy_learning_rate,
#                     distribution = distribution,
#                     device = device,
#                 )
#             logger.debug(f"policy model built: {policy_model.get_config()}")

#             # Value Func
#             # Learning Rate
#             value_learning_rate_const = get_wandb_config_value(config, model_type, "value_learning_rate_constant")
#             value_learning_rate_exp = get_wandb_config_value(config, model_type, "value_learning_rate_exponent")
#             critic_learning_rate = value_learning_rate_const * (10 ** value_learning_rate_exp)
#             logger.debug(f"value learning rate set to {critic_learning_rate}")
#             # Optimizer
#             value_optimizer = get_wandb_config_value(config, model_type, 'value_optimizer')
#             logger.debug(f"value optimizer set to {value_optimizer}")
#             value_optimizer_params = get_wandb_config_optimizer_params(config, model_type, 'value_optimizer')
#             logger.debug(f"value optimizer params set to {value_optimizer_params}")

#             # Check if CNN layers and if so, build CNN model
#             # if actor_cnn_layers:
#             #     actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
#             # else:
#             #     actor_cnn_model = None
#             # if comm is not None:
#             #     logger.debug(f"{comm.Get_name()}; Rank {rank} actor cnn layers set: {actor_cnn_layers}")
#             # else:
#             #     logger.debug(f"actor cnn layers set: {actor_cnn_layers}")

#             # if critic_cnn_layers:
#             #     critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
#             # else:
#             #     critic_cnn_model = None
#             # if comm is not None:
#             #     logger.debug(f"{comm.Get_name()}; Rank {rank} critic cnn layers set: {critic_cnn_layers}")
#             # else:
#             #     logger.debug(f"critic cnn layers set: {critic_cnn_layers}")
#             value_model = ValueModel(
#                 env = env,
#                 dense_layers = value_layers,
#                 output_layer_kernel=kernels[f'value_output_kernel'],
#                 optimizer = value_optimizer,
#                 optimizer_params = value_optimizer_params,
#                 learning_rate = critic_learning_rate,
#                 device=device,
#             )
#             logger.debug(f"value model built: {value_model.get_config()}")

#             # GAE coefficient
#             gae_coeff = get_wandb_config_value(config, model_type, 'advantage')
#             logger.debug(f"gae coeff set to {gae_coeff}")
#             # Policy clip
#             policy_clip = get_wandb_config_value(config, model_type, 'policy_clip')
#             logger.debug(f"policy clip set to {policy_clip}")
#             # Entropy coefficient
#             entropy_coeff = get_wandb_config_value(config, model_type, 'entropy')
#             logger.debug(f"entropy coeff set to {entropy_coeff}")
#             # Normalize advantages
#             normalize_advantages = get_wandb_config_value(config, model_type, 'normalize_advantage')
#             logger.debug(f"normalize advantage set to {normalize_advantages}")
#             # Normalize values
#             normalize_values = get_wandb_config_value(config, model_type, 'normalize_values')
#             logger.debug(f"normalize values set to {normalize_values}")
#             # Normalize values clip value
#             normalize_val_clip = get_wandb_config_value(config, model_type, 'normalize_values_clip')
#             if normalize_val_clip == 'infinity':
#                 normalize_val_clip = np.inf
#             logger.debug(f"normalize values clip set to {normalize_val_clip}")
#             # Policy gradient clip
#             policy_grad_clip = get_wandb_config_value(config, model_type, 'policy_grad_clip')
#             # Change value of policy_grad_clip to np.inf if == 'infinity'
#             if policy_grad_clip == "infinity":
#                 policy_grad_clip = np.inf
#             logger.debug(f"policy grad clip set to {policy_grad_clip}")

#             # Save dir
#             save_dir = get_wandb_config_value(config, model_type, 'policy_grad_clip')
#             logger.debug(f"save dir set: {save_dir}")


#             # create PPO agent
#             ppo_agent= cls(
#                 env = env,
#                 policy_model = policy_model,
#                 value_model = value_model,
#                 distribution = distribution,
#                 discount = config[model_type][f"{model_type}_discount"],
#                 gae_coefficient = gae_coeff,
#                 policy_clip = policy_clip,
#                 entropy_coefficient = entropy_coeff,
#                 normalize_advantages = normalize_advantages,
#                 normalize_values = normalize_values,
#                 value_normalizer_clip = normalize_val_clip,
#                 policy_grad_clip = policy_grad_clip,
#                 callbacks = callbacks,
#                 device = device,
#             )
#             logger.debug(f"PPO agent built: {ppo_agent.get_config()}")

#             timesteps = get_wandb_config_value(config, model_type, 'num_timesteps')
#             traj_length = get_wandb_config_value(config, model_type, 'trajectory_length')
#             batch_size = get_wandb_config_value(config, model_type, 'batch_size')
#             learning_epochs = get_wandb_config_value(config, model_type, 'learning_epochs')
#             num_envs = get_wandb_config_value(config, model_type, 'num_envs')
#             seed = get_wandb_config_value(config, model_type, 'seed')

#             ppo_agent.train(
#                 timesteps = timesteps,
#                 trajectory_length = traj_length,
#                 batch_size = batch_size,
#                 learning_epochs = learning_epochs,
#                 num_envs = num_envs,
#                 seed = seed,
#                 render_freq = 0,
#             )

#         except Exception as e:
#             logger.error(f"An error occurred: {e}", exc_info=True)

#     def train(self, timesteps, trajectory_length, batch_size, learning_epochs, num_envs, seed=None, avg_num=10, render_freq:int=0, save_dir:str=None, run_number:int=None):
#         """
#         Trains the model for 'timesteps' number of 'timesteps',
#         updating the model every 'trajectory_length' number of timesteps.

#         Args:
#             timesteps: Number of timesteps to train for.
#             trajectory_length: Number of timesteps between updates.
#             batch_size: Number of samples in a batch.
#             learning_epochs: Number of epochs to train for.
#             num_envs: Number of environments.
#             avg_num: Number of episodes to average over.
#         """

#         # Update save_dir if passed
#         if save_dir is not None and save_dir.split("/")[-2] != "ppo":
#             self.save_dir = save_dir + "/ppo/"
#             print(f'new save dir: {self.save_dir}')
#         elif save_dir is not None and save_dir.split("/")[-2] == "ppo":
#             self.save_dir = save_dir
#             print(f'new save dir: {self.save_dir}')


#         if seed is None:
#             seed = np.random.randint(100)

#         # Set render freq to 0 if None is passed
#         if render_freq == None:
#             render_freq = 0

#         # Set seeds
#         T.manual_seed(seed)
#         T.cuda.manual_seed(seed)
#         np.random.seed(seed)
#         # gym.utils.seeding.np_random.seed = seed # Seeds of envs now set in _initialize_env

#         if self.callbacks:
#             for callback in self.callbacks:
#                 self._config = callback._config(self)
#                 if isinstance(callback, WandbCallback):
#                     self._config['timesteps'] = timesteps
#                     self._config['trajectory_length'] = trajectory_length
#                     self._config['batch_size'] = batch_size
#                     self._config['learning_epochs'] = learning_epochs
#                     self._config['seed'] = seed # Add seed to config to send to wandb for logging
#                     self._config['num_envs'] = num_envs
#                     callback.on_train_begin((self.value_model, self.policy_model,), logs=self._config)
#                     # logger.debug(f'TD3.train on train begin callback complete')
#                 else:
#                     callback.on_train_begin(logs=self._config)

#         try:
#             # instantiate new vec environment
#             env = self._initialize_env(0, num_envs, seed)
#             # for e in env.envs:
#             #     print(e.spec)
#             # logger.debug(f'initiating environment with render {render}')
#         except Exception as e:
#             logger.error(f"Error in PPO.train agent._initialize_env process: {e}", exc_info=True)

#         # set best reward
#         try:
#             best_reward = self.env.reward_range
#         except:
#             best_reward = -np.inf

#         self.trajectory_length = trajectory_length
#         self.num_envs = num_envs
#         self.policy_model.train()
#         self.value_model.train()
#         # timestep = 0
#         self._step = 0
#         all_states = []
#         all_actions = []
#         all_log_probs = []
#         all_rewards = []
#         all_next_states = []
#         all_dones = []
#         # score_history = []
#         episode_scores = [[] for _ in range(num_envs)]  # Track scores for each env
#         # episode_scores = []  # Track scores for each env
#         policy_loss_history = []
#         value_loss_history = []
#         entropy_history = []
#         kl_history = []
#         time_history = []
#         lambda_values = []
#         param_history = []
#         frames = []  # List to store frames for the video
#         self.episodes = np.zeros(self.num_envs) # Tracks current episode for each env
#         episode_lengths = np.zeros(self.num_envs) # Tracks step count for each env
#         scores = np.zeros(self.num_envs) # Tracks current score for each env
#         states, _ = env.reset()

#         # set an episode rendered flag to track if an episode has yet to be rendered
#         episode_rendered = False
#         # track the previous episode number of the first env for rendering
#         prev_episode = self.episodes[0]

#         while self._step < timesteps:
#             self._step += 1 # Increment step count by 1
#             episode_lengths += 1 # increment the step count of each episode of each env by 1
#             dones = []
#             actions, log_probs = self.get_action(states)
#             # print(f'actions:{actions}')
#             if self.distribution == 'beta':
#                 acts = self.action_adapter(actions, env)
#             else:
#                 acts = actions
#             # acts = [self.action_adapter(action) if self.distribution == 'beta' else action for action in actions]
#             # acts = np.reshape(acts, env.action_space.shape)
#             acts = acts.astype(np.float32)
#             acts = np.clip(acts, env.single_action_space.low, env.single_action_space.high)
#             # print(f'acts reshape:{acts.shape}')
#             # print(f'acts:{acts}')
#             acts = acts.tolist()
#             acts = [[float(a) for a in act] for act in acts]
#             # print(f'actions after adapter:{acts}')

#             #DEBUG
#             # print(f'reshaped acts shape:{acts.shape}')

#             # if self.distribution == 'Beta':
#             #     acts = []
#             #     for action in actions:
#             #         print(f'action:{action}')
#             #         print(f'action shape:{action.shape}')
#             #         act = [self.action_adapter(a) for a in action]
#             #         print(f'act:{act}')
#             #         print(f'act shape:{np.array(act).shape}')
#             #         acts.append(act)
#             # else:
#             #     acts = actions

#             #DEBUG
#             # for e in env.envs:
#             #     print(f'continuous:{e.spec}')
#             next_states, rewards, terms, truncs, _ = env.step(acts)
#             #DEBUG
#             # print(f'terms:{terms}, truncs:{truncs}')
#             # Update scores of each episode
#             scores += rewards
#             # print(f'rewards:{rewards.mean()}')
#             self._train_step_config["step_reward"] = rewards.mean()

#             for i, (term, trunc) in enumerate(zip(terms, truncs)):
#                 if term or trunc:
#                     dones.append(True)
#                     # print(f'append true')
#                     episode_scores[i].append(scores[i])  # Store score at end of episode
#                     self._train_step_config["episode_reward"] = scores[i]
#                     scores[i] = 0  # Reset score for this environment
#                     self._train_step_config["episode_length"] = episode_lengths[i]
#                     episode_lengths[i]  = 0 # Resets the step count of the env that returned term/trunc to 0
#                 else:
#                     dones.append(False)
#                     # print(f'append false')

#             # Add frame of first env to frames array if rendering
#             # if render_freq > 0:
#             #     # Capture the frame
#             #     frame = self.env.render()[0]
#             #     # print(f'frame:{frame}')
#             #     frames.append(frame)


#             self.episodes += dones
#             # set episode rendered to false if episode number has changed
#             if prev_episode != self.episodes[0]:
#                 episode_rendered = False
#             # print(f'dones:{dones}')
#             # print(f'episodes:{episodes}')
#             self._train_episode_config['episode'] = self.episodes[0]
#             all_states.append(states)
#             all_actions.append(actions)
#             all_log_probs.append(log_probs)
#             clipped_rewards = [self.clip_reward(reward) for reward in rewards]
#             all_rewards.append(clipped_rewards)
#             all_next_states.append(next_states)
#             all_dones.append(dones)

#             # render episode if first env shows done and first env episode num % render_freq == 0
#             if render_freq > 0 and self.episodes[0] % render_freq == 0 and episode_rendered == False:
#                 print(f"Rendering episode {self.episodes[0]} during training...")
#                 # Call the test function to render an episode
#                 self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
#                 # Add render to wandb log
#                 video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.episodes[0]}.mp4")
#                 # Log the video to wandb
#                 if self.callbacks:
#                     for callback in self.callbacks:
#                         if isinstance(callback, WandbCallback):
#                             wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")})
#                 episode_rendered = True
#                 # Switch models back to train mode after rendering
#                 self.policy_model.train()
#                 self.value_model.train()

#             prev_episode = self.episodes[0]

#             env_scores = np.array([
#                 env_score[-1] if len(env_score) > 0 else np.nan
#                 for env_score in episode_scores
#             ])

#             if self._step % self.trajectory_length == 0:
#                 print(f'learning timestep: {self._step}')
#                 trajectory = (all_states, all_actions, all_log_probs, all_rewards, all_next_states, all_dones)
#                 if self.distribution == 'categorical':
#                     policy_loss, value_loss, entropy, kl, logits = self.learn(trajectory, batch_size, learning_epochs)
#                 else:
#                     policy_loss, value_loss, entropy, kl, param1, param2 = self.learn(trajectory, batch_size, learning_epochs)
#                 self._train_episode_config[f"avg_env_scores"] = np.nanmean(env_scores)
#                 self._train_episode_config["actor_loss"] = policy_loss
#                 self._train_episode_config["critic_loss"] = value_loss
#                 self._train_episode_config["entropy"] = entropy
#                 self._train_episode_config["kl_divergence"] = kl
#                 # self._train_episode_config["lambda"] = lambda_value
#                 if self.distribution == 'categorical':
#                     self._train_episode_config["logits"] = logits.mean()
#                 else:
#                     self._train_episode_config["param1"] = param1.mean()
#                     self._train_episode_config["param2"] = param2.mean()

#                 # check if best reward
#                 avg_score = np.mean([
#                     np.mean(env_score[-avg_num:]) if len(env_score) >= avg_num else np.mean(env_score)
#                     for env_score in episode_scores
#                 ])
#                 if avg_score > best_reward:
#                     best_reward = avg_score
#                     self._train_episode_config["best"] = True
#                     # save model
#                     self.save()
#                 else:
#                     self._train_episode_config["best"] = False

#                 policy_loss_history.append(policy_loss)
#                 value_loss_history.append(value_loss)
#                 entropy_history.append(entropy)
#                 kl_history.append(kl)
#                 # time_history.append(time)
#                 # lambda_values.append(lambda_value)
#                 if self.distribution == 'categorical':
#                     param_history.append(logits)
#                 else:
#                     param_history.append((param1, param2))
#                 all_states = []
#                 all_actions = []
#                 all_log_probs = []
#                 all_rewards = []
#                 all_next_states = []
#                 all_dones = []

#                 if self.callbacks:
#                     for callback in self.callbacks:
#                         callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

#             states = next_states

#             if self._step % 1000 == 0:
#                 print(f'episode: {self.episodes}; total steps: {self._step}; episodes scores: {env_scores}; avg score: {np.nanmean(env_scores)}')

#             if self.callbacks:
#                 for callback in self.callbacks:
#                     callback.on_train_step_end(step=self._step, logs=self._train_step_config)

#         if self.callbacks:
#             for callback in self.callbacks:
#                 callback.on_train_end(logs=self._train_episode_config)

#         return {
#                 'scores': episode_scores,  # Changed to episode_scores
#                 'policy loss': policy_loss_history,
#                 'value loss': value_loss_history,
#                 'entropy': entropy_history,
#                 'kl': kl_history,
#                 # 'time': time_history,
#                 'lambda': lambda_values,
#                 'params': param_history,
#                 }

#     # def learn(self, trajectory, batch_size, learning_epochs):
#     #     # Unpack trajectory
#     #     all_states, all_actions, all_log_probs, all_rewards, all_next_states, all_dones = trajectory
#     #     # Flatten the lists of numpy arrays across the num_envs dimension
#     #     states = np.concatenate(all_states, axis=0)
#     #     actions = np.concatenate(all_actions, axis=0)
#     #     log_probs = np.concatenate(all_log_probs, axis=0)
#     #     rewards = np.concatenate(all_rewards, axis=0)
#     #     next_states = np.concatenate(all_next_states, axis=0)
#     #     dones = np.concatenate(all_dones, axis=0)

#     #     # Convert to Tensors
#     #     states = T.tensor(states, dtype=T.float32, device=self.policy_model.device)
#     #     actions = T.tensor(actions, dtype=T.float32, device=self.policy_model.device)
#     #     log_probs = T.tensor(log_probs, dtype=T.float32, device=self.policy_model.device)
#     #     rewards = T.tensor(rewards, dtype=T.float32, device=self.value_model.device).unsqueeze(1)
#     #     next_states = T.tensor(next_states, dtype=T.float32, device=self.policy_model.device)
#     #     dones = T.tensor(dones, dtype=T.int, device=self.policy_model.device)


#     #     # Calculate advantages and returns
#     #     advantages, returns = self.calculate_advantages_and_returns(rewards, states, next_states, dones)

#     #     # advantages = T.tensor(advantages, dtype=T.float32, device=self.policy.device)
#     #     advantages = T.cat(advantages, dim=0)
#     #     advantages = advantages.to(self.policy_model.device, dtype=T.float32)
#     #     returns = T.cat(returns, dim=0)
#     #     returns = returns.to(self.policy_model.device, dtype=T.float32)
#     #     # returns = T.tensor(returns, dtype=T.float32, device=self.value_function.device)
#     #     # advantages = advantages.reshape(-1, 1)
#     #     # returns = returns.reshape(-1, 1)
#     #     # print(f'advantages shape:{advantages.shape}')
#     #     # print(f'returns shape:{returns.shape}')
#     #     # kl_div_loss_fn = T.nn.KLDivLoss(reduction="batchmean", log_target=True)

#     #     # Set previous distribution to none (used for KL divergence calculation)
#     #     prev_dist = None

#     #     num_batches = len(states) // batch_size
#     #     print(f'num batches:{num_batches}')

#     #     # Loop over learning_epochs epochs to train the policy and value functions
#     #     for epoch in range(learning_epochs):
#     #         times = []
#     #         start_time = time.time()
#     #         # Sample mini batch from trajectory
#     #         indices = T.randperm(len(states))
#     #         batches = [indices[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
#     #         for batch in batches:
#     #             states_batch = states[batch]
#     #             actions_batch = actions[batch]
#     #             log_probs_batch = log_probs[batch]
#     #             rewards_batch = rewards[batch]
#     #             next_states_batch = next_states[batch]
#     #             dones_batch = dones[batch]
#     #             advantages_batch = advantages[batch]
#     #             returns_batch = returns[batch]

#     def learn(self, trajectory, batch_size, learning_epochs):
#         # Unpack trajectory
#         all_states, all_actions, all_log_probs, all_rewards, all_next_states, all_dones = trajectory

#         # Convert lists to tensors without flattening
#         # This results in tensors of shape (num_steps, num_envs, ...)
#         states = T.stack([T.tensor(s, dtype=T.float32, device=self.policy_model.device) for s in all_states])
#         actions = T.stack([T.tensor(a, dtype=T.float32, device=self.policy_model.device) for a in all_actions])
#         log_probs = T.stack([T.tensor(lp, dtype=T.float32, device=self.policy_model.device) for lp in all_log_probs])
#         rewards = T.stack([T.tensor(r, dtype=T.float32, device=self.value_model.device) for r in all_rewards])
#         next_states = T.stack([T.tensor(ns, dtype=T.float32, device=self.policy_model.device) for ns in all_next_states])
#         dones = T.stack([T.tensor(d, dtype=T.int, device=self.policy_model.device) for d in all_dones])

#         # DEBUG
#         # print(f'states shape:{states.shape}')
#         # print(f'actions shape:{actions.shape}')
#         # print(f'log_probs shape:{log_probs.shape}')
#         # print(f'rewards shape:{rewards.shape}')
#         # print(f'next_states shape:{next_states.shape}')
#         # print(f'dones shape:{dones.shape}')

#         # Now, states.shape = (num_steps, num_envs, observation_space)
#         # Similarly for other variables

#         # Calculate advantages and returns
#         advantages, returns, all_values = self.calculate_advantages_and_returns(rewards, states, next_states, dones)
#         #DEBUG
#         # print(f'advantages shape:{advantages.shape}')
#         # print(f'returns shape:{returns.shape}')

#         # Proceed with the rest of the learning process
#         # Flatten the tensors along the time and environment dimensions for batching
#         num_steps, num_envs = rewards.shape
#         total_samples = num_steps * num_envs

#         # Reshape observations
#         obs_shape = states.shape[2:]  # Get observation shape
#         states = states.reshape(total_samples, *obs_shape)
#         next_states = next_states.reshape(total_samples, *obs_shape)

#         # Reshape tensors for batching
#         all_values = all_values.reshape(total_samples, -1) # Shape: (total_samples, 1)
#         # states = states.reshape(total_samples, -1)       # Shape: (total_samples, observation_space)
#         actions = actions.reshape(total_samples, -1)     # Shape: (total_samples, action_space)
#         log_probs = log_probs.reshape(total_samples, -1) # Shape: (total_samples, action_dim)
#         advantages = advantages.reshape(total_samples, 1) # Shape: (total_samples, 1)
#         returns = returns.reshape(total_samples, 1)      # Shape: (total_samples, 1)
#         #DEBUG
#         # print(f'flatenned states shape:{states.shape}')
#         # print(f'flatenned actions shape:{actions.shape}')
#         # print(f'flatenned log_probs shape:{log_probs.shape}')
#         # print(f'flatenned advantages shape:{advantages.shape}')
#         # print(f'flatenned returns shape:{returns.shape}')

#         # Set previous distribution to none (used for KL divergence calculation)
#         prev_dist = None

#         # Create random indices for shuffling
#         indices = T.randperm(total_samples)
#         num_batches = total_samples // batch_size

#         # Training loop
#         for epoch in range(learning_epochs):
#             for batch_num in range(num_batches):
#                 batch_indices = indices[batch_num * batch_size : (batch_num + 1) * batch_size]
#                 states_batch = states[batch_indices]
#                 actions_batch = actions[batch_indices]
#                 log_probs_batch = log_probs[batch_indices]
#                 advantages_batch = advantages[batch_indices]
#                 returns_batch = returns[batch_indices]
#                 #DEBUG
#                 # print(f'states batch shape:{states_batch.shape}')
#                 # print(f'actions batch shape:{actions_batch.shape}')
#                 # print(f'log_probs batch shape:{log_probs_batch.shape}')
#                 # print(f'advantages batch shape:{advantages_batch.shape}')
#                 # print(f'returns batch shape:{returns_batch.shape}')

#                 # Calculate the policy loss

#                 if self.distribution == 'categorical':
#                     dist, logits = self.policy_model(states_batch)
#                 else:
#                     dist, param1, param2 = self.policy_model(states_batch)
#                 # print(f'dist mean:{dist.loc}')
#                 # print(f'dist var:{dist.scale}')
#                 # print(f'param 1:{param1}')
#                 # print(f'param 2:{param2}')
#                 # dist_time = time.time()
#                 # Create prev_dist by recreating the distribution from the previous step's parameters
#                 if prev_dist is None:
#                     prev_dist = dist

#                 else:
#                     # Recreate prev_dist by passing in the previous parameters
#                     if self.distribution == 'beta':
#                         param1_prev = prev_dist.concentration1.clone().detach()
#                         param2_prev = prev_dist.concentration0.clone().detach()
#                         prev_dist = Beta(param1_prev, param2_prev)
#                     elif self.distribution == 'normal':
#                         param1_prev = prev_dist.loc.clone().detach()
#                         param2_prev = prev_dist.scale.clone().detach()
#                         prev_dist = Normal(param1_prev, param2_prev)
#                     elif self.distribution == 'categorical':
#                         param_prev = prev_dist.logits.clone().detach()
#                         prev_dist = Categorical(logits=param_prev)
#                     else:
#                         raise ValueError(f'Unknown distribution: {self.distribution}')
#                 # dist_delta = time.time() - dist_time
#                 # print(f'dist_delta: {dist_delta}')

#                 # Calculate new log probabilities of actions
#                 new_log_probs = dist.log_prob(actions_batch)
#                 # print(f'new_log_probs shape:{new_log_probs.shape}')
#                 # print(f'new_log_probs:{new_log_probs}')
#                 # print(f'new_log_probs shape:{new_log_probs.sum(axis=-1, keepdim=True).shape}')
#                 # print(f'log_probs shape:{log_probs_batch.sum(axis=-1, keepdim=True).shape}')

#                 # Calculate the ratios of new to old probabilities of actions
#                 prob_ratio = T.exp(new_log_probs.sum(axis=-1, keepdim=True) - log_probs_batch.sum(axis=-1, keepdim=True))
#                 # print(f'prob ratio shape:{prob_ratio.shape}')
#                 # print(f'prob ratio:{prob_ratio}')
#                 # Calculate the surrogate loss
#                 # print(f'advantages batch:{advantages_batch}')

#                 # Calculate the entropy of the distribution
#                 entropy = dist.entropy().sum(axis=-1, keepdims=True).mean()
#                 # print(f'full entropy:{dist.entropy()}')

#                 # Calculate the KL Divergence
#                 kl = kl_divergence(prev_dist, dist).sum(dim=-1, keepdim=True).mean()

#                 surr1 = (prob_ratio * advantages_batch)
#                 # print(f'surr1 shape:{surr1.shape}')
#                 surr2 = (T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages_batch)
#                 # Clipped policy loss
#                 surrogate_loss = -T.min(surr1, surr2).mean()
#                 entropy_penalty = -self.entropy_coefficient * entropy
#                 log_diff = new_log_probs - log_probs_batch
#                 kl_penalty = -log_diff.mean()
#                 kl_penalty *= self.kl_coefficient
#                 policy_loss = surrogate_loss + entropy_penalty + kl_penalty

#                 # if self.loss == 'clipped':
#                 #     lambda_value = 1.0
#                 #     entropy_penalty = -self.entropy_coefficient * entropy
#                 #     policy_loss = surrogate_loss + entropy_penalty
#                 # elif self.loss == 'kl':
#                 #     lambda_value = 0.0
#                 #     log_diff = new_log_probs - log_probs_batch
#                 #     kl_penalty = -log_diff.mean()
#                 #     kl_penalty *= self.kl_coefficient
#                 #     policy_loss = surrogate_loss + kl_penalty
#                 # elif self.loss == 'hybrid':
#                 #     # Run lambda param through sigmoid to clamp between 0 and 1
#                 #     lambda_value = T.sigmoid(self.lambda_param)
#                 #     entropy_penalty = -self.entropy_coefficient * entropy
#                 #     log_diff = new_log_probs - log_probs_batch
#                 #     kl_penalty = -log_diff.mean()
#                 #     kl_penalty *= self.kl_coefficient
#                 #     policy_loss = surrogate_loss + entropy_penalty + kl_penalty
#                 # else:
#                 #     raise ValueError(f'Unknown loss: {self.loss}')

#                 # Update the policy
#                 self.policy_model.optimizer.zero_grad()
#                 policy_loss.backward()
#                 # if self.policy_grad_clip is not None:
#                 T.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.policy_grad_clip)
#                 self.policy_model.optimizer.step()

#                 # Update the value function
#                 # value_loss = F.mse_loss(self.value_function(states_batch), returns_batch)
#                 values = self.value_model(states_batch)
#                 value_loss = (values - returns_batch).pow(2).mean()
#                 self.value_model.optimizer.zero_grad()
#                 value_loss.backward()
#                 self.value_model.optimizer.step()
#                 # epoch_time = time.time() - start_time
#                 # times.append((epoch_time, dist_delta))

#                 # set dist as previous dist
#                 prev_dist = dist

#         # if self.callbacks:
#         #     for callback in self.callbacks:
#         #         if isinstance(callback, WandbCallback):
#         #             # Reduce states to 3D embeddings
#         #             reducer = UMAP(n_components=3, random_state=42)
#         #             embeddings = reducer.fit_transform(states.cpu().numpy())  # Shape: (num_samples, 3)
#         #             # Compute the magnitude of the actions
#         #             action_magnitude = np.linalg.norm(actions.cpu().numpy(), axis=1)
#         #             df = pd.DataFrame({
#         #                 'embedding_x': embeddings[:, 0],
#         #                 'embedding_y': embeddings[:, 1],
#         #                 'embedding_z': embeddings[:, 2],
#         #                 'value': all_values.cpu().numpy().flatten(),
#         #                 'action_magnitude': action_magnitude,
#         #                 # If you want to include specific action components:
#         #                 # 'action_component_0': actions[:, 0],
#         #                 # 'action_component_1': actions[:, 1],
#         #                 # ...
#         #             })

#         #             # Create a 3D scatter plot colored by value estimates
#         #             fig_value = px.scatter_3d(
#         #                 df,
#         #                 x='embedding_x',
#         #                 y='embedding_y',
#         #                 z='embedding_z',
#         #                 color='value',
#         #                 title='State Embeddings Colored by Value Function',
#         #                 labels={'embedding_x': 'Embedding X', 'embedding_y': 'Embedding Y', 'embedding_z': 'Embedding Z', 'value': 'Value Estimate'},
#         #                 opacity=0.7
#         #             )
                    
#         #             # Create a 3D scatter plot colored by action magnitude
#         #             fig_action = px.scatter_3d(
#         #                 df,
#         #                 x='embedding_x',
#         #                 y='embedding_y',
#         #                 z='embedding_z',
#         #                 color='action_magnitude',
#         #                 title='State Embeddings Colored by Action Magnitude',
#         #                 labels={'embedding_x': 'Embedding X', 'embedding_y': 'Embedding Y', 'embedding_z': 'Embedding Z', 'action_magnitude': 'Action Magnitude'},
#         #                 opacity=0.7
#         #             )

#         #             # Log the 3D plots
#         #             wandb.log({
#         #                 "Value Function Embeddings 3D": fig_value,
#         #                 "Policy Embeddings 3D": fig_action
#         #             })

#         print(f'Policy Loss: {policy_loss.sum()}')
#         print(f'Value Loss: {value_loss}')
#         print(f'Entropy: {entropy.mean()}')
#         print(f'KL Divergence: {kl.mean()}')
#         # print(f'kl div:{kl_div.mean()}')
#         # if self.loss == 'hybrid':
#         #     print(f'Lambda: {lambda_value}')

#         if self.distribution == 'categorical':
#             return policy_loss, value_loss, entropy.mean(), kl.mean(), logits.detach().cpu().flatten()
#         else:
#             return policy_loss, value_loss, entropy.mean(), kl.mean(), param1.detach().cpu().flatten(), param2.detach().cpu().flatten()

#     def test(self, num_episodes, num_envs:int=1, seed=None, render_freq:int=0, training=False):
#         """
#         Tests the PPO agent in the environment for a specified number of episodes,
#         renders each episode, and saves the renders as video files.

#         Args:
#             num_episodes (int): Number of episodes to test the agent.
#             render_dir (str): Directory to save the rendered video files.

#         Returns:
#             dict: A dictionary containing the scores, entropy, and KL divergence for each episode.
#         """

#         # Set the policy and value function models to evaluation mode
#         self.policy_model.eval()
#         self.value_model.eval()

#         if seed is None:
#             seed = np.random.randint(100)

#         # Set render freq to 0 if None is passed
#         if render_freq == None:
#             render_freq = 0


#         print(f'seed value:{seed}')
#         # Set seeds
#         T.manual_seed(seed)
#         T.cuda.manual_seed(seed)
#         np.random.seed(seed)
#         gym.utils.seeding.np_random.seed = seed

#         # Create the render directory if it doesn't exist
#         # if not os.path.exists(save_dir):
#         #     os.makedirs(save_dir)

#         # if not training:
#         # self.env = self._initialize_env(render_freq)
#         env = self._initialize_env(render_freq, num_envs)
#         if self.callbacks and not training:
#             print('test begin callback if statement fired')
#             for callback in self.callbacks:
#                 self._config = callback._config(self)
#                 if isinstance(callback, WandbCallback):
#                     # Add to config to send to wandb for logging
#                     self._config['seed'] = seed
#                     self._config['num_envs'] = num_envs
#                 callback.on_test_begin(logs=self._config)

#         # episode_scores = [[] for _ in range(num_envs)]  # Track scores for each env
#         # reset step counter
#         step = 0
#         all_scores = []
#         all_log_probs = []

#         for episode in range(num_episodes):
#             if self.callbacks and not training:
#                 for callback in self.callbacks:
#                     callback.on_test_epoch_begin(epoch=step, logs=None)
#             done = False
#             states, _ = env.reset()
#             scores = 0
#             log_probs = []
#             frames = []  # List to store frames for the video

#             while not done:

#                 # Get action and log probability from the current policy
#                 actions, log_prob = self.get_action(states)
#                 # acts = [self.action_adapter(action, env) if self.distribution == 'beta' else action for action in actions]
#                 # acts = np.reshape(acts, env.action_space.shape)
#                 if self.distribution == 'beta':
#                     acts = self.action_adapter(actions, env)
#                 else:
#                     acts = actions
#                 acts = acts.astype(np.float32)
#                 acts = np.clip(acts, env.single_action_space.low, env.single_action_space.high)
#                 acts = acts.tolist()
#                 acts = [[float(a) for a in act] for act in acts]

#                 #  log prob to log probs list
#                 log_probs.append(log_prob)

#                 # Step the environment
#                 next_states, rewards, terms, truncs, _ = env.step(acts)
#                 # Update scores of each episode
#                 scores += rewards

#                 for i, (term, trunc) in enumerate(zip(terms, truncs)):
#                     if term or trunc:
#                         done = True
#                         # print(f'append true')
#                     # else:
#                     #     dones.append(False)

#                 if render_freq > 0:
#                     # Capture the frame
#                     frame = env.render()[0]
#                     # print(f'frame:{frame}')
#                     frames.append(frame)

#                 # Increment step count
#                 step += 1

#                 # Move to the next state
#                 states = next_states

#                 # Add metrics to test step config to log
#                 self._test_step_config['step_reward'] = rewards[0]
#                 if self.callbacks and not training:
#                     for callback in self.callbacks:
#                         callback.on_test_step_end(step=step, logs=self._test_step_config)

#             # Save the video if the episode number is divisible by render_freq
#             if (render_freq > 0) and ((episode + 1) % render_freq == 0):
#                 if training:
#                     print(f'episode number sent to renderer:{self.episodes[0]}')
#                     self.render(frames, self.episodes[0], 'train')
#                 else:
#                     self.render(frames, episode+1, 'test')

#             # Append the results for the episode
#             all_scores.append(scores)  # Store score at end of episode
#             self._test_episode_config["episode_reward"] = scores[0]

#             # Append log probs for the episode to all_log_probs list
#             all_log_probs.append(log_probs)

#             # Log to callbacks
#             if self.callbacks and not training:
#                 for callback in self.callbacks:
#                     callback.on_test_epoch_end(epoch=step, logs=self._test_episode_config)

#             print(f'Episode {episode+1}/{num_episodes} - Score: {all_scores[-1]}')

#             # Reset score for this environment
#             scores = 0
        
#         if self.callbacks and not training:
#             for callback in self.callbacks:
#                 callback.on_test_end(logs=self._test_episode_config)

#         # close the environment
#         env.close()

#         return {
#             'scores': all_scores,
#             'log probs': all_log_probs,
#             # 'entropy': entropy_list,
#             # 'kl_divergence': kl_list
#         }

#     def get_config(self):
#         return {
#                 "agent_type": self.__class__.__name__,
#                 # "env": serialize_env_spec(self.env.spec),
#                 "env": self.env.spec.to_json(),
#                 "policy": self.policy_model.get_config(),
#                 "value_model": self.value_model.get_config(),
#                 "distribution": self.distribution,
#                 "discount": self.discount,
#                 "gae_coefficient": self.gae_coefficient,
#                 "policy_clip": self.policy_clip,
#                 "entropy_coefficient": self.entropy_coefficient,
#                 "loss": self.loss,
#                 "kl_coefficient": self.kl_coefficient,
#                 "normalize_advantages":self.normalize_advantages,
#                 "normalize_values": self.normalize_values,
#                 "normalizer_clip": self.value_norm_clip,
#                 "grad_clip":self.policy_grad_clip,
#                 "reward_clip":self.reward_clip,
#                 "lambda_": self.lambda_,
#                 "callbacks": [callback.get_config() for callback in self.callbacks if self.callbacks is not None],
#                 "save_dir": self.save_dir,
#                 "device": self.device,
#                 # "seed": self.seed,
#             }

#     def save(self, save_dir=None):
#         """Saves the model."""

#         # Change self.save_dir if save_dir
#         # if save_dir is not None:
#         #     self.save_dir = save_dir + "/ddpg/"

#         config = self.get_config()

#         # makes directory if it doesn't exist
#         os.makedirs(self.save_dir, exist_ok=True)

#         # writes and saves JSON file of DDPG agent config
#         with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
#             json.dump(config, f, cls=CustomJSONEncoder)

#         # saves policy and value model
#         self.policy_model.save(self.save_dir)
#         self.value_model.save(self.save_dir)

#         # if self.normalize_inputs:
#         #     self.state_normalizer.save_state(self.save_dir + "state_normalizer.npz")

#         # if wandb callback, save wandb config
#         # if self._wandb:
#         #     for callback in self.callbacks:
#         #         if isinstance(callback, rl_callbacks.WandbCallback):
#         #             callback.save(self.save_dir + "/wandb_config.json")


#     @classmethod
#     def load(cls, config, load_weights=True):
#         """Loads the model."""

#         # create EnvSpec from config
#         # env_spec_json = json.dumps(config["env"])
#         # print(f'env spec json: {env_spec_json}')
#         env_spec = gym.envs.registration.EnvSpec.from_json(config["env"])
#         # load policy model
#         policy_model = models.StochasticContinuousPolicy.load(config['save_dir'], load_weights)
#         # load value model
#         value_model = models.ValueModel.load(config['save_dir'], load_weights)
#         # load callbacks
#         callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]

#         # return PPO agent
#         agent = cls(
#             gym.make(env_spec),
#             policy_model = policy_model,
#             value_model = value_model,
#             distribution = config["distribution"],
#             discount=config["discount"],
#             gae_coefficient = config["gae_coefficient"],
#             policy_clip = config["policy_clip"],
#             entropy_coefficient = config["entropy_coefficient"],
#             loss = config["loss"],
#             kl_coefficient = config["kl_coefficient"],
#             normalize_advantages = config["normalize_advantages"],
#             normalize_values = config["normalize_values"],
#             value_normalizer_clip = config["normalizer_clip"],
#             policy_grad_clip = config["grad_clip"],
#             reward_clip = config['reward_clip'],
#             lambda_ = config["lambda_"],
#             callbacks=callbacks,
#             save_dir=config["save_dir"],
#             device=config["device"],
#         )

#         # if agent.normalize_inputs:
#         #     agent.state_normalizer = helper.Normalizer.load_state(config['save_dir'] + "state_normalizer.npz")

#         return agent

# def load_agent_from_config_path(config_path, load_weights=True):
#     """Loads an agent from a config file path."""
#     with open(
#         Path(config_path).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
#     ) as f:
#         config = json.load(f)

#     agent_type = config["agent_type"]

#     # Use globals() to get a reference to the class
#     agent_class = globals().get(agent_type)

#     if agent_class:
#         return agent_class.load(config_path, load_weights)

#     raise ValueError(f"Unknown agent type: {agent_type}")

# def load_agent_from_config(config, load_weights=True):
#     """Loads an agent from a loaded config file."""
#     agent_type = config["agent_type"]

#     # Use globals() to get a reference to the class
#     agent_class = globals().get(agent_type)

#     if agent_class:
#         return agent_class.load(config, load_weights)

#     raise ValueError(f"Unknown agent type: {agent_type}")


# def get_agent_class_from_type(agent_type: str):
#     """Builds an agent from a passed agent type str."""

#     types = {"Actor Critic": "ActorCritic",
#              "Reinforce": "Reinforce",
#              "DDPG": "DDPG",
#              "HER_DDPG": "HER",
#              "HER": "HER",
#              "TD3": "TD3",
#              "PPO": "PPO",
#             }

#     # Use globals() to get a reference to the class
#     agent_class = globals().get(types[agent_type])

#     if agent_class:
#         return agent_class

#     raise ValueError(f"Unknown agent type: {agent_type}")

# def init_sweep(sweep_config, comm=None):
#     # rank = MPI.COMM_WORLD.Get_rank()
#     if comm is not None:
#         logger.debug(f"Rank {rank} comm detected")
#         rank = comm.Get_rank()
#         logger.debug(f"Global rank {MPI.COMM_WORLD.Get_rank()} set to comm rank {rank}")
#         logger.debug(f"Rank {rank} in {comm.Get_name()}, name {comm.Get_name()}")
    
#     try:
#         # Set the environment variable
#         os.environ['WANDB_DISABLE_SERVICE'] = 'true'
#         # logger.debug(f"{comm.Get_name()}; Rank {rank} WANDB_DISABLE_SERVICE set to true")

#         # Set seeds (Seeds now set in train.  Update each)
#         # random.seed(train_config['seed'])
#         # np.random.seed(train_config['seed'])
#         # T.manual_seed(train_config['seed'])
#         # T.cuda.manual_seed(train_config['seed'])
#         # logger.debug(f'{comm.Get_name()}; Rank {rank} random seeds set')

#         # Only primary process (rank 0) calls wandb.init() to build agent and log data
#         if comm is not None:
#             if rank == 0:
#                 # logger.debug('MPI rank 0 process fired')
#                 # try:
#                 run_number = wandb_support.get_next_run_number(sweep_config["project"])
#                 logger.debug(f"{comm.Get_name()}; Rank {rank} run number set: {run_number}")
                
#                 run = wandb.init(
#                     project=sweep_config["project"],
#                     settings=wandb.Settings(start_method='thread'),
#                     job_type="train",
#                     name=f"train-{run_number}",
#                     tags=["train"],
#                     group=f"group-{run_number}",
#                     # dir=run_dir
#                 )
#                 logger.debug("wandb.init() fired")
#                 wandb_config = dict(wandb.config)
#                 model_type = list(wandb_config.keys())[0]
                
#                 # Wait for configuration to be populated
#                 max_retries = 10
#                 retry_interval = 1  # in seconds

#                 for _ in range(max_retries):
#                     if "model_type" in wandb.config:
#                         break
#                     logger.debug(f"{comm.Get_name()}; Rank {rank} Waiting for wandb.config to be populated...")
#                     time.sleep(retry_interval)

#                 if "model_type" in wandb.config:
#                     logger.debug(f'{comm.Get_name()}; Rank {rank} wandb.config: {wandb.config}')
#                     run.tags = run.tags + (model_type,)
#                 else:
#                     logger.error("wandb.config did not populate with model_type within the expected time", exc_info=True)
                
#                 run.tags = run.tags + (model_type,)
#                 logger.debug(f"{comm.Get_name()}; Rank {rank} run.tag set")
#                 env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
#                 # save env spec to string
#                 env_spec = env.spec.to_json()
#                 logger.debug(f"{comm.Get_name()}; Rank {rank} env built: {env.spec}")
#                 callbacks = []
#                 callbacks.append(rl_callbacks.WandbCallback(project_name=sweep_config["project"], run_name=f"train-{run_number}", _sweep=True))
#                 logger.debug(f"{comm.Get_name()}; Rank {rank} callbacks created")

#             else:
#                 env_spec = None
#                 callbacks = None
#                 run_number = None
#                 wandb_config = None
            
#             # Use MPI Barrier to sync processes
#             logger.debug(f"{comm.Get_name()}; Rank {rank} init_sweep calling MPI Barrier")
#             comm.Barrier()
#             logger.debug(f"{comm.Get_name()}; Rank {rank} init_sweep MPI Barrier passed")

#             env_spec = comm.bcast(env_spec, root=0)
#             callbacks = comm.bcast(callbacks, root=0)
#             run_number = comm.bcast(run_number, root=0)
#             wandb_config = comm.bcast(wandb_config, root=0)
#             model_type = sweep_config['parameters']['model_type']
#             logger.debug(f"{comm.Get_name()}; Rank {rank} broadcasts complete")

#             agent = get_agent_class_from_type(model_type)
#             logger.debug(f"{comm.Get_name()}; Rank {rank} agent class found. Calling sweep_train")
#             agent.sweep_train(wandb_config, env_spec, callbacks, run_number, comm)
        
#         else:
#             print('comm = None')
#             run_number = wandb_support.get_next_run_number(sweep_config["project"])
#             logger.debug(f"run number set: {run_number}")
#             print(f'run number:{run_number}')
            
#             run = wandb.init(
#                 project=sweep_config["project"],
#                 settings=wandb.Settings(start_method='thread'),
#                 job_type="train",
#                 name=f"train-{run_number}",
#                 tags=["train"],
#                 group=f"group-{run_number}",
#                 # dir=run_dir
#             )
#             logger.debug("wandb.init() fired")
#             wandb_config = dict(wandb.config)
#             print(f'wandb config: {wandb_config}')
#             model_type = wandb_config['model_type']
            
#             # Wait for configuration to be populated
#             max_retries = 10
#             retry_interval = 1  # in seconds

#             for _ in range(max_retries):
#                 if "model_type" in wandb.config:
#                     break
#                 logger.debug(f"Waiting for wandb.config to be populated...")
#                 time.sleep(retry_interval)

#             if "model_type" in wandb.config:
#                 logger.debug(f'wandb.config: {wandb.config}')
#                 run.tags = run.tags + (model_type,)
#             else:
#                 logger.error("wandb.config did not populate with model_type within the expected time", exc_info=True)
            
#             run.tags = run.tags + (model_type,)
#             logger.debug(f"run.tag set")
#             # env = gym.make(**{param: value["value"] for param, value in sweep_config["parameters"]["env"]["parameters"].items()})
#             env_params = {
#                 key.replace("env_", ""): val["value"]
#                 for key, val in sweep_config["parameters"].items()
#                 if key.startswith("env_")
#             }
#             #DEBUG
#             print(f'env_params:{env_params}')
#             env = gym.make(**env_params)
#             # save env spec to string
#             env_spec = env.spec.to_json()
#             logger.debug(f"env built: {env.spec}")
#             callbacks = []
#             callbacks.append(rl_callbacks.WandbCallback(project_name=sweep_config["project"], run_name=f"train-{run_number}", _sweep=True))
#             logger.debug(f"callbacks created")
#             agent = get_agent_class_from_type(model_type)
#             logger.debug(f"agent class found. Calling sweep_train")
#             agent.sweep_train(wandb_config, env_spec, callbacks, run_number)

#     except Exception as e:
#         logger.error(f"Error in rl_agent.init_sweep: {e}", exc_info=True)

# def init_sweep(sweep_config):
#     try:
#         # Set the environment variable
#         os.environ['WANDB_DISABLE_SERVICE'] = 'true'
#         run_number = wandb_support.get_next_run_number(sweep_config["project"])
#         logger.debug(f"run number set: {run_number}")
#         run = wandb.init(
#             project=sweep_config["project"],
#             settings=wandb.Settings(start_method='thread'),
#             job_type="train",
#             name=f"train-{run_number}",
#             tags=["train"],
#             group=f"group-{run_number}",
#         )
#         logger.debug("wandb.init() fired")
#         wandb_config = dict(wandb.config)
#         model_type = list(wandb_config.keys())[0]
#         # Wait for configuration to be populated
#         max_retries = 10
#         retry_interval = 1  # in seconds
#         for _ in range(max_retries):
#             if "model_type" in wandb.config:
#                 break
#             logger.debug("Waiting for wandb.config to be populated...")
#             time.sleep(retry_interval)
#         if "model_type" in wandb.config:
#             logger.debug(f'wandb.config: {wandb.config}')
#             run.tags = run.tags + (model_type,)
#         else:
#             logger.error("wandb.config did not populate with model_type within the expected time", exc_info=True)
#         run.tags = run.tags + (model_type,)
#         logger.debug("run.tag set")
#         # Extract environment parameters from sweep_config
#         env_params = {
#             key.replace("env_", ""): val["value"]
#             for key, val in sweep_config["parameters"].items()
#             if key.startswith("env_")
#         }
#         env = gym.make(**env_params)
#         env_spec = env.spec.to_json()
#         logger.debug(f"env built: {env.spec}")
#         callbacks = []
#         callbacks.append(rl_callbacks.WandbCallback(project_name=sweep_config["project"], run_name=f"train-{run_number}", _sweep=True))
#         logger.debug(f"callbacks created")
#         agent = get_agent_class_from_type(model_type)
#         logger.debug(f"agent class found. Calling sweep_train")
#         agent.sweep_train(wandb_config, env_spec, callbacks, run_number)
#     except Exception as e:
#         logger.error(f"Error in rl_agent.init_sweep: {e}", exc_info=True)

def init_sweep(config):
    try:
        # Extract the model type (stored as a list) from the config.
        model_type_list = config.get("model_type", [])
        if not model_type_list:
            raise ValueError("No model type provided in config.")
        model_type = model_type_list[0]

        # Inject wandb settings into the config if not already provided.
        if "wandb" not in config:
            run_number = wandb_support.get_next_run_number(config["project"])
            config["wandb"] = {
                "project": config["project"],
                "name": f"train-{run_number}",
                "job_type": "train",
                "tags": ["train", model_type],
                "group": f"group-{run_number}",
            }

        # Build the environment.
        env_params = {
            key.replace("env_", ""): config[key]
            for key in config if key.startswith("env_")
        }
        env = gym.make(**env_params)
        env_spec = env.spec.to_json()
        logger.debug(f"Environment built: {env.spec}")

        # Create callbacks (using your custom WandbCallback).
        callbacks = []
        callbacks.append(WandbCallback(
            project_name=config["project"],
            run_name=config["wandb"]["name"],
            _sweep=True
        ))
        logger.debug("Callbacks created")

        # Get the appropriate agent class from the model type.
        agent = get_agent_class_from_type(model_type)
        logger.debug("Agent class found. Calling sweep_train")

        # Call the sweep_train function on the agent with the full config.
        agent.sweep_train(config, env_spec, callbacks, run_number)
    except Exception as e:
        logger.error(f"Error in init_sweep: {e}", exc_info=True)

