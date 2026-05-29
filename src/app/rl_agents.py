"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
from abc import ABC, abstractmethod
import json
import os
from typing import Protocol, Optional, Dict, List, TypeAlias, Any, runtime_checkable
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

# from .icm import ICM
from .intrinsic_motivation import IntrinsicMotivation
from .rl_callbacks import WandbCallback, Callback
from .rl_callbacks import load as callback_load
from .models import select_policy_model, StochasticContinuousPolicy, StochasticDiscretePolicy, ValueModel, ContinuousCritic, DiscreteCritic, ActorModel
from .schedulers import ScheduleWrapper
from .adaptive_kl import AdaptiveKL
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, Buffer
from .normalizer import BaseNormalizer, RewardNorm
from .noise import Noise, NormalNoise, UniformNoise
import wandb
from . import wandb_support
from .torch_utils import set_seed, get_device, move_to_device, VarianceScaling_
from .env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper, Action
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


from app.agent_utils import load_agent, get_agent_class_from_type, compute_n_step_return, compute_advantages_and_returns, compute_monte_carlo_returns, grad_norm_from_optimizer, setup_auto_entropy, soft_update


## Base Agent Class ##
class Agent(ABC):
    """Base class for all RL agents."""

    def __init__(self,
                 save_dir: str = "models/",
                 device: Optional[str | T.device] = None,
                 log_level: str = 'INFO',
                 name: str | None = None,
                 **kwargs
    ):
        self.name = name if name else self.__class__.__name__
        self.logger = get_logger(self.name, level=log_level.upper())
        self.kwargs = kwargs
        try:
            self.save_dir = self._setup_save_dir(save_dir)
            self.device = get_device(device)

            # Set internal attributes
            # self._distributed = False

            self._diag_freq = None
            self._learn_count = 0

            if self.kwargs is not None:
                for key, value in self.kwargs.items():
                    setattr(self, key, value)
           
        except Exception as e:
            self.logger.error(f"Error in Agent init: {e}", exc_info=True)

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

    # def _step(self, env: EnvWrapper, step: int, states: np.ndarray, num_episodes: int, episode_scores: np.ndarray, completed_episodes: np.ndarray, score_history: deque, learn: bool = True, training: bool = True):
    #     """Step function for the agent."""
    #     raise NotImplementedError("Subclasses must implement _step.")

    # @abstractmethod
    # def _distributed_learn(self, *args, **kwargs):
    #     """Handle distributed learning for both on-policy and off-policy agents."""
    #     raise NotImplementedError("Subclasses must implement _distributed_learn.")
    
    # @abstractmethod
    # def get_parameters(self):
    #     """Return a dictionary of model parameters: {model_name: params}."""
    #     raise NotImplementedError("Subclasses must implement get_parameters.")

    # @abstractmethod
    # def apply_parameters(self, params):
    #     """Apply the provided parameters to the agent's models."""
    #     raise NotImplementedError("Subclasses must implement apply_parameters.")

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

    def get_config(self):
        return {
            "type": self.__class__.__name__,
            "config":{
                "save_dir": self.save_dir,
                "name": self.name,
            }
        }

    @abstractmethod
    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> T.Tensor:
        """Returns an action given a state."""
        raise NotImplementedError("Subclasses must implement get_action.")

    # @abstractmethod
    # def train(self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
    #     """Trains the model for 'episodes' number of episodes."""
    #     raise NotImplementedError("Subclasses must implement train.")
    
    @abstractmethod
    def learn(self, step:int, sample:dict, **kwargs: Any)->dict:
        """Updates the model."""
        raise NotImplementedError("Subclasses must implement learn.")

    # @abstractmethod
    # def test(self, num_episodes=None, render=False, render_freq=10):
    #     """Runs a test over 'num_episodes'."""
    #     raise NotImplementedError("Subclasses must implement test.")

    @abstractmethod
    def save(self):
        """Saves the model."""
        raise NotImplementedError("Subclasses must implement save.")
    
    @classmethod
    @abstractmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        raise NotImplementedError("Subclasses must implement load.")

class Reinforce(Agent):
    def __init__(
        self,
        policy: StochasticDiscretePolicy,
        value: ValueModel|None = None,
        discount: float = 0.99,
        state_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float = 0.01,
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        save_dir: str = "models",
        device: str = None,
        **kwargs,
    ):
        """
        Reinforce Agent.

        Args:
            policy: The policy model used for action selection.
            value: The value model used for state-value prediction.
            discount: The discount factor for future rewards.
            state_normalizer: The normalizer for state inputs.
            advantage_normalizer: The normalizer for advantage/return inputs.
            reward_normalizer: The normalizer for reward inputs.
            entropy_coefficient: The coefficient for entropy regularization.
            entropy_schedule: The schedule for entropy regularization.
            auto_entropy_tuning: Whether to automatically tune the entropy coefficient.
            entropy_lr: The learning rate for the entropy coefficient.
            target_entropy_scale: The scale for the target entropy.
            save_dir: The directory to save the model.
            device: The device to use for computations.
            kwargs: Additional keyword arguments.
        """
        try:
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.value = value
            self.discount = discount
            self.state_normalizer = state_normalizer
            self.advantage_normalizer = advantage_normalizer
            self.reward_normalizer = reward_normalizer
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.auto_entropy_tuning = auto_entropy_tuning
            self.entropy_lr = entropy_lr
            self.target_entropy_scale = target_entropy_scale
            if self.auto_entropy_tuning:
                self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                    self.policy,
                    target_entropy_scale=target_entropy_scale,
                    lr=entropy_lr,
                    device=self.device,
                )
            
        except Exception as e:
            self.logger.error(f"Error in Reinforce.__init__: {e}", exc_info=True)

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> T.Tensor:
        """
        Select an action based on the current policy.
        Returns actions that are already scaled to the environment's action space.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            context: str: The context of the action (train, test).
        
        Returns:
            T.Tensor: actions.
        """
        
        if context == 'train':
            dist = self.policy(states, goals)
            actions = dist.sample()

        elif context == 'test':
            with T.no_grad():
                dist = self.policy(states, goals)
                actions = self.policy.get_mean_actions(dist)
            
        else:
            raise ValueError(f"Invalid context: {context}")

        return actions

    def learn(self, step: int, sample: list[dict], **kwargs: Any)->dict:
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False
        
        learn_metrics = {}

        all_states = [trajectory['states'] for trajectory in sample]
        all_actions = [trajectory['actions'] for trajectory in sample]
        all_rewards = [trajectory['rewards'] for trajectory in sample]
        # all_terminations = [trajectory['terminations'] for trajectory in completed_trajectories]
        # all_truncations = [trajectory['truncations'] for trajectory in completed_trajectories]

        for i, rewards in enumerate(all_rewards):
            if self.reward_normalizer:
                rewards = self.reward_normalizer.normalize(rewards)
            all_rewards[i] = rewards
        
        all_returns = [compute_monte_carlo_returns(rewards, self.discount, device=self.device) for rewards in all_rewards]

        # # Iterate over completed trajectories
        # for trajectory in completed_trajectories:
        #     all_states.append(trajectory['states'])
        #     all_actions.append(trajectory['actions'])
        #     # _return = compute_monte_carlo_returns(trajectory['rewards'], self.discount, device=self.device)
        #     # all_returns.append(_return)
        #     all_rewards.append(trajectory['rewards'])
            

        # Use T.cat to concatenate all tensors in list into single tensor of shape [total_steps, obs_dim]
        states = T.cat(all_states, dim=0)
        actions = T.cat(all_actions, dim=0)
        returns = T.cat(all_returns, dim=0).unsqueeze(-1)

        # Normalize states if using normalizer
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)

        # Clear gradients
        self.policy.optimizer.zero_grad()
        if self.value:
            self.value.optimizer.zero_grad()
        
        # Calculate advantages and value loss if using value function
        if self.value:
            values = self.value(states)
            advantages = returns.detach() - values
            value_loss = advantages.pow(2).mean()
        else:
            values = T.zeros_like(returns)
            advantages = T.zeros_like(returns)
            value_loss = 0

        # Calculate policy loss
        # Create policy_weight value based on advantages (if value) or returns
        if self.value:
            policy_weight = advantages.detach()
        else:
            policy_weight = returns.detach()

        if self.advantage_normalizer:
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(policy_weight)
            policy_weight = self.advantage_normalizer.normalize(policy_weight)
        
        # dist, logits = self.policy_model(states)
        dist = self.policy(states)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropies = dist.entropy().unsqueeze(-1)

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get policy loss
        policy_loss = -(log_probs * policy_weight + entropy_coefficient * entropies).mean()

        # Calculate gradients
        total_loss = policy_loss + value_loss
        total_loss.backward()

        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                # summarize_tensor(terminations, "terminations"),
                # summarize_tensor(truncations, "truncations"),
                summarize_tensor(values, "values"),
                summarize_tensor(advantages, "advantages"),
                summarize_tensor(returns, "returns"),
                summarize_tensor(log_probs, "log_probs"),
                summarize_tensor(entropies, "entropies"),
                f"entropy_coef={float(entropy_coefficient)}",
            )
            value_grad_norm = grad_norm_from_optimizer(self.value.optimizer)
            policy_grad_norm = grad_norm_from_optimizer(self.policy.optimizer)
            self.logger.debug(
                "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                "value_loss=%.6f policy_loss=%.6f",
                step,
                self._learn_count,
                value_grad_norm,
                policy_grad_norm,
                float(value_loss.item()),
                float(policy_loss.item()),
            )

        # Update weights
        self.policy.optimizer.step()
        if self.value:
            self.value.optimizer.step()

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            if self.policy.distribution in ['normal', 'beta']:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            else: # Discrete actor
                alpha_loss = -(self.log_alpha * ((dist.probs * log_probs).sum(dim=-1) + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        value_learning_rate = self.value.optimizer.param_groups[0]['lr']

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantages': advantages.mean().item(),
            'returns': returns.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate,
        })

        return learn_metrics

    def get_config(self):
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "policy": self.policy.get_config(),
            "value": self.value.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
        })
        return config

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
        if self.advantage_normalizer:
            self.advantage_normalizer.save(self.save_dir + "advantage_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["policy"]["env"])
        policy = StochasticDiscretePolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
        value_model = ValueModel.load(config_dir, 'value', load_weights, env=env_wrapper) if config.get('value', None) else None
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "/state_normalizer.pt") if config.get('state_normalizer', None) else None
        advantage_normalizer = BaseNormalizer.load(config["advantage_normalizer"], config["save_dir"] + "/advantage_normalizer.pt") if config.get('advantage_normalizer', None) else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None

        # return reinforce agent
        agent = cls(
            policy=policy,
            value=value_model,
            discount=config["discount"],
            state_normalizer=state_normalizer,
            advantage_normalizer=advantage_normalizer,
            reward_normalizer=reward_normalizer,
            entropy_coefficient=config["entropy_coefficient"],
            entropy_schedule=ScheduleWrapper(**config["entropy_schedule"]) if config.get("entropy_schedule", None) else None,
            auto_entropy_tuning=config["auto_entropy_tuning"],
            entropy_lr=config["entropy_lr"],
            target_entropy_scale=config["target_entropy_scale"],
            save_dir=config["save_dir"],
        )

        return agent

class ActorCritic(Agent):
    """Actor Critic Agent."""

    def __init__(
        self,
        policy: StochasticDiscretePolicy|StochasticContinuousPolicy,
        value: ValueModel,
        discount: float=0.99,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float=0.01, # Only used if auto entropy = False
        entropy_schedule: ScheduleWrapper|None = None, # Only used if auto entropy = False
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        gae_coefficient: float=0.95,
        policy_grad_clip: float=1.0,
        value_grad_clip: float=1.0,
        value_coef: float=0.5,
        bootstrap_truncations: bool=True,
        save_dir: str = "models/",
        device: Optional[str | T.device] = None,
        **kwargs,
    ):
        try:
            # super().__init__(
            #     policy,
            #     value,
            #     discount,
            #     state_normalizer,
            #     goal_normalizer,
            #     advantage_normalizer = advantage_normalizer,
            #     reward_normalizer = reward_normalizer,
            #     entropy_coefficient = entropy_coefficient,
            #     entropy_schedule = entropy_schedule,
            #     auto_entropy_tuning = auto_entropy_tuning,
            #     entropy_lr = entropy_lr,
            #     target_entropy_scale = target_entropy_scale,
            #     save_dir = save_dir,
            #     device=device,
            #     **kwargs
            # )
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.value = value
            self.discount = discount
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.advantage_normalizer = advantage_normalizer
            self.reward_normalizer = reward_normalizer
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.auto_entropy_tuning = auto_entropy_tuning
            self.entropy_lr = entropy_lr
            self.target_entropy_scale = target_entropy_scale
            if self.auto_entropy_tuning:
                self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                    self.policy,
                    target_entropy_scale=target_entropy_scale,
                    lr=entropy_lr,
                    device=self.device,
                )
            self.gae_coefficient = gae_coefficient
            self.policy_grad_clip = policy_grad_clip
            self.value_grad_clip = value_grad_clip
            self.value_coef = value_coef
            self.bootstrap_truncations = bootstrap_truncations
        except Exception as e:
            self.logger.error(f"Error in ActorCritic init: {e}", exc_info=True)

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> T.Tensor:
        """
        Select an action based on the current policy.
        Returns actions that are already scaled to the environment's action space.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            context: str: The context of the action (train, test).
        
        Returns:
            T.Tensor: actions.
        """
        
        if context == 'train':
            dist = self.policy(states, goals)
            actions = dist.sample()

        elif context == 'test':
            with T.no_grad():
                dist = self.policy(states, goals)
                actions = self.policy.get_mean_actions(dist)
            
        else:
            raise ValueError(f"Invalid context: {context}")

        return actions

    def learn(self, step:int, sample:dict, **kwargs: Any)->dict:
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        self.policy.optimizer.zero_grad()
        self.value.optimizer.zero_grad()

        # Extract trajectories from buffer
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        terminations = sample['terminations']
        truncations = sample['truncations']
        # first_steps = sample["first_steps"]
        valid_indices = sample["valid_indices"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        # Normalize states and goals
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            goals = self.goal_normalizer.normalize(goals)
            ach_goals = self.goal_normalizer.normalize(ach_goals)
        if self.reward_normalizer:
            rewards = self.reward_normalizer.normalize(rewards)

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get trajectory length, num_envs, and feature dims
        # trajectory_length, num_envs, obs_dim = states.shape
        # action_dim = actions.shape[-1]
        traj_len, num_envs = rewards.shape
        total_samples = traj_len * num_envs

        # Flatten trajectory data
        states_flat = states.reshape(total_samples, -1)
        next_states_flat = next_states.reshape(total_samples, -1)
        actions_flat = actions.reshape(total_samples, -1)
        goals_flat = goals.reshape(total_samples, -1) if goals is not None else None

        # Flatten goals if not None
        # if goals is not None:
        #     goal_dim = goals.shape[-1]
        #     flat_goals = goals.reshape(trajectory_length * num_envs, goal_dim)
        #     flat_next_ach_goals = next_ach_goals.reshape(trajectory_length * num_envs, goal_dim)
        # else:
        #     flat_goals = None
        #     flat_next_ach_goals = None

        state_values = self.value(states_flat, goals_flat).reshape(traj_len, num_envs)
        next_state_values = self.value(next_states_flat, goals_flat).reshape(traj_len, num_envs).detach()

        advantages, returns, td_errors = compute_advantages_and_returns(
            rewards,
            state_values,
            next_state_values,
            terminations,
            truncations,
            self.discount,
            self.gae_coefficient,
            self.bootstrap_truncations,
            device=self.device
        )

        # Filter phantom steps
        valid_idx = valid_indices.squeeze(-1)
        states_flat = states_flat[valid_idx]
        next_states_flat = next_states_flat[valid_idx]
        actions_flat = actions_flat[valid_idx]
        goals_flat = goals_flat[valid_idx] if goals is not None else None
        state_values_flat = state_values.reshape(total_samples)[valid_idx]
        next_state_values_flat = next_state_values.reshape(total_samples)[valid_idx]
        advantages_flat = advantages.reshape(total_samples)[valid_idx]
        returns_flat = returns.reshape(total_samples)[valid_idx]
        td_errors_flat = td_errors.reshape(total_samples)[valid_idx]

        # Calculate value loss
        value_loss = self.value_coef * (state_values_flat - returns_flat.detach()).pow(2).mean()

        # Create separate policy advantage in case using advantage normalizer
        policy_advantages = advantages_flat.detach()
        if self.advantage_normalizer:
            policy_advantages = policy_advantages.reshape(-1,1)
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(policy_advantages)
            policy_advantages = self.advantage_normalizer.normalize(policy_advantages).reshape(-1)

        # Get log probs and entropy values from current policy dist
        dist = self.policy(states_flat, goals_flat)
        # dist = self.policy.transform_distribution(base_dist)
        # reshape flat actions to be vector if categorical distribution
        if self.policy.distribution == 'categorical':
            actions_flat = actions_flat.squeeze(-1)
        log_probs = dist.log_prob(actions_flat).flatten()#.reshape(traj_len, num_envs)
        # else:
        #     log_probs = dist.log_prob(actions_flat).flatten()#.reshape(traj_len, num_envs)
        
        # Only calculate entropy if entropy coefficient > 0
        # if entropy_coefficient > 0.0:
        entropies = dist.entropy().flatten()#.reshape(traj_len, num_envs)
        # else:
        #     entropies = T.zeros_like(log_probs)

        # Calculate policy loss
        policy_loss = -(log_probs * policy_advantages + entropy_coefficient * entropies).mean()

        # Backpropogate
        value_loss.backward()
        policy_loss.backward()
        # Clip gradients if grad clips
        if self.value_grad_clip:
            value_grad_norm = T.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        if self.policy_grad_clip:
            policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_grad_clip)

        nonfinite_values = (
            count_nonfinite(state_values)
            + count_nonfinite(next_state_values)
            + count_nonfinite(td_errors)
            + count_nonfinite(advantages)
            + count_nonfinite(returns)
            + count_nonfinite(log_probs)
            + count_nonfinite(entropies)
        )

        if should_log_diag or nonfinite_values > 0:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(state_values_flat, "values"),
                summarize_tensor(next_state_values_flat, "next_values"),
                summarize_tensor(td_errors_flat, "td_errors"),
                summarize_tensor(advantages_flat, "advantages"),
                summarize_tensor(returns_flat, "returns"),
                summarize_tensor(log_probs, "log_probs"),
                summarize_tensor(entropies, "entropies"),
                f"entropy_coef={float(entropy_coefficient)}",
            )
        
            self.logger.debug(
                "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                "value_loss=%.6f policy_loss=%.6f",
                step,
                self._learn_count,
                float(value_grad_norm) if value_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(value_loss.item()),
                float(policy_loss.item()),
            )

        self.value.optimizer.step()
        self.policy.optimizer.step()

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        value_learning_rate = self.value.optimizer.param_groups[0]['lr']

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'temporal_difference': td_errors_flat.mean().item(),
            'advantages': advantages_flat.mean().item(),
            'returns': returns_flat.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate,
        })

        return learn_metrics

    def get_config(self):
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "policy": self.policy.get_config(),
            "value": self.value.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "gae_coefficient": self.gae_coefficient,
            "policy_grad_clip": self.policy_grad_clip,
            "value_grad_clip": self.value_grad_clip,
            "value_coef": self.value_coef,
            "bootstrap_truncations": self.bootstrap_truncations,
        })
        return config

    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.value.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")
        if self.advantage_normalizer:
            self.advantage_normalizer.save(self.save_dir + "advantage_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["policy"]["env"])
        policy = StochasticDiscretePolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
        value = ValueModel.load(config_dir, 'value', load_weights, env=env_wrapper)
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = BaseNormalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        advantage_normalizer = BaseNormalizer.load(config["advantage_normalizer"], config["save_dir"] + "advantage_normalizer.pt") if config["advantage_normalizer"] else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None

        agent = cls(
            policy=policy,
            value=value,
            discount=config["discount"],
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            advantage_normalizer=advantage_normalizer,
            reward_normalizer=reward_normalizer,
            entropy_coefficient=config["entropy_coefficient"],
            entropy_schedule=ScheduleWrapper(**config["entropy_schedule"]) if config.get("entropy_schedule", None) else None,
            auto_entropy_tuning=config["auto_entropy_tuning"],
            entropy_lr=config["entropy_lr"],
            target_entropy_scale=config["target_entropy_scale"],
            gae_coefficient=config["gae_coefficient"],
            policy_grad_clip=config["policy_grad_clip"],
            value_grad_clip=config["value_grad_clip"],
            value_coef=config["value_coef"],
            bootstrap_truncations=config["bootstrap_truncations"],
            save_dir=config["save_dir"],
        )

        return agent

class PPO(Agent):
    """
    Proximal Policy Optimization (PPO) agent implementation.

    Attributes:
        policy: (StochasticDiscretePolicy|StochasticContinuousPolicy): The policy model used for action selection.
        value: (ValueModel): The value model used for state-value prediction.
        discount: (float): Discount factor for future rewards.
        gae_coefficient: (float): GAE smoothing coefficient.
        state_normalizer: (BaseNormalizer): Normalizer for state inputs.
        goal_normalizer: (BaseNormalizer): Normalizer for goal inputs.
        advantage_normalizer: (BaseNormalizer): Normalizer for advantages
        reward_normalizer:(RewardNorm): Normalizer for rewards.
        entropy_coefficient: (float): Coefficient for entropy regularization.
        entropy_schedule: (ScheduleWrapper): Rate at which to decay entropy coefficient per learn epoch.
        auto_entropy_tuning: (bool): Whether to automatically tune the entropy coefficient.
        entropy_lr: (float): Learning rate for the entropy coefficient. Only used if auto entropy = True
        target_entropy_scale: (float): Scale for the target entropy. Only used if auto entropy = True and discrete action space
        kl_coefficient: (float): Coefficient for KL divergence penalty.
        kl_adapter: (AdaptiveKL): Adjusts kl_coefficient to keep KL Divergence near target.
        policy_clip: (float): Clipping value for policy ratio updates.
        policy_clip_schedule: (ScheduleWrapper): Rate at which to decay policy clip per learn epoch.
        policy_grad_clip: (float): Maximum norm for policy model gradients.
        value_clip: (float): Clipping value for value model updates.
        value_clip_schedule: (ScheduleWrapper): Rate at which to decay value clip per learn epoch.
        value_grad_clip: (float): Maximum norm for value model gradients.
        value_coef: (float): value to weight the value loss by.
        reward_clip: (float): Maximum absolute value for reward clipping.
        curiosity: (ICM|None): Intrinsic Curiosity Module for curiosity-driven learning.
        bootstrap_truncations: (bool): Whether to bootstrap the truncated returns.
        save_dir: (str): Directory to save models and configurations.
        device: (str): Device for computations ('cpu' or 'cuda').
    """

    def __init__(
        self,
        policy: StochasticContinuousPolicy | StochasticDiscretePolicy,
        value: ValueModel,
        discount: float = 0.99,
        gae_coefficient: float = 0.95,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float = 0.01,
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool = True,
        entropy_lr: float = 3e-4,
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        kl_coefficient: float = 0.0,
        kl_adapter: AdaptiveKL|None = None,
        policy_clip: float = 0.2,
        policy_clip_schedule: ScheduleWrapper|None = None,
        policy_grad_clip: float = float('inf'),
        value_clip: float = 0.2,
        value_clip_schedule: ScheduleWrapper|None = None,
        value_grad_clip: float = float('inf'),
        value_coef: float = 0.5,
        reward_clip: float = float('inf'),
        intrinsic_motivation: IntrinsicMotivation|None = None,
        bootstrap_truncations: bool=True,
        save_dir: str = 'models',
        device: str | T.device | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the PPO agent.
        Args:
            policy: (StochasticDiscretePolicy|StochasticContinuousPolicy): The policy model used for action selection.
            goal_normalizer: (BaseNormalizer): Normalizer for goal inputs.
            advantage_normalizer: (BaseNormalizer): Normalizer for advantages.
            reward_normalizer: (RewardNorm): Normalizer for rewards.
            entropy_coefficient: (float): Coefficient for entropy regularization.
            entropy_schedule: (ScheduleWrapper): Rate at which to decay entropy coefficient per learn epoch.
            auto_entropy_tuning: (bool): Whether to automatically tune the entropy coefficient.
            entropy_lr: (float): Learning rate for the entropy coefficient. Only used if auto entropy = True
            target_entropy_scale: (float): Scale for the target entropy. Only used if auto entropy = True and discrete action space
            kl_coefficient: (float): Coefficient for KL divergence penalty.
            kl_adapter: (AdaptiveKL): Adjusts kl_coefficient to keep KL Divergence near target.
            policy_clip: (float): Clipping value for policy ratio updates.
            policy_clip_schedule: (ScheduleWrapper): Rate at which to decay policy clip per learn epoch.
            policy_grad_clip: (float): Maximum norm for policy model gradients.
            value_clip: (float): Clipping value for value model updates.
            value_clip_schedule: (ScheduleWrapper): Rate at which to decay value clip per learn epoch.
            value_grad_clip: (float): Maximum norm for value model gradients.
            value_coef: (float): value to weight the value loss by.
            reward_clip: (float): Maximum absolute value for reward clipping.
            intrinsic_motivation: (IntrinsicMotivation): Intrinsic Motivation Module for curiosity-driven learning.
            bootstrap_truncations: (bool): Whether to bootstrap the truncated returns.
            save_dir: (str): Directory to save models and configurations.
            device: (str): Device for computations ('cpu' or 'cuda').
            kwargs: Additional keyword arguments.
        """
        try:
            # super().__init__(
            #     policy,
            #     value,
            #     discount,
            #     state_normalizer,
            #     None,
            #     advantage_normalizer = advantage_normalizer,
            #     reward_normalizer = reward_normalizer,
            #     entropy_coefficient = entropy_coefficient,
            #     entropy_schedule = entropy_schedule,
            #     auto_entropy_tuning = auto_entropy_tuning,
            #     entropy_lr = entropy_lr,
            #     target_entropy_scale = target_entropy_scale,
            #     save_dir = save_dir,
            #     device=device,
            #     **kwargs
            # )
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.value = value
            self.discount = discount
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.advantage_normalizer = advantage_normalizer
            self.reward_normalizer = reward_normalizer
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.auto_entropy_tuning = auto_entropy_tuning
            self.entropy_lr = entropy_lr
            self.target_entropy_scale = target_entropy_scale
            if self.auto_entropy_tuning:
                self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                    self.policy,
                    target_entropy_scale=target_entropy_scale,
                    lr=entropy_lr,
                    device=self.device,
                )
            self.gae_coefficient = gae_coefficient
            self.kl_coefficient = kl_coefficient
            self.kl_adapter = kl_adapter
            self.policy_clip = policy_clip
            self.policy_clip_schedule = policy_clip_schedule
            self.policy_grad_clip = policy_grad_clip
            self.value_clip = value_clip
            self.value_clip_schedule = value_clip_schedule
            self.value_grad_clip = value_grad_clip
            self.value_coef = value_coef
            self.reward_clip = reward_clip
            self.intrinsic_motivation = intrinsic_motivation
            self.bootstrap_truncations = bootstrap_truncations
        except Exception as e:
            self.logger.error(f"Error in PPO.__init__: {e}", exc_info=True)

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> T.Tensor:
        """
        Select an action based on the current policy.
        Returns actions that are already scaled to the environment's action space.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            context: str: The context of the action (train, test).
        
        Returns:
            T.Tensor: actions.
        """
        
        if context == 'train':
            dist = self.policy(states, goals)
            actions = dist.sample()

        elif context == 'test':
            with T.no_grad():
                dist = self.policy(states, goals)
                actions = self.policy.get_mean_actions(dist)
            
        else:
            raise ValueError(f"Invalid context: {context}")

        return actions

    def learn(self, step:int, sample:dict, learning_epochs:int, mini_batch_size:int, **kwargs: Any)->dict:
        """
        Perform learning updates using the collected trajectory.

        Args:
            step (int): Current step.
            sample (dict): Collected rollouts containing states, actions, etc.
            learning_epochs (int): Number of epochs per update.
            mini_batch_size (int): Mini batch size for training.

        Returns:
            dict: Learning metrics.
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False
        
        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        first_steps = sample["first_steps"]
        valid_indices = sample["valid_indices"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        # Get current values of policy/value clip and entropy/kl coefficients
        policy_clip = self.policy_clip
        if self.policy_clip_schedule:
            policy_clip *= self.policy_clip_schedule.get_factor()

        value_clip = self.value_clip
        if self.value_clip_schedule:
            value_clip *= self.value_clip_schedule.get_factor()

        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        kl_coefficient = self.kl_coefficient
        if self.kl_adapter:
            kl_coefficient = self.kl_adapter.get_beta()

        # Normalize states and goals
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            ach_goals = self.goal_normalizer.normalize(ach_goals)
            next_ach_goals = self.goal_normalizer.normalize(next_ach_goals)
            goals = self.goal_normalizer.normalize(goals)
        if self.reward_normalizer:
            rewards = self.reward_normalizer.normalize(rewards)

        # Clip rewards if finite and not using reward normalizer
        if T.isfinite(T.tensor(self.reward_clip)) and self.reward_normalizer is None:
            rewards = T.clamp(rewards, min=-self.reward_clip, max=self.reward_clip)

        # Get trajectory length, num envs, and total samples for reshaping
        traj_len, num_envs = rewards.shape
        total_samples = traj_len * num_envs

        # Flatten trajectory data
        states_flat = states.reshape(total_samples, -1)
        next_states_flat = next_states.reshape(total_samples, -1)
        actions_flat = actions.reshape(total_samples, -1)
        goals_flat = goals.reshape(total_samples, -1) if goals is not None else None
        # ach_goals_flat = ach_goals.reshape(total_samples, -1) if ach_goals is not None else None
        # next_ach_goals_flat = next_ach_goals.reshape(total_samples, -1) if next_ach_goals is not None else None
        # terminations_flat = terminations.reshape(total_samples, -1)
        # truncations_flat = truncations.reshape(total_samples, -1)

        # Use intrinsic rewards if ICM
        if self.intrinsic_motivation:
            curiosity_loss = self.curiosity.train(states_flat, next_states_flat, actions_flat)
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(states_flat, next_states_flat, actions_flat)
            intrinsic_reward = intrinsic_reward.reshape(num_steps, num_envs)
            if step > self.curiosity.extrinsic_threshold:
                rewards += self.curiosity.reward_weight * intrinsic_reward
            else:
                rewards = self.curiosity.reward_weight * intrinsic_reward

        # Get current log probs and values
        with T.no_grad():
            cur_dist = self.policy(states_flat, goals_flat)
            if self.policy.distribution == 'categorical':
                cur_log_probs = cur_dist.log_prob(actions_flat.view(-1)).unsqueeze(-1)
            else:
                cur_log_probs = cur_dist.log_prob(actions_flat).unsqueeze(-1)
            cur_log_probs = T.nan_to_num(cur_log_probs, nan=0.0, posinf=20.0, neginf=-20.0)
            
            cur_values = self.value(states_flat, goals_flat).reshape(traj_len, num_envs)
            cur_next_values = self.value(next_states_flat, goals_flat).reshape(traj_len, num_envs)

        # Calculate advantages and returns
        advantages, returns, td_errors = compute_advantages_and_returns(
            rewards,
            cur_values,
            cur_next_values,
            terminations,
            truncations,
            self.discount,
            self.gae_coefficient,
            self.bootstrap_truncations,
            self.device
        )

        # Filter phantom steps
        valid_idx = valid_indices.squeeze(-1)
        num_valid = valid_idx.numel()
        states_flat = states_flat[valid_idx]
        next_states_flat = next_states_flat[valid_idx]
        actions_flat = actions_flat[valid_idx]
        goals_flat = goals_flat[valid_idx] if goals is not None else None
        cur_log_probs = cur_log_probs[valid_idx]
        cur_values_flat = cur_values.reshape(total_samples, 1)[valid_idx]
        advantages_flat = advantages.reshape(total_samples, 1)[valid_idx]
        returns_flat = returns.reshape(total_samples, 1)[valid_idx]

        # Normalize advantages
        if self.advantage_normalizer:
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(advantages_flat)
            advantages_flat = self.advantage_normalizer.normalize(advantages_flat)

        # Training loop
        for epoch in range(learning_epochs):
            # Create random indices for shuffling
            indices = T.randperm(num_valid, device=self.device)
            num_batches = num_valid // mini_batch_size

            for batch_num in range(num_batches):
                batch_indices = indices[batch_num * mini_batch_size : (batch_num + 1) * mini_batch_size]
                # print("batch_indices:", batch_indices)
                states_batch = states_flat[batch_indices]
                goals_batch = goals_flat[batch_indices] if goals is not None else None
                actions_batch = actions_flat[batch_indices]
                cur_log_probs_batch = cur_log_probs[batch_indices].detach()
                cur_values_batch = cur_values_flat[batch_indices].detach()
                advantages_batch = advantages_flat[batch_indices].detach()
                returns_batch = returns_flat[batch_indices].detach()

                ## POLICY ##
                # Create new distribution
                new_dist = self.policy(states_batch, goals_batch)
                if self.policy.distribution == 'categorical':
                    new_log_probs = new_dist.log_prob(actions_batch.view(-1)).unsqueeze(-1)
                else: # Continuous Distributions
                    new_log_probs = new_dist.log_prob(actions_batch).unsqueeze(-1)
                    new_log_probs = T.nan_to_num(new_log_probs, nan=0.0, posinf=20.0, neginf=-20.0)

                # prob_ratio = T.exp(new_log_probs - cur_log_probs_batch)
                log_ratio = new_log_probs - cur_log_probs_batch
                log_ratio = T.clamp(log_ratio, min=-10.0, max=10.0)
                prob_ratio = T.exp(log_ratio)

                # Calculate Surrogate Loss
                surr1 = prob_ratio * advantages_batch
                surr2 = T.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantages_batch
                surrogate_loss = -T.min(surr1, surr2).mean()

                # Calculate Entropy penalty
                entropies = new_dist.entropy()
                mean_entropy = entropies.mean()
                entropy_penalty = mean_entropy * -entropy_coefficient

                # Calculate the KL penalty
                with T.no_grad():
                    kl = prob_ratio - 1 - log_ratio
                    mean_kl = kl.mean()
                kl_penalty = mean_kl * kl_coefficient

                # Calculate policy loss
                policy_loss = surrogate_loss + entropy_penalty + kl_penalty

                ## VALUE ##
                values = self.value(states_batch, goals_batch)
                loss = (values - returns_batch).pow(2)
                clipped_values = cur_values_batch + (values - cur_values_batch).clamp(-value_clip, value_clip)
                clipped_value_loss = (clipped_values - returns_batch).pow(2)
                value_loss = self.value_coef * (0.5 * T.max(loss, clipped_value_loss).mean())

                # Calculate gradients
                self.policy.optimizer.zero_grad()
                policy_loss.backward()
                if self.policy_grad_clip:
                    policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.policy_grad_clip)

                self.value.optimizer.zero_grad()
                value_loss.backward()
                if self.value_grad_clip:
                    value_grad_norm = T.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.value_grad_clip)

                # Log diag data
                if should_log_diag and (epoch == learning_epochs - 1) and (batch_num == num_batches - 1):
                    self.logger.debug(
                        "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                        step,
                        self._learn_count,
                        summarize_tensor(states_batch, "states batch"),
                        summarize_tensor(actions_batch, "actions batch"),
                        summarize_tensor(rewards, "rewards"),
                        summarize_tensor(next_states, "next_states"),
                        summarize_tensor(goals_batch, "goals batch"),
                        summarize_tensor(next_ach_goals, "next_ach_goals"),
                        summarize_tensor(terminations, "terminations"),
                        summarize_tensor(truncations, "truncations"),
                        summarize_tensor(cur_values_batch, "values batch"),
                        summarize_tensor(cur_next_values, "next_values"),
                        summarize_tensor(td_errors, "td_errors"),
                        summarize_tensor(advantages_batch, "advantages batch"),
                        summarize_tensor(returns_batch, "returns batch"),
                        summarize_tensor(cur_log_probs_batch, "log_probs batch"),
                        summarize_tensor(new_log_probs, "new_log_probs batch"),
                        summarize_tensor(prob_ratio, "prob_ratio batch"),
                        summarize_tensor(entropies, "entropies batch"),
                        summarize_tensor(kl, "kl batch"),
                        summarize_tensor(surr1, "surr1"),
                        summarize_tensor(surr2, "surr2"),
                        summarize_tensor(surrogate_loss, "surrogate_loss"),
                        summarize_tensor(policy_loss, "policy_loss"),
                        summarize_tensor(value_loss, "value_loss"),
                        f"entropy_coef={float(entropy_coefficient)}",
                    )

                    self.logger.debug(
                        "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                        "value_loss=%.6f policy_loss=%.6f",
                        step,
                        self._learn_count,
                        float(value_grad_norm) if value_grad_norm is not None else -1.0,
                        float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                        float(value_loss.item()),
                        float(policy_loss.item()),
                    )

                # Update models
                self.policy.optimizer.step()
                self.value.optimizer.step()

            
            if self.kl_adapter and mean_kl > self.kl_adapter.target_kl * 1.5:
                break  # Stop this learn cycle's epochs early

        # Step schedulers/adapters
        if self.kl_adapter:
            self.kl_adapter.step(mean_kl)

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        value_learning_rate = self.value.optimizer.param_groups[0]['lr']

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': mean_entropy.item(),
            'kl': mean_kl.item(),
            'prob_ratio': prob_ratio.detach().cpu().flatten().mean().item(),
            'temporal_difference': td_errors.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'advantages': advantages.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'returns': returns.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'policy_clip': policy_clip,
            'value_clip': value_clip,
            'entropy_coefficient': entropy_coefficient,
            'kl_coefficient': kl_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate
        })
        return learn_metrics

    def get_config(self):
        """
        Get the current configuration of the PPO agent.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config.update({
            "policy": self.policy.get_config(),
            "value": self.value.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "gae_coefficient": self.gae_coefficient,
            "policy_clip": self.policy_clip,
            "policy_clip_schedule": self.policy_clip_schedule.get_config() if self.policy_clip_schedule else None,
            "policy_grad_clip": self.policy_grad_clip,
            "value_clip": self.value_clip,
            "value_clip_schedule": self.value_clip_schedule.get_config() if self.value_clip_schedule else None,
            "value_grad_clip": self.value_grad_clip,
            "value_coef": self.value_coef,
            "reward_clip": self.reward_clip,
            "kl_coefficient": self.kl_coefficient,
            "kl_adapter": self.kl_adapter.get_config() if self.kl_adapter else None,
            "curiosity": self.curiosity.get_config() if self.curiosity else None,
            "bootstrap_truncations": self.bootstrap_truncations
        })
        return config

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
        self.policy.save(self.save_dir)
        self.value.save(self.save_dir)
        if self.curiosity:
            self.curiosity.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")
        if self.advantage_normalizer:
            self.advantage_normalizer.save(self.save_dir + "advantage_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

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
        env_wrapper = EnvWrapper.from_json(config["policy"]["env"])
        distribution = config['policy']['distribution']
        if distribution == 'categorical':
            policy = StochasticDiscretePolicy.load(Path(config_dir) / 'policy', load_weights, env=env_wrapper)
        elif distribution in ['beta', 'normal']:
            policy = StochasticContinuousPolicy.load(Path(config_dir) / 'policy', load_weights, env=env_wrapper)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        value = ValueModel.load(Path(config_dir) / 'value', load_weights, env=env_wrapper)
        curiosity = ICM.load(config["save_dir"], env=env_wrapper) if config["curiosity"] else None
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = BaseNormalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        advantage_normalizer = BaseNormalizer.load(config["advantage_normalizer"], config["save_dir"] + "advantage_normalizer.pt") if config["advantage_normalizer"] else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None
        agent = cls(
            policy = policy,
            value = value,
            discount=config["discount"],
            gae_coefficient = config["gae_coefficient"],
            state_normalizer = state_normalizer,
            goal_normalizer = goal_normalizer,
            advantage_normalizer = advantage_normalizer,
            reward_normalizer = reward_normalizer,
            entropy_coefficient = config["entropy_coefficient"],
            entropy_schedule = ScheduleWrapper(**config["entropy_schedule"]) if config.get("entropy_schedule", None) else None,
            auto_entropy_tuning = config["auto_entropy_tuning"],
            entropy_lr = config["entropy_lr"],
            target_entropy_scale = config["target_entropy_scale"],
            kl_coefficient = config["kl_coefficient"],
            kl_adapter = AdaptiveKL(**config["kl_adapter"]) if config.get("kl_adapter", None) else None,
            policy_clip = config["policy_clip"],
            policy_clip_schedule = ScheduleWrapper(**config["policy_clip_schedule"]) if config.get("policy_clip_schedule", None) else None,
            policy_grad_clip = config["policy_grad_clip"],
            value_clip = config["value_clip"],
            value_clip_schedule = ScheduleWrapper(**config["value_clip_schedule"]) if config.get("value_clip_schedule", None) else None,
            value_grad_clip = config["value_grad_clip"],
            value_coef = config["value_coef"],
            reward_clip = config['reward_clip'],
            curiosity = curiosity,
            bootstrap_truncations = config["bootstrap_truncations"],
            save_dir=config["save_dir"],
            device=config["device"],
        )

        return agent

class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        policy: ActorModel,
        critic: ContinuousCritic,
        *,
        discount: float=0.99,
        tau: float=0.001,
        action_epsilon: float = 0.2,
        state_normalizer: BaseNormalizer | None = None,
        goal_normalizer: BaseNormalizer | None = None,
        reward_normalizer: RewardNorm | None = None,
        noise: Noise | None = None,
        noise_schedule: ScheduleWrapper | None = None,
        noise_clip: float = 0.5,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        N: int=1, # N-steps
        intrinsic_motivation: IntrinsicMotivation | None = None,
        save_dir: str = "models",
        device: str | T.device | None = None,
        **kwargs: Any,
    ):
        try:
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.critic = critic
            self.discount = discount
            self.tau = tau
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.reward_normalizer = reward_normalizer
            self.policy_grad_clip = policy_grad_clip
            self.critic_grad_clip = critic_grad_clip
            self.N = N
            self.intrinsic_motivation = intrinsic_motivation
            self.target_policy = self.policy.clone(device=self.policy.device)
            self.target_critic = self.critic.clone(device=self.critic.device)
            self.action_epsilon = action_epsilon
            self.noise = noise
            self.noise_schedule = noise_schedule
            self.noise_clip = noise_clip

            self._use_her = False

        except Exception as e:
            self.logger.error(f"Error in DDPG init: {e}", exc_info=True)

    def _init_her(self):
        self._use_her = True

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None
    ) -> T.Tensor | np.ndarray:
        """
        Select an action based on the current policy.

        Args:
            states: T.Tensor | np.ndarray: The current states.
            goals: T.Tensor | np.ndarray | None: The current goals.
            context: str: The context of the action (train, test).
            step: int | None: The current step.
            warmup: int | None: The warmup steps.
        
        Returns:
            T.Tensor | np.ndarray: actions.
        """
        
        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                return self.policy.env.action_space.sample()
            # if random number is less than epsilon, sample random action
            if np.random.random() < self.action_epsilon:
                return self.policy.env.action_space.sample()
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.policy.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    _, actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

                return actions.detach()

        else: # context == 'test'
            with T.no_grad():
                _, actions = self.target_policy(states, goals)
            return actions.detach()

    def soft_update_targets(self):
        soft_update(self.policy, self.target_policy, self.tau)
        soft_update(self.critic, self.target_critic, self.tau)

    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            ach_goals = self.goal_normalizer.normalize(ach_goals)
            next_ach_goals = self.goal_normalizer.normalize(next_ach_goals)
            goals = self.goal_normalizer.normalize(goals)
        if self.reward_normalizer:
            rewards = self.reward_normalizer.normalize(rewards)

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = rewards.shape

        # Train Intrinsic Motivation and get intrinsic rewards
        if self.intrinsic_motivation:
            # Reshape arrays to (batch_size * N, -1) to train on all steps in N
            states_flat = states.reshape(-1, states.shape[-1])
            next_states_flat = next_states.reshape(-1, next_states.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            im_loss = self.intrinsic_motivation.train(states_flat, next_states_flat, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                states_flat,
                next_states_flat,
                actions_flat
            )
            im_learn_rewards = im_learn_rewards.reshape(batch_size, n_step_length)
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards += im_rewards
            else:
                rewards = im_rewards

        # Get target values
        with T.no_grad():
            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic.device
            ).squeeze()

            _, target_actions = self.target_policy(
                next_states[:,-1,:],
                goals[:,-1,:] if goals is not None else None
            )

            target_critic_values = self.target_critic(
                next_states[:,-1,:],
                target_actions,
                goals[:,-1,:] if goals is not None else None
            ).squeeze()

            no_dones_mask = (terminations.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values

            # Apply HER-specific clamping if needed
            if self._use_her:
                if self.intrinsic_motivation and not self.intrinsic_motivation._use_extrinsic:
                    pass
                else:
                    targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions = self.critic(
            states[:,0,:],
            actions[:,0,:],
            goals[:,0,:] if goals is not None else None
        ).squeeze()

        # Calculate TD errors
        error = targets - predictions
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss = (weights.to(self.critic.device) * error.pow(2)).mean()
        else:
            critic_loss = error.pow(2).mean()

        # Update critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip value gradient
        critic_grad_norm = T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip)
        self.critic.optimizer.step()

        # Get actor's action predictions
        _, pred_actions = self.policy(
            states[:,0,:],
            goals[:,0,:] if goals is not None else None
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic(
            states[:,0,:],
            pred_actions,
            goals[:,0,:] if goals is not None else None
        )

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

        # Clip policy gradient
        policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.policy_grad_clip)
        self.policy.optimizer.step()

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_critic_values, "target critic values"),
                summarize_tensor(targets, "targets"),
                summarize_tensor(predictions, "critic predictions"),
                summarize_tensor(error, "critic errors"),
                summarize_tensor(pred_actions, "predicted actions"),
                summarize_tensor(critic_values, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_grad_norm) if critic_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        critic_learning_rate = self.critic.optimizer.param_groups[0]['lr']
        
        learn_metrics.update({
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "critic_error": error.mean().item(),
            "policy_predictions": pred_actions.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_policy_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate
        })

        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })
        
        if self.noise_schedule:
            learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})

        return learn_metrics

    def get_config(self):
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "policy": self.policy.get_config(),
            "critic": self.critic.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
            "action_epsilon": self.action_epsilon,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "noise_clip": self.noise_clip
        })
        return config


    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.critic.save(self.save_dir)
        if self.intrinsic_motivation:
            self.intrinsic_motivation.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool = True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["policy"]["env"])
        policy = ActorModel.load(Path(config_dir) / 'policy', load_weights, env=env_wrapper)
        critic = ContinuousCritic.load(Path(config_dir) / 'critic', load_weights, env=env_wrapper)
        noise = Noise.create_instance(config["noise"]["type"], **config["noise"]["config"])
        intrinsic_motivation = IntrinsicMotivation.load(config_dir, env=env_wrapper) if config["intrinsic_motivation"] else None
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = BaseNormalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None

        agent = cls(
            policy = policy,
            critic = critic,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            reward_normalizer=reward_normalizer,
            noise=noise,
            noise_schedule=ScheduleWrapper(**config["noise_schedule"]) if config.get("noise_schedule", None) else None,
            noise_clip=config["noise_clip"],
            policy_grad_clip=config['policy_grad_clip'],
            critic_grad_clip=config['critic_grad_clip'],
            N = config['N'],
            intrinsic_motivation=intrinsic_motivation,
            save_dir=config["save_dir"],
            device=config["device"]
        )

        return agent
    

class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""
    
    def __init__(
        self,
        policy: ActorModel,
        critic: ContinuousCritic,
        critic_b: ContinuousCritic|None = None,
        *,
        discount: float = 0.99,
        tau: float = 0.005,
        action_epsilon: float = 0.0,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        noise: Noise|None = None,
        noise_schedule: ScheduleWrapper|None = None,
        target_noise: Noise|None = None,
        target_noise_schedule: ScheduleWrapper|None = None,
        noise_clip: float = 0.5,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        policy_update_delay: int = 2,
        N: int=1, # N-steps
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
        try:
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.critic = critic
            self.discount = discount
            self.tau = tau
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.reward_normalizer = reward_normalizer
            self.policy_grad_clip = policy_grad_clip
            self.critic_grad_clip = critic_grad_clip
            self.N = N
            self.intrinsic_motivation = intrinsic_motivation
            self.critic_b = critic_b
            # clone second critic (do not copy weights) if critic_b None
            if not critic_b:
                self.critic_b = self.critic.clone(copy_weights=False, device=self.critic.device)
            self.target_policy = self.policy.clone(device=self.policy.device)
            self.target_critic = self.critic.clone(device=self.critic.device)
            self.target_critic_b = self.critic_b.clone(device=self.critic_b.device)
            self.action_epsilon = action_epsilon
            self.noise = noise
            self.noise_schedule = noise_schedule
            self.noise_clip = noise_clip
            self.target_noise = target_noise
            self.target_noise_schedule = target_noise_schedule
            self.policy_update_delay = policy_update_delay

            self._use_her = False

        except Exception as e:
            self.logger.error(f"Error in TD3 init: {e}", exc_info=True)

    def _init_her(self):
        self._use_her = True

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None
    ) -> T.Tensor | np.ndarray:
        """
        Select an action based on the current policy.

        Args:
            states: T.Tensor | np.ndarray: The current states.
            goals: T.Tensor | np.ndarray | None: The current goals.
            context: str: The context of the action (train, test).
            step: int | None: The current step.
            warmup: int | None: The warmup steps.
        
        Returns:
            T.Tensor | np.ndarray: actions.
        """
        
        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                return self.policy.env.action_space.sample()
            # if random number is less than epsilon, sample random action
            if np.random.random() < self.action_epsilon:
                return self.policy.env.action_space.sample()
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.policy.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    _, actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

                return actions.detach()

        else: # context == 'test'
            with T.no_grad():
                _, actions = self.target_policy(states, goals)
            return actions.detach()

    def soft_update_targets(self):
        soft_update(self.policy, self.target_policy, self.tau)
        soft_update(self.critic, self.target_critic, self.tau)
        soft_update(self.critic_b, self.target_critic_b, self.tau)
            
    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            ach_goals = self.goal_normalizer.normalize(ach_goals)
            next_ach_goals = self.goal_normalizer.normalize(next_ach_goals)
            goals = self.goal_normalizer.normalize(goals)
        if self.reward_normalizer:
            rewards = self.reward_normalizer.normalize(rewards)

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = rewards.shape

        # Train Intrinsic Motivation and get intrinsic rewards
        if self.intrinsic_motivation:
            # Reshape arrays to (batch_size * N, -1) to train on all steps in N
            states_flat = states.reshape(-1, states.shape[-1])
            next_states_flat = next_states.reshape(-1, next_states.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            im_loss = self.intrinsic_motivation.train(states_flat, next_states_flat, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                states_flat,
                next_states_flat,
                actions_flat
            )
            im_learn_rewards = im_learn_rewards.reshape(batch_size, n_step_length)
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards += im_rewards
            else:
                rewards = im_rewards

        # Get target values
        with T.no_grad():
            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic.device
            ).squeeze()

            _, target_actions = self.target_policy(
                next_states[:,-1,:],
                goals[:,-1,:] if goals is not None else None,
            )

            noise = self.target_noise(target_actions.shape)
            # Apply noise clipping if needed
            if self.noise_clip > 0:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            # Apply noise schedule if needed
            if self.target_noise_schedule is not None:
                noise *= self.target_noise_schedule.get_factor()
                learn_metrics.update({'target_noise_anneal': self.target_noise_schedule.get_factor()})   
                
            # Add noise to target actions and clamp to env action space
            target_actions = target_actions + noise
            target_actions = target_actions.clamp(self.policy.act_space_low, self.policy.act_space_high)
            
            target_critic_values_a = self.target_critic(
                next_states[:,-1,:],
                target_actions,
                goals[:,-1,:] if goals is not None else None
            ).squeeze()
            
            target_critic_values_b = self.target_critic_b(
                next_states[:,-1,:],
                target_actions,
                goals[:,-1,:] if goals is not None else None
            ).squeeze()
            
            target_critic_values = T.minimum(target_critic_values_a, target_critic_values_b)
            no_dones_mask = (terminations.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values
            
            # Apply HER-specific clamping if needed
            if self._use_her:
                if self.curiosity and not self.curiosity._use_extrinsic:
                    pass
                else:
                    targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions_a = self.critic(
            states[:,0,:],
            actions[:,0,:],
            goals[:,0,:] if goals is not None else None
        ).squeeze()

        predictions_b = self.critic_b(
            states[:,0,:],
            actions[:,0,:],
            goals[:,0,:] if goals is not None else None
        ).squeeze()

        # Calculate TD errors (use average of both critic networks for PER)
        error_a = targets - predictions_a
        error_b = targets - predictions_b
        # error = (error_a.abs() + error_b.abs()) / 2  # Average of absolute errors for priorities
        error = T.minimum(error_a, error_b)

        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss_a = (weights.to(self.critic.device) * error_a.pow(2)).mean()
            critic_loss_b = (weights.to(self.critic_b.device) * error_b.pow(2)).mean()
            critic_loss = critic_loss_a + critic_loss_b
        else:
            critic_loss_a = error_a.pow(2).mean()
            critic_loss_b = error_b.pow(2).mean()
            critic_loss = critic_loss_a + critic_loss_b

        # Update critics
        self.critic.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        critic_loss_a.backward()
        critic_loss_b.backward()

        
        # Clip value gradient
        critic_a_grad_norm = T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip)
        critic_b_grad_norm = T.nn.utils.clip_grad_norm_(self.critic_b.parameters(), max_norm=self.critic_grad_clip)
        self.critic.optimizer.step()
        self.critic_b.optimizer.step()
        
        # Get actor's action predictions
        _, pred_actions = self.policy(
            states[:,0,:],
            goals[:,0,:] if goals is not None else None
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic(
            states[:,0,:],
            pred_actions,
            goals[:,0,:] if goals is not None else None
        )
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            actor_loss = -(weights.to(self.policy.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()
        
        # Add HER-specific regularization if using HER
        if self._use_her:
            actor_loss += raw_actions.pow(2).mean()

        
        # Update actor
        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        # Clip policy gradient
        policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.policy_grad_clip)
        if self._learn_count % self.policy_update_delay == 0:
            self.policy.optimizer.step()

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_critic_values_a, "target critic values A"),
                summarize_tensor(target_critic_values_b, "target critic values B"),
                summarize_tensor(target_critic_values, "target critic values"),
                summarize_tensor(targets, "targets"),
                summarize_tensor(predictions_a, "critic predictions A"),
                summarize_tensor(predictions_b, "critic predictions B"),
                summarize_tensor(error_a, "critic errors A"),
                summarize_tensor(error_b, "critic errors B"),
                summarize_tensor(error, "critic errors"),
                summarize_tensor(pred_actions, "predicted actions"),
                summarize_tensor(critic_values, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_a_grad_norm=%.6f critic_b_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_a_grad_norm) if critic_a_grad_norm is not None else -1.0,
                float(critic_b_grad_norm) if critic_b_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        critic_learning_rate = self.critic.optimizer.param_groups[0]['lr']
        critic_b_learning_rate = self.critic_b.optimizer.param_groups[0]['lr']

        # Add metrics to step_logs
        learn_metrics.update({
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "critic_error": error.mean().item(),
            "policy_predictions": pred_actions.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_policy_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate,
            'critic_b_learning_rate': critic_b_learning_rate,
        })

        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })

        if self.noise_schedule:
            learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})
        
        return learn_metrics

    def get_config(self):
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "policy": self.policy.get_config(),
            "critic": self.critic.get_config(),
            "critic_b": self.critic_b.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
            "action_epsilon": self.action_epsilon,
            "critic_b": self.critic_b.get_config(),
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "target_noise": self.target_noise.get_config() if self.target_noise is not None else None,
            "target_noise_schedule": self.target_noise_schedule.get_config() if self.target_noise_schedule is not None else None,
            "policy_update_delay": self.policy_update_delay,
        })
        return config

    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.critic.save(self.save_dir, 'critic')
        self.critic_b.save(self.save_dir, 'critic_b')
        if self.intrinsic_motivation:
            self.intrinsic_motivation.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool=True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["policy"]["env"])
        policy = ActorModel.load(config_dir, 'policy', load_weights, env=env_wrapper)
        critic = ContinuousCritic.load(config_dir, 'critic', load_weights, env=env_wrapper)
        critic_b = ContinuousCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        # load intrinsic motivation
        intrinsic_motivation = IntrinsicMotivation.load(config_dir, env=env_wrapper) if config["intrinsic_motivation"] else None
        # load state normalizer
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = BaseNormalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None
        noise = Noise.create_instance(config["noise"]["type"], **config["noise"]["config"])
        target_noise = Noise.create_instance(config["target_noise"]["type"], **config["target_noise"]["config"])

        agent = cls(
            policy=policy,
            critic=critic,
            critic_b=critic_b,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            reward_normalizer=reward_normalizer,
            noise=noise,
            noise_schedule=ScheduleWrapper(**config["noise_schedule"]) if config.get("noise_schedule", None) else None,
            target_noise=target_noise,
            target_noise_schedule=ScheduleWrapper(**config["target_noise_schedule"]) if config.get("target_noise_schedule", None) else None,
            noise_clip=config["noise_clip"],
            policy_grad_clip=config["policy_grad_clip"],
            critic_grad_clip=config["critic_grad_clip"],
            policy_update_delay=config["policy_update_delay"],
            N=config["N"],
            intrinsic_motivation=intrinsic_motivation,
            save_dir=config["save_dir"],
            device=config["device"],
        )
        return agent

class SAC(Agent):
    """Soft Actor Critic Agent."""

    def __init__(
        self,
        policy: StochasticDiscretePolicy|StochasticContinuousPolicy,
        critic: ContinuousCritic|DiscreteCritic,
        critic_b: ContinuousCritic|DiscreteCritic|None = None,
        *,
        discount: float=0.99,
        tau: float=0.005,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float=0.2, # Auto set to 1.0 if auto-tuning
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        N: int=1,
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
        try:
            super().__init__(save_dir, device, **kwargs)
            self.policy = policy
            self.critic = critic
            self.discount = discount
            self.tau = tau
            self.state_normalizer = state_normalizer
            self.goal_normalizer = goal_normalizer
            self.reward_normalizer = reward_normalizer
            self.policy_grad_clip = policy_grad_clip
            self.critic_grad_clip = critic_grad_clip
            self.N = N
            self.intrinsic_motivation = intrinsic_motivation
            self.critic_b = critic_b
            # clone second critic (do not copy weights) if critic_model_b None
            if not critic_b:
                self.critic_b = self.critic.clone(copy_weights=False, device=self.critic.device)
            self.target_critic = self.critic.clone(device=self.critic.device)
            self.target_critic_b = self.critic_b.clone(device=self.critic_b.device)
            self.entropy_coefficient = entropy_coefficient
            self.entropy_schedule = entropy_schedule
            self.auto_entropy_tuning = auto_entropy_tuning
            self.entropy_lr = entropy_lr
            self.target_entropy_scale = target_entropy_scale
            if self.auto_entropy_tuning:
                self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                    self.policy,
                    target_entropy_scale=target_entropy_scale,
                    lr=entropy_lr,
                    device=self.device,
                )
        except Exception as e:
            self.logger.error(f"Error in SAC init: {e}", exc_info=True)

        # set internal attributes
        try:
            # Instantiate internal attribute use_her to be switched by HER class if using DDPG
            self._use_her = False

        except Exception as e:
            self.logger.error(f"Error in DDPG init internal attributes: {e}", exc_info=True)

    def _init_her(self):
            self._use_her = True

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None
    ) -> Action:
        """
        Select an action based on the current policy.

        Args:
            states: T.Tensor | np.ndarray: The current states.
            goals: T.Tensor | np.ndarray | None: The current goals.
            context: str: The context of the action (train, test).
            step: int | None: The current step.
            warmup: int | None: The warmup steps.

        Returns:
            Action: actions.
        """

        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = self.policy.env.action_space.sample()
                actions = T.tensor(actions, device=self.device)
                log_probs = T.log(T.ones_like(actions) * (1/self.policy.num_actions))
                if log_probs.ndim > 1:
                    log_probs = log_probs.sum(-1)
            # otherwise, sample action from policy
            else:
                with T.no_grad():
                    dist = self.policy(states, goals)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

        elif context == 'test':
            with T.no_grad():
                dist = self.policy(states, goals)
                actions = self.policy.get_mean_actions(dist)
                log_probs = dist.log_prob(actions)

        else:
            raise ValueError(f"Invalid context: {context}")

        return Action(actions, log_probs=log_probs)

    def soft_update_targets(self):
        soft_update(self.critic, self.target_critic, self.tau)
        soft_update(self.critic_b, self.target_critic_b, self.tau)

    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        buf_log_probs = sample["log_probs"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            # Apply scheduling to entropy coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, -1) to train on all steps in N
        states_flat = states.reshape(-1, states.shape[-1])
        next_states_flat = next_states.reshape(-1, next_states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])
            ach_goals_flat = ach_goals.reshape(-1, ach_goals.shape[-1])
            next_ach_goals_flat = next_ach_goals.reshape(-1, next_ach_goals.shape[-1])
        
        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            ach_goals_flat = self.goal_normalizer.normalize(ach_goals_flat)
            next_ach_goals_flat = self.goal_normalizer.normalize(next_ach_goals_flat)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)
        
        # Train Intrinsic Motivation and get intrinsic rewards
        if self.intrinsic_motivation:
            
            im_loss = self.intrinsic_motivation.train(states_flat, next_states_flat, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                states_flat,
                next_states_flat,
                actions_flat
            )
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards.reshape(batch_size, n_step_length) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        with T.no_grad():
            # q_targets = compute_n_step_return(
            #     rewards,
            #     self.discount,
            #     device=self.target_critic.device
            # ).squeeze()

            # Get current policy for sampled states
            cur_dist = self.policy(
                states_flat,
                goals_flat if goals is not None else None
            )

            # Get current values of sampled states and log probs of taking the sampled actions
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                cur_log_probs = cur_dist.log_prob(actions_flat).reshape(batch_size, n_step_length)
                q_cur = T.minimum(
                    self.critic(
                        states_flat,
                        actions_flat,
                        goals_flat if goals is not None else None
                        ),
                    self.critic_b(
                        states_flat,
                        actions_flat,
                        goals_flat if goals is not None else None
                        )
                ).reshape(batch_size, n_step_length)

            else: # Discrete action space
                cur_log_probs = cur_dist.logits.gather(1, actions_flat.long()).reshape(batch_size, n_step_length)
                q_cur_all = T.minimum(
                    self.critic(
                        states_flat,
                        goals_flat if goals is not None else None
                        ),
                    self.critic_b(
                        states_flat,
                        goals_flat if goals is not None else None
                        )
                )
                q_cur = q_cur_all.gather(1, actions_flat.long()).reshape(batch_size, n_step_length)

            ## Critic Update ##
            next_dist = self.policy(
                next_states_flat,
                goals_flat if goals is not None else None
            )

            # Continuous critic target values
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                target_actions = next_dist.sample()
                next_log_probs = next_dist.log_prob(target_actions)
                q_next = T.minimum(
                    self.target_critic(
                        next_states_flat,
                        target_actions,
                        goals_flat if goals is not None else None
                        ),
                    self.target_critic_b(
                        next_states_flat,
                        target_actions,
                        goals_flat if goals is not None else None
                        )
                ).squeeze(-1)
                target_q = (q_next - entropy_coefficient * next_log_probs).reshape(batch_size, n_step_length)

            else: # Discrete critic target values
                target_actions = next_dist.sample().float()
                next_log_probs = next_dist.logits
                q_next = T.minimum(
                    self.target_critic(
                        next_states_flat,
                        goals_flat if goals is not None else None
                    ),
                    self.target_critic_b(
                        next_states_flat,
                        goals_flat if goals is not None else None
                    )
                )
                target_q = (next_dist.probs * (q_next - entropy_coefficient * next_log_probs)).sum(-1).reshape(batch_size, n_step_length)

            # Compute TD errors across n-step window
            td_errors = rewards + self.discount * (1 - terminations.float()) * target_q.detach() - q_cur.detach()

            # Compute IS ratios
            is_ratio = T.clamp(T.exp(cur_log_probs - buf_log_probs), max=1.0)
            # Mask IS ratios from terminated_state +1 : N
            mask = T.ones(batch_size, n_step_length, device=self.device)
            dones = T.logical_or(terminations, truncations)
            for k in range(1, n_step_length):
                mask[:, k] = mask[:, k-1] * (1 - dones[:, k-1].float())
            is_ratio = is_ratio * mask

            # Compute q retrace
            cum_c = T.ones(batch_size, device=self.device)
            retrace_sum = T.zeros(batch_size, device=self.device)

            for k in range(n_step_length):
                gamma = self.discount ** k
                retrace_sum += gamma * cum_c * td_errors[:, k]
                # Update cumulative weight IS ratio
                if k < n_step_length - 1:
                    cum_c = cum_c * is_ratio[:, k+1]

            q_retrace = q_cur[:, 0] + retrace_sum
            
            # no_dones_mask = (terminations.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            # gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            # q_targets += no_dones_mask * gamma_pow * target_values

            # Apply HER-specific clamping if needed
            # if self._use_her:
            #     if self.curiosity and not self.curiosity._use_extrinsic:
            #         pass
            #     else:
            #         q_targets = T.clamp(q_targets, min=-1/(1-self.discount))

        # Continuous critic predictions
        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            q1_preds = self.critic(
                states[:,0,:],
                actions[:,0,:],
                goals[:,0,:] if goals is not None else None).squeeze()
            q2_preds = self.critic_b(
                states[:,0,:],
                actions[:,0,:],
                goals[:,0,:] if goals is not None else None).squeeze()

        else: # Discrete critic predictions
            q1 = self.critic(
                states[:,0,:],
                goals[:,0,:] if goals is not None else None
            )
            q2 = self.critic_b(
                states[:,0,:],
                goals[:,0,:] if goals is not None else None
            )
            buffer_actions = actions[:,0,:].squeeze(-1).long().unsqueeze(1)
            q1_preds = q1.gather(1, buffer_actions).squeeze(1)
            q2_preds = q2.gather(1, buffer_actions).squeeze(1)

        # Calculate errors
        q1_loss = (q1_preds - q_retrace.detach()).pow(2)
        q2_loss = (q2_preds - q_retrace.detach()).pow(2)
        # Get min error across losses (used to update priorities)
        errors = (T.minimum(q1_preds, q2_preds) - q_retrace).detach().flatten()
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            q1_loss = weights.to(self.critic.device) * q1_loss
            q2_loss = weights.to(self.critic_b.device) * q2_loss
        critic_loss = 0.5 * (q1_loss.mean() + q2_loss.mean())

        self.critic.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        critic_loss.backward()
        # if self.grad_clip:
        critic_a_grad_norm = T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        critic_b_grad_norm = T.nn.utils.clip_grad_norm_(self.critic_b.parameters(), self.critic_grad_clip)
        self.critic.optimizer.step()
        self.critic_b.optimizer.step()

        ## Update Policy ##
        dist = self.policy(
            states[:,0,:],
            goals[:,0,:] if goals is not None else None
        )
        # Continuous policy update
        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            new_actions = dist.rsample()
            log_probs = dist.log_prob(new_actions)
            q1 = self.critic(states[:,0,:], new_actions, goals[:,0,:] if goals is not None else None).squeeze()
            q2 = self.critic_b(states[:,0,:], new_actions, goals[:,0,:] if goals is not None else None).squeeze()
            min_q = T.minimum(q1, q2)
            actor_loss = entropy_coefficient * log_probs - min_q

        else: # Discrete policy update
            new_actions = dist.sample().float()
            log_probs = dist.logits
            q1 = self.critic(states[:,0,:], goals[:,0,:] if goals is not None else None)
            q2 = self.critic_b(states[:,0,:], goals[:,0,:] if goals is not None else None)
            min_q = T.minimum(q1, q2)
            actor_loss = (dist.probs * (entropy_coefficient * log_probs - min_q)).sum(-1)


        if weights is not None:
            actor_loss = weights.to(self.policy.device) * actor_loss

        actor_loss = actor_loss.mean()

        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        # if self.grad_clip:
        policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_grad_clip)
        self.policy.optimizer.step()

        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            entropy = -log_probs
        else:  # Discrete actor
            entropy = -(dist.probs * log_probs).sum(dim=-1)
        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (-entropy.mean() + self.target_entropy).detach())
            alpha_loss.backward()
            self.entropy_optimizer.step()

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_q, "target critic values"),
                summarize_tensor(q1_preds, "critic predictions A"),
                summarize_tensor(q2_preds, "critic predictions B"),
                summarize_tensor(q1_loss, "critic errors A"),
                summarize_tensor(q2_loss, "critic errors B"),
                summarize_tensor(critic_loss, "critic errors"),
                summarize_tensor(new_actions, "predicted actions"),
                summarize_tensor(log_probs, "log probs"),
                summarize_tensor(entropy, "entropy"),
                summarize_tensor(min_q, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_a_grad_norm=%.6f critic_b_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_a_grad_norm) if critic_a_grad_norm is not None else -1.0,
                float(critic_b_grad_norm) if critic_b_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.policy.optimizer.param_groups[0]['lr']
        critic_learning_rate = self.critic.optimizer.param_groups[0]['lr']
        critic_b_learning_rate = self.critic_b.optimizer.param_groups[0]['lr']                                                     ###

        learn_metrics.update({
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": errors,
            "td_error": td_errors.mean().item(),
            "policy_predictions": new_actions.mean().item(),
            "critic_predictions": min_q.mean().item(),
            "target_critic_predictions": target_q.mean().item(),
            "entropy_coefficient": entropy_coefficient,
            "entropy": float(entropy.mean().item()),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate,
            'critic_b_learning_rate': critic_b_learning_rate,
        })
        
        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })

        return learn_metrics

    def get_config(self):

        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "policy": self.policy.get_config(),
            "critic": self.critic.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
            "critic_b": self.critic_b.get_config(),
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
        })
        return config

    def save(self):
        """Saves the model."""
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.policy.save(self.save_dir)
        self.critic.save(self.save_dir, 'critic')
        self.critic_b.save(self.save_dir, 'critic_b')
        if self.intrinsic_motivation:
            self.intrinsic_motivation.save(self.save_dir)
        if self.state_normalizer:
            self.state_normalizer.save(self.save_dir + "state_normalizer.pt")
        if self.goal_normalizer:
            self.goal_normalizer.save(self.save_dir + "goal_normalizer.pt")
        if self.reward_normalizer:
            self.reward_normalizer.save(self.save_dir + "reward_normalizer.pt")

    @classmethod
    def load(cls, config_dir:str | Path, load_weights:bool=True):
        """Loads the model."""
        config = json.load(open(Path(config_dir) / 'config.json'))
        env_wrapper = EnvWrapper.from_json(config["env"])
        distribution = config['policy_model']['distribution']
        if distribution == 'categorical':
            policy = StochasticDiscretePolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
            critic = DiscreteCritic.load(config_dir, 'critic', load_weights, env=env_wrapper)
            critic_b = DiscreteCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        elif distribution in ['beta', 'normal']:
            policy = StochasticContinuousPolicy.load(config_dir, 'policy', load_weights, env=env_wrapper)
            critic = ContinuousCritic.load(config_dir, 'critic', load_weights, env=env_wrapper)
            critic_b = ContinuousCritic.load(config_dir, 'critic_b', load_weights, env=env_wrapper)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        intrinsic_motivation = IntrinsicMotivation.load(config_dir, env=env_wrapper) if config["intrinsic_motivation"] else None
        state_normalizer = BaseNormalizer.load(config["state_normalizer"], config["save_dir"] + "state_normalizer.pt") if config["state_normalizer"] else None
        goal_normalizer = BaseNormalizer.load(config["goal_normalizer"], config["save_dir"] + "goal_normalizer.pt") if config["goal_normalizer"] else None
        reward_normalizer = RewardNorm.load(config["reward_normalizer"], config["save_dir"] + "reward_normalizer.pt") if config["reward_normalizer"] else None

        agent = cls(
            policy = policy,
            critic = critic,
            critic_b = critic_b,
            discount=config["discount"],
            tau=config["tau"],
            state_normalizer=state_normalizer,
            goal_normalizer=goal_normalizer,
            reward_normalizer=reward_normalizer,
            entropy_coefficient=config["entropy_coefficient"],
            entropy_schedule = ScheduleWrapper(**config["entropy_schedule"]) if config.get("entropy_schedule", None) else None,
            auto_entropy_tuning=config["auto_entropy_tuning"],
            entropy_lr=config["entropy_lr"],
            target_entropy_scale=config["target_entropy_scale"],
            policy_grad_clip=config["policy_grad_clip"],
            critic_grad_clip=config["critic_grad_clip"],
            N = config['N'],
            intrinsic_motivation=intrinsic_motivation,
            save_dir=config["save_dir"],
            device=config["device"]
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

    def _step(self,
              step: int,
              trajectories: list[list[tuple]],
              states: dict,
              max_episodes: int,
              episode_scores: np.ndarray,
              completed_episodes: np.ndarray,
              score_history: deque[float],
              best_reward: float,
              success_counter: float,
              training: bool = True):
        """
        Perform a single training step.
        """
        obs, goals = self._preprocess_inputs(states)
        actions = self.base_agent.get_action(obs, goals, step=step, context='train' if training else 'test')
        actions = self.base_agent.env.format_actions(actions)
        next_states, rewards, dones, infos = self.base_agent.env.step(actions)

        # Ensure states, actions, rewards, next_states, and dones are tensors
        obs, actions, rewards, next_states, dones = (
            T.tensor(obs, dtype=T.float32, device=self.device) if isinstance(obs, np.ndarray) else obs,
            T.tensor(actions, dtype=T.float32, device=self.device) if isinstance(actions, np.ndarray) else actions,
            T.tensor(rewards, dtype=T.float32, device=self.device) if isinstance(rewards, np.ndarray) else rewards,
            T.tensor(next_states, dtype=T.float32, device=self.device) if isinstance(next_states, np.ndarray) else next_states,
            T.tensor(dones, dtype=T.int8, device=self.device) if isinstance(dones, np.ndarray) else dones,
        )

        episode_scores += rewards
        step_logs = {f'step_reward': rewards.mean()}
        
        # Store transitions in the env trajectory
        for i in range(self.base_agent.env.num_envs):
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

    def train(self, num_epochs: int, num_cycles: int, num_episodes: int, num_updates: int, render_freq: int = 0, seed: int | None = None):
        """
        Train the HER agent with a vectorized environment setup, following the HER paper's experiment structure.

        Args:
            num_epochs (int): Number of training epochs.
            num_cycles (int): Number of cycles per epoch.
            num_episodes (int): Number of episodes to collect per cycle across all environments.
            num_updates (int): Number of optimization steps per cycle after collecting episodes.
            render_freq (int): Frequency of rendering (in total completed episodes).
            seed (int, optional): Random seed for reproducibility.
        """
        
        init_dict = self._initialize_run(seed, num_episodes=num_episodes, num_epochs=num_epochs, num_cycles=num_cycles, num_updates=num_updates)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        trajectories = [[] for _ in range(self.base_agent.env.num_envs)]
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

                    step_result = self._step(step, trajectories, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, success_counter)
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
                        self.agent.soft_update(self.base_agent.policy, self.base_agent.target_policy)
                        self.agent.soft_update(self.base_agent.critic, self.base_agent.target_critic)
                        self.agent.soft_update(self.base_agent.critic_b, self.base_agent.target_critic_b)
                    elif isinstance(self.base_agent, SAC):
                        self.agent.soft_update(self.base_agent.critic, self.base_agent.target_critic)
                        self.agent.soft_update(self.base_agent.critic_b, self.base_agent.target_critic_b)

                else:
                    learn_logs = None

                if self.base_agent.callbacks:
                    for callback in self.base_agent.callbacks:
                        callback.on_train_epoch_end(epoch=step, logs=learn_logs)

        if self.base_agent.callbacks:
            for callback in self.base_agent.callbacks:
                callback.on_train_end(logs=episode_log)

        self.base_agent.env.close()
    
    def test(self, num_episodes: int, render_freq: int = 0, seed: int | None = None):
        """Runs a test over 'num_episodes'."""
        
        init_dict = self._initialize_run(seed, training=False)
        step = init_dict['step']
        states = init_dict['states']
        episode_scores = init_dict['episode_scores']
        completed_episodes = init_dict['completed_episodes']
        score_history = init_dict['score_history']
        best_reward = init_dict['best_reward']
        trajectories = [[] for _ in range(self.base_agent.env.num_envs)]
        success_counter = 0.0

        while completed_episodes.sum() < num_episodes:
            step += 1
            step_result = self._step(step, trajectories, states, num_episodes, episode_scores, completed_episodes, score_history, best_reward, success_counter, training=False)
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

        self.base_agent.env.close()

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
    
# @runtime_checkable
# class HasNoise(Protocol):
#     def reset_noise(self, env_indices: T.Tensor) -> None: ...

@runtime_checkable
class HasTargetNetworks(Protocol):
    def soft_update_targets(self) -> None: ...


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

# def init_sweep(config):
#     try:
#         # Extract the model type (stored as a list) from the config.
#         model_type_list = config.get("model_type", [])
#         if not model_type_list:
#             raise ValueError("No model type provided in config.")
#         model_type = model_type_list[0]

#         # Inject wandb settings into the config if not already provided.
#         if "wandb" not in config:
#             run_number = wandb_support.get_next_run_number(config["project"])
#             config["wandb"] = {
#                 "project": config["project"],
#                 "name": f"train-{run_number}",
#                 "job_type": "train",
#                 "tags": ["train", model_type],
#                 "group": f"group-{run_number}",
#             }

#         # Build the environment.
#         env_params = {
#             key.replace("env_", ""): config[key]
#             for key in config if key.startswith("env_")
#         }
#         env = gym.make(**env_params)
#         env_spec = env.spec.to_json()
#         logger.debug(f"Environment built: {env.spec}")

#         # Create callbacks (using your custom WandbCallback).
#         callbacks = []
#         callbacks.append(WandbCallback(
#             project_name=config["project"],
#             run_name=config["wandb"]["name"],
#             _sweep=True
#         ))
#         logger.debug("Callbacks created")

#         # Get the appropriate agent class from the model type.
#         agent = get_agent_class_from_type(model_type)
#         logger.debug("Agent class found. Calling sweep_train")

#         # Call the sweep_train function on the agent with the full config.
#         agent.sweep_train(config, env_spec, callbacks, run_number)
#     except Exception as e:
#         logger.error(f"Error in init_sweep: {e}", exc_info=True)

