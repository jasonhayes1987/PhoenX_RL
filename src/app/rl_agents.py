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
from .models import select_policy_model, StochasticContinuousPolicy, StochasticDiscretePolicy, ValueModel, ContinuousCritic, DiscreteCritic, ActorModel, build_model
from .schedulers import ScheduleWrapper
from .adaptive_kl import AdaptiveKL
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, Buffer
from .normalizer import BaseNormalizer, RewardNorm, create_normalizer
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
# import gymnasium as gym
# import gymnasium_robotics
from gymnasium.envs.registration import EnvSpec
import numpy as np

from isaaclab.app import AppLauncher


from app.agent_utils import compute_n_step_return, compute_advantages_and_returns, compute_monte_carlo_returns, compute_q_retrace, grad_norm_from_optimizer, setup_auto_entropy, soft_update


## Base Agent Class ##
class Agent(ABC):
    """Base class for all RL agents.

    Serialization contract (uniform across every agent):
        - ``get_config()``  -> ``{"type", "config"}`` architecture description.
        - ``from_config(config, env)`` -> rebuild architecture (fresh tensors),
          the env injected as a live object.
        - ``save_state(dir)`` / ``load_state(dir)`` -> dump/restore every tensor
          (model weights + optimizers + schedule progress + normalizer stats +
          entropy temperature + intrinsic motivation), driven entirely by the
          class-level component-attribute declarations below.

    Subclasses only declare *which* attributes hold each kind of component; the
    base class handles the (de)serialization uniformly.
    """

    # Attribute names that hold trainable Models (weights + optimizer + schedule).
    MODEL_ATTRS: tuple[str, ...] = ()
    # Target/EMA networks (weights are saved; rebuilt as clones on construction).
    TARGET_ATTRS: tuple[str, ...] = ()
    # BaseNormalizer attributes (running statistics).
    NORMALIZER_ATTRS: tuple[str, ...] = (
        "state_normalizer", "goal_normalizer", "reward_normalizer", "advantage_normalizer",
    )
    # Agent-level ScheduleWrapper attributes (progress persisted via get_state).
    SCHEDULE_ATTRS: tuple[str, ...] = ()
    # IntrinsicMotivation attributes (self-contained sub-artifacts).
    IM_ATTRS: tuple[str, ...] = ("intrinsic_motivation",)

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
            self._nstep_retrace_stats = deque(maxlen=2048)

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

    def get_nstep_diagnostics(self) -> dict:
        """Return and clear accumulated n-step + retrace boundary diagnostics."""
        if not self._nstep_retrace_stats:
            return {}

        final_cum_c = []
        max_leakage = []

        for stats in self._nstep_retrace_stats:
            final_cum_c.extend(stats.get("done_window_final_cum_c", []))
            max_leakage.extend(stats.get("done_window_max_leakage", []))

        self._nstep_retrace_stats.clear()

        out = {}
        if final_cum_c:
            out["nstep/avg_final_cum_c_on_done_windows"] = float(sum(final_cum_c) / len(final_cum_c))
        if max_leakage:
            out["nstep/max_leakage_in_mask_after_done"] = float(max(max_leakage))

        return out

    def get_config(self):
        return {
            "type": self.__class__.__name__,
            "config":{
                "save_dir": self.save_dir,
                "name": self.name,
                "device": self.device.type,
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

    def _live_env(self) -> EnvWrapper:
        """Return the single live env instance shared by the agent's models."""
        for name in self.MODEL_ATTRS or ("policy",):
            model = getattr(self, name, None)
            if model is not None:
                return model.env
        raise AttributeError(f"{self.__class__.__name__} has no model to source env from.")

    @classmethod
    def from_config(cls, config: dict, env: EnvWrapper) -> "Agent":
        """Rebuild an agent (architecture + fresh tensors) from an inner config.

        Every sub-component is reconstructed and the single live ``env`` is
        injected into all models. Tensor state (weights, optimizers, stats,
        entropy temperature) and intrinsic-motivation modules are restored
        separately by :meth:`load_state`.
        """
        cfg = dict(config)
        for key in ("policy", "critic", "critic_b", "value"):
            if cfg.get(key) is not None:
                cfg[key] = build_model(cfg[key], env)
        for key in ("state_normalizer", "goal_normalizer",
                    "reward_normalizer", "advantage_normalizer"):
            if cfg.get(key) is not None:
                cfg[key] = create_normalizer(cfg[key])
        for key in ("noise", "target_noise"):
            if cfg.get(key) is not None:
                cfg[key] = Noise.create_instance(cfg[key]["type"], **cfg[key]["config"])
        if cfg.get("kl_adapter") is not None:
            cfg["kl_adapter"] = AdaptiveKL(**cfg["kl_adapter"])
        for key in list(cfg):
            if key.endswith("_schedule") and isinstance(cfg.get(key), dict):
                cfg[key] = ScheduleWrapper.from_config(cfg[key])
        # Intrinsic motivation is a self-contained artifact rebuilt in load_state.
        for key in ("intrinsic_motivation"):
            if key in cfg:
                cfg[key] = None
        return cls(**cfg)

    def save_state(self, save_dir: str | Path) -> None:
        """Dump every tensor of the agent under ``save_dir`` (mirrors the tree)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name in self.MODEL_ATTRS + self.TARGET_ATTRS:
            model = getattr(self, name, None)
            if model is not None:
                model.save_state(save_dir / f"{name}.pt")

        for name in self.NORMALIZER_ATTRS:
            normalizer = getattr(self, name, None)
            if normalizer is not None:
                normalizer.save_state(save_dir / "normalizers" / f"{name}.pt")

        for name in self.IM_ATTRS:
            intrinsic = getattr(self, name, None)
            if intrinsic is not None:
                intrinsic.save(save_dir)  # writes save_dir/intrinsic_motivation/...

        extra: dict = {}
        for name in self.SCHEDULE_ATTRS:
            schedule = getattr(self, name, None)
            if schedule is not None:
                extra[name] = schedule.get_state()
        if getattr(self, "auto_entropy_tuning", False):
            extra["log_alpha"] = self.log_alpha.detach().cpu()
            extra["entropy_optimizer"] = self.entropy_optimizer.state_dict()
        kl_adapter = getattr(self, "kl_adapter", None)
        if kl_adapter is not None and hasattr(kl_adapter, "get_state"):
            extra["kl_adapter"] = kl_adapter.get_state()
        if extra:
            T.save(extra, save_dir / "agent_state.pt")

    def load_state(self, save_dir: str | Path, load_weights: bool = True) -> None:
        """Restore every tensor written by :meth:`save_state` (in place)."""
        save_dir = Path(save_dir)

        for name in self.MODEL_ATTRS + self.TARGET_ATTRS:
            model = getattr(self, name, None)
            path = save_dir / f"{name}.pt"
            if model is not None and path.exists():
                model.load_state(path, load_weights=load_weights)

        for name in self.NORMALIZER_ATTRS:
            normalizer = getattr(self, name, None)
            path = save_dir / "normalizers" / f"{name}.pt"
            if normalizer is not None and path.exists():
                normalizer.load_state(path)

        if (save_dir / "intrinsic_motivation" / "config.json").is_file():
            intrinsic = IntrinsicMotivation.load(save_dir, env=self._live_env())
            for name in self.IM_ATTRS:
                setattr(self, name, intrinsic)

        extra_path = save_dir / "agent_state.pt"
        if extra_path.exists():
            extra = T.load(extra_path, map_location=self.device, weights_only=False)
            for name in self.SCHEDULE_ATTRS:
                schedule = getattr(self, name, None)
                if schedule is not None and extra.get(name) is not None:
                    schedule.set_state(extra[name])
            if getattr(self, "auto_entropy_tuning", False) and "log_alpha" in extra:
                with T.no_grad():
                    self.log_alpha.data.copy_(extra["log_alpha"].to(self.device))
                if extra.get("entropy_optimizer") is not None:
                    self.entropy_optimizer.load_state_dict(extra["entropy_optimizer"])
            kl_adapter = getattr(self, "kl_adapter", None)
            if kl_adapter is not None and hasattr(kl_adapter, "set_state") and "kl_adapter" in extra:
                kl_adapter.set_state(extra["kl_adapter"])

class Reinforce(Agent):
    MODEL_ATTRS = ("policy", "value")
    SCHEDULE_ATTRS = ("entropy_schedule",)
    IM_ATTRS = ()

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
        config["config"].update({
            "policy": self.policy.get_config(),
            "value": self.value.get_config() if self.value is not None else None,
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

class ActorCritic(Agent):
    """Actor Critic Agent."""

    MODEL_ATTRS = ("policy", "value")
    SCHEDULE_ATTRS = ("entropy_schedule",)
    IM_ATTRS = ()

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
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
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

    MODEL_ATTRS = ("policy", "value")
    SCHEDULE_ATTRS = ("entropy_schedule", "policy_clip_schedule", "value_clip_schedule")

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
        policy_grad_clip: float = 40.0,
        value_clip: float = 0.2,
        value_clip_schedule: ScheduleWrapper|None = None,
        value_grad_clip: float = 40.0,
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
    ) -> Action:
        """
        Select an action based on the current policy.
        Returns actions that are already scaled to the environment's action space.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            context: str: The context of the action (train, test).
        
        Returns:
            Action: actions.
        """
        raw_actions = None
        with T.no_grad():
            dist = self.policy(states, goals)
            if context == 'train':
                if self.policy.distribution == 'categorical':
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.sample_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
            elif context == 'test':
                if self.policy.distribution == 'categorical':
                    actions = self.policy.get_mean_actions(dist)
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.mean_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
            else:
                raise ValueError(f"Invalid context: {context}")

        return Action(actions, raw_actions=raw_actions, log_probs=log_probs)

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
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
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

        # Get trajectory length, num envs, and total samples for reshaping
        traj_len, num_envs = extrinsic_rewards.shape
        total_samples = traj_len * num_envs

        # Flatten trajectory data
        states_flat = states.reshape(total_samples, -1)
        next_states_flat = next_states.reshape(total_samples, -1)
        actions_flat = actions.reshape(total_samples, -1)
        raw_actions_flat = raw_actions.reshape(total_samples, -1)
        extrinsic_rewards_flat = extrinsic_rewards.reshape(total_samples, -1)
        goals_flat = goals.reshape(total_samples, -1) if goals is not None else None
        # ach_goals_flat = ach_goals.reshape(total_samples, -1) if ach_goals is not None else None
        # next_ach_goals_flat = next_ach_goals.reshape(total_samples, -1) if next_ach_goals is not None else None
        # terminations_flat = terminations.reshape(total_samples, -1)
        # truncations_flat = truncations.reshape(total_samples, -1)



        # Normalize states and goals
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            # ach_goals = self.goal_normalizer.normalize(ach_goals)
            # next_ach_goals = self.goal_normalizer.normalize(next_ach_goals)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Clip rewards if finite and not using reward normalizer
        if T.isfinite(T.tensor(self.reward_clip)) and self.reward_normalizer is None:
            extrinsic_rewards_flat = T.clamp(extrinsic_rewards_flat, min=-self.reward_clip, max=self.reward_clip)

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
            im_rewards = im_learn_rewards.reshape(traj_len, num_envs) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(traj_len, num_envs) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(traj_len, num_envs)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        # Get current log probs and values
        with T.no_grad():
            cur_dist = self.policy(states_flat, goals_flat)
            if self.policy.distribution == 'categorical':
                cur_log_probs = cur_dist.log_prob(actions_flat.view(-1)).unsqueeze(-1)
            else:
                cur_log_probs = cur_dist.log_prob_from_z(raw_actions_flat).unsqueeze(-1)
            
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
        raw_actions_flat = raw_actions_flat[valid_idx]
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
                states_batch = states_flat[batch_indices]
                goals_batch = goals_flat[batch_indices] if goals is not None else None
                actions_batch = actions_flat[batch_indices]
                raw_actions_batch = raw_actions_flat[batch_indices]
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
                    new_log_probs = new_dist.log_prob_from_z(raw_actions_batch).unsqueeze(-1)

                log_ratio = new_log_probs - cur_log_probs_batch
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
        """
        Get the current configuration of the PPO agent.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config["config"].update({
            "policy": self.policy.get_config(),
            "value": self.value.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
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
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation else None,
            "bootstrap_truncations": self.bootstrap_truncations
        })
        return config

class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agent."""

    MODEL_ATTRS = ("policy", "critic")
    TARGET_ATTRS = ("target_policy", "target_critic")
    SCHEDULE_ATTRS = ("noise_schedule",)

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
        raw_action_l2_coef: float = 0.0,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        critic_huber_delta: float = 1.0,
        N: int=1, # N-steps
        intrinsic_motivation: IntrinsicMotivation | None = None,
        save_dir: str = "models",
        device: str | T.device | None = None,
        **kwargs: Any,
    ):
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
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
        self.N = N
        self.intrinsic_motivation = intrinsic_motivation
        self.target_policy = self.policy.clone(device=self.policy.device)
        self.target_critic = self.critic.clone(device=self.critic.device)
        self.action_epsilon = action_epsilon
        self.noise = noise
        self.noise_schedule = noise_schedule
        self.noise_clip = noise_clip
        self.raw_action_l2_coef = raw_action_l2_coef

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
        
        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.policy.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                actions = T.as_tensor(self.policy.env.action_space.sample(), device=self.device)
                raw_actions = actions
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
                    raw_actions, actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

        else: # context == 'test'
            with T.no_grad():
                raw_actions, actions = self.policy(states, goals)
        
        return Action(actions, raw_actions=raw_actions)

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
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        # ach_goals = sample["state_achieved_goals"]
        # next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, -1) to train on all steps in N
        states_flat = states.reshape(-1, states.shape[-1])
        next_states_flat = next_states.reshape(-1, next_states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])
            # ach_goals_flat = ach_goals.reshape(-1, ach_goals.shape[-1])
            # next_ach_goals_flat = next_ach_goals.reshape(-1, next_ach_goals.shape[-1])

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            # ach_goals_flat = self.goal_normalizer.normalize(ach_goals_flat)
            # next_ach_goals_flat = self.goal_normalizer.normalize(next_ach_goals_flat)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Create normalized tensors reshaped to [batch_size, n_step_length]
        states_norm = states_flat.reshape(batch_size, n_step_length, -1)
        next_states_norm = next_states_flat.reshape(batch_size, n_step_length, -1)
        extrinsic_rewards_norm = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
        if goals is not None:
            goals_norm = goals_flat.reshape(batch_size, n_step_length, -1)
            # ach_goals_norm = ach_goals_flat.reshape(batch_size, n_step_length, -1)
            # next_ach_goals_norm = next_ach_goals_flat.reshape(batch_size, n_step_length, -1)
        else:
            goals_norm = None

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

        # Get target values
        with T.no_grad():
            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic.device
            ).squeeze()

            _, target_actions = self.target_policy(
                next_states_norm[:,-1,:],
                goals_norm[:,-1,:] if goals is not None else None
            )

            target_critic_values = self.target_critic(
                next_states_norm[:,-1,:],
                target_actions,
                goals_norm[:,-1,:] if goals is not None else None
            ).squeeze()

            no_dones_mask = (terminations.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values

            targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions = self.critic(
            states_norm[:,0,:],
            actions[:,0,:],
            goals_norm[:,0,:] if goals is not None else None
        ).squeeze()

        # Calculate TD errors (kept as raw signed differences for PER priorities and logging)
        error = targets - predictions

        # Per-sample Huber loss; apply IS weights before averaging if using PER
        per_sample_loss = self.critic_loss_fn(predictions, targets)
        if weights is not None:
            critic_loss = (weights.to(self.critic.device) * per_sample_loss).mean()
        else:
            critic_loss = per_sample_loss.mean()

        # Update critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip value gradient
        critic_grad_norm = T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip)
        self.critic.optimizer.step()

        # Get actor's action predictions
        pred_raw_actions, pred_actions = self.policy(
            states_norm[:,0,:],
            goals_norm[:,0,:] if goals is not None else None
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic(
            states_norm[:,0,:],
            pred_actions,
            goals_norm[:,0,:] if goals is not None else None
        )

        if weights is not None:
            actor_loss = -(weights.to(self.policy.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()

        # Add raw action l2 regularization if coef > 0
        actor_loss += self.raw_action_l2_coef * pred_raw_actions.pow(2).mean()

        # Update actor
        self.policy.optimizer.zero_grad()
        actor_loss.backward()

        # Clip policy gradient
        policy_grad_norm = T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.policy_grad_clip)
        self.policy.optimizer.step()

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(extrinsic_rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
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
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "td_error": error.mean().item(),
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
            "action_epsilon": self.action_epsilon,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "raw_action_l2_coef": self.raw_action_l2_coef,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config


class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""

    MODEL_ATTRS = ("policy", "critic", "critic_b")
    TARGET_ATTRS = ("target_policy", "target_critic", "target_critic_b")
    SCHEDULE_ATTRS = ("noise_schedule", "target_noise_schedule")

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
        raw_action_l2_coef: float = 0.0,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        critic_huber_delta: float = 1.0,
        policy_update_delay: int = 2,
        N: int=1, # N-steps
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
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
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
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
        self.raw_action_l2_coef = raw_action_l2_coef
        self.target_noise = target_noise
        self.target_noise_schedule = target_noise_schedule
        self.policy_update_delay = policy_update_delay

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
        
        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.policy.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                actions = T.as_tensor(self.policy.env.action_space.sample(), device=self.device)
                raw_actions = actions
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
                    raw_actions, actions = self.policy(states, goals)
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

        else: # context == 'test'
            with T.no_grad():
                raw_actions, actions = self.policy(states, goals)

        return Action(actions, raw_actions=raw_actions)

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
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        # ach_goals = sample["state_achieved_goals"]
        # next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, -1) to train on all steps in N
        states_flat = states.reshape(-1, states.shape[-1])
        next_states_flat = next_states.reshape(-1, next_states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])
            # ach_goals_flat = ach_goals.reshape(-1, ach_goals.shape[-1])
            # next_ach_goals_flat = next_ach_goals.reshape(-1, next_ach_goals.shape[-1])

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            # ach_goals_flat = self.goal_normalizer.normalize(ach_goals_flat)
            # next_ach_goals_flat = self.goal_normalizer.normalize(next_ach_goals_flat)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Create normalized tensors reshaped to [batch_size, n_step_length]
        states_norm = states_flat.reshape(batch_size, n_step_length, -1)
        next_states_norm = next_states_flat.reshape(batch_size, n_step_length, -1)
        extrinsic_rewards_norm = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
        if goals is not None:
            goals_norm = goals_flat.reshape(batch_size, n_step_length, -1)
            # ach_goals_norm = ach_goals_flat.reshape(batch_size, n_step_length, -1)
            # next_ach_goals_norm = next_ach_goals_flat.reshape(batch_size, n_step_length, -1)
        else:
            goals_norm = None

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

        # Get target values
        with T.no_grad():
            targets = compute_n_step_return(
                rewards,
                self.discount,
                device=self.target_critic.device
            ).squeeze()

            _, target_actions = self.target_policy(
                next_states_norm[:,-1,:],
                goals_norm[:,-1,:] if goals is not None else None,
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
                next_states_norm[:,-1,:],
                target_actions,
                goals_norm[:,-1,:] if goals is not None else None
            ).squeeze()
            
            target_critic_values_b = self.target_critic_b(
                next_states_norm[:,-1,:],
                target_actions,
                goals_norm[:,-1,:] if goals is not None else None
            ).squeeze()
            
            target_critic_values = T.minimum(target_critic_values_a, target_critic_values_b)
            no_dones_mask = (terminations.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** trajectory_lengths # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values
            
            
            targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions
        predictions_a = self.critic(
            states_norm[:,0,:],
            actions[:,0,:],
            goals_norm[:,0,:] if goals is not None else None
        ).squeeze()

        predictions_b = self.critic_b(
            states_norm[:,0,:],
            actions[:,0,:],
            goals_norm[:,0,:] if goals is not None else None
        ).squeeze()

        # Calculate TD errors (kept as raw signed differences for PER priorities and logging)
        error_a = targets - predictions_a
        error_b = targets - predictions_b
        # error = (error_a.abs() + error_b.abs()) / 2  # Average of absolute errors for priorities
        error = T.minimum(error_a, error_b)

        # Per-sample Huber loss; apply IS weights before averaging if using PER
        per_sample_loss_a = self.critic_loss_fn(predictions_a, targets)
        per_sample_loss_b = self.critic_loss_fn(predictions_b, targets)
        if weights is not None:
            critic_loss_a = (weights.to(self.critic.device) * per_sample_loss_a).mean()
            critic_loss_b = (weights.to(self.critic_b.device) * per_sample_loss_b).mean()
            critic_loss = critic_loss_a + critic_loss_b
        else:
            critic_loss_a = per_sample_loss_a.mean()
            critic_loss_b = per_sample_loss_b.mean()
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
        pred_raw_actions, pred_actions = self.policy(
            states_norm[:,0,:],
            goals_norm[:,0,:] if goals is not None else None
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic(
            states_norm[:,0,:],
            pred_actions,
            goals_norm[:,0,:] if goals is not None else None
        )
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            actor_loss = -(weights.to(self.policy.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()
        
        # Add raw action l2 regularization if coef > 0
        actor_loss += self.raw_action_l2_coef * pred_raw_actions.pow(2).mean()

        
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
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(extrinsic_rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
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
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "td_error": error.mean().item(),
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
            "action_epsilon": self.action_epsilon,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "target_noise": self.target_noise.get_config() if self.target_noise is not None else None,
            "target_noise_schedule": self.target_noise_schedule.get_config() if self.target_noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "raw_action_l2_coef": self.raw_action_l2_coef,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "policy_update_delay": self.policy_update_delay,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config

class SAC(Agent):
    """Soft Actor Critic Agent."""

    MODEL_ATTRS = ("policy", "critic", "critic_b")
    TARGET_ATTRS = ("target_critic", "target_critic_b")
    SCHEDULE_ATTRS = ("entropy_schedule",)

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
        critic_huber_delta: float = 1.0,
        N: int=1,
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
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
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
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
        raw_actions = None
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.policy.env.action_space.sample(), device=self.device)
                if isinstance(self.policy, StochasticContinuousPolicy): # Continuous
                    with T.no_grad():
                        dist = self.policy(states, goals)
                        raw_actions = dist.z_from_action(actions)
                    delta = T.as_tensor(
                        self.policy.act_space.high - self.policy.act_space.low,
                        device=self.device,
                    )
                    log_probs = (-T.log(delta).sum(-1)) * T.ones(actions.shape[0], device=self.device)
                else: # Discrete
                    num_actions = T.as_tensor(self.policy.act_space.n, device=self.device)
                    log_probs = T.full((actions.shape[0],), -T.log(num_actions), device=self.device)
            
            else: # Sample action from policy
                with T.no_grad():
                    dist = self.policy(states, goals)
                    if isinstance(self.policy, StochasticContinuousPolicy):
                        actions, raw_actions = dist.sample_with_z()
                        log_probs = dist.log_prob_from_z(raw_actions)
                    else: # Discrete
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

        elif context == 'test':
            with T.no_grad():
                dist = self.policy(states, goals)
                if isinstance(self.policy, StochasticContinuousPolicy):
                    actions, raw_actions = dist.mean_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
                else: # Discrete
                    actions = self.policy.get_mean_actions(dist)
                    log_probs = dist.log_prob(actions)

        else:
            raise ValueError(f"Invalid context: {context}")

        return Action(actions, raw_actions=raw_actions, log_probs=log_probs)

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
        raw_actions = sample["raw_actions"]
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
        raw_actions_flat = raw_actions.reshape(-1, raw_actions.shape[-1])
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
            # ach_goals_flat = self.goal_normalizer.normalize(ach_goals_flat)
            # next_ach_goals_flat = self.goal_normalizer.normalize(next_ach_goals_flat)
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
            # Get current policy for sampled states
            cur_dist = self.policy(
                states_flat,
                goals_flat if goals is not None else None
            )

            # Get current values of sampled states and log probs of taking the sampled actions
            # Continuous Action Space
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                cur_log_probs = cur_dist.log_prob_from_z(raw_actions_flat).reshape(batch_size, n_step_length)
                q_cur = T.minimum(
                    self.target_critic(
                        states_flat,
                        actions_flat,
                        goals_flat if goals is not None else None
                        ),
                    self.target_critic_b(
                        states_flat,
                        actions_flat,
                        goals_flat if goals is not None else None
                        )
                ).reshape(batch_size, n_step_length)

            else: # Discrete Action Space
                cur_log_probs = cur_dist.logits.gather(1, actions_flat.long()).reshape(batch_size, n_step_length)
                q_cur_all = T.minimum(
                    self.target_critic(
                        states_flat,
                        goals_flat if goals is not None else None
                        ),
                    self.target_critic_b(
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
                target_actions, target_z = next_dist.sample_with_z()
                next_log_probs = next_dist.log_prob_from_z(target_z)
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

            
            q_retrace, q_metrics = compute_q_retrace(
                rewards,
                terminations,
                truncations,
                trajectory_lengths,
                q_cur,
                target_q,
                cur_log_probs,
                buf_log_probs,
                self.discount,
                device=self.device
            )
            # Collect retrace boundary diagnostics
            if q_metrics.get("done_window_final_cum_c") or q_metrics.get("done_window_max_leakage"):
                self._nstep_retrace_stats.append({
                    "done_window_final_cum_c": q_metrics["done_window_final_cum_c"],
                    "done_window_max_leakage": q_metrics["done_window_max_leakage"],
                })

            # Set low bound of q-retrace to -1/1-self.discount
            q_retrace = T.clamp(q_retrace, min=-1/(1-self.discount))

        # Reshape flat states, goals, actions to [batch_size, n-step, feature_dim]
        states_reshaped = states_flat.reshape(batch_size, n_step_length, -1)
        actions_reshaped = actions_flat.reshape(batch_size, n_step_length, -1)
        goals_reshaped = goals_flat.reshape(batch_size, n_step_length, -1) if goals is not None else None
        
        # Continuous critic predictions
        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            q1_preds = self.critic(
                states_reshaped[:,0,:],
                actions_reshaped[:,0,:],
                goals_reshaped[:,0,:] if goals_reshaped is not None else None).squeeze()
            q2_preds = self.critic_b(
                states_reshaped[:,0,:],
                actions_reshaped[:,0,:],
                goals_reshaped[:,0,:] if goals_reshaped is not None else None).squeeze()

        else: # Discrete critic predictions
            q1 = self.critic(
                states_reshaped[:,0,:],
                goals_reshaped[:,0,:] if goals_reshaped is not None else None
            )
            q2 = self.critic_b(
                states_reshaped[:,0,:],
                goals_reshaped[:,0,:] if goals_reshaped is not None else None
            )
            buffer_actions = actions_reshaped[:,0,:].squeeze(-1).long().unsqueeze(1)
            q1_preds = q1.gather(1, buffer_actions).squeeze(1)
            q2_preds = q2.gather(1, buffer_actions).squeeze(1)

        # Per-sample Huber loss for each critic
        q1_loss = self.critic_loss_fn(q1_preds, q_retrace.detach())
        q2_loss = self.critic_loss_fn(q2_preds, q_retrace.detach())
        # Get min raw TD error (used to update PER priorities — kept as signed difference, not Huber-transformed)
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
            states_reshaped[:,0,:],
            goals_reshaped[:,0,:] if goals_reshaped is not None else None
        )
        # Continuous policy update
        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            new_actions, new_z = dist.rsample_with_z()
            log_probs = dist.log_prob_from_z(new_z)
            q1 = self.critic(states_reshaped[:,0,:], new_actions, goals_reshaped[:,0,:] if goals_reshaped is not None else None).squeeze()
            q2 = self.critic_b(states_reshaped[:,0,:], new_actions, goals_reshaped[:,0,:] if goals_reshaped is not None else None).squeeze()
            min_q = T.minimum(q1, q2)
            actor_loss = entropy_coefficient * log_probs - min_q

        else: # Discrete policy update
            new_actions = dist.sample().float()
            log_probs = dist.logits
            q1 = self.critic(states_reshaped[:,0,:], goals_reshaped[:,0,:] if goals_reshaped is not None else None)
            q2 = self.critic_b(states_reshaped[:,0,:], goals_reshaped[:,0,:] if goals_reshaped is not None else None)
            min_q = T.minimum(q1, q2)
            actor_loss = (dist.probs * (entropy_coefficient * log_probs - min_q)).sum(-1)


        # if weights is not None:
        #     actor_loss = weights.to(self.policy.device) * actor_loss

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
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states_reshaped, "states"),
                summarize_tensor(actions_reshaped, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals_reshaped, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_q, "target critic values"),
                summarize_tensor(q_retrace, "q retrace"),
                summarize_tensor(q_metrics["td_errors"], "td errors"),
                summarize_tensor(q_metrics["mask"], "mask"),
                summarize_tensor(q_metrics["is_ratio"], "is ratio"),
                summarize_tensor(q_metrics["cum_c"], "cum c"),
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

            mask = q_metrics["mask"]
            cum_c_final = q_metrics["cum_c"]
            td = q_metrics["td_errors"]
            # Only look at rows that actually had a termination or truncation
            has_done = (terminations | truncations).any(dim=1)
            if has_done.any():
                for i in range(batch_size):
                    if has_done[i]:
                        L = int(trajectory_lengths[i].item())
                        done_pos = (terminations[i, :L] | truncations[i, :L]).nonzero(as_tuple=True)[0]
                        self.logger.debug(
                            "[RETRACE-DIAG] learn_count=%d "
                            "row=%d L=%d done_at=%s "
                            "final_cum_c=%.4f "
                            "mask_tail=%s",
                            step,
                            i,
                            L,
                            done_pos.tolist(),
                            cum_c_final[i].item(),
                            mask[i, done_pos[-1]+1:L].tolist() if done_pos.numel() > 0 else 'N/A',
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
            "td_error": q_metrics["td_errors"].mean().item(),
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
            "critic_b": self.critic_b.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config


@runtime_checkable
class HasTargetNetworks(Protocol):
    def soft_update_targets(self) -> None: ...


# Registry of every concrete agent class, keyed by class name (the "type" tag
# emitted by Agent.get_config). Used by build_agent to reconstruct from a config.
AGENT_REGISTRY: Dict[str, type] = {
    "Reinforce": Reinforce,
    "ActorCritic": ActorCritic,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}


def build_agent(config: dict, env: EnvWrapper) -> "Agent":
    """Rebuild an agent from a ``{"type", "config"}`` dict, injecting ``env``.

    This is the single entry point for rebuilding a saved agent config
    back into a live (fresh-tensor) agent; tensor state is restored afterwards
    via :meth:`Agent.load_state`.
    """
    agent_type = config["type"]
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. Available: {list(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[agent_type].from_config(config["config"], env)